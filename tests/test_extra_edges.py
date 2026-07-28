"""Tests for `MeshNeuron.extra_edges` - edges that are not part of any face.

These express connectivity that the mesh surface itself does not have (e.g.
bridges between disconnected fragments). Two things need testing: that the
edges survive the things we do to neurons (storage/invariants), and that the
code deriving connectivity from a mesh actually sees them (plumbing).
"""

import navis
import numpy as np
import pytest
import trimesh as tm

from navis.graph.graph_utils import _connected_components
from navis.utils.subclasses import TrimeshPlus, validate_extra_edges

HAS_FASTCORE = navis.utils.fastcore is not None


@pytest.fixture(params=["builtin", "fastcore"])
def backend(request, monkeypatch):
    """Run a test against both the numpy/igraph and the fastcore backend."""
    if request.param == "builtin":
        monkeypatch.setattr(navis.utils, "fastcore", None)
    elif not HAS_FASTCORE:
        pytest.skip("navis-fastcore not installed")
    return request.param


@pytest.fixture
def mesh():
    """The example mesh - which happens to consist of 70 fragments."""
    return navis.example_neurons(1, kind="mesh")


@pytest.fixture
def fragments(mesh):
    """Connected components of the example mesh, largest first."""
    return sorted(_connected_components(mesh), key=len, reverse=True)


@pytest.fixture
def bridged(mesh, fragments):
    """Example mesh with the two largest fragments bridged by one edge."""
    x = mesh.copy()
    x.extra_edges = [[int(fragments[0][0]), int(fragments[1][0])]]
    return x


# ---------------------------------------------------------------------------
# Storage & invariants
# ---------------------------------------------------------------------------


def test_default_is_empty(mesh):
    assert mesh.extra_edges.shape == (0, 2)
    assert mesh.n_extra_edges == 0
    # An empty set of extra edges must not clutter the summary
    assert "n_extra_edges" not in mesh.summary().index


def test_summary_toggles(mesh):
    mesh.extra_edges = [[0, 1]]
    assert mesh.summary()["n_extra_edges"] == 1
    mesh.extra_edges = None
    assert "n_extra_edges" not in mesh.summary().index


def test_canonicalization(mesh):
    # Duplicates (in either orientation) and self-loops must go, and the result
    # must be sorted both within and across rows
    mesh.extra_edges = [[10, 5], [5, 10], [3, 3], [1, 2]]
    assert mesh.extra_edges.tolist() == [[1, 2], [5, 10]]
    assert mesh.extra_edges.dtype == np.int64


def test_single_edge(mesh):
    mesh.extra_edges = [3, 7]
    assert mesh.extra_edges.tolist() == [[3, 7]]


@pytest.mark.parametrize(
    "edges,error",
    [
        ([[0, 1, 2]], ValueError),  # wrong shape
        ([[0.5, 1.5]], TypeError),  # not integers
        ([[0, -1]], ValueError),  # negative index
        ([[0, 10**9]], ValueError),  # out of bounds
    ],
)
def test_validation(mesh, edges, error):
    with pytest.raises(error):
        mesh.extra_edges = edges


def test_hash_reflects_extra_edges(mesh):
    before = mesh.core_md5
    mesh.extra_edges = [[0, 1]]
    assert mesh.core_md5 != before


def test_copy_is_independent(bridged):
    copy = bridged.copy()
    assert copy.extra_edges.tolist() == bridged.extra_edges.tolist()
    copy.extra_edges = None
    assert bridged.n_extra_edges == 1


def test_pickle_roundtrip(bridged):
    import pickle

    restored = pickle.loads(pickle.dumps(bridged))
    assert restored.extra_edges.tolist() == bridged.extra_edges.tolist()


def test_constructor_inherits_from_trimesh(mesh):
    m = TrimeshPlus(mesh.vertices, mesh.faces, process=False)
    m.add_extra_edges([[0, 100]])
    assert navis.MeshNeuron(m, process=False).extra_edges.tolist() == [[0, 100]]


@pytest.mark.parametrize("op", ["mul", "div", "add", "sub"])
def test_arithmetic_preserves(bridged, op):
    """Scaling/offsetting moves vertices but does not renumber them."""
    out = {
        "mul": lambda x: x * 2,
        "div": lambda x: x / 2,
        "add": lambda x: x + 10,
        "sub": lambda x: x - 10,
    }[op](bridged)
    assert out.extra_edges.tolist() == bridged.extra_edges.tolist()


def test_dropped_when_vertex_count_changes(bridged, caplog):
    """Extra edges are indices - they can't survive a renumbering."""
    with caplog.at_level("WARNING"):
        bridged.vertices = bridged.vertices[:-1]
    assert bridged.n_extra_edges == 0
    assert "extra edges" in caplog.text


def test_kept_when_vertex_count_is_stable(bridged):
    """E.g. laplacian smoothing moves vertices but keeps their indices."""
    edges = bridged.extra_edges.tolist()
    bridged.vertices = bridged.vertices * 1.1
    assert bridged.extra_edges.tolist() == edges


def test_faces_do_not_invalidate(bridged):
    """Extra edges index vertices, so re-ordering faces leaves them valid."""
    edges = bridged.extra_edges.tolist()
    bridged.faces = bridged.faces[::-1]
    assert bridged.extra_edges.tolist() == edges


# ---------------------------------------------------------------------------
# TrimeshPlus
# ---------------------------------------------------------------------------


def test_trimesh_is_plus_and_carries_edges(bridged):
    assert isinstance(bridged.trimesh, TrimeshPlus)
    assert bridged.trimesh.extra_edges.tolist() == bridged.extra_edges.tolist()


def test_trimesh_internals_unaffected(mesh, bridged):
    """Extra edges must not leak into anything trimesh derives from the faces."""
    plain, plus = mesh.trimesh, bridged.trimesh

    assert len(plus.edges) == 3 * len(bridged.faces)
    assert len(plus.edges) == len(plus.edges_face)
    assert plus.edges_unique.shape == plain.edges_unique.shape
    # This one raises outright if `edges` and the faces get out of sync
    assert plus.faces_unique_edges.shape == (len(bridged.faces), 3)
    # Geometry must be untouched - healing topology is not cosmetic surgery
    assert plus.volume == plain.volume
    assert plus.area == plain.area


def test_graph_edges(bridged):
    plus = bridged.trimesh
    expected = len(plus.edges_unique) + len(plus.extra_edges)
    assert len(plus.graph_edges) == expected


def test_trimesh_copy_keeps_class_and_edges(bridged):
    copy = bridged.trimesh.copy()
    assert isinstance(copy, TrimeshPlus)
    assert copy.extra_edges.tolist() == bridged.extra_edges.tolist()


def test_add_extra_edges_drops_face_edges(mesh):
    m = TrimeshPlus(mesh.vertices, mesh.faces, process=False)
    face_edge = m.edges_unique[0]
    m.add_extra_edges([face_edge, [0, 1000]])
    assert m.extra_edges.tolist() == [[0, 1000]]


def test_add_extra_edges_append_and_replace(mesh):
    m = TrimeshPlus(mesh.vertices, mesh.faces, process=False)
    m.add_extra_edges([[0, 1000]])
    m.add_extra_edges([[1, 1001]])
    assert m.extra_edges.tolist() == [[0, 1000], [1, 1001]]
    m.add_extra_edges([[2, 1002]], replace=True)
    assert m.extra_edges.tolist() == [[2, 1002]]


def test_validate_extra_edges_empty():
    for empty in (None, [], np.zeros((0, 2))):
        assert validate_extra_edges(empty).shape == (0, 2)


# ---------------------------------------------------------------------------
# Plumbing: things that derive connectivity from a mesh
# ---------------------------------------------------------------------------


def test_mesh_unique_edges(mesh, bridged):
    plain = navis.utils.mesh_unique_edges(mesh)
    with_extra, lengths = navis.utils.mesh_unique_edges(bridged, return_lengths=True)

    assert len(with_extra) == len(plain) + 1
    assert with_extra[-1].tolist() == bridged.extra_edges[0].tolist()

    expected = np.linalg.norm(np.subtract(*bridged.vertices[bridged.extra_edges[0]]))
    assert lengths[-1] == pytest.approx(expected)

    # ... and the opt-out
    assert len(navis.utils.mesh_unique_edges(bridged, extra_edges=False)) == len(plain)


def test_trimesh_cache_stays_faces_only(bridged):
    """`mesh_unique_edges` seeds trimesh's cache - with face edges only."""
    trimesh = bridged.trimesh
    navis.utils.mesh_unique_edges(trimesh, return_lengths=True)
    assert len(trimesh.edges_unique) == len(trimesh.edges_unique_length)
    assert len(trimesh.edges_unique) < len(trimesh.graph_edges)


def test_connected_components(mesh, fragments, bridged, backend):
    merged = sorted(_connected_components(bridged), key=len, reverse=True)

    assert len(merged) == len(fragments) - 1
    assert set(merged[0]) == set(fragments[0]) | set(fragments[1])


def test_break_fragments(mesh, bridged, backend):
    assert len(navis.break_fragments(bridged)) == len(navis.break_fragments(mesh)) - 1


def test_drop_fluff(mesh, fragments, bridged, backend):
    """The bridged fragment must now count as part of the main component."""
    fluffless = navis.drop_fluff(bridged)
    assert fluffless.n_vertices == len(fragments[0]) + len(fragments[1])
    assert fluffless.n_extra_edges == 1


def test_igraph_and_nx(mesh, bridged):
    assert bridged.igraph.ecount() == mesh.igraph.ecount() + 1
    assert bridged.graph.number_of_edges() == mesh.graph.number_of_edges() + 1
    assert len(bridged.igraph.components(mode="WEAK")) == len(
        mesh.igraph.components(mode="WEAK")
    ) - 1


def test_geodesic_matrix(mesh, bridged, backend):
    a, b = [int(v) for v in bridged.extra_edges[0]]

    # Without the bridge the two fragments are unreachable from one another
    assert np.isinf(navis.geodesic_matrix(mesh, from_=[a], to_=[b]).values[0, 0])

    dist = navis.geodesic_matrix(bridged, from_=[a], to_=[b]).values[0, 0]
    expected = np.linalg.norm(bridged.vertices[a] - bridged.vertices[b])
    assert dist == pytest.approx(expected, rel=1e-5)

    # Unweighted = hop count
    hops = navis.geodesic_matrix(bridged, from_=[a], to_=[b], weight=None).values[0, 0]
    assert hops == 1


@pytest.mark.skipif(not HAS_FASTCORE, reason="navis-fastcore not installed")
def test_geodesic_matrix_without_graph_variant(bridged, monkeypatch):
    """Older fastcore has no `geodesic_matrix_graph` - we must not use the mesh
    variant then, as it would silently ignore the extra edges."""
    a, b = [int(v) for v in bridged.extra_edges[0]]
    expected = navis.geodesic_matrix(bridged, from_=[a], to_=[b]).values[0, 0]

    monkeypatch.delattr(navis.utils.fastcore, "geodesic_matrix_graph", raising=False)
    assert navis.geodesic_matrix(bridged, from_=[a], to_=[b]).values[
        0, 0
    ] == pytest.approx(expected, rel=1e-5)


def test_geodesic_matrix_unchanged_within_fragment(mesh, fragments, bridged, backend):
    within = [int(v) for v in fragments[0][:20]]
    before = navis.geodesic_matrix(mesh, from_=within, to_=within)
    after = navis.geodesic_matrix(bridged, from_=within, to_=within)
    assert np.allclose(before.values, after.values)


def test_geodesic_clusters(mesh, bridged, backend):
    """Clusters are grown along the graph, so they may now cross the bridge."""
    a, b = [int(v) for v in bridged.extra_edges[0]]
    bridge_len = np.linalg.norm(bridged.vertices[a] - bridged.vertices[b])

    cl = navis.graph.geodesic_clusters(bridged, max_dist=bridge_len * 2)
    assert cl[a] == cl[b]

    cl_plain = navis.graph.geodesic_clusters(mesh, max_dist=bridge_len * 2)
    assert cl_plain[a] != cl_plain[b]


def test_sampling_resolution_ignores_extra_edges(mesh, bridged):
    """Sampling resolution describes the surface, not the connectivity."""
    assert bridged.sampling_resolution == mesh.sampling_resolution


def test_subset_remaps(bridged, fragments):
    keep = np.concatenate([fragments[0], fragments[1]])
    sub = navis.subset_neuron(bridged, keep)

    assert sub.n_extra_edges == 1
    # The remapped edge must still connect the same two points in space
    assert np.allclose(
        bridged.vertices[bridged.extra_edges[0]], sub.vertices[sub.extra_edges[0]]
    )
    assert len(_connected_components(sub)) == 1


def test_subset_drops_dangling(bridged, fragments):
    """An edge with only one surviving endpoint has to go."""
    sub = navis.subset_neuron(bridged, fragments[0])
    assert sub.n_extra_edges == 0


def test_subset_empty(bridged):
    assert navis.subset_neuron(bridged, []).n_extra_edges == 0


def test_combine_neurons_offsets(mesh, bridged):
    comb = navis.combine_neurons(bridged, bridged.copy())

    assert comb.n_vertices == 2 * bridged.n_vertices
    assert comb.n_extra_edges == 2
    # Second copy's edge must be shifted by the first copy's vertex count
    assert np.array_equal(
        comb.extra_edges[1], bridged.extra_edges[0] + bridged.n_vertices
    )
    assert np.allclose(
        comb.vertices[comb.extra_edges[1]], bridged.vertices[bridged.extra_edges[0]]
    )


def test_write_mesh_warns(bridged, tmp_path, caplog):
    with caplog.at_level("WARNING"):
        navis.write_mesh(bridged, tmp_path / "mesh.obj")
    assert "extra edges" in caplog.text


def test_smooth_mesh_preserves(bridged):
    """Laplacian smoothing keeps vertex indices, so the edges stay valid."""
    smoothed = navis.smooth_mesh(bridged, iterations=1)
    assert smoothed.extra_edges.tolist() == bridged.extra_edges.tolist()


def test_read_write_roundtrip_drops(bridged, tmp_path):
    """Mesh formats have no place for extra edges - they must not resurface."""
    fp = tmp_path / "mesh.obj"
    navis.write_mesh(bridged, fp)
    assert navis.read_mesh(fp).n_extra_edges == 0
