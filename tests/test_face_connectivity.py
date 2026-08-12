"""Tests for the two face readings of mesh connectivity - `connectivity="face"`
and `connectivity="manifold"` - i.e. components of faces joined by shared
*edges* rather than components of vertices joined by shared faces.

Each reading is strictly finer than the one before it, and each drops a kind of
junction. `"face"` drops the pinch points: faces meeting at a single vertex are
one piece under `"vertex"` and two under `"face"`. `"manifold"` also drops the
seams: sheets meeting along an edge three faces deep are one piece under
`"face"` and one each under `"manifold"`, which is what `trimesh.split` means by
a piece.

Everything worth testing follows from that - the labels are per-face rather than
per-vertex, the components `_connected_components` hands back can share a vertex,
and `drop_fluff` can therefore drop a piece whose pinch vertex survives with the
piece next to it.
"""

from collections import defaultdict

import navis
import numpy as np
import pytest
import trimesh as tm

from scipy.sparse import coo_matrix, csgraph

from navis.graph.graph_utils import (
    _connected_components,
    _mesh_component_labels,
    _resolve_connectivity,
)

#: The readings that label faces - most of what follows holds for both
FACE_READINGS = ["face", "manifold"]


@pytest.fixture
def mesh():
    """The example mesh - 14 vertex, 24 face and 502 manifold components."""
    return navis.example_neurons(1, kind="mesh")


@pytest.fixture
def pinched():
    """Two triangles sharing vertex 2, plus a vertex belonging to no face."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 0], [3, 2, 0], [9, 9, 9]], dtype=float
    )
    faces = np.array([[0, 1, 2], [2, 3, 4]])
    return navis.Mesh({"vertices": verts, "faces": faces}, process=False)


@pytest.fixture
def fins():
    """Three triangles along the spine `(0, 1)` - an edge three faces deep."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=float
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    return navis.Mesh({"vertices": verts, "faces": faces}, process=False)


@pytest.fixture
def faceless():
    """Three vertices and no faces at all."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    return navis.Mesh(
        {"vertices": verts, "faces": np.zeros((0, 3), dtype=int)}, process=False
    )


def face_components(faces, manifold_only=False):
    """Ground truth: join the faces that name the same edge.

    With `manifold_only` an edge has to carry exactly two faces to join them,
    which is the reading `trimesh.graph.face_adjacency` takes.
    """
    by_edge = defaultdict(list)
    for i, f in enumerate(faces):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            by_edge[(min(a, b), max(a, b))].append(i)

    groups = [g for g in by_edge.values() if not manifold_only or len(g) == 2]
    rows = [f for group in groups for f in group[1:]]
    cols = [group[0] for group in groups for _ in group[1:]]
    adj = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(faces),) * 2
    )
    return csgraph.connected_components(adj, directed=False)[1]


def canonical(labels):
    """Relabel a labelling `0, 1, 2, ...` in order of first appearance."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.empty(len(first), dtype=int)
    order[np.argsort(first)] = np.arange(len(first))
    return order[inverse.reshape(-1)]


def same_partition(a, b):
    """Whether two labellings group their items the same way."""
    return np.array_equal(canonical(a), canonical(b))


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("connectivity, expected", [("face", 24), ("manifold", 502)])
def test_labels_match_ground_truth(mesh, connectivity, expected):
    labels, n = _mesh_component_labels(mesh, connectivity=connectivity)
    truth = face_components(np.asarray(mesh.faces), connectivity == "manifold")

    assert len(labels) == len(mesh.faces)  # per face, not per vertex
    assert same_partition(labels, truth)
    assert n == expected


@pytest.mark.parametrize("connectivity", FACE_READINGS)
def test_labels_are_contiguous(mesh, connectivity):
    labels, n = _mesh_component_labels(mesh, connectivity=connectivity)

    assert set(np.unique(labels)) == set(range(n))


def test_manifold_is_trimesh_split(mesh):
    """The whole point of `"manifold"`: it is what `trimesh` calls a piece."""
    labels, n = _mesh_component_labels(mesh, connectivity="manifold")

    assert n == len(mesh.trimesh.split(only_watertight=False))

    # ... and not just in count - the same faces in the same pieces
    faces = np.asarray(mesh.faces)
    adjacency = tm.graph.face_adjacency(faces)
    graph = coo_matrix(
        (
            np.ones(len(adjacency), dtype=np.int8),
            (adjacency[:, 0], adjacency[:, 1]),
        ),
        shape=(len(faces),) * 2,
    )
    assert same_partition(labels, csgraph.connected_components(graph, False)[1])


def test_each_reading_is_finer_than_the_last(mesh):
    """Vertex, face, manifold - each partition refines the one before it."""
    by_vertex = _mesh_component_labels(mesh, connectivity="vertex")[0]
    by_face, n_face = _mesh_component_labels(mesh, connectivity="face")
    by_manifold, n_manifold = _mesh_component_labels(mesh, connectivity="manifold")

    faces = np.asarray(mesh.faces)
    assert n_manifold >= n_face >= len(np.unique(by_vertex))

    # Each face component's faces all sit in one vertex component ...
    for label in range(n_face):
        assert len(np.unique(by_vertex[faces[by_face == label]])) == 1

    # ... and each manifold component's faces all sit in one face component
    for label in range(n_manifold):
        assert len(np.unique(by_face[by_manifold == label])) == 1


def test_pinch_splits_face_but_not_vertex(pinched):
    """The junction `"vertex"` and `"face"` disagree about."""
    by_vertex, n_vertex = _mesh_component_labels(pinched, connectivity="vertex")
    by_face, n_face = _mesh_component_labels(pinched, connectivity="face")

    # Under "vertex" the two triangles are one piece and the loose vertex is the
    # other; under "face" they are one piece each and the vertex is nothing at all
    assert n_vertex == 2
    assert same_partition(by_vertex, [0, 0, 0, 0, 0, 1])
    assert n_face == 2
    assert same_partition(by_face, [0, 1])

    # Nothing here is more than two faces deep, so "manifold" says the same
    assert same_partition(by_face, _mesh_component_labels(pinched, "manifold")[0])


def test_seam_splits_manifold_but_not_face(fins):
    """The junction `"face"` and `"manifold"` disagree about."""
    assert _mesh_component_labels(fins, connectivity="vertex")[1] == 1
    assert _mesh_component_labels(fins, connectivity="face")[1] == 1

    by_manifold, n = _mesh_component_labels(fins, connectivity="manifold")
    assert n == 3
    assert same_partition(by_manifold, [0, 1, 2])


@pytest.mark.parametrize("connectivity", FACE_READINGS)
def test_no_faces(faceless, connectivity):
    """A mesh with no faces has no face components - not one per vertex."""
    assert _mesh_component_labels(faceless, connectivity="vertex")[1] == 3
    assert _mesh_component_labels(faceless, connectivity=connectivity)[1] == 0
    assert _connected_components(faceless, connectivity=connectivity) == []


@pytest.mark.parametrize("connectivity", FACE_READINGS)
def test_empty_mesh(connectivity):
    empty = navis.Mesh(None)

    assert _mesh_component_labels(empty, connectivity=connectivity)[1] == 0
    assert _connected_components(empty, connectivity=connectivity) == []


def test_bad_connectivity(mesh):
    with pytest.raises(ValueError):
        _mesh_component_labels(mesh, connectivity="edge")


# ---------------------------------------------------------------------------
# Extra edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("connectivity", FACE_READINGS)
def test_extra_edge_merges_face_components(pinched, connectivity):
    """A bridge between two vertices has to join the faces using them."""
    bridged = pinched.copy()
    bridged.extra_edges = [[0, 3]]

    assert _mesh_component_labels(bridged, connectivity=connectivity)[1] == 1


@pytest.mark.parametrize("connectivity", FACE_READINGS)
def test_extra_edge_to_a_faceless_vertex_does_nothing(pinched, connectivity):
    """Vertex 5 is in no face, so there is nothing for the bridge to join."""
    bridged = pinched.copy()
    bridged.extra_edges = [[0, 5]]

    assert _mesh_component_labels(bridged, connectivity=connectivity)[1] == 2


def test_a_bridge_can_weld_a_seam_back_together(fins):
    """Bridging two fins joins them - even where the seam kept them apart."""
    bridged = fins.copy()
    bridged.extra_edges = [[2, 3]]

    assert _mesh_component_labels(bridged, connectivity="manifold")[1] == 2


def test_a_bridge_welds_every_piece_at_its_endpoint(fins):
    """The one place face components come out coarser than they would like.

    A bridge joins the faces at one end to the faces at the other, so where its
    endpoint is a junction - a pinch, or the spine of these fins - the pieces
    meeting there are welded to each other too. No labelling can do otherwise:
    they cannot stay apart while both connect to the far end.
    """
    verts = np.vstack([fins.vertices, [[5, 5, 5], [6, 5, 5], [5, 6, 5]]])
    faces = np.vstack([fins.faces, [[5, 6, 7]]])
    detached = navis.Mesh({"vertices": verts, "faces": faces}, process=False)

    assert _mesh_component_labels(detached, connectivity="manifold")[1] == 4

    # Landing on vertex 2, which only fin 0 uses, joins just those two
    to_a_tip = detached.copy()
    to_a_tip.extra_edges = [[2, 5]]
    assert _mesh_component_labels(to_a_tip, connectivity="manifold")[1] == 3

    # Landing on the spine, which all three fins use, joins all four
    to_the_spine = detached.copy()
    to_the_spine.extra_edges = [[0, 5]]
    assert _mesh_component_labels(to_the_spine, connectivity="manifold")[1] == 1


@pytest.mark.parametrize("connectivity", FACE_READINGS)
def test_healed_mesh_counts_its_bridges(mesh, connectivity):
    """`heal_mesh` welds the vertex components - each bridge joining two faces.

    It knows nothing of the finer face components, so what it leaves behind is
    the pinches (and, for `"manifold"`, the seams): still apart. Each bridge
    takes *at least* one component with it - more where it lands on a junction,
    which is why this is an inequality (13 bridges: 24 -> 11 face components,
    but 502 -> 487 manifold ones rather than 489).
    """
    before = _mesh_component_labels(mesh, connectivity=connectivity)[1]
    healed = navis.heal_mesh(mesh)

    assert _mesh_component_labels(healed, connectivity="vertex")[1] == 1
    assert (
        1
        <= _mesh_component_labels(healed, connectivity=connectivity)[1]
        <= before - healed.n_extra_edges
    )


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("connectivity, expected", [("face", 24), ("manifold", 502)])
def test_components_come_back_as_vertices(mesh, connectivity, expected):
    """Face labels, but vertex indices - like every other neuron type here."""
    cc = _connected_components(mesh, connectivity=connectivity)
    all_vertices = np.concatenate(cc)

    assert len(cc) == expected
    assert all_vertices.max() < mesh.n_vertices
    # Every vertex used by a face is in there, and shared ones in several
    assert set(np.unique(all_vertices)) == set(np.unique(mesh.faces))
    assert len(all_vertices) > len(np.unique(all_vertices))


@pytest.mark.parametrize("connectivity", FACE_READINGS)
def test_components_are_the_pieces_they_label(pinched, connectivity):
    cc = sorted(
        (c.tolist() for c in _connected_components(pinched, connectivity=connectivity)),
        key=min,
    )

    assert cc == [[0, 1, 2], [2, 3, 4]]


# ---------------------------------------------------------------------------
# `drop_fluff`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("connectivity", FACE_READINGS)
def test_drop_fluff_drops_the_pinched_off_piece(pinched, connectivity):
    """Face 1 hangs off the rest by a single vertex - which itself stays."""
    verts = pinched.vertices.copy()

    kept = navis.drop_fluff(pinched, connectivity=connectivity, n_largest=1)

    assert kept.faces.tolist() == [[0, 1, 2]]
    assert np.array_equal(kept.vertices, verts[[0, 1, 2]])

    # Whereas by default the two triangles are one piece and both survive
    assert len(navis.drop_fluff(pinched).faces) == 2


def test_drop_fluff_drops_the_other_fins(fins):
    """A seam is fluff under `"manifold"` and nothing at all under `"face"`."""
    kept = navis.drop_fluff(fins, connectivity="manifold")

    assert len(kept.faces) == 1
    assert len(navis.drop_fluff(fins, connectivity="face").faces) == 3


def test_drop_fluff_keeps_less_the_finer_the_reading(mesh):
    """Same mesh, finer pieces, so the largest one is smaller."""
    kept = [
        navis.drop_fluff(mesh, connectivity=c).n_vertices
        for c in ("vertex", "face", "manifold")
    ]

    assert kept == sorted(kept, reverse=True)


@pytest.mark.parametrize("connectivity", FACE_READINGS)
@pytest.mark.parametrize("n_largest", [1, 2, 5])
def test_drop_fluff_keeps_the_n_largest_components(mesh, connectivity, n_largest):
    """Shared pinch vertices are kept once, not once per component."""
    cc = sorted(
        _connected_components(mesh, connectivity=connectivity), key=len, reverse=True
    )
    expected = np.unique(np.concatenate(cc[:n_largest]))

    kept = navis.drop_fluff(mesh, connectivity=connectivity, n_largest=n_largest)

    assert np.array_equal(kept.vertices, mesh.vertices[expected])


@pytest.mark.parametrize("connectivity", FACE_READINGS)
def test_drop_fluff_keep_size(mesh, connectivity):
    """`keep_size` counts vertices, as it does under vertex connectivity."""
    cc = _connected_components(mesh, connectivity=connectivity)
    expected = np.unique(np.concatenate([c for c in cc if len(c) >= 100]))

    kept = navis.drop_fluff(mesh, connectivity=connectivity, keep_size=100)

    assert np.array_equal(kept.vertices, mesh.vertices[expected])


@pytest.mark.parametrize("connectivity", FACE_READINGS)
def test_drop_fluff_on_a_mesh_with_no_faces(faceless, connectivity):
    """Nothing to keep, rather than an index error."""
    assert navis.drop_fluff(faceless, connectivity=connectivity).n_vertices == 0


# ---------------------------------------------------------------------------
# The `connectivity` argument itself
# ---------------------------------------------------------------------------


def test_defaults_per_neuron_type(mesh):
    assert _resolve_connectivity(mesh, None) == "vertex"
    assert _resolve_connectivity(navis.voxelize(mesh, pitch=2000), None) == 26
    assert _resolve_connectivity(navis.example_neurons(1, kind="skeleton"), None) is None


@pytest.mark.parametrize("connectivity", FACE_READINGS)
def test_a_value_meant_for_another_neuron_type_is_an_error(mesh, connectivity):
    with pytest.raises(ValueError):
        navis.drop_fluff(mesh, connectivity=26)

    with pytest.raises(ValueError):
        navis.drop_fluff(navis.voxelize(mesh, pitch=2000), connectivity=connectivity)


def test_ignored_where_it_means_nothing(caplog):
    """Skeletons come with their edges - saying so beats doing nothing quietly."""
    skeleton = navis.example_neurons(1, kind="skeleton")

    with caplog.at_level("WARNING"):
        navis.drop_fluff(skeleton, connectivity="manifold")

    assert "does not apply" in caplog.text
