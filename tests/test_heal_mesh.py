"""Tests for `navis.heal_mesh`.

Note that the example mesh genuinely consists of several connected components
(one main body plus a handful of small bits), so it doubles as a real-world
fixture here - no artificial fragmentation needed. The exact counts depend on
whatever mesh is currently vendored, so the tests below derive them from the
fixture rather than hard-coding them.
"""

import navis
import numpy as np
import pytest
import trimesh as tm

from scipy.sparse.csgraph import minimum_spanning_tree

from navis.graph.graph_utils import _mesh_component_labels

@pytest.fixture
def mesh():
    return navis.example_neurons(1, kind="mesh")


@pytest.fixture
def two_boxes():
    """Two unit cubes 10 units apart - one bridge of known length."""
    a = tm.creation.box((1, 1, 1))
    b = tm.creation.box((1, 1, 1))
    b.apply_translation([10, 0, 0])
    comb = tm.util.concatenate([a, b])
    return navis.Mesh((np.asarray(comb.vertices), np.asarray(comb.faces)))


def bridge_lengths(x, edges):
    """Euclidean length of each `edges` pair of vertex indices into `x`."""
    if not len(edges):
        return np.zeros(0)
    return np.linalg.norm(np.diff(x.vertices[edges], axis=1)[:, 0], axis=1)


def added_length(x, healed):
    """Total length of the bridges `healed` added to `x`."""
    return float(bridge_lengths(x, healed.extra_edges[x.n_extra_edges :]).sum())


def frag_sizes(x):
    """Number of vertices per connected component."""
    labels, n = _mesh_component_labels(x)
    return np.bincount(labels, minlength=n)


def brute_force_mst(x):
    """Total added length of a true MST over the fragments.

    Builds the complete inter-fragment distance matrix by brute force (closest
    pair of vertices between every pair of fragments) and runs a global MST over
    it. This is the ground truth `heal_mesh` must reproduce.
    """
    labels, n = _mesh_component_labels(x)
    co = x.vertices
    frags = [np.where(labels == i)[0] for i in range(n)]

    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(
                co[frags[i]][:, None, :] - co[frags[j]][None, :, :], axis=-1
            ).min()
            dists[i, j] = dists[j, i] = d

    return minimum_spanning_tree(dists).toarray().sum()


def n_fragments(x):
    return _mesh_component_labels(x)[1]


# ---------------------------------------------------------------------------
# The basics
# ---------------------------------------------------------------------------


def test_heals(mesh):
    n_frags = n_fragments(mesh)
    assert n_frags > 1  # the example mesh is genuinely fragmented

    healed = navis.heal_mesh(mesh)

    assert n_fragments(healed) == 1
    assert healed.n_extra_edges == n_frags - 1  # one bridge per fragment merged


def test_geometry_is_untouched(mesh):
    """This is topological repair, not cosmetic surgery."""
    healed = navis.heal_mesh(mesh)

    assert np.array_equal(healed.vertices, mesh.vertices)
    assert np.array_equal(healed.faces, mesh.faces)
    assert healed.volume == mesh.volume
    assert healed.trimesh.area == mesh.trimesh.area
    # Extra edges must not leak into the surface
    assert len(healed.trimesh.edges_unique) == len(mesh.trimesh.edges_unique)


def test_is_minimum_spanning_tree(mesh):
    """Added length must match a brute-force MST over the fragments."""
    healed = navis.heal_mesh(mesh)
    assert added_length(mesh, healed) == pytest.approx(brute_force_mst(mesh))


def test_two_boxes(two_boxes):
    healed = navis.heal_mesh(two_boxes)

    assert healed.n_extra_edges == 1
    assert n_fragments(healed) == 1
    # Closest corners of two unit cubes 10 apart are 9 apart
    assert added_length(two_boxes, healed) == pytest.approx(9)


def test_bridges_connect_different_fragments(mesh):
    labels, _ = _mesh_component_labels(mesh)
    healed = navis.heal_mesh(mesh)

    a, b = healed.extra_edges[:, 0], healed.extra_edges[:, 1]
    assert (labels[a] != labels[b]).all()


def test_bridges_are_not_face_edges(mesh):
    """A bridge duplicating a face edge would be a no-op (and a bug)."""
    healed = navis.heal_mesh(mesh)

    face_edges = {tuple(e) for e in np.sort(mesh.trimesh.edges_unique, axis=1)}
    assert not any(tuple(e) in face_edges for e in healed.extra_edges)


def test_copy_semantics(mesh):
    healed = navis.heal_mesh(mesh)
    assert mesh.n_extra_edges == 0
    assert healed is not mesh

    inplace = mesh.copy()
    out = navis.heal_mesh(inplace, inplace=True)
    assert out is inplace
    assert inplace.n_extra_edges == n_fragments(mesh) - 1


def test_neuronlist(mesh):
    n_bridges = n_fragments(mesh) - 1
    nl = navis.NeuronList([mesh, mesh.copy()])
    healed = navis.heal_mesh(nl)

    assert [n.n_extra_edges for n in healed] == [n_bridges, n_bridges]
    assert [n.n_extra_edges for n in nl] == [0, 0]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_max_dist(mesh):
    """`max_dist` caps the length of any single bridge."""
    full = navis.heal_mesh(mesh)
    cutoff = np.median(bridge_lengths(mesh, full.extra_edges))

    limited = navis.heal_mesh(mesh, max_dist=cutoff)

    assert 0 < limited.n_extra_edges < full.n_extra_edges
    assert n_fragments(limited) > 1
    assert (bridge_lengths(mesh, limited.extra_edges) <= cutoff).all()


def test_max_dist_with_units(mesh):
    """Neurons with `.units` accept a string."""
    assert mesh.units.to("nm").magnitude == 8
    by_string = navis.heal_mesh(mesh, max_dist="500 nm")
    by_number = navis.heal_mesh(mesh, max_dist=500 / 8)

    assert by_string.n_extra_edges == by_number.n_extra_edges


def test_max_dist_too_small(mesh):
    """Nothing within reach = nothing to do (and no crash)."""
    healed = navis.heal_mesh(mesh, max_dist=1e-6)
    assert healed.n_extra_edges == 0


def test_min_size(mesh):
    """Fragments below `min_size` are ignored and stay disconnected."""
    sizes = frag_sizes(mesh)
    # A cutoff at the second-largest fragment leaves exactly the two largest
    min_size = sorted(sizes)[-2]
    assert (sizes >= min_size).sum() == 2

    healed = navis.heal_mesh(mesh, min_size=min_size)

    assert healed.n_extra_edges == 1
    assert n_fragments(healed) == len(sizes) - 1


def test_mask(mesh):
    """Only masked vertices may serve as bridge endpoints."""
    labels, _ = _mesh_component_labels(mesh)
    two_largest = np.argsort(frag_sizes(mesh))[-2:]
    sel = np.where(np.isin(labels, two_largest))[0]

    by_index = navis.heal_mesh(mesh, mask=sel)
    assert by_index.n_extra_edges == 1

    bool_mask = np.zeros(mesh.n_vertices, dtype=bool)
    bool_mask[sel] = True
    by_mask = navis.heal_mesh(mesh, mask=bool_mask)
    assert np.array_equal(by_mask.extra_edges, by_index.extra_edges)

    # The bridge must sit between the two masked fragments
    assert set(labels[by_index.extra_edges[0]]) == set(two_largest)


def test_mask_wrong_length(mesh):
    with pytest.raises(ValueError):
        navis.heal_mesh(mesh, mask=np.ones(5, dtype=bool))


def test_keep_largest(mesh):
    """Whatever is left fragmented after healing gets dropped."""
    sizes = sorted(frag_sizes(mesh))

    healed = navis.heal_mesh(mesh, min_size=sizes[-2], keep_largest=True)

    assert n_fragments(healed) == 1
    assert healed.n_vertices == sizes[-1] + sizes[-2]
    # The bridge must have survived (and been remapped)
    assert healed.n_extra_edges == 1


def test_keep_largest_noop_when_connected(mesh):
    healed = navis.heal_mesh(mesh, keep_largest=True)
    assert healed.n_vertices == mesh.n_vertices


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_already_connected_is_noop(two_boxes):
    healed = navis.heal_mesh(two_boxes)
    again = navis.heal_mesh(healed)

    assert np.array_equal(again.extra_edges, healed.extra_edges)


def test_existing_extra_edges_are_kept(mesh):
    """Pre-existing bridges count as connections and must survive."""
    labels, n_frags = _mesh_component_labels(mesh)
    a = int(np.where(labels == labels[0])[0][0])
    b = int(np.where(labels != labels[0])[0][0])

    pre = mesh.copy()
    pre.extra_edges = [[a, b]]

    healed = navis.heal_mesh(pre)

    assert n_fragments(healed) == 1
    # One fewer bridge needed - those two fragments were already connected -
    # but the pre-existing one is kept, so the total is unchanged
    assert healed.n_extra_edges == (n_frags - 1) - 1 + 1
    assert [a, b] in healed.extra_edges.tolist()


def test_empty_mesh():
    assert navis.heal_mesh(navis.Mesh(None)).n_extra_edges == 0


def test_wrong_type():
    skeleton = navis.example_neurons(1, kind="skeleton")
    with pytest.raises(TypeError):
        navis.heal_mesh(skeleton)


def test_heal_skeleton_points_at_heal_mesh(mesh):
    with pytest.raises(TypeError, match="heal_mesh"):
        navis.heal_skeleton(mesh)
