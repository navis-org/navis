"""Tests for `navis.connected_components` - the one primitive everything else
in the component family now goes through.

The properties worth pinning down are the ones the API *promises*: the labels
are a partition of the elements, they are sorted largest-first (so `== 0` is
always the biggest piece), `element` decides what they are of, and `mask`
labels the induced sub-neuron rather than the whole thing.
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest
import trimesh as tm

import navis

from navis._deprecated import DEPRECATED_PROPERTIES, reset_deprecation_warnings
from navis.graph.graph_utils import _component_ids


@pytest.fixture(scope="module")
def skeleton():
    return navis.example_neurons(1, kind="skeleton")


@pytest.fixture(scope="module")
def mesh():
    """The example mesh - 14 vertex components, 24 face, 502 manifold."""
    return navis.example_neurons(1, kind="mesh")


@pytest.fixture(scope="module")
def fragmented(skeleton):
    """A skeleton in three pieces, by orphaning two nodes."""
    n = skeleton.copy()
    ids = n.nodes.node_id.values
    n.nodes.loc[n.nodes.node_id.isin([ids[500], ids[1500]]), "parent_id"] = -1
    n._clear_temp_attr()
    return n


@pytest.fixture
def voxels():
    """One big blob (64 voxels), one small (8) and one speck (1)."""
    coords = np.vstack(
        [
            np.argwhere(np.ones((4, 4, 4))),
            np.argwhere(np.ones((2, 2, 2))) + 20,
            np.array([[40, 40, 40]]),
        ]
    )
    return navis.Voxels(coords, units="8 nm")


@pytest.fixture
def dotprops(skeleton):
    return navis.make_dotprops(skeleton, k=5)


# ---------------------------------------------------------------------------
# The shape of the answer
# ---------------------------------------------------------------------------


def test_labels_are_a_partition(fragmented):
    """One label per node, contiguous from zero, nothing left over."""
    labels = navis.connected_components(fragmented)

    assert labels.shape == (fragmented.n_nodes,)
    assert set(np.unique(labels).tolist()) == set(range(labels.max() + 1))
    assert (labels >= 0).all()


@pytest.mark.parametrize("kind", ["skeleton", "mesh", "dotprops", "voxels"])
def test_every_type_answers(kind, request):
    """The whole point of promoting this: all four types, one call."""
    x = request.getfixturevalue(kind)
    labels = navis.connected_components(x)

    assert labels.ndim == 1
    assert labels.max() + 1 == x.n_components


def test_sorted_largest_first(mesh):
    counts = np.bincount(navis.connected_components(mesh))

    assert (np.diff(counts) <= 0).all(), "components are not size-sorted"
    assert counts[0] == max(counts)


def test_label_zero_is_the_largest_component(fragmented):
    labels = navis.connected_components(fragmented)
    sizes = np.bincount(labels)

    assert int((labels == 0).sum()) == sizes.max()


def test_sorting_is_deterministic_across_ties():
    """Equal-size components are ordered by their first element, not by luck."""
    # Four isolated nodes: four components of one node each, all tied
    sk = navis.Skeleton(
        pd.DataFrame(
            {
                "node_id": [10, 11, 12, 13],
                "parent_id": [-1, -1, -1, -1],
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0, 0.0],
            }
        )
    )

    assert navis.connected_components(sk).tolist() == [0, 1, 2, 3]


def test_empty_neuron():
    """An empty node table has no components - and must not raise saying so."""
    n = navis.Skeleton(
        pd.DataFrame(
            {
                "node_id": pd.Series([], dtype=np.int64),
                "parent_id": pd.Series([], dtype=np.int64),
                "x": pd.Series([], dtype=float),
                "y": pd.Series([], dtype=float),
                "z": pd.Series([], dtype=float),
            }
        )
    )

    assert navis.connected_components(n).shape == (0,)
    assert n.n_components == 0
    assert _component_ids(n) == []


# ---------------------------------------------------------------------------
# `element`
# ---------------------------------------------------------------------------


def test_element_defaults_to_vertices_for_a_mesh(mesh):
    labels = navis.connected_components(mesh)

    assert labels.shape == (mesh.n_vertices,)


def test_element_face_labels_faces(mesh):
    labels = navis.connected_components(mesh, element="face")

    assert labels.shape == (len(mesh.faces),)
    # A face's corners are always in one vertex component, so switching the
    # element cannot change how many components there are
    assert labels.max() == navis.connected_components(mesh).max()


@pytest.mark.parametrize("connectivity", ["face", "manifold"])
def test_face_connectivity_defaults_to_face_elements(mesh, connectivity):
    """Only a labelling of faces is a partition under these readings."""
    labels = navis.connected_components(mesh, connectivity=connectivity)

    assert labels.shape == (len(mesh.faces),)
    assert (labels >= 0).all()


@pytest.mark.parametrize("connectivity", ["face", "manifold"])
def test_face_connectivity_as_vertices_takes_the_largest(mesh, connectivity):
    """A pinch vertex is in several components; it gets the biggest one."""
    per_face = navis.connected_components(mesh, connectivity=connectivity)
    per_vertex = navis.connected_components(
        mesh, connectivity=connectivity, element="vertex"
    )

    assert per_vertex.shape == (mesh.n_vertices,)
    # Every vertex a face uses is labelled, and with the smallest (= largest
    # component) label among the faces meeting there
    for face, label in zip(np.asarray(mesh.faces), per_face):
        assert (per_vertex[face] <= label).all()

    used = np.unique(np.asarray(mesh.faces))
    assert (per_vertex[used] >= 0).all()


def test_element_rejects_a_foreign_value(skeleton, mesh):
    with pytest.raises(ValueError, match="element"):
        navis.connected_components(skeleton, element="vertex")
    with pytest.raises(ValueError, match="element"):
        navis.connected_components(mesh, element="node")


def test_vertices_no_face_uses_are_not_a_component():
    """Under a face reading a loose vertex belongs to nothing - hence -1."""
    # Through trimesh with `process=False`: `navis.Mesh` drops the loose vertex
    # on construction, which is the very thing under test here
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=float)
    m = tm.Trimesh(verts, np.array([[0, 1, 2]]), process=False)

    labels = navis.connected_components(m, connectivity="face", element="vertex")

    assert labels.tolist() == [0, 0, 0, -1]


# ---------------------------------------------------------------------------
# `mask`
# ---------------------------------------------------------------------------


def test_mask_excludes_and_disconnects(skeleton):
    """Masked-out nodes get -1 *and* stop conducting."""
    ids = skeleton.nodes.node_id.values
    keep = ids[::2]

    labels = navis.connected_components(skeleton, mask=keep)

    assert int((labels == -1).sum()) == len(ids) - len(keep)
    # Dropping every other node of a single tree shatters it
    assert labels.max() + 1 > skeleton.n_components


# N.B. the partition a mask produces is checked against a plain-Python
# union-find ground truth in `tests/test_graph_primitives.py` - what is left to
# pin down here is the `-1`s and the non-skeleton types.


def test_mask_accepts_a_boolean_array(skeleton):
    keep = np.zeros(skeleton.n_nodes, dtype=bool)
    keep[:100] = True

    by_bool = navis.connected_components(skeleton, mask=keep)
    by_id = navis.connected_components(
        skeleton, mask=skeleton.nodes.node_id.values[:100]
    )

    assert np.array_equal(by_bool, by_id)


def test_empty_mask_keeps_nothing(skeleton):
    labels = navis.connected_components(skeleton, mask=[])

    assert (labels == -1).all()
    assert _component_ids(skeleton, mask=[]) == []


def test_mask_on_a_mesh_drops_faces(mesh):
    """A face survives only if all three corners do."""
    keep = np.zeros(mesh.n_vertices, dtype=bool)
    keep[np.unique(np.asarray(mesh.faces)[:50])] = True

    labels = navis.connected_components(mesh, mask=keep)

    assert (labels[~keep] == -1).all()
    assert (labels[keep] >= 0).all()


def test_mask_on_voxels(voxels):
    keep = np.zeros(len(voxels.voxels), dtype=bool)
    keep[:64] = True  # the big blob only

    labels = navis.connected_components(voxels, mask=keep)

    assert labels.max() + 1 == 1
    assert int((labels == -1).sum()) == 9


# ---------------------------------------------------------------------------
# Neuron methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["skeleton", "mesh", "dotprops", "voxels"])
def test_method_matches_function(kind, request):
    x = request.getfixturevalue(kind)

    assert np.array_equal(x.connected_components(), navis.connected_components(x))


def test_method_passes_kwargs_through(mesh):
    assert mesh.connected_components(element="face").shape == (len(mesh.faces),)


def test_n_components(fragmented, mesh):
    assert fragmented.n_components == 3
    assert mesh.n_components == navis.connected_components(mesh).max() + 1


# ---------------------------------------------------------------------------
# The rest of the family agrees with the primitive
# ---------------------------------------------------------------------------


def test_split_components_agrees(fragmented):
    labels = navis.connected_components(fragmented)
    frags = navis.split_components(fragmented)

    assert len(frags) == labels.max() + 1
    # Largest first, same as the labels
    assert [f.n_nodes for f in frags] == np.bincount(labels).tolist()


def test_split_components_min_size(fragmented):
    sizes = np.bincount(navis.connected_components(fragmented))
    cutoff = int(sorted(sizes)[-2])

    frags = navis.split_components(fragmented, min_size=cutoff)

    assert len(frags) == int((sizes >= cutoff).sum())


def test_drop_fluff_keeps_label_zero(mesh):
    kept = navis.drop_fluff(mesh)
    labels = navis.connected_components(mesh)

    assert kept.n_vertices == int((labels == 0).sum())


def test_heal_keep_largest_keeps_label_zero(fragmented):
    labels = navis.connected_components(fragmented)

    healed = navis.heal_skeleton(fragmented, max_dist=1, keep_largest=True)

    assert healed.n_nodes == int((labels == 0).sum())


# ---------------------------------------------------------------------------
# Deprecated spellings
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_warnings():
    """Warn-once state is global; each test needs it clean to observe a warning."""
    reset_deprecation_warnings()


@pytest.mark.parametrize(
    "old,new", sorted(DEPRECATED_PROPERTIES["Skeleton"].items())
)
def test_deprecated_skeleton_properties(skeleton, old, new):
    # The match asserts the warning names *both* spellings; `new` may carry
    # call parens (`connected_components()`), which are not regex-safe.
    pattern = rf"`Skeleton\.{old}`.*`Skeleton\.{re.escape(new)}`"
    with pytest.warns(DeprecationWarning, match=pattern):
        getattr(skeleton, old)


def test_deprecated_subtrees_still_returns_node_ids(fragmented):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        subtrees = fragmented.subtrees

    sizes = np.bincount(navis.connected_components(fragmented))

    assert [len(s) for s in subtrees] == sizes.tolist()
    assert set(subtrees[0].tolist()) <= set(fragmented.nodes.node_id.values.tolist())


# N.B. the renamed *functions* (`break_fragments`, `split_into_fragments`) are
# covered by `tests/test_deprecated_names.py`, which parametrises off the same
# `_deprecated` tables and also checks warns-once and `-W error`.
