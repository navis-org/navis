"""Tests for the code paths that route through navis-fastcore.

Most of these functions have two implementations - a fastcore fast path and an
igraph/scipy fallback - and CI normally only ever exercises the former. The
tests here pin the *behaviour* so the two cannot drift apart, and they check the
fallback explicitly by monkeypatching `utils.fastcore` to None.
"""

import numpy as np
import pandas as pd
import pytest

import navis
from navis import graph as G
from navis import utils


@pytest.fixture
def n():
    x = navis.example_neurons(1, kind="skeleton")
    navis.graph.classify_nodes(x)
    return x


@pytest.fixture
def no_fastcore(monkeypatch):
    """Force the igraph/scipy fallback."""
    monkeypatch.setattr(utils, "fastcore", None)


# ---------------------------------------------------------------- geodesic_matrix


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("weight", ["weight", None])
def test_geodesic_matrix_to(n, directed, weight):
    """`to_` must equal slicing the columns of a `from_`-only matrix."""
    leafs = n.nodes[n.nodes.type == "end"].node_id.values[:50]
    bps = n.nodes[n.nodes.type == "branch"].node_id.values[:30]

    full = G.geodesic_matrix(n, from_=leafs, directed=directed, weight=weight)
    block = G.geodesic_matrix(
        n, from_=leafs, to_=bps, directed=directed, weight=weight
    )

    assert block.shape == (len(np.unique(leafs)), len(np.unique(bps)))
    # Compare label-wise - that's the contract, not the raw column order
    expected = full[np.unique(bps)]
    assert np.allclose(block.values, expected.values, equal_nan=True)
    assert list(block.columns) == list(np.unique(bps))
    assert list(block.index) == list(np.unique(leafs))


def test_geodesic_matrix_to_matches_fallback(n, monkeypatch):
    """The fastcore path and the scipy fallback must agree - including order."""
    leafs = n.nodes[n.nodes.type == "end"].node_id.values[:40]
    bps = n.nodes[n.nodes.type == "branch"].node_id.values[:20]

    fast = G.geodesic_matrix(n, from_=leafs, to_=bps)

    m = n.copy()
    m._clear_temp_attr()
    monkeypatch.setattr(utils, "fastcore", None)
    slow = G.geodesic_matrix(m, from_=leafs, to_=bps)

    assert list(fast.index) == list(slow.index)
    assert list(fast.columns) == list(slow.columns)
    assert np.allclose(fast.values, slow.values, rtol=1e-5, atol=1e-5)


def test_geodesic_matrix_to_missing_id(n):
    with pytest.raises(ValueError):
        G.geodesic_matrix(n, to_=[-1])


# --------------------------------------------------------------------- distal_to


def test_distal_to_fastcore_matches_fallback(n, monkeypatch):
    leafs = n.nodes[n.nodes.type == "end"].node_id.values[:50]
    bps = n.nodes[n.nodes.type == "branch"].node_id.values[:30]

    fast = navis.distal_to(n, leafs, bps)

    m = n.copy()
    m._clear_temp_attr()
    monkeypatch.setattr(utils, "fastcore", None)
    slow = navis.distal_to(m, leafs, bps)

    assert list(fast.index) == list(slow.index)
    assert list(fast.columns) == list(slow.columns)
    assert (fast.values == slow.values).all()


def test_distal_to_scalar(n):
    """A single node pair must still come back as a plain bool."""
    root = n.root[0]
    leaf = n.nodes[n.nodes.type == "end"].node_id.values[0]

    assert navis.distal_to(n, int(leaf), int(root)) is np.True_ or bool(
        navis.distal_to(n, int(leaf), int(root))
    )
    # ...and not the other way around
    assert not bool(navis.distal_to(n, int(root), int(leaf)))


# ------------------------------------------------------------------ dist_between


def test_dist_between_scalar_unchanged(n):
    """Scalar in -> scalar out. This is the pre-existing contract."""
    a, b = n.nodes.node_id.values[:2]
    d = G.dist_between(n, int(a), int(b))
    assert isinstance(d, float)
    assert d > 0


def test_dist_between_pairs(n):
    """Matched arrays must give the same answers as looping one pair at a time."""
    rng = np.random.default_rng(0)
    ids = n.nodes.node_id.values
    a, b = rng.choice(ids, 50), rng.choice(ids, 50)

    batch = G.dist_between(n, a, b)
    loop = np.array([G.dist_between(n, int(i), int(j)) for i, j in zip(a, b)])

    assert isinstance(batch, np.ndarray)
    assert batch.shape == (50,)
    assert np.allclose(batch, loop, rtol=1e-4)


def test_dist_between_pairs_fallback(n, monkeypatch):
    """The igraph fallback must agree with the fastcore path."""
    rng = np.random.default_rng(1)
    ids = n.nodes.node_id.values
    a, b = rng.choice(ids, 50), rng.choice(ids, 50)

    fast = G.dist_between(n, a, b)

    m = n.copy()
    m._clear_temp_attr()
    monkeypatch.setattr(utils, "fastcore", None)
    slow = G.dist_between(m, a, b)

    assert np.allclose(fast, slow, rtol=1e-4)


def test_dist_between_broadcast(n):
    """One node against many."""
    ids = n.nodes.node_id.values[:20]
    root = int(n.root[0])

    d = G.dist_between(n, root, ids)
    assert d.shape == (20,)

    expected = np.array([G.dist_between(n, root, int(i)) for i in ids])
    assert np.allclose(d, expected, rtol=1e-4)


def test_dist_between_length_mismatch(n):
    ids = n.nodes.node_id.values
    with pytest.raises(ValueError):
        G.dist_between(n, ids[:5], ids[:3])


def test_dist_between_unreachable():
    """Unreachable pairs are inf, not -1 (which is what fastcore returns).

    Regression test for a navis-fastcore bug (fixed in 0.6.0) where
    `geodesic_pairs` returned a bogus `1.0` instead of `-1` for pairs sitting in
    different fragments.
    """
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3],
            "parent_id": [-1, 0, -1, 2],  # two separate fragments
            "x": [0.0, 1.0, 10.0, 11.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
        }
    )
    x = navis.TreeNeuron(nodes)
    d = G.dist_between(x, [0, 0], [1, 3])
    assert np.isfinite(d[0])
    assert np.isinf(d[1])


# -------------------------------------------------------------- stitch_skeletons


def _two_fragments():
    """Two disjoint bars, 10 units apart along x.

    Deliberately NOT built with `cut_skeleton`: that duplicates the cut node, so
    the fragments share a node ID and `stitch_skeletons` remaps it - which makes
    a caller-supplied list of node IDs ambiguous (see its docstring).
    """
    def bar(ids, x0):
        return navis.TreeNeuron(
            pd.DataFrame(
                {
                    "node_id": ids,
                    "parent_id": [-1] + list(ids[:-1]),
                    "x": np.arange(len(ids), dtype=float) + x0,
                    "y": np.zeros(len(ids)),
                    "z": np.zeros(len(ids)),
                }
            )
        )

    return navis.NeuronList([bar([0, 1, 2], 0.0), bar([10, 11, 12], 10.0)])


def test_stitch_skeletons_node_list():
    """`method=<list of node IDs>` is documented but used to raise AssertionError."""
    frags = _two_fragments()
    allowed = [2, 10]  # the two facing tips

    stitched = navis.stitch_skeletons(frags, method=allowed)

    assert isinstance(stitched, navis.TreeNeuron)
    assert stitched.n_nodes == 6
    assert len(stitched.root) == 1  # fragments joined up


def test_stitch_skeletons_node_list_restricts():
    """Nodes outside the list must not be used to bridge."""
    frags = _two_fragments()

    before = {
        frozenset((c, p))
        for f in frags
        for c, p in zip(f.nodes.node_id.values, f.nodes.parent_id.values)
        if p >= 0
    }

    # Force the bridge to use node 0 (the FAR tip of the first bar) even though
    # node 2 is much closer to the second fragment.
    stitched = navis.stitch_skeletons(frags, method=[0, 10])

    after = {
        frozenset((c, p))
        for c, p in zip(stitched.nodes.node_id.values, stitched.nodes.parent_id.values)
        if p >= 0
    }
    new_edges = after - before
    assert len(new_edges) == 1
    assert new_edges == {frozenset((0, 10))}


def test_stitch_skeletons_bad_method():
    n = navis.example_neurons(1, kind="skeleton")
    frags = navis.cut_skeleton(n, int(n.nodes.node_id.values[100]))
    with pytest.raises(ValueError):
        navis.stitch_skeletons(frags, method="NOT_A_METHOD")


# ------------------------------------------------------------------------ plot1d


def test_plot1d_segment_lengths():
    """The bars must add up to the neuron's cable length.

    Used to sum to only ~28% of it: the code took the first *two* nodes of each
    segment rather than the first and the last.
    """
    import matplotlib

    matplotlib.use("Agg")

    n = navis.example_neurons(1, kind="skeleton")
    ax = navis.plot1d(n)

    assert np.isclose(ax.get_xlim()[1], n.cable_length, rtol=1e-4)


# ---------------------------------------------------------------- small_segments


def test_break_segments_order_matches_fallback(n, monkeypatch):
    """Both backends must return the segments in the same order.

    The order ends up in the output of consumers that `enumerate()` the segments
    (`segment_analysis`, the NEURON interface, `resample_skeleton`), so it must not
    depend on whether fastcore is installed. The igraph branch used to walk a Python
    set, i.e. in hash order.
    """
    fast = [list(s) for s in G._break_segments(n)]

    m = n.copy()
    m._clear_temp_attr()
    monkeypatch.setattr(utils, "fastcore", None)
    slow = [list(s) for s in G._break_segments(m)]

    assert fast == slow


def test_break_segments_order_is_node_table_order(monkeypatch):
    """Segments come back ordered by the node table position of their (distal) seed.

    Note this is *position*, not node ID - so use a node table that is not sorted by
    node ID, otherwise the two are indistinguishable.
    """
    nodes = pd.DataFrame(
        {
            "node_id": [50, 90, 7, 20, 3],  # deliberately unsorted
            "parent_id": [-1, 7, 50, 90, 7],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0] * 5,
            "z": [0.0] * 5,
        }
    )
    # seeds are 7 (branch, row 2), 20 (leaf, row 3) and 3 (leaf, row 4)
    expected = [[7, 50], [20, 90, 7], [3, 7]]

    got = [list(s) for s in G._break_segments(navis.TreeNeuron(nodes.copy()))]
    assert got == expected

    monkeypatch.setattr(utils, "fastcore", None)
    got = [list(s) for s in G._break_segments(navis.TreeNeuron(nodes.copy()))]
    assert got == expected


def test_segment_analysis_row_order_matches_fallback(n, monkeypatch):
    """The row order of `segment_analysis` must not depend on the backend."""
    fast = navis.segment_analysis(n.copy())

    m = n.copy()
    m._clear_temp_attr()
    monkeypatch.setattr(utils, "fastcore", None)
    slow = navis.segment_analysis(m)

    assert fast.equals(slow)


# ----------------------------------------------------------------- classify_nodes


@pytest.mark.parametrize("categorical", [True, False])
def test_classify_nodes_matches_fallback(n, categorical, monkeypatch):
    fast = navis.graph.classify_nodes(n.copy(), categorical=categorical).nodes.type

    monkeypatch.setattr(utils, "fastcore", None)
    slow = navis.graph.classify_nodes(n.copy(), categorical=categorical).nodes.type

    assert (np.asarray(fast).astype(str) == np.asarray(slow).astype(str)).all()
    assert fast.dtype == slow.dtype


@pytest.mark.parametrize(
    "node_ids,parent_ids,expected",
    [
        # a root, a branch point and two leafs
        ([0, 1, 2, 3], [-1, 0, 1, 1], ["root", "branch", "end", "end"]),
        # a plain, unbranched chain -> the middle node is a slab
        ([0, 1, 2], [-1, 0, 1], ["root", "slab", "end"]),
        # two separate fragments -> two roots
        ([0, 1, 2, 3], [-1, 0, -1, 2], ["root", "end", "root", "end"]),
        # a single, isolated node is a root (not an end)
        ([0], [-1], ["root"]),
        # node IDs need be neither small nor sorted
        ([100, 7, 55, 3], [-1, 100, 7, 7], ["root", "branch", "end", "end"]),
    ],
)
def test_classify_nodes_topologies(node_ids, parent_ids, expected, monkeypatch):
    """The fastcore and numpy paths must agree on the awkward topologies."""
    def build():
        return navis.TreeNeuron(
            pd.DataFrame(
                {
                    "node_id": node_ids,
                    "parent_id": parent_ids,
                    "x": np.arange(len(node_ids), dtype=float),
                    "y": np.zeros(len(node_ids)),
                    "z": np.zeros(len(node_ids)),
                }
            )
        )

    fast = list(navis.graph.classify_nodes(build()).nodes.type.astype(str))
    assert fast == expected

    monkeypatch.setattr(utils, "fastcore", None)
    slow = list(navis.graph.classify_nodes(build()).nodes.type.astype(str))
    assert slow == expected


def test_classify_nodes_uint64(monkeypatch):
    """uint64 node IDs used to trip up `np.isin` - make sure both paths cope."""
    nodes = pd.DataFrame(
        {
            "node_id": np.array([0, 1, 2, 3], dtype=np.uint64),
            "parent_id": np.array([-1, 0, 1, 1], dtype=np.int64),
            "x": np.arange(4, dtype=float),
            "y": np.zeros(4),
            "z": np.zeros(4),
        }
    )
    expected = ["root", "branch", "end", "end"]
    assert list(navis.graph.classify_nodes(navis.TreeNeuron(nodes)).nodes.type.astype(str)) == expected

    monkeypatch.setattr(utils, "fastcore", None)
    assert list(navis.graph.classify_nodes(navis.TreeNeuron(nodes)).nodes.type.astype(str)) == expected


# ------------------------------------------------------- geodesic_matrix (meshes)


@pytest.mark.parametrize("weight", ["weight", None])
def test_geodesic_matrix_mesh_matches_fallback(weight, monkeypatch):
    """The mesh fastcore path and the scipy fallback must agree."""
    m = navis.example_neurons(1, kind="mesh")
    rng = np.random.default_rng(0)
    src = np.sort(rng.choice(len(m.vertices), 40, replace=False))
    tgt = np.sort(rng.choice(len(m.vertices), 25, replace=False))

    fast = G.geodesic_matrix(m, from_=src, to_=tgt, weight=weight)

    q = m.copy()
    q._clear_temp_attr()
    monkeypatch.setattr(utils, "fastcore", None)
    slow = G.geodesic_matrix(q, from_=src, to_=tgt, weight=weight)

    assert list(fast.index) == list(slow.index)
    assert list(fast.columns) == list(slow.columns)
    assert np.allclose(fast.values, slow.values, rtol=1e-4, atol=1e-3, equal_nan=True)


def test_geodesic_matrix_mesh_limit(monkeypatch):
    """`limit` must mark the same pairs as unreachable in both backends."""
    m = navis.example_neurons(1, kind="mesh")
    rng = np.random.default_rng(1)
    src = np.sort(rng.choice(len(m.vertices), 40, replace=False))

    fast = G.geodesic_matrix(m, from_=src, limit=5000)

    q = m.copy()
    q._clear_temp_attr()
    monkeypatch.setattr(utils, "fastcore", None)
    slow = G.geodesic_matrix(q, from_=src, limit=5000)

    assert (np.isfinite(fast.values) == np.isfinite(slow.values)).all()
    assert np.isfinite(fast.values).any()  # ...and the limit isn't cutting everything
    assert np.allclose(fast.values, slow.values, rtol=1e-4, atol=1e-3, equal_nan=True)


# -------------------------------------------------------------- _subtree_height


@pytest.mark.parametrize("weight", ["weight", None])
def test_subtree_height_matches_fallback(n, weight, monkeypatch):
    from navis.morpho.manipulation import _subtree_height

    fast = _subtree_height(n, weight=weight)

    m = n.copy()
    m._clear_temp_attr()
    monkeypatch.setattr(utils, "fastcore", None)
    slow = _subtree_height(m, weight=weight)

    assert list(fast.index) == list(slow.index)
    assert np.allclose(fast.values, slow.values, rtol=1e-4, atol=1e-4)


def test_subtree_height_definition(monkeypatch):
    """Height = distance down to the farthest leaf below. Leafs are 0."""
    #  0 - 1 - 2 - 3     (3 is 3 hops below 0)
    #       \- 4
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3, 4],
            "parent_id": [-1, 0, 1, 2, 1],
            "x": [0.0, 1.0, 2.0, 3.0, 2.0],
            "y": [0.0, 0.0, 0.0, 0.0, 5.0],
            "z": [0.0] * 5,
        }
    )
    expected = [3.0, 2.0, 1.0, 0.0, 0.0]  # hop counts

    from navis.morpho.manipulation import _subtree_height

    x = navis.TreeNeuron(nodes)
    assert list(_subtree_height(x, weight=None).loc[[0, 1, 2, 3, 4]]) == expected

    monkeypatch.setattr(utils, "fastcore", None)
    x = navis.TreeNeuron(nodes)
    assert list(_subtree_height(x, weight=None).loc[[0, 1, 2, 3, 4]]) == expected


def test_subtree_height_fragmented(monkeypatch):
    """Each fragment's root gets the height of its own component."""
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3],
            "parent_id": [-1, 0, -1, 2],  # two fragments
            "x": [0.0, 1.0, 10.0, 11.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0] * 4,
        }
    )
    from navis.morpho.manipulation import _subtree_height

    fast = _subtree_height(navis.TreeNeuron(nodes), weight="weight")
    monkeypatch.setattr(utils, "fastcore", None)
    slow = _subtree_height(navis.TreeNeuron(nodes), weight="weight")

    assert np.allclose(fast.values, slow.values, rtol=1e-4)
    assert np.allclose(fast.loc[[0, 1, 2, 3]].values, [1.0, 0.0, 1.0, 0.0], rtol=1e-4)


# -------------------------------------------------------------- longest_neurite


@pytest.mark.parametrize("k", [1, 2, 3])
def test_longest_neurite_matches_fallback(n, k, monkeypatch):
    """A diameter has two ends and float32 rounding decides which one wins.

    Picking the other end reroots the neuron elsewhere, which changes the segment
    decomposition (and thus the output) for k >= 2. Pin both backends to the same
    choice.

    NOTE: `from_root=True` is the default and skips the geodesic branch entirely,
    so it *must* be False here or this tests nothing.
    """
    fast = navis.longest_neurite(n, n=k, from_root=False, inplace=False)

    m = n.copy()
    m._clear_temp_attr()
    monkeypatch.setattr(utils, "fastcore", None)
    slow = navis.longest_neurite(m, n=k, from_root=False, inplace=False)

    assert list(fast.root) == list(slow.root)
    assert sorted(fast.nodes.node_id) == sorted(slow.nodes.node_id)
    assert np.isclose(fast.cable_length, slow.cable_length, rtol=1e-4)
