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
    """Unreachable pairs are inf, not -1 (which is what fastcore returns)."""
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
