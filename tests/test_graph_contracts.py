"""Behavioural contracts for the graph primitives fastcore implements.

These were differential tests: every function was run with and without
navis-fastcore and the two answers compared. Both fallbacks and oracle are gone
(fastcore is a hard requirement now) - and the oracle was the weaker one anyway,
since two implementations can agree and both be wrong. `FASTCORE_DISCREPANCIES.md`
has a worked example of exactly that.

What is here instead: definitions checked on hand-computed topologies, internal
consistency (a block of a matrix must equal that block of the whole matrix), and
- where a reference is genuinely useful - an oracle computed *in the test* with
scipy. Note the difference from a fallback: this reference is a few lines in one
test, not a shipped code path that has to be kept bit-identical forever.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import navis
from navis import graph as G


@pytest.fixture
def n():
    x = navis.example_neurons(1, kind="skeleton")
    navis.graph.classify_nodes(x)
    return x


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


def test_geodesic_matrix_is_a_metric(n):
    """Undirected geodesic distances must be symmetric, zero on the diagonal and
    obey the triangle inequality. Any of those failing is a real bug, whatever a
    second implementation might agree to.
    """
    rng = np.random.default_rng(0)
    ids = np.sort(rng.choice(n.nodes.node_id.values, 60, replace=False))
    d = G.geodesic_matrix(n, from_=ids, to_=ids, directed=False).values

    assert np.allclose(np.diag(d), 0)
    assert np.allclose(d, d.T, rtol=1e-5)
    # d[i, k] <= d[i, j] + d[j, k] for every j. Distances come back as float32,
    # so the slack has to scale with their magnitude rather than be absolute.
    tol = 1e-4 * float(d.max())
    assert (d[:, None, :] <= d[:, :, None] + d[None, :, :] + tol).all()


def test_geodesic_matrix_matches_a_scipy_reference(n):
    """Ground truth: Dijkstra over the skeleton's own edge list."""
    ids = n.nodes.node_id.values
    pos = pd.Index(ids)
    coords = n.nodes[["x", "y", "z"]].values.astype(float)

    has_parent = n.nodes.parent_id.values >= 0
    child = np.arange(len(ids))[has_parent]
    parent = pos.get_indexer(n.nodes.parent_id.values[has_parent])
    w = np.linalg.norm(coords[child] - coords[parent], axis=1)

    adj = csr_matrix(
        (np.concatenate([w, w]),
         (np.concatenate([child, parent]), np.concatenate([parent, child]))),
        shape=(len(ids), len(ids)),
    )

    rng = np.random.default_rng(1)
    src = np.sort(rng.choice(len(ids), 20, replace=False))
    expected = dijkstra(adj, directed=False, indices=src)

    got = G.geodesic_matrix(n, from_=ids[src], directed=False)
    assert np.allclose(got.loc[ids[src], ids].values, expected, rtol=1e-4, atol=1e-4)


def test_geodesic_matrix_to_missing_id(n):
    with pytest.raises(ValueError):
        G.geodesic_matrix(n, to_=[-1])


# --------------------------------------------------------------------- distal_to


def test_distal_to_matches_walking_to_the_root(n):
    """Ground truth: A is distal to B iff B is on A's path to the root."""
    parents = dict(
        zip(n.nodes.node_id.values.tolist(), n.nodes.parent_id.values.tolist())
    )

    def ancestors(a):
        out, seen = set(), set()
        while a >= 0 and a not in seen:
            out.add(a)
            seen.add(a)
            a = parents.get(a, -1)
        return out

    leafs = n.nodes[n.nodes.type == "end"].node_id.values[:25]
    bps = n.nodes[n.nodes.type == "branch"].node_id.values[:15]

    got = navis.distal_to(n, leafs, bps)
    for a in leafs:
        anc = ancestors(int(a))
        for b in bps:
            assert bool(got.loc[a, b]) == (int(b) in anc), (a, b)


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


def test_dist_between_agrees_with_geodesic_matrix(n):
    """The pairwise and the matrix route must give the same distances."""
    rng = np.random.default_rng(1)
    ids = np.sort(rng.choice(n.nodes.node_id.values, 30, replace=False))
    a = np.repeat(ids[:10], 3)
    b = np.tile(ids[-3:], 10)

    pairs = G.dist_between(n, a, b)
    mat = G.geodesic_matrix(n, from_=ids[:10], to_=ids[-3:], directed=False)

    assert np.allclose(pairs, [mat.loc[i, j] for i, j in zip(a, b)], rtol=1e-4)


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
    x = navis.Skeleton(nodes)
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
        return navis.Skeleton(
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

    assert isinstance(stitched, navis.Skeleton)
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


def test_break_segments_partition_the_neuron(n):
    """Every node must appear in a segment, and only the shared branch/root nodes
    may appear in more than one.
    """
    segs = [list(s) for s in G._break_segments(n)]
    flat = [node for s in segs for node in s]

    assert set(flat) == set(n.nodes.node_id.values.tolist())

    # A node may repeat only as the proximal end of a segment (i.e. a branch
    # point or root shared with the segment above it).
    seen = {}
    for s in segs:
        for i, node in enumerate(s):
            seen.setdefault(node, []).append(i == len(s) - 1)
    for node, positions in seen.items():
        if len(positions) > 1:
            assert sum(not last for last in positions) <= 1, (
                f"node {node} appears mid-segment more than once"
            )


def test_break_segments_order_is_node_table_order():
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

    got = [list(s) for s in G._break_segments(navis.Skeleton(nodes.copy()))]
    assert got == expected


def test_segment_analysis_rows_follow_small_segments(n):
    """`segment_analysis` enumerates `small_segments`, so its row order is that
    order - and its per-row length must be that segment's length.
    """
    segs = [list(s) for s in n.small_segments]
    res = navis.segment_analysis(n.copy())

    assert len(res) == len(segs)
    # Sum the child->parent distances along each segment. N.B. deliberately not
    # `G.segment_length` - that is the helper `segment_analysis` uses internally,
    # so it would compare the implementation against itself.
    w = dict(
        zip(
            n.nodes.node_id.values.tolist(),
            navis.morpho.mmetrics.parent_dist(n, root_dist=0),
        )
    )
    expected = [sum(w[i] for i in seg[:-1]) for seg in segs]
    assert np.allclose(res["length"].values, expected, rtol=1e-4)


# ----------------------------------------------------------------- classify_nodes


@pytest.mark.parametrize("categorical", [True, False])
def test_classify_nodes_matches_the_definitions(n, categorical):
    """root = no parent; end = nobody's parent; branch = >1 child; else slab."""
    x = navis.graph.classify_nodes(n.copy(), categorical=categorical)

    nid = x.nodes.node_id.values
    pid = x.nodes.parent_id.values
    got = np.asarray(x.nodes.type).astype(str)

    n_children = pd.Series(pid[pid >= 0]).value_counts()
    expected = np.where(
        pid < 0,
        "root",
        np.where(
            ~np.isin(nid, pid),
            "end",
            np.where(pd.Series(nid).map(n_children).fillna(0).values > 1,
                     "branch", "slab"),
        ),
    )
    assert (got == expected).all()


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
def test_classify_nodes_topologies(node_ids, parent_ids, expected):
    """Hand-computed classifications for the awkward topologies."""
    def build():
        return navis.Skeleton(
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

    assert list(navis.graph.classify_nodes(build()).nodes.type.astype(str)) == expected


def test_classify_nodes_uint64():
    """uint64 node IDs used to trip up `np.isin` - make sure they still work."""
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
    assert list(navis.graph.classify_nodes(navis.Skeleton(nodes)).nodes.type.astype(str)) == expected


# ------------------------------------------------------- geodesic_matrix (meshes)


@pytest.mark.parametrize("weight", ["weight", None])
def test_geodesic_matrix_mesh_matches_a_scipy_reference(weight):
    """Ground truth: Dijkstra over the mesh's own unique edges."""
    m = navis.example_neurons(1, kind="mesh")
    edges, lengths = navis.utils.mesh_unique_edges(m, return_lengths=True)
    w = lengths if weight == "weight" else np.ones(len(edges))

    n_verts = len(m.vertices)
    adj = csr_matrix(
        (np.concatenate([w, w]),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(n_verts, n_verts),
    )

    rng = np.random.default_rng(0)
    src = np.sort(rng.choice(n_verts, 20, replace=False))
    tgt = np.sort(rng.choice(n_verts, 15, replace=False))

    expected = dijkstra(adj, directed=False, indices=src)[:, tgt]
    got = G.geodesic_matrix(m, from_=src, to_=tgt, weight=weight)

    assert list(got.index) == list(src)
    assert list(got.columns) == list(tgt)
    assert np.allclose(got.values, expected, rtol=1e-4, atol=1e-3, equal_nan=True)


def test_geodesic_matrix_mesh_limit():
    """`limit` marks everything beyond it unreachable and nothing within it."""
    m = navis.example_neurons(1, kind="mesh")
    rng = np.random.default_rng(1)
    src = np.sort(rng.choice(len(m.vertices), 40, replace=False))

    limit = 5000
    capped = G.geodesic_matrix(m, from_=src, max_dist=limit)
    full = G.geodesic_matrix(m, from_=src)

    assert np.isfinite(capped.values).any(), "the limit is cutting everything"
    # Kept pairs keep their exact distance; dropped pairs were all over the limit
    kept = np.isfinite(capped.values)
    assert np.allclose(capped.values[kept], full.values[kept], rtol=1e-4, atol=1e-3)
    assert (full.values[~kept] > limit).all()


# -------------------------------------------------------------- _subtree_height


@pytest.mark.parametrize("weight", ["weight", None])
def test_subtree_height_matches_its_definition(n, weight):
    """Height(v) = max over leafs below v of depth(leaf) - depth(v).

    Computed here straight off `dist_to_root`, which is the definition rather
    than a second implementation of the sweep.
    """
    from navis.morpho.manipulation import _subtree_height

    got = _subtree_height(n, weight=weight)

    depth = G.dist_to_root(n, weight=weight)
    parents = dict(
        zip(n.nodes.node_id.values.tolist(), n.nodes.parent_id.values.tolist())
    )
    leafs = n.nodes[n.nodes.type == "end"].node_id.values.tolist()

    expected = {int(k): 0.0 for k in n.nodes.node_id.values}
    for leaf in leafs:
        dl, v = depth[leaf], leaf
        while v != -1:
            expected[v] = max(expected[v], dl - depth[v])
            v = parents.get(v, -1)

    ids = n.nodes.node_id.values
    assert np.allclose(
        got.loc[ids].values, [expected[int(i)] for i in ids], rtol=1e-4, atol=1e-4
    )


def test_subtree_height_definition():
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

    x = navis.Skeleton(nodes)
    assert list(_subtree_height(x, weight=None).loc[[0, 1, 2, 3, 4]]) == expected


def test_subtree_height_fragmented():
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

    got = _subtree_height(navis.Skeleton(nodes), weight="weight")

    assert np.allclose(got.loc[[0, 1, 2, 3]].values, [1.0, 0.0, 1.0, 0.0], rtol=1e-4)


# -------------------------------------------------------------- longest_neurite


@pytest.mark.parametrize("k", [1, 2, 3])
def test_longest_neurite_is_a_growing_subset(n, k):
    """`n=k` must be a connected subset of the neuron, and larger `k` may only add.

    NOTE: `from_root=True` is the default and skips the geodesic branch entirely,
    so it *must* be False here or this tests nothing.
    """
    got = navis.longest_neurite(n, n=k, from_root=False, inplace=False)

    assert set(got.nodes.node_id) <= set(n.nodes.node_id)
    assert got.n_components == 1
    assert got.cable_length <= n.cable_length

    if k > 1:
        smaller = navis.longest_neurite(n, n=k - 1, from_root=False, inplace=False)
        assert set(smaller.nodes.node_id) <= set(got.nodes.node_id)
        assert got.cable_length >= smaller.cable_length


def test_longest_neurite_k1_is_the_diameter(n):
    """With `from_root=False`, `n=1` is the longest leaf-to-leaf path - i.e. its
    cable must equal the largest geodesic distance between any two leafs.
    """
    got = navis.longest_neurite(n, n=1, from_root=False, inplace=False)

    leafs = n.nodes[n.nodes.type.isin(("end", "root"))].node_id.values
    dmat = G.geodesic_matrix(n, from_=leafs, to_=leafs, directed=False).values.copy()
    dmat[~np.isfinite(dmat)] = -1

    assert got.cable_length == pytest.approx(dmat.max(), rel=1e-4)


def test_collapse_nodes_preserves_structure(n):
    """Collapsing a group must merge exactly that group and leave a valid tree."""
    which = n.nodes.node_id.values[[100, 101, 102, 103]]

    got = navis.collapse_nodes(n, which, inplace=False)

    # Exactly the non-representative members are gone
    assert got.n_nodes == n.n_nodes - (len(which) - 1)
    assert set(got.nodes.node_id) == set(n.nodes.node_id) - set(which[1:])
    assert got.is_acyclic
    # Collapsing does not re-root the neuron
    assert list(got.root) == list(n.root)


def test_collapse_nodes_with_large_node_ids():
    """Node IDs are IDs, not row numbers.

    Segmentation backends hand out node IDs in the 7e17 range. This used to build an
    igraph contraction mapping of vertex *indices* and write node IDs into it, which
    only worked while IDs happened to run 1..N - on real IDs igraph tried to allocate
    a vector sized by the ID and raised `MemoryError`.
    """
    offset = 720575940379000000
    x = navis.example_neurons(1, kind="skeleton").copy()
    pid = x.nodes.parent_id.values.astype(np.int64)
    x.nodes["node_id"] = x.nodes.node_id.values.astype(np.int64) + offset
    x.nodes["parent_id"] = np.where(pid >= 0, pid + offset, -1)
    x._clear_temp_attr()

    which = x.nodes.node_id.values[[500, 501, 502]]
    got = navis.collapse_nodes(x, which, inplace=False)

    assert got.n_nodes == x.n_nodes - 2
    assert got.is_acyclic
    assert set(got.nodes.node_id) == set(x.nodes.node_id) - set(which[1:])
