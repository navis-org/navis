"""Tests for `navis.heal_skeleton` and the machinery behind it."""

import tracemalloc

import navis
import numpy as np
import pandas as pd
import pytest

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree

from navis.morpho.manipulation import _segment_radii, _stitch_edges

HAS_FASTCORE = navis.utils.fastcore is not None and hasattr(
    navis.utils.fastcore, "heal_skeleton"
)


@pytest.fixture(params=["builtin", "fastcore"], autouse=True)
def stitch_backend(request, monkeypatch):
    """Run every test in this module against both stitching backends.

    `heal_skeleton` uses fastcore when it is installed, so without this the
    built-in numpy implementation would silently stop being tested.
    """
    if request.param == "builtin":
        monkeypatch.setattr(navis.utils, "fastcore", None)
    elif not HAS_FASTCORE:
        pytest.skip("navis-fastcore with `heal_skeleton` not installed")
    return request.param


def fragment(neuron, n_breaks, seed=0):
    """Break `n_breaks` random edges to produce a fragmented skeleton.

    Note this leaves the fragments *touching*: no cable is removed, so the gap
    between them is zero. See `fragment_with_gap` for the more realistic case.
    """
    x = neuron.copy()
    rng = np.random.default_rng(seed)
    has_parent = np.where(x.nodes.parent_id.values >= 0)[0]
    breaks = rng.choice(has_parent, size=min(n_breaks, len(has_parent)), replace=False)
    parents = x.nodes.parent_id.values.copy()
    parents[breaks] = -1
    x.nodes["parent_id"] = parents
    x._clear_temp_attr()
    return x


def fragment_with_gap(neuron, n_breaks, gap, seed=0):
    """Excise `n_breaks` runs of cable of length ~`gap` from `neuron`.

    Unlike `fragment` this leaves a real gap between the fragments, which is what
    a skeleton from an imperfect segmentation actually looks like - and which is
    far harder to stitch, because a node deep inside a fragment no longer has a
    foreign node anywhere near it.
    """
    x = neuron.copy()
    rng = np.random.default_rng(seed)

    nodes = x.nodes
    ix = pd.Series(np.arange(len(nodes)), index=nodes.node_id.values)
    par_ix = ix.reindex(nodes.parent_id.values).values
    has_parent = ~np.isnan(par_ix)
    par_ix = np.where(has_parent, np.nan_to_num(par_ix), 0).astype(int)

    pos = nodes[["x", "y", "z"]].values.astype(np.float64)
    dist = np.zeros(len(nodes))
    dist[has_parent] = np.linalg.norm(
        pos[has_parent] - pos[par_ix[has_parent]], axis=1
    )

    candidates = np.where(has_parent)[0]
    starts = rng.choice(
        candidates, size=min(n_breaks, len(candidates)), replace=False
    )

    # Walk from each cut point towards the root, dropping nodes until we have
    # removed `gap` worth of cable.
    remove = np.zeros(len(nodes), dtype=bool)
    for start in starts:
        walked, at = 0.0, int(start)
        while walked < gap and has_parent[at] and not remove[at]:
            remove[at] = True
            walked += dist[at]
            at = int(par_ix[at])

    x = navis.subset_neuron(x, nodes.node_id.values[~remove])
    x._clear_temp_attr()
    return x


def added_cable(before, after):
    """Total length of the edges that healing added.

    Computed in float64 from the coordinates rather than via `.cable_length`,
    which accumulates in float32 and is too coarse to verify MST exactness.
    """
    def edges(x):
        nodes = x.nodes[x.nodes.parent_id >= 0]
        pos = x.nodes.set_index("node_id")[["x", "y", "z"]].astype(np.float64)
        child = pos.loc[nodes.node_id.values].values
        parent = pos.loc[nodes.parent_id.values].values
        return np.linalg.norm(child - parent, axis=1).sum()

    return edges(after) - edges(before)


def brute_force_mst_cable(x, cols=("x", "y", "z")):
    """Total added cable of a true MST over the fragments.

    Builds the complete inter-fragment distance matrix by brute force (closest
    pair of nodes between every pair of fragments) and runs a global MST over
    it. This is the ground truth `_stitch_edges` must reproduce.
    """
    labels = navis.graph.graph_utils._connected_components(x)
    coords = x.nodes[list(cols)].values.astype(float)
    ids = x.nodes.node_id.values
    pos = {n: i for i, n in enumerate(ids)}
    frags = [np.array([pos[n] for n in c]) for c in labels]

    n_frags = len(frags)
    dists = np.zeros((n_frags, n_frags))
    for i in range(n_frags):
        for j in range(i + 1, n_frags):
            d = np.linalg.norm(
                coords[frags[i]][:, None, :] - coords[frags[j]][None, :, :], axis=-1
            )
            dists[i, j] = dists[j, i] = d.min()

    return minimum_spanning_tree(dists).toarray().sum()


def assert_valid_forest(x):
    """The parent array must describe a forest: no cycles, one root per component.

    Every non-root node contributes exactly one parent edge, so a component
    without a root necessarily contains a cycle. Checking that the number of
    connected components equals the number of roots therefore rules out cycles -
    and does so in linear time (walking parents from every node would be
    quadratic on a long chain).
    """
    node_ids = x.nodes.node_id.values
    parent_ids = x.nodes.parent_id.values
    n_nodes = len(node_ids)

    pos = pd.Series(np.arange(n_nodes), index=node_ids)
    has_parent = parent_ids >= 0
    children = np.arange(n_nodes)[has_parent]
    parents = pos.loc[parent_ids[has_parent]].values

    adj = coo_matrix(
        (np.ones(len(children)), (children, parents)), shape=(n_nodes, n_nodes)
    )
    n_comps = connected_components(adj, directed=False, return_labels=False)
    n_roots = int((~has_parent).sum())

    assert n_comps == n_roots, (
        f"{n_roots} root(s) but {n_comps} component(s) - the parent array is not a "
        "valid forest (a component without a root contains a cycle)"
    )


@pytest.mark.parametrize("n_breaks", [1, 10, 100])
def test_heal_reconnects_everything(n_breaks):
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment(n, n_breaks)

    assert len(frag.root) == n_breaks + 1

    healed = navis.heal_skeleton(frag)

    assert len(healed.root) == 1
    assert healed.n_nodes == n.n_nodes
    assert_valid_forest(healed)


@pytest.mark.parametrize("n_breaks", [1, 10, 100])
def test_heal_is_a_true_mst(n_breaks):
    """The added cable must match a brute-force minimum spanning tree."""
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment(n, n_breaks)

    healed = navis.heal_skeleton(frag)
    expected = brute_force_mst_cable(frag)

    assert added_cable(frag, healed) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("gap", [50, 200, 2000])
def test_heal_is_a_true_mst_with_real_gaps(gap):
    """Same, but with cable actually removed so the fragments don't touch.

    This is the case that matters: with `gap=0` a stitcher can lean on the
    fragments being adjacent, so it is possible to pass every zero-gap test while
    being quadratic on any real skeleton.
    """
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment_with_gap(n, n_breaks=40, gap=gap)

    assert len(frag.root) > 1

    healed = navis.heal_skeleton(frag)
    expected = brute_force_mst_cable(frag)

    assert len(healed.root) == 1
    assert added_cable(frag, healed) == pytest.approx(expected, rel=1e-6)
    assert_valid_forest(healed)


@pytest.mark.skipif(not HAS_FASTCORE, reason="navis-fastcore not installed")
@pytest.mark.parametrize("gap", [0, 500])
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="ALL"),
        dict(method="LEAFS"),
        dict(method="ALL", max_dist=300),
        dict(method="ALL", min_size=10),
    ],
)
def test_fastcore_and_builtin_agree(gap, kwargs, monkeypatch, stitch_backend):
    """Both backends must add the same amount of cable and leave the same roots.

    They may pick *different* bridges where several are exactly the same length -
    the MST isn't unique then - so we compare the total added cable rather than
    the edges themselves.

    `use_radius` is deliberately not covered here; see
    `test_use_radius_agrees_only_approximately_across_backends`.
    """
    if stitch_backend != "fastcore":
        pytest.skip("runs once, comparing the two backends against each other")

    n = navis.example_neurons(1, kind="skeleton")
    frag = (
        fragment(n, 40) if gap == 0 else fragment_with_gap(n, n_breaks=40, gap=gap)
    )

    with_fc = navis.heal_skeleton(frag, **kwargs)

    monkeypatch.setattr(navis.utils, "fastcore", None)
    without_fc = navis.heal_skeleton(frag, **kwargs)

    assert added_cable(frag, with_fc) == pytest.approx(
        added_cable(frag, without_fc), rel=1e-6
    )
    assert set(np.asarray(with_fc.root).tolist()) == set(
        np.asarray(without_fc.root).tolist()
    )
    assert_valid_forest(with_fc)


@pytest.mark.skipif(not HAS_FASTCORE, reason="navis-fastcore not installed")
def test_use_radius_agrees_only_approximately_across_backends(monkeypatch, stitch_backend):
    """With `use_radius` the two backends may differ slightly - and that's expected.

    `_segment_radii` assigns each node the mean radius of *its* segment, but a
    branch point belongs to several segments, so it ends up with whichever segment
    is written last. The backends enumerate the same segments in a different
    order, so branch points get different radii and the two end up minimising in
    slightly different 4D spaces. Each still finds a true MST - of its own space.

    Guard the size of that divergence so it stays a rounding-level artefact rather
    than growing into a real disagreement.
    """
    if stitch_backend != "fastcore":
        pytest.skip("runs once, comparing the two backends against each other")

    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment_with_gap(n, n_breaks=40, gap=500)

    with_fc = navis.heal_skeleton(frag, method="ALL", use_radius=5)

    monkeypatch.setattr(navis.utils, "fastcore", None)
    without_fc = navis.heal_skeleton(frag, method="ALL", use_radius=5)

    # Both fully heal the neuron and land on the same root ...
    assert len(with_fc.root) == len(without_fc.root) == 1
    assert_valid_forest(with_fc)
    assert_valid_forest(without_fc)

    # ... and the added cable agrees to well under a percent.
    assert added_cable(frag, with_fc) == pytest.approx(
        added_cable(frag, without_fc), rel=0.01
    )


def test_heal_single_component_is_noop():
    n = navis.example_neurons(1, kind="skeleton")
    assert len(n.root) == 1

    healed = navis.heal_skeleton(n)

    assert len(healed.root) == 1
    assert healed.cable_length == pytest.approx(n.cable_length)


def test_heal_leafs_only():
    """`method="LEAFS"` may pick worse attachment points but must still connect."""
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment(n, 20)

    healed = navis.heal_skeleton(frag, method="LEAFS")

    assert len(healed.root) == 1
    assert_valid_forest(healed)
    # Restricting the candidates can never produce *less* cable than using all nodes
    assert added_cable(frag, healed) >= added_cable(frag, navis.heal_skeleton(frag))


def test_heal_max_dist_leaves_neuron_fragmented():
    """Bridges longer than `max_dist` must not be added - result stays a forest."""
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment(n, 50)

    healed = navis.heal_skeleton(frag, max_dist=10)

    # Too small to bridge everything, so the neuron must remain fragmented ...
    assert len(healed.root) > 1
    # ... but must still be a valid forest (this exercises the multi-seed rewire)
    assert_valid_forest(healed)
    assert healed.n_nodes == n.n_nodes

    # A generous max_dist heals completely and matches the unrestricted result
    healed = navis.heal_skeleton(frag, max_dist=np.inf)
    assert len(healed.root) == 1
    assert healed.cable_length == pytest.approx(navis.heal_skeleton(frag).cable_length)


def test_heal_min_size_skips_small_fragments():
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment(n, 50)

    healed = navis.heal_skeleton(frag, min_size=100)

    # Fragments below `min_size` are ignored and stay disconnected
    assert len(healed.root) > 1
    assert_valid_forest(healed)
    assert healed.n_nodes == n.n_nodes


def test_heal_drop_disc():
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment(n, 50)

    healed = navis.heal_skeleton(frag, max_dist=10, drop_disc=True)

    assert len(healed.root) == 1
    assert healed.n_nodes < n.n_nodes


def test_heal_mask():
    """Only nodes in the mask may be used to bridge."""
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment(n, 10)

    mask = frag.nodes.node_id.values
    healed = navis.heal_skeleton(frag, mask=mask)
    assert len(healed.root) == 1

    # Same thing but as a boolean mask
    healed_bool = navis.heal_skeleton(frag, mask=np.ones(frag.n_nodes, dtype=bool))
    assert healed_bool.cable_length == pytest.approx(healed.cable_length)


def test_heal_use_radius():
    """`use_radius` must actually be honoured (it used to be silently ignored)."""
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment(n, 50)

    plain = navis.heal_skeleton(frag)
    weighted = navis.heal_skeleton(frag, use_radius=10)

    assert len(weighted.root) == 1
    assert_valid_forest(weighted)
    # Weighting by radius changes which nodes get bridged, so the resulting
    # cable must differ from the unweighted healing.
    assert weighted.cable_length != pytest.approx(plain.cable_length)


def test_use_radius_fallback_for_isolated_nodes():
    """Isolated nodes must fall back to their *own* radius, scaled by `use_radius`.

    An isolated node (a root without children) belongs to no segment and so has no
    segment radius. The fallback used to pass a `{node_id: radius}` dict straight
    to `Series.fillna()`, which fills by *index label* rather than by value - so
    such a node was handed the radius of whichever node's ID happened to equal its
    DataFrame index, and was never scaled by `use_radius` either.
    """
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment(n, 50)
    use_radius = 5

    radii = _segment_radii(frag, frag.nodes, use_radius)

    # Nodes that belong to no segment - i.e. the ones that hit the fallback
    in_segment = {nd for seg in frag.small_segments for nd in seg}
    isolated = np.where(~frag.nodes.node_id.isin(in_segment).values)[0]
    assert len(isolated), "test needs at least one node without a segment radius"

    own_radius = frag.nodes.radius.values
    for i in isolated:
        assert radii.iloc[i] == pytest.approx(own_radius[i] * use_radius)

    # Every node's value must be scaled by `use_radius` - doubling it doubles them
    doubled = _segment_radii(frag, frag.nodes, use_radius * 2)
    assert np.allclose(doubled.values, radii.values * 2)


def test_heal_inplace():
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment(n, 10)

    copy = navis.heal_skeleton(frag, inplace=False)
    assert len(frag.root) == 11, "`inplace=False` must not modify the input"
    assert len(copy.root) == 1

    assert navis.heal_skeleton(frag, inplace=True) is not None or True
    assert len(frag.root) == 1


def test_stitch_edges_handles_noncontiguous_labels():
    """Component labels are not contiguous once `min_size` has dropped fragments."""
    n = navis.example_neurons(1, kind="skeleton")
    frag = fragment(n, 20)

    cc = navis.graph.graph_utils._connected_components(frag)
    labels = frag.nodes.node_id.map({nd: i for i, c in enumerate(cc) for nd in c})

    # Drop every other fragment so the remaining labels are 0, 2, 4, ...
    keep = labels % 2 == 0
    to_use = frag.nodes[keep.values]
    labels = labels[keep.values]

    edges = _stitch_edges(to_use, labels, ["x", "y", "z"], np.inf)

    assert len(edges) == labels.nunique() - 1
    # Every bridge must connect two *different* fragments
    lookup = dict(zip(to_use.node_id.values, labels.values))
    assert all(lookup[a] != lookup[b] for a, b in edges)


def _chain_fragments(n_frags, per_frag, sep):
    """`n_frags` dense chains of `per_frag` nodes, separated by `sep` along x."""
    frames = []
    for f in range(n_frags):
        ids = np.arange(per_frag) + f * per_frag
        parents = ids - 1
        parents[0] = -1
        frames.append(
            pd.DataFrame({
                "node_id": ids,
                "parent_id": parents,
                "x": np.arange(per_frag) * 0.5 + f * sep,
                "y": (ids % 7) * 0.5,
                "z": (ids % 5) * 0.5,
                "radius": 1.0,
            })
        )
    return navis.TreeNeuron(pd.concat(frames, ignore_index=True))


@pytest.mark.parametrize("sep", [10, 10_000])
def test_heal_large_well_separated_fragments(sep):
    """Large, well-separated fragments must not blow up time or memory.

    Every node's `k` nearest neighbours are its own fragment-mates until `k`
    approaches the fragment size. A naive k-escalating search degrades to O(n^2)
    here (both in time and in the size of the neighbour matrix it materialises),
    so this guards against that regression.
    """
    x = _chain_fragments(2, 10_000, sep)
    assert len(x.root) == 2

    tracemalloc.start()
    healed = navis.heal_skeleton(x)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(healed.root) == 1
    assert_valid_forest(healed)

    # The (nodes x k) neighbour matrix must stay bounded. A quadratic search would
    # need >1GB here; anything in the low hundreds of MB means it stayed linear.
    assert peak < 300e6, f"peak memory was {peak / 1e6:.0f} MB - search went quadratic"

    # And the single bridge must still be the exact closest pair between the two
    # fragments (found here with a KDTree rather than an O(n^2) brute force)
    coords = x.nodes[["x", "y", "z"]].values.astype(np.float64)
    half = len(coords) // 2
    expected = cKDTree(coords[half:]).query(coords[:half], k=1)[0].min()

    assert added_cable(x, healed) == pytest.approx(expected, rel=1e-6)


def test_stitch_skeletons():
    a, b = navis.example_neurons(2, kind="skeleton")

    stitched = navis.stitch_skeletons(a, b, method="LEAFS")

    assert len(stitched.root) == 1
    assert stitched.n_nodes == a.n_nodes + b.n_nodes
    assert_valid_forest(stitched)
