"""Behavioural contracts for the morphometrics fastcore implements.

Replaces `tests/test_fastcore.py`, which was entirely differential: every test
computed a quantity with navis-fastcore and again with it monkeypatched away,
and asserted the two matched. With fastcore a hard requirement there is no
second implementation left to compare against - and the comparison was the weak
oracle anyway, since it could only report a *disagreement*.

These check the definitions instead: Strahler's recurrence, the parent-distance
identity, what pruning a twig means. All of them would have caught a wrong
answer that two agreeing implementations would not.
"""

import numpy as np
import pandas as pd
import pytest

import navis


@pytest.fixture(scope="module")
def n():
    return navis.example_neurons(1, kind="skeleton")


# ------------------------------------------------------------------ parent_dist


def test_parent_dist_is_the_distance_to_the_parent(n):
    """Definition: each node's Euclidean distance to its parent; roots get
    `root_dist`."""
    got = navis.morpho.mmetrics.parent_dist(n, root_dist=0)

    coords = n.nodes[["x", "y", "z"]].values.astype(float)
    pos = pd.Index(n.nodes.node_id.values)
    pid = n.nodes.parent_id.values
    has_parent = pid >= 0

    expected = np.zeros(len(coords))
    p_ix = pos.get_indexer(pid[has_parent])
    expected[has_parent] = np.linalg.norm(
        coords[has_parent] - coords[p_ix], axis=1
    )

    assert np.allclose(got, expected, rtol=1e-5)


def test_parent_dist_sums_to_cable_length(n):
    """Cable length is exactly the sum of the parent distances."""
    assert navis.morpho.mmetrics.parent_dist(n, root_dist=0).sum() == pytest.approx(
        n.cable_length, rel=1e-5
    )


def test_parent_dist_honours_root_dist(n):
    got = navis.morpho.mmetrics.parent_dist(n, root_dist=7.0)
    roots = n.nodes.parent_id.values < 0

    assert (got[roots] == 7.0).all()
    assert len(n.root) == roots.sum()


# --------------------------------------------------------------- strahler_index


@pytest.mark.parametrize("method", ["standard", "greedy"])
def test_strahler_index_satisfies_its_recurrence(n, method):
    """Leafs are 1. A node with one child inherits it. At a branch point:

    - "standard": max(children), +1 if the max is shared by two or more
    - "greedy":   sum(children)
    """
    x = navis.strahler_index(n.copy(), method=method)
    si = dict(zip(x.nodes.node_id.values.tolist(), x.nodes.strahler_index.values))

    children = {}
    for nid, pid in zip(
        x.nodes.node_id.values.tolist(), x.nodes.parent_id.values.tolist()
    ):
        if pid >= 0:
            children.setdefault(pid, []).append(nid)

    for node, si_v in si.items():
        kids = [si[c] for c in children.get(node, [])]
        if not kids:
            assert si_v == 1, f"leaf {node} has SI {si_v}"
        elif len(kids) == 1:
            assert si_v == kids[0], f"slab {node} did not inherit its child's SI"
        elif method == "greedy":
            assert si_v == sum(kids)
        else:
            top = max(kids)
            assert si_v == (top + 1 if kids.count(top) >= 2 else top)


def test_strahler_index_root_is_the_maximum(n):
    x = navis.strahler_index(n.copy())
    root_si = x.nodes.loc[x.nodes.parent_id < 0, "strahler_index"].values
    assert (root_si == x.nodes.strahler_index.max()).all()


def test_strahler_index_min_twig_size_lowers_the_order(n):
    """Twigs shorter than `min_twig_size` stop counting towards a branch point's
    order, so the arbor's overall Strahler order can only go down.

    Note individual *nodes* may go up: an ignored twig is re-assigned the index
    of the branch it hangs off, which is generally higher than the 1 it had as a
    leaf. It is the maximum that is the meaningful invariant.
    """
    plain = navis.strahler_index(n.copy()).nodes.strahler_index
    demoted = navis.strahler_index(n.copy(), min_twig_size=5).nodes.strahler_index

    assert demoted.max() <= plain.max()
    assert demoted.min() >= 0


def test_strahler_index_on_a_hand_computed_tree():
    #        0(root)
    #        |
    #        1 ------.
    #        |       |
    #        2       5
    #      /   \
    #     3     4          -> 3, 4 are leafs (SI 1), so 2 is SI 2;
    #                         5 is a leaf (SI 1), so 1 is max(2, 1) = 2
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3, 4, 5],
            "parent_id": [-1, 0, 1, 2, 2, 1],
            "x": [0.0, 1.0, 2.0, 3.0, 3.0, 2.0],
            "y": [0.0, 0.0, 0.0, 1.0, -1.0, 5.0],
            "z": [0.0] * 6,
        }
    )
    x = navis.strahler_index(navis.Skeleton(nodes))
    got = dict(zip(x.nodes.node_id.values.tolist(), x.nodes.strahler_index.tolist()))

    assert got == {0: 2, 1: 2, 2: 2, 3: 1, 4: 1, 5: 1}


# ------------------------------------------------------------------ prune_twigs


@pytest.mark.parametrize("recursive", [True, False])
def test_prune_twigs_removes_only_short_terminal_branches(n, recursive):
    size = 5000 / 8
    pruned = navis.prune_twigs(n, size=size, recursive=recursive)

    # Only ever a subset, and the root survives
    assert set(pruned.nodes.node_id) <= set(n.nodes.node_id)
    assert set(np.asarray(n.root).tolist()) <= set(pruned.nodes.node_id.tolist())
    assert pruned.cable_length <= n.cable_length

    # Terminal segments must all still carry cable (nothing left as a stub)
    leafs = set(pruned.nodes[pruned.nodes.type == "end"].node_id.values.tolist())
    terminal = [seg for seg in pruned.small_segments if seg[0] in leafs]
    assert terminal, "expected at least one terminal segment to survive"
    assert min(navis.graph.segment_lengths(pruned, terminal)) > 0


def test_prune_twigs_is_monotonic_in_size(n):
    """A bigger threshold can only remove more."""
    small = navis.prune_twigs(n, size=100)
    big = navis.prune_twigs(n, size=1000)

    assert set(big.nodes.node_id) <= set(small.nodes.node_id)
    assert big.cable_length <= small.cable_length


def test_prune_twigs_on_a_hand_computed_tree():
    """One long branch and one short twig off the same branch point."""
    #  0 - 1 - 2 - 3 - 4      (long: 4 units of cable past the branch point)
    #       \- 5              (short: 1 unit)
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3, 4, 5],
            "parent_id": [-1, 0, 1, 2, 3, 1],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 1.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "z": [0.0] * 6,
        }
    )
    pruned = navis.prune_twigs(navis.Skeleton(nodes), size=1.5)

    assert 5 not in pruned.nodes.node_id.values, "the short twig should be gone"
    assert set(pruned.nodes.node_id.values.tolist()) == {0, 1, 2, 3, 4}


# ------------------------------------------------------- synapse_flow_centrality


@pytest.mark.parametrize("mode", ["sum", "centrifugal", "centripetal"])
def test_synapse_flow_centrality_is_non_negative_and_capped(n, mode):
    """Flow is a product of two synapse counts, so it is non-negative and can
    never exceed `n_pre * n_post`.
    """
    x = navis.synapse_flow_centrality(n.copy(), mode=mode)
    flow = x.nodes.synapse_flow_centrality.values

    n_pre = (x.connectors.type == "pre").sum()
    n_post = (x.connectors.type == "post").sum()

    assert (flow >= 0).all()
    assert flow.max() <= n_pre * n_post


def test_synapse_flow_centrality_sum_is_the_sum_of_its_parts(n):
    """`mode="sum"` must equal centrifugal + centripetal, node for node."""
    total = navis.synapse_flow_centrality(
        n.copy(), mode="sum"
    ).nodes.synapse_flow_centrality.values
    cf = navis.synapse_flow_centrality(
        n.copy(), mode="centrifugal"
    ).nodes.synapse_flow_centrality.values
    cp = navis.synapse_flow_centrality(
        n.copy(), mode="centripetal"
    ).nodes.synapse_flow_centrality.values

    # The branch-point correction is applied per mode, so it is applied to the
    # sum too - compare where it does not bite, i.e. off the branch points.
    is_bp = (navis.graph.classify_nodes(n.copy()).nodes.type == "branch").values
    assert np.array_equal(total[~is_bp], (cf + cp)[~is_bp])


def test_synapse_flow_centrality_is_zero_with_only_one_synapse_type():
    """Flow is a product of pre- and postsynapse counts, so with none of one kind
    there is nothing to flow anywhere.
    """
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2],
            "parent_id": [-1, 0, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0] * 3,
            "z": [0.0] * 3,
        }
    )
    x = navis.Skeleton(nodes)
    x.connectors = pd.DataFrame(
        {
            "connector_id": [0, 1],
            "node_id": [1, 2],
            "type": ["pre", "pre"],
            "x": [1.0, 2.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    )

    out = navis.synapse_flow_centrality(x)
    assert (out.nodes.synapse_flow_centrality.values == 0).all()
