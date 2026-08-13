"""Tests for the fastcore-backed graph primitives.

These used to be *differential* tests: run each primitive with and without
navis-fastcore and assert the two agree. That oracle is gone with the fallbacks
(fastcore is a hard requirement now), and it was never the right one anyway - it
could only ever catch a *disagreement*, which is neither necessary nor
sufficient for a bug. See `FASTCORE_DISCREPANCIES.md`, where the one genuinely
dangerous bug it turned up was invisible to it because both backends were wrong
in the same way.

What replaces it is what should always have been here: properties that must hold
of the answer itself, checked against a ground truth built independently of the
implementation.
"""

import numpy as np
import pytest

import navis
from navis.graph.graph_utils import _component_ids, skeleton_edges


@pytest.fixture(scope="module")
def neuron():
    return navis.example_neurons(1, kind="skeleton")


@pytest.fixture(scope="module")
def fragmented(neuron):
    """A neuron broken into three fragments, by orphaning two nodes."""
    n = neuron.copy()
    ids = n.nodes.node_id.values
    n.nodes.loc[n.nodes.node_id.isin([ids[500], ids[1500]]), "parent_id"] = -1
    n._clear_temp_attr()
    return n


def _components_by_walking(x, keep):
    """Ground truth: union-find over the induced edges, in plain Python.

    Deliberately shares no machinery with the implementation - no numpy, no
    fastcore, no igraph - so it can only agree with `connected_components(mask=)`
    by both being right. Comparing the whole partition against this also settles
    connectivity and maximality of each component, so neither needs its own test.
    """
    keep = set(int(i) for i in keep)
    parent = {n: n for n in keep}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for nid, pid in zip(
        x.nodes.node_id.values.tolist(), x.nodes.parent_id.values.tolist()
    ):
        if nid in keep and pid in keep:
            ra, rb = find(nid), find(pid)
            if ra != rb:
                parent[ra] = rb

    groups = {}
    for n in keep:
        groups.setdefault(find(n), set()).add(n)
    return sorted(sorted(g) for g in groups.values())


def _partition(components):
    """Normalise a list of node-ID sets for comparison."""
    return sorted(sorted(int(i) for i in c) for c in components)


# --------------------------------------------- connected_components(mask=...)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_masked_components_matches_ground_truth(neuron, step):
    keep = neuron.nodes.node_id.values[::step]
    assert _partition(_component_ids(neuron, mask=keep)) == _components_by_walking(
        neuron, keep
    )


def test_masked_components_ground_truth_when_fragmented(fragmented):
    keep = fragmented.nodes.node_id.values[::3]
    assert _partition(
        _component_ids(fragmented, mask=keep)
    ) == _components_by_walking(fragmented, keep)


def test_masked_components_accepts_a_set(neuron):
    """Regression: `keep` is routinely a set.

    `np.asarray(a_set)` produces a 0-d *object* array which `np.isin` matches
    against nothing, so this silently returned no components at all - which
    surfaced as `max() arg is an empty sequence` in `find_soma_label`, not as
    anything pointing here.
    """
    keep = set(neuron.nodes.node_id.values[::2].tolist())
    comps = _component_ids(neuron, mask=keep)

    assert len(comps) > 0
    assert _partition(comps) == _components_by_walking(neuron, keep)


def test_masked_components_covers_exactly_keep(neuron):
    """Every kept node lands in exactly one component; nothing else does."""
    keep = set(neuron.nodes.node_id.values[::2].tolist())
    comps = _component_ids(neuron, mask=keep)

    flat = [n for c in comps for n in c]
    assert len(flat) == len(set(flat)), "a node appeared in two components"
    assert set(flat) == keep


def test_masked_components_empty_keep(neuron):
    assert _component_ids(neuron, mask=[]) == []


# ---------------------------------------------------------------- edge helper


def test_skeleton_edges_shape_and_mapping(neuron):
    edges, node_ids = skeleton_edges(neuron)

    assert len(node_ids) == neuron.n_nodes
    # One edge per non-root node
    assert len(edges) == neuron.n_nodes - len(neuron.root)
    assert edges.min() >= 0 and edges.max() < neuron.n_nodes

    # Spot-check: the edge for a known node points at its parent
    nid = neuron.nodes.node_id.values[100]
    pid = neuron.nodes.parent_id.values[100]
    ix = np.flatnonzero(node_ids == nid)[0]
    row = edges[edges[:, 0] == ix][0]
    assert node_ids[row[1]] == pid


def test_skeleton_edges_reproduce_the_node_table(neuron):
    """The edge list is just the node table in index space - going back must
    return exactly the child -> parent pairs we started with.
    """
    edges, node_ids = skeleton_edges(neuron)

    recovered = {(int(node_ids[a]), int(node_ids[b])) for a, b in edges}
    expected = {
        (int(n), int(p))
        for n, p in zip(neuron.nodes.node_id.values, neuron.nodes.parent_id.values)
        if p >= 0
    }
    assert recovered == expected


# -------------------------------------------------------------- end-to-end


def test_heal_skeleton_reconnects_every_fragment(fragmented):
    """Healing must end with one tree and may only *add* cable."""
    assert fragmented.n_components > 1

    healed = navis.heal_skeleton(fragmented, inplace=False)

    assert healed.n_components == 1
    assert healed.n_nodes == fragmented.n_nodes
    assert healed.cable_length >= fragmented.cable_length


# N.B. the label-only soma path is the other consumer of
# `_component_ids` and is covered end-to-end by `tests/test_soma.py` -
# which is what caught the set-input bug above.
