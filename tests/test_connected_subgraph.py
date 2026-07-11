"""Tests for the optimized `navis.graph.graph_utils.connected_subgraph`.

The optimized implementation is checked against a frozen copy of the original
implementation (`_connected_subgraph_old`) to prove identical results, except in
the documented edge case (subset contains an ancestor of the branch-point LCA)
where the new implementation returns a connected *superset* with the same root.
"""

import navis
import networkx as nx
import numpy as np

import pytest

from navis.graph import graph_utils as gu


def _connected_subgraph_old(g, ss):
    """Frozen copy of the original (pre-optimization) implementation.

    Operates directly on an nx.DiGraph so we can drive it with arbitrary
    subgraphs in the tests.
    """
    ss = set(ss)
    missing = ss - set(g.nodes)
    if np.any(missing):
        missing = np.array(list(missing)).astype(str)
        raise ValueError(f"Nodes not found: {','.join(missing)}")

    g_ss = g.subgraph(ss)
    in_degree = dict(g_ss.in_degree)
    leafs = ss & {n for n, d in in_degree.items() if not d}

    include = set()
    new_roots = []
    for cc in nx.connected_components(g.to_undirected()):
        paths = []
        for n in leafs & cc:
            this_path = []
            while n is not None:
                this_path.append(n)
                n = next(g.successors(n), None)
            paths.append(this_path)

        if not paths:
            continue

        common = set.intersection(*[set(p) for p in paths])

        longest_path = sorted(paths, key=lambda x: len(x))[-1]
        first_common = sorted(common, key=lambda x: longest_path.index(x))[0]

        for p in paths:
            it = iter(p)
            n = next(it, None)
            while n is not None:
                if n in include:
                    break
                if n == first_common:
                    include.add(n)
                    break
                include.add(n)
                n = next(it, None)

        this_ss = ss & cc
        if this_ss - include:
            nr = sorted(this_ss - include, key=lambda x: longest_path.index(x))[-1]
            new_roots.append(nr)
            include = set.union(include, this_ss)
        else:
            new_roots.append(first_common)

    return np.array(list(include)), new_roots


@pytest.fixture
def neuron():
    return navis.example_neurons(1, kind="skeleton")


def test_connected_subgraph_matches_old(neuron):
    """New implementation must match the old one on random subsets."""
    n = neuron
    g = n.graph
    rng = np.random.default_rng(0)
    ids = n.nodes.node_id.values

    for _ in range(50):
        k = int(rng.integers(2, len(ids)))
        ss = rng.choice(ids, size=k, replace=False)

        new_inc, new_roots = gu.connected_subgraph(g, ss)
        old_inc, old_roots = _connected_subgraph_old(g, ss)

        # New result is identical to (or a connected superset of) the old result.
        assert set(new_inc) >= set(old_inc)
        # The chosen roots must be identical.
        assert set(new_roots) == set(old_roots)

        # Connectivity invariant: the included nodes induce exactly one connected
        # component per returned root.
        sub = g.subgraph(new_inc)
        assert nx.number_weakly_connected_components(sub) == len(new_roots)


def test_connected_subgraph_treeneuron_input(neuron):
    """Passing the TreeNeuron directly must match passing its graph."""
    n = neuron
    rng = np.random.default_rng(1)
    ids = n.nodes.node_id.values
    ss = rng.choice(ids, size=len(ids) // 2, replace=False)

    inc_neuron, roots_neuron = gu.connected_subgraph(n, ss)
    inc_graph, roots_graph = gu.connected_subgraph(n.graph, ss)

    assert set(inc_neuron) == set(inc_graph)
    assert set(roots_neuron) == set(roots_graph)


def test_connected_subgraph_doctest(neuron):
    """Asking for all terminals + root must return the whole neuron."""
    n = neuron
    ends = n.nodes[n.nodes.type.isin(["end", "root"])].node_id.values
    sg, root = gu.connected_subgraph(n, ends)
    assert sg.shape[0] == n.nodes.shape[0]


def test_connected_subgraph_multi_component(neuron):
    """Subgraph spanning several components -> one root per component."""
    n = neuron
    g = n.graph

    # Build a forest by subsetting to nodes from two separate branches. We pick two
    # leaves and take the union of their segments (geodesic paths to root) but cut
    # the trunk so they end up disconnected.
    leaves = n.nodes[n.nodes.type == "end"].node_id.values
    root = n.root[0]

    # Two leaves; paths to root.
    p1 = nx.shortest_path(g, source=int(leaves[0]), target=int(root))
    p2 = nx.shortest_path(g, source=int(leaves[-1]), target=int(root))

    # Drop the shared trunk so the two paths form separate components.
    shared = set(p1) & set(p2)
    comp_nodes = (set(p1) | set(p2)) - shared
    sub = g.subgraph(comp_nodes)

    n_comp = nx.number_weakly_connected_components(sub)
    assert n_comp >= 2  # sanity

    ss = list(comp_nodes)
    inc, roots = gu.connected_subgraph(sub, ss)

    assert len(roots) == n_comp
    assert nx.number_weakly_connected_components(sub.subgraph(inc)) == len(roots)
    # Compare against old implementation on the same multi-component subgraph.
    old_inc, old_roots = _connected_subgraph_old(sub, ss)
    assert set(inc) >= set(old_inc)
    assert set(roots) == set(old_roots)


def test_connected_subgraph_proximal_ancestor_fix(neuron):
    """Edge case: subset = two sibling tips + an ancestor of their branch point.

    The new implementation returns a *connected* subtree (the old one left a gap),
    with the new root == that ancestor.
    """
    n = neuron
    g = n.graph

    # Find a branch point with at least two end-bearing children, then pick two
    # tips below it and an ancestor of the branch point.
    bp = None
    tips = None
    for node in n.nodes[n.nodes.type == "branch"].node_id.values:
        # Children (predecessors, since edges point child->parent).
        children = list(g.predecessors(int(node)))
        if len(children) < 2:
            continue
        # Collect a terminal tip reachable below each of two children.
        found = []
        for c in children:
            # Walk down (predecessors) until a leaf.
            cur = c
            while True:
                preds = list(g.predecessors(cur))
                if not preds:
                    break
                cur = preds[0]
            found.append(cur)
            if len(found) == 2:
                break
        if len(found) == 2:
            bp = int(node)
            tips = found
            break

    assert bp is not None, "Could not find a suitable branch point"

    # An ancestor of the branch point (its parent's parent if available).
    anc = next(g.successors(bp), None)
    assert anc is not None
    anc2 = next(g.successors(anc), None)
    ancestor = int(anc2 if anc2 is not None else anc)

    ss = [int(tips[0]), int(tips[1]), ancestor]
    inc, roots = gu.connected_subgraph(g, ss)

    # Result is connected and rooted at the ancestor.
    assert roots == [ancestor]
    sub = g.subgraph(inc)
    assert nx.number_weakly_connected_components(sub) == 1
    # All requested nodes are present.
    assert set(ss) <= set(inc)
    # And the chain from the branch point up to the ancestor is fully present
    # (this is what the old implementation failed to include).
    chain = nx.shortest_path(g, source=bp, target=ancestor)
    assert set(chain) <= set(inc)
