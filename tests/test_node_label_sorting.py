"""Tests for `node_label_sorting`.

The sort used to be driven by a directed breaks-by-breaks geodesic matrix (4.5GB on a
71k node skeleton). It now uses two O(N) quantities instead - the subtree height and a
difference of root distances - which are mathematically the same thing. The reference
implementation below is the *old* matrix formulation; it exists to pin the new one.

The keys feed a `sorted()`, so a float epsilon in either term could silently reorder a
near-tie. That is what these tests guard against.
"""

import numpy as np
import pandas as pd
import pytest

import navis
from navis import graph
from navis.graph import graph_utils as gu


@pytest.fixture(params=[0, 1, 2, 3])
def neuron(request):
    n = navis.example_neurons(kind="skeleton")[request.param].copy()
    navis.graph.classify_nodes(n)
    if len(n.root) > 1:
        pytest.skip("node_label_sorting does not support multi-root neurons")
    return n


def _reference_sorting(x, weighted=False):
    """The old, matrix-based implementation - kept here purely as a reference."""
    term = x.nodes[x.nodes.type == "end"].node_id.values
    breaks = x.nodes[x.nodes.type.isin(("end", "root", "branch"))].node_id.values

    geo = gu.geodesic_matrix(
        x, from_=breaks, to_=breaks, directed=True, weight="weight" if weighted else None
    )
    dist_mat = pd.DataFrame(
        np.where(geo == float("inf"), np.nan, geo), columns=geo.columns, index=geo.index
    )

    G = graph.simplify_graph(x.graph)
    curr_points = sorted(
        list(G.predecessors(x.root[0])),
        key=lambda n: dist_mat[n].max() + dist_mat.loc[n, x.root[0]],
        reverse=True,
    )

    nodes_walked = []
    while curr_points:
        nodes_walked.append(curr_points.pop(0))
        if nodes_walked[-1] not in term:
            curr_points = (
                sorted(
                    list(G.predecessors(nodes_walked[-1])),
                    key=lambda n: dist_mat[n].max() + dist_mat.loc[n, nodes_walked[-1]],
                    reverse=True,
                )
                + curr_points
            )

    node_list = [x.root[0:]]
    seg_dict = {s[0]: s[::-1] for s in gu._break_segments(x)}
    for n in nodes_walked:
        node_list.append(seg_dict[n][1:])
    return [n for s in node_list for n in s]


@pytest.mark.parametrize("weighted", [False, True])
def test_node_label_sorting_matches_matrix(neuron, weighted):
    """The O(N) key must give the exact same order as the geodesic matrix did."""
    got = list(graph.node_label_sorting(neuron, weighted=weighted))
    expected = _reference_sorting(neuron, weighted=weighted)

    assert got == expected


def test_node_label_sorting_is_complete(neuron):
    """Every node exactly once, starting at the root."""
    srt = list(graph.node_label_sorting(neuron))

    assert srt[0] == neuron.root[0]
    assert len(srt) == neuron.n_nodes
    assert sorted(srt) == sorted(neuron.nodes.node_id.tolist())


def test_node_label_sorting_multi_root():
    n = navis.example_neurons(1, kind="skeleton").copy()
    n.nodes.loc[n.nodes.index[10], "parent_id"] = -1  # make a second root
    n._clear_temp_attr()
    with pytest.raises(ValueError):
        graph.node_label_sorting(n)
