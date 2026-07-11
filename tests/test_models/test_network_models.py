import pytest
import networkx as nx
import numpy as np
import pandas as pd

from navis.models.network_models import (TraversalModel, BayesianTraversalModel)

def test_traversal_models():
    models = (TraversalModel, BayesianTraversalModel)

    G = nx.path_graph(10, create_using=nx.DiGraph)
    G.add_edge(0, 9)
    G.add_node(10)
    edges = nx.to_pandas_edgelist(G)
    edges['weight'] = np.ones(edges.shape[0])

    results = {}
    for m in models:
        model = m(edges, seeds=[1], max_steps=8)
        model.run(iterations=1)
        res = model.summary
        if res.index.name != 'node':  # issue here with older pandas versions
            res.set_index('node', inplace=True)
        assert 0 not in res.index
        assert 9 not in res.index
        assert 10 not in res.index
        for i in range(1, 9):
            row = res.loc[i]
            assert row.layer_min == row.layer_max
        results[m] = res

    for m in models:
        pd.testing.assert_frame_equal(results[TraversalModel], results[m])


def test_bayesian_matches_montecarlo_diamond():
    """Regression test for #194.

    On a probabilistic diamond (a single point of reconvergence) the
    deterministic BayesianTraversalModel must match the Monte-Carlo
    TraversalModel. Previously an independence-across-time assumption made the
    sink node appear traversed too early (layer_mean ~3.878 instead of ~3.963).
    """
    # Diamond: 0->1, 0->2, 1->3, 2->3, seed 0.
    # linear_activation_p maps weight 0.15 -> traversal probability 0.5.
    edges = pd.DataFrame({
        'source': [0, 0, 1, 2],
        'target': [1, 2, 3, 3],
        'weight': [0.15, 0.15, 0.15, 0.15],
    })

    np.random.seed(0)
    tm = TraversalModel(edges, seeds=[0], max_steps=15)
    tm.run(iterations=100000)
    ts = tm.summary
    if ts.index.name != 'node':
        ts.set_index('node', inplace=True)

    bm = BayesianTraversalModel(edges, seeds=[0], max_steps=15)
    bm.run()
    bs = bm.summary
    if bs.index.name != 'node':
        bs.set_index('node', inplace=True)

    # Sink node must match Monte-Carlo (~3.96), not the old biased ~3.878.
    assert bs.loc[3, 'layer_mean'] == pytest.approx(ts.loc[3, 'layer_mean'], abs=0.05)
    assert bs.loc[3, 'layer_mean'] > 3.9

    # Every node's mean traversal step should track Monte-Carlo closely.
    for node in ts.index.intersection(bs.index):
        assert bs.loc[node, 'layer_mean'] == pytest.approx(
            ts.loc[node, 'layer_mean'], abs=0.05)
