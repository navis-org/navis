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
