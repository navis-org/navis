import functools
import itertools
from collections import defaultdict

import pytest
import networkx as nx
import numpy as np
import pandas as pd

from navis.models.network_models import (TraversalModel, BayesianTraversalModel)

METHODS = ('conditioned', 'legacy')


def test_traversal_models():
    models = (TraversalModel, BayesianTraversalModel,
              functools.partial(BayesianTraversalModel, method='legacy'))

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


def test_bayesian_invalid_method():
    edges = pd.DataFrame({'source': [0], 'target': [1], 'weight': 1.})
    with pytest.raises(ValueError, match='method'):
        BayesianTraversalModel(edges, seeds=[0], method='fast')


@pytest.mark.parametrize('method', METHODS)
def test_bayesian_batched_path_dag(method):
    """On a path graph the batched model hits each node at exactly one step.

    cmf entries before that step must be 0, entries at or after it must be 1
    (weight 1 => certain traversal).
    """
    G = nx.path_graph(6, create_using=nx.DiGraph)
    edges = nx.to_pandas_edgelist(G)
    edges['weight'] = 1.0

    model = BayesianTraversalModel(edges, seeds=[0], max_steps=6, method=method)
    res = model.run()
    cmf_by_node = dict(zip(res['node'], res['cmf']))

    # Node i is reached at step i (seed at step 0).
    for i in range(6):
        cmf = np.asarray(cmf_by_node[i])
        assert np.all(cmf[:i] == 0.0), (i, cmf)
        assert np.all(cmf[i:] == 1.0), (i, cmf)


@pytest.mark.parametrize('method', METHODS)
def test_bayesian_batched_summary_columns(method):
    """The summary DataFrame must expose the documented layer columns."""
    G = nx.path_graph(5, create_using=nx.DiGraph)
    edges = nx.to_pandas_edgelist(G)
    edges['weight'] = 1.0
    model = BayesianTraversalModel(edges, seeds=[0], max_steps=5, method=method)
    model.run()
    s = model.summary
    for col in ('layer_min', 'layer_max', 'layer_mean', 'layer_median'):
        assert col in s.columns


# --- Bayesian vs. exact ------------------------------------------------------
#
# The two methods make different independence assumptions. These two minimal
# graphs isolate one each, so it is unambiguous which assumption a deviation
# from the truth comes from. Weight .15 maps to a per-step traversal
# probability of 0.5 under linear_activation_p (max_w = .3).

# Independence *across time* within a single edge.
CHAIN_EDGES = pd.DataFrame({'source': [0, 1], 'target': [1, 2], 'weight': .15})

# Independence *across parents*: 2 and 3 share node 1's random activation time.
CORRELATED_EDGES = pd.DataFrame({'source': [0, 1, 1, 2, 3],
                                 'target': [1, 2, 3, 4, 4], 'weight': .15})

MAX_STEPS = 30


def _exact_layer_means(edges, seeds, p, max_steps):
    """Exact mean layer per node under `TraversalModel`'s dynamics.

    Steps the distribution over sets of traversed nodes: seeds are traversed at
    layer 1 and at layer `i` every edge from a traversed to an untraversed node
    fires independently with probability `p`. Like
    `BayesianTraversalModel.make_summary`, probability mass beyond `max_steps`
    is dropped. Exponential in the number of edges, so tiny graphs only.
    """
    src, tgt = edges['source'].values, edges['target'].values
    states = {frozenset(seeds): 1.}
    means = {n: 1. for n in seeds}
    for layer in range(2, max_steps + 1):
        next_states = defaultdict(float)
        for traversed, prob in states.items():
            open_edges = [(s, t) for s, t in zip(src, tgt)
                          if s in traversed and t not in traversed]
            for fired in itertools.product((False, True), repeat=len(open_edges)):
                w = prob * np.prod([p if f else 1 - p for f in fired])
                reached = {t for (_, t), f in zip(open_edges, fired) if f}
                for n in reached:
                    means[n] = means.get(n, 0.) + w * layer
                next_states[traversed | reached] += w
        states = next_states
    return pd.Series(means)


def _bayesian_layer_means(edges, method):
    m = BayesianTraversalModel(edges, seeds=[0], max_steps=MAX_STEPS,
                               method=method)
    m.run()
    s = m.summary
    if s.index.name != 'node':
        s = s.set_index('node')
    return s['layer_mean']


def test_exact_reference_chain():
    """Each edge takes a geometric number of steps (mean 2), so node 2 sits at
    layer 1 + 2 + 2 = 5."""
    exact = _exact_layer_means(CHAIN_EDGES, [0], .5, MAX_STEPS)
    assert exact[2] == pytest.approx(5., abs=1e-4)


@pytest.mark.parametrize('edges, node, method, expected', [
    # Regression test for #194 in minimal form: exact on a chain.
    pytest.param(CHAIN_EDGES, 2, 'conditioned', 'exact', id='chain-conditioned'),
    # Independence across time: truth is 5.0.
    pytest.param(CHAIN_EDGES, 2, 'legacy', 4.7685, id='chain-legacy'),
    # Independence across parents: truth is 5.963.
    pytest.param(CORRELATED_EDGES, 4, 'conditioned', 5.6914,
                 id='correlated-conditioned'),
    # Both assumptions: truth is 5.963.
    pytest.param(CORRELATED_EDGES, 4, 'legacy', 5.4179, id='correlated-legacy'),
])
def test_bayesian_vs_exact(edges, node, method, expected):
    """Exact where the method's assumptions hold; otherwise the known early bias
    is pinned so that it cannot silently grow."""
    exact = _exact_layer_means(edges, [0], .5, MAX_STEPS)[node]
    mean = _bayesian_layer_means(edges, method)[node]
    if expected == 'exact':
        assert mean == pytest.approx(exact, abs=1e-4)
    else:
        assert mean == pytest.approx(expected, abs=1e-3)
        assert mean < exact
