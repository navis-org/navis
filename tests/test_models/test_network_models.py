import pytest
import networkx as nx
import numpy as np
import pandas as pd

from navis.models.network_models import (TraversalModel, BayesianTraversalModel,
                                         ConditionedBayesianTraversalModel)

def test_traversal_models():
    models = (TraversalModel, BayesianTraversalModel,
              ConditionedBayesianTraversalModel)

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


def test_bayesian_batched_path_dag():
    """On a path graph the batched model hits each node at exactly one step.

    cmf entries before that step must be 0, entries at or after it must be 1
    (weight 1 => certain traversal).
    """
    G = nx.path_graph(6, create_using=nx.DiGraph)
    edges = nx.to_pandas_edgelist(G)
    edges['weight'] = 1.0

    model = BayesianTraversalModel(edges, seeds=[0], max_steps=6)
    res = model.run()
    cmf_by_node = dict(zip(res['node'], res['cmf']))

    # Node i is reached at step i (seed at step 0).
    for i in range(6):
        cmf = np.asarray(cmf_by_node[i])
        assert np.all(cmf[:i] == 0.0), (i, cmf)
        assert np.all(cmf[i:] == 1.0), (i, cmf)


def test_bayesian_batched_summary_columns():
    """The summary DataFrame must expose the documented layer columns."""
    G = nx.path_graph(5, create_using=nx.DiGraph)
    edges = nx.to_pandas_edgelist(G)
    edges['weight'] = 1.0
    model = BayesianTraversalModel(edges, seeds=[0], max_steps=5)
    model.run()
    s = model.summary
    for col in ('layer_min', 'layer_max', 'layer_mean', 'layer_median'):
        assert col in s.columns


# --- Bayesian vs. Monte-Carlo ----------------------------------------------
#
# The two Bayesian models make two different independence assumptions. These
# two minimal graphs isolate one each, so it is unambiguous which assumption a
# deviation from Monte-Carlo comes from. Weight .15 maps to a per-step
# traversal probability of 0.5 under linear_activation_p (max_w = .3).

# Independence *across time* within a single edge.
CHAIN_EDGES = pd.DataFrame({'source': [0, 1], 'target': [1, 2], 'weight': .15})

# Independence *across parents*: 2 and 3 share node 1's random activation time.
CORRELATED_EDGES = pd.DataFrame({'source': [0, 1, 1, 2, 3],
                                 'target': [1, 2, 3, 4, 4], 'weight': .15})


def _montecarlo_layer_means(edges, iterations=50_000, max_steps=30):
    np.random.seed(0)
    tm = TraversalModel(edges, seeds=[0], max_steps=max_steps)
    tm.run(iterations=iterations)
    s = tm.summary
    if s.index.name != 'node':
        s = s.set_index('node')
    return s['layer_mean']


def _bayesian_layer_means(model_cls, edges, max_steps=30):
    m = model_cls(edges, seeds=[0], max_steps=max_steps)
    m.run()
    s = m.summary
    if s.index.name != 'node':
        s = s.set_index('node')
    return s['layer_mean']


# TraversalModel is slow (a Python loop per iteration), so compute each graph's
# Monte-Carlo reference once and share it across both parametrizations.
@pytest.fixture(scope='module')
def chain_montecarlo():
    return _montecarlo_layer_means(CHAIN_EDGES)


@pytest.fixture(scope='module')
def correlated_montecarlo():
    return _montecarlo_layer_means(CORRELATED_EDGES)


XFAIL_ACROSS_TIME = pytest.mark.xfail(
    strict=True,
    reason='BayesianTraversalModel assumes independence across time within an '
           'edge; biased early whenever a parent activates at a random time (#194)')
XFAIL_ACROSS_PARENTS = pytest.mark.xfail(
    strict=True,
    reason="assumes parents activate independently; parents 2 and 3 share node "
           "1's random activation time, so the reconvergence node is mistimed")


@pytest.mark.parametrize('model_cls', [
    pytest.param(ConditionedBayesianTraversalModel, id='conditioned'),
    pytest.param(BayesianTraversalModel, id='fast', marks=XFAIL_ACROSS_TIME),
])
def test_bayesian_chain_matches_montecarlo(model_cls, chain_montecarlo):
    """Independence across time - regression for #194 in minimal form.

    Truth (Monte-Carlo) is ~5.0 for node 2; the conditioned model gives exactly
    5.0, the fast one ~4.77.
    """
    bm = _bayesian_layer_means(model_cls, CHAIN_EDGES)
    for node in chain_montecarlo.index.intersection(bm.index):
        assert bm.loc[node] == pytest.approx(chain_montecarlo.loc[node], abs=0.05)


@pytest.mark.parametrize('model_cls', [
    pytest.param(ConditionedBayesianTraversalModel, id='conditioned',
                 marks=XFAIL_ACROSS_PARENTS),
    pytest.param(BayesianTraversalModel, id='fast', marks=XFAIL_ACROSS_TIME),
])
def test_bayesian_correlated_reconvergence_vs_montecarlo(model_cls,
                                                         correlated_montecarlo):
    """Independence across parents - the documented residual approximation.

    Truth (Monte-Carlo) is ~5.96 for node 4; the conditioned model gives ~5.69,
    the fast one ~5.42. Only node 4 is asserted: nodes 1-3 are chain-like and
    already covered by the chain test above.
    """
    bm = _bayesian_layer_means(model_cls, CORRELATED_EDGES)
    assert bm.loc[4] == pytest.approx(correlated_montecarlo.loc[4], abs=0.05)
