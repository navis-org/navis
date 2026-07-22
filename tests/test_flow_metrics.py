"""Tests for the flow metrics on *fragmented* neurons.

These metrics work out how many synapses/leafs are proximal to a node as
`total - distal`. That identity only holds on a single-rooted tree; on a forest, nodes
in another fragment are neither distal nor proximal, and counting them as proximal used
to inflate the flow.

The ground truth used here is simple and backend-independent: a fragment's flow cannot
depend on what else happens to sit in the node table, so computing each fragment on its
own must give the same answer as computing them together as one (disconnected) neuron.
"""

import numpy as np
import pandas as pd
import pytest

import navis
from navis import utils


# (function, name of the column it writes)
METRICS = [
    (navis.flow_centrality, "flow_centrality"),
    (navis.synapse_flow_centrality, "synapse_flow_centrality"),
    (navis.arbor_segregation_index, "segregation_index"),
]
IDS = [f.__name__ for f, _ in METRICS]

# Two disjoint, branched fragments, each with one pre- and one postsynapse.
#   A:  0(root) - 1 - 2,  1 - 3      post@0, pre@2
#   B: 10(root) - 11 - 12, 11 - 13   post@10, pre@12
NODES = pd.DataFrame(
    {
        "node_id": [0, 1, 2, 3, 10, 11, 12, 13],
        "parent_id": [-1, 0, 1, 1, -1, 10, 11, 11],
        "x": [0.0, 1.0, 2.0, 2.0, 100.0, 101.0, 102.0, 102.0],
        "y": [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0],
        "z": [0.0] * 8,
    }
)
CONNECTORS = pd.DataFrame(
    {
        "connector_id": [0, 1, 2, 3],
        "node_id": [0, 2, 10, 12],
        "type": [1, 0, 1, 0],  # 1 = post, 0 = pre
        "x": [0.0, 2.0, 100.0, 102.0],
        "y": [0.0] * 4,
        "z": [0.0] * 4,
    }
)
FRAG_A = [0, 1, 2, 3]
FRAG_B = [10, 11, 12, 13]


def _neuron(ids=None):
    ids = NODES.node_id.values if ids is None else ids
    n = navis.TreeNeuron(NODES[NODES.node_id.isin(ids)].copy())
    n.connectors = CONNECTORS[CONNECTORS.node_id.isin(ids)].copy()
    return n


@pytest.fixture(params=[True, False], ids=["fastcore", "fallback"])
def backend(request, monkeypatch):
    if not request.param:
        monkeypatch.setattr(utils, "fastcore", None)
    return request.param


@pytest.mark.parametrize("func,col", METRICS, ids=IDS)
def test_flow_is_per_fragment(func, col, backend):
    """A fragment's flow must not depend on the other fragments in the node table."""
    truth = {}
    for ids in (FRAG_A, FRAG_B):
        out = func(_neuron(ids))
        truth.update(dict(zip(out.nodes.node_id, out.nodes[col])))

    out = func(_neuron())  # both fragments as one (disconnected) neuron
    got = dict(zip(out.nodes.node_id, out.nodes[col]))

    for node in NODES.node_id:
        assert np.isclose(got[node], truth[node], rtol=1e-6), (
            f"{col} at node {node}: {got[node]} as a forest vs "
            f"{truth[node]} for the fragment on its own"
        )


@pytest.mark.parametrize("func,col", METRICS, ids=IDS)
def test_flow_backends_agree_on_fragmented_neuron(func, col, monkeypatch):
    """fastcore and the igraph/scipy fallback must agree - including on a forest."""
    if utils.fastcore is None:
        pytest.skip("navis-fastcore not installed")

    n = navis.example_neurons(kind="skeleton")[4]  # this one has two roots
    assert len(n.root) > 1, "expected a fragmented example neuron"

    fast = func(n.copy()).nodes[col].values.astype(float)

    monkeypatch.setattr(utils, "fastcore", None)
    slow = func(n.copy()).nodes[col].values.astype(float)

    assert np.allclose(fast, slow, rtol=1e-5, equal_nan=True)


def test_synapse_flow_centrality_known_values(backend):
    """Hand-checked values on the minimal two-fragment neuron.

    Node 1 has one presynapse distal to it (node 2) and one postsynapse proximal to it
    (node 0), both inside its own fragment -> centrifugal flow 1 * 1 = 1. The
    postsynapse at node 10 is in the *other* fragment and must not contribute.
    """
    out = navis.synapse_flow_centrality(_neuron())
    flow = dict(zip(out.nodes.node_id, out.nodes.synapse_flow_centrality))

    assert flow == {0: 0, 1: 1, 2: 1, 3: 0, 10: 0, 11: 1, 12: 1, 13: 0}


def test_flow_centrality_connected_neuron_unchanged(backend):
    """The single-root case must be untouched by the per-fragment totals."""
    n = navis.example_neurons(1, kind="skeleton")
    assert len(n.root) == 1

    flow = navis.flow_centrality(n.copy()).nodes.flow_centrality.values
    # Root has every leaf distal and none proximal -> no flow passes through it
    assert flow.max() > 0
    assert not np.isnan(flow).any()
