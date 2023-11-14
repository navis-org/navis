from typing import List, Tuple, Dict

import pytest
import pandas as pd
import numpy as np

import navis
from navis.connectivity import NeuronConnector
from navis import NeuronList


def test_neuron_connector():
    nrns = []
    for n in navis.example_neurons():
        n.name = f"{n.name}_{n.id}"
        nrns.append(n)

    conn = NeuronConnector(nrns)

    adj = conn.to_adjacency()
    assert len(adj) == len(nrns) + 1
    assert adj.to_numpy().sum() == 0

    dg = conn.to_digraph()
    assert dg.number_of_nodes() == len(nrns) + 1
    assert dg.number_of_edges() == 0

    mdg = conn.to_multidigraph()
    assert mdg.number_of_nodes() == dg.number_of_nodes()
    assert mdg.number_of_edges() == 0


def path_neuron(path: List[int]):
    nrn = navis.TreeNeuron(None)
    nrn.name = "".join(str(n) for n in path)
    dtypes = {
        "node_id": np.uint64,
        "parent_id": np.int64,
        "x": float,
        "y": float,
        "z": float,
    }
    prev = -1
    rows = []
    for n in path:
        rows.append([n, prev, 0, 0, 0])
        prev = n
    df = pd.DataFrame(rows, columns=list(dtypes)).astype(dtypes)
    nrn.nodes = df
    return nrn


def add_connectors(
    nrn: navis.TreeNeuron,
    incoming: List[Tuple[int, int]],
    outgoing: List[Tuple[int, int]],
):
    """Add connectors to a neuron.

    Parameters
    ----------
    incoming : list[tuple[int, int]]
        List of connector_id, node_id pairs.
    outgoing : list[tuple[int, int]]
        List of connector_id, node_id pairs
    """
    dtypes = {
        "connector_id": np.uint64,
        "node_id": np.uint64,
        "x": float,
        "y": float,
        "z": float,
        "type": np.uint64,
    }
    rows = []
    for tc_rel, ids in [(1, incoming), (0, outgoing)]:
        for conn, node in ids:
            rows.append([conn, node, 0, 0, 0, tc_rel])
    df = pd.DataFrame(rows, columns=list(dtypes)).astype(dtypes)
    nrn.connectors = df


@pytest.fixture
def simple_network():
    """
    2 neurons, "456" and "789".
    4 connectors, 0/1/2/3.

        4     7
        |     |
    =0> 5 =1> 8 =2>
        |     |
        6 =3> 9
    """
    n456 = path_neuron([4, 5, 6])
    add_connectors(n456, [(0, 5)], [(1, 5), (3, 6)])
    n789 = path_neuron([7, 8, 9])
    add_connectors(n789, [(1, 8), (3, 9)], [(2, 8)])
    return [n456, n789]


def test_neuron_connector_synthetic(simple_network):
    nconn = NeuronConnector(simple_network)

    adj = nconn.to_adjacency()
    assert len(adj) == len(simple_network) + 1
    assert adj.to_numpy().sum() == 4

    expected_edges = sorted([
        ("__OTHER__", "456"),
        ("456", "789"),
        ("456", "789"),
        ("789", "__OTHER__"),
    ])

    dg = nconn.to_digraph()
    assert dg.number_of_nodes() == len(simple_network) + 1
    assert dg.number_of_edges() == 3
    assert set(dg.edges()) == set(expected_edges)

    mdg = nconn.to_multidigraph()
    assert mdg.number_of_nodes() == dg.number_of_nodes()
    assert mdg.number_of_edges() == 4
    assert sorted(mdg.edges()) == expected_edges


def test_neuron_connector_real(
    neuron_connections: Tuple[NeuronList, Dict[int, Dict[int, int]]]
):
    nl, exp = neuron_connections
    nc = NeuronConnector(nl)
    dg = nc.to_digraph(include_other=False)
    for pre_n, post_n, edata in dg.edges(data=True):
        pre_skid = dg.nodes[pre_n]["neuron"].id
        post_skid = dg.nodes[post_n]["neuron"].id
        n = edata["weight"]

        assert exp[pre_skid][post_skid] == n
