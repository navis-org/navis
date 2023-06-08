import navis
from navis.connectivity import NeuronConnector


def test_neuron_connector():
    nrns = []
    for n in navis.example_neurons():
        n.name = f"{n.name}_{n.id}"
        nrns.append(n)

    conn = NeuronConnector(nrns)

    adj = conn.to_adjacency()
    assert len(adj) == len(nrns) + 1

    dg = conn.to_digraph()
    assert dg.number_of_nodes() == len(nrns) + 1

    mdg = conn.to_multidigraph()
    assert mdg.number_of_nodes() == dg.number_of_nodes()

    # todo: check edges
