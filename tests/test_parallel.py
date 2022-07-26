import navis

import numpy as np


def test_parallel():
    # Load example neurons
    nl = navis.example_neurons(kind='skeleton')

    # Test decorator
    pr = navis.prune_by_strahler(nl, 1, parallel=True, inplace=False)

    assert isinstance(pr, navis.NeuronList)
    assert len(pr) == len(nl)
    assert pr[0].n_nodes < nl[0].n_nodes

    # Test apply
    pr = nl.apply(navis.prune_by_strahler, to_prune=1, inplace=False,
                  parallel=True)
    assert isinstance(pr, navis.NeuronList)
    assert len(pr) == len(nl)
    assert all(pr.n_nodes < nl.n_nodes)


def test_parallel_inplace():
    # Load example neurons
    nl = navis.example_neurons(kind='skeleton')

    # Test decorator with inplace=True
    pr = nl.copy()
    pr2 = navis.prune_by_strahler(pr, 1, parallel=True, inplace=True)
    assert len(pr) == len(pr2) == len(nl)
    assert pr[0].n_nodes == pr2[0].n_nodes

    # Test apply with inplace=True -> this should not work
    #pr = nl.apply(navis.prune_by_strahler, to_prune=1, inplace=True,
    #              parallel=True)
    #assert len(pr) == len(nl)
    #assert all(pr.n_nodes < nl.n_nodes)


def test_apply():
    # Load example neurons
    nl = navis.example_neurons(kind='skeleton')

    # Test apply
    ids = nl.apply(lambda x: x.id, parallel=False)
    assert isinstance(ids, list)
    assert len(ids) == len(nl)
    assert all(np.array(ids) == nl.id)

    # Test apply
    pr = nl.apply(navis.prune_by_strahler, to_prune=1, inplace=False,
                  parallel=False)
    assert isinstance(pr, navis.NeuronList)
    assert len(pr) == len(nl)
    assert all(pr.n_nodes < nl.n_nodes)

    # Test apply with inplace=True
    pr = nl.apply(navis.prune_by_strahler, to_prune=1, inplace=True,
                  parallel=False)
    assert isinstance(pr, navis.NeuronList)
    assert len(pr) == len(nl)
    assert all(pr.n_nodes == nl.n_nodes)
