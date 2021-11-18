from copy import deepcopy

import navis

import pytest

from .common import with_igraph


def test_deepcopy():
    nrn = navis.core.BaseNeuron()
    deepcopy(nrn)


def test_from_swc(swc_source):
    n = navis.read_swc(swc_source)
    assert isinstance(n, navis.TreeNeuron)


@pytest.mark.parametrize("parallel", ["auto", True, 2, False])
def test_from_swc_multi(swc_source_multi, parallel):
    n = navis.read_swc(swc_source_multi, parallel=parallel)
    assert isinstance(n, navis.NeuronList)


@with_igraph
def test_from_gml():
    n = navis.example_neurons(n=1, source='gml')
    assert isinstance(n, navis.TreeNeuron)
