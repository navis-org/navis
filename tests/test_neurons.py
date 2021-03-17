import navis

from .common import with_igraph


def test_from_swc(swc_source):
    n = navis.read_swc(swc_source)
    assert isinstance(n, navis.TreeNeuron)


def test_from_swc_many


@with_igraph
def test_from_gml():
    n = navis.example_neurons(n=1, source='gml')
    assert isinstance(n, navis.TreeNeuron)
