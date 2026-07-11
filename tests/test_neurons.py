from copy import deepcopy

import navis
import numpy as np
import pandas as pd

import pytest


def test_deepcopy():
    nrn = navis.core.BaseNeuron()
    deepcopy(nrn)


@pytest.mark.parametrize("op", ["mul", "truediv", "add", "sub"])
def test_mesh_math_connectors(op):
    """Arithmetic must not choke on integer connector coordinates."""
    n = navis.example_neurons(1, kind="mesh")
    # Make sure the example neuron actually has integer connectors
    assert n.has_connectors
    assert np.issubdtype(n.connectors[["x", "y", "z"]].values.dtype, np.integer)

    verts = np.asarray(n.vertices).copy()
    conns = n.connectors[["x", "y", "z"]].values.astype(float).copy()

    if op == "mul":
        m = n * 2
        assert np.allclose(m.vertices, verts * 2)
        assert np.allclose(m.connectors[["x", "y", "z"]].values, conns * 2)
    elif op == "truediv":
        m = n / 2
        assert np.allclose(m.vertices, verts / 2)
        assert np.allclose(m.connectors[["x", "y", "z"]].values, conns / 2)
    elif op == "add":
        m = n + 5
        assert np.allclose(m.vertices, verts + 5)
        assert np.allclose(m.connectors[["x", "y", "z"]].values, conns + 5)
    elif op == "sub":
        m = n - 5
        assert np.allclose(m.vertices, verts - 5)
        assert np.allclose(m.connectors[["x", "y", "z"]].values, conns - 5)


@pytest.mark.parametrize("op", ["mul", "truediv", "add", "sub"])
def test_skeleton_math_integer_nodes(op):
    """Arithmetic must cast integer node coordinates to float when needed."""
    n = navis.example_neurons(1, kind="skeleton")
    # Force integer node (and connector) coordinates
    n.nodes[["x", "y", "z"]] = n.nodes[["x", "y", "z"]].round().astype("int64")
    if n.has_connectors:
        n.connectors[["x", "y", "z"]] = (
            n.connectors[["x", "y", "z"]].round().astype("int64")
        )

    nodes = n.nodes[["x", "y", "z"]].values.astype(float).copy()

    if op == "mul":
        assert np.allclose((n * 2.5).nodes[["x", "y", "z"]].values, nodes * 2.5)
    elif op == "truediv":
        assert np.allclose((n / 2.5).nodes[["x", "y", "z"]].values, nodes / 2.5)
    elif op == "add":
        assert np.allclose((n + 2.5).nodes[["x", "y", "z"]].values, nodes + 2.5)
    elif op == "sub":
        assert np.allclose((n - 2.5).nodes[["x", "y", "z"]].values, nodes - 2.5)


@pytest.mark.parametrize("op", ["mul", "truediv", "add", "sub"])
def test_skeleton_math_connectors(op):
    """Arithmetic must not choke on integer connector coordinates."""
    n = navis.example_neurons(1, kind="skeleton")
    assert n.has_connectors
    n.connectors[["x", "y", "z"]] = (
        n.connectors[["x", "y", "z"]].round().astype("int64")
    )
    conns = n.connectors[["x", "y", "z"]].values.astype(float).copy()

    if op == "mul":
        assert np.allclose((n * 2).connectors[["x", "y", "z"]].values, conns * 2)
    elif op == "truediv":
        assert np.allclose((n / 2).connectors[["x", "y", "z"]].values, conns / 2)
    elif op == "add":
        assert np.allclose((n + 5).connectors[["x", "y", "z"]].values, conns + 5)
    elif op == "sub":
        assert np.allclose((n - 5).connectors[["x", "y", "z"]].values, conns - 5)


@pytest.mark.parametrize("kind", ["mesh", "skeleton"])
def test_neuronlist_inplace_math(kind):
    """``nl *= x`` / ``nl /= x`` must mutate in place without copying neurons."""
    nl = navis.example_neurons(3, kind=kind)

    neuron_ids = [id(n) for n in nl]
    before = [np.asarray(n.vertices if kind == "mesh" else
                         n.nodes[["x", "y", "z"]].values).copy() for n in nl]

    nl *= 3

    # Same neuron objects -> no copies were made
    assert [id(n) for n in nl] == neuron_ids
    assert isinstance(nl, navis.NeuronList)
    for n, b in zip(nl, before):
        coords = n.vertices if kind == "mesh" else n.nodes[["x", "y", "z"]].values
        assert np.allclose(coords, b * 3)

    nl /= 3
    for n, b in zip(nl, before):
        coords = n.vertices if kind == "mesh" else n.nodes[["x", "y", "z"]].values
        assert np.allclose(coords, b)


def test_neuronlist_inplace_mesh_no_copy():
    """In-place ``*=`` on a mesh NeuronList must not reallocate vertex arrays."""
    nl = navis.example_neurons(2, kind="mesh")
    vertex_ids = [id(n._vertices) for n in nl]

    nl *= 2

    # The underlying vertex arrays must be the very same objects (mutated in
    # place) rather than freshly allocated copies.
    assert [id(n._vertices) for n in nl] == vertex_ids


def test_from_swc(swc_source):
    n = navis.read_swc(swc_source)
    assert isinstance(n, navis.TreeNeuron)


@pytest.mark.parametrize("parallel", ["auto", True, 2, False])
def test_from_swc_multi(swc_source_multi, parallel):
    n = navis.read_swc(swc_source_multi, parallel=parallel)
    assert isinstance(n, navis.NeuronList)


def test_from_gml():
    n = navis.example_neurons(n=1, source='gml')
    assert isinstance(n, navis.TreeNeuron)


def test_empty_skeleton_graph_functions():
    """Graph functions must not choke on a neuron without nodes.

    An empty node table produces a zero-vertex igraph, and `bool()` on such a
    graph is False - so these used to quietly take a networkx code path.
    """
    nodes = pd.DataFrame(
        {
            "node_id": pd.Series([], dtype=np.int64),
            "parent_id": pd.Series([], dtype=np.int64),
            "x": pd.Series([], dtype=float),
            "y": pd.Series([], dtype=float),
            "z": pd.Series([], dtype=float),
        }
    )
    n = navis.TreeNeuron(nodes)

    assert n.n_nodes == 0
    assert not bool(n.igraph)

    assert len(n.segments) == 0
    assert len(navis.graph.graph_utils._connected_components(n)) == 0
    assert len(navis.graph.graph_utils._break_segments(n)) == 0
    assert navis.geodesic_matrix(n).empty
    assert len(n.root) == 0
