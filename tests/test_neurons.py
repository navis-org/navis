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


@pytest.mark.parametrize("validate", [True, False])
def test_edges2neuron(validate):
    """`validate=False` used to raise an UnboundLocalError."""
    # A simple Y: 0 -> 1 -> 2, with 3 branching off 1
    verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=float)
    # Edges are (child, parent) - which is what `validate=False` trusts them to be
    edges = np.array([[0, 1], [2, 1], [3, 1]])

    n = navis.edges2neuron(edges, vertices=verts, validate=validate)

    assert isinstance(n, navis.TreeNeuron)
    assert n.n_nodes == 4
    assert n.is_tree
    # Same tree either way, regardless of how it ends up rooted
    edge_set = {
        frozenset((c, p))
        for c, p in zip(n.nodes.node_id.values, n.nodes.parent_id.values)
        if p >= 0
    }
    assert edge_set == {frozenset((0, 1)), frozenset((2, 1)), frozenset((3, 1))}


def test_edges2neuron_cycle():
    """Cycles must be broken (only relevant with validate=True)."""
    verts = np.zeros((4, 3), dtype=float)
    verts[:, 0] = np.arange(4)
    # 0-1-2-0 is a cycle, 3 hangs off 2
    edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3]])

    n = navis.edges2neuron(edges, vertices=verts)

    assert n.n_nodes == 4
    assert n.is_tree  # cycle was broken


# --------------------------------------------------------------------------- #
# memory_usage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["skeleton", "mesh", "dotprops", "voxels"])
def test_memory_usage_estimate_matches_the_slow_path(kind):
    """`estimate=True` prices columns from their dtype instead of walking them.

    It is allowed to come in a little low, but it has to work: it used to raise
    on any neuron carrying connectors, because from pandas 3 a text column is a
    `StringDtype`, which has no `itemsize`.
    """
    if kind in ("skeleton", "mesh"):
        n = navis.example_neurons(1, kind=kind)
    elif kind == "dotprops":
        n = navis.make_dotprops(navis.example_neurons(1), k=5)
    else:
        n = navis.voxelize(navis.example_neurons(1), pitch="2 microns")

    exact = n.memory_usage(estimate=False)
    estimate = n.memory_usage(estimate=True)

    assert estimate > 0
    assert 0.8 <= estimate / exact <= 1.0


def test_memory_usage_handles_extension_dtypes():
    """Neurons with connectors carry text columns; those are the awkward ones."""
    n = navis.example_neurons(1)
    assert n.has_connectors
    # An extension dtype with no `itemsize` at all is the case that used to blow up
    assert any(not hasattr(dt, "itemsize") for dt in n.connectors.dtypes)

    # Not just non-zero - the connectors have to actually be counted
    without = navis.example_neurons(1)
    without._connectors = None
    assert n.memory_usage(estimate=True) > without.memory_usage(estimate=True)


def test_memory_usage_counts_categorical_codes():
    """The per-row codes dwarf the categories, so they cannot be skipped."""
    n = navis.example_neurons(1)
    cat_cols = [c for c, dt in n.nodes.dtypes.items()
                if isinstance(dt, pd.CategoricalDtype)]
    assert cat_cols, "example skeleton should have categorical node columns"

    estimate = n.memory_usage(estimate=True)
    # One byte per row per categorical column is the floor for the codes alone
    assert estimate > n.n_nodes * len(cat_cols)


def test_neuronlist_memory_usage():
    nl = navis.example_neurons(5)

    assert nl.memory_usage(estimate=True) > 0
    assert isinstance(nl.memory_usage(estimate=True), int)
    # Roughly the sum of its parts
    parts = sum(n.memory_usage(estimate=True) for n in nl)
    assert nl.memory_usage(estimate=True) == parts

    # Sampling extrapolates from every 10th neuron, so it is approximate but
    # must be in the right ballpark rather than zero
    sampled = nl.memory_usage(estimate=True, sample=True)
    assert 0.2 <= sampled / parts <= 5

    assert navis.NeuronList([]).memory_usage() == 0


def test_neuronlist_memory_usage_survives_a_broken_neuron():
    """It backs `__str__`, so it must not be why a NeuronList cannot print."""
    class Broken(navis.TreeNeuron):
        def memory_usage(self, deep=False, estimate=False):
            raise RuntimeError("no idea")

    nl = navis.NeuronList([Broken(navis.example_neurons(1))])
    assert nl.memory_usage(estimate=True) == 0
    assert "NeuronList" in str(nl)
