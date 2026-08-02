"""Tests for `navis.downsample_neuron` on skeletons.

The focus is on what downsampling has to keep intact besides the node table:
connectors and tags refer to nodes by ID, and most of those nodes disappear.
"""

import navis
import numpy as np
import pandas as pd

import pytest


@pytest.fixture
def neuron():
    return navis.example_neurons(1)


def toy_neuron(coords, parents, **kwargs):
    """Build a Skeleton from explicit coordinates and parents."""
    nodes = pd.DataFrame(np.asarray(coords, dtype=np.float32), columns=["x", "y", "z"])
    nodes["node_id"] = np.arange(len(nodes))
    nodes["parent_id"] = np.asarray(parents, dtype=np.int64)
    nodes["radius"] = kwargs.pop("radius", 1.0)
    return navis.Skeleton(nodes, **kwargs)


def line(n_nodes, step=1.0):
    """Straight line of `n_nodes` along x, rooted at node 0."""
    coords = np.zeros((n_nodes, 3))
    coords[:, 0] = np.arange(n_nodes) * step
    return toy_neuron(coords, [-1] + list(range(n_nodes - 1)))


def dangling(n):
    """Connector node IDs that aren't in the node table."""
    if not n.has_connectors:
        return np.array([])
    return n.connectors.node_id.values[
        ~np.isin(n.connectors.node_id.values, n.nodes.node_id.values)
    ]


# =========================================================================== #
# Node table
# =========================================================================== #
@pytest.mark.parametrize("factor", [2, 5, 10, 100, float("inf")])
def test_downsampling_reduces_nodes(neuron, factor):
    ds = navis.downsample_neuron(neuron, factor)
    assert ds.n_nodes < neuron.n_nodes
    assert ds.id == neuron.id


def test_original_is_untouched(neuron):
    before = neuron.n_nodes
    navis.downsample_neuron(neuron, 10)
    assert neuron.n_nodes == before
    assert len(dangling(neuron)) == 0


@pytest.mark.parametrize("factor", [2, 5, 10, 100, float("inf")])
def test_root_branches_and_leafs_survive(neuron, factor):
    """These are the fix points - downsampling only ever thins slabs."""
    ds = navis.downsample_neuron(neuron, factor)
    kept = set(ds.nodes.node_id.values)
    assert set(np.asarray(neuron.root).ravel().tolist()) <= kept
    assert set(neuron.branch_points.node_id.values) <= kept
    assert set(neuron.ends.node_id.values) <= kept


def test_infinite_factor_keeps_only_fix_points(neuron):
    ds = navis.downsample_neuron(neuron, float("inf"))
    expected = (
        set(neuron.branch_points.node_id.values)
        | set(neuron.ends.node_id.values)
        | set(np.asarray(neuron.root).ravel().tolist())
        | {neuron.soma}
    )
    assert set(ds.nodes.node_id.values) == expected


def test_factor_of_one_raises(neuron):
    with pytest.raises(ValueError):
        navis.downsample_neuron(neuron, 1)


# =========================================================================== #
# Connectors and tags must not be left pointing at deleted nodes
# =========================================================================== #
@pytest.mark.parametrize("factor", [2, 5, 10, 100, float("inf")])
def test_connectors_never_dangle(neuron, factor):
    ds = navis.downsample_neuron(neuron, factor)
    assert len(ds.connectors) == len(neuron.connectors)
    assert len(dangling(ds)) == 0


@pytest.mark.parametrize("factor", [2, 10, float("inf")])
def test_tags_never_dangle(neuron, factor):
    n = neuron.copy()
    n.tags = {"soma": [n.soma], "spread": n.nodes.node_id.values[::37].tolist()}

    ds = navis.downsample_neuron(n, factor)
    kept = set(ds.nodes.node_id.values)

    assert set(ds.tags) == set(n.tags)
    for tag, nodes in ds.tags.items():
        assert len(nodes) == len(n.tags[tag])
        assert set(nodes) <= kept


def test_connectors_move_to_the_nearest_surviving_node():
    """On a straight line the survivor is unambiguous, so we can name it."""
    n = line(11)
    # One connector on every node
    n.connectors = pd.DataFrame(
        {
            "connector_id": np.arange(11),
            "node_id": np.arange(11),
            "type": 0,
            "x": n.nodes.x.values,
            "y": 0.0,
            "z": 0.0,
        }
    )

    # No branch points, so only root (0) and leaf (10) are fix points
    ds = navis.downsample_neuron(n, float("inf"))
    assert set(ds.nodes.node_id.values) == {0, 10}

    # Nodes 0-4 are closer to the root, 6-10 to the leaf; node 5 is equidistant
    # and may go either way.
    mapped = dict(zip(ds.connectors.connector_id, ds.connectors.node_id))
    assert [mapped[i] for i in range(5)] == [0] * 5
    assert [mapped[i] for i in range(6, 11)] == [10] * 5
    assert mapped[5] in (0, 10)


def test_connectors_stay_put_with_preserve_nodes(neuron):
    """`preserve_nodes="connectors"` pins them exactly, so nothing moves."""
    ds = navis.downsample_neuron(neuron, 10, preserve_nodes="connectors")

    assert len(dangling(ds)) == 0
    pd.testing.assert_series_equal(
        ds.connectors.node_id, neuron.connectors.node_id, check_dtype=False
    )


def test_connectors_do_not_jump_branches():
    """A spatially close node on another branch must not win over the real one.

    Two parallel branches run close together; the connector sits on one of them.
    A nearest-neighbour search in space could snap it across the gap, a
    geodesic one cannot.
    """
    # Branch A along y=0, branch B along y=0.1, both hanging off a shared root
    coords = [[0, 0, 0]]
    parents = [-1]
    for i in range(1, 11):  # branch A: nodes 1-10
        coords.append([i, 0, 0])
        parents.append(i - 1)
    for i in range(1, 11):  # branch B: nodes 11-20
        coords.append([i, 0.1, 0])
        parents.append(0 if i == 1 else 9 + i)
    n = toy_neuron(coords, parents)

    # Connector in the middle of branch A
    n.connectors = pd.DataFrame(
        {"connector_id": [0], "node_id": [5], "type": [0], "x": [5.0], "y": [0.0], "z": [0.0]}
    )

    ds = navis.downsample_neuron(n, float("inf"))

    # Survivors are the root and the two tips; the connector must land on
    # branch A's tip (10) or the root (0) - never on branch B's tip (20).
    assert ds.connectors.node_id.iloc[0] in (0, 10)


def test_no_connectors_or_tags_is_fine(neuron):
    n = neuron.copy()
    n.connectors = None
    n.tags = None
    ds = navis.downsample_neuron(n, 10)
    assert ds.n_nodes < neuron.n_nodes
    assert not ds.has_connectors


def test_fragmented_neuron_remaps_within_each_fragment(neuron):
    """Every fragment keeps its own root, so there is always a survivor."""
    for frag in navis.cut_skeleton(neuron, neuron.nodes.node_id.values[500]):
        ds = navis.downsample_neuron(frag, 10)
        assert len(dangling(ds)) == 0


def test_neuronlist_roundtrip():
    nl = navis.example_neurons(2)
    ds = navis.downsample_neuron(nl, 10)
    assert isinstance(ds, navis.NeuronList)
    for before, after in zip(nl, ds):
        assert after.n_nodes < before.n_nodes
        assert len(dangling(after)) == 0
