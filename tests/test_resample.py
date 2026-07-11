import navis
import numpy as np
import pandas as pd

import pytest


@pytest.fixture
def neuron():
    return navis.example_neurons(1)


def toy_neuron(coords, parents, **kwargs):
    """Build a TreeNeuron from explicit coordinates and parents."""
    nodes = pd.DataFrame(np.asarray(coords, dtype=np.float32), columns=["x", "y", "z"])
    nodes["node_id"] = np.arange(len(nodes))
    nodes["parent_id"] = np.asarray(parents, dtype=np.int64)
    nodes["radius"] = kwargs.pop("radius", 1.0)
    return navis.TreeNeuron(nodes, **kwargs)


def line(n_nodes, step=1.0):
    """Straight line of `n_nodes` along x, rooted at the far end."""
    coords = np.zeros((n_nodes, 3))
    coords[:, 0] = np.arange(n_nodes) * step
    return toy_neuron(coords, [-1] + list(range(n_nodes - 1)))


def test_resolution_moves_toward_target(neuron):
    """The achieved sampling resolution should track the requested one."""
    res = {
        to: navis.resample_skeleton(neuron, resample_to=to, inplace=False)
        for to in (50, 125, 500)
    }

    # A coarser target must give a coarser neuron
    achieved = [res[to].sampling_resolution for to in (50, 125, 500)]
    assert achieved == sorted(achieved)

    # For targets that the neuron's segments can actually accommodate we should
    # land close to what was asked for
    assert res[50].sampling_resolution == pytest.approx(50, rel=0.1)
    assert res[125].sampling_resolution == pytest.approx(125, rel=0.15)

    # The coarsest target undershoots, and that's expected: segments shorter than
    # `resample_to` are collapsed to their two ends and keep contributing nodes
    assert res[500].sampling_resolution < 500


def test_preserves_topology(neuron):
    """Root, leafs and branch points must survive resampling."""
    res = navis.resample_skeleton(neuron, resample_to=125, inplace=False)

    assert res.n_trees == neuron.n_trees
    assert res.n_branches == neuron.n_branches
    assert res.n_leafs == neuron.n_leafs
    assert set(navis.utils.make_iterable(res.root)) == set(
        navis.utils.make_iterable(neuron.root)
    )
    # Cable length is approximately preserved. Resampling cuts corners, so the
    # resampled neuron can only ever be shorter than the original.
    assert res.cable_length <= neuron.cable_length
    assert res.cable_length == pytest.approx(neuron.cable_length, rel=0.1)


def test_node_ids_unique(neuron):
    res = navis.resample_skeleton(neuron, resample_to=40, inplace=False)

    assert not res.nodes.node_id.duplicated().any()
    # Every parent must exist (or be a root)
    parents = res.nodes.parent_id.values
    assert np.isin(parents[parents >= 0], res.nodes.node_id.values).all()


def test_straight_line():
    """A straight line is resampled into nodes spaced exactly `resample_to` apart."""
    n = line(101)  # 101 nodes, 100 units long

    res = navis.resample_skeleton(n, resample_to=10, inplace=False)

    # 100 units at one node every 10 units == 11 nodes (10 intervals + 1)
    x = np.sort(res.nodes.x.values)
    assert res.n_nodes == 11
    assert np.allclose(x, np.arange(0, 101, 10))


@pytest.mark.parametrize("resample_to", [4, 7, 10, 25])
def test_spacing_matches_target(resample_to):
    """Nodes must actually end up `resample_to` apart - not a fraction more."""
    n = line(101)  # straight, 100 units long

    res = navis.resample_skeleton(n, resample_to=resample_to, inplace=False)

    x = np.sort(res.nodes.x.values)
    # The segment can't always be divided evenly, so the spacing is the closest
    # we can get while still landing on both ends
    expected = 100 / round(100 / resample_to)
    assert np.allclose(np.diff(x), expected)
    assert abs(expected - resample_to) <= resample_to * 0.5


def test_interpolates_radius():
    """Radius must be interpolated, not dropped or zeroed."""
    n = line(11)
    n.nodes["radius"] = np.arange(11, dtype=np.float32)  # 0 at root end .. 10

    res = navis.resample_skeleton(n, resample_to=5, inplace=False)

    # Radius should still track x, since both are linear along the line
    assert np.allclose(res.nodes.radius.values, res.nodes.x.values, atol=1e-4)


def test_remaps_connectors_and_soma(neuron):
    assert neuron.has_connectors
    res = navis.resample_skeleton(neuron, resample_to=125, inplace=False)

    # Connectors must point at nodes that actually exist in the new neuron
    assert res.n_connectors == neuron.n_connectors
    assert np.isin(res.connectors.node_id.values, res.nodes.node_id.values).all()

    # Soma must be pinned to an existing node, and stay in roughly the same place
    assert res.soma is not None
    soma = navis.utils.make_iterable(res.soma)
    assert np.isin(soma, res.nodes.node_id.values).all()
    assert np.allclose(res.soma_pos, neuron.soma_pos, atol=125)


def test_remaps_tags(neuron):
    neuron = neuron.copy()
    neuron.tags = {"of interest": list(neuron.nodes.node_id.values[:5])}

    res = navis.resample_skeleton(neuron, resample_to=125, inplace=False)

    assert set(res.tags) == {"of interest"}
    assert np.isin(res.tags["of interest"], res.nodes.node_id.values).all()


@pytest.mark.parametrize("method", ["linear", "cubic", "quadratic", "nearest"])
def test_methods(neuron, method):
    res = navis.resample_skeleton(neuron, resample_to=125, method=method, inplace=False)

    assert res.n_nodes < neuron.n_nodes
    assert res.nodes[["x", "y", "z"]].notnull().values.all()


def test_map_columns_numeric(neuron):
    neuron = neuron.copy()
    # A column that increases linearly with node index interpolates predictably
    neuron.nodes["depth"] = np.arange(neuron.n_nodes, dtype=float)

    res = navis.resample_skeleton(
        neuron, resample_to=125, map_columns=["depth"], inplace=False
    )

    assert "depth" in res.nodes.columns
    assert res.nodes.depth.notnull().all()
    # Interpolated values must stay inside the original range
    assert res.nodes.depth.min() >= neuron.nodes.depth.min()
    assert res.nodes.depth.max() <= neuron.nodes.depth.max()


def test_map_columns_categorical(neuron):
    """Non-numeric columns are mapped by nearest neighbour, not interpolated."""
    res = navis.resample_skeleton(
        neuron, resample_to=125, map_columns=["label"], inplace=False
    )

    assert "label" in res.nodes.columns
    # Every label must be one that existed in the original neuron - a categorical
    # must never be averaged into a value that was never there
    assert set(res.nodes.label.unique()).issubset(set(neuron.nodes.label.unique()))


def test_map_columns_categorical_with_nan(neuron):
    """NaNs in a non-numeric column must not decode into a real category."""
    neuron = neuron.copy()
    labels = neuron.nodes.label.astype(object).values.copy()
    labels[::7] = np.nan
    neuron.nodes["tag"] = labels

    res = navis.resample_skeleton(
        neuron, resample_to=125, map_columns=["tag"], inplace=False
    )

    assert pd.isna(res.nodes.tag).any()
    values = set(pd.unique(res.nodes.tag.dropna()))
    assert values.issubset(set(pd.unique(neuron.nodes.tag.dropna())))


def test_map_columns_string():
    """`map_columns` accepts a bare string as well as a list."""
    n = line(51)
    n.nodes["tag"] = "a"

    res = navis.resample_skeleton(n, resample_to=10, map_columns="tag", inplace=False)

    assert (res.nodes.tag == "a").all()


def test_map_columns_missing(neuron):
    with pytest.raises(ValueError, match="not found"):
        navis.resample_skeleton(
            neuron, resample_to=125, map_columns=["nonexistent"], inplace=False
        )


def test_short_segments_kept():
    """Segments shorter than the target resolution collapse to first + last node."""
    n = line(11)  # 10 units long

    res = navis.resample_skeleton(n, resample_to=1000, inplace=False)

    # Whole neuron is one segment, far shorter than 1000 -> just the two ends
    assert res.n_nodes == 2
    assert set(res.nodes.x.values) == {0.0, 10.0}


def test_zero_length_edges():
    """Coincident nodes must not blow up the distance calculation."""
    coords = np.zeros((11, 3))
    coords[:, 0] = [0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7]  # several duplicate positions
    n = toy_neuron(coords, [-1] + list(range(10)))

    res = navis.resample_skeleton(n, resample_to=2, inplace=False)

    assert res.nodes[["x", "y", "z"]].notnull().values.all()
    assert not res.nodes.node_id.duplicated().any()
    assert res.cable_length == pytest.approx(n.cable_length, rel=0.01)


def test_single_node_neuron():
    n = toy_neuron([[0, 0, 0]], [-1])

    res = navis.resample_skeleton(n, resample_to=10, inplace=False)

    assert res.n_nodes == 1
    assert res.nodes.parent_id.iloc[0] == -1


def test_inplace(neuron):
    original = neuron.copy()

    out = navis.resample_skeleton(neuron, resample_to=125, inplace=True)

    assert out is neuron
    assert neuron.n_nodes != original.n_nodes


def test_neuronlist():
    nl = navis.example_neurons(2)

    res = navis.resample_skeleton(nl, resample_to=125, inplace=False)

    assert isinstance(res, navis.NeuronList)
    assert len(res) == 2
    assert all(r.n_nodes < n.n_nodes for r, n in zip(res, nl))


def test_units(neuron):
    """A string resolution is resolved via the neuron's units."""
    by_str = navis.resample_skeleton(neuron, resample_to="1 micron", inplace=False)
    # Example neurons are in 8x8x8 nm voxels, so 1 micron == 125 voxels
    by_num = navis.resample_skeleton(neuron, resample_to=125, inplace=False)

    assert by_str.n_nodes == by_num.n_nodes


def test_dtypes_preserved(neuron):
    """Resampling must not silently upcast the node table."""
    res = navis.resample_skeleton(neuron, resample_to=125, inplace=False)

    for col in ("x", "y", "z", "radius"):
        assert res.nodes[col].dtype == neuron.nodes[col].dtype
    assert np.issubdtype(res.nodes.node_id.dtype, np.integer)
    assert np.issubdtype(res.nodes.parent_id.dtype, np.integer)


def test_skip_errors():
    """`skip_errors` falls back to the original nodes instead of raising."""
    # "quadratic" needs >= 3 points; a 2-node segment can't be fitted
    n = line(2, step=100)

    res = navis.resample_skeleton(
        n, resample_to=10, method="quadratic", skip_errors=True, inplace=False
    )

    # Segment could not be interpolated -> original nodes are kept
    assert res.n_nodes == n.n_nodes
    assert np.allclose(
        np.sort(res.nodes.x.values), np.sort(n.nodes.x.values.astype(float))
    )

    with pytest.raises(ValueError):
        navis.resample_skeleton(
            n, resample_to=10, method="quadratic", skip_errors=False, inplace=False
        )
