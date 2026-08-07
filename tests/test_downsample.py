"""Tests for `navis.downsample_neuron` on skeletons.

The focus is on what downsampling has to keep intact besides the node table:
connectors and tags refer to nodes by ID, and most of those nodes disappear.
"""

import navis
import numpy as np
import pandas as pd

import pytest

# The methods that thin a skeleton by *shape* rather than by counting. Their
# second argument is a distance tolerance, not a factor, so anything
# parametrised over both families has to supply the number per method.
SHAPE_METHODS = ["rdp", "vw"]

# `(method, amount)` spanning all three methods, from barely-thinning to
# as-hard-as-it-goes. Everything that has to hold whatever the method and
# however hard it is pushed is parametrised over this. The example neurons are
# in 8nm voxels, so the tolerances run from a third of a micron to 40 of them.
ALL_SETTINGS = [
    ("simple", 2),
    ("simple", 5),
    ("simple", 10),
    ("simple", 100),
    ("simple", float("inf")),
    ("rdp", 40),
    ("rdp", 500),
    ("rdp", 5000),
    ("vw", 40),
    ("vw", 500),
    ("vw", 5000),
]

# One setting per method, for the tests that only need "some downsampling
# happened" and would learn nothing from running eleven times.
ONE_PER_METHOD = [("simple", 10), ("rdp", 500), ("vw", 500)]


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
    # `range` rather than `[-1] + ...` so that `n_nodes=0` gives no parents
    # rather than a root with no node to be.
    return toy_neuron(coords, list(range(-1, n_nodes - 1)))


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
@pytest.mark.parametrize("method,amount", ALL_SETTINGS)
def test_downsampling_reduces_nodes(neuron, method, amount):
    ds = navis.downsample_neuron(neuron, amount, method=method)
    assert ds.n_nodes < neuron.n_nodes
    assert ds.id == neuron.id


def test_original_is_untouched(neuron):
    before = neuron.n_nodes
    navis.downsample_neuron(neuron, 10)
    assert neuron.n_nodes == before
    assert len(dangling(neuron)) == 0


@pytest.mark.parametrize("method,amount", ALL_SETTINGS)
def test_root_branches_and_leafs_survive(neuron, method, amount):
    """These are the fix points - downsampling only ever thins slabs."""
    ds = navis.downsample_neuron(neuron, amount, method=method)
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
@pytest.mark.parametrize("method,amount", ALL_SETTINGS)
def test_connectors_never_dangle(neuron, method, amount):
    ds = navis.downsample_neuron(neuron, amount, method=method)
    assert len(ds.connectors) == len(neuron.connectors)
    assert len(dangling(ds)) == 0


@pytest.mark.parametrize(
    "method,amount", [("simple", 2), ("simple", float("inf")), ("rdp", 500), ("vw", 500)]
)
def test_tags_never_dangle(neuron, method, amount):
    n = neuron.copy()
    n.tags = {"soma": [n.soma], "spread": n.nodes.node_id.values[::37].tolist()}

    ds = navis.downsample_neuron(n, amount, method=method)
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
    # and goes to the root, ties being broken towards it.
    mapped = dict(zip(ds.connectors.connector_id, ds.connectors.node_id))
    assert [mapped[i] for i in range(5)] == [0] * 5
    assert [mapped[i] for i in range(6, 11)] == [10] * 5
    assert mapped[5] == 0


@pytest.mark.parametrize("method,amount", ONE_PER_METHOD)
def test_connectors_stay_put_with_preserve_nodes(neuron, method, amount):
    """`preserve_nodes="connectors"` pins them exactly, so nothing moves."""
    ds = navis.downsample_neuron(
        neuron, amount, method=method, preserve_nodes="connectors"
    )

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


@pytest.mark.parametrize("method,amount", ONE_PER_METHOD)
def test_fragmented_neuron_remaps_within_each_fragment(neuron, method, amount):
    """Every fragment keeps its own root, so there is always a survivor."""
    for frag in navis.cut_skeleton(neuron, neuron.nodes.node_id.values[500]):
        ds = navis.downsample_neuron(frag, amount, method=method)
        assert len(dangling(ds)) == 0


@pytest.mark.parametrize("method,amount", ONE_PER_METHOD)
def test_neuronlist_roundtrip(method, amount):
    nl = navis.example_neurons(2)
    ds = navis.downsample_neuron(nl, amount, method=method)
    assert isinstance(ds, navis.NeuronList)
    for before, after in zip(nl, ds):
        assert after.n_nodes < before.n_nodes
        assert len(dangling(after)) == 0


# =========================================================================== #
# Thinning by shape rather than by count
# =========================================================================== #
@pytest.mark.parametrize("method", SHAPE_METHODS)
def test_higher_tolerance_keeps_fewer_nodes(neuron, method):
    """The whole point of a tolerance: it has to buy something monotonically."""
    counts = [
        navis.downsample_neuron(neuron, tol, method=method).n_nodes
        for tol in (10, 100, 1000, 10000)
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_rdp_at_zero_only_drops_collinear_nodes():
    """`0` is the identity for RDP everywhere the neuron actually bends."""
    # A perfectly straight line has nothing but collinear interior nodes, so it
    # collapses to its two ends...
    assert navis.downsample_neuron(line(11), 0, method="rdp").n_nodes == 2

    # ...and one node nudged off the line has to survive, however slight the
    # nudge. Its two neighbours survive with it: node 4 is exactly on the line
    # 3 -> 5 only while node 5 is, and it no longer is.
    kinked = line(11)
    kinked.nodes.loc[5, "y"] = 0.001
    kept = navis.downsample_neuron(kinked, 0, method="rdp").nodes.node_id.values
    assert set(kept) == {0, 4, 5, 6, 10}


@pytest.mark.parametrize("method", SHAPE_METHODS)
def test_tolerance_is_a_distance_in_neuron_units(neuron, method):
    """Both shape methods read the same number as the same distance.

    `vw`'s underlying threshold is an *area*, but navis squares the tolerance
    for it so that `method` can be swapped without also rescaling the number -
    and so that a unit string means one thing rather than two. Example neurons
    are in 8nm voxels, so 125 voxels is a micron.
    """
    by_number = navis.downsample_neuron(neuron, 125, method=method)
    by_string = navis.downsample_neuron(neuron, "1 micron", method=method)
    assert by_number.n_nodes == by_string.n_nodes

    # Both methods land in the same ballpark for the same tolerance - not the
    # same node count (they differ by design), but the same order of thinning.
    other = "vw" if method == "rdp" else "rdp"
    assert 0.5 < by_number.n_nodes / navis.downsample_neuron(
        neuron, 125, method=other
    ).n_nodes < 2


@pytest.mark.parametrize("method", SHAPE_METHODS)
@pytest.mark.parametrize("tol", [-1, float("inf"), float("nan")])
def test_bad_tolerance_raises(neuron, method, tol):
    """Must be a `ValueError` here - fastcore aborts a worker thread on these."""
    with pytest.raises(ValueError):
        navis.downsample_neuron(neuron, tol, method=method)


@pytest.mark.parametrize("method", SHAPE_METHODS)
def test_tolerance_below_one_is_allowed(neuron, method):
    """A factor below 1 is nonsense; a tolerance below 1 is an ordinary request."""
    ds = navis.downsample_neuron(neuron, 0.5, method=method)
    assert ds.n_nodes <= neuron.n_nodes


# =========================================================================== #
# `method` now means something for every neuron type, so it has to be checked
# =========================================================================== #
@pytest.mark.parametrize("method", SHAPE_METHODS + ["nonsense"])
def test_shape_methods_are_skeleton_only(neuron, method):
    """Silently ignoring these would hand back something and call it shape-aware."""
    dp = navis.make_dotprops(neuron, k=5)
    with pytest.raises(ValueError, match="Unknown"):
        navis.downsample_neuron(dp, 5, method=method)


@pytest.mark.parametrize("method", ["uniform", "fps", "decimate", "nonsense"])
def test_dotprops_methods_are_dotprops_only(neuron, method):
    with pytest.raises(ValueError, match="Unknown"):
        navis.downsample_neuron(neuron, 5, method=method)


@pytest.mark.parametrize("method", ["simple", "uniform", "fps", "decimate"])
def test_dotprops_methods_all_work(neuron, method):
    """The other half of the above: what a Dotprops *does* understand."""
    dp = navis.make_dotprops(neuron, k=5)
    assert navis.downsample_neuron(dp, 5, method=method).n_points < dp.n_points


@pytest.mark.parametrize("method", SHAPE_METHODS + ["uniform", "nonsense"])
def test_method_is_not_checked_for_types_we_cannot_downsample(method):
    """An unsupported *type* must not be reported as a bad `method`.

    Listing the methods a `str` does not have would imply it had the others.
    Which error it *does* raise is not this test's business - only that it is
    not a complaint about `method`.
    """
    with pytest.raises(Exception) as exc:
        navis.downsample_neuron("not a neuron", 5, method=method)
    assert "Unknown (down-)sampling method" not in str(exc.value)


# =========================================================================== #
# Degenerate skeletons
# =========================================================================== #
@pytest.mark.parametrize("method,amount", [("simple", 5), ("rdp", 1), ("vw", 1)])
@pytest.mark.parametrize("n_nodes", [0, 1, 2])
def test_tiny_skeletons_survive(method, amount, n_nodes):
    """Nothing to thin, but also nothing to trip over."""
    ds = navis.downsample_neuron(line(n_nodes), amount, method=method)
    assert ds.n_nodes == n_nodes


# =========================================================================== #
# Aligned data
# =========================================================================== #
@pytest.mark.parametrize("method,amount", ONE_PER_METHOD)
def test_attached_data_carries_onto_the_right_nodes(neuron, method, amount):
    """Survivors *are* the old nodes, so anything aligned to them comes along."""
    n = neuron.copy()
    n.attach("carried", np.arange(n.n_nodes), axis="nodes", on_rebuild="carry")

    ds = navis.downsample_neuron(n, amount, method=method)

    # `carried[i] == i` on the original, so the carried value must be the
    # surviving node's row index in the *original* table.
    expected = pd.Index(n.nodes.node_id.values).get_indexer(ds.nodes.node_id.values)
    assert np.array_equal(ds.carried, expected)


@pytest.mark.parametrize("method,amount", ALL_SETTINGS)
def test_downsampling_shortens_the_neuron(neuron, method, amount):
    """Survivors keep their coordinates, so replacement edges cut corners.

    Documented in `downsample_neuron`'s Notes: cable length is *not* preserved.
    Pinned here so the docs and the behaviour cannot drift apart.
    """
    ds = navis.downsample_neuron(neuron, amount, method=method)
    assert ds.cable_length < neuron.cable_length
    assert ds.cable_length > neuron.cable_length * 0.85
