"""Behavioural contracts for `navis.smooth_skeleton`.

Both kernels live in navis-fastcore, so these check the properties navis
promises rather than the arithmetic: what moves, what is pinned, what the
topology looks like afterwards, and which of `window`/`sigma` was asked for.

The numeric ones use toy neurons small enough to work out by hand - a spike on
a straight wire is the whole of what a smoother does, and its expected output
is a mean of three numbers.
"""

import numpy as np
import pandas as pd

import pytest

import navis


@pytest.fixture
def neuron():
    return navis.example_neurons(1, kind="skeleton")


def toy_neuron(coords, parents, **kwargs):
    """Build a Skeleton from explicit coordinates and parents."""
    nodes = pd.DataFrame(np.asarray(coords, dtype=np.float32), columns=["x", "y", "z"])
    nodes["node_id"] = np.arange(len(nodes))
    nodes["parent_id"] = np.asarray(parents, dtype=np.int64)
    nodes["radius"] = kwargs.pop("radius", 1.0)
    return navis.Skeleton(nodes, **kwargs)


def spike(n_nodes=7, at=3, height=3.0):
    """Straight line along x with a single node pushed off it in y."""
    coords = np.zeros((n_nodes, 3))
    coords[:, 0] = np.arange(n_nodes)
    coords[at, 1] = height
    return toy_neuron(coords, list(range(-1, n_nodes - 1)))


def sine(step, span=40.0, wavelength=10.0, amp=1.0):
    """A sine wave along x, sampled every `step`. Same curve at any `step`."""
    x = np.arange(0, span + step / 2, step)
    coords = np.zeros((len(x), 3))
    coords[:, 0] = x
    coords[:, 1] = amp * np.sin(x * 2 * np.pi / wavelength)
    return toy_neuron(coords, list(range(-1, len(x) - 1)))


def fork():
    """A Y: a stem of three nodes, then two branches of three each."""
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0], [4, 2, 0], [5, 3, 0],
         [3, -1, 0], [4, -2, 0], [5, -3, 0]],
        dtype=float,
    )
    # Node 2 is the branch point; it carries a kink in y that a smoother would
    # happily average away were it not pinned.
    coords[2, 1] = 2.0
    return toy_neuron(coords, [-1, 0, 1, 2, 3, 4, 2, 6, 7])


def ends_of(x):
    """Roots, branch points and leafs - the nodes both kernels pin."""
    return np.concatenate(
        [
            np.atleast_1d(x.root),
            x.branch_points.node_id.values,
            x.leafs.node_id.values,
        ]
    )


def positions(x, node_ids):
    return x.nodes.set_index("node_id").loc[node_ids, ["x", "y", "z"]].values


# The toy neurons below are built at unit spacing; the example neuron is in
# nanometres, where a `sigma` of 2 would be a no-op.
KERNELS = [pytest.param(dict(window=5), id="window"),
           pytest.param(dict(sigma=2.0), id="sigma")]
NEURON_KERNELS = [pytest.param(dict(window=5), id="window"),
                  pytest.param(dict(sigma=2000.0), id="sigma")]


# --------------------------------------------------------------- what is pinned


@pytest.mark.parametrize("kernel", KERNELS)
def test_segment_ends_do_not_move(kernel):
    """Roots, branch points and leafs are pinned: a branch point that drifted
    would drag its three neurites apart."""
    x = fork()
    smoothed = navis.smooth_skeleton(x, **kernel)

    ends = ends_of(x)
    assert len(ends) == 4  # root, one branch point, two leafs
    assert np.allclose(positions(smoothed, ends), positions(x, ends))


@pytest.mark.parametrize("kernel", NEURON_KERNELS)
def test_ends_are_pinned_on_a_real_neuron(neuron, kernel):
    smoothed = navis.smooth_skeleton(neuron, **kernel)

    ends = ends_of(neuron)
    assert np.allclose(positions(smoothed, ends), positions(neuron, ends))


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_straight_line_is_its_own_smoothing(kernel):
    """Nothing to take out: an unbranched straight wire comes back unchanged.

    The Gaussian earns this one by reflecting about the endpoints rather than
    truncating there - a one-sided window would pull the second node inwards.
    """
    x = spike(height=0.0)
    smoothed = navis.smooth_skeleton(x, **kernel)

    assert np.allclose(smoothed.nodes[["x", "y", "z"]], x.nodes[["x", "y", "z"]])


# -------------------------------------------------------------- what is smoothed


def test_window_is_a_centred_mean():
    """`window=3` makes each interior node the mean of itself and its two
    neighbours; the spike is spread over three nodes, not dragged distally."""
    x = spike(n_nodes=7, at=3, height=3.0)
    smoothed = navis.smooth_skeleton(x, window=3)

    # Nodes 2, 3 and 4 each see the spike once out of three
    assert np.allclose(smoothed.nodes.y.values, [0, 0, 1, 1, 1, 0, 0])
    # ... and x, which was already evenly spaced, is untouched
    assert np.allclose(smoothed.nodes.x.values, np.arange(7))


def test_the_window_is_symmetric_so_even_values_round_down():
    """The window is centred, so it can only ever hold an odd number of nodes."""
    x = spike()
    assert np.allclose(
        navis.smooth_skeleton(x, window=4).nodes.y.values,
        navis.smooth_skeleton(x, window=3).nodes.y.values,
    )


@pytest.mark.parametrize("window", [0, 1, 2])
def test_a_window_of_one_node_is_a_no_op(window):
    """`2` included: rounded down to the odd value below, it is a window of 1."""
    x = spike()
    smoothed = navis.smooth_skeleton(x, window=window)

    assert np.allclose(smoothed.nodes[["x", "y", "z"]], x.nodes[["x", "y", "z"]])


def test_a_wider_kernel_smooths_more():
    x = spike(n_nodes=21, at=10, height=5.0)

    peaks = [
        navis.smooth_skeleton(x, window=w).nodes.y.values[10] for w in (3, 5, 9)
    ]
    assert peaks[0] > peaks[1] > peaks[2]

    peaks = [
        navis.smooth_skeleton(x, sigma=s).nodes.y.values[10] for s in (1.0, 2.0, 4.0)
    ]
    assert peaks[0] > peaks[1] > peaks[2]


def test_sigma_is_a_distance_not_a_node_count():
    """The point of the Gaussian: how much smoothing a given `sigma` does is a
    property of the *curve*, not of how densely it happens to be sampled."""
    # The same sine wave at 1x and 4x the node density. Every 4th node of the
    # fine one sits exactly on a node of the coarse one.
    coarse, fine = sine(step=1.0), sine(step=0.25)

    a = navis.smooth_skeleton(coarse, sigma=1.5).nodes.y.values
    b = navis.smooth_skeleton(fine, sigma=1.5).nodes.y.values[::4]
    assert np.allclose(a, b, atol=1e-2)
    assert np.abs(a).max() < 0.8  # and it did flatten the wave

    # A node-count window, by contrast, reaches a quarter as far along the
    # curve once there are four times the nodes, so it barely smooths at all
    a = navis.smooth_skeleton(coarse, window=5).nodes.y.values
    b = navis.smooth_skeleton(fine, window=5).nodes.y.values[::4]
    assert np.abs(b).max() > 1.4 * np.abs(a).max()


# ------------------------------------------------------------------- topology


@pytest.mark.parametrize("kernel", NEURON_KERNELS)
def test_smoothing_moves_nodes_but_keeps_the_skeleton(neuron, kernel):
    """Only values move: every node keeps its ID and its parent, so anything
    attached to a node is still attached to it afterwards."""
    smoothed = navis.smooth_skeleton(neuron, **kernel)

    assert smoothed.n_nodes == neuron.n_nodes
    assert np.array_equal(smoothed.nodes.node_id.values, neuron.nodes.node_id.values)
    assert np.array_equal(
        smoothed.nodes.parent_id.values, neuron.nodes.parent_id.values
    )
    assert len(smoothed.connectors) == len(neuron.connectors)
    # ... and it did do something
    assert not np.allclose(smoothed.nodes[["x", "y", "z"]], neuron.nodes[["x", "y", "z"]])


@pytest.mark.parametrize("kernel", NEURON_KERNELS)
def test_attached_data_is_carried(neuron, kernel):
    """`smooth_skeleton` is a move, not a rebuild - see the "attaching data"
    tutorial's table of what carries."""
    neuron = neuron.copy()
    neuron.attach("mydata", np.arange(neuron.n_nodes), axis="nodes")

    smoothed = navis.smooth_skeleton(neuron, **kernel)

    assert np.array_equal(smoothed.mydata, neuron.mydata)


@pytest.mark.parametrize("n_nodes", [0, 1, 2])
@pytest.mark.parametrize("kernel", KERNELS)
def test_degenerate_skeletons(n_nodes, kernel):
    """Nothing here has an interior node to smooth; all of it survives intact."""
    coords = np.zeros((n_nodes, 3))
    coords[:, 0] = np.arange(n_nodes)
    x = toy_neuron(coords, list(range(-1, n_nodes - 1)))

    smoothed = navis.smooth_skeleton(x, **kernel)

    assert smoothed.n_nodes == n_nodes
    assert np.allclose(smoothed.nodes[["x", "y", "z"]], x.nodes[["x", "y", "z"]])


@pytest.mark.parametrize("kernel", NEURON_KERNELS)
def test_inplace(neuron, kernel):
    neuron = neuron.copy()
    before = neuron.nodes[["x", "y", "z"]].values.copy()

    out = navis.smooth_skeleton(neuron, inplace=True, **kernel)

    assert out is neuron
    assert not np.allclose(neuron.nodes[["x", "y", "z"]].values, before)


@pytest.mark.parametrize("kernel", NEURON_KERNELS)
def test_not_inplace_leaves_the_original_alone(neuron, kernel):
    neuron = neuron.copy()
    before = neuron.nodes[["x", "y", "z"]].values.copy()

    out = navis.smooth_skeleton(neuron, inplace=False, **kernel)

    assert out is not neuron
    assert np.array_equal(neuron.nodes[["x", "y", "z"]].values, before)


# ------------------------------------------------------------------ to_smooth


def test_smoothing_a_radius_leaves_the_geometry_alone(neuron):
    smoothed = navis.smooth_skeleton(neuron, to_smooth="radius", window=5)

    assert np.allclose(smoothed.nodes[["x", "y", "z"]], neuron.nodes[["x", "y", "z"]])
    assert not np.allclose(smoothed.nodes.radius, neuron.nodes.radius)


def test_smoothing_the_geometry_leaves_a_radius_alone(neuron):
    smoothed = navis.smooth_skeleton(neuron, window=5)

    assert np.array_equal(smoothed.nodes.radius, neuron.nodes.radius)


@pytest.mark.parametrize("kernel", NEURON_KERNELS)
def test_columns_smooth_independently(neuron, kernel):
    """Stacking columns into one call is only an optimisation - each column is
    still smoothed on its own."""
    together = navis.smooth_skeleton(neuron, to_smooth=["x", "y", "radius"], **kernel)

    for col in ("x", "y", "radius"):
        alone = navis.smooth_skeleton(neuron, to_smooth=col, **kernel)
        assert np.allclose(together.nodes[col], alone.nodes[col])


def test_the_gaussian_kernel_is_measured_over_the_geometry():
    """Smoothing a radius with `sigma` weighs neighbours by *distance along the
    neurite*, so stretching the neuron out changes the answer - whereas the
    radii themselves are the same numbers either way."""
    coords = np.zeros((7, 3))
    coords[:, 0] = np.arange(7)
    radius = [1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0]

    tight = toy_neuron(coords, list(range(-1, 6)), radius=radius)
    stretched = toy_neuron(coords * 4, list(range(-1, 6)), radius=radius)

    a = navis.smooth_skeleton(tight, sigma=2.0, to_smooth="radius")
    b = navis.smooth_skeleton(stretched, sigma=2.0, to_smooth="radius")

    # Four times the spacing, so 2.0 reaches a quarter as far along the wire
    assert b.nodes.radius.values[3] > a.nodes.radius.values[3]


@pytest.mark.parametrize("kernel", NEURON_KERNELS)
def test_dtypes_are_preserved(neuron, kernel):
    neuron = neuron.copy()
    neuron.nodes["count"] = np.arange(neuron.n_nodes, dtype=np.int32)

    smoothed = navis.smooth_skeleton(neuron, to_smooth=["x", "count"], **kernel)

    assert smoothed.nodes.x.dtype == neuron.nodes.x.dtype
    assert smoothed.nodes["count"].dtype == neuron.nodes["count"].dtype


# --------------------------------------------------------------------- errors


def test_window_and_sigma_are_mutually_exclusive(neuron):
    with pytest.raises(ValueError, match="one or the other"):
        navis.smooth_skeleton(neuron, window=5, sigma=2.0)


@pytest.mark.parametrize("window", [-1, 2.7])
def test_window_must_be_a_non_negative_integer(neuron, window):
    """Without this, `-1` surfaces as an `OverflowError` from the Rust boundary
    and `2.7` is silently truncated to a *narrower* window than `3`."""
    with pytest.raises(ValueError, match="non-negative integer"):
        navis.smooth_skeleton(neuron, window=window)


def test_missing_column(neuron):
    with pytest.raises(ValueError, match="not found in node table"):
        navis.smooth_skeleton(neuron, to_smooth="nonexistent")


def test_non_numeric_column(neuron):
    neuron = neuron.copy()
    neuron.nodes["label"] = "a"

    with pytest.raises(ValueError, match="numeric"):
        navis.smooth_skeleton(neuron, to_smooth="label")


def test_only_skeletons(neuron):
    with pytest.raises(TypeError, match="Can only process Skeletons"):
        navis.smooth_skeleton(navis.example_neurons(1, kind="mesh"))
