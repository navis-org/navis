"""Tests for `navis.sholl_analysis`, in particular the handling of `center`.

`center` accepts several forms - the "centermass"/"root"/"soma" presets, a node ID or an
x/y/z coordinate. The dispatch used to resolve "centermass" into a coordinate array *before*
the remaining branches compared `center` against the string presets. `array == "soma"` is an
elementwise comparison, so the following `elif` raised "truth value of an array is ambiguous"
and the default arguments were unusable.

The same block only recognised node IDs via `isinstance(center, int)`, which is False for the
numpy integers that come out of `x.root` / `x.nodes.node_id`. Such a center silently skipped
the node -> coordinate lookup and was broadcast as a scalar into the distance computation,
i.e. distances were measured from (v, v, v) rather than from the node. That produced wrong
numbers without raising.

The ground truth used here is that all the ways of naming the *same* center must agree:
"root", the root's node ID (as Python int and as numpy int) and the root's coordinates must
all give identical results.
"""

import numpy as np
import pandas as pd
import pytest

import navis


COLUMNS = ["intersections", "cable_length", "branch_points"]


@pytest.fixture
def n():
    return navis.example_neurons(1, kind="skeleton")


def test_sholl_default_center(n):
    """`sholl_analysis(n)` with default center="centermass" must not raise."""
    res = navis.sholl_analysis(n)

    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == COLUMNS
    assert len(res) == 10  # default radii=10
    assert res.intersections.sum() > 0


@pytest.mark.parametrize(
    "center",
    ["centermass", "root", "soma", "node_id", "numpy_node_id", "coordinate"],
)
def test_sholl_center_forms(n, center):
    """Every documented form of `center` must work."""
    if center == "node_id":
        center = int(n.root[0])
    elif center == "numpy_node_id":
        center = n.root[0]  # numpy integer
    elif center == "coordinate":
        center = n.nodes[["x", "y", "z"]].values[0]

    res = navis.sholl_analysis(n, radii=5, center=center)

    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == COLUMNS
    assert len(res) == 5


def test_sholl_root_forms_agree(n):
    """Naming the root by preset, node ID or coordinate must give identical results.

    The numpy-integer case is the regression guard: it used to be broadcast as a scalar
    instead of being looked up as a node, silently returning different numbers.
    """
    root = n.root[0]
    root_xyz = n.nodes.set_index("node_id").loc[int(root), ["x", "y", "z"]].values

    by_preset = navis.sholl_analysis(n, radii=10, center="root")
    by_int = navis.sholl_analysis(n, radii=10, center=int(root))
    by_numpy_int = navis.sholl_analysis(n, radii=10, center=root)
    by_coord = navis.sholl_analysis(n, radii=10, center=root_xyz)

    for other in (by_int, by_numpy_int, by_coord):
        pd.testing.assert_frame_equal(by_preset, other)


def test_sholl_centermass_is_not_a_node(n):
    """The default center is the mean node position, not some node's position."""
    res = navis.sholl_analysis(n, radii=10, center="centermass")
    expected = navis.sholl_analysis(
        n, radii=10, center=n.nodes[["x", "y", "z"]].mean(axis=0).values
    )

    pd.testing.assert_frame_equal(res, expected)


@pytest.mark.parametrize("center", ["root", "soma", "node_id"])
def test_sholl_geodesic(n, center):
    """Geodesic distances work for any node-based center."""
    if center == "node_id":
        center = n.root[0]

    res = navis.sholl_analysis(n, radii=5, center=center, geodesic=True)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 5


@pytest.mark.parametrize(
    "center",
    ["centermass", "coordinate"],
    ids=["centermass", "coordinate"],
)
def test_sholl_geodesic_requires_node(n, center):
    """Geodesic distances are undefined for a center that is not on the arbor."""
    if center == "coordinate":
        center = [1000.0, 2000.0, 3000.0]

    with pytest.raises(ValueError):
        navis.sholl_analysis(n, radii=5, center=center, geodesic=True)


@pytest.mark.parametrize(
    "center",
    ["cenetrmass", "CENTERMASS", 99999999, [1.0, 2.0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
    ids=["typo", "wrong_case", "unknown_node_id", "too_few_coords", "not_1d"],
)
def test_sholl_invalid_center(n, center):
    """Bad input must raise a ValueError - not AttributeError or wrong numbers."""
    with pytest.raises(ValueError):
        navis.sholl_analysis(n, radii=5, center=center)


def test_sholl_numpy_radii(n):
    """`radii` must accept numpy integers as well as Python ints."""
    expected = navis.sholl_analysis(n, radii=10, center="root")
    res = navis.sholl_analysis(n, radii=np.int64(10), center="root")

    pd.testing.assert_frame_equal(res, expected)


def test_sholl_explicit_radii(n):
    """A list of radii is used as given (with 0 prepended)."""
    res = navis.sholl_analysis(n, radii=[1000, 2000, 3000], center="root")

    assert list(res.index) == [1000, 2000, 3000]


def test_sholl_neuronlist():
    """The @map_neuronlist decorator still maps over a NeuronList."""
    nl = navis.example_neurons(2, kind="skeleton")

    res = navis.sholl_analysis(nl, radii=5)

    assert len(res) == 2
    assert all(isinstance(r, pd.DataFrame) for r in res)


@pytest.fixture
def star():
    """Root at the origin with four straight 10-unit arms along +x/-x/+y/-y.

    Every sphere of radius r < 10 must cut each arm exactly once, so the correct
    number of intersections is 4 - regardless of whether distances are measured
    Euclidean or geodesic, which for this geometry are identical.
    """
    rows = [dict(node_id=0, parent_id=-1, x=0.0, y=0.0, z=0.0, radius=1)]
    nid = 1
    for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        parent = 0
        for step in range(1, 11):
            xyz = [0.0, 0.0, 0.0]
            xyz[axis] = float(sign * step)
            rows.append(
                dict(
                    node_id=nid,
                    parent_id=parent,
                    x=xyz[0],
                    y=xyz[1],
                    z=xyz[2],
                    radius=1,
                )
            )
            parent = nid
            nid += 1
    return navis.TreeNeuron(pd.DataFrame(rows), units="1 nm")


@pytest.mark.parametrize("geodesic", [False, True], ids=["euclidean", "geodesic"])
def test_sholl_intersections_count_both_directions(star, geodesic):
    """An edge intersects a sphere if exactly one of its ends is inside it.

    Regression guard: the count used to be `(dists <= r) & (pdists > r)`, i.e. only
    edges crossing *inwards*. For an arbor rooted at the center virtually every edge
    points outwards, so intersections were massively undercounted; with geodesic
    distances - where a child is always further out than its parent - the condition
    can never hold and the column was identically zero.
    """
    res = navis.sholl_analysis(
        star, radii=[2, 4, 6, 8], center="root", geodesic=geodesic
    )

    assert res.intersections.tolist() == [4, 4, 4, 4]


def test_sholl_intersections_nonzero_geodesic(n):
    """Geodesic Sholl analysis must actually find intersections on a real neuron."""
    res = navis.sholl_analysis(n, radii=10, center="soma", geodesic=True)

    assert res.intersections.sum() > 0
