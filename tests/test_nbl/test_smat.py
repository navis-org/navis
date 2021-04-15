from navis.core.neurons import Dotprops
import pytest

import numpy as np

from navis.nbl.smat import (
    wrap_bounds, LookupNd, Lookup2d, LookupDistDotBuilder
)


SMALLEST_DIM_SIZE = 3
SEED = 1991


@pytest.mark.parametrize(
    ["arr", "left", "right", "expected"],
    [
        ([1, 2], -np.inf, np.inf, [-np.inf, 1, 2, np.inf]),
        ([-np.inf, 1, 2, np.inf], -np.inf, np.inf, [-np.inf, 1, 2, np.inf]),
        ([0, 1, 2, 3], 0, 3, [-np.inf, 1, 2, np.inf]),
    ],
)
def test_wrap_bounds(arr, left, right, expected):
    assert np.allclose(wrap_bounds(arr, left, right), expected)


def test_wrap_bounds_error():
    with pytest.raises(ValueError):
        wrap_bounds([1, 2, 1])


def lookup_args(ndim):
    f"""
    Create arguments for an ND lookup table.

    The first dimension is of size {SMALLEST_DIM_SIZE},
    and subsequent dimensions are 1 longer than the previous.
    The data in the cells are the sequence from 0
    to the size of the array.
    The boundaries are 0 to the length of the dimension,
    with the left and rightmost values replaced with -inf and inf respectively.

    Examples
    --------
    >>> lookup_args(2)
    (
        [
            array([-inf, 1, 2, inf]),
            array([-inf, 1, 2, 3, inf]),
        ],
        array([
            [ 0,  1,  2,  3 ],
            [ 4,  5,  6,  7 ],
            [ 8,  9, 10, 11 ],
        ]),
    )
    """
    shape = tuple(range(SMALLEST_DIM_SIZE, SMALLEST_DIM_SIZE + ndim))
    cells = np.arange(np.product(shape)).reshape(shape)
    boundaries = []
    for s in shape:
        b = np.arange(s + 1, dtype=float)
        b[0] = -np.inf
        b[-1] = np.inf
        boundaries.append(b)
    return boundaries, cells


def fmt_array(arg):
    if np.isscalar(arg):
        return str(arg)
    else:
        return "[" + ",".join(fmt_array(v) for v in arg) + "]"


@pytest.mark.parametrize(
    ["ndim"], [[1], [2], [3], [4], [5]], ids=lambda x: f"{x}D"
)
@pytest.mark.parametrize(
    ["arg"],
    [
        (-1000,),
        (0,),
        (1,),
        (1.5,),
        (2,),
        (1000,),
        ([-1000, 0, 1, 1.5, 2, 1000],),
    ],
    ids=fmt_array,
)
def test_lookupNd(ndim, arg):
    lookup = LookupNd(*lookup_args(ndim))

    args = [arg for _ in range(ndim)]
    expected_arr_idx = np.floor([
        np.clip(arg, 0, dim + SMALLEST_DIM_SIZE - 1) for dim in range(ndim)
    ]).astype(int)
    expected_val = np.ravel_multi_index(
        tuple(expected_arr_idx), lookup.cells.shape
    )

    response = lookup(*args)
    assert np.all(response == expected_val)


@pytest.mark.parametrize(["strict"], [(False,), (True,)])
def test_lookup2d_roundtrip(strict):
    lookup = Lookup2d(*lookup_args(2))
    df = lookup.to_dataframe()
    lookup2 = Lookup2d.from_dataframe(df, strict=strict)
    assert np.allclose(lookup.cells, lookup2.cells)
    for b1, b2 in zip(lookup.boundaries, lookup2.boundaries):
        assert np.allclose(b1, b2)


def test_calc_dist_boundaries(neurons):
    dotprops = [Dotprops(n.nodes[["x", "y", "z"]], k=5) for n in neurons]
    builder = LookupDistDotBuilder(dotprops, [])
    n_bins = 5
    bounds = builder.calc_dist_boundaries(n_bins)
    assert bounds.shape == (n_bins + 1,)
    assert bounds[0] == -np.inf
    assert bounds[-1] == np.inf


def test_calc_dot_boundaries(neurons):
    dotprops = [Dotprops(n.nodes[["x", "y", "z"]], k=5) for n in neurons]
    builder = LookupDistDotBuilder(dotprops, [])
    n_bins = 5
    bounds = builder.calc_dot_boundaries(n_bins)
    assert bounds.shape == (n_bins + 1,)
    assert bounds[0] == -np.inf
    assert bounds[-1] == np.inf


def prepare_lookupdistdotbuilder(neurons, alpha=False, k=5, with_bounds=False):
    dotprops = [Dotprops(n.nodes[["x", "y", "z"]], k) for n in neurons]
    n_orig = len(dotprops)

    # make jittered copies of these neurons
    rng = np.random.default_rng(SEED)
    jitter_sigma = 50
    matching_sets = []
    for idx, dp in enumerate(dotprops[:]):
        dotprops.append(
            Dotprops(
                dp.points + rng.normal(0, jitter_sigma, dp.points.shape), k
            )
        )
        # assign each neuron its jittered self as a match
        matching_sets.append({idx, idx + n_orig})

    # original neurons should all not match each other
    nonmatching = list(range(n_orig))

    builder = LookupDistDotBuilder(
        dotprops,
        matching_sets,
        nonmatching,
        alpha,
        seed=SEED + 1,
    )
    if with_bounds:
        n_bins = 5
        builder.calc_dist_boundaries(n_bins)
        builder.calc_dot_boundaries(n_bins)
    return builder


@pytest.mark.parametrize(["threads"], [(None,), (0,), (2,)])
@pytest.mark.parametrize(["alpha"], [(True,), (False,)])
def test_lookupdistdotbuilder_builds(neurons, threads, alpha):
    builder = prepare_lookupdistdotbuilder(neurons, alpha, with_bounds=True)
    lookup = builder.build(threads)
    # `pytest -rP` to see output
    print(lookup.to_dataframe())
