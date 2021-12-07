from navis import Dotprops
import pytest

import numpy as np

from navis.nbl.smat import (
    Digitizer, LookupNd, Lookup2d, LookupDistDotBuilder
)


SMALLEST_DIM_SIZE = 3
SEED = 1991


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
    digitizers = [Digitizer(np.arange(s + 1, dtype=float)) for s in shape]
    return digitizers, cells


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


def test_lookup2d_roundtrip():
    digs, cells = lookup_args(2)
    lookup = Lookup2d(*digs, cells=cells)
    df = lookup.to_dataframe()
    lookup2 = Lookup2d.from_dataframe(df)
    assert np.allclose(lookup.cells, lookup2.cells)
    for b1, b2 in zip(lookup.axes, lookup2.axes):
        assert b1 == b2


def prepare_lookupdistdotbuilder(neurons, alpha=False, k=5):
    k = 5
    dotprops = [Dotprops(n.nodes[["x", "y", "z"]], k) for n in neurons]
    n_orig = len(dotprops)

    # make jittered copies of these neurons
    rng = np.random.default_rng(SEED)
    jitter_sigma = 50
    matching_lists = []
    for idx, dp in enumerate(dotprops[:]):
        dotprops.append(
            Dotprops(
                dp.points + rng.normal(0, jitter_sigma, dp.points.shape), k
            )
        )
        # assign each neuron its jittered self as a match
        matching_lists.append([idx, idx + n_orig])

    # original neurons should all not match each other
    nonmatching = list(range(n_orig))

    # max distance between any 2 points in the data
    # for calculating dist boundaries
    max_dist = np.linalg.norm(
        np.ptp(
            np.concatenate([dp.points for dp in dotprops], axis=0), axis=0,
        )
    )

    return LookupDistDotBuilder(
        dotprops,
        matching_lists,
        nonmatching,
        alpha,
        seed=SEED + 1,
    ).with_digitizers([
        Digitizer.from_geom(10, max_dist, 5),
        Digitizer.from_linear(0, 1, 5),
    ])


@pytest.mark.parametrize(["alpha"], [(True,), (False,)])
@pytest.mark.parametrize(["threads"], [(0,), (2,), (None,)])
def test_lookupdistdotbuilder_builds(treeneuron_dfs, threads, alpha):
    builder = prepare_lookupdistdotbuilder(treeneuron_dfs, alpha)
    lookup = builder.build(threads)
    # `pytest -rP` to see output
    print(lookup.to_dataframe())
