from __future__ import annotations
from itertools import permutations
import sys
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator, Sequence, Callable, List, Iterable, Any, Tuple, Union
import logging
from pathlib import Path
from functools import lru_cache
from copy import deepcopy
import operator
import math

import numpy as np
import pandas as pd

from ..core.neurons import Dotprops

logger = logging.getLogger(__name__)

DEFAULT_SEED = 1991

epsilon = sys.float_info.epsilon
cpu_count = max(1, (os.cpu_count() or 2) - 1)
IMPLICIT_INTERVAL = "[)"

fp = Path(__file__).resolve().parent
smat_path = fp / "score_mats"


def chunksize(it_len, cpu_count, min_chunk=50):
    return max(min_chunk, int(it_len / (cpu_count * 4)))


def yield_not_same(pairs: Iterable[Tuple[Any, Any]]) -> Iterator[Tuple[Any, Any]]:
    for a, b in pairs:
        if a != b:
            yield a, b


class LookupNdBuilder:
    def __init__(
        self,
        dotprops,
        matching_sets,
        digitizers: Sequence[Digitizer],
        match_fn: Callable[[Dotprops, Dotprops], List[np.ndarray]],
        nonmatching=None,
        seed=DEFAULT_SEED,
    ) -> None:
        f"""Class for building an N-dimensional score lookup for NBLAST.
        Parameters
        ----------
        dotprops : dict or list of Dotprops
            An indexable sequence of all neurons which will be used as the training set,
            as Dotprops objects.
        matching_sets : list of sets of int
            Sets of neurons, as indices into ``dotprops``, which should be considered matches.
        digitizers : sequence of Digitizers
            The length is the dimensionality of the lookup table.
        match_fn : Callable[[Dotprops, Dotprops], List[np.ndarray[float]]]
            Function taking 2 arguments,
            both instances of ``navis.core.neurons.Dotprops``,
            and returning a list of 1D ``numpy.ndarray``s of floats.
            The length of the list must be the same as the length of ``boundaries``.
            The length of the ``array``s must be the same
            as the number of points in the first argument.
            This function returns values describing the quality of
            point matches from a query to a target neuron.
        nonmatching : set of int, optional
            Set of neurons, as indices into ``dotprops``,
            which should not be considered matches.
            If not given, all ``dotprops`` will be used
            (on the assumption that matches are a small subset of possible pairs).
        seed : int, optional
            Non-matching pairs are drawn at random using this seed,
            by default {DEFAULT_SEED}
        """
        self.dotprops = dotprops
        self.matching_sets = matching_sets
        self._nonmatching = nonmatching
        self.match_fn = match_fn
        self.digitizers = list(digitizers)

        self.rng = np.random.default_rng(seed)

    def _dotprop_keys(self):
        try:
            return self.dotprops.keys()
        except AttributeError:
            return range(len(self.dotprops))

    @property
    def nonmatching(self):
        """Indices of nonmatching set of neurons"""
        if self._nonmatching is None:
            return set(self._dotprop_keys())
        return self._nonmatching

    def _yield_matching_pairs(self):
        for ms in self.matching_sets:
            yield from yield_not_same(permutations(ms, 2))

    def _yield_nonmatching_pairs(self):
        # todo: this could be much better, use meshgrid or shuffle index arrays
        return yield_not_same(permutations(self.nonmatching, 2))

    def _empty_counts(self):
        shape = [len(b) for b in self.digitizers]
        return np.zeros(shape, int)

    def _query_to_idxs(self, q_idx, t_idx, counts=None):
        response = self.match_fn(self.dotprops[q_idx], self.dotprops[t_idx])
        idxs = [dig(r) for dig, r in zip(self.digitizers, response)]

        if counts is None:
            counts = self._empty_counts()

        for idx in zip(*idxs):
            counts[idx] += 1

        return counts

    def _counts_array(self, idx_pairs, threads=None):
        counts = self._empty_counts()
        if threads is None or threads == 0 and cpu_count == 1:
            for q_idx, t_idx in idx_pairs:
                counts = self._query_to_idxs(q_idx, t_idx, counts)
            return counts

        threads = threads or cpu_count
        idx_pairs = np.asarray(idx_pairs, dtype=int)
        chunks = chunksize(len(idx_pairs), threads)

        with ProcessPoolExecutor(threads) as exe:
            for these_counts in exe.map(
                self._query_to_idxs,
                idx_pairs[:, 0],
                idx_pairs[:, 1],
                chunksize=chunks,
            ):
                counts += these_counts

        return counts

    def _pick_nonmatching_pairs(self, n_matching_qual_vals):
        # pre-calculating which pairs we're going to use,
        # rather than drawing them as we need them,
        # means that we can parallelise the later step more effectively.
        # Slowdowns here are practically meaningless
        # because of how long distdot calculation will take
        all_nonmatching_pairs = list(self._yield_nonmatching_pairs())
        nonmatching_pairs = []
        n_nonmatching_qual_vals = 0
        while n_nonmatching_qual_vals < n_matching_qual_vals:
            idx = self.rng.integers(0, len(all_nonmatching_pairs))
            nonmatching_pair = all_nonmatching_pairs.pop(idx)
            nonmatching_pairs.append(nonmatching_pair)
            n_nonmatching_qual_vals += len(self.dotprops[nonmatching_pair[0]])

        return nonmatching_pairs

    def _build(self, threads) -> Tuple[List[Digitizer], np.ndarray]:
        matching_pairs = list(set(self._yield_matching_pairs()))
        # need to know the eventual distdot count
        # so we know how many non-matching pairs to draw
        q_idx_count = Counter(p[0] for p in matching_pairs)
        n_matching_qual_vals = sum(
            len(self.dotprops[q_idx]) * n_reps for q_idx, n_reps in q_idx_count.items()
        )

        nonmatching_pairs = self._pick_nonmatching_pairs(n_matching_qual_vals)

        match_counts = self._counts_array(matching_pairs, threads)
        nonmatch_counts = self._counts_array(nonmatching_pairs, threads)

        # account for there being different total numbers of match/nonmatch dist dots
        matching_factor = nonmatch_counts.sum() / match_counts.sum()
        if np.any(match_counts + nonmatch_counts == 0):
            logger.warning("Some lookup cells have no data in them")

        cells = np.log2(
            (match_counts * matching_factor + epsilon) / (nonmatch_counts + epsilon)
        )

        return self.digitizers, cells

    def build(self, threads=None) -> LookupNd:
        """Build the score matrix.
        All non-identical neuron pairs within all matching sets are selected,
        and distdots calculated for those pairs.
        Then, the minimum number of non-matching pairs are randomly drawn
        so that at least as many distdots can be calculated for non-matching
        pairs.
        In each bin of the score matrix, the log2 odds ratio of a distdot
        in that bin belonging to a match vs. non-match is calculated.
        Parameters
        ----------
        threads : int, optional
            If None, act in serial.
            If 0, use cpu_count - 1.
            Otherwise, use the given value.
        Returns
        -------
        pd.DataFrame
            Suitable for passing to navis.nbl.ScoringFunction
        Raises
        ------
        ValueError
            If dist_bins or dot_bins are not set.
        """
        dig, cells = self._build(threads)
        return LookupNd(dig, cells)


def dist_dot(q: Dotprops, t: Dotprops):
    return list(q.dist_dots(t))


def dist_dot_alpha(q: Dotprops, t: Dotprops):
    dist, dot, alpha = q.dist_dots(t, alpha=True)
    return [dist, dot * np.sqrt(alpha)]


class LookupDistDotBuilder(LookupNdBuilder):
    def __init__(
        self,
        dotprops,
        matching_sets,
        dist_digitizer: Digitizer,
        dot_digitizer: Digitizer,
        nonmatching=None,
        use_alpha=False,
        seed=DEFAULT_SEED,
    ):
        f"""Class for building a 2-dimensional score lookup for NBLAST.
        The scores are
        1. The distances between best-matching points
        2. The dot products of direction vectors around those points,
            optionally scaled by the colinearity ``alpha``.
        Parameters
        ----------
        dotprops : dict or list of Dotprops
            An indexable sequence of all neurons which will be used as the training set,
            as Dotprops objects.
        matching_sets : list of sets of int
            Sets of neurons, as indices into ``dotprops``, which should be considered matches.
        dist_boundaries : array-like of floats
            Monotonically increasing boundaries between distance bins
            (i.e. N values makes N-1 bins).
            If the first value > 0, -inf will be prepended,
            otherwise the first value will be replaced with -inf.
            inf will be appended unless the last value is already inf.
            Consider using ``numpy.logspace`` or ``numpy.geomspace``
            to generate the inner boundaries of the bins.
        dot_boundaries : array-like of floats
            Monotonically increasing boundaries between
            (possibly alpha-scaled) dot product bins
            (i.e. N values makes N-1 bins).
            If the first value > 0, -inf will be prepended,
            otherwise the first value will be replaced with -inf.
            If the last value < 1, inf will be appended,
            otherwise the last value will be replaced with inf.
            Consider using ``numpy.linspace`` between 0 and 1.
        nonmatching : set of int, optional
            Set of neurons, as indices into ``dotprops``,
            which should not be considered matches.
            If not given, all ``dotprops`` will be used
            (on the assumption that matches are a small subset of possible pairs).
        use_alpha : bool, optional
            If true, multiply the dot product by the geometric mean
            of the matched points' alpha values
            (i.e. ``sqrt(alpha1 * alpha2)``).
        seed : int, optional
            Non-matching pairs are drawn at random using this seed,
            by default {DEFAULT_SEED}
        """
        match_fn = dist_dot_alpha if use_alpha else dist_dot
        super().__init__(
            dotprops,
            matching_sets,
            [dist_digitizer, dot_digitizer],
            match_fn,
            nonmatching,
            seed,
        )

    def build(self, threads=None) -> Lookup2d:
        (dig0, dig1), cells = self._build(threads)
        return Lookup2d(dig0, dig1, cells)


def is_monotonically_increasing(lst):
    for prev_idx, item in enumerate(lst[1:]):
        if item <= lst[prev_idx]:
            return False
    return True


def parse_boundary(item: str):
    explicit_interval = item[0] + item[-1]
    if explicit_interval == "[)":
        right = False
    elif explicit_interval == "(]":
        right = True
    else:
        raise ValueError(
            f"Enclosing characters '{explicit_interval}' do not match a half-open interval"
        )
    return tuple(float(i) for i in item[1:-1].split(",")), right


class Digitizer:
    def __init__(
        self,
        boundaries: Sequence[float],
        clip: Tuple[bool, bool] = (True, True),
        right=False,
    ):
        self.right = right

        boundaries = list(boundaries)
        self._min = boundaries[0]
        if clip[0]:
            boundaries[0] = -math.inf
        elif boundaries[0] != -math.inf:
            boundaries.insert(0, -math.inf)

        self._max = boundaries[-1]
        if clip[1]:
            boundaries[-1] = math.inf
        elif boundaries[-1] != math.inf:
            boundaries.append(math.inf)

        if not is_monotonically_increasing(boundaries):
            raise ValueError("Boundaries are not monotonically increasing")

        self.boundaries = np.asarray(boundaries)

    def __len__(self):
        return len(self.boundaries) - 1

    def __call__(self, value: float):
        # searchsorted is marginally faster than digitize as it skips monotonicity checks
        return (
            np.searchsorted(
                self.boundaries, value, side="left" if self.right else "right"
            )
            - 1
        )

    def to_strings(self) -> List[str]:
        if self.right:
            lb = "("
            rb = "]"
        else:
            lb = "["
            rb = ")"

        return [
            f"{lb}{lower},{upper}{rb}"
            for lower, upper in zip(self.boundaries[:-1], self.boundaries[1:])
        ]

    @classmethod
    def from_strings(
        cls, bound_strs: Sequence[str]
    ):
        """Set digitizer boundaries based on a sequence of interval expressions.

        e.g. ``["(0, 1]", "(1, 5]", "(5, 10]"]``

        The lowermost and uppermost boundaries are converted to -infinity and infinity respectively.

        Parameters
        ----------
        bound_strs : Sequence[str]
            Strings representing intervals, which must abut and have open/closed boundaries
            specified by brackets.

        Returns
        -------
        Digitizer
        """
        bounds: List[float] = []
        last_upper = None
        last_right = None
        for item in bound_strs:
            (lower, upper), right = parse_boundary(item)
            bounds.append(float(lower))

            if last_right is not None:
                if right != last_right:
                    raise ValueError("Inconsistent half-open interval")
            else:
                last_right = right

            if last_upper is not None:
                if lower != last_upper:
                    raise ValueError("Half-open intervals do not abut")

            last_upper = upper

        bounds.append(float(last_upper))
        return cls(bounds, right=last_right)

    @classmethod
    def from_linear(cls, lower: float, upper: float, nbins: int, right=False):
        """Choose digitizer boundaries spaced linearly between two values.

        Parameters
        ----------
        lower : float
            Lowest value
        upper : float
            Highest value
        nbins : int
            Number of bins
        right : bool, optional
            Whether bins should include their right (rather than left) boundary,
            by default False

        Returns
        -------
        Digitizer
        """
        arr = np.linspace(lower, upper, nbins + 1, endpoint=True)
        return cls(arr, right=right)

    @classmethod
    def from_geom(cls, lowest_upper: float, upper: float, nbins: int, right=False):
        """Choose digitizer boundaries in a geometric sequence.

        Parameters
        ----------
        lowest_upper : float
            Upper bound of the lowest bin.
            The lower bound of the lowest bin is often 0, which cannot be represented in a nontrivial geometric sequence.
        upper : float
            Upper bound of the highest bin.
            This will be replaced by +inf, but the value is used for setting the rest of the sequence.
        nbins : int
            Number of bins
        right : bool, optional
            Whether bins should include their right (rather than left) boundary,
            by default False

        Returns
        -------
        Digitizer
        """
        # Because geom can't start at 0, instead start at the upper bound
        # of the lowest bin, then prepend -inf.
        # Because the extra bin is added, do not need to add 1 to nbins in geomspace.
        arr = np.geomspace(lowest_upper, upper, nbins, True)
        return cls(arr, clip=(False, True), right=right)

    @classmethod
    def from_data(cls, data: Sequence[float], nbins: int, right=False):
        """Choose digitizer boundaries to evenly partition the given values.

        Parameters
        ----------
        data : Sequence[float]
            Data which should be evenly partitioned by the resulting digitizer.
        nbins : int
            Number of bins
        right : bool, optional
            Whether bins should include their right (rather than left) boundary,
            by default False

        Returns
        -------
        Digitizer
        """
        arr = np.quantile(data, np.linspace(0, 1, nbins + 1, True))
        return cls(arr, right=right)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Digitizer):
            return NotImplemented
        return self.right == other.right and np.allclose(self.boundaries, other.boundaries)


class LookupNd:
    def __init__(self, digitizers: List[Digitizer], cells: np.ndarray):
        if [len(b) for b in digitizers] != list(cells.shape):
            raise ValueError("boundaries and cells have inconsistent bin counts")
        self.digitizers = digitizers
        self.cells = cells

    def __call__(self, *args):
        if len(args) != len(self.digitizers):
            raise TypeError(
                f"Lookup takes {len(self.digitizers)} arguments but {len(args)} were given"
            )

        idxs = tuple(d(arg) for d, arg in zip(self.digitizers, args))
        out = self.cells[idxs]
        return out


class Lookup2d(LookupNd):
    """Convenience class inheriting from LookupNd for the common 2D case.
    Provides IO with pandas DataFrames.
    """

    def __init__(self, digitizer0: Digitizer, digitizer1: Digitizer, cells: np.ndarray):
        super().__init__([digitizer0, digitizer1], cells)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.cells,
            self.digitizers[0].to_strings(),
            self.digitizers[1].to_strings(),
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        f"""Parse score matrix from a dataframe with string index and column labels.

        Expects the index and column labels to specify an interval
        like ``f"[{{lower}},{{upper}})"``.
        Will replace the lowermost and uppermost bound with -inf and inf
        if they are not already.
        """
        return cls(
            Digitizer.from_strings(df.index),
            Digitizer.from_strings(df.columns),
            df.to_numpy(),
        )


@lru_cache(maxsize=None)
def _smat_fcwb(alpha=False):
    # cached private function defers construction
    # until needed (speeding startup),
    # but avoids repeated reads (speeding later uses)
    fname = ("smat_fcwb.csv", "smat_alpha_fcwb.csv")[alpha]
    fpath = smat_path / fname

    return Lookup2d.from_dataframe(pd.read_csv(fpath, index_col=0))


def smat_fcwb(alpha=False):
    # deepcopied so that mutations do not propagate to cache
    return deepcopy(_smat_fcwb(alpha))


def check_score_fn(fn: Callable, nargs=2, scalar=True, array=True):
    """Checks functionally that the callable can be used as a score function.
    Parameters
    ----------
    nargs : optional int, default 2
        How many positional arguments the score function should have.
    scalar : optional bool, default True
        Check that the function can be used on ``nargs`` scalars.
    array : optional bool, default True
        Check that the function can be used on ``nargs`` 1D ``numpy.ndarray``s.
    Raises
    ------
    ValueError
        If the score function is not appropriate.
    """
    if scalar:
        scalars = [0.5] * nargs
        if not isinstance(fn(*scalars), float):
            raise ValueError("smat does not take 2 floats and return a float")

    if array:
        test_arr = np.array([0.5] * 3)
        arrs = [test_arr] * nargs
        try:
            out = fn(*arrs)
        except Exception as e:
            raise ValueError(f"Failed to use smat with numpy arrays: {e}")

        if out.shape != test_arr.shape:
            raise ValueError(
                f"smat produced inconsistent shape: input {test_arr.shape}; output {out.shape}"
            )


SCORE_FN_DESCR = """
NBLAST score functions take 2 floats or N-length numpy arrays of floats
(for matched dotprop points/tangents, distance and dot product;
the latter possibly scaled by the geometric mean of the alpha colinearity values)
and returns a float or N-length numpy array of floats.
""".strip().replace("\n", " ")


def parse_score_fn(smat, alpha=False):
    f"""Interpret ``smat`` as a score function.
    Primarily for backwards compatibility.
    {SCORE_FN_DESCR}
    Parameters
    ----------
    smat : None | "auto" | str | os.PathLike | pandas.DataFrame | Callable[[float, float], float]
        If ``None``, use ``operator.mul``.
        If ``"auto"``, use ``navis.nbl.smat.smat_fcwb(alpha)``.
        If a dataframe, use ``navis.nbl.smat.Lookup2d.from_dataframe(smat)``.
        If another string or path-like, load from CSV in a dataframe and uses as above.
        Also checks the signature of the callable.
        Raises an error, probably a ValueError, if it can't be interpreted.
    alpha : optional bool, default False
        If ``smat`` is ``"auto"``, choose whether to use the FCWB matrices
        with or without alpha.
    Returns
    -------
    Callable
    Raises
    ------
    ValueError
        If score function cannot be interpreted.
    """
    if smat is None:
        smat = operator.mul
    elif smat == "auto":
        smat = smat_fcwb(alpha)

    if isinstance(smat, (str, os.PathLike)):
        smat = pd.read_csv(smat, index_col=0)

    if isinstance(smat, pd.DataFrame):
        smat = Lookup2d.from_dataframe(smat)

    if not callable(smat):
        raise ValueError(
            "smat should be a callable, a path, a pandas.DataFrame, or 'auto'"
        )

    check_score_fn(smat)

    return smat
