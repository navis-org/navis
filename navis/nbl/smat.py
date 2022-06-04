from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import permutations
import sys
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import (
    Generic,
    Hashable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Callable,
    List,
    Iterable,
    Any,
    Tuple,
    TypeVar,
    Union,
)
import logging
from pathlib import Path
from functools import lru_cache
from copy import deepcopy
import operator
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from ..core.neurons import Dotprops

logger = logging.getLogger(__name__)

DEFAULT_SEED = 1991

epsilon = sys.float_info.epsilon
cpu_count = max(1, (os.cpu_count() or 2) - 1)

fp = Path(__file__).resolve().parent
smat_path = fp / "score_mats"


def chunksize(it_len, cpu_count, min_chunk=50):
    return max(min_chunk, int(it_len / (cpu_count * 4)))


def yield_not_same(pairs: Iterable[Tuple[Any, Any]]) -> Iterator[Tuple[Any, Any]]:
    for a, b in pairs:
        if a != b:
            yield a, b


def concat_results(results: Iterable[List[np.ndarray]]) -> List[np.ndarray]:
    intermediate = defaultdict(list)
    for result_lst in results:
        for idx, array in enumerate(result_lst):
            intermediate[idx].append(array)

    return [np.concatenate(arrs) for arrs in intermediate.values()]


DotpropKey = Hashable


class LookupNdBuilder:
    def __init__(
        self,
        dotprops: Union[List[Dotprops], Mapping[DotpropKey, Dotprops]],
        matching_lists: List[List[DotpropKey]],
        match_fn: Callable[[Dotprops, Dotprops], List[np.ndarray]],
        nonmatching_list: Optional[List[DotpropKey]] = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        f"""Class for building an N-dimensional score lookup for NBLAST.

        Once instantiated, the axes of the lookup table must be defined.
        Call ``.with_digitizers()`` to manually define them,
        or ``.with_bin_counts()`` to learn them from the matched-pair data.

        Then call ``.build()`` to build the lookup table.

        Parameters
        ----------

        dotprops : dict or list of Dotprops
            An indexable, consistently-ordered sequence of all neurons
            which will be used as the training set, as Dotprops objects.
        matching_sets : list of lists of index into dotprops
            Lists of neurons, as indices into ``dotprops``, which should be considered matches.
        match_fn : Callable[[Dotprops, Dotprops], List[np.ndarray[float]]]
            Function taking 2 arguments,
            both instances of ``navis.core.neurons.Dotprops``,
            and returning a list of 1D ``numpy.ndarray``s of floats.
            The length of the list must be the same as the length of ``boundaries``.
            The length of the ``array``s must be the same
            as the number of points in the first argument.
            This function returns values describing the quality of
            point matches from a query to a target neuron.
        nonmatching : list of index into dotprops, optional
            List of neurons, as indices into ``dotprops``,
            which should not be considered matches.
            If not given, all ``dotprops`` will be used
            (on the assumption that matches are a small subset of possible pairs).
        seed : int, optional
            Non-matching pairs are drawn at random using this seed,
            by default {DEFAULT_SEED}
        """
        self.dotprops = dotprops
        self.matching_lists = matching_lists
        self._nonmatching_list = nonmatching_list
        self.match_fn = match_fn

        self.digitizers: Optional[List[Digitizer]] = None
        self.bin_counts: Optional[List[int]] = None

        self.seed = seed
        self._ndim: Optional[int] = None

    @property
    def ndim(self) -> int:
        if self._ndim is None:
            idx1, idx2 = self._dotprop_keys()[:2]
            self._ndim = len(self._query(idx1, idx2))
            self._query.cache_clear()
        return self._ndim

    def with_digitizers(self, digitizers: List[Digitizer]):
        """Specify the axes of the output lookup table directly.

        Parameters
        ----------
        digitizers : List[Digitizer]

        Returns
        -------
        self
            For chaining convenience.
        """
        if len(digitizers) != self.ndim:
            raise ValueError(
                f"Match function returns {self.ndim} values "
                f"but provided {len(digitizers)} digitizers"
            )

        self.digitizers = digitizers
        self.bin_counts = None
        return self

    def with_bin_counts(self, bin_counts: List[int]):
        """Specify the number of bins on each axis of the output lookup table.

        The bin boundaries will be determined by evenly partitioning the data
        from the matched pairs into quantiles, in each dimension.

        Parameters
        ----------
        bin_counts : List[int]

        Returns
        -------
        self
            For chaining convenience.
        """
        if len(bin_counts) != self.ndim:
            raise ValueError(
                f"Match function returns {self.ndim} values "
                f"but provided {len(bin_counts)} bin counts"
            )

        self.bin_counts = bin_counts
        self.digitizers = None
        return self

    def _dotprop_keys(self) -> Sequence[DotpropKey]:
        """Get all indices into dotprops instance member"""
        try:
            return self.dotprops.keys()
        except AttributeError:
            return range(len(self.dotprops))

    @property
    def nonmatching(self) -> List[DotpropKey]:
        """Indices of nonmatching set of neurons"""
        if self._nonmatching_list is None:
            return list(self._dotprop_keys())
        return self._nonmatching_list

    def _yield_matching_pairs(self) -> Iterator[Tuple[DotpropKey, DotpropKey]]:
        """Yield all index pairs within all matching pairs"""
        for ms in self.matching_lists:
            yield from yield_not_same(permutations(ms, 2))

    def _yield_nonmatching_pairs(self) -> Iterator[Tuple[DotpropKey, DotpropKey]]:
        """Yield all index pairs within nonmatching list"""
        # todo: this could be much better, use meshgrid or shuffle index arrays
        return yield_not_same(permutations(self.nonmatching, 2))

    def _empty_counts(self) -> np.ndarray:
        """Create an empty array in which to store counts; shape determined by digitizer sizes."""
        shape = [len(b) for b in self.digitizers]
        return np.zeros(shape, int)

    @lru_cache(None)
    def _query(self, q_idx, t_idx) -> List[np.ndarray]:
        """Get the results of applying the match function to dotprops specified by indices"""
        return self.match_fn(self.dotprops[q_idx], self.dotprops[t_idx])

    def _query_many(self, idx_pairs, threads=None) -> Iterator[List[np.ndarray]]:
        """Yield results from querying many pairs of dotprop indices"""
        if threads is None or threads == 0 and cpu_count == 1:
            for q_idx, t_idx in idx_pairs:
                yield self._query(q_idx, t_idx)
            return

        threads = threads or cpu_count
        idx_pairs = np.asarray(idx_pairs)
        chunks = chunksize(len(idx_pairs), threads)

        with ProcessPoolExecutor(threads) as exe:
            yield from exe.map(
                self._query, idx_pairs[:, 0], idx_pairs[:, 1], chunksize=chunks
            )

    def _query_to_idxs(self, q_idx, t_idx, counts=None):
        """Produce a digitized counts array from a given query-target pair"""
        return self._count_results(self._query(q_idx, t_idx), counts)

    def _count_results(self, results: List[np.ndarray], counts=None):
        """Convert raw match function ouput into a digitized counts array.

        Requires digitizers.
        """
        idxs = [dig(r) for dig, r in zip(self.digitizers, results)]

        if counts is None:
            counts = self._empty_counts()

        for idx in zip(*idxs):
            counts[idx] += 1

        return counts

    def _counts_array(self, idx_pairs, threads=None, cache=False):
        """Convert index pairs into a digitized counts array.

        Requires digitizers.
        """
        counts = self._empty_counts()
        if threads is None or threads == 0 and cpu_count == 1:
            for q_idx, t_idx in idx_pairs:
                counts = self._query_to_idxs(q_idx, t_idx, counts)
            return counts

        threads = threads or cpu_count
        idx_pairs = np.asarray(idx_pairs, dtype=int)
        chunks = chunksize(len(idx_pairs), threads)

        # because digitizing is not necessarily free,
        # keep this parallelisation separate to that in _query_many
        with ProcessPoolExecutor(threads) as exe:
            for these_counts in exe.map(
                self._query_to_idxs,
                idx_pairs[:, 0],
                idx_pairs[:, 1],
                chunksize=chunks,
            ):
                counts += these_counts
                if not cache:
                    self._query.cache_clear

        return counts

    def _pick_nonmatching_pairs(self, n_matching_qual_vals):
        """Using the seeded RNG, pick which nonmatching pairs to use."""
        # pre-calculating which pairs we're going to use,
        # rather than drawing them as we need them,
        # means that we can parallelise the later step more effectively.
        # Slowdowns here are practically meaningless
        # because of how long distdot calculation will take
        all_nonmatching_pairs = list(self._yield_nonmatching_pairs())
        nonmatching_pairs = []
        n_nonmatching_qual_vals = 0
        rng = np.random.default_rng(self.seed)
        while n_nonmatching_qual_vals < n_matching_qual_vals:
            idx = rng.integers(0, len(all_nonmatching_pairs))
            nonmatching_pair = all_nonmatching_pairs.pop(idx)
            nonmatching_pairs.append(nonmatching_pair)
            n_nonmatching_qual_vals += len(self.dotprops[nonmatching_pair[0]])

        return nonmatching_pairs

    def _get_pairs(self):
        matching_pairs = list(set(self._yield_matching_pairs()))
        # need to know the eventual distdot count
        # so we know how many non-matching pairs to draw
        q_idx_count = Counter(p[0] for p in matching_pairs)
        n_matching_qual_vals = sum(
            len(self.dotprops[q_idx]) * n_reps for q_idx, n_reps in q_idx_count.items()
        )

        nonmatching_pairs = self._pick_nonmatching_pairs(n_matching_qual_vals)
        return matching_pairs, nonmatching_pairs

    def _build(self, threads, cache=False) -> Tuple[List[Digitizer], np.ndarray]:
        # Asking for more threads than available CPUs seems to crash on Github
        # actions
        if threads and threads >= cpu_count:
            threads = cpu_count

        if self.digitizers is None and self.bin_counts is None:
            raise ValueError(
                "Builder needs either digitizers or bin_counts; see with_* methods"
            )

        matching_pairs, nonmatching_pairs = self._get_pairs()

        if self.digitizers:
            match_counts = self._counts_array(matching_pairs, threads, cache)
        else:
            match_results = concat_results(self._query_many(matching_pairs, threads))
            self.digitizers = [
                Digitizer.from_data(data, nbins)
                for data, nbins in zip(match_results, self.bin_counts)
            ]
            if not cache:
                self._query.cache_clear()
            match_counts = self._count_results(match_results)

        nonmatch_counts = self._counts_array(nonmatching_pairs, threads)

        # account for there being different total numbers of match/nonmatch dist dots
        matching_factor = nonmatch_counts.sum() / match_counts.sum()
        if np.any(match_counts + nonmatch_counts == 0):
            logger.warning("Some lookup cells have no data in them")

        cells = np.log2(
            (match_counts * matching_factor + epsilon) / (nonmatch_counts + epsilon)
        )

        return self.digitizers, cells

    def build(self, threads=None, cache=False) -> LookupNd:
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
            Will be clipped at number of available cores - 1.

        Returns
        -------
        LookupNd
        """
        dig, cells = self._build(threads, cache)
        return LookupNd(dig, cells)


def dist_dot(q: Dotprops, t: Dotprops):
    return list(q.dist_dots(t))


def dist_dot_alpha(q: Dotprops, t: Dotprops):
    dist, dot, alpha = q.dist_dots(t, alpha=True)
    return [dist, dot * np.sqrt(alpha)]


class LookupDistDotBuilder(LookupNdBuilder):
    def __init__(
        self,
        dotprops: Union[List[Dotprops], Mapping[DotpropKey, Dotprops]],
        matching_lists: List[List[DotpropKey]],
        nonmatching_list: Optional[List[DotpropKey]] = None,
        use_alpha: bool = False,
        seed: int = DEFAULT_SEED,
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
        matching_lists : list of lists of indices into dotprops
            List of neurons, as indices into ``dotprops``, which should be considered matches.
        nonmatching_list : list of indices into dotprops, optional
            List of neurons, as indices into ``dotprops``,
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
            matching_lists,
            match_fn,
            nonmatching_list,
            seed,
        )
        self._ndim = 2

    def build(self, threads=None, cache=False) -> Lookup2d:
        (dig0, dig1), cells = self._build(threads, cache)
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


T = TypeVar("T")


class LookupAxis(ABC, Generic[T]):
    """Class converting some data into a linear index."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of bins represented by this instance."""
        pass

    @abstractmethod
    def __call__(self, value: Union[T, Sequence[T]]) -> Union[int, Sequence[int]]:
        """Convert some data into a linear index.

        Parameters
        ----------
        value : Union[T, Sequence[T]]
            Value to convert into an index

        Returns
        -------
        Union[int, Sequence[int]]
            If a scalar was given, return a scalar; otherwise, a numpy array of ints.
        """
        pass


class SimpleLookup(LookupAxis[Hashable]):
    def __init__(self, items: List[Hashable]):
        """Look up in a list of items and return their index.

        Parameters
        ----------
        items : List[Hashable]
            The item's position in the list is the index which will be returned.

        Raises
        ------
        ValueError
            items are non-unique.
        """
        self.items = {item: idx for idx, item in enumerate(items)}
        if len(self.items) != len(items):
            raise ValueError("Items are not unique")

    def __len__(self) -> int:
        return len(self.items)

    def __call__(self, value: Union[Hashable, Sequence[Hashable]]) -> Union[int, Sequence[int]]:
        if np.isscalar(value):
            return self.items[value]
        else:
            return np.array([self.items[v] for v in value], int)


class Digitizer(LookupAxis[float]):
    def __init__(
        self,
        boundaries: Sequence[float],
        clip: Tuple[bool, bool] = (True, True),
        right=False,
    ):
        """Class converting continuous values into discrete indices.

        Parameters
        ----------
        boundaries : Sequence[float]
            N boundaries specifying N-1 bins.
            Must be monotonically increasing.
        clip : Tuple[bool, bool], optional
            Whether to set the bottom and top boundaries to -infinity and infinity respectively,
            effectively clipping incoming values: by default (True, True).
            False means "add a new bin for out-of-range values".
        right : bool, optional
            Whether bins should include their right (rather than left) boundary,
            by default False
        """
        self.right = right

        boundaries = list(boundaries)
        self._min = -math.inf
        if clip[0]:
            self._min = boundaries[0]
            boundaries[0] = -math.inf
        elif boundaries[0] != -math.inf:
            self._min = -math.inf
            boundaries.insert(0, -math.inf)

        self._max = math.inf
        if clip[1]:
            self._max = boundaries[-1]
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

        b = self.boundaries.copy()
        b[0] = self._min
        b[-1] = self._max
        return [
            f"{lb}{lower},{upper}{rb}"
            for lower, upper in zip(b[:-1], b[1:])
        ]

    @classmethod
    def from_strings(cls, interval_strs: Sequence[str]):
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
        for item in interval_strs:
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

        Input values will be clipped to fit within the given interval.

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
    def from_geom(cls, lowest_upper: float, highest_lower: float, nbins: int, right=False):
        """Choose digitizer boundaries in a geometric sequence.

        Additional bins will be added above and below the given values.

        Parameters
        ----------
        lowest_upper : float
            Upper bound of the lowest bin.
            The lower bound of the lowest bin is often 0, which cannot be represented in a nontrivial geometric sequence.
        highest_lower : float
            Lower bound of the highest bin.
        nbins : int
            Number of bins
        right : bool, optional
            Whether bins should include their right (rather than left) boundary,
            by default False

        Returns
        -------
        Digitizer
        """
        arr = np.geomspace(lowest_upper, highest_lower, nbins - 1, True)
        return cls(arr, clip=(False, False), right=right)

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
        return self.right == other.right and np.allclose(
            self.boundaries, other.boundaries
        )


class LookupNd:
    def __init__(self, axes: List[LookupAxis], cells: np.ndarray):
        if [len(b) for b in axes] != list(cells.shape):
            raise ValueError("boundaries and cells have inconsistent bin counts")
        self.axes = axes
        self.cells = cells

    def __call__(self, *args):
        if len(args) != len(self.axes):
            raise TypeError(
                f"Lookup takes {len(self.axes)} arguments but {len(args)} were given"
            )

        idxs = tuple(d(arg) for d, arg in zip(self.axes, args))
        out = self.cells[idxs]
        return out


class Lookup2d(LookupNd):
    """Convenience class inheriting from LookupNd for the common 2D float case.
    Provides IO with pandas DataFrames.
    """

    def __init__(self, axis0: Digitizer, axis1: Digitizer, cells: np.ndarray):
        """2D lookup table for convert NBLAST matches to scores.

        Commonly read from a ``pandas.DataFrame``
        or trained on data using a ``LookupDistDotBuilder``.

        Parameters
        ----------
        digitizer0 : Digitizer
            How to convert continuous values into an index for the first axis.
        digitizer1 : Digitizer
            How to convert continuous values into an index for the second axis.
        cells : np.ndarray
            Values to look up in the table.
        """
        super().__init__([axis0, axis1], cells)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the lookup table into a ``pandas.DataFrame``.

        From there, it can be shared, saved, and so on.

        The index and column labels describe the intervals represented by that axis.

        Returns
        -------
        pd.DataFrame
        """
        return pd.DataFrame(
            self.cells,
            self.axes[0].to_strings(),
            self.axes[1].to_strings(),
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
""".strip().replace(
    "\n", " "
)


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
