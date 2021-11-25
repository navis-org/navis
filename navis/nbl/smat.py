from __future__ import annotations
from itertools import permutations
import sys
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator, Sequence, Callable, List, Iterable, Any, Tuple
import logging
from pathlib import Path
from functools import lru_cache
from copy import deepcopy
import operator

import numpy as np
import pandas as pd

from ..core.neurons import Dotprops

logger = logging.getLogger(__name__)

DEFAULT_SEED = 1991

epsilon = sys.float_info.epsilon
cpu_count = max(1, (os.cpu_count() or 2) - 1)
IMPLICIT_INTERVAL = "[)"

fp = Path(__file__).resolve().parent
smat_path = fp / 'score_mats'


def chunksize(it_len, cpu_count, min_chunk=50):
    return max(min_chunk, int(it_len / (cpu_count * 4)))


def wrap_bounds(arr: np.ndarray, left: float = -np.inf, right: float = np.inf):
    """Ensures that boundaries cover -inf to inf.
    If the left or rightmost values lie within the interval ``(left, right)``,
    ``-inf`` or ``inf`` will be prepended or appended respectively.
    Otherwise, the left and right values will be set to
    ``-inf`` and ``inf`` respectively.
    Parameters
    ----------
    arr : array-like of floats
        Array of boundaries
    left : float, optional
        If the first value is greater than this, prepend -inf;
        otherwise replace that value with -inf.
        Defaults to -inf if None.
    right : float, optional
        If the last value is less than this, append inf;
        otherwise replace that value with inf.
        Defaults to inf if None.
    Returns
    -------
    np.ndarray of floats
        Boundaries from -inf to inf
    Raises
    ------
    ValueError
        If values of ``arr`` are not monotonically increasing.
    """
    if left is None:
        left = -np.inf
    if right is None:
        right = np.inf

    if not np.all(np.diff(arr) > 0):
        raise ValueError("Boundaries are not monotonically increasing")

    items = list(arr)
    if items[0] <= left:
        items[0] = -np.inf
    else:
        items.insert(0, -np.inf)

    if items[-1] >= right:
        items[-1] = np.inf
    else:
        items.append(np.inf)

    return np.array(items)


def yield_not_same(pairs: Iterable[Tuple[Any, Any]]) -> Iterator[Tuple[Any, Any]]:
    for a, b in pairs:
        if a != b:
            yield a, b


class LookupNdBuilder:
    def __init__(
        self,
        dotprops,
        matching_sets,
        boundaries: Sequence[Sequence[float]],
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
        boundaries : sequence of array-likes of floats
            List of lists, where the inner lists are monotonically increasing
            from -inf to inf.
            The length of the outer list is the dimensionality of the lookup table.
            The inner lists are the boundaries of bins for that dimension,
            i.e. an inner list of N items describes N-1 bins.
            If an inner list is not inf-bounded,
            -inf and inf will be prepended and appended.
            See the ``wrap_bounds`` convenience function.
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
        self.boundaries = [wrap_bounds(b) for b in boundaries]

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
        shape = [len(b) - 1 for b in self.boundaries]
        return np.zeros(shape, int)

    def _query_to_idxs(self, q_idx, t_idx, counts=None):
        response = self.match_fn(self.dotprops[q_idx], self.dotprops[t_idx])
        idxs = [np.digitize(r, b) - 1 for r, b in zip(response, self.boundaries)]

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

    def _build(self, threads) -> Tuple[List[np.ndarray], np.ndarray]:
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

        return self.boundaries, cells

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
        return LookupNd(*self._build(threads))


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
        dist_boundaries,
        dot_boundaries,
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
            [wrap_bounds(dist_boundaries, 0), wrap_bounds(dot_boundaries, 0, 1)],
            match_fn,
            nonmatching,
            seed,
        )

    def build(self, threads=None) -> Lookup2d:
        return Lookup2d(*self._build(threads))


class LookupNd:
    def __init__(self, boundaries: List[np.ndarray], cells: np.ndarray):
        if [len(b) - 1 for b in boundaries] != list(cells.shape):
            raise ValueError("boundaries and cells have inconsistent bin counts")
        self.boundaries = boundaries
        self.cells = cells

    def __call__(self, *args):
        if len(args) != len(self.boundaries):
            raise TypeError(
                f"Lookup takes {len(self.boundaries)} arguments but {len(args)} were given"
            )

        idxs = tuple(
            np.digitize(r, b) - 1 for r, b in zip(args, self.boundaries)
        )
        out = self.cells[idxs]
        return out


def format_boundaries(arr):
    return [f"[{lower},{upper})" for lower, upper in zip(arr[:-1], arr[1:])]


def parse_boundary(item, strict=False):
    explicit_interval = item[0] + item[-1]
    if strict and item[0] + item[-1] != IMPLICIT_INTERVAL:
        raise ValueError(
            f"Enclosing characters {explicit_interval} "
            f"do not match implicit interval specifiers {IMPLICIT_INTERVAL}"
        )
    return tuple(float(i) for i in item[1:-1].split(","))


def parse_boundaries(items, strict=False):
    # declaring upper first and then checking for None
    # pleases the type checker, and handles the case where
    # len(items) == 0
    upper = None
    for item in items:
        lower, upper = parse_boundary(item, strict)
        yield lower
    if upper is None:
        return
    yield upper


class Lookup2d(LookupNd):
    """Convenience class inheriting from LookupNd for the common 2D case.
    Provides IO with pandas DataFrames.
    """

    def __init__(self, boundaries: List[np.ndarray], cells: np.ndarray):
        if len(boundaries) != 2:
            raise ValueError("boundaries must be of length 2; cells must be 2D")
        super().__init__(boundaries, cells)

    def to_dataframe(self) -> pd.DataFrame:
        # numpy.digitize includes left, excludes right, i.e. "[left,right)"
        return pd.DataFrame(
            self.cells,
            format_boundaries(self.boundaries[0]),
            format_boundaries(self.boundaries[1]),
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, strict=False):
        f"""Parse score matrix from a dataframe with string index and column labels.
        Expects the index and column labels to specify an interval
        like ``f"[{{lower}},{{upper}})"``.
        Will replace the lowermost and uppermost bound with -inf and inf
        if they are not already.
        Parameters
        ----------
        strict : bool, optional
            If falsey (default), ignores parentheses and brackets,
            effectively normalising to
            ``{IMPLICIT_INTERVAL[0]}lower,upper{IMPLICIT_INTERVAL[1]}``
            as an implementation detail.
            Otherwise, raises a ValueError if parens/brackets
            do not match the implementation.
        """
        boundaries = []
        for arr in (df.index, df.columns):
            b = np.array(list(parse_boundaries(arr, strict)))
            b[0] = -np.inf
            b[-1] = np.inf
            boundaries.append(b)

        return cls(boundaries, df.to_numpy())


@lru_cache(maxsize=None)
def _smat_fcwb(alpha=False):
    # cached private function defers construction
    # until needed (speeding startup),
    # but avoids repeated reads (speeding later uses)
    fname = ("smat_fcwb.csv", "smat_alpha_fcwb.csv")[alpha]
    fpath = smat_path / fname

    return Lookup2d.from_dataframe(
        pd.read_csv(fpath, index_col=0)
    )


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
NBLAST score functions take 2 floats or numpy arrays of floats of length N
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
        If ``smat`` is None, choose whether to use the FCWB matrices
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
    elif smat == 'auto':
        smat = smat_fcwb(alpha)

    if isinstance(smat, (str, os.PathLike)):
        smat = pd.read_csv(smat, index_col=0)

    if isinstance(smat, pd.DataFrame):
        smat = Lookup2d.from_dataframe(smat)

    if not callable(smat):
        raise ValueError("smat should be a callable, a path, a pandas.DataFrame, or 'auto'")

    check_score_fn(smat)

    return smat
