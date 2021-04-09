from itertools import permutations
import sys
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np

epsilon = sys.float_info.epsilon
cpu_count = max(1, (os.cpu_count() or 2) - 1)


def ensure_inf_bounds(numbers, right_one=True):
    lst = list(numbers)
    if lst[0] <= 0:
        lst[0] = -np.inf
    else:
        lst.insert(0, -np.inf)

    if right_one and lst[-1] >= 1:
        lst[-1] = np.inf
    elif lst[-1] != np.inf:
        lst.append(np.inf)

    return np.array(lst)


def chunksize(it_len, cpu_count, min_chunk=50):
    return max(min_chunk, int(it_len / (cpu_count * 4)))


class ScoreMatrixBuilder:
    def __init__(
        self,
        dotprops,
        matching_sets,
        nonmatching=None,
        alpha=False,
        seed=1991,
    ):
        """Class for building a score matrix for NBLAST.

        dist_bins and dot_bins must additionally be given using the set_* or calc_* methods.

        Parameters
        ----------
        dotprops : dict or list of Dotprops
            An indexable sequence of all neurons which will be used as the training set,
            as Dotprops objects.
        matching_sets : list of sets of indices into dotprops
            Sets of neurons, as indices into dotprops, which should be considered matches.
        nonmatching : set of indices into dotprops, optional
            Set of neurons, as indices into dotprops, which should not be considered matches.
            If not given, all dotprops will be used
            (on the assumption that matches are a small subset of possible pairs).
        alpha : bool, optional
            Whether to multiply dot product by alpha (local colinearity)
            when calculating dist-dots.
        seed : int, optional
            Non-matching pairs are drawn at random using this seed, by default 1991
        """
        self.rng = np.random.default_rng(seed)
        self.dotprops = dotprops
        self.matching_sets = matching_sets
        self.alpha = alpha
        self._nonmatching = nonmatching
        self.dist_bins = None
        self.dot_bins = None

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

    def set_dist_boundaries(self, boundaries):
        """Set distance bins to use for score matrix by their internal boundaries.

        If the lowest boundary is <=0, it will be changed to -inf.
        If the lowest boundary is >0, -inf will be prepended.

        If the highest boundary is not inf, it will be appended.

        The first and last values are set to -inf and inf respectively,
        so that the bins explicitly cover the entire domain,
        even though negatives are not possible, in principle.

        Parameters
        ----------
        bins : list of float
            First and last values are effectively ignored, as above.

        Returns
        -------
        np.ndarray of float
            The modified bin boundaries.
        """
        self.dist_bins = ensure_inf_bounds(boundaries)
        return self.dist_bins

    def calc_dist_bins(self, n_bins, base, min_exp, max_exp):
        """Calculate distance boundaries in a logarithmic sequence.

        base**min_exp will be the upper bound of the lowest bin;
        base**max_exp will be the lower bound of the highest bin.

        n_bins - 1 values are spread evenly between min_exp and max_exp (inclusive).
        These are transformed into boundaries by base**values.

        Parameters
        ----------
        n_bins : int
            Number of bins (i.e. number of boundaries - 1)
        base : float
            Value to be raised to some power to create the boundaries.
        min_exp : float
            Exponent of the lowest boundary (actual boundary will be replaced by -inf)
        max_exp : float
            Exponent of the highest boundary (the lower bound of the inf-containing bin)

        Returns
        -------
        np.ndarray of float
            The modified bin boundaries.
        """
        # n_bins - 1 because inf outer boundaries are added
        return self.set_dist_boundaries(
            np.logspace(min_exp, max_exp, n_bins - 1, base=base)
        )

    def set_dot_boundaries(self, boundaries):
        """Set dot product bins to use for score matrix by their internal boundaries.

        Dot products, even normalised by alpha values,
        should be in the range 0,1.
        However, due to float imprecision, they sometimes aren't,
        so the lowest bound is set to -inf and highest bound set to inf
        just in case.

        Parameters
        ----------
        bins : list of float
            First and last values are effectively ignored, as above.

        Returns
        -------
        np.ndarray of float
            The modified bin boundaries.
        """
        self.dot_bins = ensure_inf_bounds(boundaries, True)
        return self.dot_bins

    def calc_dot_bins(self, n_bins):
        """Calculate dot product bins in a linear sequence between 0 and 1.

        Internally, 0 and 1 will be replaced with -inf and inf respectively.

        Parameters
        ----------
        n_bins : int
            Number of bins (i.e. number of boundaries - 1).

        Returns
        -------
        np.ndarray of float
            The modified bin boundaries.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        return self.set_dot_boundaries(bins[1:-1])

    def _yield_matching_pairs(self):
        for ms in self.matching_sets:
            for q, t in permutations(ms, 2):
                if q != t:
                    yield q, t

    def _yield_nonmatching_pairs(self):
        for q, t in permutations(self.nonmatching):
            if q != t:
                yield q, t

    def _query_to_dist_dot_idxs(self, q_idx, t_idx, counts=None):
        q = self.dotprops[q_idx]
        response = q.dist_dots(self.dotprops[t_idx], use_alpha=self.alpha)
        if self.alpha:
            dists, dots, alphas = response
            dots *= alphas
        else:
            dists, dots = response

        dist_idxs = np.digitize(dists, self.dist_bins) - 1
        dot_idxs = np.digitize(dots, self.dot_bins) - 1
        if counts is None:
            counts = np.zeros(
                (len(self.dist_bins), len(self.dot_bins)),
                dtype=int,
            )
        for dist_idx, dot_idx in zip(dist_idxs, dot_idxs):
            counts[dist_idx, dot_idx] += 1
        return counts

    def _counts_array(self, idx_pairs, threads=None):
        counts = np.zeros(
            (len(self.dist_bins), len(self.dot_bins)),
            dtype=int,
        )
        if threads is None or threads == 0 and cpu_count == 1:
            for q_idx, t_idx in idx_pairs:
                counts = self._query_to_dist_dot_idxs(q_idx, t_idx, counts)
            return counts

        threads = threads or cpu_count

        with ProcessPoolExecutor(threads) as exe:
            for these_counts in exe.map(
                self._query_to_dist_dot_idxs,
                idx_pairs[:, 0],
                idx_pairs[:, 1],
                chunksize=chunksize(len(idx_pairs), threads)
            ):
                counts += these_counts

        return counts

    def build(self, threads=None):
        """Build the score matrix.

        All non-identical neuron pairs within all matching sets are selected,
        and distdots calculated for those pairs.
        Then, the minimum number of non-matching pairs are randomly drawn
        so that at least as many distdots can be calculated for non-matching
        pairs.

        In each bin of the 2D score matrix, the log2 odds ratio of a distdot
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
        if self.dot_bins is None or self.dist_bins is None:
            raise ValueError("dot_bins and dist_bins must be set or calculated")

        matching_pairs = set(self._yield_matching_pairs())
        # need to know the eventual distdot count
        # so we know how many non-matching pairs to draw
        q_idx_count = Counter(p[0] for p in matching_pairs)
        n_matching_dist_dots = sum(
            len(self.dotprops[q_idx]) * n_reps for q_idx, n_reps in q_idx_count.items()
        )

        # pre-calculating which pairs we're going to use,
        # rather than drawing them as we need them,
        # means that we can parallelise the later step more effectively.
        # Slowdowns here are basically meaningless
        # because of how long distdot calculation will take
        all_nonmatching_pairs = list(self._yield_nonmatching_pairs())
        nonmatching_pairs = []
        n_nonmatching_dist_dots = 0
        while n_nonmatching_dist_dots < n_matching_dist_dots:
            idx = self.rng.integers(0, len(all_nonmatching_pairs) + 1)
            nonmatching_pairs.append(all_nonmatching_pairs.pop(idx))
            n_nonmatching_dist_dots += len(self.dotprops[nonmatching_pairs[-1][0]])

        match_counts = self._counts_array(matching_pairs, threads)
        nonmatch_counts = self._counts_array(nonmatching_pairs, threads)

        # account for there being different total numbers of match/nonmatch dist dots
        matching_factor = nonmatch_counts.sum() / match_counts.sum()

        cells = np.log2(
            (match_counts * matching_factor + epsilon) / (nonmatch_counts + epsilon)
        )

        index = [
            f"({left},{right}]"
            for left, right in zip(
                [-np.inf] + list(self.dist_bins), list(self.dist_bins) + [np.inf]
            )
        ]
        columns = [
            f"({left},{right}]"
            for left, right in zip(
                [-np.inf] + list(self.dot_bins), list(self.dot_bins) + [np.inf]
            )
        ]

        return pd.DataFrame(cells, index, columns)
