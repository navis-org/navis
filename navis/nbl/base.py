#    This script is part of navis (http://www.github.com/navis-org/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

"""Module containing base classes for BLASTING."""

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Union, List

from .. import utils, config

INT_DTYPES = {16: np.int16, 32: np.int32, 64: np.int64, None: None,
              'single': np.int32, 'double': np.int64}
FLOAT_DTYPES = {16: np.float16, 32: np.float32, 64: np.float64, None: None,
                'single': np.float32, 'double': np.float64}

logger = config.logger

NestedIndices = Union[int, List['NestedIndices']]


class Blaster(ABC):
    """Base class for blasting."""

    def __init__(self, dtype=np.float64, progress=True):
        """Initialize class."""
        self.dtype = dtype
        self.progress = progress
        self.desc = "Blasting"
        self.self_hits = []
        self.neurons = []
        self.ids = []

    def __len__(self):
        return len(self.neurons)

    @abstractmethod
    def append(self, neurons) -> NestedIndices:
        """Append neurons."""
        pass

    @abstractmethod
    def calc_self_hit(self, neurons):
        """Non-normalized value for self hit."""
        pass

    @abstractmethod
    def single_query_target(self, q_idx, t_idx, scores='forward'):
        """Query single target against single target."""
        pass

    @property
    def dtype(self):
        """Data type used for scores."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        try:
            self._dtype = np.dtype(dtype)
        except TypeError:
            try:
                self._dtype = FLOAT_DTYPES[dtype]
            except KeyError:
                raise ValueError(
                    f'Unknown precision/dtype {dtype}. Expected on of the following: 16, 32 or 64 (default)'
                )

    def pair_query_target(self, pairs, scores='forward'):
        """BLAST multiple pairs.

        Parameters
        ----------
        pairs :             tuples
                            Tuples of (query_ix, target_ix) to query.
        scores :            "forward" | "mean" | "min" | "max" | "both"
                            Which scores to return.

        """
        # See `multi_query_target` for explanation on progress bars
        scr = []
        for p in config.tqdm_classic(pairs,
                             desc=f'{self.desc} pairs',
                             leave=False,
                             position=getattr(self,
                                              'pbar_position',
                                              None),
                             disable=not self.progress):
            scr.append(self.single_query_target(p[0], p[1], scores=scores))

        return scr

    def multi_query_target(self, q_idx, t_idx, scores='forward'):
        """BLAST multiple queries against multiple targets.

        Parameters
        ----------
        q_idx,t_idx :       iterable
                            Iterable of query/target neuron indices to BLAST.
        scores :            "forward" | "mean" | "min" | "max" | "both"
                            Which scores to return.

        """
        # There are currently a few issues with Jupyter (lab?) and tqdm:
        # 1. Subprocesses don't know that they were spawned from a Jupyter
        #    environment and consequently use classic tqdm. This by itself would
        #    be an easy fix but see below:
        # 2. Even forcing tqdm.notebook in subprocesses does not produce a Jupyter
        #    widget progress bar in the notebook - it just shows nothing until
        #    the process finishes. Even that empty print(' ', end='', flush=True)
        #    does not do the trick anymore.
        # 3. Using classic tqdm from multiple processes from inside a Jupyter
        #    enviroment leads to only one progress bar being shown... UNLESS
        #    `position!=None` in which case every update is printed on a new
        #    line which produces a horrendous mess. With `position=None` we
        #    only ever see a single classic progress bar but at least there is
        #    some feedback for the user without stdout exploding.
        # So it seems the only viable solution for now is:
        # - always use classic tqdm
        # - only use position when spawned outside a Jupyter environment
        # We could allow Jupyter progress bars on single cores but how often
        # does that happen?

        shape = (len(q_idx), len(t_idx)) if scores != 'both' else (len(q_idx), len(t_idx), 2)
        res = np.empty(shape, dtype=self.dtype)
        for i, q in enumerate(config.tqdm_classic(q_idx,
                                          desc=self.desc,
                                          leave=False,
                                          position=getattr(self,
                                                           'pbar_position',
                                                           None),
                                          disable=not self.progress)):
            for k, t in enumerate(t_idx):
                res[i, k] = self.single_query_target(q, t, scores=scores)

        # Generate results
        if res.ndim == 2:
            res = pd.DataFrame(res)
            res.columns = [self.ids[t] for t in t_idx]
            res.index = [self.ids[q] for q in q_idx]
            res.index.name = 'query'
            res.columns.name = 'target'
        else:
            # For scores='both' we will create a DataFrame with multi-index
            ix = pd.MultiIndex.from_product([[self.ids[q] for q in q_idx],
                                             ['forward', 'reverse']],
                                            names=["query", "score"])
            res = pd.DataFrame(np.hstack((res[:, :, 0],
                                          res[:, :, 1])).reshape(len(q_idx) * 2,
                                                                 len(t_idx)),
                               index=ix,
                               columns=[self.ids[t] for t in t_idx])
            res.columns.name = 'target'

        return res

    def all_by_all(self, scores='forward'):
        """BLAST all-by-all neurons."""
        res = self.multi_query_target(range(len(self.neurons)),
                                      range(len(self.neurons)),
                                      scores='forward')

        # For all-by-all BLAST we can get the mean score by
        # transposing the scores
        if scores == 'mean':
            res = (res + res.T) / 2
        elif scores == 'min':
            res.loc[:, :] = np.dstack((res, res.T)).min(axis=2)
        elif scores == 'max':
            res.loc[:, :] = np.dstack((res, res.T)).max(axis=2)
        elif scores == 'both':
            ix = pd.MultiIndex.from_product([res.index, ['forward', 'reverse']],
                                            names=["query", "score"])
            res = pd.DataFrame(np.hstack((res[:, :, 0],
                                          res[:, :, 1])).reshape(res.shape[0] * 2,
                                                                 res.shape[1]),
                               index=ix,
                               columns=res.columns)

        return res

    def __len__(self):
        return len(self.neurons)


def extract_matches(scores, N=1, axis=0, distances=False):
    """Extract top N matches from score matrix.

    Parameters
    ----------
    scores :    pd.DataFrame
                Score matrix (e.g. from :func:`navis.nblast`).
    N :         int
                Number of matches to extract.
    axis :      0 | 1
                For which axis to produce matches.
    distances : bool
                Set to True if input is distances instead of similarities (i.e.
                we need to look for the lowest instead of the highest values).

    Returns
    -------
    pd.DataFrame

    """
    assert axis in (0, 1), '`axis` must be 0 or 1'

    # Transposing is easier than dealing with the different axis further down
    if axis == 1:
        scores = scores.T

    if not distances:
        if N > 1:
            # This partitions of the largest N values (faster than argsort)
            # Their correct order, however, is not guaranteed
            top_n = np.argpartition(scores.values, -N, axis=-1)[:, -N:]
        else:
            # For N=1 this is still faster
            top_n = np.argmax(scores.values, axis=-1).reshape(-1, 1)
    else:
        if N > 1:
            top_n = np.argpartition(scores.values, N, axis=-1)[:, :N]
        else:
            top_n = np.argmin(scores.values, axis=-1).reshape(-1, 1)

    # This make sure we order them properly
    top_scores = scores.values[np.arange(len(scores)).reshape(-1, 1), top_n]
    ind_ordered = np.argsort(top_scores, axis=1)

    if distances:
        ind_ordered = ind_ordered[:, ::-1]

    top_n = top_n[np.arange(len(top_n)).reshape(-1, 1), ind_ordered]
    top_scores = top_scores[np.arange(len(top_scores)).reshape(-1, 1), ind_ordered]

    # Now collate matches
    matches = pd.DataFrame()
    matches['id'] = scores.index.values
    for i in range(N):
        matches[f'match_{i + 1}'] = scores.columns[top_n[:, -(i + 1)]]
        matches[f'score_{i + 1}'] = top_scores[:, -(i + 1)]

    return matches


def update_scores(queries, targets, scores_ex, nblast_func, **kwargs):
    """Update score matrix by running only new query->target pairs.

    Parameters
    ----------
    queries :       Dotprops
    targets :       Dotprops
    scores_ex :     pandas.DataFrame
                    DataFrame with existing scores.
    nblast_func :   callable
                    The NBLAST to use. For example: ``navis.nblast``.
    **kwargs
                    Argument passed to ``nblast_func``.

    Returns
    -------
    pandas.DataFrame
                    Updated scores.

    Examples
    --------

    Mostly for testing but also illustrates the principle:

    >>> import navis
    >>> import numpy as np
    >>> nl = navis.example_neurons(n=5)
    >>> dp = navis.make_dotprops(nl, k=5) / 125
    >>> # Full NBLAST
    >>> scores = navis.nblast(dp, dp, n_cores=1)
    >>> # Subset and fill in
    >>> scores2 = navis.nbl.update_scores(dp, dp,
    ...                                   scores_ex=scores.iloc[:3, 2:],
    ...                                   nblast_func=navis.nblast,
    ...                                   n_cores=1)
    >>> np.all(scores == scores2)
    True

    """
    if not callable(nblast_func):
        raise TypeError('`nblast_func` must be callable.')
    # The np.isin query is much faster if we force any strings to <U18 by
    # converting to arrays
    is_new_q = ~np.isin(queries.id, np.array(scores_ex.index))
    is_new_t = ~np.isin(targets.id, np.array(scores_ex.columns))

    logger.info(f'Found {is_new_q.sum()} new queries and '
                f'{is_new_t.sum()} new targets.')

    # Reindex old scores
    scores = scores_ex.reindex(index=queries.id, columns=targets.id).copy()

    # NBLAST new queries against all targets
    if 'precision' not in kwargs:
        kwargs['precision'] = scores.values.dtype

    if any(is_new_q):
        logger.info(f'Updating new queries -> targets scores')
        qt = nblast_func(queries[is_new_q], targets, **kwargs)
        scores.loc[qt.index, qt.columns] = qt.values

    # NBLAST all old queries against new targets
    if any(is_new_t):
        logger.info(f'Updating old queries -> new targets scores')
        tq = nblast_func(queries[~is_new_q], targets[is_new_t], **kwargs)
        scores.loc[tq.index, tq.columns] = tq.values

    return scores
