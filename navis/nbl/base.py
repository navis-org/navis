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

INT_DTYPES = {16: np.int16, 32: np.int32, 64: np.int64, None: None}
FLOAT_DTYPES = {16: np.float16, 32: np.float32, 64: np.float64, None: None}


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
        scores :            "forward" | "mean" | "min" | "max"
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
        scores :            "forward" | "mean" | "min" | "max"
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

        res = np.zeros((len(q_idx), len(t_idx)),
                       dtype=self.dtype)
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
        res = pd.DataFrame(res)
        res.columns = [self.ids[t] for t in t_idx]
        res.index = [self.ids[q] for q in q_idx]

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

        return res

    def __len__(self):
        return len(self.neurons)
