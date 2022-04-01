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

import atexit

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from multiprocessing import shared_memory

from .. import utils, config

INT_DTYPES = {16: np.int16, 32: np.int32, 64: np.int64, None: None}
FLOAT_DTYPES = {16: np.float16, 32: np.float32, 64: np.float64, None: None}


class Blaster(ABC):
    """Base class for Blasting."""

    def __init__(self, dtype=np.float64, progress=True):
        """Initialize class."""
        self.dtype = dtype
        self.progress = progress
        self.desc = "Blasting"
        self.self_hits = []
        self.neurons = []
        self.ids = []

    @abstractmethod
    def append(self, neurons):
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
        self._dtype = parse_precision(dtype)

    def pair_query_target(self, pairs, scores='forward'):
        """BLAST multiple pairs.

        Parameters
        ----------
        pairs :             tuples
                            Tuples of (query_ix, target_ix) to query.
        scores :            "forward" | "mean" | "min" | "max"
                            Which scores to return.

        """
        if utils.is_jupyter() and config.tqdm == config.tqdm_notebook:
            # Jupyter does not like the progress bar position for some reason
            position = None

            # For some reason we have to do this if we are in a Jupyter environment
            # and are using multi-processing because otherwise the progress bars
            # won't show. See this issue:
            # https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308
            print(' ', end='', flush=True)
        else:
            position = getattr(self, 'pbar_position', 0)

        scr = []
        for p in config.tqdm(pairs,
                             desc=f'{self.desc} pairs',
                             leave=False,
                             position=position,
                             disable=not self.progress):
            scr.append(self.single_query_target(p[0], p[1], scores=scores))

        return scr

    def multi_query_target(self, q_idx, t_idx, scores='forward', out=None):
        """BLAST multiple queries against multiple targets.

        Parameters
        ----------
        q_idx,t_idx :       iterable
                            Iterable of query/target neuron indices to BLAST.
        scores :            "forward" | "mean" | "min" | "max"
                            Which scores to return.
        out :               np.ndarray, optional
                            Array to write results to.

        """
        if utils.is_jupyter() and config.tqdm == config.tqdm_notebook:
            # Jupyter does not like the progress bar position for some reason
            position = None

            # For some reason we have to do this if we are in a Jupyter environment
            # and are using multi-processing because otherwise the progress bars
            # won't show. See this issue:
            # https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308
            print(' ', end='', flush=True)
        else:
            position = getattr(self, 'pbar_position', 0)

        shape = (len(q_idx), len(t_idx))
        if not isinstance(out, np.ndarray):
            out = np.zeros(shape, dtype=self.dtype)
        elif out.shape != shape:
            raise TypeError(f'Expected out array of shape {shape}, got {out.shape}')
        elif out.dtype != self.dtype:
            raise TypeError(f'Expected out array to be {self.dtype}, got {out.dtype}')

        for i, q in enumerate(config.tqdm(q_idx,
                                          desc=self.desc,
                                          leave=False,
                                          position=position,
                                          disable=not self.progress)):
            for k, t in enumerate(t_idx):
                out[i, k] = self.single_query_target(q, t, scores=scores)

        # Generate results
        res = pd.DataFrame(out)
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


class SharedBlaster(Blaster, ABC):
    """Version of the Blaster that works with shared memory buffers."""

    def create_array(self, shm, shape, offset=None):
        """Create array from shared memory buffer."""
        # Generate the out array from shared memory buffer
        arr = np.ndarray(shape, dtype=self.dtype, buffer=shm.buf)

        if not isinstance(offset, type(None)):
            arr = arr[offset]

        return arr

    def multi_query_target(self, q_idx, t_idx, shm, shape, offset=None, scores='forward'):
        """BLAST multiple queries against multiple targets.

        Parameters
        ----------
        q_idx,t_idx :   iterable
                        Iterable of query/target neuron indices to BLAST.
        shm :           multiprocessing.shared_memory.SharedMemory                        
        shape :         tuple (N, M)
                        Shape of the array in the memory buffer.
        offset :        tuple (slice, slice)
                        The view inside the full array to pass through as `out` to
                        `multi_query_target`.
        scores :        "forward" | "mean" | "min" | "max"
                        Which scores to produce.

        """
        out = self.create_array(shm, shape, offset=offset)

        _ = super().multi_query_target(q_idx, t_idx, scores=scores, out=out)

    def all_by_all(self, shm, shape, offset=None, scores='forward'):
        """BLAST all-by-all neurons."""
        out = self.create_array(shm, shape, offset=offset)

        _ = super().multi_query_target(range(len(self.neurons)),
                                       range(len(self.neurons)),
                                       scores='forward',
                                       out=out)

        # For all-by-all BLAST we can get the mean score by
        # transposing the scores
        if scores == 'mean':
            out[:, :] = (out + out.T) / 2
        elif scores == 'min':
            out[:, :] = np.dstack((out, out.T)).min(axis=2)
        elif scores == 'max':
            out[:, :] = np.dstack((out, out.T)).max(axis=2)


def create_shared_array(shape, dtype):
    """Create shared array.

    Parameters
    ----------
    shape :     tuple
    dtype :     str | np.dtype

    Returns
    -------
    multiprocessing.shared_memory.SharedMemory
                The shared memory buffer for the array.
    np.ndarray
                A numpy array accessing the shared memory buffer.

    """
    # Get the number of items in the requested array
    if utils.is_iterable(shape):
        items = np.prod(shape)
    else:
        items = shape

    # Force dtype to numpy dtype
    dtype = np.dtype(dtype)
    # Calculate required size for memory buffer
    size = dtype.itemsize * items

    # Create shared memory buffer
    shm = shared_memory.SharedMemory(create=True, size=size)

    # We need to make sure that the memory is released on exit of this process
    # Note: order is reverse -> last registered is executed first
    atexit.register(shm.unlink)
    atexit.register(shm.close)

    # Create array based on buffer
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    return shm, arr


def parse_precision(dtype):
    """Parse precision into numpy dtype."""
    try:
        return np.dtype(dtype)
    except TypeError:
        try:
            return FLOAT_DTYPES[dtype]
        except KeyError:
            raise ValueError(f'Unknown precision/dtype {dtype}. Expected one '
                             'of the following: 16, 32 or 64 (default)')
