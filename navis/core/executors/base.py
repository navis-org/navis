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

import itertools
import os

from typing import Optional

from ... import config


try:
    #from pathos.multiprocessing import ProcessingPool
    # pathos' ProcessingPool apparently ignores chunksize
    # (see https://stackoverflow.com/questions/55611806/how-to-set-chunk-size-when-using-pathos-processingpools-map)
    import pathos
    ProcessingPool = pathos.pools._ProcessPool
except ImportError:
    ProcessingPool = None


def kwargs_of_iter_to_iter_of_kwargs(**kwargs):
    """Convert a dict with iterable values to an iterator of dicts."""
    if kwargs:
        kwargs_iters = zip(*kwargs.values())
        kwargs_iter = (dict(zip(kwargs.keys(), vs)) for vs in kwargs_iters)
    else:
        kwargs_iter = itertools.repeat({})

    return kwargs_iter


class Executor:
    """Simple analogue to `concurrent.futures.Executor` with progress and error handling."""
    progress: bool

    def __init__(
        self,
        progress: bool = True,
    ):
        self.progress = progress

    def submit(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def map(self, fn, *iterables, desc: Optional[str]=None, chunksize=1, **kwargs):
        kwargs_iter = kwargs_of_iter_to_iter_of_kwargs(**kwargs)

        res = list(map(
                lambda x, kw: self.submit(*x, **kw),
                config.tqdm(
                    zip(fn, *iterables),
                    desc=desc,
                    disable=(config.pbar_hide
                            or not self.progress),
                    leave=config.pbar_leave),
                kwargs_iter,
            )
        )

        return iter(res)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    @classmethod
    def supports_inplace(cls):
        return True


class PathosProcessPoolExecutor(Executor):
    def __init__(
        self,
        n_cores: int = os.cpu_count() // 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not ProcessingPool:
            raise ImportError('navis relies on pathos for multiprocessing!'
                                'Please install pathos and try again:\n'
                                '  pip3 install pathos -U')
        self.pool = ProcessingPool(n_cores)

    def __enter__(self):
        self.pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.pool.__exit__(exc_type, exc_val, exc_tb)

    def map(self, fn, *iterables, desc: Optional[str]=None, chunksize=1, **kwargs):
        kwargs_iter = kwargs_of_iter_to_iter_of_kwargs(**kwargs)

        res = list(config.tqdm(self.pool.imap(
                        lambda x: x[0](*x[1], **x[2]),
                        zip(fn, zip(*iterables), kwargs_iter),
                        chunksize=chunksize,
                    ),
                    desc=desc,
                    disable=(config.pbar_hide
                            or not self.progress),
                    leave=config.pbar_leave),
        )

        return iter(res)

    @classmethod
    def supports_inplace(cls):
        return False
