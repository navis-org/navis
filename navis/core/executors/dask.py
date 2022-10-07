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

import contextlib

import dask
import tqdm.dask

from typing import Optional

from ... import config
from . import Executor
from .base import kwargs_of_iter_to_iter_of_kwargs


class SimpleDaskExecutor(Executor):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def map(self, fn, *iterables, desc: Optional[str]=None, chunksize=1, **kwargs):
        kwargs_iter = kwargs_of_iter_to_iter_of_kwargs(**kwargs)

        if self.progress:
            pbar = tqdm.dask.TqdmCallback(
                tqdm_class=config.tqdm,
                desc=desc,
                disable=(config.pbar_hide
                        or not self.progress),
                leave=config.pbar_leave)
        else:
            pbar = contextlib.nullcontext()

        res = [dask.delayed(f)(*a, **kw) for f, a, kw in
                zip(fn, zip(*iterables), kwargs_iter)
        ]

        with pbar:
            res = dask.compute(res)[0]

        return iter(res)

    @classmethod
    def supports_inplace(cls):
        return True
