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

"""The joblib backend.

Worth having even though joblib's default executor (loky) is also reachable as
a plain `concurrent.futures.Executor`: `joblib.parallel_config` is the plug
that dask, ray and ipyparallel all provide, so this one adapter reaches several
more schedulers.

Its other advantage is that loky *reuses* its worker pool between calls, which
in benchmarking was the single biggest factor for repeated `parallel=True`
calls - starting workers costs far more than the work itself for small jobs.
"""

from ... import config
from .base import ParallelBackend

logger = config.get_logger(__name__)

__all__ = ['JoblibBackend']

try:
    import joblib
except ModuleNotFoundError:
    joblib = None


class JoblibBackend(ParallelBackend):
    """Run via `joblib.Parallel`.

    Serialises with `cloudpickle`, so lambdas and notebook-defined functions
    work. Workers are separate processes (loky) that are kept alive between
    calls.
    """

    name = 'joblib'
    priority = 10

    isolated = True
    pickles_by_value = True

    def available(self):
        return joblib is not None

    def _parallel(self, n_workers):
        """Build a `Parallel` that streams results as they land.

        `return_as="generator_unordered"` needs joblib >= 1.4; older versions
        can only hand back a list, which just means the progress bar jumps to
        100% at the end.
        """
        for return_as in ('generator_unordered', 'generator'):
            try:
                return joblib.Parallel(n_jobs=n_workers, return_as=return_as)
            except (TypeError, ValueError):
                continue
        return joblib.Parallel(n_jobs=n_workers)

    def map(self, func, payloads, *, n_workers):
        par = self._parallel(n_workers)
        yield from par(joblib.delayed(func)(p) for p in payloads)
