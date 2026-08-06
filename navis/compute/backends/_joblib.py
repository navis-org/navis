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

import contextlib

from ... import config
from .base import ParallelBackend

logger = config.get_logger(__name__)

__all__ = ['JoblibBackend']


class JoblibBackend(ParallelBackend):
    """Run via `joblib.Parallel`.

    The default where installed. Serialises with `cloudpickle`, so lambdas and
    notebook-defined functions work, and its workers (loky) are kept alive
    between calls - which makes a sequence of `parallel=True` calls noticeably
    faster than rebuilding a pool each time.
    """

    name = 'joblib'
    priority = 20
    requires = 'joblib'

    isolated = True
    pickles_by_value = True

    def _parallel(self, joblib, n_workers):
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

    def _thread_cap(self, joblib, threads):
        """Tell joblib the per-worker thread cap, or nothing if there isn't one.

        Two things come of this, and the second is the important one:

        - joblib sets `OMP_NUM_THREADS` and friends in the workers *as it starts
          them*, which is the only moment that actually works for a BLAS library
          - by the time our own hook runs in the worker, numpy has been imported
          and the variables have been read.
        - the value is part of loky's executor identity, so changing it makes
          loky tear the pool down and start fresh workers rather than resizing
          the one it has. Without that, a second call at a different cap would
          silently reuse workers whose thread pools are already built and can no
          longer be resized (`_resize()` keeps existing workers alive).

        Contrast `n_jobs`, which loky *does* only resize on - which is why the
        cap has to be stated separately rather than inferred from it.
        """
        if threads is None:
            # No cap. Leave joblib to its own default, which is the same
            # arithmetic we would do (`cpu_count // n_jobs`) and which - unlike
            # an explicit value - defers to an `OMP_NUM_THREADS` the user
            # exported on purpose.
            return contextlib.nullcontext()

        # joblib only accepts `inner_max_num_threads` together with an explicit
        # backend, and naming one is exactly what we must not do in general:
        # `joblib.parallel_config` is how somebody points joblib at dask, ray or
        # ipyparallel, and forcing 'loky' here would quietly undo that. So ask
        # what is active, and only step in for a backend that says it supports
        # the parameter - which is the same flag `parallel_config` itself
        # validates against, rather than a class name that a subclass or a
        # rename would silently break.
        from joblib.parallel import get_active_backend

        active = get_active_backend()[0]
        if not getattr(active, 'supports_inner_max_num_threads', False):
            return contextlib.nullcontext()

        # Named rather than passed as the object: joblib builds a fresh backend
        # from the name, so the user's own instance is left alone.
        return joblib.parallel_config(backend='loky',
                                      inner_max_num_threads=threads)

    #: Whether we have ever asked joblib to run anything - i.e. whether the
    #: pool `shutdown` would tear down is plausibly ours to tear down.
    _used = False

    def map(self, func, payloads, *, n_workers, threads=None):
        import joblib

        self._used = True
        with self._thread_cap(joblib, threads):
            par = self._parallel(joblib, n_workers)
            yield from par(joblib.delayed(func)(p) for p in payloads)

    def shutdown(self):
        """Tear down loky's reusable pool, if one is up because of us.

        Keeping the workers alive is most of why this backend is the default,
        but each one is a whole navis interpreter, so somebody who is done with
        them has to be able to get that memory back.

        loky's pool is a process-wide global that a caller's own
        `joblib.Parallel` code shares, and there is no public handle on it -
        `get_reusable_executor()` would *create* one, which is the opposite of
        what is wanted here. So: only if navis put work on it, and treat any
        change in loky's shape as "nothing to shut down" rather than an error.
        Shutting the pool down early is a courtesy; loky expires it on its own
        timer regardless.
        """
        if not self._used:
            return

        try:
            from joblib.externals.loky import reusable_executor
        except ImportError:
            return

        executor = getattr(reusable_executor, '_executor', None)
        if executor is not None:
            executor.shutdown(wait=False)
