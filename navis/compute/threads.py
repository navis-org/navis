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

"""Capping the native thread pools inside a process.

navis spreads work over *processes*; several of the libraries underneath it -
navis-fastcore above all - spread work over *threads*, and by default take
every core they can see. Neither knows about the other, so `n_cores=20` on a
224-core machine is 20 workers x 224 threads over 224 cores. That is not a
theoretical cost: measured on exactly that machine, healing 40 large skeletons
this way was *slower* than not parallelising at all, at 2.3x the CPU.

This module is the other half of the bargain: it tells one process how much of
the machine it is entitled to. :func:`navis.set_num_threads` is the public
front door; the dispatcher applies the same thing to every worker it starts.

Important: this module must not import from `navis.core` or `navis.utils` -
see the note at the top of `dispatch.py`.

"""

import os

from typing import Optional

from .. import config

logger = config.get_logger(__name__)

__all__ = ['set_num_threads', 'limit_native_threads']


#: Environment variables that cap a native thread pool, and who reads them.
#: These are only read when the library in question first builds its pool, so
#: setting them is worth doing early and worth nothing late - hence the other
#: two mechanisms below.
THREAD_ENV_VARS = (
    'RAYON_NUM_THREADS',        # rust's rayon, i.e. navis-fastcore
    'OMP_NUM_THREADS',          # OpenMP: pykdtree, scipy, scikit-learn, ...
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'BLIS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',   # macOS Accelerate
    'NUMBA_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
)

#: The limit currently in force in this process, so that re-applying the same
#: one - which happens once per chunk in a worker - costs nothing.
_APPLIED: Optional[int] = None


def set_num_threads(n: int) -> None:
    """Cap the number of threads navis and its dependencies use.

    Applies to *this* process: navis-fastcore's thread pool, and the BLAS /
    OpenMP pools underneath numpy and friends.

    You do not need this for `parallel=True` - navis caps its own workers (see
    [`navis.set_parallel_backend`][]). It is for the other direction: when
    **you** are the one running the pool and navis is the thing inside it,
    nothing tells a worker that it is one of twenty, so it has to be told.

    ```python
    import multiprocessing as mp

    def work(neuron):
        navis.set_num_threads(1)      # or once, in the pool's `initializer`
        return navis.heal_skeleton(neuron)

    with mp.Pool(20) as pool:
        healed = pool.map(work, neurons)
    ```

    Call it before doing any real work. Thread pools are built once, lazily, by
    whatever needs one first; this can size a pool that does not exist yet but
    cannot resize one that does. Repeating the same value is free, which is what
    makes it safe in a worker initializer that fires more than once.

    Parameters
    ----------
    n :         int
                Maximum number of threads. Must be >= 1.

    See Also
    --------
    [`navis.set_parallel_backend`][]
                Where `parallel=True` runs its work, and how many threads each
                of its workers may use.

    Examples
    --------
    >>> import navis
    >>> navis.set_num_threads(1)                          # doctest: +SKIP

    Skipped rather than run: thread pools are built once per process, so a
    doctest that really capped this one would hand every later test a
    single-threaded interpreter and no way to get it back.

    """
    n = int(n)
    if n < 1:
        raise ValueError(f'Number of threads must be >= 1, got {n}')
    limit_native_threads(n)


def limit_native_threads(n: Optional[int]) -> None:
    """Apply a thread cap to this process. `None` or 0 caps nothing.

    Best-effort by design, and silent about what it could not do: this runs in
    a worker whose stderr nobody may ever read, once per chunk, and failing to
    cap a thread pool is a performance problem rather than a correctness one.

    **Only ever call this in a process navis owns.** Setting the environment
    variables in a parent process is not harmless: `OMP_NUM_THREADS` and
    friends are part of the environment loky builds its workers from, so
    changing them mid-session makes joblib consider its pool stale and rebuild
    it - twice, once on the way in and once on the way out.
    """
    global _APPLIED

    if not n or _APPLIED == n:
        return
    _APPLIED = n

    _set_thread_env(n)
    _limit_fastcore(n)
    _limit_threadpools(n)


def _set_thread_env(n: int) -> None:
    """Cap anything that builds its pool from the environment.

    Late for a library that is already loaded, but this is the only mechanism
    that covers one loaded *after* us - and, for navis-fastcore, it is what
    makes the cap work on versions predating `set_num_threads`, since rayon
    reads `RAYON_NUM_THREADS` when it lazily builds its global pool.
    """
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n)


def _limit_fastcore(n: int) -> None:
    """Size navis-fastcore's (rayon) thread pool."""
    import navis_fastcore as fastcore

    # Added in navis-fastcore 0.11; `_set_thread_env` covers older versions
    if not hasattr(fastcore, 'set_num_threads'):
        return

    try:
        fastcore.set_num_threads(n)
    except RuntimeError as e:
        # The pool is built once per process and cannot be resized. Reaching
        # here means something already built it at a different size - most
        # likely a reused worker that an earlier call capped differently.
        logger.debug(f'Could not cap navis-fastcore to {n} thread(s): {e}')


def _limit_threadpools(n: int) -> None:
    """Resize the BLAS/OpenMP pools that are already loaded.

    `_set_thread_env` cannot reach these: a worker imports navis - and
    therefore numpy, and therefore BLAS - before anything here runs, and by
    then the variables have been read. `threadpoolctl` talks to the loaded
    libraries directly, which is the only thing that does work at this point.
    It is an optional dependency, hence the soft failure.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return

    try:
        # Not as a context manager: we want the limit to outlive this call
        threadpool_limits(limits=n)
    except Exception as e:  # pragma: no cover
        logger.debug(f'Could not cap native thread pools to {n}: {e}')
