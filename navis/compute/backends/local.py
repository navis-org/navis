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

"""Dependency-free backends: serial, threads and the stdlib process pool."""

import os
import threading
import multiprocessing as mp
import concurrent.futures as cf

from concurrent.futures.process import BrokenProcessPool

from ... import config
from .base import ParallelBackend, ExecutorBackend

logger = config.get_logger(__name__)

__all__ = ['SerialBackend', 'ThreadBackend', 'ProcessBackend']

#: Seconds an unused process pool is kept alive. Starting one costs ~2s (the
#: workers re-import navis), so keeping it around makes a sequence of parallel
#: calls much faster; holding it forever would keep N idle interpreters
#: resident. Set to 0 to tear the pool down after every call.
IDLE_TIMEOUT = 60


class SerialBackend(ParallelBackend):
    """Run everything in the calling process, one task after another.

    Useful for debugging something that misbehaves only in a worker, and the
    backend the dispatcher degrades to when there is nothing to parallelise.
    """

    name = 'serial'
    priority = -20
    auto_select = False

    concurrent = False
    isolated = False
    pickles_by_value = True     # nothing is serialised at all

    def map(self, func, payloads, *, n_workers):
        for payload in payloads:
            yield func(payload)


class ThreadBackend(ExecutorBackend):
    """Run in threads.

    Only helps for work that releases the GIL, and shares memory with the
    caller - hence `isolated = False`.
    """

    name = 'threads'
    priority = -10
    auto_select = False

    isolated = False
    pickles_by_value = True     # nothing is serialised at all

    def get_executor(self, n_workers):
        return cf.ThreadPoolExecutor(max_workers=n_workers)


# --------------------------------------------------------------------------- #
# Process pool, with reuse
# --------------------------------------------------------------------------- #
_POOL = None
_POOL_KEY = None
_POOL_TIMER = None
_POOL_LOCK = threading.RLock()


def _default_mp_context():
    """Pick a start method, avoiding `fork`.

    Forking a process that already has threads running risks deadlocks, which
    is why CPython deprecated it in 3.12 and, from 3.14, no longer defaults to
    it on Linux. navis pulls in plenty of threaded libraries, so we opt out
    early rather than inherit the platform default where that is still `fork`.
    """
    ctx = mp.get_context()
    if ctx.get_start_method() != 'fork':
        return ctx
    if 'forkserver' in mp.get_all_start_methods():
        return mp.get_context('forkserver')
    return mp.get_context('spawn')


def _get_pool(n_workers):
    """Return a (possibly reused) process pool."""
    global _POOL, _POOL_KEY, _POOL_TIMER

    ctx = _default_mp_context()
    # The pid is part of the key so a forked child never inherits - and then
    # deadlocks on - its parent's pool.
    key = (n_workers, ctx.get_start_method(), os.getpid())

    with _POOL_LOCK:
        if _POOL_TIMER is not None:
            _POOL_TIMER.cancel()
            _POOL_TIMER = None

        if _POOL is not None and _POOL_KEY != key:
            _POOL.shutdown(wait=False)
            _POOL = None

        if _POOL is None:
            _POOL = cf.ProcessPoolExecutor(max_workers=n_workers,
                                           mp_context=ctx)
            _POOL_KEY = key

        return _POOL


def _idle_pool():
    """Start the countdown to shutting an unused pool down."""
    global _POOL_TIMER

    with _POOL_LOCK:
        if _POOL_TIMER is not None:
            _POOL_TIMER.cancel()
            _POOL_TIMER = None
        if IDLE_TIMEOUT and _POOL is not None:
            _POOL_TIMER = threading.Timer(IDLE_TIMEOUT, shutdown_pool)
            _POOL_TIMER.daemon = True
            _POOL_TIMER.start()
        elif not IDLE_TIMEOUT:
            shutdown_pool()


def shutdown_pool():
    """Tear down the shared process pool, if there is one."""
    global _POOL, _POOL_KEY, _POOL_TIMER

    with _POOL_LOCK:
        if _POOL_TIMER is not None:
            _POOL_TIMER.cancel()
            _POOL_TIMER = None
        if _POOL is not None:
            _POOL.shutdown(wait=False)
            _POOL = None
            _POOL_KEY = None


class ProcessBackend(ExecutorBackend):
    """Run in separate processes using the standard library.

    The default when nothing better is installed: it needs no dependencies at
    all. The trade-off is that it serialises with plain `pickle`, so it cannot
    ship lambdas, closures or functions defined in a notebook.
    """

    name = 'processes'
    priority = 0

    isolated = True
    pickles_by_value = False

    def get_executor(self, n_workers):
        return _get_pool(n_workers)

    def release_executor(self, executor):
        # Pool is shared and outlives this call - just start the idle timer
        _idle_pool()

    def map(self, func, payloads, *, n_workers):
        try:
            yield from super().map(func, payloads, n_workers=n_workers)
        except BrokenProcessPool:
            # A dead worker (OOM kill, segfault) poisons the whole executor -
            # drop it so the next call gets a fresh one.
            shutdown_pool()
            raise

    def shutdown(self):
        shutdown_pool()
