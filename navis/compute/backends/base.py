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

"""Backend abstraction and registry for parallel processing.

A *backend* is somewhere `parallel=True` can run work: a process pool, a thread
pool, or - once the cluster adapters land - a scheduler on another set of
machines. Backends are pure executor adapters: they receive a picklable
function and a list of payloads and know nothing about neurons. That is what
lets a `dask.distributed.Client` or a `submitit.AutoExecutor` be dropped in
without touching the dispatch layer.

Third-party libraries can register their own via :func:`register_backend`.

"""

import concurrent.futures as cf

from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional, Sequence

from ... import config

logger = config.get_logger(__name__)

__all__ = ['ParallelBackend', 'ExecutorBackend', 'register_backend',
           'get_backend', 'list_backends', 'available_backends',
           'resolve_backend', 'set_parallel_backend']

# Registry of name -> backend instance
_BACKENDS = {}


class ParallelBackend(ABC):
    """Base class for parallel processing backends.

    Attributes
    ----------
    name :          str
                    Unique name used to select this backend (e.g. via the
                    `backend` parameter or `navis.set_parallel_backend`).
    priority :      int
                    Higher-priority backends are preferred when
                    `backend="auto"`.
    auto_select :   bool
                    Whether `backend="auto"` may pick this backend at all. False
                    for backends that are only ever a deliberate choice (e.g.
                    `serial`, which is always "available" and would otherwise
                    make the "nothing can run this" branch unreachable).
    concurrent :    bool
                    Whether work actually overlaps. False for `serial`.
    isolated :      bool
                    Whether workers get their own address space. This decides
                    two things: whether it is safe to turn a caller's
                    `inplace=False` into an in-place operation (it is not, if
                    the "worker" shares our memory), and whether the parent's
                    `navis.config` has to be shipped along.
    pickles_by_value : bool
                    Whether the backend can send functions that cannot be
                    imported by name - lambdas, closures, notebook-defined
                    functions. `pickle` cannot; `dill` and `cloudpickle` can.

    """

    name: str = 'base'
    priority: int = 0
    auto_select: bool = True

    concurrent: bool = True
    isolated: bool = True
    pickles_by_value: bool = False

    def __repr__(self):
        return f"<ParallelBackend '{self.name}' (priority={self.priority})>"

    def available(self) -> bool:
        """Whether this backend's dependencies are importable."""
        return True

    def unsupported(self, **requirements) -> list:
        """Return reasons why this backend cannot serve `requirements`.

        An empty list means the request is supported. Non-empty entries are
        human-readable strings used both to skip a backend during `"auto"`
        resolution and to build the error when one was asked for by name.

        Recognised requirements:

        by_value :  bool
                    The work includes a callable that cannot be imported by
                    name.
        """
        reasons = []
        if requirements.get('by_value') and not self.pickles_by_value:
            reasons.append(
                f"'{self.name}' can only send functions that are importable by "
                "name (no lambdas, closures or notebook-defined functions)"
            )
        return reasons

    @staticmethod
    def _remedy(**requirements) -> str:
        """How to get past an `unsupported()` rejection."""
        if requirements.get('by_value'):
            return (' Lambdas, closures and functions defined in a notebook '
                    'need a backend that serialises by value - select one with '
                    "`navis.set_parallel_backend('joblib')` (or 'pathos'), or "
                    'run with `parallel=False`.')
        return ''

    def chunksize(self, n_tasks: int, n_workers: int,
                  requested: Optional[int] = None,
                  size_hint: Optional[Callable[[], float]] = None) -> int:
        """How many tasks to bundle into one unit of work.

        One task per unit is right for a local pool, where handing over a task
        is cheap. It is badly wrong for a scheduler, where each unit may be a
        separate job - hence the hook. `size_hint` is a *callable* returning
        the estimated bytes per task, so backends that don't need it never pay
        for computing it.
        """
        if requested is not None:
            return max(1, int(requested))
        return 1

    @abstractmethod
    def map(self, func: Callable, payloads: Sequence, *,
            n_workers: int) -> Iterator:
        """Call `func` on each payload, yielding results as they complete.

        Implementations must yield - not return a list - so progress bars can
        advance during the run, and may yield in **any** order: the caller
        restores the original order from an index inside the payload.

        `n_workers` is a hint. A backend wrapping a user-configured executor is
        free to ignore it.
        """

    def shutdown(self) -> None:
        """Release any persistent resources (e.g. a reused worker pool)."""


class ExecutorBackend(ParallelBackend):
    """Backend for anything implementing `concurrent.futures.Executor`.

    That covers the stdlib pools, loky, `mpi4py.futures`, ipyparallel and -
    via `Client.get_executor()` - `dask.distributed`, whose `ClientExecutor`
    is a real `cf.Executor` handing back real `cf.Future` objects.
    """

    @abstractmethod
    def get_executor(self, n_workers: int) -> cf.Executor:
        """Return the executor to submit to."""

    def release_executor(self, executor: cf.Executor) -> None:
        """Called when a `map` finishes. No-op if the executor is reused."""
        executor.shutdown()

    def map(self, func, payloads, *, n_workers):
        executor = self.get_executor(n_workers)
        futures = [executor.submit(func, p) for p in payloads]
        try:
            for f in cf.as_completed(futures):
                yield f.result()
        except BaseException:
            # Don't leave the rest of the work running after an error
            for f in futures:
                f.cancel()
            raise
        finally:
            self.release_executor(executor)


class WrappedExecutorBackend(ExecutorBackend):
    """Adapter for an `Executor` instance handed to us by the user.

    This is what makes `navis.set_parallel_backend(client.get_executor())`
    work, and is the seam the cluster adapters will slot into.
    """

    auto_select = False

    def __init__(self, executor, *, isolated=None, pickles_by_value=None):
        self.executor = executor
        self.name = f'custom:{type(executor).__name__}'

        inferred_isolated, inferred_by_value = _infer_capabilities(executor)
        self.isolated = inferred_isolated if isolated is None else bool(isolated)
        self.pickles_by_value = (inferred_by_value if pickles_by_value is None
                                 else bool(pickles_by_value))

    def get_executor(self, n_workers):
        return self.executor

    def release_executor(self, executor):
        # The user owns this executor - never shut it down for them
        pass


def _infer_capabilities(executor):
    """Guess (isolated, pickles_by_value) for a user-supplied executor.

    We can recognise the common ones. For anything else we deliberately guess
    the *safe* values: `isolated=False` means we never turn a caller's
    `inplace=False` into an in-place operation, which at worst costs a copy.
    """
    if isinstance(executor, cf.ThreadPoolExecutor):
        return False, True
    if isinstance(executor, cf.ProcessPoolExecutor):
        return True, False

    mod = type(executor).__module__ or ''
    if mod.startswith('joblib') or 'loky' in mod:
        return True, True
    if mod.startswith('distributed') or mod.startswith('dask'):
        # dask uses cloudpickle; workers are separate processes by default but
        # a LocalCluster(processes=False) shares memory - hence the safe guess.
        return False, True

    logger.debug(f'Unrecognised executor {type(executor).__name__}; assuming '
                 'it shares memory with the parent. Pass `isolated=True` to '
                 '`navis.set_parallel_backend` if it does not.')
    return False, False


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def register_backend(backend: ParallelBackend, name: str = None):
    """Register a parallel processing backend.

    Parameters
    ----------
    backend :   ParallelBackend
                The backend instance to register.
    name :      str, optional
                Name to register under. Defaults to `backend.name`.

    """
    if not isinstance(backend, ParallelBackend):
        raise TypeError(f'Expected ParallelBackend, got {type(backend)}')
    _BACKENDS[name or backend.name] = backend


def get_backend(name: str) -> ParallelBackend:
    """Get a registered backend by name."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown parallel backend '{name}'. "
                         f"Available: {list_backends()}")
    return _BACKENDS[name]


def list_backends() -> list:
    """List names of all registered backends."""
    return list(_BACKENDS)


def available_backends() -> list:
    """List backend instances whose dependencies are importable."""
    return [b for b in _BACKENDS.values() if b.available()]


def resolve_backend(backend=None, *, parallel: bool = True,
                    n_tasks: int = 2, n_workers: Optional[int] = None,
                    **requirements) -> ParallelBackend:
    """Select the backend to run a piece of work on.

    Parameters
    ----------
    backend :       str | ParallelBackend | concurrent.futures.Executor, optional
                    Either "auto" (highest-priority available backend that can
                    serve the request), the name of a backend, or an object to
                    wrap. `None` falls back to
                    `navis.config.default_parallel_backend`.
    parallel :      bool
                    If False, resolves to `serial` regardless of everything
                    else.
    n_tasks :       int
                    Number of tasks about to be run. A single task never
                    warrants spinning up workers.
    n_workers :     int, optional
                    Requested worker count. One worker means no concurrency, so
                    we run inline instead of paying to start a pool.
    **requirements
                    Passed to `unsupported()` - currently `by_value`.

    Returns
    -------
    ParallelBackend

    """
    # Not parallel, or nothing to parallelise over: don't start a pool just to
    # tear it down. This matters beyond performance - the caller uses the
    # resolved backend's `isolated` flag to decide whether it may force
    # `inplace=True`, and an inline run must not be treated as isolated.
    if not parallel or n_tasks <= 1 or (n_workers is not None and n_workers <= 1):
        return get_backend('serial')

    # Fall back to the configured default *before* dispatching on type - the
    # config may itself hold a backend instance or an executor.
    if backend is None:
        backend = config.default_parallel_backend

    if isinstance(backend, ParallelBackend):
        return backend

    if isinstance(backend, cf.Executor):
        return WrappedExecutorBackend(backend)

    if backend in (None, 'auto'):
        rejected = {}
        for be in sorted(available_backends(), key=lambda b: -b.priority):
            if not be.auto_select:
                continue
            reasons = be.unsupported(**requirements)
            if not reasons:
                return be
            rejected[be.name] = reasons

        # Nothing available fits. Say what would have, and how to get it.
        missing = [b.name for b in _BACKENDS.values()
                   if not b.available() and b.auto_select
                   and not b.unsupported(**requirements)]
        msg = 'No available parallel backend can run this work.'
        if missing:
            msg += (f' Backend(s) {missing} could, but are not installed'
                    f' (`pip install -U {" ".join(missing)}`).')
        if rejected:
            detail = '; '.join(f"{n}: {', '.join(r)}"
                               for n, r in rejected.items())
            msg += f' Rejected: {detail}.'
        msg += ParallelBackend._remedy(**requirements) or \
            ' Use `parallel=False` to run serially.'
        raise ValueError(msg)

    be = get_backend(backend)

    if not be.available():
        raise ValueError(
            f"Parallel backend '{be.name}' is not available - its optional "
            f"dependencies are not installed (`pip install -U {be.name}`)."
        )

    reasons = be.unsupported(**requirements)
    if reasons:
        raise ValueError(
            f"Parallel backend '{be.name}' cannot run this work: "
            f"{'; '.join(reasons)}.{be._remedy(**requirements)}"
        )

    return be


# --------------------------------------------------------------------------- #
# Public setter
# --------------------------------------------------------------------------- #
class _BackendSetter:
    """Applies a backend on creation and can restore the previous one.

    Doubles as a context manager so the same function serves both
    `navis.set_parallel_backend('joblib')` and
    `with navis.set_parallel_backend(client.get_executor()): ...`.
    """

    def __init__(self, backend, n_workers):
        self.previous = config.default_parallel_backend
        self.previous_n_workers = config.default_n_workers

        if backend is not None:
            config.default_parallel_backend = backend
        if n_workers is not None:
            config.default_n_workers = int(n_workers)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.restore()
        return False

    def restore(self):
        config.default_parallel_backend = self.previous
        config.default_n_workers = self.previous_n_workers


def set_parallel_backend(backend=None, *, n_workers=None, isolated=None,
                         pickles_by_value=None):
    """Set where `parallel=True` runs its work.

    Parameters
    ----------
    backend :   str | ParallelBackend | concurrent.futures.Executor, optional
                One of the registered backend names (see
                [`navis.list_parallel_backends`][]), `"auto"` to pick the best
                available, a backend instance, or any
                `concurrent.futures.Executor`. The last of these is how you
                point navis at a cluster: configure your executor with its own
                library's API and hand it over. `None` leaves the setting
                unchanged.
    n_workers : int, optional
                Default number of workers. `None` leaves it unchanged.
    isolated :  bool, optional
                Only for a bare executor: whether its workers have their own
                address space. Inferred where we recognise the executor,
                otherwise assumed False (the safe guess).
    pickles_by_value : bool, optional
                Only for a bare executor: whether it can ship lambdas and
                closures (i.e. uses `dill`/`cloudpickle` rather than `pickle`).

    Returns
    -------
    A restorer that can be used as a context manager to scope the change.

    Examples
    --------
    >>> import navis
    >>> # Set globally
    >>> _ = navis.set_parallel_backend('serial')
    >>> # ... or just for a block
    >>> with navis.set_parallel_backend('serial'):
    ...     nl = navis.example_neurons(2)
    >>> _ = navis.set_parallel_backend('auto')

    """
    if isinstance(backend, cf.Executor):
        backend = WrappedExecutorBackend(backend, isolated=isolated,
                                         pickles_by_value=pickles_by_value)
    elif isinstance(backend, str) and backend != 'auto':
        # Fail loudly here rather than at the next `parallel=True`
        get_backend(backend)

    return _BackendSetter(backend, n_workers)
