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
pool, or a scheduler on another set of machines. Backends are pure executor
adapters: they receive a picklable function and a list of payloads and know
nothing about neurons. That is what lets a `dask.distributed.Client` or a
`submitit.AutoExecutor` be dropped in without touching the dispatch layer.

Third-party libraries can register their own via :func:`register_backend`.

"""

import concurrent.futures as cf

from abc import ABC, abstractmethod
from importlib.util import find_spec
from typing import Callable, Iterator, Optional, Sequence, Tuple

from ... import config

logger = config.get_logger(__name__)

__all__ = ['ParallelBackend', 'ExecutorBackend', 'register_backend',
           'get_backend', 'list_backends', 'available_backends',
           'resolve_backend', 'set_parallel_backend', 'non_forking_context',
           'auto_chunksize', 'adopt_object', 'apply_overrides']

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
    chunks_per_worker : int | None
                    How many units of work to aim for per worker. `None` (the
                    default) means "don't bundle": one task per unit, which is
                    right for a local pool where handing over a task costs
                    microseconds. See :func:`auto_chunksize`.
    max_chunk_bytes : int
                    Ceiling on the estimated payload of one unit. Only consulted
                    when `chunks_per_worker` is set.
    marshals_exceptions : bool
                    Whether an exception raised in a worker arrives here as
                    itself. False for transports that only carry a "the job
                    failed" marker plus text (submitit); the dispatcher then
                    brings failures back as data instead, so callers see the
                    same exception they would have seen locally.

    """

    name: str = 'base'
    priority: int = 0
    auto_select: bool = True

    concurrent: bool = True
    isolated: bool = True
    pickles_by_value: bool = False
    marshals_exceptions: bool = True

    chunks_per_worker: Optional[int] = None
    max_chunk_bytes: int = 128 * 1024 ** 2

    #: Module this backend needs, if any. Drives `available()` and the "not
    #: installed" hint, which is why it is the import name rather than the
    #: backend name - `pip install dask` would not get you `distributed`.
    requires: Optional[str] = None

    #: Top-level modules whose objects this backend may claim. Checked before
    #: `_adopt` is called, so an implementation never has to import its library
    #: just to find out that the object isn't its own.
    adopts: Tuple[str, ...] = ()

    def __repr__(self):
        return f"<ParallelBackend '{self.name}' (priority={self.priority})>"

    def available(self) -> bool:
        """Whether this backend's dependencies are importable."""
        # A spec lookup rather than an import: this runs on every `"auto"`
        # resolution, and importing e.g. `distributed` costs far more than
        # users who never select it should pay.
        return self.requires is None or find_spec(self.requires) is not None

    def adopt(self, obj, **overrides) -> Optional['ParallelBackend']:
        """Return a backend wrapping `obj`, or None if it isn't ours.

        This is how an object that is *not* a `concurrent.futures.Executor` -
        a `submitit.AutoExecutor`, a `dask.distributed.Client` - can still be
        handed to [`navis.set_parallel_backend`][].

        Every registered backend is asked, including ones whose dependencies
        are missing, so nothing may be imported until `adopts` has matched.
        Override `_adopt`, not this.
        """
        if type(obj).__module__.split('.')[0] not in self.adopts:
            return None
        return self._adopt(obj, **overrides)

    def _adopt(self, obj, **overrides) -> Optional['ParallelBackend']:
        """Claim `obj`, or return None. Only called once `adopts` matched."""
        return None

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

        An explicit `requested` always wins; otherwise backends that set
        `chunks_per_worker` get the policy in :func:`auto_chunksize` and the
        rest get one task per unit. Override `worker_count`, not this, if the
        number of workers is not the one the caller thinks it is.
        """
        if requested is not None:
            return max(1, int(requested))
        if not self.chunks_per_worker:
            return 1
        # Cheap out before asking a remote backend to count its workers: with
        # this few tasks the answer is one apiece whatever it says.
        if n_tasks <= self.chunks_per_worker:
            return 1
        return auto_chunksize(n_tasks, self.worker_count(n_workers),
                              chunks_per_worker=self.chunks_per_worker,
                              max_bytes=self.max_chunk_bytes,
                              size_hint=size_hint)

    def worker_count(self, hint: Optional[int]) -> Optional[int]:
        """How many workers will actually run this, for sizing purposes.

        `hint` is what the caller asked for, which for a local pool is the
        answer. A backend talking to a cluster knows better - `n_cores`
        defaults to half of *this* machine's cores and says nothing about how
        big the cluster is.
        """
        return hint

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

    def __init__(self, executor, **overrides):
        self.executor = executor
        self.name = f'custom:{type(executor).__name__}'

        for key, value in _infer_capabilities(executor).items():
            setattr(self, key, value)
        apply_overrides(self, **overrides)

    def get_executor(self, n_workers):
        return self.executor

    def release_executor(self, executor):
        # The user owns this executor - never shut it down for them
        pass


# --------------------------------------------------------------------------- #
# Chunking policy
# --------------------------------------------------------------------------- #
def auto_chunksize(n_tasks: int, n_workers: int, *,
                   chunks_per_worker: int,
                   max_bytes: Optional[int] = None,
                   size_hint: Optional[Callable[[], float]] = None) -> int:
    """Bundle `n_tasks` into units sized for an expensive transport.

    Locally, one neuron per unit is fine - handing a task to a worker on the
    same machine costs microseconds. Once the unit has to cross a network, or
    *is* a scheduler job, that stops being true: 10,000 neurons would mean
    10,000 round trips, or 10,000 SLURM jobs.

    Two bounds decide the size:

    - **How many units.** Aim for `chunks_per_worker` units per worker. One
      unit each would leave every worker idle the moment it finishes early, so
      a small multiple buys load balancing; a large one just pays the transport
      cost again. This is the bound that normally binds.
    - **How big a unit.** With `size_hint`, no unit is allowed past `max_bytes`
      of estimated payload. This only binds for very large jobs, and it is a
      safety valve rather than a target: it keeps a worker from having to hold
      an unreasonable slice of the data, and limits how much work one failure
      takes down with it.

    Parameters
    ----------
    n_tasks :           int
    n_workers :         int
    chunks_per_worker : int
                        Units of work to aim for per worker.
    max_bytes :         int, optional
                        Ceiling on the estimated payload of a unit.
    size_hint :         callable, optional
                        Returns the estimated bytes of a single task. Only
                        called when it could actually change the answer.

    Returns
    -------
    int
                        Tasks per unit; always >= 1.

    """
    n_workers = max(1, int(n_workers or 1))
    units = max(1, n_workers * int(chunks_per_worker))
    cs = -(-n_tasks // units)  # ceil

    # Don't ask for a size estimate we can't act on - it costs a sample of the
    # data on the neuron path.
    if max_bytes and size_hint is not None and cs > 1:
        try:
            per_task = float(size_hint())
        except Exception as e:
            logger.debug(f'Size hint failed ({e}); chunking by task count only.')
            per_task = 0
        if per_task > 0:
            cs = min(cs, max(1, int(max_bytes // per_task)))

    return max(1, min(cs, n_tasks))


def non_forking_context(mp):
    """Return a start-method context for `mp` that is not `fork`.

    `mp` is either the standard library's `multiprocessing` or pathos'
    `multiprocess`, which mirrors its API but defaults to `fork` on every
    platform - including macOS, where the standard library does not.

    Forking is unsafe once a process has done real work. Only the forking
    thread survives into the child, so any native thread pool the parent had
    spun up - BLAS/OpenMP, or Accelerate on macOS - is gone, while the locks
    those pools held are inherited in whatever state they were in. The first
    call in the child that touches one then blocks forever, with no error and
    no output. navis reaches that state almost immediately: skeletonizing a
    mesh is enough. CPython deprecated fork-with-threads in 3.12 and stopped
    defaulting to it on Linux in 3.14; we opt out everywhere.
    """
    ctx = mp.get_context()
    if ctx.get_start_method() != 'fork':
        return ctx
    # forkserver forks from a process started before any of that work, so it
    # keeps most of fork's cheapness without inheriting the hazard.
    for method in ('forkserver', 'spawn'):
        if method in mp.get_all_start_methods():
            return mp.get_context(method)
    return ctx


#: Capability flags a user may override when handing over an executor, and how
#: to coerce them. One vocabulary, so a typo is an error rather than an ignored
#: keyword that silently leaves the inferred value in place - and so that
#: everything the protocol documents as a capability can actually be stated.
_CAPABILITIES = {'isolated': bool, 'pickles_by_value': bool,
                 'marshals_exceptions': bool, 'chunks_per_worker': int}


def apply_overrides(backend, **overrides):
    """Set capability flags a user stated explicitly. `None` means "keep"."""
    for key, value in overrides.items():
        if key not in _CAPABILITIES:
            raise TypeError(f"Unknown capability '{key}'. Expected one of "
                            f'{list(_CAPABILITIES)}.')
        if value is not None:
            setattr(backend, key, _CAPABILITIES[key](value))
    return backend


def _infer_capabilities(executor) -> dict:
    """Guess the capability flags for a user-supplied executor.

    Only for executors no backend claimed through `adopt` - anything a backend
    recognises describes itself, and does so from the real thing rather than
    from a guess. For the rest we deliberately guess the *safe* values:
    `isolated=False` means we never turn a caller's `inplace=False` into an
    in-place operation, which at worst costs a copy.
    """
    if isinstance(executor, cf.ThreadPoolExecutor):
        return dict(isolated=False, pickles_by_value=True)
    if isinstance(executor, cf.ProcessPoolExecutor):
        return dict(isolated=True, pickles_by_value=False)

    mod = type(executor).__module__ or ''
    if mod.startswith('joblib') or 'loky' in mod:
        return dict(isolated=True, pickles_by_value=True)

    logger.debug(f'Unrecognised executor {type(executor).__name__}; assuming '
                 'it shares memory with the parent. Pass `isolated=True` to '
                 '`navis.set_parallel_backend` if it does not.')
    return dict(isolated=False, pickles_by_value=False)


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


def adopt_object(obj, **overrides) -> Optional[ParallelBackend]:
    """Wrap a user-supplied executor-like object in a backend.

    Asks each registered backend whether it recognises `obj` (see
    :meth:`ParallelBackend.adopt`) and falls back to the generic wrapper for
    anything implementing `concurrent.futures.Executor`. Returns None if
    nothing claims it.
    """
    for be in list(_BACKENDS.values()):
        adopted = be.adopt(obj, **overrides)
        if adopted is not None:
            return adopted

    if isinstance(obj, cf.Executor):
        return WrappedExecutorBackend(obj, **overrides)

    return None


def _adopt_or_raise(obj, **overrides) -> ParallelBackend:
    """`adopt_object`, but say what was expected instead of returning None."""
    adopted = adopt_object(obj, **overrides)
    if adopted is None:
        raise TypeError(
            f'Cannot run navis on a {type(obj).__name__}. Expected the name of '
            f'a backend ({list_backends()}), a ParallelBackend, a '
            '`concurrent.futures.Executor`, or a scheduler object one of the '
            'installed backends recognises (e.g. a `dask.distributed.Client` '
            'or a `submitit.AutoExecutor`).'
        )
    return adopted


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

    if not isinstance(backend, (str, type(None))):
        return _adopt_or_raise(backend)

    if backend in (None, 'auto'):
        rejected = {}
        # `auto_select` before `available()`: the latter is a filesystem lookup
        # and there is no point paying it for a backend we would skip anyway.
        for be in sorted(_BACKENDS.values(), key=lambda b: -b.priority):
            if not be.auto_select or not be.available():
                continue
            reasons = be.unsupported(**requirements)
            if not reasons:
                return be
            rejected[be.name] = reasons

        # Nothing available fits. Say what would have, and how to get it.
        missing = [b for b in _BACKENDS.values()
                   if not b.available() and b.auto_select
                   and not b.unsupported(**requirements)]
        msg = 'No available parallel backend can run this work.'
        if missing:
            names = [b.name for b in missing]
            install = ' '.join(b.requires or b.name for b in missing)
            msg += (f' Backend(s) {names} could, but are not installed'
                    f' (`pip install -U {install}`).')
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
            f"dependencies are not installed "
            f"(`pip install -U {be.requires or be.name}`)."
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
                         pickles_by_value=None, chunks_per_worker=None,
                         marshals_exceptions=None):
    """Set where `parallel=True` runs its work.

    Parameters
    ----------
    backend :   str | ParallelBackend | executor, optional
                One of the registered backend names (see
                [`navis.list_parallel_backends`][]), `"auto"` to pick the best
                available, or a backend instance. `None` leaves the setting
                unchanged.

                It also takes a scheduler object directly - any
                `concurrent.futures.Executor`, a `dask.distributed.Client` or
                a `submitit.Executor` - which is how you point navis at a
                cluster. Configure it with its own library's API and hand it
                over; navis deliberately has no `slurm_partition`-style
                parameters of its own.
    n_workers : int, optional
                Default number of workers. `None` leaves it unchanged.
    isolated :  bool, optional
                Only for a bare executor: whether its workers have their own
                address space. Inferred where we recognise the executor,
                otherwise assumed False (the safe guess).
    pickles_by_value : bool, optional
                Only for a bare executor: whether it can ship lambdas and
                closures (i.e. uses `dill`/`cloudpickle` rather than `pickle`).
    chunks_per_worker : int, optional
                Only for a bare executor: how many units of work to aim for per
                worker. `None` (the default) sends one neuron per unit, which is
                right on one machine and wasteful across a network - recognised
                remote executors are bundled automatically, so this is for
                tuning that or for an executor we don't recognise.
    marshals_exceptions : bool, optional
                Only for a bare executor: whether an exception raised in a
                worker arrives here as itself. Set False for a transport that
                only reports *that* the work failed (submitit does this), and
                navis will bring failures back as data instead so you still
                catch your own exception.

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
    overrides = dict(isolated=isolated, pickles_by_value=pickles_by_value,
                     chunks_per_worker=chunks_per_worker,
                     marshals_exceptions=marshals_exceptions)

    # The capability flags describe a specific executor, so they only mean
    # something on the branch that adopts one.
    describes_executor = (backend is not None
                          and not isinstance(backend, (str, ParallelBackend)))
    if not describes_executor and any(v is not None for v in overrides.values()):
        raise TypeError(f'{list(_CAPABILITIES)} describe an executor - pass '
                        'them together with one, not with a backend name.')

    if describes_executor:
        backend = _adopt_or_raise(backend, **overrides)
    elif isinstance(backend, str) and backend != 'auto':
        # Fail loudly here rather than at the next `parallel=True`
        be = get_backend(backend)
        reasons = be.unsupported()
        if reasons:
            raise ValueError(f"Parallel backend '{be.name}' cannot be used as "
                             f"it stands: {'; '.join(reasons)}")

    return _BackendSetter(backend, n_workers)
