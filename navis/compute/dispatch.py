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

"""The backend-agnostic half of parallel processing.

This module knows how to turn a list of `(func, args, kwargs)` tasks into
chunks, hand them to *some* backend, and put the results back in order. It
knows nothing about which backend, and backends know nothing about neurons -
that separation is what lets a cluster executor drop in later.

Important: this module must not import from `navis.core` or `navis.utils`. It is
imported from `navis/core/core_utils.py` (and eventually from the decorators),
and the note at the top of `navis/utils/decorators.py` applies here too - extra
imports are an easy way to break pickling of the functions we ship to workers.

"""

import os
import sys
import pickle
import functools
import traceback

from dataclasses import dataclass
from typing import (Any, Callable, Iterator, List, NamedTuple, Optional,
                    Sequence, Tuple)

from .. import config

logger = config.get_logger(__name__)

__all__ = ['map_tasks', 'imap_tasks', 'default_n_workers', 'FailedRun']


def default_n_workers() -> int:
    """Number of workers to use when the caller didn't say.

    `navis.config.default_n_workers` if set, else half the available cores.
    Never zero: `os.cpu_count()` returns None on platforms that can't tell us,
    which would otherwise produce a `TypeError`.
    """
    return config.default_n_workers or max(1, (os.cpu_count() or 1) // 2)


# --------------------------------------------------------------------------- #
# Worker context
# --------------------------------------------------------------------------- #
# `navis.config` holds mutable module-level settings. A *forked* worker inherits
# the parent's mutations for free; a spawned worker (the default on macOS and
# Windows, and on Linux from Python 3.14) re-imports navis and gets the
# defaults back. Without this, `navis.set_pbars(hide=True)` and friends would
# silently stop applying the moment the backend changed.

#: Callables run in a worker before each chunk, after the config has been
#: applied. Use this for state that lives outside `navis.config` - e.g.
#: `navis.patch_cloudvolume()`. They must be picklable (importable by name) and
#: cheap enough to re-run, since there is no uniform "once per worker" hook
#: across the executors we support.
worker_init_hooks: List[Callable[[], None]] = []


@dataclass(frozen=True)
class WorkerContext:
    """Parent state that a worker in a fresh interpreter cannot see."""

    log_level: int
    settings: Tuple[Tuple[str, Any], ...] = ()
    hooks: Tuple[Callable[[], None], ...] = ()

    @classmethod
    def snapshot(cls) -> 'WorkerContext':
        # `hasattr` so this survives a setting being renamed or dropped
        return cls(
            log_level=config.logger.getEffectiveLevel(),
            settings=tuple((k, getattr(config, k)) for k in config.WORKER_SETTINGS
                           if hasattr(config, k)),
            hooks=tuple(worker_init_hooks),
        )

    def apply(self) -> None:
        """Re-establish the parent's state. Runs inside the worker."""
        # Only if it differs: `setLevel` invalidates the logging module's level
        # cache for *every* logger, and this runs once per chunk.
        if config.logger.getEffectiveLevel() != self.log_level:
            config.logger.setLevel(self.log_level)
        for key, value in self.settings:
            setattr(config, key, value)
        for hook in self.hooks:
            hook()


# --------------------------------------------------------------------------- #
# The unit of work
# --------------------------------------------------------------------------- #
class FailedRun:
    """Class representing a failed run."""

    def __init__(self, func, args, kwargs, exception='NA'):
        self.args = args
        self.func = func
        self.kwargs = kwargs
        self.exception = exception

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (f'Failed run(function={self.func}, args={self.args}, '
                f'kwargs={self.kwargs}, exception={self.exception})')


class RemoteTraceback(Exception):
    """Carries a worker-side traceback into the parent's.

    A traceback object cannot be pickled, so an exception that has travelled
    arrives with `__traceback__` stripped and would otherwise point only at the
    dispatcher. Chaining this as the `__cause__` puts the worker's frames back
    in the printed traceback - the same trick `concurrent.futures` uses.
    """


class _FailedTask:
    """A failure on its way back from a worker.

    Deliberately *not* a `FailedRun`: that one keeps `args`, i.e. the neuron,
    so returning it would ship the whole payload back across the wire for every
    failure. The parent already has the args and rebuilds the `FailedRun`.
    """

    __slots__ = ('exception', 'traceback')

    def __init__(self, exception, traceback=None):
        self.exception = exception
        self.traceback = traceback

    def reraise(self):
        """Raise the original exception, worker frames and all."""
        if self.traceback:
            raise self.exception from RemoteTraceback(f'\n{self.traceback}')
        raise self.exception


class Chunk(NamedTuple):
    """One unit of work, as it travels to a worker.

    Module level (and a plain tuple underneath) so it pickles by reference on
    backends that serialise with stdlib `pickle`.
    """

    #: Position in the input, so the parent can restore order from results
    #: that arrive in completion order.
    index: int
    #: Parent state a fresh interpreter cannot see. None for in-process
    #: backends, where applying it would clobber the parent's own config.
    context: Optional[WorkerContext]
    #: Hand failures back as data instead of raising them.
    catch: bool
    #: Also pay for a formatted traceback, because the parent is going to
    #: re-raise and would otherwise lose the worker's frames.
    want_traceback: bool
    tasks: Sequence[Tuple[Callable, Sequence, dict]]


def run_chunk(chunk: Chunk):
    """Run one chunk of tasks. Runs in the worker.

    Module level so it pickles by reference - a closure or lambda here would
    force every backend to serialise by value.
    """
    if chunk.context is not None:
        chunk.context.apply()

    results = []
    for func, args, kwargs in chunk.tasks:
        try:
            results.append(func(*args, **kwargs))
        except BaseException as e:
            if not chunk.catch:
                raise
            results.append(_FailedTask(
                e, traceback.format_exc() if chunk.want_traceback else None))

    return chunk.index, results


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def picklable_by_reference(func) -> bool:
    """Whether `func` can be shipped by plain `pickle`.

    `pickle` stores a function as `module.qualname` and, on unpickling, checks
    that the name still resolves to the *same* object. Lambdas, closures and
    anything defined in a notebook fail that check.

    Deliberately conservative: a wrong `False` only means we pick a more
    capable backend, whereas a wrong `True` surfaces as a `PicklingError`.

    A callable object can answer for itself by exposing a
    `__picklable_by_reference__` attribute. That is the escape hatch for a
    *composite* callable: one that is not importable by name itself, but
    travels fine as long as each function it is built from does.
    """
    # Peel the wrappers first, then ask whatever is actually inside.

    # `functools.partial` reduces to (func, args, kwargs), so it travels
    # exactly as far as the function it wraps - and it proxies no attributes,
    # so asking it directly would miss a declaration underneath. Unpicklable
    # *arguments* are a separate matter and surface via `_serialisation_hint`,
    # same as for any other task.
    while isinstance(func, functools.partial):
        func = func.func

    # Unwrap bound methods: `n.resample` pickles fine if the neuron does
    func = getattr(func, '__func__', func)

    # Asked before the name check, which rejects *any* instance: an instance
    # has no `__qualname__` of its own (that lives on the class).
    declared = getattr(func, '__picklable_by_reference__', None)
    if declared is not None:
        return bool(declared)

    module = getattr(func, '__module__', None)
    qualname = getattr(func, '__qualname__', None)
    if not module or not qualname:
        return False

    # `<lambda>`, `<locals>` - never importable
    if '<' in qualname:
        return False

    # A REPL/notebook `__main__` has no file to re-import
    if module == '__main__' and not hasattr(sys.modules.get('__main__'), '__file__'):
        return False

    obj = sys.modules.get(module)
    for part in qualname.split('.'):
        obj = getattr(obj, part, None)
        if obj is None:
            return False

    return obj is func or getattr(obj, '__func__', None) is func


#: Exception types a failed pickle can surface as, depending on where in the
#: machinery it blew up. `AttributeError`/`TypeError` are in here because that
#: is what the stdlib raises for a local object or an unpicklable type - hence
#: the message check below, so a user's own error of those types is left alone.
_PICKLE_ERRORS = (pickle.PickleError, AttributeError, TypeError)
_PICKLE_MARKERS = ('pickle', 'not the same object', 'local object')


def _serialisation_hint(exc, backend) -> Optional[str]:
    """Return an actionable message if `exc` looks like a pickling failure.

    Returns None for anything else, so that a `TypeError` raised by the user's
    own function is re-raised untouched rather than blamed on the backend.
    """
    if not isinstance(exc, _PICKLE_ERRORS):
        return None
    if not any(m in str(exc).lower() for m in _PICKLE_MARKERS):
        return None

    return (
        f"The '{backend.name}' backend could not serialise this work: {exc}\n\n"
        "It can only send functions that are importable by name. Lambdas, "
        "closures and functions defined in a notebook need a backend that "
        "serialises by value:\n"
        "    navis.set_parallel_backend('joblib')   # or 'pathos'\n"
        "Note this also applies to arguments - e.g. a lambda passed as a "
        "keyword argument. Alternatively, run serially with `parallel=False`."
    )


# --------------------------------------------------------------------------- #
# The dispatcher
# --------------------------------------------------------------------------- #
def _iter_completed(backend, payloads, n_workers) -> Iterator:
    """Yield `(chunk index, results)` as units of work come back.

    A thin wrapper whose only job is to translate a pickling failure into
    something actionable - and to do so *without* wrapping the caller's own
    loop body, which is free to re-raise an exception a worker sent home.
    """
    try:
        yield from backend.map(run_chunk, payloads, n_workers=n_workers)
    except BaseException as e:
        hint = _serialisation_hint(e, backend)
        if hint is None:
            raise
        raise RuntimeError(hint) from e


def imap_tasks(tasks: Sequence[Tuple[Callable, Sequence, dict]],
               *,
               backend,
               n_workers: Optional[int] = None,
               chunksize: Optional[int] = None,
               omit_failures: bool = False,
               desc: Optional[str] = None,
               disable: bool = False,
               size_hint: Optional[Callable[[], float]] = None) -> Iterator:
    """Run `tasks` on `backend`, yielding `(index, result)` as they land.

    Same contract as :func:`map_tasks` except for the order: results arrive as
    they complete, each tagged with the position of its task in `tasks`. Use
    this when a result can be consumed - written into an output array, say - as
    soon as it arrives, rather than held until the last one lands. For a big
    enough job that is the difference between one copy of the output and two.

    See :func:`map_tasks` for the parameters.
    """
    if not len(tasks):
        return

    n_workers = n_workers or default_n_workers()

    cs = backend.chunksize(len(tasks), n_workers,
                           requested=chunksize, size_hint=size_hint)
    cs = max(1, int(cs))

    chunks = [tasks[i:i + cs] for i in range(0, len(tasks), cs)]
    if cs > 1:
        logger.debug(f"'{backend.name}': {len(tasks)} tasks in {len(chunks)} "
                     f'units of up to {cs}.')

    # Only ship the context where it's needed: applying it in-process would
    # clobber the parent's own config.
    context = WorkerContext.snapshot() if backend.isolated else None

    # A transport that can't carry an exception home intact (submitit turns
    # every one of them into a generic "job failed") would otherwise make the
    # error a caller sees depend on which backend is configured. Have the
    # worker hand failures back as data instead and raise them here.
    reraises_here = not backend.marshals_exceptions and not omit_failures
    payloads = [Chunk(index=i, context=context,
                      catch=omit_failures or reraises_here,
                      want_traceback=reraises_here, tasks=c)
                for i, c in enumerate(chunks)]

    with config.tqdm(total=len(tasks), desc=desc, disable=disable,
                     leave=config.pbar_leave) as pbar:
        for index, results in _iter_completed(backend, payloads, n_workers):
            pbar.update(len(chunks[index]))
            for offset, (task, result) in enumerate(zip(chunks[index], results)):
                if isinstance(result, _FailedTask):
                    if not omit_failures:
                        result.reraise()
                    # Rebuild the full FailedRun here, where the args still live
                    result = FailedRun(*task, exception=result.exception)
                yield index * cs + offset, result


def map_tasks(tasks: Sequence[Tuple[Callable, Sequence, dict]],
              *,
              backend,
              n_workers: Optional[int] = None,
              chunksize: Optional[int] = None,
              omit_failures: bool = False,
              desc: Optional[str] = None,
              disable: bool = False,
              size_hint: Optional[Callable[[], float]] = None) -> list:
    """Run `tasks` on `backend` and return the results in *input* order.

    Parameters
    ----------
    tasks :         list of (func, args, kwargs)
    backend :       ParallelBackend
    n_workers :     int, optional
                    Hint for how many workers to use. A backend wrapping a
                    user-configured executor may ignore it.
    chunksize :     int, optional
                    Tasks per unit of work. `None` lets the backend decide.
    omit_failures : bool
                    If True, failures come back as `FailedRun` instead of
                    propagating.
    desc :          str, optional
                    Progress bar description.
    disable :       bool
                    Whether to hide the progress bar.
    size_hint :     callable, optional
                    Returns the estimated size of one task in bytes. Only
                    called if the backend's chunking policy needs it.

    Returns
    -------
    list
                    One entry per task, in the order the tasks were given.

    """
    # Results come back in completion order - that's the only contract every
    # transport can honour - so we put them back using the index they carry.
    out: List[Any] = [None] * len(tasks)
    for index, result in imap_tasks(tasks, backend=backend, n_workers=n_workers,
                                    chunksize=chunksize, desc=desc,
                                    disable=disable, size_hint=size_hint,
                                    omit_failures=omit_failures):
        out[index] = result
    return out
