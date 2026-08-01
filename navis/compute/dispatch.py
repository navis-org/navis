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

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

from .. import config

logger = config.get_logger(__name__)

__all__ = ['map_tasks', 'default_n_workers', 'FailedRun']


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


class _FailedTask:
    """A failure on its way back from a worker.

    Deliberately *not* a `FailedRun`: that one keeps `args`, i.e. the neuron,
    so returning it would ship the whole payload back across the wire for every
    failure. The parent already has the args and rebuilds the `FailedRun`.
    """

    __slots__ = ('exception',)

    def __init__(self, exception):
        self.exception = exception


def run_chunk(payload):
    """Run one chunk of tasks. Runs in the worker.

    Module level so it pickles by reference - a closure or lambda here would
    force every backend to serialise by value.
    """
    index, context, omit_failures, tasks = payload

    if context is not None:
        context.apply()

    results = []
    for func, args, kwargs in tasks:
        try:
            results.append(func(*args, **kwargs))
        except BaseException as e:
            if not omit_failures:
                raise
            results.append(_FailedTask(e))

    return index, results


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
    """
    # Unwrap bound methods: `n.resample` pickles fine if the neuron does
    func = getattr(func, '__func__', func)

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
    if not len(tasks):
        return []

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
    payloads = [(i, context, omit_failures, c) for i, c in enumerate(chunks)]

    # Results come back in completion order - that's the only contract every
    # transport can honour - so we put them back using the index we sent.
    out: List[Optional[list]] = [None] * len(chunks)
    with config.tqdm(total=len(tasks), desc=desc, disable=disable,
                     leave=config.pbar_leave) as pbar:
        try:
            for index, results in backend.map(run_chunk, payloads,
                                              n_workers=n_workers):
                out[index] = results
                pbar.update(len(chunks[index]))
        except BaseException as e:
            hint = _serialisation_hint(e, backend)
            if hint is None:
                raise
            raise RuntimeError(hint) from e

    # Rebuild the full FailedRun here, where the args already live
    res = []
    for chunk, results in zip(chunks, out):
        for (func, args, kwargs), r in zip(chunk, results):
            if isinstance(r, _FailedTask):
                res.append(FailedRun(func, args, kwargs, r.exception))
            else:
                res.append(r)

    return res
