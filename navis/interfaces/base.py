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

"""Shared machinery for the remote data-source interfaces.

Every interface in this package does the same handful of things - fan a fetch
out over a thread pool, talk HTTP, lean on an optional third-party client - and
until this module existed each of them did so in its own way. The differences
were never deliberate: fetches came back in nondeterministic order, per-neuron
failures were reported with a bare `print` and then swallowed, and whether a
missing dependency failed at import or at first use depended on which interface
you happened to be importing.

The pieces here are deliberately small and free-standing. They are for the
interfaces to *use*, not to inherit from.
"""

import importlib

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, partial

from .. import config
from ..utils.http import get_session  # noqa: F401 - re-exported for interfaces

__all__ = [
    "FetchError",
    "cached",
    "clear_cache",
    "fetch_parallel",
    "get_session",
    "optional_import",
    "register_cache",
    "resolve_errors",
]

logger = config.get_logger(__name__)

#: What to do about a failed item when the caller did not say. Strict mode
#: (`navis.config.strict`) flips this to "raise".
DEFAULT_ERRORS = "log"

ERROR_POLICIES = ("raise", "log", "ignore")


class FetchError(Exception):
    """Raised when fetching an individual item from a remote source failed.

    The underlying error - an `HTTPError`, a missing-ID `ValueError`, whatever
    the source raised - is always attached as `__cause__`:

        try:
            ...
        except FetchError as e:
            original = e.__cause__

    """


def resolve_errors(errors=None):
    """Resolve an error policy, applying the strict-mode default.

    Parameters
    ----------
    errors :    "raise" | "log" | "ignore" | None
                `None` means "use the default", which is "log" normally and
                "raise" under `navis.config.strict`.

    Returns
    -------
    str
                One of `ERROR_POLICIES`.

    """
    if errors is None:
        return "raise" if config.strict else DEFAULT_ERRORS

    if errors not in ERROR_POLICIES:
        raise ValueError(
            f'`errors` must be one of {ERROR_POLICIES}, got "{errors}"'
        )

    return errors


def _apply_error_policy(label, exc, errors):
    """Apply `errors` to a failed item.

    Returns the exception the caller should raise, or `None` if the failure is
    to be tolerated. Returning rather than raising keeps this frame out of the
    traceback the user ends up reading.
    """
    if errors == "raise":
        return FetchError(f"Failed to fetch {label}: {exc}")
    if errors == "log":
        logger.error(f"Failed to fetch {label}: {exc}")
    return None


def fetch_parallel(
    func,
    items,
    *,
    labels=None,
    errors=None,
    parallel=True,
    max_threads=4,
    desc="Fetching",
    progress=True,
    initializer=None,
    initargs=(),
    **kwargs,
):
    """Map `func` over `items` in a thread pool.

    The one implementation of "fetch a bunch of things from a server" for all of
    `navis.interfaces`. Three properties are worth knowing about:

    1. **Results come back in the order of `items`**, regardless of the order
       the requests happened to finish in. Callers therefore do not need to
       re-sort afterwards to match their input.
    2. **A failed item is `None` in the output** (unless `errors="raise"`, in
       which case it raises). So the usual call site is
       `[n for n in results if n is not None]` - or `NeuronList` of the same,
       which does the filtering itself.
    3. **Progress is reported as requests finish**, not in input order, so the
       bar stays responsive.

    Parameters
    ----------
    func :          callable
                    Called as `func(item, **kwargs)` for each item.
    items :         iterable
                    What to fetch. Consumed eagerly (generators are fine).
    labels :        iterable, optional
                    Used to name items in error messages - typically the IDs.
                    Must match `items` in length. Defaults to the items.
    errors :        "raise" | "log" | "ignore", optional
                    What to do when an item fails:

                      - "raise" re-raises as a `FetchError`, with the original
                        error as `__cause__`
                      - "log" logs an error and carries on
                      - "ignore" carries on silently

                    Defaults to "log", or to "raise" under `navis.config.strict`.
    parallel :      bool
                    Whether to use threads at all. `False` is equivalent to
                    `max_threads=1`.
    max_threads :   int
                    Max number of threads. Be a good citizen: most of the
                    services on the other end are public academic servers.
    desc :          str
                    Label for the progress bar.
    progress :      bool
                    Set to False to suppress the progress bar entirely - for
                    when the caller runs one of its own.
    initializer :   callable, optional
                    Run once per worker thread before any item, as
                    `initializer(*initargs)`. Also run once, on the calling
                    thread, when running serially - so that thread-locals it
                    sets up are in place either way.
    initargs :      tuple
    **kwargs
                    Passed through to every `func` call.

    Returns
    -------
    list
                    Same length and order as `items`, with `None` for failures.

    Examples
    --------
    >>> from navis.interfaces.base import fetch_parallel
    >>> fetch_parallel(lambda x: x * 2, [1, 2, 3], max_threads=2)
    [2, 4, 6]

    Failures are holes, not exceptions:

    >>> def flaky(x):
    ...     if x == 2:
    ...         raise ValueError("nope")
    ...     return x
    >>> fetch_parallel(flaky, [1, 2, 3], errors="ignore")
    [1, None, 3]

    """
    items = list(items)

    if not items:
        return []

    errors = resolve_errors(errors)

    if labels is None:
        labels = items
    else:
        labels = list(labels)
        if len(labels) != len(items):
            raise ValueError(
                f"Got {len(labels)} labels for {len(items)} items - must match."
            )

    n_threads = 1 if not parallel else max(1, int(max_threads))

    results = [None] * len(items)

    prog = partial(
        config.tqdm,
        desc=desc,
        total=len(items),
        leave=config.pbar_leave,
        # A progress bar over a single item is just noise.
        disable=not progress or len(items) == 1 or config.pbar_hide,
    )

    if n_threads == 1:
        if initializer is not None:
            initializer(*initargs)

        with prog() as pbar:
            for i, item in enumerate(items):
                try:
                    results[i] = func(item, **kwargs)
                except Exception as exc:
                    err = _apply_error_policy(labels[i], exc, errors)
                    if err is not None:
                        raise err from exc
                finally:
                    pbar.update(1)

        return results

    with ThreadPoolExecutor(
        max_workers=n_threads, initializer=initializer, initargs=initargs
    ) as executor:
        # Keyed by future so we can put each result back at its input index.
        futures = {
            executor.submit(func, item, **kwargs): i for i, item in enumerate(items)
        }

        with prog() as pbar:
            for f in as_completed(futures):
                i = futures[f]
                pbar.update(1)
                try:
                    results[i] = f.result()
                except Exception as exc:
                    err = _apply_error_policy(labels[i], exc, errors)
                    if err is not None:
                        raise err from exc

    return results


#: `cache_clear` callables for every registered cache. Kept so that there is one
#: place to empty them all from - the interfaces memoize datastack listings,
#: clients and metadata, and until this existed each had to be hunted down by
#: name (or was simply unreachable).
_CACHE_CLEARERS = []


def cached(func=None, *, maxsize=None):
    """`functools.lru_cache`, registered with `clear_cache`.

    Use in place of a bare `@lru_cache` anywhere in `navis.interfaces`.

    Parameters
    ----------
    maxsize :   int, optional
                As for `functools.lru_cache`. `None` (the default) is unbounded.

    Examples
    --------
    >>> from navis.interfaces.base import cached, clear_cache
    >>> @cached
    ... def expensive(x):
    ...     return x * 2
    >>> expensive(21)
    42
    >>> clear_cache()

    """

    def wrap(f):
        wrapped = lru_cache(maxsize)(f)
        _CACHE_CLEARERS.append(wrapped.cache_clear)
        return wrapped

    return wrap if func is None else wrap(func)


def register_cache(clear):
    """Register a cache that navis does not own the decorator for.

    For caches that are not `lru_cache`s - a plain dict, say. `clear` is any
    zero-argument callable that empties it.

    Returns `clear`, so it can be used as a decorator on the clearing function.
    """
    _CACHE_CLEARERS.append(clear)
    return clear


def clear_cache():
    """Empty every cache held by the `navis.interfaces` modules.

    Covers memoized datastack listings, clients and downloaded metadata. Caches
    refill on demand, so this is always safe to call - reach for it when a
    remote source has changed under you, or to release memory.

    Examples
    --------
    >>> import navis.interfaces as interfaces
    >>> interfaces.clear_cache()

    """
    for clear in _CACHE_CLEARERS:
        clear()


class _MissingModule:
    """Stand-in for an optional dependency that is not installed.

    Exists so that importing a navis interface never fails on a dependency the
    user may not need for what they are about to do - and, more importantly, so
    that when they *do* need it the error says how to install it instead of
    surfacing as an `AttributeError` on `None` half a stack frame later.

    Falsy, so that the `if not <module>:` guards the interfaces already use keep
    working.
    """

    def __init__(self, name, message):
        self._name = name
        self._message = message

    def _raise(self):
        raise ModuleNotFoundError(self._message, name=self._name)

    # Our own state, which `__getattr__` must answer for rather than recurse on.
    _OWN = ("_name", "_message")

    def __getattr__(self, key):
        # Two kinds of lookup must NOT produce the install message:
        #
        # - dunders, because `copy` and pickle probe for `__deepcopy__`,
        #   `__setstate__` & co and only tolerate an AttributeError. Otherwise a
        #   `deepcopy` of an object that merely *holds* one of these would blow
        #   up with "please pip install ...".
        # - our own attributes, because `copy` reconstructs instances *without*
        #   calling `__init__`: `_name`/`_message` are then briefly absent from
        #   `__dict__`, land here, and recurse until the stack gives out.
        if key in self._OWN or (key.startswith("__") and key.endswith("__")):
            raise AttributeError(key)
        self._raise()

    def __call__(self, *args, **kwargs):
        self._raise()

    def __bool__(self):
        return False

    def __repr__(self):
        return f"<not installed: {self._name}>"


def optional_import(name, *, pip=None, hint=None):
    """Import an optional dependency, deferring failure to first use.

    Parameters
    ----------
    name :  str
            Module to import, e.g. "caveclient" or "allensdk.core.swc".
    pip :   str, optional
            What to tell the user to `pip install`. Defaults to the top-level
            package of `name`. Pass e.g. `"allensdk --no-deps"` where the plain
            name is not enough.
    hint :  str, optional
            Extra line appended to the error message.

    Returns
    -------
    module
            The module, or a stand-in that raises `ModuleNotFoundError` with an
            actionable message on first use.

    Examples
    --------
    >>> from navis.interfaces.base import optional_import
    >>> np = optional_import("numpy")
    >>> nope = optional_import("not_a_real_package")
    >>> bool(nope)
    False

    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        top = name.split(".")[0]
        msg = (
            f"`{name}` is required for this but is not installed. "
            f"Please install it using pip:\n\n"
            f"    pip install {pip or top} -U\n"
        )
        if hint:
            msg += f"\n{hint}\n"
        # A missing dependency *of* the package we asked for reads as "the
        # package is broken", which is a different problem from "not installed"
        # and deserves to be said out loud rather than papered over.
        if e.name and e.name.split(".")[0] != top:
            msg += (
                f"\n(`{top}` itself appears to be installed - the import failed "
                f"on `{e.name}`: {e})\n"
            )
        return _MissingModule(name, msg)
