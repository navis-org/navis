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

"""Machinery for names that navis has renamed.

Four kinds of rename live here, each served by one helper so that the shims,
the docs and the tests cannot drift apart:

- **classes** - navis 2.0 renamed the neuron classes to `Skeleton`, `Mesh` and
  `Voxels` (`DEPRECATED_NEURON_CLASSES`);
- **top-level functions** - 2.0 settled on "component" as the one word for a
  connected piece of a neuron (`DEPRECATED_FUNCTIONS`);
- **properties** - the same, on `Skeleton` (`DEPRECATED_PROPERTIES`);
- **keyword arguments** - one name per concept: `max_dist` for a distance cap,
  `min_size` for a count, `min_length` for a distance (`renamed_kwargs`).

The first two resolve in the top-level `navis` namespace via PEP 562 module
`__getattr__` - and only there, because a plain module global is found by
ordinary attribute lookup before `__getattr__` is ever consulted, so the warning
would never fire.

Elsewhere the old class names *are* plain aliases (see the tail of
`navis/core/skeleton.py` & co), because `pickle` resolves classes by their
defining module and must find them without a warning. Aliases either way, so
`isinstance` and subclassing are unaffected by the rename.
"""

import functools
import sys
import warnings

__all__ = [
    "DEPRECATED_NEURON_CLASSES",
    "DEPRECATED_FUNCTIONS",
    "DEPRECATED_KWARGS",
    "DEPRECATED_PROPERTIES",
    "DEPRECATED_TOP_LEVEL",
    "deprecated_getattr",
    "deprecated_property",
    "caller_stacklevel",
    "renamed_kwargs",
    "reset_deprecation_warnings",
    "warn_renamed",
]

#: Old name -> new name. `Dotprops` was not renamed.
DEPRECATED_NEURON_CLASSES = {
    "TreeNeuron": "Skeleton",
    "MeshNeuron": "Mesh",
    "VoxelNeuron": "Voxels",
}

#: Old name -> new name for the top-level functions 2.0 renamed, as part of
#: settling on "component" as the one word for a connected piece of a neuron.
DEPRECATED_FUNCTIONS = {
    "break_fragments": "split_components",
    "split_into_fragments": "split_neurites",
}

#: Old name -> new name for the renamed neuron properties, per class. Unlike the
#: two above these cannot be served by a module `__getattr__`; the classes build
#: `deprecated_property` shims off this table (see `navis/core/skeleton.py`).
DEPRECATED_PROPERTIES = {
    "Skeleton": {
        "subtrees": "connected_components()",
        "n_trees": "n_components",
        "n_skeletons": "n_components",
        "is_tree": "is_acyclic",
    },
}

#: Qualified function name -> `{old kwarg: new kwarg}`, filled in by
#: `renamed_kwargs` as each decorated function is defined. Unlike the three
#: tables above this cannot be written by hand: the decorator has to sit at each
#: `def`, in eight modules. Registering from there keeps the docs and the tests
#: derivable from the shims rather than hand-copied alongside them.
DEPRECATED_KWARGS = {}

#: Everything the `navis` namespace serves under an old name.
DEPRECATED_TOP_LEVEL = {**DEPRECATED_NEURON_CLASSES, **DEPRECATED_FUNCTIONS}

# Names already warned about, so that a loop over a NeuronList doesn't produce
# one warning per neuron.
_warned = set()


#: Top-level package name, for `caller_stacklevel`.
_ROOT_PACKAGE = __name__.partition(".")[0]


def caller_stacklevel():
    """`stacklevel` of the first frame outside navis, i.e. the user's.

    Counting frames by hand is what `stacklevel=` normally asks for, but the
    count depends on how many of navis' own decorators a function happens to
    carry - so adding one silently re-points every warning underneath it at
    navis' own source. Python's default filters only surface a
    `DeprecationWarning` attributed to `__main__`, so a misblamed warning is one
    nobody ever sees.

    "Inside navis" is decided by the frame's module rather than by its filename:
    a path prefix is an artefact of one install shape, and would take a sibling
    distribution (`navis-fastcore`) for our own and lose track of us entirely
    under zipimport or `exec(compile(...))`.

    N.B. Python 3.12 has `warnings.warn(skip_file_prefixes=...)` for this;
    `setup.py` still allows 3.10.
    """
    # Frame 1 is our caller, which is the frame `warnings.warn` counts as
    # level 1 - so that is where the walk starts.
    frame, level = sys._getframe(1), 1
    while frame is not None and (
        frame.f_globals.get("__name__", "").partition(".")[0] == _ROOT_PACKAGE
    ):
        frame, level = frame.f_back, level + 1
    return level


def warn_renamed(old, new, stacklevel=None):
    """Warn once per session that `old` has been renamed to `new`.

    `stacklevel` defaults to whatever points at the caller - see
    `caller_stacklevel`.
    """
    if old in _warned:
        return

    warnings.warn(
        f"`{old}` is deprecated and will be removed in a future version - "
        f"use `{new}` instead.",
        DeprecationWarning,
        stacklevel=caller_stacklevel() if stacklevel is None else stacklevel,
    )
    # Recorded only once the warning is through: under `-W
    # error::DeprecationWarning` the line above raises, and marking the name
    # first would let every later access pass silently.
    _warned.add(old)


def deprecated_property(cls_name, old, fget=None):
    """Build a read-only property that warns once, then answers the old question.

    The new name comes from `DEPRECATED_PROPERTIES[cls_name][old]`, so a shim
    cannot drift from the table the docs and tests read.

    By default the property simply forwards to that new name. `fget` is for the
    renames that also changed what comes back - `subtrees` hands out node IDs
    where `connected_components` hands out labels - and spells out the *old*
    answer.

    Parameters
    ----------
    cls_name :  str
                Class the property lives on, e.g. `"Skeleton"`.
    old :       str
                Old property name, e.g. `"subtrees"`.
    fget :      callable, optional
                Takes the neuron, returns what the old property returned.
                Defaults to reading the new name off the neuron.

    """
    new = DEPRECATED_PROPERTIES[cls_name][old]
    # The table spells methods with their call parens, for the message
    attr = new.rstrip("()")

    if fget is None:
        def fget(self, _attr=attr):
            return getattr(self, _attr)

    def getter(self):
        warn_renamed(f"{cls_name}.{old}", f"{cls_name}.{new}")
        return fget(self)

    getter.__name__ = old
    return property(getter, doc=f"Deprecated. Use `{cls_name}.{new}` instead.")


def renamed_kwargs(**renames):
    """Decorator: keep accepting a function's old keyword argument names.

    Each old name warns once per session and is then forwarded to the new one.
    Passing both is an error rather than a coin toss over which wins.

    Apply this **innermost** - directly above the `def`, below
    `@map_neuronlist` and friends. `map_neuronlist` dispatches to worker
    processes by pickling *its own* wrapper, and pickle resolves a function by
    looking its qualified name up in the module and checking the result is the
    same object; a decorator above it takes over that name and the check fails
    with a `PicklingError`. Nothing in navis' decorators binds the full keyword
    set, so translating from underneath them is safe.

    The warning still blames the caller: `warn_renamed` finds the first frame
    outside navis rather than counting a fixed number of them.

    Note the wrapper keeps the *new* signature (via `functools.wraps`), so
    `inspect.signature` and anything validating against it - e.g.
    [`navis.Pipeline`][] - still see only the current names. This shim is for
    direct calls.

    Parameters
    ----------
    **renames
                `old_name="new_name"` pairs.

    Examples
    --------
    >>> from navis._deprecated import renamed_kwargs
    >>> @renamed_kwargs(size="min_length")
    ... def prune(x, min_length=1):
    ...     return min_length
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     prune(None, size=5)
    5

    """

    def decorator(func):
        # `__qualname__` so that a method reads as `Skeleton.prune_twigs`
        name = func.__qualname__

        # Wrapping a `map_neuronlist` wrapper is the mistake this cannot survive
        # (see above) and it only shows up under `parallel=True`, as a
        # `PicklingError` blaming the user's own arguments. Fail at import.
        if getattr(func, "__maps_neuronlist__", False):
            raise TypeError(
                f"`renamed_kwargs` must be applied below `@map_neuronlist` "
                f"(i.e. closer to the `def`), but is above it on `{name}`. "
                "Above it, it takes over the module-level name that pickle "
                "resolves the mapped wrapper by, and parallel dispatch fails."
            )

        DEPRECATED_KWARGS[name] = dict(renames)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old, new in renames.items():
                if old not in kwargs:
                    continue
                if new in kwargs:
                    raise TypeError(
                        f"`{name}()` got both `{old}` and `{new}`. `{old}` is "
                        f"the old name for `{new}` - pass only `{new}`."
                    )
                warn_renamed(f"{name}(..., {old}=...)", f"{new}=...")
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_getattr(module_name, renames=DEPRECATED_TOP_LEVEL):
    """Build a PEP 562 `__getattr__` that serves renamed attributes.

    Each name warns once per session, then resolves to the current attribute.

    Parameters
    ----------
    module_name :   str
                    `__name__` of the module being given a `__getattr__`.
    renames :       dict
                    Maps deprecated name -> current name.

    Returns
    -------
    callable
                    Suitable for assigning to a module's `__getattr__`.

    """

    def __getattr__(name):
        new = renames.get(name)
        if new is None:
            # Must be an AttributeError (not e.g. KeyError) so that `hasattr`,
            # `copy` and pickle's `__reduce_ex__` probing keep behaving.
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            )

        warn_renamed(f"{module_name}.{name}", f"{module_name}.{new}")

        # Via the module rather than a captured `globals()` so that a rename
        # pointing at a missing target raises AttributeError, not KeyError.
        return getattr(sys.modules[module_name], new)

    return __getattr__


def reset_deprecation_warnings():
    """Forget which deprecated names have already been warned about.

    Warnings are emitted once per name per session; tests that need to observe
    a second warning call this to clear the record.

    """
    _warned.clear()
