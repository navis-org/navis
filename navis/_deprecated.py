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

Three kinds of rename live here, each with a table so that the shims and the
tests read from one place:

- **classes** - navis 2.0 renamed the neuron classes to `Skeleton`, `Mesh` and
  `Voxels`;
- **top-level functions** - 2.0 settled on "component" as the one word for a
  connected piece of a neuron;
- **properties** - the same, on `Skeleton`.

The first two resolve in the top-level `navis` namespace via PEP 562 module
`__getattr__` - and only there, because a plain module global is found by
ordinary attribute lookup before `__getattr__` is ever consulted, so the warning
would never fire.

Elsewhere the old class names *are* plain aliases (see the tail of
`navis/core/skeleton.py` & co), because `pickle` resolves classes by their
defining module and must find them without a warning. Aliases either way, so
`isinstance` and subclassing are unaffected by the rename.
"""

import sys
import warnings

__all__ = [
    "DEPRECATED_NEURON_CLASSES",
    "DEPRECATED_FUNCTIONS",
    "DEPRECATED_PROPERTIES",
    "DEPRECATED_TOP_LEVEL",
    "deprecated_getattr",
    "deprecated_property",
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

#: Everything the `navis` namespace serves under an old name.
DEPRECATED_TOP_LEVEL = {**DEPRECATED_NEURON_CLASSES, **DEPRECATED_FUNCTIONS}

# Names already warned about, so that a loop over a NeuronList doesn't produce
# one warning per neuron.
_warned = set()


def warn_renamed(old, new, stacklevel=3):
    """Warn once per session that `old` has been renamed to `new`."""
    if old in _warned:
        return

    warnings.warn(
        f"`{old}` is deprecated and will be removed in a future version - "
        f"use `{new}` instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
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

        warn_renamed(f"{module_name}.{name}", f"{module_name}.{new}", stacklevel=2)

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
