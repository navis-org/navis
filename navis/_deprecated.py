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

navis 2.0 renamed the neuron classes to `Skeleton`, `Mesh` and `Voxels`. The old
names still resolve, but only in the top-level `navis` namespace and only via
PEP 562 module `__getattr__` - a plain module global is found by ordinary
attribute lookup before `__getattr__` is ever consulted, so the warning would
never fire.

Elsewhere the old names *are* plain aliases (see the tail of
`navis/core/skeleton.py` & co), because `pickle` resolves classes by their
defining module and must find them without a warning. Aliases either way, so
`isinstance` and subclassing are unaffected by the rename.
"""

import sys
import warnings

__all__ = [
    "DEPRECATED_NEURON_CLASSES",
    "deprecated_getattr",
    "reset_deprecation_warnings",
]

#: Old name -> new name. `Dotprops` was not renamed.
DEPRECATED_NEURON_CLASSES = {
    "TreeNeuron": "Skeleton",
    "MeshNeuron": "Mesh",
    "VoxelNeuron": "Voxels",
}

# Names already warned about, so that a loop over a NeuronList doesn't produce
# one warning per neuron.
_warned = set()


def deprecated_getattr(module_name, renames=DEPRECATED_NEURON_CLASSES):
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

        if name not in _warned:
            warnings.warn(
                f"`{module_name}.{name}` is deprecated and will be removed in a "
                f"future version - use `{module_name}.{new}` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Recorded only once the warning is through: under
            # `-W error::DeprecationWarning` the line above raises, and marking
            # the name first would let every later access pass silently.
            _warned.add(name)

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
