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

"""Pluggable backends for NBLAST.

See :mod:`navis.nbl.backends.base` for the abstraction. Third-party libraries
can register their own backends via :func:`register_backend`::

    from navis.nbl.backends import NblastBackend, register_backend

    class MyGPUBackend(NblastBackend):
        name = "mygpu"
        priority = 20
        def available(self): ...
        def nblast(self, query, target, **params): ...

    register_backend(MyGPUBackend())

"""

from .base import (NblastBackend, register_backend, get_backend,
                   list_backends, available_backends, resolve_backend)
from .builtin import BuiltinBackend
from .fastcore import FastcoreBackend

# Register the backends shipped with navis. The built-in backend must always be
# present as it is the fallback for ``backend="auto"``.
register_backend(BuiltinBackend())
register_backend(FastcoreBackend())

__all__ = ["NblastBackend", "register_backend", "get_backend", "list_backends",
           "available_backends", "resolve_backend", "BuiltinBackend",
           "FastcoreBackend"]
