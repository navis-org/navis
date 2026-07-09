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

"""Backend abstraction and registry for NBLAST.

A *backend* is an implementation of one or more NBLAST operations (``nblast``,
``nblast_allbyall``, ``nblast_smart``, ...). The public ``navis.nblast*``
functions do the shared parameter validation and preflight checks and then
delegate the actual work to a backend selected via the ``backend`` parameter.

Backends only implement the operations they support (as methods of the same
name) and declare, via :meth:`NblastBackend.unsupported`, which
operation/parameter combinations they *cannot* handle. This keeps the system
open: a backend may be a full reimplementation (e.g. ``navis-fastcore``, in
Rust) or merely a different work dispatcher (e.g. joblib instead of the
built-in multiprocessing), and it need not cover every operation or parameter.

Third-party libraries can add their own backends by calling
:func:`register_backend` at import time.
"""

from abc import ABC

from ... import config

logger = config.get_logger(__name__)

__all__ = ["NblastBackend", "register_backend", "get_backend",
           "list_backends", "available_backends", "resolve_backend"]

# Operation name -> is implemented by looking up an attribute of the same name
# on the backend. This is the canonical list of dispatchable operations.
OPERATIONS = ("nblast", "nblast_allbyall", "nblast_smart", "synblast")

# Registry of name -> backend instance
_BACKENDS = {}


class NblastBackend(ABC):
    """Base class for NBLAST backends.

    A backend implements one or more of the operations listed in
    :data:`OPERATIONS` as methods of the same name. It does not need to
    implement all of them.

    Attributes
    ----------
    name :      str
                Unique name used to select this backend (e.g. via the
                ``backend`` parameter or ``navis.config.default_nblast_backend``).
    priority :  int
                Higher-priority backends are preferred when ``backend="auto"``.
                The built-in backend uses ``0``; faster backends should use a
                higher value.

    """

    name: str = "base"
    priority: int = 0

    def __repr__(self):
        return f"<NblastBackend '{self.name}' (priority={self.priority})>"

    def available(self) -> bool:
        """Whether this backend's dependencies are importable.

        Backends whose optional dependencies are missing should return False
        so they are skipped during ``backend="auto"`` resolution.
        """
        return True

    def implements(self, operation: str) -> bool:
        """Whether this backend implements `operation` at all."""
        return callable(getattr(self, operation, None))

    def unsupported(self, operation: str, **params) -> list:
        """Return reasons why this backend cannot run `operation` with `params`.

        An empty list means the request is supported. Non-empty entries are
        human-readable strings explaining the limitation (used both to skip a
        backend during ``"auto"`` resolution and to build a helpful error
        message when a backend was explicitly requested).

        The default implementation only checks whether the operation is
        implemented at all; subclasses should extend it to reject unsupported
        parameter combinations (e.g. ``approx_nn=True``).
        """
        if not self.implements(operation):
            return [f"backend '{self.name}' does not implement '{operation}'"]
        return []


def register_backend(backend: NblastBackend, name: str = None):
    """Register an NBLAST backend.

    Parameters
    ----------
    backend :   NblastBackend
                The backend instance to register.
    name :      str, optional
                Name to register under. Defaults to ``backend.name``.

    """
    if not isinstance(backend, NblastBackend):
        raise TypeError(f"Expected NblastBackend, got {type(backend)}")
    _BACKENDS[name or backend.name] = backend


def get_backend(name: str) -> NblastBackend:
    """Get a registered backend by name."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown NBLAST backend '{name}'. "
                         f"Available: {list_backends()}")
    return _BACKENDS[name]


def list_backends() -> list:
    """List names of all registered backends."""
    return list(_BACKENDS)


def available_backends() -> list:
    """List backend instances whose dependencies are importable."""
    return [b for b in _BACKENDS.values() if b.available()]


def resolve_backend(operation: str, backend="auto", **params) -> NblastBackend:
    """Select a backend for `operation` given `params`.

    Parameters
    ----------
    operation : str
                The operation to run (e.g. "nblast"). Must be one of
                :data:`OPERATIONS`.
    backend :   str | NblastBackend
                Either "auto" (pick the highest-priority available backend that
                supports the request, falling back to "builtin"), or the name of
                a specific backend, or a backend instance.
    **params
                The call parameters (e.g. ``approx_nn``, ``scores``, ``smat``).
                Used to check whether a backend supports the request.

    Returns
    -------
    NblastBackend

    """
    # Allow passing a backend instance directly
    if isinstance(backend, NblastBackend):
        return backend

    if backend in (None, "auto"):
        # Highest priority first; builtin (priority 0) is the fallback
        for be in sorted(available_backends(), key=lambda b: -b.priority):
            if not be.unsupported(operation, **params):
                return be
        # Nothing supported the request - fall back to builtin so we can raise
        # a meaningful error (or run it, if builtin does support it).
        return get_backend("builtin")

    be = get_backend(backend)

    if not be.available():
        raise ValueError(
            f"NBLAST backend '{be.name}' is not available - its optional "
            "dependencies are not installed."
        )

    reasons = be.unsupported(operation, **params)
    if reasons:
        raise ValueError(
            f"NBLAST backend '{be.name}' cannot run '{operation}' with the "
            f"given parameters: {'; '.join(reasons)}."
        )

    return be
