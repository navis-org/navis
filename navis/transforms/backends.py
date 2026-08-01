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

"""Backend selection for point transforms.

Several of navis' transforms have a navis-fastcore implementation alongside the
original one, and pick between them at call time:

- CMTK and elastix transforms otherwise shell out to the external binaries
  (`streamxform`, `transformix`); fastcore implements them in-process in Rust,
  needing no binaries and skipping the subprocess/temp-file overhead entirely.
- The landmark transforms (thin-plate spline, moving least squares) otherwise
  run on `morphops` / `molesq`; fastcore implements them in Rust without
  materialising the (n_points, n_landmarks) intermediates those libraries
  build, so there is no `batch_size` to tune and no landmark-count ceiling.

The two families differ only in what the *non*-fastcore option is called:
"binary" for CMTK/elastix, "python" for the landmark transforms. Both spellings
are accepted everywhere and mean the same thing - "do not use fastcore" - so a
single `navis.config.default_transform_backend` can steer both families.

Deprecated
----------
Both non-fastcore backends are deprecated as of 2.0 and will be removed in 3.0,
now that navis-fastcore is a hard requirement. Selecting one emits a
`DeprecationWarning`; "auto" (the default) has always preferred fastcore, so
nothing changes for callers who never set a backend explicitly.

"""

import functools
import warnings

from .. import config

__all__ = ["set_transform_backend", "clear_transform_cache"]

#: The non-fastcore backends, i.e. what a transform falls back to. Which name a
#: given transform reports is a property of that transform (`_fallback_backend`).
FALLBACK_BACKENDS = ("binary", "python")

BACKENDS = ("auto", "fastcore") + FALLBACK_BACKENDS


#: Backends already warned about, so that a deprecated backend left set in the
#: config does not warn on every `.backend` read. `warnings`' own per-callsite
#: registry cannot do this for us: the warning would be attributed to whichever
#: navis-internal frame happened to read the property.
_WARNED_BACKENDS = set()


def warn_if_deprecated_backend(backend: str, stacklevel: int) -> None:
    """Warn once per session that `backend` is on its way out.

    Called where a backend is *chosen* - `set_transform_backend`, or a transform
    resolving one for the first time - rather than on every `.backend` read: the
    property is resolved lazily and internal code hits it several times per
    transform per call.

    Parameters
    ----------
    stacklevel : int
                Passed straight to `warnings.warn`, i.e. counted from *this*
                function: 2 blames the caller, 3 its caller, and so on. It has to
                land on user code or Python's default filters hide the warning
                entirely (only `__main__` gets `default::DeprecationWarning`).

    """
    if backend not in FALLBACK_BACKENDS or backend in _WARNED_BACKENDS:
        return

    _WARNED_BACKENDS.add(backend)
    warnings.warn(
        f'Transform backend "{backend}" is deprecated and will be removed in '
        "navis 3.0. navis-fastcore is a hard requirement now and does these "
        "transforms in-process - faster, with no external binaries to install, "
        'and (for elastix) invertible. Drop the explicit backend or use "auto".',
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def resolve_backend(requested=None, fallback="binary") -> str:
    """Resolve a backend request to either "fastcore" or `fallback`.

    Parameters
    ----------
    requested : "auto" | "fastcore" | "binary" | "python" | None
                `None` defers to `navis.config.default_transform_backend`.
                "binary" and "python" both mean "do not use fastcore"; which of
                the two comes back is decided by `fallback`, not by the request.
    fallback :  str
                What to return when fastcore is not used.

    """
    if requested is None:
        requested = getattr(config, "default_transform_backend", "auto")

    if requested not in BACKENDS:
        raise ValueError(
            f'Unknown transform backend "{requested}". Must be one of {BACKENDS}.'
        )

    if requested in FALLBACK_BACKENDS:
        warn_if_deprecated_backend(requested, stacklevel=4)
        return fallback

    # Both "auto" and "fastcore" resolve to fastcore - it is a hard requirement,
    # so there is nothing left to probe for.
    return "fastcore"


class BackendMixin:
    """Gives a transform a `.backend` property.

    `self._backend` holds only what the user explicitly asked for (`None` means
    "follow the config"). The effective backend is resolved lazily, on access.

    That laziness is load-bearing: template packages such as `flybrains` build
    all their transforms at import time, i.e. long before a user gets a chance
    to touch `navis.config`. If we snapshotted the backend in `__init__`,
    setting `navis.config.default_transform_backend` after `import flybrains`
    would silently have no effect.

    Subclasses whose non-fastcore implementation is not the CMTK/elastix one
    override `_fallback_backend`.

    """

    _backend = None

    #: What this transform calls its non-fastcore implementation.
    _fallback_backend = "binary"

    @property
    def backend(self) -> str:
        """The backend this transform will actually use."""
        return resolve_backend(self._backend, fallback=self._fallback_backend)


# Parsing a registration costs 3.5-37ms, so we cache the parsed objects. The
# cache is deliberately *module*-level rather than per-instance: transforms get
# copied a lot (`TransformSequence` copies every transform it is handed), and
# copying a fastcore object re-reads it from disk - so an instance-level cache
# would never actually hit. Keeping the Rust objects out of the instances also
# keeps the transforms cheap to pickle for multiprocessing.
#
# Note the cache keys on the files *only*. Direction is a property of the
# traversal, not of the parse: fastcore takes `invert` at xform time, so one
# parsed object serves every direction. A forward chain, its inverse, and any
# mixed traversal of the same files all share a single entry.
#
# The key is derived from the transform's current state at call time rather than
# stored on it. `copy()`, `__add__`, `append()` and `__neg__` all mutate
# `regs`/`directions`, so a stored handle would need invalidating in half a
# dozen places; deriving the key means a mutated transform simply looks up a
# different entry and there is no stale-cache failure mode to get wrong.


@functools.lru_cache(maxsize=64)
def get_cmtk_reg(paths: tuple):
    """Get a (cached) fastcore CmtkRegistration for these files."""
    from .. import utils

    return utils.fastcore.CmtkRegistration(list(paths))


@functools.lru_cache(maxsize=64)
def get_elastix_transform(path: str):
    """Get a (cached) fastcore ElastixTransform for this file."""
    from .. import utils

    return utils.fastcore.ElastixTransform(path)


@functools.lru_cache(maxsize=256)
def elastix_is_invertible(path: str) -> bool:
    """Whether this elastix transform can be inverted.

    Uses fastcore's header-only probe, which is ~10x cheaper than parsing the
    file - it does not read the B-spline coefficients. That matters because this
    is asked of every registered transform whenever the bridging graph is built.

    """
    from .. import utils

    return bool(utils.fastcore.probe_elastix_invertible(path))


def clear_transform_cache():
    """Drop all cached (parsed) fastcore transforms.

    Mostly useful if a registration changed on disk - we do not stat the files
    on every call.

    """
    get_cmtk_reg.cache_clear()
    get_elastix_transform.cache_clear()
    elastix_is_invertible.cache_clear()


def set_transform_backend(backend: str = None, elastix_invertible: bool = None):
    """Set the backend used for point transforms.

    Applies to CMTK and elastix transforms, and to the landmark transforms
    ([`navis.transforms.TPStransform`][] and
    [`navis.transforms.MovingLeastSquaresTransform`][]).

    Parameters
    ----------
    backend :   "auto" | "fastcore" | "binary" | "python", optional
                - "auto" (default) and "fastcore" both use navis-fastcore's
                  in-process Rust implementation
                - "binary"/"python" force the original implementation: the
                  external binaries (`streamxform`, `transformix`) for
                  CMTK/elastix, `morphops`/`molesq` for the landmark
                  transforms. The two names are interchangeable - each
                  transform reports whichever fits it. **Both are deprecated
                  and will be removed in 3.0**; selecting one warns.
    elastix_invertible : bool, optional
                Whether the bridging graph may traverse an elastix registration
                backwards (only possible on the fastcore backend). Off by
                default - see `navis.config.elastix_invertible`. Note this
                re-routes a substantial fraction of the bridging paths.

    Examples
    --------
    >>> import navis
    >>> navis.transforms.set_transform_backend('fastcore')
    >>> navis.transforms.set_transform_backend('auto')

    """
    if backend is not None:
        if backend not in BACKENDS:
            raise ValueError(
                f'Unknown transform backend "{backend}". Must be one of {BACKENDS}.'
            )
        # Warn *here*, where the choice is actually made: `stacklevel=2` puts the
        # blame on the caller's frame, which is what Python's default filters need
        # to show it at all.
        warn_if_deprecated_backend(backend, stacklevel=3)
        config.default_transform_backend = backend

    if elastix_invertible is not None:
        config.elastix_invertible = bool(elastix_invertible)

    # Both of the above feed the transforms' `invertible` property, which in turn
    # decides the shape of the (cached) bridging graph - so both caches go.
    clear_transform_cache()

    from .templates import registry

    registry.clear_caches()
