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

import logging
import pint
import os

import matplotlib as mpl

logger = logging.getLogger("navis")


def default_logging():
    """Add a formatted stream handler to the `navis` logger.

    Called by default when navis is imported for the first time.
    To prevent this behaviour, set an environment variable:
    `NAVIS_SKIP_LOG_SETUP=True`.
    """
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter("%(levelname)-5s : %(message)s (%(name)s)")
        sh.setFormatter(formatter)
        logger.addHandler(sh)


def remove_log_handlers():
    """Remove all handlers from the `navis` logger.

    It may be preferable to skip navis' default log handler being added in the
    first place.
    Do this by setting an environment variable before the first import:
    `NAVIS_SKIP_LOG_SETUP=True`.
    """
    logger.handlers.clear()


skip_log_setup = os.environ.get("NAVIS_SKIP_LOG_SETUP", "").lower() == "true"
if not skip_log_setup:
    default_logging()


def get_logger(name: str):
    if skip_log_setup:
        return logging.getLogger(name)
    return logger


# Set up numpy number representation, see NEP51
# Once numpy<=2 is dropped from requirements, the doctest comparissons
# should become `np.float64(1.074)` instead of `1.074`
if os.environ.get("NAVIS_TEST_ENV", "").lower() == "true":
    import numpy as np

    np.set_printoptions(legacy="1.25")

# Default settings for progress bars
pbar_hide = False
pbar_leave = False


class quiet_logger:
    """Silence navis' logger - and optionally its progress bars - for a block.

    For the common case of calling a chatty navis function for a side effect,
    where its warnings are about a temporary object the caller will never see
    (a throwaway downsampled copy, say).

    There is only *one* navis logger, so putting the level back is not
    housekeeping - an exception escaping the block would otherwise leave the
    whole library silenced for the rest of the session, with nothing to tell
    anyone why. That is what makes this a context manager rather than a pair of
    `setLevel` calls at each site.

    Parameters
    ----------
    level :     str | int
                Level to silence *to*. Only ever quietens: a logger already at
                `ERROR` is left where it is by `quiet_logger('WARNING')`.
    pbars :     bool
                Whether to also hide progress bars for the duration.

    Examples
    --------
    >>> from navis import config
    >>> with config.quiet_logger():
    ...     pass

    """

    def __init__(self, level="ERROR", pbars=False):
        if isinstance(level, str):
            level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            raise ValueError(f"Not a logging level: {level!r}")
        self.level = level
        self.pbars = pbars

    def __enter__(self):
        # Restore what was explicitly set, but decide from what is in effect:
        # a logger inheriting its level has `.level == NOTSET`, and writing that
        # back would be a no-op where writing back `getEffectiveLevel()` would
        # pin it.
        self._previous = logger.level
        self._changed = logger.getEffectiveLevel() < self.level
        if self._changed:
            logger.setLevel(self.level)

        if self.pbars:
            global pbar_hide
            self._pbar_hide = pbar_hide
            pbar_hide = True
        return self

    def __exit__(self, *args):
        # `setLevel` drops the level cache of *every* logger in the process, so
        # don't pay for it where we changed nothing.
        if self._changed:
            logger.setLevel(self._previous)
        if self.pbars:
            global pbar_hide
            pbar_hide = self._pbar_hide
        return False


# Default settings for caching
warn_caching = True

# Default backend for `parallel=True`, i.e. where per-neuron work is run:
#   "auto" (default) picks the highest-priority installed backend that can run
#   the request - `joblib`, then `pathos`, then the dependency-free stdlib
#   process pool. Name one ("joblib", "pathos", "processes", "threads",
#   "serial") to force it.
# Prefer `navis.set_parallel_backend()` over setting this directly: it
# validates the name up front and also accepts a `concurrent.futures.Executor`,
# which is how you point navis at a cluster.
default_parallel_backend = "auto"

# Default number of workers for `parallel=True`. None means "half the available
# cores". Set via `navis.set_parallel_backend(n_workers=...)`.
default_n_workers = None

# Settings carried into worker processes. A forked worker inherits these for
# free, but a spawned one re-imports navis and would otherwise silently revert
# to the defaults - so anything a user can change and would expect to still
# apply inside a worker belongs here.
#
# Everything listed must be picklable, since it travels with the work. That
# rules out `ureg`, `logger` and the `tqdm` callables (objects rather than
# settings - the log *level* is carried separately) and, notably,
# `default_parallel_backend`, which may hold a live executor. Not carrying it
# also means a worker never inherits "run this on the cluster", so nested
# `parallel=True` stays local instead of trying to recurse into the scheduler.
WORKER_SETTINGS = (
    "pbar_hide",
    "pbar_leave",
    "warn_caching",
    "add_units",
    "default_n_workers",
    "default_nblast_backend",
    "default_transform_backend",
    "elastix_invertible",
    "default_connector_colors",
    "max_grid_size",
)

# Default backend for NBLAST functions:
#   "builtin" (default) uses navis' own multiprocessing implementation.
#   Set to "auto" to instead pick the fastest available backend that supports
#   the requested operation and parameters (usually "fastcore"), or name a
#   specific backend (e.g. "fastcore") to force it.
default_nblast_backend = "builtin"

# Default backend for point transforms - CMTK/elastix and the landmark
# transforms (thin-plate spline, moving least squares):
#   "auto" (default) uses navis-fastcore's in-process Rust implementation.
#   "binary"/"python" force the original implementation instead - the external
#   binaries (`streamxform`, `transformix`) for CMTK/elastix, `morphops`/`molesq`
#   for the landmark transforms. Those two names are interchangeable; each
#   transform reports whichever fits it. Both are DEPRECATED as of 2.0 and will
#   be removed in 3.0 - selecting one warns.
# Prefer `navis.transforms.set_transform_backend()` over setting this directly:
# elastix transforms are only invertible on the fastcore backend, so changing
# the backend also has to invalidate the cached bridging graph.
default_transform_backend = "auto"

# Whether elastix transforms may be inverted *by the bridging graph*.
# The fastcore backend can invert an elastix transform (the `transformix` binary
# cannot), and `-transform` / `TransformSequence` always honour that. This flag
# only controls whether the template registry is also allowed to traverse an
# elastix registration backwards when it plots a route between two templates.
#
# It stays False while the deprecated binary backend is still selectable: that
# backend cannot invert at all, so enabling this would let the two find different
# routes. Flip it to True in 3.0, when the binary backend goes away.
#
# It is otherwise safe to turn on: on `flybrains` it changes nothing whatsoever
# (no re-routing, no new routes, and no route actually uses an inverted elastix),
# because every elastix registration there already ships with a purpose-built
# reverse. Its only effect is to provide a route where somebody registered an
# elastix transform without one.
elastix_invertible = False

# Maximum size (in bytes) of a dense voxel grid that navis will allocate.
# Voxel grids are allocated from a *shape*, not from the number of filled
# voxels: `Voxels.shape` is derived from the voxel coordinates, so a
# handful of far-apart voxels can imply a grid of terabytes. On systems that
# overcommit, numpy hands out such an array without complaint and the process
# is then OOM-killed (SIGKILL, no traceback) once the pages are touched - hence
# we check up-front instead of relying on `MemoryError`.
# Set to 0 or None to disable the check.
max_grid_size = int(os.environ.get("NAVIS_MAX_GRID_SIZE", 4 * 1024**3))  # 4 GiB

# Default color for neurons
default_color = (0.95, 0.65, 0.04)

# Unit registry
ureg = pint.UnitRegistry()

# Whether to add units to certain spatial neuron properties
add_units = False

# Set to true to prevent Viewer from ever showing
headless = os.environ.get("NAVIS_HEADLESS", "False").lower() == "true"
if headless:
    logger.info("Running in headless mode.")
    mpl.use("template")
    pbar_hide = True

# Default connector color palette
default_connector_colors = {
    0: {"name": "Presynapses", "color": (1, 0, 0)},
    1: {"name": "Postsynapses", "color": (0, 0.75, 0.75)},
    2: {"name": "Gap junctions", "color": (0, 1, 0)},
    "display": "lines",  # can also be 'circles'
    "size": 2,  # for "circles" only
}

# Set some synonyms
default_connector_colors["pre"] = default_connector_colors["Pre"] = (
    default_connector_colors[0]
)
default_connector_colors["post"] = default_connector_colors["Post"] = (
    default_connector_colors[1]
)
default_connector_colors["gap"] = default_connector_colors["Gap"] = (
    default_connector_colors[2]
)
default_connector_colors["gap_junction"] = default_connector_colors["Gap_junction"] = (
    default_connector_colors[2]
)
default_connector_colors["gap_junctions"] = default_connector_colors[
    "Gap_junctions"
] = default_connector_colors[2]


def _type_of_script():
    """Returns context in which navis is run."""
    try:
        ipy_str = str(type(get_ipython()))  # noqa
        if "zmqshell" in ipy_str:
            return "jupyter"
        if "terminal" in ipy_str:
            return "ipython"
    except BaseException:
        return "terminal"


def is_jupyter():
    """Test if navis is run in a Jupyter notebook."""
    return _type_of_script() == "jupyter"


# Here, we import tqdm and determine whether we use classic notebook tbars
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.notebook import trange as trange_notebook
from tqdm import tqdm as tqdm_classic
from tqdm import trange as trange_classic

# Keep this because `tqdm_notebook` is only a wrapper (type "function")
tqdm_class = tqdm_classic

if is_jupyter():
    tqdm = tqdm_notebook
    trange = trange_notebook
else:
    tqdm = tqdm_classic
    trange = trange_classic
