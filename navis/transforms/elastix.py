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

"""Functions to use elastix transformations."""

import copy
import functools
import os
import pathlib
import subprocess
import shutil
import tempfile
import platform

import numpy as np
import pandas as pd

from .backends import BackendMixin, elastix_is_invertible, get_elastix_transform
from .base import BaseTransform, parse_points
from .. import config
from ..utils import make_iterable

logger = config.get_logger(__name__)

_search_path = [i for i in os.environ["PATH"].split(os.pathsep) if len(i) > 0]


def find_elastixbin(tool: str = "transformix") -> str:
    """Find directory with elastix binaries."""
    for path in _search_path:
        path = pathlib.Path(path)
        if not path.is_dir():
            continue

        try:
            return next(path.glob(tool)).resolve().parent
        except StopIteration:
            continue
        except BaseException:
            raise


if platform.system() == "Windows":
    # On Windows, we have to search for `transformix.exe`
    # We can still invoke it as `transformix` via the command line though
    _elastixbin = find_elastixbin("transformix.exe")
else:
    _elastixbin = find_elastixbin("transformix")


def setup_elastix():
    """Set up to make elastix work from inside a Python session.

    Briefly: elastix requires the `LD_LIBRARY_PATH` (Linux) or `LDY_LIBRARY_PATH`
    (OSX) environment variables to (also) point to the directory with the
    elastix `lib` directory. For reasons unknown to me, these variables do not
    make it into the Python session. Hence, we have to set them here explicitly.

    Above info is based on: https://github.com/jasper-tms/pytransformix

    """
    # Don't do anything if no elastixbin
    if not _elastixbin:
        return

    # Check if this variable already exists
    var = os.environ.get("LD_LIBRARY_PATH", os.environ.get("LDY_LIBRARY_PATH", ""))

    # Get the actual path
    path = (_elastixbin.parent / "lib").absolute()

    if str(path) not in var:
        var = f"{path}{os.pathsep}{var}" if var else str(path)

    # Note that `LD_LIBRARY_PATH` works for both Linux and OSX
    os.environ["LD_LIBRARY_PATH"] = var
    # As per navis/issues/112
    os.environ["DYLD_LIBRARY_PATH"] = var


setup_elastix()


def requires_elastix(func):
    """Check if elastix is available."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _elastixbin:
            raise ValueError(
                "Could not find elastix binaries. Please download "
                "the releases page at https://github.com/SuperElastix/elastix, "
                "unzip at a convenient location and add that "
                "location to your PATH variable. Note that you "
                "will also have to set a LD_LIBRARY_PATH (Linux) "
                "or DYLD_LIBRARY_PATH (OSX) variable. See the "
                "elastic manual (release page) for details."
            )
        return func(*args, **kwargs)

    return wrapper


@requires_elastix
def elastix_version(as_string=False):
    """Get elastix version."""
    p = subprocess.run([_elastixbin / "elastix", "--version"], capture_output=True)
    if p.stderr:
        raise BaseException(f"Error running elastix:\n{p.stderr.decode()}")

    version = p.stdout.decode("utf-8").rstrip()

    # Extract version from "elastix version: 5.0.1"
    version = version.split(":")[-1]

    if as_string:
        return version
    else:
        return tuple(int(v) for v in version.split("."))


class ElastixTransform(BackendMixin, BaseTransform):
    """Elastix transforms of 3D spatial data.

    Requires either [Elastix](https://github.com/SuperElastix/elastix/) to be
    installed, or navis-fastcore - the latter implements elastix transforms in
    Rust and needs no external binaries. See the `backend` parameter.

    Based on code by Jasper Phelps (<https://github.com/jasper-tms/pytransformix>).

    Note that elastix transforms can only be inverted on the "fastcore" backend:
    the `transformix` binary has no way to compute an inverse.

    Parameters
    ----------
    file :              str
                        Filepath to elastix transformation file.
    copy_files :        filepath | list, optional
                        Any files that need to be copied into the temporary
                        directory where we perform the transform. These are
                        typically files supplemental to the main transform
                        file (e.g. defining an additional affine transform).

                        Only relevant for the "binary" backend, and only because
                        `transformix` looks for chained transform files in the
                        current directory. fastcore resolves them itself - it
                        follows the recorded path and, failing that, looks for
                        that file's basename next to the transform that named it,
                        which is the same thing copying them into one directory
                        achieves. On that backend this argument is ignored.
    backend :           "auto" | "fastcore" | "binary", optional
                        Which implementation to use. `None` (default) defers to
                        `navis.config.default_transform_backend`.

    Examples
    --------
    >>> from navis import transforms
    >>> tr = transforms.ElastixTransform('/path/to/transform/transform')
    >>> tr.xform(points) # doctest: +SKIP

    Notes
    -----
    Inverting an elastix transform is a Gauss-Newton solve per point: ~80x the cost
    of going forwards (and ~4x a dedicated reverse registration), it drops the ~0.04%
    of points that have no preimage, and where the warp folds ~4% of points come back
    somewhere other than they started - inherent to inverting a non-injective map,
    not a solver defect.

    So a purpose-built reverse registration should always beat an on-the-fly
    inversion. It does: `_pick_edge` prefers a forward registration over the inverse
    of its counterpart unconditionally, and `inverse_weight_factor` (above) keeps an
    inverse hop from looking like a shortcut when routing.

    `navis.config.elastix_invertible` nonetheless defaults to *off*, and should stay
    that way while navis-fastcore is an optional dependency: the "binary" backend
    cannot invert at all, so turning it on would let the two backends find different
    routes. Revisit when fastcore is required. It is safe to enable - on `flybrains`
    it changes nothing at all, because every elastix registration there already ships
    with a reverse - so the only thing it buys today is a route where somebody
    registered an elastix transform and no reverse for it.

    One thing worth knowing before anyone tunes this: a dedicated reverse registration
    is *not* the inverse of its forward twin - they are independent fits. Ours agree
    to a median of ~0.2 world units, but the dedicated reverse only round-trips through
    the forward map to a median of 0.17 (p95 128), where the numerical inverse
    round-trips to ~1e-13. Preferring the dedicated registration is a choice about
    *which map you want* - the one its authors fitted, and the one the field uses - not
    a claim that it is more accurate.

    """

    # Traversing an elastix transform backwards is a Gauss-Newton solve per point:
    # ~80x the cost of going forwards, and it is *lossy* - points whose preimage
    # does not exist come back as NaN (~0.04%), and where the warp folds another
    # ~4% land somewhere other than they started.
    #
    # 5 means navis will detour through up to four extra forward transforms rather
    # than invert an elastix registration once. Deliberately not the 80x runtime
    # ratio: routing trades accuracy, not seconds, and every extra hop carries its
    # own interpolation error and loses its own out-of-domain points. Five is a
    # judgement call - it says "this is a last resort", without licensing an absurd
    # detour to avoid it.
    inverse_weight_factor = 5

    _invert = False

    def __init__(self, file: str, copy_files=None, backend: str = None):
        self.file = pathlib.Path(file)
        self.copy_files = copy_files if copy_files is not None else []
        self._backend = backend

    def __eq__(self, other: "ElastixTransform") -> bool:
        """Implement equality comparison."""
        if isinstance(other, ElastixTransform):
            if self.file == other.file and self._invert == other._invert:
                return True
        return False

    def __neg__(self) -> "ElastixTransform":
        """Invert transform."""
        # Note this is defined unconditionally (so that `hasattr(tr, '__neg__')`
        # is True) but the honest answer lives in `.can_invert`/`.invertible` -
        # which is why the registry must ask *those* instead.
        if self.backend != "fastcore":
            raise NotImplementedError(
                "Inverting elastix transforms requires navis-fastcore "
                "(`pip install navis-fastcore`): the `transformix` binary "
                "cannot compute an inverse."
            )

        if not self.can_invert:
            raise NotImplementedError(
                f"Elastix transform {self.file} cannot be inverted: its chain "
                'combines transforms via "Add".'
            )

        x = self.copy()
        x._invert = not x._invert
        return x

    @property
    def can_invert(self) -> bool:
        """Whether `-transform` works.

        Needs the fastcore backend (`transformix` cannot invert) *and* a file
        that is actually invertible - a chain combined via "Add" is not. The
        latter is settled with fastcore's header-only probe, so this stays cheap
        enough to ask of every transform in the registry.

        """
        if self.backend != "fastcore" or not self.file.is_file():
            return False
        return elastix_is_invertible(str(self.file))

    @property
    def invertible(self) -> bool:
        """Whether the bridging graph may traverse this transform backwards.

        Deliberately *narrower* than `can_invert`: inverting works whenever we're
        on the fastcore backend with an invertible file, but we only let the
        registry route through an inverse if `navis.config.elastix_invertible`
        says so. Every elastix registration we know of ships with a purpose-built
        reverse registration, so the inverse buys no new connectivity - it just
        adds a cheaper-looking parallel edge that drags unrelated routes through
        it.

        Note the config check comes first, so that in the default case this costs
        a single lookup and never touches the disk - it is evaluated for every
        registered transform each time the bridging graph is built.

        """
        if not getattr(config, "elastix_invertible", False):
            return False
        return self.can_invert

    def check_if_possible(self, on_error: str = "raise"):
        """Check if this transform is possible."""
        msg = None
        if not self.file.is_file():
            msg = f"Transformation file {self.file} not found."
        elif self.backend == "binary":
            if not _elastixbin:
                msg = (
                    "Folder with elastix binaries not found. Make sure the "
                    "directory is in your PATH environment variable - or install "
                    "navis-fastcore to transform points without elastix."
                )
            elif self._invert:
                msg = (
                    "Inverse elastix transforms require navis-fastcore: the "
                    "`transformix` binary cannot compute an inverse."
                )
        elif self._invert and not elastix_is_invertible(str(self.file)):
            # Header-only probe - no need to parse the coefficients just to find
            # out we can't use them.
            msg = (
                f"Elastix transform {self.file} cannot be inverted: its "
                'chain combines transforms via "Add".'
            )

        if msg and on_error == "raise":
            raise BaseException(msg)

        return msg

    def copy(self) -> "ElastixTransform":
        """Return copy."""
        # Attributes not to copy
        no_copy = []
        # Generate new empty transform
        x = self.__class__(self.file)
        # Override with this neuron's data
        x.__dict__.update(
            {k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy}
        )

        return x

    def write_input_file(self, points, filepath):
        """Write a numpy array in format required by transformix."""
        with open(filepath, "w") as f:
            f.write("point\n{}\n".format(len(points)))
            for x, y, z in points:
                f.write(f"{x:f} {y:f} {z:f}\n")

    def read_output_file(self, filepath) -> np.ndarray:
        """Load output file.

        Parameter
        ---------
        filepath :      str
                        Filepath to output file.

        Returns
        -------
        pointsxf :      (N, 3) numpy array
                        The parse transformed points.

        """
        points = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                output = line.split("OutputPoint = [ ")[1].split(" ]")[0]
                points.append([float(i) for i in output.split(" ")])
        return np.array(points)

    def xform(self, points: np.ndarray, return_logs=False) -> np.ndarray:
        """Xform data.

        Parameters
        ----------
        points :        (N, 3) numpy array | pandas.DataFrame
                        Points to xform. DataFrame must have x/y/z columns.
        return_logs :   bool
                        If True, will return logs instead of transformed points.
                        Really only useful for debugging.

        Returns
        -------
        pointsxf :      (N, 3) numpy array
                        Transformed points.

        """
        self.check_if_possible(on_error="raise")

        if self.backend == "fastcore":
            if return_logs:
                raise ValueError(
                    "`return_logs=True` requires the `transformix` binary "
                    "(use `backend='binary'`)."
                )

            if self.copy_files:
                logger.debug(
                    "`copy_files` is ignored on the fastcore backend: it resolves "
                    "chained transform files itself."
                )

            # One parsed object serves both directions - fastcore takes `invert`
            # at xform time - so a transform and its inverse share a cache entry.
            tr = get_elastix_transform(str(self.file))
            return tr.xform(parse_points(points), invert=self._invert)

        if isinstance(points, pd.DataFrame):
            # Make sure x/y/z columns are present
            if np.any([c not in points for c in ["x", "y", "z"]]):
                raise ValueError("points DataFrame must have x/y/z columns.")
            points = points[["x", "y", "z"]].values
        elif not (
            isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 3
        ):
            raise TypeError(
                "`points` must be numpy array of shape (N, 3) or "
                "pandas DataFrame with x/y/z columns"
            )

        # Everything happens in a temporary directory
        with tempfile.TemporaryDirectory() as tempdir:
            p = pathlib.Path(tempdir)

            # If required, copy additional files into the temporary directory
            if self.copy_files:
                for f in make_iterable(self.copy_files):
                    _ = pathlib.Path(shutil.copy(f, p))

            # Write points to file
            in_file = p / "inputpoints.txt"
            self.write_input_file(points, in_file)

            out_file = p / "outputpoints.txt"

            # Prepare the command
            command = [
                _elastixbin / "transformix",
                "-out",
                str(p),
                "-tp",
                str(self.file),
                "-def",
                str(in_file),
            ]

            # Keep track of current working directory
            cwd = os.getcwd()
            try:
                # Change working directory to the temporary directory
                # This is apparently required because elastix stupidly expects
                # any secondary transform files to be in the current directory
                # (as opposed to where the main transform is)
                os.chdir(p)
                # Run the actual transform
                proc = subprocess.run(command, stdout=subprocess.PIPE)
            except BaseException:
                raise
            finally:
                # This makes sure we land on our feet even in case of an error
                os.chdir(cwd)

            if return_logs:
                logfile = p / "transformix.log"
                if not logfile.is_file():
                    raise FileNotFoundError("No log file found.")
                with open(logfile) as f:
                    logs = f.read()
                return logs

            if not out_file.is_file():
                raise FileNotFoundError(
                    "Elastix transform did not produce any "
                    f"output:\n {proc.stdout.decode()}"
                )

            points_xf = self.read_output_file(out_file)

        return points_xf
