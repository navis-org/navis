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

import numpy as np
import pandas as pd

from .base import BaseTransform
from ..utils import make_iterable

_search_path = [i for i in os.environ['PATH'].split(os.pathsep) if len(i) > 0]


def find_elastixbin(tool: str = 'transformix') -> str:
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


_elastixbin = find_elastixbin()


def setup_elastix():
    """Set up to make elastix work from inside a Python session.

    Briefly: elastix requires the `LD_LIBRARY_PATH` (Linux) or `LDY_LIBRARY_PATH`
    (OSX) environment variables to (also) point to the directory with the
    elastix `lib` directory. For reasons unknown to me, these varibles do not
    make it into the Python session. Hence, we have to set them here explicitly.

    Above info is based on: https://github.com/jasper-tms/pytransformix

    """
    # Don't do anything if no elastixbin
    if not _elastixbin:
        return

    # Check if this variable already exists
    var = os.environ.get('LD_LIBRARY_PATH', os.environ.get('LDY_LIBRARY_PATH', ''))

    # Get the actual path
    path = (_elastixbin.parent / 'lib').absolute()

    if str(path) not in var:
        var = f'{path}{os.pathsep}{var}' if var else str(path)

    # Note that `LD_LIBRARY_PATH` works for both Linux and OSX
    os.environ['LD_LIBRARY_PATH'] = var
    # As per navis/issues/112
    os.environ['DYLD_LIBRARY_PATH'] = var


setup_elastix()


def requires_elastix(func):
    """Check if elastix is available."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _elastixbin:
            raise ValueError("Could not find elastix binaries. Please download "
                             "the releases page at https://github.com/SuperElastix/elastix, "
                             "unzip at a convenient location and add that "
                             "location to your PATH variable. Note that you "
                             "will also have to set a LD_LIBRARY_PATH (Linux) "
                             "or DYLD_LIBRARY_PATH (OSX) variable. See the "
                             "elastic manual (release page) for details.")
        return func(*args, **kwargs)
    return wrapper


@requires_elastix
def elastix_version(as_string=False):
    """Get elastix version."""
    p = subprocess.run([_elastixbin / 'elastix', '--version'],
                       capture_output=True)
    if p.stderr:
        raise BaseException(f'Error running elastix:\n{p.stderr.decode()}')

    version = p.stdout.decode('utf-8').rstrip()

    # Extract version from "elastix version: 5.0.1"
    version = version.split(':')[-1]

    if as_string:
        return version
    else:
        return tuple(int(v) for v in version.split('.'))


class ElastixTransform(BaseTransform):
    """Elastix transforms of 3D spatial data.

    Requires `Elastix <https://github.com/SuperElastix/elastix/>`_. Based on
    code by Jasper Phelps (https://github.com/jasper-tms/pytransformix).

    Note that elastix transforms can not be inverted!

    Parameters
    ----------
    file :              str
                        Filepath to elastix transformation file.
    copy_files :        filepath | list, optional
                        Any files that need to be copied into the temporary
                        directory where we perform the transform. These are
                        typically files supplemental to the main transform
                        file (e.g. defining an additional affine transform).

    Examples
    --------
    >>> from navis import transforms
    >>> tr = transforms.ElastixTransform('/path/to/transform/transform')
    >>> tr.xform(points) # doctest: +SKIP

    """

    def __init__(self, file: str, copy_files=[]):
        self.file = pathlib.Path(file)
        self.copy_files = copy_files

    def __eq__(self, other: 'ElastixTransform') -> bool:
        """Implement equality comparison."""
        if isinstance(other, ElastixTransform):
            if self.file == other.file:
                return True
        return False

    def check_if_possible(self, on_error: str = 'raise'):
        """Check if this transform is possible."""
        if not _elastixbin:
            msg = 'Folder with elastix binaries not found. Make sure the ' \
                  'directory is in your PATH environment variable.'
            if on_error == 'raise':
                raise BaseException(msg)
            return msg
        if not self.file.is_file():
            msg = f'Transformation file {self.file} not found.'
            if on_error == 'raise':
                raise BaseException(msg)
                return msg

    def copy(self) -> 'ElastixTransform':
        """Return copy."""
        # Attributes not to copy
        no_copy = []
        # Generate new empty transform
        x = self.__class__(self.file)
        # Override with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        return x

    def write_input_file(self, points, filepath):
        """Write a numpy array in format required by transformix."""
        with open(filepath, 'w') as f:
            f.write('point\n{}\n'.format(len(points)))
            for x, y, z in points:
                f.write(f'{x:f} {y:f} {z:f}\n')

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
        with open(filepath, 'r') as f:
            for line in f.readlines():
                output = line.split('OutputPoint = [ ')[1].split(' ]')[0]
                points.append([float(i) for i in output.split(' ')])
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
        self.check_if_possible(on_error='raise')

        if isinstance(points, pd.DataFrame):
            # Make sure x/y/z columns are present
            if np.any([c not in points for c in ['x', 'y', 'z']]):
                raise ValueError('points DataFrame must have x/y/z columns.')
            points = points[['x', 'y', 'z']].values
        elif not (isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 3):
            raise TypeError('`points` must be numpy array of shape (N, 3) or '
                            'pandas DataFrame with x/y/z columns')

        # Everything happens in a temporary directory
        with tempfile.TemporaryDirectory() as tempdir:
            p = pathlib.Path(tempdir)

            # If required, copy additional files into the temporary directory
            if self.copy_files:
                for f in make_iterable(self.copy_files):
                    _ = pathlib.Path(shutil.copy(f, p))

            # Write points to file
            in_file = p / 'inputpoints.txt'
            self.write_input_file(points, in_file)

            out_file = p / 'outputpoints.txt'

            # Prepare the command
            command = [_elastixbin / 'transformix', '-out', str(p), '-tp', str(self.file), '-def', str(in_file)]

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
                logfile = p / 'transformix.log'
                if not logfile.is_file():
                    raise FileNotFoundError('No log file found.')
                with open(logfile) as f:
                    logs = f.read()
                return logs

            if not out_file.is_file():
                raise FileNotFoundError('Elastix transform did not produce any '
                                        f'output:\n {proc.stdout.decode()}')

            points_xf = self.read_output_file(out_file)

        return points_xf
