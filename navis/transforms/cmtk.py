#    This script is part of navis (http://www.github.com/schlegelp/navis).
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

"""Functions to use CMTK transforms."""

import copy
import functools
import os
import pathlib
import platform
import re
import subprocess

import numpy as np
import pandas as pd

from .. import utils
from .base import BaseTransform, TransformSequence

__all__ = ['xform_cmtk']

_search_path = os.environ['PATH']
_search_path = [i for i in _search_path.split(';') if len(i) > 0]
_search_path += ['~/bin',
                 '/usr/lib/cmtk/bin/',
                 '/usr/local/lib/cmtk/bin',
                 '/usr/local/bin',
                 '/opt/local/bin',
                 '/opt/local/lib/cmtk/bin/',
                 '/Applications/IGSRegistrationTools/bin']

if platform.system() == 'Windows':
    _search_path += [r'C:\cygwin64\usr\local\lib\cmtk\bin',
                     r'C:\Program Files\CMTK-3.3\CMTK\lib\cmtk\bin']


def find_cmtkbin(tool: str = 'streamxform') -> str:
    """Find directory with CMTK binaries."""
    for path in _search_path:
        path = pathlib.Path(path)
        if not path.is_dir():
            continue

        try:
            next(path.glob(tool))
            return path
        except StopIteration:
            continue
        except BaseException:
            raise


_cmtkbin = find_cmtkbin()


def requires_cmtk(func):
    """Check if CMTK is available."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _cmtkbin:
            raise ValueError("Cannot find CMTK. Please install from "
                             "http://www.nitrc.org/projects/cmtk and "
                             "make sure that it is your path!")
        return func(*args, **kwargs)
    return wrapper


@requires_cmtk
def cmtk_version(as_string=False):
    """Get CMTK version."""
    p = subprocess.run([_cmtkbin / 'streamxform', '--version'],
                       capture_output=True)
    version = p.stdout.decode('utf-8').rstrip()

    if as_string:
        return version
    else:
        return tuple(int(v) for v in version.split('.'))


@requires_cmtk
def xform_cmtk(points: np.ndarray, transforms, inverse: bool = False,
               affine_fallback: bool = False, **kwargs) -> np.ndarray:
    """Xform 3d coordinates.

    Parameters
    ----------
    points :            (N, 3) array | pandas.DataFrame
                        Points to transform. DataFrame must have x/y/z columns.
    transforms :        filepath(s) | CMTKtransform(s)
                        Either filepath to CMTK transform or ``CMTKtransform``.
                        Multiple regs must be given as list and will be applied
                        sequentially in the order provided.
    inverse :           bool | list thereof
                        Whether to invert transforms. If single boolean will
                        apply to all transforms. Can also provide ``inverse` as
                        list of booleans.
    affine_fallback :   bool
                        If True, points that failed to transform during warping
                        transform will be transformed using only the affine
                        transform.

    Returns
    -------
    pointsxf :          (N, 3) numpy.ndarray
                        Transformed points. Will contain `np.nan` for points
                        that did not transform.

    """
    transforms = list(utils.make_iterable(transforms))

    if isinstance(inverse, bool):
        inverse = [inverse] * len(transforms)

    directions = ['forward' if not i else 'inverse' for i in inverse]

    for i, r in enumerate(transforms):
        if not isinstance(r, CMTKtransform):
            if not isinstance(r, (str, pathlib.Path)):
                raise TypeError('`reg` must be filepath or CMTKtransform')
            transforms[i] = CMTKtransform(r, directions=directions[i])

    # Combine all transforms into a sequence of transforms
    seq = TransformSequence(*transforms)

    # Transform points
    xf = seq.xform(points)

    # If requested, try again with affine only for points that failed to xform
    if affine_fallback:
        isnan = np.any(np.isnan(xf), axis=1)
        if np.any(isnan):
            xf[isnan] = seq.xform(points[isnan], affine_only=True)

    return xf


class CMTKtransform(BaseTransform):
    """CMTK transforms of 3D spatial data.

    Requires `CMTK <https://www.nitrc.org/projects/cmtk/>`_ to be installed.

    Parameters
    ----------
    regs :          str | list of str
                    Path(s) to CMTK transformations(s).
    directions :    "forward" | "inverse" | list thereof
                    Direction of transformation. Must provide one direction per
                    ``reg``.
    threads :       int, optional
                    Number of threads to use.

    Examples
    --------
    >>> from navis import transforms
    >>> tr = transforms.cmtk.CMTKtransform('/path/to/CMTK_directory.list')
    >>> tr.xform(points) # doctest: +SKIP

    """

    def __init__(self, regs: list, directions: str = 'forward', threads: int = None):
        self.directions = list(utils.make_iterable(directions))
        for d in self.directions:
            assert d in ('forward', 'inverse'), ('`direction` must be "foward"'
                                                 f'or "inverse", not "{d}"')

        self.regs = list(utils.make_iterable(regs))
        self.command = 'streamxform'
        self.threads = threads

        if len(directions) == 1 and len(regs) >= 1:
            directions = directions * len(regs)

        if len(self.regs) != len(self.directions):
            raise ValueError('Must provide one direction per regs')

    def __eq__(self, other: 'CMTKtransform') -> bool:
        """Implements equality comparison."""
        if isinstance(other, CMTKtransform):
            if len(self) == len(other):
                if all([self.regs[i] == other.regs[i] for i in range(len(self))]):
                    if all([self.directions[i] == other.directions[i] for i in range(len(self))]):
                        return True
        return False

    def __len__(self) -> int:
        return len(self.regs)

    def __neg__(self) -> 'CMTKtransform':
        """Invert direction."""
        x = self.copy()

        # Swap directions
        x.directions = [{'forward': 'inverse',
                         'inverse': 'forward'}[d] for d in x.directions]

        # Reverse order
        x.regs = x.regs[::-1]
        x.directions = x.directions[::-1]

        return x

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'CMTKtransform with {len(self)} transform(s)'

    @staticmethod
    def from_file(filepath: str, **kwargs) -> 'CMTKtransform':
        """Generate CMTKtransform from file.

        Parameters
        ----------
        filepath :  str
                    Path to CMTK transform.
        **kwargs
                    Keyword arguments passed to CMTKtransform.__init__

        Returns
        -------
        CMTKtransform

        """
        defaults = {'directions': 'forward'}
        defaults.update(kwargs)
        return CMTKtransform(str(filepath), **defaults)

    def make_args(self, affine_only: bool = False) -> list:
        """Generate arguments passed to subprocess."""
        # Generate the arguments
        # The actual command (i.e. streamxform)
        args = [str(_cmtkbin / self.command)]

        if affine_only:
            args.append('--affine-only')
        else:
            args.append('--')

        if self.threads:
            args.append(f'--threads {int(self.threads)}')

        # Add the regargs
        args += self.regargs

        return args

    @property
    def regargs(self) -> list:
        """Generate regargs."""
        regargs = []
        for reg, dir in zip(self.regs, self.directions):
            if dir == 'inverse':
                regargs.append('--inverse')
            # Note no double quotes!
            regargs.append(f'{reg}')
        return regargs

    def append(self, transform: 'CMTKtransform', direction: str = None):
        """Add another transform.

        Parameters
        ----------
        transform :     str | CMTKtransform
                        Either another CMTKtransform or filepath to registration.
        direction :     "forward" | "inverse"
                        Only relevant if transform is filepath.

        """
        if isinstance(transform, CMTKtransform):
            if self.command != transform.command:
                raise ValueError('Unable to merge CMTKtransforms using '
                                 'different commands.')

            self.regs += transform.regs
            self.directions += transform.directions
        elif isinstance(transform, str):
            if not direction:
                raise ValueError('Must provide direction along with new transform')
            self.regs.append(transform)
            self.directions.append(direction)
        else:
            raise NotImplementedError(f'Unable to append {type(transform)} to {type(self)}')

    def check_if_possible(self, on_error: str = 'raise'):
        """Check if this transform is possible."""
        if not _cmtkbin:
            msg = 'Folder with CMTK binaries not found. Make sure the ' \
                  'directory is in your PATH environment variable.'
            if on_error == 'raise':
                raise BaseException(msg)
            return msg
        for r in self.regs:
            if not os.path.isdir(r) and not os.path.isfile(r):
                msg = f'Registration {r} not found.'
                if on_error == 'raise':
                    raise BaseException(msg)
                return msg

    def copy(self) -> 'CMTKtransform':
        """Return copy."""
        # Attributes not to copy
        no_copy = []
        # Generate new empty transform
        x = self.__class__(None)
        # Override with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        return x

    def parse_cmtk_output(self, output: str, fail_value=np.nan) -> np.ndarray:
        r"""Parse CMTK output.

        Briefly, CMTK output will be a byte literal like this:

            b'311 63 23 \n275 54 25 \n'

        In case of failed transforms we will get something like this where the
        original coordinates are returned with a "FAILED" flag

            b'343 72 23 \n-10 -10 -10 FAILED \n'

        Parameter
        ---------
        output :        tuple of (b'', None)
                        Stdout of CMTK call.
        fail_value
                        Value to use for points that failed to transform. By
                        default we use ``np.nan``.

        Returns
        -------
        pointsxf :      (N, 3) numpy array
                        The parse transformed points.

        """
        # The original stout is tuple where we care only about the second one
        if isinstance(output, tuple):
            output = output[0]

        pointsx = []
        # Split string into rows - lazily using a generator
        for row in (x.group(1) for x in re.finditer(r"(.*?) \n",
                                                    output.decode())):
            # Split into values
            values = row.split(' ')

            # If this point failed
            if len(values) != 3:
                values = [fail_value] * 3
            else:
                values = [float(v) for v in values]

            pointsx.append(values)

        return np.asarray(pointsx)

    def xform(self, points: np.ndarray,
              affine_only: bool = False,
              affine_fallback: bool = False) -> np.ndarray:
        """Xform data.

        Parameters
        ----------
        points :            (N, 3) numpy array | pandas.DataFrame
                            Points to xform. DataFrame must have x/y/z columns.
        affine_only :       bool
                            Whether to apply only the non-rigid affine
                            transform. This is useful if points are outside
                            the deformation field and would therefore not
                            transform properly.
        affine_fallback :   bool
                            If True and some points did not transform during the
                            non-rigid part of the transformation, we will apply
                            only the affine transformation to those points.

        Returns
        -------
        pointsxf :      (N, 3) numpy array
                        Transformed points. Points that failed to transform will
                        be ``np.nan``.

        """
        self.check_if_possible(on_error='raise')

        if isinstance(points, pd.DataFrame):
            # Make sure x/y/z columns are present
            if np.any([c not in points for c in ['x', 'y', 'z']]):
                raise ValueError('points DataFrame must have x/y/z columns.')
        elif isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] == 3:
            points = pd.DataFrame(points, columns=['x', 'y', 'z'])
        else:
            raise TypeError('`points` must be numpy array of shape (N, 3) or '
                            'pandas DataFrame with x/y/z columns')

        # Generate the result
        args = self.make_args(affine_only=affine_only)
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        # Pipe in the points
        points_str = points[['x', 'y', 'z']].to_string(index=False,
                                                       header=False,)

        # Do not use proc.stdin.write to avoid output buffer becoming full
        # before we finish piping in stdin.
        #proc.stdin.write(points_str.encode())

        # Read out results
        output = proc.communicate(input=points_str.encode())

        # If no output, something went wrong
        if not output[0]:
            raise utils.CMTKError('CMTK produced no output. Check points?')

        # Xformed points
        xf = self.parse_cmtk_output(output, fail_value=np.nan)

        # Check if any points not xformed
        if affine_fallback and not affine_only:
            not_xf = np.any(np.isnan(xf), axis=1)
            if np.any(not_xf):
                xf[not_xf] = self.xform(xf[not_xf], affine_only=True)

        return xf
