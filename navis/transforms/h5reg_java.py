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

"""Functions to use the Saalfeld lab's h5 transforms. Requires jpype."""

import os

import numpy as np
import pandas as pd

from .base import BaseTransform, trigger_init

from .. import config

# jpype is a soft dependency - defer import errors until we first try to use it
try:
    import jpype
except ImportError as e:
    jpype = None
    jpype_import_error = e
except BaseException:
    raise

# Path for the compiled transform-helpers jar which contains the required classes
fp = os.path.dirname(__file__)
jar_path = os.path.join(fp, 'jars/*')


class H5JavaTransform(BaseTransform):
    """Hdf5 transform using the Saalfeld lab's Java implementation.

    This was written mostly for validation of the pure Python implementation.
    Since it requires the compiled java code in ./jars/ which is not packaged
    with the wheel, it will only work if you cloned navis' Github repository.

    Requires ``jpype``:

        pip3 install JPype1

    See `here <https://github.com/saalfeldlab/template-building/wiki/Hdf5-Deformation-fields>`_
    for specifications of the format.

    Parameters
    ----------
    regs :          str
                    Path to h5 transformation.
    direction :     "forward" | "inverse"
                    Direction of transformation.
    level :         int
                    What level of detail to use. Negative values default to the
                    highest available resolution.

    """

    def __init__(self, reg, direction='forward', level=-1):
        """Init class."""
        assert direction in ('forward', 'inverse'), ('`direction` must be "fowrard"'
                                                     f'or "inverse", not "{direction}"')

        self.reg = reg
        self.direction = direction
        self.level = level
        self.initialized = False

    def __delayed_init__(self, ):
        """Delayed initialization."""
        # Check if jpype is present...
        if not jpype:
            # ... else raise import exception
            raise jpype_import_error

        # Check if JVM is running
        if not jpype.isJVMStarted():
            # This takes ~1 sec and we should not do this repeatedly
            # Also note that we NEED to give it the path to the
            # compiled jar
            jpype.startJVM(classpath=[jar_path])

        # This must be set before we start accessing/setting other attributes
        # that require the transform to be initialized - to prevent infinite
        # loops
        self.initialized = True

        # Import java classes
        # -> these are imported from the transform-helpers.jar
        # -> note that for some reason the "from ch. .... import" does not work
        HDF5Factory = jpype.JPackage("ch").systemsx.cisd.hdf5.HDF5Factory
        N5HDF5Reader = jpype.JPackage("org").janelia.saalfeldlab.n5.hdf5.N5HDF5Reader
        N5DisplacementField = jpype.JPackage("org").janelia.saalfeldlab.n5.imglib2.N5DisplacementField

        # Open the h5 stack
        self.hdf5Reader = HDF5Factory.openForReading(self.reg)

        # Generate n5 reader
        self.n5 = N5HDF5Reader(self.hdf5Reader, [16, 16, 16])

        # Parse/Check level
        if self.level < 0:
            self.level = self.available_levels[0]
        elif not self.n5.exists(f'/{self.level}'):
            raise ValueError(f'`level` {self.level} does not exists in n5 stack')

        # Generate transform
        path = f'/{self.level}'
        path += '/invdfield' if self.direction == 'inverse' else '/dfield'
        self.transform = N5DisplacementField.open(self.n5, path,
                                                  self.direction == 'inverse')

    def __eq__(self, other):
        """Compare with other Transform."""
        if isinstance(other, H5JavaTransform):
            if self.reg == other.reg:
                if self.direction == other.direction:
                    if self.level == other.level:
                        return True
        return False

    def __neg__(self):
        """Invert direction."""
        # Swap direction
        new_direction = {'forward': 'inverse',
                         'inverse': 'forward'}[self.direction]
        # We will re-iniatialize
        x = H5JavaTransform(self.reg, direction=new_direction, level=self.level)

        return x

    @property
    @trigger_init
    def available_levels(self):
        """Check which levels exists in h5 stack."""
        levels = []
        for i in range(0, 10):
            if self.n5.exists(f'/{i}'):
                levels.append(i)
        return levels

    def copy(self):
        """Return copy."""
        return H5JavaTransform(self.reg,
                               direction=self.direction,
                               level=int(self.level))

    @staticmethod
    def from_file(filepath, **kwargs):
        """Generate H5transform from file.

        Parameters
        ----------
        filepath :  str
                    Path to H5 transform.
        **kwargs
                    Keyword arguments passed to H5transform.__init__

        Returns
        -------
        H5transform

        """
        defaults = {'direction': 'forward'}
        defaults.update(kwargs)
        return H5JavaTransform(str(filepath), **defaults)

    @trigger_init
    def xform(self, points, progress=False):
        """Xform data.

        Parameters
        ----------
        points :        (N, 3) numpy array | pandas.DataFrame
                        Points to xform. DataFrame must have x/y/z columns.
        progress :      bool
                        Whether to show a progress bar.

        Returns
        -------
        pointsxf :      (N, 3) numpy array
                        Transformed points. Points that failed to transform will
                        be ``np.nan``.

        """
        if isinstance(points, pd.DataFrame):
            # Make sure x/y/z columns are present
            if np.any([c not in points for c in ['x', 'y', 'z']]):
                raise ValueError('points DataFrame must have x/y/z columns.')
            points = points[['x', 'y', 'z']].values

        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise TypeError('`points` must be numpy array of shape (N, 3) or '
                            'pandas DataFrame with x/y/z columns')

        # An [x, y, z] array for Java to write into
        q = jpype.JArray(jpype.JFloat, 1)([0, 0, 0])

        # It looks like we have to xform one point at a time
        xf = []
        for p in config.tqdm(points,
                             disable=not progress,
                             leave=False,
                             desc='Xforming'):
            # Xform this point
            self.transform.apply(p, q)
            # Keep track of new coordinates before moving to the next
            xf.append(list(q))

        return np.array(xf)
