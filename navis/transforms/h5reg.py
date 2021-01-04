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

"""Functions to use the Saalfeld lab's h5 transforms."""

import concurrent.futures
import h5py

import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator

from .base import BaseTransform
from .affine import AffineTransform

from .. import config


class H5transform(BaseTransform):
    """Class to run h5 xform.

    Requires ``jpype``:

        pip3 install JPype1

    See `here <https://github.com/saalfeldlab/template-building/wiki/Hdf5-Deformation-fields>`_
    for specifications of the format.

    Parameters
    ----------
    f :             str
                    Path to h5 transformation.
    direction :     "forward" | "inverse"
                    Direction of transformation.
    level :         int
                    What level of detail to use. Negative values default to the
                    highest available resolution.
    cache :         bool
                    If True, we will cache the deformation field for subsequent
                    future transforms. This can speed up calculations in the
                    future but will take up additional memory.
    full_ingest :   bool
                    If True, will read and cache the full deformation field at
                    initialization. This additional upfront cost can pay off if
                    you are about to make many transforms across the volume.

    """

    def __init__(self, f: str, direction: str = 'forward',
                 level: int = -1, cache: bool = False,
                 full_ingest: bool = False):
        """Init class."""
        assert direction in ('forward', 'inverse'), ('`direction` must be "forward"'
                                                     f'or "inverse", not "{direction}"')

        self.file = f
        self.direction = direction
        self.field = {'forward': 'dfield', 'inverse': 'invdfield'}[direction]
        self.use_cache = cache
        self.full_ingest = full_ingest

        # Trying to avoid the file repeatedly so we are making these initial
        # adjustments all in one go even though it would be more Pythonic to
        # delegate to property getter/setter methods
        with h5py.File(self.file, 'r') as h5:
            # Get the available levels
            available_levels = []
            for k in h5.keys():
                try:
                    available_levels.append(int(k))
                except ValueError:
                    continue
            available_levels = sorted(available_levels)

            if level < 0:
                ix = level * -1 - 1
                level = available_levels[ix]

            # Set level
            self.level = str(level)

            # Shape of deformation field
            self.shape = h5[self.level][self.field].shape

            if self.full_ingest:
                # Ingest the whole deformation field
                self.cache = h5[self.level][self.field][:, :, :]
                self.use_cache = True
            elif self.use_cache:
                # Set up for cache:
                # This is the actual cache
                self.cache = np.zeros(self.shape,
                                      dtype=h5[self.level][self.field].dtype)
                # This is a mask that tells us which values have already been cached
                self.cached = np.zeros(self.shape[:-1],
                                       dtype=np.bool)

                # Note: should explore whether we can use sparse N-dimensional
                # arrays for caching to save memory
                # See https://github.com/pydata/sparse/

    def __eq__(self, other) -> bool:
        """Compare with other Transform."""
        if isinstance(other, H5transform):
            if self.reg == other.reg:
                if self.direction == other.direction:
                    if self.level == other.level:
                        return True
        return False

    def __neg__(self) -> 'H5transform':
        """Invert direction."""
        # Swap direction
        new_direction = {'forward': 'inverse',
                         'inverse': 'forward'}[self.direction]
        # We will re-iniatialize
        x = H5transform(self.file, direction=new_direction, level=int(self.level),
                        cache=self.use_cache, full_ingest=self.full_ingest)

        return x

    @staticmethod
    def from_file(filepath: str, **kwargs) -> 'H5transform':
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
        return H5transform(str(filepath), **kwargs)

    def xform(self, points: np.ndarray) -> np.ndarray:
        """Xform data.

        Parameters
        ----------
        points :        (N, 3) numpy array | pandas.DataFrame
                        Points to xform. DataFrame must have x/y/z columns.

        Returns
        -------
        pointsxf :      (N, 3) numpy array
                        Transformed points. Points outside the deformation field
                        will have only the affine part of the transform applied.

        """
        if isinstance(points, pd.DataFrame):
            # Make sure x/y/z columns are present
            if np.any([c not in points for c in ['x', 'y', 'z']]):
                raise ValueError('points DataFrame must have x/y/z columns.')
            points = points[['x', 'y', 'z']].values

        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise TypeError('`points` must be numpy array of shape (N, 3) or '
                            'pandas DataFrame with x/y/z columns')

        # Read the file
        with h5py.File(self.file, 'r') as h5:
            if 'affine' in h5[self.level][self.field].attrs:
                # The affine part of the transform is a 4 x 4 matrix where the upper
                # 3 x 4 part (row x columns) is an attribute of the h5 dataset
                M = np.ones((4, 4))
                M[:3, :4] = h5[self.level][self.field].attrs['affine'].reshape(3, 4)
                affine = AffineTransform(M)
            else:
                affine = False

            # Get quantization multiplier for later use
            quantization_multiplier = h5[self.level][self.field].attrs['quantization_multiplier']

            # For forward direction, the affine part is applied first
            if self.direction == 'inverse' and affine:
                xf = affine.xform(points)
            else:
                xf = points

            # Translate points into voxel space
            spacing = h5[self.level][self.field].attrs['spacing']
            # Note that we invert because spacing is given in (z, y, x)
            xf_voxel = xf / spacing[::-1]
            # Digitize points into voxels
            xf_indices = xf_voxel.round().astype(int)
            # Determine the bounding box of the deformation vectors we need
            # Note that we are grabbing a bit more than required - this is
            # necessary for interpolation later down the line
            mn = xf_indices.min(axis=0) - 2
            mx = xf_indices.max(axis=0) + 2

            # Make sure we are within bounds
            mn = np.clip(mn, 0, self.shape[:-1][::-1])
            mx = np.clip(mx, 0, self.shape[:-1][::-1])

            # Check if we can use cached values
            if self.use_cache and (self.full_ingest
                                   or np.all(self.cached[mn[2]: mx[2],
                                                         mn[1]: mx[1],
                                                         mn[0]: mx[0]])):
                offsets = self.cache[mn[2]: mx[2], mn[1]: mx[1], mn[0]: mx[0]]
            else:
                # Load the deformation values for this bounding box
                # This is faster than grabbing individual voxels and
                offsets = h5[self.level][self.field][mn[2]: mx[2],
                                                     mn[1]: mx[1],
                                                     mn[0]: mx[0]]

                if self.use_cache:
                    # Write these offsets to cache
                    self.cache[mn[2]: mx[2], mn[1]: mx[1], mn[0]: mx[0]] = offsets
                    self.cached[mn[2]: mx[2], mn[1]: mx[1], mn[0]: mx[0]] = True

        # For interpolation, we need to split the offsets into their x, y
        # and z component
        xgrid = offsets[:, :, :, 0]
        ygrid = offsets[:, :, :, 1]
        zgrid = offsets[:, :, :, 2]

        xx = np.arange(mn[0], mx[0])
        yy = np.arange(mn[1], mx[1])
        zz = np.arange(mn[2], mx[2])

        # The RegularGridInterpolator is the fastest one but the results are
        # are ever so slightly (4th decimal) different from the Java impl
        xinterp = RegularGridInterpolator((zz, yy, xx),
                                          xgrid)
        yinterp = RegularGridInterpolator((zz, yy, xx),
                                          ygrid)
        zinterp = RegularGridInterpolator((zz, yy, xx),
                                          zgrid)

        # Interpolate coordinats and re-combine to an x/y/z array
        offset_vxl = np.vstack((xinterp(xf_voxel[:, [2, 1, 0]], method='linear'),
                                yinterp(xf_voxel[:, [2, 1, 0]], method='linear'),
                                zinterp(xf_voxel[:, [2, 1, 0]], method='linear'))).T

        # Turn offsets into real-world coordinates
        offset_real = offset_vxl * quantization_multiplier

        # Apply offsets
        has_offset = ~np.any(np.isnan(offset_real), axis=1)
        xf[has_offset] += offset_real[has_offset]

        # For inverse direction, the affine part is applied second
        if self.direction == 'forward' and affine:
            xf = affine.xform(xf)

        return xf


def read_points_threaded(voxels, filepath, level, dir, threads=5):
    """Some tinkering with using multiple processes to read voxels."""
    splits = np.array_split(voxels, threads)
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        chunks = [(arr, filepath, level, dir) for arr in splits]
        futures = executor.map(_read_points, chunks)
        offset = np.vstack(list(futures))

    return offset


def _read_points(params):
    voxels, filepath, level, dir = params
    f = h5py.File(filepath, 'r', libver='latest', swmr=True)
    # Get all these voxels
    data = []
    for vx in config.tqdm(voxels):
        data.append(f[level][dir][vx[2], vx[1], vx[0]])
    return np.vstack(data)
