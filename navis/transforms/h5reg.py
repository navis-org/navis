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
from typing import Union

from .base import BaseTransform
from .affine import AffineTransform

from .. import config

logger = config.logger


class H5transform(BaseTransform):
    """Hdf5 transform of 3D spatial data.

    Requires ``jpype``:

        pip3 install JPype1

    See `here <https://github.com/saalfeldlab/template-building/wiki/Hdf5-Deformation-fields>`_
    for specifications of the format.

    Parameters
    ----------
    f :             str
                    Path to Hdf5 transformation.
    direction :     "forward" | "inverse"
                    Direction of transformation.
    level :         int
                    What level of detail to use. Negative values go backwards
                    from the highest available resolution: -1 = highest, -2 =
                    second highest, etc.
    cache :         bool
                    If True, we will cache the deformation field for subsequent
                    future transforms. This will speed up future calculations
                    in the future but comes at a memory cost.
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
            self._level = str(level)

            # Shape of deformation field
            self.shape = h5[self.level][self.field].shape

            # Data type of deformation field
            self.dtype = h5[self.level][self.field].dtype

        # Prepare cache if applicable
        if full_ingest:
            # Ingest the whole deformation field
            self.full_ingest()
        elif cache:
            self.use_cache = True

    def __eq__(self, other) -> bool:
        """Compare with other Transform."""
        if isinstance(other, H5transform):
            if self.file == other.file:
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
                        cache=self.use_cache, full_ingest=False)

        return x

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        raise ValueError('`level` cannot be changed after initialization.')

    @property
    def use_cache(self):
        """Whether to cache the deformation field."""
        if not hasattr(self, '_use_cache'):
            self._use_cache = False
        return self._use_cache

    @use_cache.setter
    def use_cache(self, value):
        """Set whether to cache the deformation field."""
        assert isinstance(value, bool)

        # If was False and now set to True, build the cache
        if not getattr(self, '_use_cache', False) and value:
            # This is the actual cache
            self.cache = np.zeros(self.shape,
                                  dtype=self.dtype)
            # This is a mask that tells us which values have already been cached
            self.cached = np.zeros(self.shape[:-1],
                                   dtype=np.bool)
            self._use_cache = True
        # If was True and now is set to False, deconstruct cache
        elif getattr(self, '_use_cache', False) and not value:
            del self.cache
            del self.cached
            self._use_cache = False

            if hasattr(self, '_fully_ingested'):
                del self._fully_ingested

        # Note: should explore whether we can use sparse N-dimensional
        # arrays for caching to save memory
        # See https://github.com/pydata/sparse/

    def copy(self):
        """Return copy."""
        return H5transform(self.file,
                           direction=self.direction,
                           level=int(self.level),
                           cache=self.use_cache,
                           full_ingest=False)

    def full_ingest(self):
        """Fully ingest the deformation field."""
        with h5py.File(self.file, 'r') as h5:
            # Read in the entire field
            self.cache = h5[self.level][self.field][:, :, :]
            # Keep a flag of this
            self._fully_ingested = True
            # We set `cached` to True instead of using a mask
            self.cached = True
            # Keep track of the caching
            self._use_cache = True

    def precache(self, bbox: Union[list, np.ndarray], padding=True):
        """Cache deformation field for given bounding box.

        Parameters
        ----------
        bbox :      list | array
                    Must be ``[[x1, x2], [y1, y2], [z1, z2]]``.
        padding :   bool
                    If True, will add the (required!) padding to the bounding
                    box.

        """
        bbox = np.asarray(bbox)

        if bbox.ndim != 2 or bbox.shape != (3, 2):
            raise ValueError(f'Expected (3, 2) bounding box, got {bbox.shape}')

        # Set use_cache=True -> this also prepares the cache array(s)
        self.use_cache = True

        with h5py.File(self.file, 'r') as h5:
            spacing = h5[self.level][self.field].attrs['spacing']

            # Note that we invert because spacing is given in (z, y, x)
            bbox_vxl = (bbox.T / spacing[::-1]).T
            # Digitize into voxels
            bbox_vxl = bbox_vxl.round().astype(int)

            if padding:
                bbox_vxl[:, 0] -= 2
                bbox_vxl[:, 1] += 2

            # Make sure we are within bounds
            bbox_vxl = np.clip(bbox_vxl.T, 0, self.shape[:-1][::-1]).T

            # Extract values
            x1, x2, y1, y2, z1, z2 = bbox_vxl.flatten()

            # Cache values in this bounding box
            self.cache[z1:z2, y1:y2, x1:x2] = h5[self.level][self.field][z1:z2, y1:y2, x1:x2]
            self.cached[z1:z2, y1:y2, x1:x2] = True

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

    def xform(self, points: np.ndarray, force_deform: bool = True) -> np.ndarray:
        """Xform data.

        Parameters
        ----------
        points :        (N, 3) numpy array | pandas.DataFrame
                        Points to xform. DataFrame must have x/y/z columns.
        force_deform :  bool
                        If True, points outside the deformation field be
                        deformed using the closest point inside the deformation
                        field.

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
            if self.use_cache and (hasattr(self, '_fully_ingested')
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
        xinterp = RegularGridInterpolator((zz, yy, xx), xgrid,
                                          bounds_error=False, fill_value=0)
        yinterp = RegularGridInterpolator((zz, yy, xx), ygrid,
                                          bounds_error=False, fill_value=0)
        zinterp = RegularGridInterpolator((zz, yy, xx), zgrid,
                                          bounds_error=False, fill_value=0)

        # Before we interpolate check how many points are outside the
        # deformation field -> these will only receive the affine part of the
        # transform
        is_out = (xf_voxel.min(axis=1) < 0) | np.any(xf_voxel >= self.shape[:-1][::-1], axis=1)

        # If more than 20% (arbitrary number) of voxels are out, there is
        # something suspicious going on
        frac_out = is_out.sum() / xf_voxel.shape[0]
        if frac_out > 0.2:
            logger.warning(f'A suspiciously a large fraction ({frac_out:.1%}) '
                           'of points appear to be outside the H5 '
                           'deformation field. Please make doubly sure '
                           'that the input coordinates are in the correct '
                           'space/units')

        # If all points are outside the volume, the interpolation complains
        if frac_out < 1 or force_deform:
            if force_deform:
                # For the purpose of finding offsets, we will snap points
                # outside the deformation field to the closest inside voxel
                q_voxel = np.clip(xf_voxel,
                                  a_min=0,
                                  a_max=np.array(self.shape[:-1][::-1]) - 1)
            else:
                q_voxel = xf_voxel

            # Interpolate coordinates and re-combine to an x/y/z array
            offset_vxl = np.vstack((xinterp(q_voxel[:, ::-1], method='linear'),
                                    yinterp(q_voxel[:, ::-1], method='linear'),
                                    zinterp(q_voxel[:, ::-1], method='linear'))).T

            # Turn offsets into real-world coordinates
            offset_real = offset_vxl * quantization_multiplier

            # Apply offsets
            # Please note that we must not use += here
            # That's to avoid running into data type errors where numpy
            # will refuse to add e.g. float64 to int64.
            # By using "+" instead of "+=" we are creating a new array that
            # is potentially upcast from e.g. int64 to float64
            xf = xf + offset_real

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
