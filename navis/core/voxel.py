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

import copy
import numbers
import pint
import warnings

import numpy as np
import pandas as pd

from typing import Union, Optional

from .. import utils, config
from .base import BaseNeuron
from .core_utils import temp_property

try:
    import xxhash
except ImportError:
    xxhash = None


__all__ = ['VoxelNeuron']

# Set up logging
logger = config.get_logger(__name__)

# This is to prevent pint to throw a warning about numpy integration
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


class VoxelNeuron(BaseNeuron):
    """Neuron represented as voxels.

    Parameters
    ----------
    x
                    Data to construct neuron from:
                     - a 2D (N, 3) array of voxel positions (x, y, z)
                     - a 2D (N, 4) array of voxel positions + values (x, y, z, value)
                     - a 3D (N, M, J) array representing the voxel grid

    offset :        (3, ) array, optional
                    An (optional) offset in voxels. This is useful to keep the
                    voxel grid small while still maintaining correct positioning
                    e.g. for plotting.
    cache :         bool
                    Whether to cache different representations (i.e. grid
                    and voxels) of the data. Set to False to save some memory.
    units :         str | pint.Units | pint.Quantity
                    Units (scales) for voxels. Defaults to ``1`` (dimensionless).
                    Strings must be parsable by pint: e.g. "nm", "um",
                    "micrometer" or "8 nanometers".
    **metadata
                    Any additional data to attach to neuron.

    """

    connectors: Optional[pd.DataFrame]

    #: (N, 3) array of x/y/z voxels locations
    voxels: np.ndarray
    #: (N, ) array of values for each voxel
    values: np.ndarray
    # (N, M, K) voxel grid
    grid: np.ndarray
    # shape of voxel grid
    shape: tuple

    soma: Optional[Union[list, np.ndarray]]

    #: Attributes used for neuron summary
    SUMMARY_PROPS = ['type', 'name', 'units', 'shape', 'dtype']

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ['name', 'shape', 'dtype']

    #: Temporary attributes that need clearing when neuron data changes
    TEMP_ATTR = ['_memory_usage', '_shape', '_voxels', '_grid']

    #: Core data table(s) used to calculate hash
    CORE_DATA = ['_data']

    def __init__(self,
                 x: Union[np.ndarray],
                 offset: Optional[np.ndarray] = None,
                 cache: bool = True,
                 units: Union[pint.Unit, str] = None,
                 **metadata
                 ):
        """Initialize Voxel Neuron."""
        super().__init__()

        if not isinstance(x, (np.ndarray, type(None))):
            raise utils.ConstructionError(f'Unable to construct VoxelNeuron from "{type(x)}".')

        if isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[1] in [3, 4]:
                # Contiguous arrays are required for hashing and we save a lot
                # of time by doing this once up-front
                self._data = np.ascontiguousarray(x)
            elif x.ndim == 3:
                self._data = np.ascontiguousarray(x)
            else:
                raise utils.ConstructionError(f'Unable to construct VoxelNeuron from {x.shape} array.')

        for k, v in metadata.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                raise AttributeError(f"Unable to set neuron's `{k}` attribute.")

        self.cache = cache
        self.units = units
        self.offset = offset

    def __getstate__(self):
        """Get state (used e.g. for pickling)."""
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}

        SKIP = []
        for s in SKIP:
            if s in state:
                _ = state.pop(s)

        return state

    def __setstate__(self, d):
        """Update state (used e.g. for pickling)."""
        self.__dict__.update(d)

    def __truediv__(self, other, copy=True):
        """Implement division for coordinates (units, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units / other).to_compact()

            n.offset = n.offset / other
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] /= other

            self._clear_temp_attr()

            return n
        return NotImplemented

    def __mul__(self, other, copy=True):
        """Implement multiplication for coordinates (units, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units * other).to_compact()

            n.offset = n.offset * other
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] *= other

            self._clear_temp_attr()

            return n
        return NotImplemented

    @property
    def _base_data_type(self) -> str:
        """Type of data (grid or voxels) underlying this neuron."""
        if self._data.ndim == 3:
            return 'grid'
        else:
            return 'voxels'

    @property
    def dtype(self) -> type:
        """Data type of voxel values."""
        return self._data.dtype

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box (includes connectors) in units."""
        mn = self.offset
        if self._base_data_type == 'voxels':
            mx = np.max(self.voxels, axis=0) * self.units.magnitude + self.offset
        else:
            mx = np.array(self.grid.shape) * self.units.magnitude + self.offset

        if self.has_connectors:
            cn_mn = np.min(self.connectors[['x', 'y', 'z']].values, axis=0)
            cn_mx = np.max(self.connectors[['x', 'y', 'z']].values, axis=0)

            mn = np.min(np.vstack((mn, cn_mn)), axis=0)
            mx = np.max(np.vstack((mx, cn_mx)), axis=0)

        return np.vstack((mn, mx)).T

    @property
    def volume(self) -> float:
        """Volume of neuron."""
        # Get volume of a single voxel
        voxel_volume = self.units_xyz[0] * self.units_xyz[2] * self.units_xyz[2]
        voxel_volume = voxel_volume.to_compact()
        return self.voxels.shape[0] * voxel_volume

    @temp_property
    def voxels(self):
        """Voxels making up the neuron."""
        if self._base_data_type == 'voxels':
            return self._data[:, :3]

        if hasattr(self, '_voxels'):
            return self._voxels

        voxels = np.dstack(np.where(self._data))[0]
        if self.cache:
            self._voxels = voxels
        return voxels

    @voxels.setter
    def voxels(self, voxels):
        if not isinstance(voxels, np.ndarray):
            raise TypeError(f'Voxels must be numpy array, got "{type(voxels)}"')
        if voxels.ndim != 2 or voxels.shape[1] != 3:
            raise ValueError('Voxels must be (N, 3) array')
        if 'float' in str(voxels.dtype):
            voxels = voxels.astype(np.int64)
        self._data = voxels
        self._clear_temp_attr()

    @temp_property
    def grid(self):
        """Voxel grid representation."""
        if self._base_data_type == 'grid':
            return self._data

        if hasattr(self, '_grid'):
            return self._grid

        grid = np.zeros(self.shape, dtype=self.values.dtype)
        grid[self._data[:, 0],
             self._data[:, 1],
             self._data[:, 2]] = self.values

        if self.cache:
            self._grid = grid
        return grid

    @grid.setter
    def grid(self, grid):
        if not isinstance(grid, np.ndarray):
            raise TypeError(f'Grid must be numpy array, got "{type(grid)}"')
        if grid.ndim != 3:
            raise ValueError('Grid must be 3D array')
        self._data = grid
        self._clear_temp_attr()

    @temp_property
    def values(self):
        """Values for each voxel (can be None)."""
        if self._base_data_type == 'grid':
            values = self._data.flatten()
            return values[values > 0]
        else:
            if not isinstance(getattr(self, '_values', None), type(None)):
                return self._values
            else:
                return np.ones(self._data.shape[0])

    @values.setter
    def values(self, values):
        if self._base_data_type == 'grid':
            raise ValueError('Unable to set values for VoxelNeurons that were '
                             'initialized with a grid')

        if isinstance(values, type(None)):
            if hasattr(self, '_values'):
                delattr(self, '_values')
            return

        if not isinstance(values, np.ndarray):
            raise TypeError(f'Values must be numpy array, got "{type(values)}"')
        elif values.ndim != 1 or values.shape[0] != self.voxels.shape[0]:
            raise ValueError('Voxels must be (N, ) array of the same length as voxels')

        self._values = values
        self._clear_temp_attr()

    @property
    def offset(self) -> np.ndarray:
        """Offset (in voxels)."""
        return self._offset

    @offset.setter
    def offset(self, offset):
        if isinstance(offset, type(None)):
            self._offset = np.array((0, 0, 0))
        else:
            offset = np.asarray(offset)
            if offset.ndim != 1 or offset.shape[0] != 3:
                raise ValueError('Offset must be (3, ) array of x/y/z coordinates.')
            self._offset = offset

        self._clear_temp_attr()

    @temp_property
    def shape(self):
        """Shape of voxel grid."""
        if not hasattr(self, '_shape'):
            if self._base_data_type == 'voxels':
                self._shape = tuple(self.voxels.max(axis=0) + 1)
            else:
                self._shape = self._data.shape
        return self._shape

    @property
    def type(self) -> str:
        """Neuron type."""
        return 'navis.VoxelNeuron'

    def copy(self) -> 'VoxelNeuron':
        """Return a copy of the neuron."""
        no_copy = ['_lock']

        # Generate new neuron
        x = self.__class__(None)
        # Override with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        return x

    def strip(self, inplace=False) -> 'VoxelNeuron':
        """Strip empty voxels (leading/trailing planes of zeros)."""
        x = self
        if not inplace:
            x = x.copy()

        # Get offset until first filled voxel
        voxels = x.voxels
        mn = voxels.min(axis=0)
        x.offset = np.array(x.offset) + mn * x.units_xyz.magnitude

        # Drop empty planes
        if x._base_data_type == 'voxels':
            x._data = voxels - mn
        else:
            mx = voxels.max(axis=0)
            x._data = x._data[mn[0]: mx[0] + 1,
                              mn[1]: mx[1] + 1,
                              mn[2]: mx[2] + 1]

        if not inplace:
            return x

    def threshold(self, threshold, inplace=False) -> 'VoxelNeuron':
        """Drop below-threshold voxels."""
        x = self
        if not inplace:
            x = x.copy()

        if x._base_data_type == 'grid':
            x._data[x._data < threshold] = 0
        else:
            x._data = x._data[x.values >= threshold]

        x._clear_temp_attr()

        if not inplace:
            return x

    def min(self) -> Union[int, float]:
        """Minimum value (excludes zeros)."""
        return self.values.min()

    def max(self) -> Union[int, float]:
        """Maximum value (excludes zeros)."""
        return self.values.max()
