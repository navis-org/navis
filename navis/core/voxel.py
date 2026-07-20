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
import math
import numbers
import pint
import warnings

import numpy as np
import pandas as pd

from typing import Union, Optional

from .. import utils, config
from .base import BaseNeuron
from .core_utils import temp_property, add_units

try:
    import xxhash
except ModuleNotFoundError:
    xxhash = None


__all__ = ["VoxelNeuron"]

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
                    An (optional) offset in units (not voxels), i.e. the
                    position of voxel (0, 0, 0). This is useful to keep the
                    voxel grid small while still maintaining correct positioning
                    e.g. for plotting.
    cache :         bool
                    Whether to cache different representations (i.e. grid
                    and voxels) of the data. Set to False to save some memory.
    units :         str | pint.Units | pint.Quantity
                    Units (scales) for voxels. Defaults to `1` (dimensionless).
                    Strings must be parsable by pint: e.g. "nm", "um",
                    "micrometer" or "8 nanometers".
    sparsify :      "auto" | bool
                    Whether to store a dense grid as sparse voxels (see
                    [`navis.VoxelNeuron.sparsify`][]). Neurons are typically
                    sparse enough that this saves a lot of memory - the example
                    neurons here shrink by 7-46x. Ignored if `x` is already
                    sparse.
                     - "auto" (default) sparsifies only if it cuts memory by at
                       least `SPARSIFY_MARGIN`
                     - True always sparsifies
                     - False keeps the dense grid
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
    SUMMARY_PROPS = ["type", "name", "units", "shape", "dtype"]

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ["name", "shape", "dtype"]

    #: Temporary attributes that need clearing when neuron data changes
    TEMP_ATTR = ["_memory_usage", "_shape", "_voxels", "_grid"]

    #: Core data table(s) used to calculate hash
    CORE_DATA = ["_data", "_values"]

    #: How much smaller the sparse representation must be before `sparsify="auto"`
    #: switches to it. The margin keeps neurons near the break-even point from
    #: flipping representation on trivial differences.
    SPARSIFY_MARGIN = 2

    def __init__(
        self,
        x: Union[np.ndarray],
        offset: Optional[np.ndarray] = None,
        cache: bool = True,
        units: Union[pint.Unit, str] = None,
        sparsify: Union[str, bool] = "auto",
        **metadata,
    ):
        """Initialize Voxel Neuron."""
        super().__init__()

        if not isinstance(x, (np.ndarray, type(None))):
            raise utils.ConstructionError(
                f'Unable to construct VoxelNeuron from "{type(x)}".'
            )

        if isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[1] in [3, 4]:
                # Contiguous arrays are required for hashing and we save a lot
                # of time by doing this once up-front.
                # Coordinates and values are kept in separate arrays because
                # they will typically have different dtypes.
                coords = x[:, :3]
                # Coordinates must be integers to be usable as grid indices
                # (same as in the `voxels` setter). Note this matters in
                # particular for (N, 4) inputs where float values would
                # otherwise force the coordinates to float as well.
                if "float" in str(coords.dtype):
                    coords = coords.astype(np.int64)
                self._data = np.ascontiguousarray(coords)
                if x.shape[1] == 4:
                    self._values = np.ascontiguousarray(x[:, 3])
            elif x.ndim == 3:
                self._data = np.ascontiguousarray(x)
            else:
                raise utils.ConstructionError(
                    f"Unable to construct VoxelNeuron from {x.shape} array."
                )

        for k, v in metadata.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                raise AttributeError(f"Unable to set neuron's `{k}` attribute.")

        self.cache = cache
        self.units = units
        self.offset = offset

        if sparsify not in ("auto", True, False):
            raise ValueError(
                f'`sparsify` must be "auto", True or False, got "{sparsify}"'
            )

        # Note this has to happen last: `.sparsify()` recomputes the core hash
        # (via `_clear_temp_attr`) and that must see the finished neuron.
        # Voxel-backed neurons are already sparse, hence the data type check.
        if (
            sparsify is not False
            and getattr(self, "_data", None) is not None
            and self._base_data_type == "grid"
            and (sparsify is True or self._sparsify_saves_memory())
        ):
            # This grid exists, so it fits in memory whatever `max_grid_size`
            # says. Record that before dropping it: `.grid` must be able to hand
            # it back rather than refusing to rebuild what we were just given.
            # Deliberately not a TEMP_ATTR - it outlives the data it describes.
            self._materialized_nbytes = self.grid_nbytes
            self.sparsify(inplace=True)

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
        """Implement division for coordinates (units, connectors, offset)."""
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
                # Note: reassign (instead of in-place /=) so that integer
                # connector coordinates can be cast to float if necessary
                n.connectors[["x", "y", "z"]] = n.connectors[["x", "y", "z"]] / other

            self._clear_temp_attr()

            return n
        return NotImplemented

    def __mul__(self, other, copy=True):
        """Implement multiplication for coordinates (units, connectors, offset)."""
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
                # Note: reassign (instead of in-place *=) so that integer
                # connector coordinates can be cast to float if necessary
                n.connectors[["x", "y", "z"]] = n.connectors[["x", "y", "z"]] * other

            self._clear_temp_attr()

            return n
        return NotImplemented

    def __add__(self, other, copy=True):
        """Implement addition for coordinates (offset, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self

            n.offset = n.offset + other
            if n.has_connectors:
                # Note: reassign (instead of in-place +=) so that integer
                # connector coordinates can be cast to float if necessary
                n.connectors[["x", "y", "z"]] = n.connectors[["x", "y", "z"]] + other

            self._clear_temp_attr()

            return n
        return NotImplemented

    def __sub__(self, other, copy=True):
        """Implement subtraction for coordinates (offset, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self

            n.offset = n.offset - other
            if n.has_connectors:
                # Note: reassign (instead of in-place -=) so that integer
                # connector coordinates can be cast to float if necessary
                n.connectors[["x", "y", "z"]] = n.connectors[["x", "y", "z"]] - other

            self._clear_temp_attr()

            return n
        return NotImplemented

    @property
    def _base_data_type(self) -> str:
        """Type of data (grid or voxels) underlying this neuron."""
        if self._data.ndim == 3:
            return "grid"
        else:
            return "voxels"

    @property
    def dtype(self) -> type:
        """Data type of voxel values."""
        if self._base_data_type == "grid":
            return self._data.dtype
        # For voxel-backed neurons it's the values - not the coordinates -
        # that determine the dtype. Note that we avoid going through `.values`
        # here because that would materialize the default array of ones.
        values = getattr(self, "_values", None)
        return np.dtype(float) if values is None else values.dtype

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box (includes connectors) in units."""
        mn = self.offset
        # Note that `.shape` is the shape of the voxel grid for either backing
        # and hence gives the same bounding box for both
        mx = np.array(self.shape) * self.units_xyz.magnitude + self.offset

        if self.has_connectors:
            cn_mn = np.min(self.connectors[["x", "y", "z"]].values, axis=0)
            cn_mx = np.max(self.connectors[["x", "y", "z"]].values, axis=0)

            mn = np.min(np.vstack((mn, cn_mn)), axis=0)
            mx = np.max(np.vstack((mx, cn_mx)), axis=0)

        return np.vstack((mn, mx)).T

    @property
    @add_units(compact=True, power=3)
    def volume(self) -> float:
        """Volume of neuron."""
        # Get volume of a single voxel
        voxel_volume = self.units_xyz[0] * self.units_xyz[1] * self.units_xyz[2]
        return (self.nnz * voxel_volume).to_compact()

    @property
    @temp_property
    def voxels(self):
        """Voxels making up the neuron."""
        if self._base_data_type == "voxels":
            return self._data[:, :3]

        if hasattr(self, "_voxels"):
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
            raise ValueError("Voxels must be (N, 3) array")
        if "float" in str(voxels.dtype):
            voxels = voxels.astype(np.int64)
        self._data = voxels
        self._clear_temp_attr()

    @property
    @temp_property
    def grid(self):
        """Voxel grid representation."""
        if self._base_data_type == "grid":
            return self._data

        if hasattr(self, "_grid"):
            return self._grid

        # The grid's shape is derived from the voxel coordinates, so it can be
        # orders of magnitude larger than the sparse data we actually hold.
        # Exception: a grid we were handed at construction demonstrably fits in
        # memory, so rebuilding it must never fail - see `_materialized_nbytes`.
        # Comparing sizes (rather than trusting a flag) means anything that
        # grows the grid - more voxels, a promoted dtype - re-engages the guard.
        if self.grid_nbytes > getattr(self, "_materialized_nbytes", 0):
            utils.check_grid_size(
                self.shape,
                self.dtype,
                hint=(
                    f"This neuron holds only {self.nnz:,} non-zero voxels: consider "
                    "working with `.voxels`/`.values` instead of `.grid`, or use "
                    "`.strip()`/`navis.downsample_neuron()` to shrink the grid."
                ),
            )

        grid = np.zeros(self.shape, dtype=self.values.dtype)
        grid[self._data[:, 0], self._data[:, 1], self._data[:, 2]] = self.values

        if self.cache:
            self._grid = grid
        return grid

    @grid.setter
    def grid(self, grid):
        if not isinstance(grid, np.ndarray):
            raise TypeError(f'Grid must be numpy array, got "{type(grid)}"')
        if grid.ndim != 3:
            raise ValueError("Grid must be 3D array")
        self._data = grid
        self._clear_temp_attr()

    @property
    @temp_property
    def values(self):
        """Values for each voxel (can be None)."""
        if self._base_data_type == "grid":
            # Note: "filled" means non-zero, not positive. `.voxels`, `.nnz` and
            # `count_nonzero()` all use non-zero, so using `> 0` here would make
            # `.values` shorter than `.voxels` for grids holding negative values.
            # Note: boolean-index the grid directly instead of flattening it
            # first - `.flatten()` copies the entire grid, which for a large
            # neuron doubles peak memory for no reason. Both give the filled
            # values in C order.
            return self._data[self._data != 0]
        else:
            if not isinstance(getattr(self, "_values", None), type(None)):
                return self._values
            else:
                return np.ones(self._data.shape[0])

    @values.setter
    def values(self, values):
        if self._base_data_type == "grid":
            raise ValueError(
                "Unable to set values for VoxelNeurons that were "
                "initialized with a grid"
            )

        if isinstance(values, type(None)):
            if hasattr(self, "_values"):
                delattr(self, "_values")
            return

        if not isinstance(values, np.ndarray):
            raise TypeError(f'Values must be numpy array, got "{type(values)}"')
        elif values.ndim != 1 or values.shape[0] != self.voxels.shape[0]:
            raise ValueError("Voxels must be (N, ) array of the same length as voxels")

        self._values = values
        self._clear_temp_attr()

    @property
    def offset(self) -> np.ndarray:
        """Offset in units (not voxels), i.e. the position of voxel (0, 0, 0)."""
        return self._offset

    @offset.setter
    def offset(self, offset):
        if isinstance(offset, type(None)):
            self._offset = np.array((0, 0, 0))
        else:
            offset = np.asarray(offset)
            if offset.ndim != 1 or offset.shape[0] != 3:
                raise ValueError("Offset must be (3, ) array of x/y/z coordinates.")
            self._offset = offset

        self._clear_temp_attr()

    @property
    @temp_property
    def shape(self):
        """Shape of voxel grid."""
        if not hasattr(self, "_shape"):
            if self._base_data_type == "voxels":
                # Coordinates only give the data's bounding box. A neuron that
                # came from a grid (or was flipped) has a canvas that may be
                # larger - e.g. trailing empty planes - and losing it would
                # silently reshape the neuron. Note `_canvas_shape` deliberately
                # lives outside TEMP_ATTR: `_shape` alone gets cleared on the
                # next data change and the canvas would not survive it.
                shape = self.voxels.max(axis=0) + 1
                canvas = getattr(self, "_canvas_shape", None)
                if canvas is not None:
                    # Take the larger of the two so the canvas can never clip
                    # data that has grown past it
                    shape = np.maximum(shape, canvas)
                self._shape = tuple(shape)
            else:
                self._shape = self._data.shape
        return self._shape

    @property
    def grid_nbytes(self) -> int:
        """Size of the dense voxel grid in bytes.

        For voxel-backed neurons this is the size the grid *would* have if
        materialized - which, because the shape is derived from the voxel
        coordinates, can be orders of magnitude larger than the neuron's actual
        memory footprint. Use this to check before touching `.grid`.
        """
        # Python ints: the product overflows int64 for pathologically sparse
        # neurons, which are exactly the ones worth checking
        return math.prod(int(s) for s in self.shape) * self.dtype.itemsize

    @property
    def voxels_nbytes(self) -> int:
        """Size of the sparse voxel representation in bytes.

        For grid-backed neurons this is the size the voxels *would* have if
        materialized. Compare against `.grid_nbytes` to see which representation
        is the more compact one for this neuron.
        """
        # Coordinates are int64 (3 per voxel, see `.sparsify()`) and sit
        # alongside one value per voxel
        return self.nnz * (3 * np.dtype(np.int64).itemsize + self.dtype.itemsize)

    def _sparsify_saves_memory(self) -> bool:
        """Whether sparsifying would cut memory by at least `SPARSIFY_MARGIN`.

        Note this is deliberately a memory comparison rather than a density
        cutoff: the break-even density depends on the dtype (~0.04 for bool but
        ~0.25 for float64), so a single density threshold would be wrong for
        most dtypes.
        """
        # Counting non-zeros is a single vectorized pass with no allocation and
        # costs ~1% of the sparsify it gates - cheap enough to always run
        return self.voxels_nbytes * self.SPARSIFY_MARGIN <= self.grid_nbytes

    @property
    def type(self) -> str:
        """Neuron type."""
        return "navis.VoxelNeuron"

    @property
    def density(self) -> float:
        """Fraction of filled voxels."""
        return self.nnz / np.prod(self.shape)

    @property
    def nnz(self) -> int:
        """Number of non-zero voxels."""
        return self.count_nonzero()

    def count_nonzero(self) -> int:
        """Count non-zero voxels."""
        if self._base_data_type == "grid":
            return np.count_nonzero(self.grid)
        elif self._base_data_type == "voxels":
            return np.count_nonzero(self.values)

        raise TypeError(f"Unexpected data type: {self._base_data_type}")

    def convert_units(
        self, to: Union[pint.Unit, str], inplace: bool = False
    ) -> Optional["VoxelNeuron"]:
        """Convert coordinates to different unit.

        For `VoxelNeurons` this is purely a re-labelling: the voxel grid is
        left untouched (its coordinates are indices, not coordinates) and the
        neuron keeps occupying the exact same space.

        Note that this overrides the base implementation which works by
        scaling the neuron's coordinates - something we can not do here.

        Parameters
        ----------
        to :        pint.Unit | str
                    Units to convert to. If string, must be parsable by pint.
                    See examples.
        inplace :   bool, optional
                    If True will convert in place. If not will return a
                    copy.

        Examples
        --------
        >>> import navis
        >>> n = navis.example_neurons(1, kind='mesh')
        >>> vx = navis.voxelize(n, pitch='1 micron')
        >>> vx.units
        <Quantity(1000.0, 'nanometer')>
        >>> vx.convert_units('um').units
        <Quantity(1.0, 'micrometer')>

        """
        n = self if inplace else self.copy()

        old = n.units_xyz
        new = old.to(to)

        # The offset is carried in the same base unit as `.units` and hence
        # has to be converted along with it
        n.offset = (n.offset * old.units).to(new.units).magnitude
        n.units = [str(u) for u in new]

        return n

    def copy(self, deepcopy=False) -> "VoxelNeuron":
        """Return a copy of the neuron."""
        copy_fn = copy.deepcopy if deepcopy else copy.copy
        no_copy = ["_lock"]

        # Generate new neuron
        x = self.__class__(None)
        # Override with this neuron's data
        x.__dict__.update(
            {k: copy_fn(v) for k, v in self.__dict__.items() if k not in no_copy}
        )

        return x

    def flip(self, axis: str, inplace: bool = False) -> Optional["VoxelNeuron"]:
        """Flip the volume along the specified axis.

        Parameters
        ----------
        axis :       "x" | "y" | "z"
                    Axis to flip along.
        inplace :   bool, optional
                    If False, will return flipped copy.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                    Flipped copy of original neuron. Only if `inplace=False`.

        """
        assert axis in ("x", "y", "z"), (
            f'Unknown axis "{axis}". Allowed axes: "x", "y", "z"'
        )

        x = self
        if not inplace:
            x = x.copy()

        ix = "xyz".index(axis)

        # Grab the shape before modifying the data: for voxel-backed neurons
        # it is derived from the voxel coordinates
        shape = x.shape

        # Flip voxels
        if x._base_data_type == "voxels":
            x._data[:, ix] = shape[ix] - 1 - x._data[:, ix]
        else:
            x._data = np.flip(x._data, axis=ix)

        # Note that the offset stays as is: flipping the data in place leaves
        # the neuron occupying the exact same bounding box

        # Flip connectors. Note that - unlike voxels - connectors are in units
        # and hence have to be mirrored across the grid's bounding box.
        if x.has_connectors:
            mx = x.offset[ix] + (shape[ix] - 1) * x.units_xyz.magnitude[ix]
            x.connectors.loc[:, axis] = mx + x.offset[ix] - x.connectors[axis]

        x._clear_temp_attr()

        # For voxel-backed neurons the shape is derived from the coordinates,
        # so flipping would shrink it whenever the trailing planes are empty.
        # Restore it explicitly to keep the neuron on the same canvas - without
        # this, flipping twice does not get you back to the original.
        x._canvas_shape = tuple(shape)

        if not inplace:
            return x

    def strip(self, inplace=False) -> "VoxelNeuron":
        """Strip empty voxels (leading/trailing planes of zeros)."""
        x = self
        if not inplace:
            x = x.copy()

        # Get offset until first filled voxel
        voxels = x.voxels
        mn = voxels.min(axis=0)
        x.offset = np.array(x.offset) + mn * x.units_xyz.magnitude

        # Shrinking the canvas to the data is the whole point of `strip()`, so
        # any previously preserved canvas has to go - otherwise `.shape` would
        # keep reporting the empty planes we just dropped
        if hasattr(x, "_canvas_shape"):
            del x._canvas_shape

        # Drop empty planes
        if x._base_data_type == "voxels":
            x._data = voxels - mn
        else:
            mx = voxels.max(axis=0)
            x._data = x._data[mn[0] : mx[0] + 1, mn[1] : mx[1] + 1, mn[2] : mx[2] + 1]

        x._clear_temp_attr()

        if not inplace:
            return x

    def threshold(self, threshold, inplace=False) -> "VoxelNeuron":
        """Drop below-threshold voxels."""
        x = self
        if not inplace:
            x = x.copy()

        if x._base_data_type == "grid":
            x._data[x._data < threshold] = 0
        else:
            keep = x.values >= threshold
            x._data = x._data[keep]
            if getattr(x, "_values", None) is not None:
                x._values = x._values[keep]

        x._clear_temp_attr()

        if not inplace:
            return x

    def normalize(self, inplace=False) -> "VoxelNeuron":
        """Normalize voxel values.

        For float data, this scales values to the [0-1] range.
        For integer data, this scales values to the [0-max]
        range depending on the datatype.
        """
        mx = np.iinfo(self.dtype).max if np.issubdtype(self.dtype, np.integer) else 1.0

        x = self
        if not inplace:
            x = x.copy()

        if x._base_data_type == "grid":
            x._data = (x._data / x._data.max()) * mx
        else:
            # Note that we must scale the values, not the coordinates
            x.values = (x.values / x.values.max()) * mx

        x._clear_temp_attr()

        if not inplace:
            return x

    def densify(self, inplace: bool = False) -> Optional["VoxelNeuron"]:
        """Convert to a dense grid representation.

        VoxelNeurons are backed by either a dense 3D grid or a sparse (N, 3)
        array of voxel coordinates plus values. Both views are available via
        `.grid` and `.voxels`/`.values` whatever the backing - this method
        changes which one is actually stored. See
        [`navis.VoxelNeuron.sparsify`][] for the inverse.

        Note that the grid's shape is derived from the voxel coordinates and
        can be orders of magnitude larger than the sparse data it replaces.
        Check `.grid_nbytes` before densifying a sparse neuron.

        Parameters
        ----------
        inplace :   bool, optional
                    If False, will return a densified copy.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                    Densified copy of original neuron. Only if `inplace=False`.

        Examples
        --------
        >>> import navis
        >>> import numpy as np
        >>> n = navis.VoxelNeuron(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
        >>> n._base_data_type
        'voxels'
        >>> dense = n.densify()
        >>> dense._base_data_type
        'grid'
        >>> dense.shape
        (3, 3, 3)

        """
        x = self
        if not inplace:
            x = x.copy()

        if x._base_data_type != "grid":
            # Going through `.grid` re-uses the memory guard (and any cached
            # grid) instead of re-implementing the conversion here
            x._data = x.grid

            # Grid-backed neurons carry their values in the grid itself. A
            # leftover `_values` would be ignored by `.values` but still feeds
            # into the core hash (see `CORE_DATA`), so it has to go.
            if hasattr(x, "_values"):
                delattr(x, "_values")

            x._clear_temp_attr()

        if not inplace:
            return x

    def sparsify(self, inplace: bool = False) -> Optional["VoxelNeuron"]:
        """Convert to a sparse voxel representation.

        VoxelNeurons are backed by either a dense 3D grid or a sparse (N, 3)
        array of voxel coordinates plus values. Both views are available via
        `.grid` and `.voxels`/`.values` whatever the backing - this method
        changes which one is actually stored. See
        [`navis.VoxelNeuron.densify`][] for the inverse.

        Parameters
        ----------
        inplace :   bool, optional
                    If False, will return a sparsified copy.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                    Sparsified copy of original neuron. Only if `inplace=False`.

        Examples
        --------
        >>> import navis
        >>> import numpy as np
        >>> grid = np.zeros((3, 3, 3))
        >>> grid[0, 0, 0] = grid[1, 1, 1] = 1
        >>> n = navis.VoxelNeuron(grid, sparsify=False)  # keep the dense grid
        >>> n._base_data_type
        'grid'
        >>> sparse = n.sparsify()
        >>> sparse._base_data_type
        'voxels'
        >>> sparse.voxels
        array([[0, 0, 0],
               [1, 1, 1]])
        >>> sparse.shape  # the grid's shape is preserved
        (3, 3, 3)

        """
        x = self
        if not inplace:
            x = x.copy()

        if x._base_data_type != "voxels":
            # Grab the shape before swapping the data: for voxel-backed neurons
            # it is derived from the coordinates and would otherwise shrink
            # whenever the trailing planes are empty (same dance as in `.flip()`)
            shape = x.shape

            # Note we derive coordinates and values from a single mask rather
            # than going through `.voxels`/`.values`: that is one pass instead
            # of two and makes it structurally impossible for the two to drift
            # out of sync.
            mask = x._data != 0
            # Both must be read off the grid *before* `_data` is reassigned -
            # afterwards the neuron is voxel-backed and `_data` is coordinates
            coords, values = np.argwhere(mask), x._data[mask]

            # Contiguous arrays are required for hashing (see `__init__`)
            x._data = np.ascontiguousarray(coords)
            x._values = np.ascontiguousarray(values)

            x._clear_temp_attr()

            # Restore the original canvas - without this a densify/sparsify
            # round-trip can silently shrink the neuron
            x._canvas_shape = tuple(shape)

        if not inplace:
            return x

    def min(self) -> Union[int, float]:
        """Minimum value (excludes zeros)."""
        return self.values.min()

    def max(self) -> Union[int, float]:
        """Maximum value (excludes zeros)."""
        return self.values.max()
