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
import sparsecubes

from typing import Union, Optional

from .. import utils, config
from .base import BaseNeuron
from .core_utils import temp_property

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
    def volume(self) -> float:
        """Volume of neuron: number of filled voxels times a voxel's volume.

        Note this is *not* wrapped in `add_units`: the arithmetic below already
        produces a `pint` Quantity, so decorating it would apply the units a
        second time and report a volume in e.g. micrometer**6.
        """
        # Get volume of a single voxel
        voxel_volume = self.units_xyz[0] * self.units_xyz[1] * self.units_xyz[2]
        return (self.nnz * voxel_volume).to_compact()

    @property
    def surface_area(self) -> float:
        """Area of the neuron's exposed (blocky, staircase) surface.

        This is the area of the voxel faces that border the background, so it
        follows the voxel grid rather than a smoothed surface - meshing the
        neuron first and taking `MeshNeuron.area` will give you a somewhat
        smaller number.

        Requires sparse-cubes >= 0.4.0.

        Examples
        --------
        >>> import navis
        >>> n = navis.example_neurons(1, kind='mesh')
        >>> vx = navis.voxelize(n, pitch='2 microns')
        >>> vx.surface_area  # doctest: +SKIP
        <Quantity(21776.0, 'micrometer ** 2')>

        """
        area = sparsecubes.measure.surface_area(self.voxels, spacing=self.units_xyz.magnitude)
        return (area * self.units_xyz.units**2).to_compact()

    @property
    def centroid(self) -> np.ndarray:
        """Centre of mass of the neuron, in units (i.e. including the offset).

        Note this is the centroid of the *occupied voxels* - it is not weighted
        by their values.

        Requires sparse-cubes >= 0.4.0.
        """
        centre = sparsecubes.measure.centroid(self.voxels, spacing=self.units_xyz.magnitude)
        # `.centroid` is in the voxel grid's own frame; shift it into the same
        # space as `.bbox` and the connectors
        return centre + self.offset

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

        # Values are aligned row for row with the voxels, so they can not
        # survive an assignment that changes their number. Dropping them (after
        # which `.values` falls back to ones) is the only safe option: keeping
        # them would leave `.nnz`/`.volume` silently wrong and make `.grid`
        # raise. Note this is easy to hit now that grids auto-sparsify, i.e.
        # even a grid-constructed neuron carries `_values`.
        values = getattr(self, "_values", None)
        if values is not None and len(values) != len(voxels):
            logger.warning(
                f"Dropping this neuron's {len(values):,} values: they no "
                f"longer match the {len(voxels):,} voxels assigned. Set "
                "`.values` afterwards to attach new ones."
            )
            del self._values

        # Contiguous arrays are required for hashing (see `__init__`)
        self._data = np.ascontiguousarray(voxels)
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
                voxels = self.voxels
                canvas = getattr(self, "_canvas_shape", None)
                if not len(voxels):
                    # Empty neurons have no coordinates to reduce over. They are
                    # not exotic: an all-zero grid sparsifies to nothing, and
                    # `erode()`/`thin()`/`difference()` can consume a neuron
                    # entirely. Fall back to the canvas they were left on so
                    # `.shape` (and with it `.grid`, `.bbox`, `repr()`, ...)
                    # keeps working.
                    shape = (0, 0, 0) if canvas is None else canvas
                else:
                    shape = voxels.max(axis=0) + 1
                    if canvas is not None:
                        # Take the larger of the two so the canvas can never
                        # clip data that has grown past it
                        shape = np.maximum(shape, canvas)
                # Plain ints: the shape is user-facing (`repr`, `summary()`) and
                # numpy scalars leak into it via `_canvas_shape`
                self._shape = tuple(int(s) for s in shape)
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
        # A neuron with a zero-sized shape (i.e. empty and without a canvas)
        # has no cells to be a fraction of - report 0 rather than a nan
        n_cells = math.prod(int(s) for s in self.shape)
        return self.nnz / n_cells if n_cells else 0.0

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
        if not len(voxels):
            # Nothing to strip - and no coordinates to derive the crop from.
            # Dropping the canvas is still the right call: stripping an empty
            # neuron to its (non-existent) data leaves a zero-shaped one.
            if hasattr(x, "_canvas_shape"):
                del x._canvas_shape
                x._clear_temp_attr()
            return None if inplace else x
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

        if not x.nnz:
            # Nothing to scale - and no maximum to scale by. Bailing keeps
            # `normalize()` usable on neurons that e.g. `erode()` emptied.
            return None if inplace else x

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

    def mesh(self, **kwargs):
        """Convert this neuron into a mesh.

        Shorthand for [`navis.mesh`][] - see there for details.

        Parameters
        ----------
        **kwargs
                    Keyword arguments are passed through to
                    [`navis.conversion.voxels2mesh`][].

        Returns
        -------
        [`navis.MeshNeuron`][]
                    Note that data tables (e.g. `connectors`) are not carried
                    over from the input neuron.

        Examples
        --------
        >>> import navis
        >>> n = navis.example_neurons(1, kind='mesh')
        >>> vx = navis.voxelize(n, pitch='2 microns')
        >>> m = vx.mesh()
        >>> type(m)
        <class 'navis.core.mesh.MeshNeuron'>

        """
        from ..conversion import mesh

        return mesh(self, **kwargs)

    def skeletonize(self, **kwargs):
        """Convert this neuron into a skeleton.

        Shorthand for [`navis.skeletonize`][] - see there for details.

        Parameters
        ----------
        **kwargs
                    Keyword arguments are passed through to
                    [`navis.conversion.voxels2skeleton`][], e.g. `method`
                    ("teasar" or "thin") or `heal`.

        Returns
        -------
        [`navis.TreeNeuron`][]
                    Note that data tables (e.g. `connectors`) are not carried
                    over from the input neuron.

        Examples
        --------
        >>> import navis
        >>> n = navis.example_neurons(1, kind='mesh')
        >>> vx = navis.voxelize(n, pitch='2 microns')
        >>> sk = vx.skeletonize()
        >>> type(sk)
        <class 'navis.core.skeleton.TreeNeuron'>

        """
        from ..conversion import skeletonize

        return skeletonize(self, **kwargs)

    def distance_transform(self, spacing=None, workers: int = -1) -> np.ndarray:
        """Distance from each voxel to the nearest background voxel.

        Requires sparse-cubes >= 0.4.0.

        Parameters
        ----------
        spacing :   (3, ) array, optional
                    Voxel size to scale distances by. Defaults to this neuron's
                    units, i.e. distances come out in the neuron's own units
                    rather than in voxels. Pass `[1, 1, 1]` for voxel counts.
        workers :   int, optional
                    Number of threads to use. Default (-1) uses all cores.

        Returns
        -------
        np.ndarray
                    (N, ) array of distances, aligned row for row with
                    `.voxels` (and hence with `.values`).

        """

        if spacing is None:
            spacing = self.units_xyz.magnitude

        return sparsecubes.measure.distance_transform(self.voxels, spacing=spacing, workers=workers)

    def connected_components(self, connectivity: int = 26) -> np.ndarray:
        """Label the neuron's connected components.

        Requires sparse-cubes >= 0.4.0.

        Parameters
        ----------
        connectivity :  6 | 18 | 26, optional
                        Which neighbours count as connected: 6 = faces only,
                        18 = faces + edges, 26 = faces + edges + corners.

        Returns
        -------
        np.ndarray
                    (N, ) array of component labels (`0 .. n_components - 1`),
                    aligned row for row with `.voxels`.

        See Also
        --------
        [`navis.drop_fluff`][]
                    Uses this to remove small disconnected fragments.

        """
        _, labels = sparsecubes.measure.connected_components(self.voxels, connectivity=connectivity)
        return labels

    def dilate(
        self,
        iterations: int = 1,
        connectivity: int = 6,
        fill=None,
        inplace: bool = False,
    ) -> Optional["VoxelNeuron"]:
        """Grow the neuron by its voxel neighbourhood.

        Adds every background voxel adjacent to the neuron, `iterations` times
        over. Note that iterating a 6-connected step grows a diamond, not a
        ball - this matches `scipy.ndimage.binary_dilation`'s semantics.

        Parameters
        ----------
        iterations :    int, optional
                        Number of dilation steps.
        connectivity :  6 | 18 | 26, optional
                        Which neighbours count as adjacent: 6 = faces only,
                        18 = faces + edges, 26 = faces + edges + corners.
        fill :          int | float, optional
                        Value given to the newly added voxels. Defaults to this
                        neuron's maximum value, i.e. a binary neuron stays
                        binary. Existing voxels always keep their values.
        inplace :       bool, optional
                        If False, will return a modified copy.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                        Only if `inplace=False`.

        See Also
        --------
        [`navis.VoxelNeuron.erode`][]
                        The inverse operation.

        Examples
        --------
        >>> import navis
        >>> import numpy as np
        >>> n = navis.VoxelNeuron(np.array([[5, 5, 5]]))
        >>> n.dilate().nnz  # doctest: +SKIP
        7

        """
        out = sparsecubes.binary.dilate(
            self.voxels, iterations=iterations, connectivity=connectivity
        )
        return self._replace_voxels(out, self._carry_values(out, fill), inplace)

    def erode(
        self,
        iterations: int = 1,
        connectivity: int = 6,
        inplace: bool = False,
    ) -> Optional["VoxelNeuron"]:
        """Shrink the neuron by peeling voxels that touch the background.

        A voxel survives only if *all* of its `connectivity` neighbours are also
        part of the neuron, so the outer shell is always removed. Note this can
        consume a thin neuron entirely - use [`navis.VoxelNeuron.thin`][] if you
        want to preserve topology.

        Parameters
        ----------
        iterations :    int, optional
                        Number of erosion steps.
        connectivity :  6 | 18 | 26, optional
                        See [`navis.VoxelNeuron.dilate`][].
        inplace :       bool, optional
                        If False, will return a modified copy.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                        Only if `inplace=False`. Surviving voxels keep their
                        values.

        """
        out = sparsecubes.binary.erode(self.voxels, iterations=iterations, connectivity=connectivity)
        return self._replace_voxels(out, self._carry_values(out), inplace)

    def opening(
        self,
        iterations: int = 1,
        connectivity: int = 6,
        fill=None,
        inplace: bool = False,
    ) -> Optional["VoxelNeuron"]:
        """Erode, then dilate: strips specks and thin spurs, keeps bulk shape.

        Small structures that the erosion destroys do not come back, which makes
        this the standard way to remove surface noise before meshing or
        skeletonizing. See [`navis.VoxelNeuron.dilate`][] for the parameters.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                        Only if `inplace=False`.

        """
        out = sparsecubes.binary.opening(
            self.voxels, iterations=iterations, connectivity=connectivity
        )
        return self._replace_voxels(out, self._carry_values(out, fill), inplace)

    def closing(
        self,
        iterations: int = 1,
        connectivity: int = 6,
        fill=None,
        inplace: bool = False,
    ) -> Optional["VoxelNeuron"]:
        """Dilate, then erode: bridges narrow gaps and fills small pits.

        Note this **can fuse structures** that pass within `2 * iterations`
        voxels of each other. To fill enclosed voids without that risk use
        [`navis.VoxelNeuron.fill_cavities`][], which is topology-safe. See
        [`navis.VoxelNeuron.dilate`][] for the parameters.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                        Only if `inplace=False`.

        """
        out = sparsecubes.binary.closing(
            self.voxels, iterations=iterations, connectivity=connectivity
        )
        return self._replace_voxels(out, self._carry_values(out, fill), inplace)

    def thin(self, inplace: bool = False, **kwargs) -> Optional["VoxelNeuron"]:
        """Thin the neuron to a one-voxel-wide medial curve (centerline).

        Unlike [`navis.VoxelNeuron.erode`][] this preserves topology and
        endpoints. Note that - unlike [`navis.thin_voxels`][], which goes
        through scikit-image and a dense grid - this works straight off the
        sparse voxels and never allocates the bounding box.

        Parameters
        ----------
        inplace :   bool, optional
                    If False, will return a modified copy.
        **kwargs
                    Passed through to `sparsecubes.binary.thin`, e.g.
                    `preserve_endpoints` or `max_iterations`.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                    Only if `inplace=False`. Surviving voxels keep their values.

        """
        out = sparsecubes.binary.thin(self.voxels, **kwargs)
        return self._replace_voxels(out, self._carry_values(out), inplace)

    def fill_cavities(
        self, fill=None, inplace: bool = False, **kwargs
    ) -> Optional["VoxelNeuron"]:
        """Fill enclosed background voids.

        Unlike [`navis.VoxelNeuron.closing`][] this is topology-safe: it only
        fills voids that are actually enclosed and can never fuse two separate
        structures.

        Parameters
        ----------
        fill :      int | float, optional
                    Value given to the newly filled voxels. Defaults to this
                    neuron's maximum value.
        inplace :   bool, optional
                    If False, will return a modified copy.
        **kwargs
                    Passed through to `sparsecubes.binary.fill_cavities`, e.g.
                    `mode` or `max_cavity_size`.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                    Only if `inplace=False`.

        """
        out = sparsecubes.binary.fill_cavities(self.voxels, **kwargs)
        return self._replace_voxels(out, self._carry_values(out, fill), inplace)

    def union(
        self, *others, fill=None, inplace: bool = False
    ) -> Optional["VoxelNeuron"]:
        """Voxels present in this neuron *or* any of the others.

        Note that `others` must live on the same voxel lattice: same voxel size,
        and offsets differing by a whole number of voxels. Differing offsets are
        translated automatically; anything else raises.

        Parameters
        ----------
        *others :   VoxelNeuron | (N, 3) array
                    One or more neurons (or raw voxel indices) to combine with.
        fill :      int | float, optional
                    Value given to voxels contributed by `others`. Defaults to
                    this neuron's maximum. Voxels already in this neuron keep
                    their own values, so this neuron always wins on overlap.
        inplace :   bool, optional
                    If False, will return a modified copy.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                    Only if `inplace=False`.

        """
        sets = [self.voxels] + [self._voxels_in_frame(o) for o in others]
        out = sparsecubes.binary.union(*sets)
        return self._replace_voxels(out, self._carry_values(out, fill), inplace)

    def intersection(self, *others, inplace: bool = False) -> Optional["VoxelNeuron"]:
        """Voxels present in this neuron *and* in every one of the others.

        See [`navis.VoxelNeuron.union`][] for how neurons are aligned.

        Parameters
        ----------
        *others :   VoxelNeuron | (N, 3) array
                    One or more neurons (or raw voxel indices) to intersect with.
        inplace :   bool, optional
                    If False, will return a modified copy.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                    Only if `inplace=False`. The result is a subset of this
                    neuron, so all voxels keep their values.

        """
        sets = [self.voxels] + [self._voxels_in_frame(o) for o in others]
        out = sparsecubes.binary.intersection(*sets)
        return self._replace_voxels(out, self._carry_values(out), inplace)

    def difference(self, other, inplace: bool = False) -> Optional["VoxelNeuron"]:
        """Voxels in this neuron but not in `other` (set subtraction).

        See [`navis.VoxelNeuron.union`][] for how neurons are aligned.

        Parameters
        ----------
        other :     VoxelNeuron | (N, 3) array
                    Neuron (or raw voxel indices) to subtract.
        inplace :   bool, optional
                    If False, will return a modified copy.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                    Only if `inplace=False`. The result is a subset of this
                    neuron, so all voxels keep their values.

        """
        out = sparsecubes.binary.difference(self.voxels, self._voxels_in_frame(other))
        return self._replace_voxels(out, self._carry_values(out), inplace)

    def symmetric_difference(
        self, other, fill=None, inplace: bool = False
    ) -> Optional["VoxelNeuron"]:
        """Voxels in exactly one of this neuron and `other` (set XOR).

        See [`navis.VoxelNeuron.union`][] for how neurons are aligned.

        Parameters
        ----------
        other :     VoxelNeuron | (N, 3) array
                    Neuron (or raw voxel indices) to compare against.
        fill :      int | float, optional
                    Value given to voxels contributed by `other`. Defaults to
                    this neuron's maximum.
        inplace :   bool, optional
                    If False, will return a modified copy.

        Returns
        -------
        [`navis.VoxelNeuron`][]
                    Only if `inplace=False`.

        """
        out = sparsecubes.binary.symmetric_difference(self.voxels, self._voxels_in_frame(other))
        return self._replace_voxels(out, self._carry_values(out, fill), inplace)

    def iou(self, other) -> float:
        """Intersection over union (Jaccard index) with `other`.

        A pure overlap measure on the occupied voxels - values are ignored.
        See [`navis.VoxelNeuron.union`][] for how the two neurons are aligned.

        Parameters
        ----------
        other :     VoxelNeuron | (N, 3) array
                    Neuron (or raw voxel indices) to compare against.

        Returns
        -------
        float
                    Between 0 (disjoint) and 1 (identical).

        See Also
        --------
        [`navis.VoxelNeuron.dice`][]
                    The Dice coefficient, which weights the overlap higher.

        Examples
        --------
        >>> import navis
        >>> import numpy as np
        >>> a = navis.VoxelNeuron(np.array([[0, 0, 0], [1, 1, 1]]))
        >>> b = navis.VoxelNeuron(np.array([[1, 1, 1], [2, 2, 2]]))
        >>> a.iou(b)
        0.3333333333333333

        """
        return sparsecubes.measure.iou(self.voxels, self._voxels_in_frame(other))

    def dice(self, other) -> float:
        """Dice coefficient (F1 score) with `other`.

        Like [`navis.VoxelNeuron.iou`][] but counts the intersection twice, so
        it is more forgiving of partial overlap. Values are ignored.

        Parameters
        ----------
        other :     VoxelNeuron | (N, 3) array
                    Neuron (or raw voxel indices) to compare against.

        Returns
        -------
        float
                    Between 0 (disjoint) and 1 (identical).

        """
        return sparsecubes.measure.dice(self.voxels, self._voxels_in_frame(other))

    def _voxels_in_frame(self, other) -> np.ndarray:
        """Express `other`'s voxels in this neuron's voxel frame.

        Voxel coordinates are grid indices, so they only mean the same thing in
        two neurons whose grids share a scale and an origin. Rather than
        silently combining misaligned neurons we translate where we can and
        raise where we cannot.
        """
        if isinstance(other, np.ndarray):
            if other.ndim != 2 or other.shape[1] != 3:
                raise ValueError(
                    f"Expected (N, 3) array of voxel indices, got {other.shape}"
                )
            return other.astype(np.int64, copy=False)

        if not isinstance(other, VoxelNeuron):
            raise TypeError(
                f'Expected VoxelNeuron or (N, 3) array, got "{type(other)}"'
            )

        try:
            other_units = other.units_xyz.to(self.units_xyz.units).magnitude
            other_offset = (
                (other.offset * other.units_xyz.units)
                .to(self.units_xyz.units)
                .magnitude
            )
        except pint.DimensionalityError:
            raise ValueError(
                "Unable to combine VoxelNeurons with incompatible units: "
                f"{self.units} vs {other.units}."
            )

        if not np.allclose(other_units, self.units_xyz.magnitude):
            raise ValueError(
                "Unable to combine VoxelNeurons with different voxel sizes "
                f"({self.units} vs {other.units}). Resample one of them first."
            )

        # Offsets may differ - that just means the grids start at different
        # places - but only if they still land on the same lattice
        delta = (other_offset - self.offset) / self.units_xyz.magnitude
        if not np.allclose(delta, np.round(delta)):
            raise ValueError(
                "Unable to combine VoxelNeurons whose offsets differ by a "
                f"non-integer number of voxels (differ by {delta} voxels). "
                "Their grids do not line up."
            )

        return other.voxels + np.round(delta).astype(np.int64)

    def _carry_values(self, voxels: np.ndarray, fill=None) -> np.ndarray:
        """Values for `voxels`, taken from this neuron wherever they overlap.

        Voxels the operation invented have no value of their own and get `fill`
        (by default this neuron's maximum, so a binary neuron stays binary).
        """
        values = self.values

        if fill is None:
            fill = values.max() if len(values) else 1

        out = np.full(len(voxels), fill, dtype=values.dtype)
        ix = sparsecubes.binary.index_of(voxels, self.voxels)
        hit = ix >= 0
        out[hit] = values[ix[hit]]

        return out

    def _replace_voxels(
        self, voxels: np.ndarray, values: np.ndarray, inplace: bool
    ) -> Optional["VoxelNeuron"]:
        """Return this neuron with its data swapped for `voxels`/`values`."""
        x = self if inplace else self.copy()

        # Grab the canvas before swapping the data (same dance as `.sparsify()`)
        shape = np.array(x.shape)
        voxels = np.asarray(voxels)

        # Growing an object that touches index 0 pushes coordinates negative,
        # which a voxel grid cannot represent. Shift the frame rather than
        # dropping them: moving the origin and compensating the offset keeps the
        # neuron in the exact same physical place.
        if len(voxels):
            shift = np.minimum(voxels.min(axis=0), 0)
            if (shift < 0).any():
                voxels = voxels - shift
                x.offset = np.asarray(x.offset, dtype=float) + (
                    shift * x.units_xyz.magnitude
                )
                shape = shape - shift

        x._data = np.ascontiguousarray(voxels.astype(np.int64, copy=False))
        x._values = np.ascontiguousarray(values)

        x._clear_temp_attr()
        x._canvas_shape = tuple(shape)

        if not inplace:
            return x

    def min(self) -> Union[int, float]:
        """Minimum value (excludes zeros)."""
        return self.values.min()

    def max(self) -> Union[int, float]:
        """Maximum value (excludes zeros)."""
        return self.values.max()
