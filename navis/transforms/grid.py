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

"""Functions for using deformation fields."""

import concurrent.futures

import numpy as np
import pandas as pd

from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from .base import BaseTransform
from .affine import AffineTransform
from .thinplate import TPStransform

from .. import config

logger = config.get_logger(__name__)


class GridTransform(BaseTransform):
    """Deformation or coordinate field transform of 3D spatial data.

    This is effectively a simpler version of the H5transform class and
    only supports deformation fields stored as numpy arrays in memory.

    Parameters
    ----------
    field :         (Nx, Ny, Nz, 3) numpy array
                    Deformation/coordinate field. The last dimension must
                    contain the x/y/z coordinates.
    type :          "offsets" | "coordinates"
                    Whether the field contains absolute coordinates or offsets/displacements.
                    Offsets are added to the input coordinates, while
                    coordinates returned as is.
    spacing :       tuple | list | numpy array, optional
                    Spacing of the deformation field in x/y/z. If not provided,
                    spacing of 1 is assumed.
    offset :        tuple | list | numpy array, optional
                    Offset of the deformation field in x/y/z. If not provided,
                    offset of 0 is assumed. N.B. offsets are applied _after_ spacing,
                    i.e. need to be provided in voxel space of the deformation field.

    """

    def __init__(
        self, field: np.ndarray, type: str = "offsets", spacing=None, offset=None
    ):
        """Init class."""
        assert (
            field.ndim == 4 and field.shape[-1] == 3
        ), "Field must be a 4D numpy array with the last dimension of size 3."
        assert type in (
            "coordinates",
            "offsets",
        ), "type must be 'coordinates' or 'offsets'."
        self.field = field
        self.type = type
        self.dtype = field.dtype
        self.spacing = spacing
        self.offset = offset

    def __eq__(self, other) -> bool:
        """Compare with other Transform."""
        if isinstance(other, GridTransform):
            if np.array_equal(self.field, other.field):
                return True
        return False

    def __neg__(self) -> "GridTransform":
        """Invert direction."""
        # Note to future self: we could implement this by computing the inverse
        # deformation field using fixed-point iteration, but that's
        # non-trivial and not needed right now.
        raise NotImplementedError("Inversion of GridTransform is not implemented.")

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, value):
        if value is None:
            self._spacing = None
        else:
            value = np.asarray(value)
            assert (
                value.ndim == 1 and value.size == 3
            ), "spacing must be a tuple/list/array of size 3."
            self._spacing = np.array(value, dtype=self.dtype)

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        if value is None:
            self._offset = None
        else:
            value = np.asarray(value)
            assert (
                value.ndim == 1 and value.size == 3
            ), "offset must be a tuple/list/array of size 3."
            self._offset = np.array(value, dtype=self.dtype)

    @property
    def affine(self) -> AffineTransform:
        """Return affine part of the transform."""
        # We're delaying the calculation of the affine part until it's needed
        if not hasattr(self, "_affine"):
            self.calculate_affine()
        return self._affine

    @property
    def shape(self) -> tuple:
        """Return shape of the deformation field."""
        return self.field.shape

    def calculate_affine(self) -> None:
        """Calculate affine part of the transform."""
        # The strategy here is this:
        # 1. Take the 8 corners of the deformation field
        # 2. Transform them using the deformation field
        # 3. Treat them as landmarks and compute the affine transform using morphops

        mx = np.array(self.shape) - 1  # max indices in each dimension
        points = np.array(
            [
                [0, 0, 0],
                [0, 0, mx[2]],
                [0, mx[1], 0],
                [0, mx[1], mx[2]],
                [mx[0], 0, 0],
                [mx[0], 0, mx[2]],
                [mx[0], mx[1], 0],
                [mx[0], mx[1], mx[2]],
            ]
        )

        if self.offset is not None:
            points = points - self.offset[np.newaxis, :]

        if self.spacing is not None:
            points = points * self.spacing[np.newaxis, :]

        points_xf = self.xform(points, affine_fallback=False)

        m = TPStransform(points, points_xf).matrix_rigid

        # Calculate the affine part as the mean displacement across the field
        self._affine = AffineTransform(m)

    def copy(self, copy_data: bool = False) -> "GridTransform":
        """Return copy."""
        if copy_data:
            return GridTransform(self.field.copy(), type=self.type)
        else:
            return GridTransform(self.field, type=self.type)

    @classmethod
    def from_file(cls, filepath: str) -> "GridTransform":
        """Create GridTransform a file.

        Parameters
        ----------
        file :          str
                        Path to file. Currently supported formats:
                          - NRRD files with deformation fields
                          - Numpy .npy files with deformation fields
                          - Nifti files (.nii, .nii.gz) with deformation fields

        Returns
        -------
        GridTransform instance.

        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist.")

        if filepath.suffix in [".npy"]:
            field = np.load(filepath)
        elif filepath.suffix in [".nii", ".nii.gz"]:
            try:
                import nibabel as nib
            except ModuleNotFoundError:
                raise ImportError(
                    "`nibabel` package is required to read Nifti files (.nii, .nii.gz)"
                )

            img = nib.load(str(filepath))
            field = img.get_fdata()
        elif filepath.suffix in [".nrrd"]:
            import nrrd

            field, _ = nrrd.read(str(filepath))
        else:
            raise ValueError(
                f"Unsupported file format: {filepath.suffix}. "
                "Supported formats are: .npy, .nii, .nii.gz, .nrrd"
            )

        return cls(field)

    @classmethod
    def from_warpfield(cls, warpfield):
        """Create GridTransform from a Warpfield deformation field.

        Parameters
        ----------
        warpfield :     warpfield.WarpMap | str
                        Warpfield WarpMap instance or path to a WarpMap h5 file.

        """
        if isinstance(warpfield, str):
            import h5py

            with h5py.File(warpfield, "r") as h5:
                wm = h5["warp_map"]
                field = wm["warp_field"][:]
                block_size = wm["block_size"][:]
                block_stride = wm["block_stride"][:]
                # mov_shape = wm["moving_shape"][:]
                # ref_shape = wm["ref_shape"][:]
        else:
            field = warpfield.warp_field
            block_size = warpfield.block_size
            block_stride = warpfield.block_stride
            # mov_shape = warpfield.mov_shape
            # ref_shape = warpfield.ref_shape

        # Reshape the field from (3, X, Y, Z) to (X, Y, Z, 3)
        field = np.moveaxis(field, 0, -1)

        spacing = block_stride
        offset = -block_size / block_stride / 2
        # Note to self regarding the offset: this code is taken straight from warpfield.

        return cls(field, type="offsets", spacing=spacing, offset=offset)

    def xform(
        self,
        points: np.ndarray,
        affine_fallback: bool = False,
    ) -> np.ndarray:
        """Xform data.

        Parameters
        ----------
        points :            (N, 3) numpy array | pandas.DataFrame
                            Points to xform. DataFrame must have x/y/z columns.
        affine_fallback :   bool
                            If True, points that are outside the deformation field
                            will be transformed using an affine transform defined by
                            the affine part of the deformation field.
                            If False, points outside the field will be set to np.nan.

        Returns
        -------
        pointsxf :          (N, 3) numpy.ndarray
                            Transformed points. Will contain `np.nan` for points
                            that did not transform.

        """
        if isinstance(points, pd.DataFrame):
            # Make sure x/y/z columns are present
            if np.any([c not in points for c in ["x", "y", "z"]]):
                raise ValueError("points DataFrame must have x/y/z columns.")
            points = points[["x", "y", "z"]].values

        if (
            not isinstance(points, np.ndarray)
            or points.ndim != 2
            or points.shape[1] != 3
        ):
            raise TypeError(
                "`points` must be numpy array of shape (N, 3) or "
                "pandas DataFrame with x/y/z columns"
            )

        points_vxl = points

        if self.spacing is not None:
            points_vxl = points_vxl / self.spacing[np.newaxis, :]

        if self.offset is not None:
            points_vxl = points_vxl + self.offset[np.newaxis, :]

        # For interpolation, we need to split the offsets into their x, y
        # and z component
        xgrid = self.field[:, :, :, 0]
        ygrid = self.field[:, :, :, 1]
        zgrid = self.field[:, :, :, 2]

        # Prepare points for interpolation
        xx = np.arange(0, self.shape[0])
        yy = np.arange(0, self.shape[1])
        zz = np.arange(0, self.shape[2])

        # The RegularGridInterpolator is the fastest but the results are
        # are ever so slightly (4th decimal) different from the Java implementation
        xinterp = RegularGridInterpolator(
            (xx, yy, zz), xgrid, bounds_error=False, fill_value=0
        )
        yinterp = RegularGridInterpolator(
            (xx, yy, zz), ygrid, bounds_error=False, fill_value=0
        )
        zinterp = RegularGridInterpolator(
            (xx, yy, zz), zgrid, bounds_error=False, fill_value=0
        )

        # Before we interpolate check how many points are outside the deformation field
        is_out = (points_vxl.min(axis=1) < 0) | np.any(points_vxl >= self.shape[:-1], axis=1)

        # Prepare output array
        if self.type == "coordinates":
            points_xf = np.zeros(points.shape, dtype=self.dtype)
        else:  # offsets
            points_xf = points.astype(self.dtype, copy=True)

        if is_out.any():
            if not affine_fallback:
                points_xf[is_out, :] = np.nan
            else:
                # Apply affine part to out-of-bounds points
                points_xf[is_out, :] = self.affine.xform(
                    points[is_out, :],
                )

        # Interpolate coordinates, re-combine to an x/y/z array and
        # add to the input points (if coordinates, the points are zeroed out above)
        if not is_out.all():
            points_xf[~is_out, :] += np.vstack(
                (
                    xinterp(points_vxl[~is_out, :], method="linear"),
                    yinterp(points_vxl[~is_out, :], method="linear"),
                    zinterp(points_vxl[~is_out, :], method="linear"),
                )
            ).T.astype(self.dtype)

        return points_xf
