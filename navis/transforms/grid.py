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
                    Whether the field contains absolute coordinates or offsets.
                    Offsets are added to the input coordinates, while
                    coordinates returned as is.

    """

    def __init__(self, field: np.ndarray, type: str = "offsets"):
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
        self.shape = field.shape
        self.dtype = field.dtype

        # Calculate the affine part as the mean displacement across the field
        self.affine = AffineTransform(np.eye(4))
        mean_disp = np.mean(
            self.field.reshape(-1, 3), axis=0
        )  # mean displacement in x/y/z
        self.affine.matrix[0:3, 3] = mean_disp

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
                            Coordinates are expected to be in voxel space matching
                            the resolution of the deformation field.
        affine_fallback :   bool
                            If True, points that are outside the deformation field
                            will be transformed using an affine transform defined
                            by the mean displacement of the field. If False, points
                            outside the field will be set to np.nan.

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
        is_out = (points.min(axis=1) < 0) | np.any(points >= self.shape[:-1], axis=1)

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
                    xinterp(points[~is_out, :], method="linear"),
                    yinterp(points[~is_out, :], method="linear"),
                    zinterp(points[~is_out, :], method="linear"),
                )
            ).T

        return points_xf
