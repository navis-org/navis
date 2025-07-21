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

"""Functions to perform thin plate spline transforms. Requires morphops."""

import morphops as mops
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

from .base import BaseTransform


def distance_matrix(X, Y):
    """For (p1,k)-shaped X and (p2,k)-shaped Y, returns the (p1,p2) matrix
    where the element at [i,j] is the distance between X[i,:] and Y[j,:].

    This is a re-implementation of a morphops function using scipy to speed
    things up ~4 orders of magnitude.
    """
    return cdist(X, Y)


# Replace morphops's original slow distance_matrix function
mops.lmk_util.distance_matrix = distance_matrix


class TPStransform(BaseTransform):
    """Thin Plate Spline transforms of 3D spatial data.

    Notes
    -----
    At least in my hands, `TPStransforms` are significantly faster than
    `MovingLeastSquaresTransforms`. The results are similar but not identical,
    so make sure to use the one that works best for your use case.

    Parameters
    ----------
    landmarks_source :  (M, 3) numpy array
                        Source landmarks as x/y/z coordinates.
    landmarks_target :  (M, 3) numpy array
                        Target landmarks as x/y/z coordinates.
    batch_size :        int, optional
                        Batch size for transforming points. The
                        thin-plate spline generating a (N, M) distance
                        matrix, where N is the number of points and M
                        is the number of source landmarks. Because
                        this can get prohibitively expensive, we're
                        batching the transformation by default.
                        Please note that the the overhead from batching
                        seems negligible.

    Examples
    --------
    >>> from navis import transforms
    >>> import numpy as np
    >>> # Generate some mock landmarks
    >>> src = np.array([[0, 0, 0], [10, 10, 10], [100, 100, 100], [80, 10, 30]])
    >>> trg = np.array([[1, 15, 5], [9, 18, 21], [80, 99, 120], [5, 10, 80]])
    >>> tr = transforms.thinplate.TPStransform(src, trg)
    >>> points = np.array([[0, 0, 0], [50, 50, 50]])
    >>> tr.xform(points)
    array([[ 1.        , 15.        ,  5.        ],
           [40.55555556, 54.        , 65.        ]])

    """

    def __init__(
        self,
        landmarks_source: np.ndarray,
        landmarks_target: np.ndarray,
        batch_size: int = 100_000,
    ):
        """Initialize class."""
        self.batch_size = batch_size
        self.source = np.asarray(landmarks_source)
        self.target = np.asarray(landmarks_target)

        # Some checks
        if self.source.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {self.source.shape}")
        if self.target.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {self.target.shape}")

        if self.source.shape[0] != self.target.shape[0]:
            raise ValueError(
                "Number of source landmarks must match number of target landmarks."
            )

        self._W, self._A = None, None

    def __eq__(self, other) -> bool:
        """Implement equality comparison."""
        if isinstance(other, TPStransform):
            if self.source.shape[0] == other.source.shape[0]:
                if np.all(self.source == other.source):
                    if np.all(self.target == other.target):
                        return True
        return False

    def __neg__(self) -> "TPStransform":
        """Invert direction."""
        # Switch source and target
        return TPStransform(self.target, self.source)

    def _calc_tps_coefs(self):
        # Calculate thinplate coefficients
        self._W, self._A = mops.tps_coefs(self.source, self.target)

    @property
    def W(self):
        if isinstance(self._W, type(None)):
            # Calculate coefficients
            self._calc_tps_coefs()
        return self._W

    @property
    def A(self):
        if isinstance(self._A, type(None)):
            # Calculate coefficients
            self._calc_tps_coefs()
        return self._A

    @property
    def matrix_rigid(self):
        """Return the rigid transformation matrix."""
        # The first row in self.A is the translation vector
        # The next 3x3 block is the rotation matrix
        # Let's combine these into a typical 4x4 transformation matrix
        # where the last row is [0, 0, 0, 1]
        m = np.zeros((4, 4))
        m[0:3, 0:3] = self.A[1:4, :].T
        m[0:3, 3] = self.A[0, :]
        m[3] = [0, 0, 0, 1]
        return m

    def copy(self):
        """Make copy."""
        x = TPStransform(self.source, self.target)

        x.__dict__.update(self.__dict__)

        return x

    def xform(self, points: np.ndarray) -> np.ndarray:
        """Transform points.

        Parameters
        ----------
        points :    (N, 3) array
                    Points to transform.

        Returns
        -------
        pointsxf :  (N, 3) array
                    Transformed points.

        """
        if isinstance(points, pd.DataFrame):
            if any(c not in points for c in ["x", "y", "z"]):
                raise ValueError("DataFrame must have x/y/z columns.")
            points = points[["x", "y", "z"]].values

        batch_size = self.batch_size if self.batch_size else points.shape[0]
        points_xf = []
        for i in range(0, points.shape[0], batch_size):
            # Get the current batch of points
            batch = points[i : i + batch_size]

            # N.B. U is of shape (N, M) where N is the number of points and M is the
            # number of source landmarks. This can get fairly expensive
            # (which is precisely why we batch the transformation)!
            U = mops.K_matrix(batch, self.source)
            P = mops.P_matrix(batch)
            # The warped pts are the affine part + the non-uniform part
            points_xf.append(np.matmul(P, self.A) + np.matmul(U, self.W))

        # Concatenate all batches
        return np.concatenate(points_xf, axis=0)
