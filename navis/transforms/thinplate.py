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

"""Functions to perform thin plate spline transforms. Requires morphops."""

import morphops as mops
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

from .base import BaseTransform


def distance_matrix(X,Y):
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

    Parameters
    ----------
    landmarks_source :  (M, 3) numpy array
                        Source landmarks as x/y/z coordinates.
    landmarks_target :  (M, 3) numpy array
                        Target landmarks as x/y/z coordinates.

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

    def __init__(self, landmarks_source: np.ndarray,
                 landmarks_target: np.ndarray,
                 direction: str = 'forward'):
        """Initialize class."""
        assert direction in ('forward', 'inverse')

        # Some checks
        self.source = np.asarray(landmarks_source)
        self.target = np.asarray(landmarks_target)

        if direction == 'inverse':
            self.source, self.target = self.target, self.source

        if self.source.shape[1] != 3:
            raise ValueError(f'Expected (N, 3) array, got {self.source.shape}')
        if self.target.shape[1] != 3:
            raise ValueError(f'Expected (N, 3) array, got {self.target.shape}')

        if self.source.shape[0] != self.target.shape[0]:
            raise ValueError('Number of source landmarks must match number of '
                             'target landmarks.')

        # Calculate coefficients
        self._calc_tps_coefs()

    def __eq__(self, other) -> bool:
        """Implement equality comparison."""
        if isinstance(other, TPStransform):
            if self.source.shape[0] == other.source.shape[0]:
                if np.all(self.source == other.source):
                    if np.all(self.target == other.target):
                        return True
        return False

    def __neg__(self) -> 'TPStransform':
        """Invert direction."""
        # Switch source and target
        return TPStransform(self.target, self.source)

    def _calc_tps_coefs(self):
        # Calculate thinplate coefficients
        self.W, self.A = mops.tps_coefs(self.source, self.target)

    def copy(self):
        """Make copy."""
        x = TPStransform(self.source, self.target, self.direction)

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
            if any([c not in points for c in ['x', 'y', 'z']]):
                raise ValueError('DataFrame must have x/y/z columns.')
            points = points[['x', 'y', 'z']].values

        U = mops.K_matrix(points, self.source)
        P = mops.P_matrix(points)
        # The warped pts are the affine part + the non-uniform part
        return np.matmul(P, self.A) + np.matmul(U, self.W)
