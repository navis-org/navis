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

import copy

import numpy as np

from .base import BaseTransform


class AffineTransform(BaseTransform):
    """Affine transformation of 3D spatial data.

    Parameters
    ----------
    matrix :        (4, 4) np.ndarray
                    Affine matrix.

    Examples
    --------
    A simple scaling transform

    >>> from navis import transforms
    >>> import numpy as np
    >>> M = np.diag([1e3, 1e3, 1e3, 1])
    >>> tr = transforms.affine.AffineTransform(M)
    >>> points = np.array([[0, 0, 0], [1, 1, 1]])
    >>> tr.xform(points)
    array([[   0.,    0.,    0.],
           [1000., 1000., 1000.]])

    """

    def __init__(self, matrix: np.ndarray, direction: str = 'forward'):
        """Initialize transform."""
        assert direction in ('forward', 'inverse')

        self.matrix = matrix

        if direction == 'inverse':
            self.matrix = np.linalg.inv(self.matrix)

    def __eq__(self, other: 'AffineTransform') -> bool:
        """Implements equality comparison."""
        if isinstance(other, AffineTransform):
            if np.all(self.matrix == other.matrix):
                return True
        return False

    def __neg__(self):
        """Invert direction."""
        x = self.copy()

        # Invert affine matrix
        x.matrix = np.linalg.inv(x.matrix)

        return x

    def copy(self) -> 'AffineTransform':
        """Return copy of transform."""
        # Attributes not to copy
        no_copy = []
        # Generate new empty transform
        x = self.__class__(None)
        # Override with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        return x

    def xform(self, points: np.ndarray, invert: bool = False) -> np.ndarray:
        """Apply transform to points.

        Parameters
        ----------
        points :        np.ndarray
                        (N, 3) array of x/y/z locations.
        invert :        bool
                        If True, will invert the transform.

        Returns
        -------
        pointsxf :      np.ndarray
                        The transformed points.

        """
        points = np.asarray(points)

        if points.ndim != 2 and points.shape[1] != 3:
            raise ValueError('`points` must be of shape (N, 3)')

        # Add a fourth column to points
        points_mat = np.ones((points.shape[0], 4))
        points_mat[:, :3] = points

        # Apply transform
        if not invert:
            mat = self.matrix
        else:
            mat = np.linalg.inv(self.matrix)

        return np.dot(mat, points_mat.T).T[:, :3]
