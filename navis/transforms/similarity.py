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

"""Functions to calculate and apply similarity transforms."""

import copy

import numpy as np
import pandas as pd

from .base import BaseTransform


class SimilarityTransform(BaseTransform):
    """Calculate and apply a similarity transform.

    Similarity transforms are a special case of affine transforms that consist
    of a rotation, a translation and a single uniform scaling factor but no
    shear and no reflection.

    The transform is the least-squares optimal fit between the two sets of
    landmarks and is found in closed form via a singular value decomposition
    (see Umeyama, 1991). Because a similarity transform has only 7 degrees of
    freedom (3 rotation, 3 translation, 1 scale), it will generally *not* map
    the source landmarks onto the target landmarks exactly - unlike, say, a
    [`navis.transforms.thinplate.TPStransform`][].

    Parameters
    ----------
    landmarks_source :  (M, 3) array
                        Source landmarks as x/y/z coordinates.
    landmarks_target :  (M, 3) array
                        Target landmarks as x/y/z coordinates. Must have the
                        same number of landmarks as `landmarks_source`.
    weights :           (M, ) array, optional
                        Relative weight for each landmark pair. If `None`, all
                        landmarks are weighted equally.
    scale :             bool
                        Whether to also estimate a uniform scaling factor. If
                        False, the transform is a rigid transform (rotation +
                        translation only).

    Examples
    --------
    >>> from navis import transforms
    >>> import numpy as np
    >>> # Generate some mock landmarks
    >>> src = np.array([[0, 0, 0], [10, 10, 10], [100, 100, 100], [80, 10, 30]])
    >>> trg = np.array([[1, 15, 5], [9, 18, 21], [80, 99, 120], [5, 10, 80]])
    >>> tr = transforms.similarity.SimilarityTransform(src, trg)
    >>> tr.xform(np.array([[0, 0, 0], [50, 50, 50]]))
    array([[-0.27589955, 10.98503005,  4.26630007],
           [38.44936026, 53.04761397, 62.7346341 ]])

    Note that - unlike a thin-plate spline transform - the landmarks are not
    mapped exactly because a similarity transform has only 7 degrees of freedom:

    >>> round(tr.scale_factor, 4)
    0.9443

    """

    def __init__(
        self,
        landmarks_source: np.ndarray,
        landmarks_target: np.ndarray,
        weights: np.ndarray = None,
        scale: bool = True,
    ):
        self.source = landmarks_source
        self.target = landmarks_target

        if self.source.ndim != 2 or self.source.shape[1] != 3:
            raise ValueError(f"Expected (M, 3) array, got {self.source.shape}")
        if self.target.ndim != 2 or self.target.shape[1] != 3:
            raise ValueError(f"Expected (M, 3) array, got {self.target.shape}")

        if self.source.shape[0] != self.target.shape[0]:
            raise ValueError(
                "Number of source landmarks must match number of target "
                f"landmarks: {self.source.shape[0]} != {self.target.shape[0]}"
            )

        if self.n <= 1:
            raise ValueError("Need at least two landmarks.")

        if weights is None:
            weights = np.ones(self.n)
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (self.n,):
                raise ValueError(
                    f"Expected {self.n} weights, got {weights.shape[0]}"
                )
            if np.any(weights < 0):
                raise ValueError("Weights must not be negative.")
            if not weights.sum() > 0:
                raise ValueError("Weights must not sum to zero.")
        self.weights = weights

        self.scale = scale

        self.update()

    def __eq__(self, other) -> bool:
        """Implement equality comparison."""
        if not isinstance(other, SimilarityTransform):
            return False
        return (
            self.source.shape == other.source.shape
            and np.all(self.source == other.source)
            and np.all(self.target == other.target)
            and np.all(self.weights == other.weights)
            and self.scale == other.scale
        )

    def __neg__(self) -> "SimilarityTransform":
        """Invert direction."""
        # Simply switching source and target would re-fit the transform which
        # - for an imperfect fit - is not the same as inverting it. Instead we
        # invert the matrix itself.
        x = self.copy()
        x.source, x.target = self.target, self.source
        x.matrix = np.linalg.inv(self.matrix)
        return x

    @property
    def n(self) -> int:
        """Number of landmarks."""
        return self.source.shape[0]

    @property
    def source(self) -> np.ndarray:
        return self._source

    @source.setter
    def source(self, value):
        self._source = np.asarray(value, dtype=float)

    @property
    def target(self) -> np.ndarray:
        return self._target

    @target.setter
    def target(self, value):
        self._target = np.asarray(value, dtype=float)

    def update(self):
        """(Re-)calculate the transform from the current landmarks."""
        # Normalize weights so they sum to 1
        w = self.weights / self.weights.sum()

        # Weighted centroids
        mu_src = w @ self.source
        mu_trg = w @ self.target

        # Center the landmarks
        src_c = self.source - mu_src
        trg_c = self.target - mu_trg

        # Weighted cross-covariance (target x source)
        cov = (trg_c * w[:, None]).T @ src_c

        U, D, Vt = np.linalg.svd(cov)

        # Guard against the SVD producing a reflection instead of a rotation:
        # flipping the sign of the least significant singular value gives the
        # best proper rotation (det = +1).
        S = np.ones(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[-1] = -1

        R = U @ np.diag(S) @ Vt

        if self.scale:
            var_src = (w * (src_c**2).sum(axis=1)).sum()
            # If the source landmarks are all coincident there is no scale to
            # recover - fall back to 1 rather than dividing by zero.
            c = (D * S).sum() / var_src if var_src > 0 else 1.0
        else:
            c = 1.0

        matrix = np.eye(4)
        matrix[:3, :3] = c * R
        matrix[:3, 3] = mu_trg - c * R @ mu_src

        self.matrix = matrix

    @property
    def rotation(self) -> np.ndarray:
        """The (3, 3) rotation matrix."""
        return self.matrix[:3, :3] / self.scale_factor

    @property
    def scale_factor(self) -> float:
        """The uniform scaling factor."""
        # The matrix' linear part is `scale * R` and R is orthonormal, so each
        # of its columns has length `scale`.
        return float(np.linalg.norm(self.matrix[:3, 0]))

    @property
    def translation(self) -> np.ndarray:
        """The (3, ) translation vector."""
        return self.matrix[:3, 3]

    def copy(self) -> "SimilarityTransform":
        """Return copy of transform."""
        x = self.__class__.__new__(self.__class__)
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items()})
        return x

    def xform(self, points: np.ndarray) -> np.ndarray:
        """Transform points.

        Parameters
        ----------
        points :    (N, 3) array | pandas.DataFrame
                    Points to transform. DataFrames must have x/y/z columns.

        Returns
        -------
        pointsxf :  (N, 3) array
                    Transformed points.

        """
        if isinstance(points, pd.DataFrame):
            if any(c not in points for c in ["x", "y", "z"]):
                raise ValueError("DataFrame must have x/y/z columns.")
            points = points[["x", "y", "z"]].values

        points = np.asarray(points, dtype=float)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {points.shape}")

        return points @ self.matrix[:3, :3].T + self.matrix[:3, 3]
