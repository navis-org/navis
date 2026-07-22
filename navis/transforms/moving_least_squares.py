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

from copy import deepcopy

import numpy as np
import pandas as pd
from molesq import Transformer

from .backends import BackendMixin, fastcore_landmarks_available
from .base import BaseTransform


class MovingLeastSquaresTransform(BackendMixin, BaseTransform):
    _fallback_backend = "python"
    _fastcore_available = staticmethod(fastcore_landmarks_available)

    def __init__(
        self,
        landmarks_source: np.ndarray,
        landmarks_target: np.ndarray,
        direction: str = "forward",
        batch_size: int = 100_000,
        backend: str = None,
    ) -> None:
        """Moving Least Squares transforms of 3D spatial data.

        Uses the [molesq](https://github.com/clbarnes/molesq) library, which packages the
        [implementation](https://github.com/ceesem/catalysis/blob/master/catalysis/transform.py)
        by Casey Schneider-Mizell of the affine algorithm published in
        [Schaefer et al. 2006](https://dl.acm.org/doi/pdf/10.1145/1179352.1141920).
        If navis-fastcore is installed, its Rust implementation of the same
        algorithm is used instead - see the `backend` parameter.

        Notes
        -----
        At least in my hands, `TPStransforms` are significantly faster than
        `MovingLeastSquaresTransforms`. The results are similar but not identical,
        so make sure to use the one that works best for your use case. The two
        backends agree with each other to ~1e-13.

        Parameters
        ----------
        landmarks_source : np.ndarray
            Source landmarks as x/y/z coordinates.
        landmarks_target : np.ndarray
            Target landmarks as x/y/z coordinates.
        direction : str
            'forward' (default) or 'inverse' (treat the target as the source and vice versa)
        batch_size : int, optional
            Batch size for transforming points. At one point during the transformation,
            molesq generates a (N, M) distance matrix, where N is the number of landmarks
            and M is the number of locations, which can get prohibitively expensive.
            We avoid the issue by batching the transformation by default. Note that the
            overhead from batching seems negligible.
            Ignored on the "fastcore" backend, which reduces over landmarks
            instead of materialising that matrix.
        backend : "auto" | "fastcore" | "python", optional
            Which implementation to use. `None` (default) defers to
            `navis.config.default_transform_backend`.

        Examples
        --------
        >>> from navis import transforms
        >>> import numpy as np
        >>> # Generate some mock landmarks
        >>> src = np.array([[0, 0, 0], [10, 10, 10], [100, 100, 100], [80, 10, 30]])
        >>> trg = np.array([[1, 15, 5], [9, 18, 21], [80, 99, 120], [5, 10, 80]])
        >>> tr = transforms.MovingLeastSquaresTransform(src, trg)
        >>> points = np.array([[0, 0, 0], [50, 50, 50]])
        >>> tr.xform(points)                                        # doctest: +SKIP
        array([[  1.        ,  15.        ,   5.        ],
               [ 81.56361725, 155.32071504, 187.3147564 ]])
        >>> # The global affine part of the transform
        >>> tr.matrix_affine.shape
        (4, 4)

        """
        assert direction in ("forward", "inverse")
        self.transformer = Transformer(landmarks_source, landmarks_target)
        self.reverse = direction == "inverse"
        self.batch_size = int(batch_size)
        self._backend = backend
        self._fc = None

    @property
    def _fastcore_mls(self):
        """The (lazily built) fastcore transform.

        Moving least squares has no fit - construction just stores the
        landmarks - so building this is cheap. It is lazy only so that an
        install without fastcore never touches the attribute.

        Always built in the forward direction; `xform` passes `reverse` per
        call. That matters because navis keeps the direction in a plain
        `self.reverse` which `__neg__` (and users) flip in place, so a cached
        object with the direction baked in would go stale.

        """
        if self._fc is None:
            from .. import utils

            self._fc = utils.fastcore.MlsTransform(
                self.transformer.control_points,
                self.transformer.deformed_control_points,
            )
        return self._fc

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
            if any([c not in points for c in ["x", "y", "z"]]):
                raise ValueError("DataFrame must have x/y/z columns.")
            points = points[["x", "y", "z"]].values

        if self.backend == "fastcore":
            # No batching here: everything but the result is a reduction over
            # landmarks, so peak memory is just the output array.
            return self._fastcore_mls.xform(
                np.asarray(points), reverse=self.reverse
            )

        batch_size = self.batch_size if self.batch_size else len(points)
        points_xf = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            points_xf.append(self.transformer.transform(batch, reverse=self.reverse))

        return np.concatenate(points_xf, axis=0)

    @property
    def matrix_affine(self):
        """Return the affine transformation matrix.

        Note that moving least squares is a *locally* weighted affine transform:
        every point effectively gets its own affine matrix. This property returns
        the *global* affine, i.e. the least-squares fit of the source onto the
        target landmarks. That is the transform the moving least squares converges
        to far away from the landmarks, where the distance weights even out.

        """
        source, target = self._control_points()
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        ndim = source.shape[1]

        # Least-squares fit of `source_homogeneous @ coefs = target`
        source_hom = np.ones((source.shape[0], ndim + 1))
        source_hom[:, :ndim] = source
        coefs = np.linalg.lstsq(source_hom, target, rcond=None)[0]

        # Combine into a typical (4, 4) transformation matrix
        # where the last row is [0, 0, 0, 1]
        m = np.eye(ndim + 1)
        m[:ndim, :ndim] = coefs[:ndim].T
        m[:ndim, ndim] = coefs[ndim]
        return m

    def __neg__(self) -> "MovingLeastSquaresTransform":
        """Invert direction"""
        out = self.copy()
        out.reverse = not self.reverse
        return out

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, MovingLeastSquaresTransform):
            return False
        for cp_this, cp_that in zip(self._control_points(), o._control_points()):
            if not np.array_equal(cp_this, cp_that):
                return False
        return True

    def _control_points(self):
        cp1 = self.transformer.control_points
        cp2 = self.transformer.deformed_control_points
        if self.reverse:
            cp2, cp1 = cp1, cp2
        return cp1, cp2

    def copy(self):
        """Make a copy"""
        return deepcopy(self)
