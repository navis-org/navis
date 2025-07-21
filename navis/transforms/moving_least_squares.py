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

from .base import BaseTransform


class MovingLeastSquaresTransform(BaseTransform):
    def __init__(
        self,
        landmarks_source: np.ndarray,
        landmarks_target: np.ndarray,
        direction: str = "forward",
        batch_size: int = 100_000,
    ) -> None:
        """Moving Least Squares transforms of 3D spatial data.

        Uses the [molesq](https://github.com/clbarnes/molesq) library, which packages the
        [implementation](https://github.com/ceesem/catalysis/blob/master/catalysis/transform.py)
        by Casey Schneider-Mizell of the affine algorithm published in
        [Schaefer et al. 2006](https://dl.acm.org/doi/pdf/10.1145/1179352.1141920).

        Notes
        -----
        At least in my hands, `TPStransforms` are significantly faster than
        `MovingLeastSquaresTransforms`. The results are similar but not identical,
        so make sure to use the one that works best for your use case.

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

        """
        assert direction in ("forward", "inverse")
        self.transformer = Transformer(landmarks_source, landmarks_target)
        self.reverse = direction == "inverse"
        self.batch_size = int(batch_size)

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

        batch_size = self.batch_size if self.batch_size else len(points)
        points_xf = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            points_xf.append(self.transformer.transform(batch, reverse=self.reverse))

        return np.concatenate(points_xf, axis=0)

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
