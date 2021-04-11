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
        direction: str = 'forward',
    ) -> None:
        """Moving Least Squares transforms of 3D spatial data.

        Uses the
        `molesq <https://github.com/clbarnes/molesq>`_
        library, which packages the
        `implementation by Casey Schneider-Mizell <https://github.com/ceesem/catalysis/blob/master/catalysis/transform.py>`_
        of the affine algorithm published in
        `Schaefer et al. 2006 <https://dl.acm.org/doi/pdf/10.1145/1179352.1141920>`_.

        Parameters
        ----------
        landmarks_source : np.ndarray
            Source landmarks as x/y/z coordinates.
        landmarks_target : np.ndarray
            Target landmarks as x/y/z coordinates.
        direction : str
            'forward' (default) or 'inverse' (treat the target as the source and vice versa)

        Examples
        --------
        >>> from navis import transforms
        >>> import numpy as np
        >>> # Generate some mock landmarks
        >>> src = np.array([[0, 0, 0], [10, 10, 10], [100, 100, 100], [80, 10, 30]])
        >>> trg = np.array([[1, 15, 5], [9, 18, 21], [80, 99, 120], [5, 10, 80]])
        >>> tr = transforms.MovingLeastSquaresTransform(src, trg)
        >>> points = np.array([[0, 0, 0], [50, 50, 50]])
        >>> tr.xform(points)
        array([[ 1.        , 15.        ,  5.        ],
               [17.56361725, 43.32071504, 59.3147564 ]])

        """
        assert direction in ('forward', 'inverse')
        self.transformer = Transformer(landmarks_source, landmarks_target)
        self.reverse = direction == 'inverse'

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

        return self.transformer.transform(points, reverse=self.reverse)

    def __neg__(self) -> 'MovingLeastSquaresTransform':
        """Invert direction"""
        out = self.copy()
        out.reverse = not self.reverse
        return out

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, MovingLeastSquaresTransform):
            return False
        for cp_this, cp_that in zip(
            self._control_points(), o._control_points()
        ):
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
