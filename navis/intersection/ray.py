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


""" This module contains functions for intersections.
"""

import numpy as np

from typing import Sequence

from ..core import Volume

try:
    from pyoctree import pyoctree
except ImportError:
    pyoctree = None


def ray_in_volume(points: np.ndarray,
                  volume: Volume,
                  multi_ray: bool = False) -> Sequence[bool]:
    """ Uses pyoctree's raycsasting to test if points are within a given
    volume.
    """

    tree = getattr(volume, 'pyoctree', None)

    if not tree:
        # Create octree from scratch
        tree = pyoctree.PyOctree(np.array(volume.vertices, dtype=float, order='C'),
                                 np.array(volume.faces, dtype=np.int32, order='C')
                                 )
        volume.pyoctree = tree  # type: ignore

    # Get min max of volume
    mx = np.array(volume.vertices).max(axis=0)
    mn = np.array(volume.vertices).min(axis=0)

    # Get points outside of bounding box
    bbox_out = (points > mx).any(axis=1) | (points < mn).any(axis=1)
    isin = ~bbox_out
    in_points = points[isin]

    # Perform ray intersection on points inside bounding box
    rayPointList = np.array([[[p[0], mn[1], mn[2]], p] for p in in_points],
                            dtype=np.float32)

    # Get intersections and extract coordinates of intersection
    intersections = [np.array([i.p for i in tree.rayIntersection(ray)]) for ray in rayPointList]

    # In a few odd cases we can get the multiple intersections at the exact
    # same coordinate (something funny with the faces).
    unique_int = [np.unique(np.round(i), axis=0) if np.any(i) else i for i in intersections]

    # Unfortunately rays are bidirectional -> we have to filter intersections
    # to those that occur "above" the point we are querying
    unilat_int = [i[i[:, 2] >= p] if np.any(i) else i for i, p in zip(unique_int, in_points[:, 2])]

    # Count intersections
    int_count = [i.shape[0] for i in unilat_int]

    # Get odd (= in volume) numbers of intersections
    is_odd = np.remainder(int_count, 2) != 0

    # If we want to play it safe, run the above again with two additional rays
    # and decide by majority
    if multi_ray:
        # Run ray from left back
        rayPointList = np.array([[[mn[0], p[1], mn[2]], p] for p in in_points],
                                dtype=np.float32)
        intersections = [np.array([i.p for i in tree.rayIntersection(ray)]) for ray in rayPointList]
        unique_int = [np.unique(i, axis=0) if np.any(i) else i for i in intersections]
        unilat_int = [i[i[:, 0] >= p] if np.any(i) else i for i, p in zip(unique_int, in_points[:, 0])]
        int_count = [i.shape[0] for i in unilat_int]
        is_odd2 = np.remainder(int_count, 2) != 0

        # Run ray from lower left
        rayPointList = np.array([[[mn[0], mn[1], p[2]], p] for p in in_points],
                                dtype=np.float32)
        intersections = [np.array([i.p for i in tree.rayIntersection(ray)]) for ray in rayPointList]
        unique_int = [np.unique(i, axis=0) if np.any(i) else i for i in intersections]
        unilat_int = [i[i[:, 1] >= p] if np.any(i) else i for i, p in zip(unique_int, in_points[:, 1])]
        int_count = [i.shape[0] for i in unilat_int]
        is_odd3 = np.remainder(int_count, 2) != 0

        # Find majority consensus
        is_odd = is_odd.astype(int) + is_odd2.astype(int) + is_odd3.astype(int)
        is_odd = is_odd >= 2

    isin[isin] = is_odd
    return isin
