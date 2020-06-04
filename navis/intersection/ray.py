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

from typing import Sequence, Optional

from ..core import Volume

try:
    from pyoctree import pyoctree
except ImportError:
    pyoctree = None

try:
    import ncollpyde
except ImportError:
    ncollpyde = None


def in_volume_ncoll(points: np.ndarray,
                    volume: Volume,
                    n_rays: Optional[int] = 3) -> Sequence[bool]:
    """Use ncollpyde to test if points are within a given volume."""
    if isinstance(n_rays, type(None)):
        n_rays = 3

    if not isinstance(n_rays, (int, np.integer)):
        raise TypeError(f'n_rays must be integer, got "{type(n_rays)}"')

    if n_rays <= 0:
        raise ValueError('n_rays must be > 0')

    coll = ncollpyde.Volume(volume.vertices, volume.faces, n_rays=n_rays)

    return coll.contains(points)


def in_volume_pyoc(points: np.ndarray,
                   volume: Volume,
                   n_rays: Optional[int] = 1) -> Sequence[bool]:
    """Use pyoctree's raycasting to test if points are within a given volume."""
    if isinstance(n_rays, type(None)):
        n_rays = 1

    if not isinstance(n_rays, (int, np.integer)):
        raise TypeError(f'n_rays must be integer, got "{type(n_rays)}"')

    if n_rays <= 0:
        raise ValueError('n_rays must be > 0')

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
    dm = mx - mn

    # Remove points outside of bounding box
    is_out = (points > mx).any(axis=1) | (points < mn).any(axis=1)

    # Cast rays
    # There is no point of vectorizing this because pyoctree's rayIntersection
    # does only take a single ray at a time...
    for i in range(n_rays):
        # Process only point that we think could be in
        in_points = points[~is_out]

        # If no in points left, break out
        if in_points.size == 0:
            break

        # Pick a random point inside the volumes bounding box as origin
        origin = np.random.rand(3) * dm + mn

        # Generate ray point list:
        rayPointList = np.array([[origin, p] for p in in_points],
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

        # Get even (= outside volume) numbers of intersections
        is_even = np.remainder(int_count, 2) == 0

        # Set outside points
        is_out[~is_out] = is_even

    return ~is_out
