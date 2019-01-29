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

try:
    from pyoctree import pyoctree
except ImportError:
    pyoctree = None


def ray_in_volume(points, volume):
    """ Uses pyoctree's raycsasting to test if points are within a given
    volume.
    """

    tree = getattr(volume, 'pyoctree', None)

    if not tree:
        # Create octree from scratch
        tree = pyoctree.PyOctree(np.array(volume.vertices, dtype='d', order='C'),
                                 np.array(volume.faces, dtype=np.int32, order='C')
                                 )
        volume.pyoctree = tree

    # Get min max of volume
    mx = np.array(volume.vertices).max(axis=0)
    mn = np.array(volume.vertices).min(axis=0)

    # Get points outside of bounding box
    out = (points > mx).any(axis=1) | (points < mn).any(axis=1)
    isin = ~out
    in_points = points[~out]

    # Perform ray intersection on points inside bounding box
    rayPointList = np.array([[[p[0], p[1], mn[2]], [p[0], p[1], mx[2]]] for p in in_points],
                            dtype=np.float32)

    # Unfortunately rays are bidirectional -> we have to filter intersections
    # to those that occur "above" the point we are querying
    intersections = [len([i for i in tree.rayIntersection(
        ray) if i.p[2] >= in_points[k][2]]) for k, ray in enumerate(rayPointList)]

    # Count intersections and return True for odd counts
    # [i % 2 != 0 for i in intersections]
    isin[~out] = np.remainder(list(intersections), 2) != 0

    return isin
