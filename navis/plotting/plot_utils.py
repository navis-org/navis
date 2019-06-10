#    This script is part of navis (http://www.github.com/schlegelp/navis).
#    Copyright (C) 2017 Philipp Schlegel
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
#
#    You should have received a copy of the GNU General Public License
#    along

""" Module contains functions to plot neurons in 2D and 3D.
"""

import random
import math

import numpy as np

from typing import Tuple, Optional, List, Dict

from .. import config, core

__all__ = ['tn_pairs_to_coords', 'segments_to_coords', 'fibonacci_sphere']

logger = config.logger


def tn_pairs_to_coords(x: core.TreeNeuron,
                       modifier: Optional[Tuple[float,
                                                float,
                                                float]] = (1, 1, 1)
                       ) -> np.ndarray:
    """Returns pairs of treenode -> parent node coordinates.

    Parameters
    ----------
    x :         pandas DataFrame | TreeNeuron
                Must contain the nodes.
    modifier :  ints, optional
                Use to modify/invert x/y/z axes.

    Returns
    -------
    coords :    np.array
                ``[[[x1, y1, z1], [x2, y2, z2]], [[x3, y3, y4], [x4, y4, z4]]]``

    """

    if not isinstance(modifier, np.ndarray):
        modifier = np.array(modifier)

    nodes = x.nodes[x.nodes.parent_id >= 0]

    tn_co = nodes.loc[:, ['x', 'y', 'z']].values
    parent_co = x.nodes.set_index('node_id',
                                  inplace=False).loc[nodes.parent_id.values,
                                                     ['x', 'y', 'z']].values

    tn_co *= modifier
    parent_co *= modifier

    coords = np.append(tn_co, parent_co, axis=1)

    return coords.reshape((coords.shape[0], 2, 3))


def segments_to_coords(x: core.TreeNeuron,
                       segments: List[List[int]],
                       modifier: Optional[Tuple[float,
                                                float,
                                                float]] = (1, 1, 1)
                       ) -> List[np.ndarray]:
    """ Turns lists of node IDs into coordinates.

    Parameters
    ----------
    x :         pandas DataFrame | TreeNeuron
                Must contain the nodes
    segments :  list of treenode IDs
    modifier :  ints, optional
                Use to modify/invert x/y/z axes.

    Returns
    -------
    coords :    list of tuples
                [(x, y, z), (x, y, z), ... ]

    """

    if not isinstance(modifier, np.ndarray):
        modifier = np.array(modifier)

    locs: Dict[int, Tuple[float, float, float]]
    locs = {r.node_id: (r.x, r.y, r.z) for r in x.nodes.itertuples()}  # type: ignore

    coords = ([np.array([locs[tn] for tn in s]) * modifier for s in segments])

    return coords


def fibonacci_sphere(samples: int = 1,
                     randomize: bool = True) -> list:
    """ Calculates points on a sphere
    """
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2. / samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return points
