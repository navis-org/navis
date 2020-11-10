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
from .. import config, core

import collections
import math
import random
import warnings

import numpy as np

from typing import Tuple, Optional, List, Dict

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from vispy.util.transforms import rotate

__all__ = ['tn_pairs_to_coords', 'segments_to_coords', 'fibonacci_sphere', 'make_tube']

logger = config.logger


def tn_pairs_to_coords(x: core.TreeNeuron,
                       modifier: Optional[Tuple[float,
                                                float,
                                                float]] = (1, 1, 1)
                       ) -> np.ndarray:
    """Return pairs of child->parent node coordinates.

    Parameters
    ----------
    x :         TreeNeuron
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
    parent_co = x.nodes.set_index('node_id').loc[nodes.parent_id.values,
                                                 ['x', 'y', 'z']].values

    coords = np.append(tn_co, parent_co, axis=1)

    if any(modifier != 1):
        coords *= modifier

    return coords.reshape((coords.shape[0], 2, 3))


def segments_to_coords(x: core.TreeNeuron,
                       segments: List[List[int]],
                       modifier: Optional[Tuple[float,
                                                float,
                                                float]] = (1, 1, 1),
                       node_colors: Optional[np.ndarray] = None,
                       ) -> List[np.ndarray]:
    """Turn lists of node IDs into coordinates.

    Parameters
    ----------
    x :             TreeNeuron
                    Must contain the nodes
    segments :      list of lists node IDs
    node_colors :   numpy.ndarray, optional
                    A color for each node in ``x.nodes``. If provided, will
                    also return a list of colors sorted to match coordinates.
    modifier :      ints, optional
                    Use to modify/invert x/y/z axes.

    Returns
    -------
    coords :        list of tuples
                    [(x, y, z), (x, y, z), ... ]
    colors :        list of colors
                    If ``node_colors`` provided will return a copy of it sorted
                    to match ``coords``.

    """
    if not isinstance(modifier, np.ndarray):
        modifier = np.array(modifier)

    # Using a dictionary here is orders of manitude faster than .loc[]!
    locs: Dict[int, Tuple[float, float, float]]
    # Oddly, this is also the fastest way to generate the dictionary
    locs = {r.node_id: (r.x, r.y, r.z) for r in x.nodes.itertuples()}  # type: ignore
    coords = [np.array([locs[tn] for tn in s]) for s in segments]

    if any(modifier != 1):
        coords = [c * modifier for c in coords]

    if not isinstance(node_colors, type(None)):
        ilocs = dict(zip(x.nodes.node_id.values,
                         np.arange(x.nodes.shape[0])))
        colors = [node_colors[[ilocs[tn] for tn in s]] for s in segments]

        return coords, colors

    return coords


def fibonacci_sphere(samples: int = 1,
                     randomize: bool = True) -> list:
    """Generate points on a sphere."""
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


def make_tube(segments, radii=1.0, tube_points=8, use_normals=True):
    """Generate tube mesh (vertices + faces) from lines.

    This code was modified from the vispy library.

    Parameters
    ----------
    segments :      list
                    List of lists of x/y/z coordinates.
    radii :         float | list of floats
                    Either a single radius used for all nodes or list of lists of
                    floats with the same shape as ``segments``.
    tube_points :   int
                    Number of points making up the circle of the cross-section
                    of the tube.
    use_normals :   bool
                    If True will rotate tube along it's curvature.

    Returns
    -------
    vertices :      np.ndarray
    faces :         np.ndarray

    """
    vertices = np.empty((0, 3), dtype=np.float)
    indices = np.empty((0, 3), dtype=np.uint32)

    if not isinstance(radii, collections.Iterable):
        radii = [[radii] * len(points) for points in segments]

    for points, radius in zip(segments, radii):
        # Need to make sure points are floats
        points = np.array(points).astype(float)

        if use_normals:
            tangents, normals, binormals = _frenet_frames(points)
        else:
            tangents = normals = binormals = np.ones((len(points), 3))

        n_segments = len(points) - 1

        if not isinstance(radius, collections.Iterable):
            radius = [radius] * len(points)

        radius = np.array(radius)

        # Vertices for each point on the circle
        verts = np.repeat(points, tube_points, axis=0)

        v = np.arange(tube_points,
                      dtype=np.float) / tube_points * 2 * np.pi

        all_cx = (radius * -1. * np.tile(np.cos(v), points.shape[0]).reshape((tube_points, points.shape[0]), order='F')).T
        cx_norm = (all_cx[:, :, np.newaxis] * normals[:, np.newaxis, :]).reshape(verts.shape)

        all_cy = (radius * np.tile(np.sin(v), points.shape[0]).reshape((tube_points, points.shape[0]), order='F')).T
        cy_norm = (all_cy[:, :, np.newaxis] * binormals[:, np.newaxis, :]).reshape(verts.shape)

        verts = verts + cx_norm + cy_norm

        # Generate indices for the first segment
        ix = np.arange(0, tube_points)

        # Repeat indices n_segments-times
        ix = np.tile(ix, n_segments)

        # Offset indices by number segments and tube points
        offsets = np.repeat((np.arange(0, n_segments)) * tube_points, tube_points)
        ix += offsets

        # Turn indices into faces
        ix_a = ix
        ix_b = ix + tube_points

        ix_c = ix_b.reshape((n_segments, tube_points))
        ix_c = np.append(ix_c[:, 1:], ix_c[:, [0]], axis=1)
        ix_c = ix_c.ravel()

        ix_d = ix_a.reshape((n_segments, tube_points))
        ix_d = np.append(ix_d[:, 1:], ix_d[:, [0]], axis=1)
        ix_d = ix_d.ravel()

        faces1 = np.concatenate((ix_a, ix_b, ix_d), axis=0).reshape((n_segments * tube_points, 3), order='F')
        faces2 = np.concatenate((ix_b, ix_c, ix_d), axis=0).reshape((n_segments * tube_points, 3), order='F')

        faces = np.append(faces1, faces2, axis=0)

        # Offset faces against already existing vertices
        faces += vertices.shape[0]

        # Add vertices and faces to total collection
        vertices = np.append(vertices, verts, axis=0)
        indices = np.append(indices, faces, axis=0)

    return vertices, indices


def _frenet_frames(points):
    """Calculate and return the tangents, normals and binormals for the tube.

    This code was modified from the vispy library.

    """
    tangents = np.zeros((len(points), 3))
    normals = np.zeros((len(points), 3))

    epsilon = 0.0001

    # Compute tangent vectors for each segment
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)

    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]

    mags = np.sqrt(np.sum(tangents * tangents, axis=1))
    tangents /= mags[:, np.newaxis]

    # Get initial normal and binormal
    t = np.abs(tangents[0])

    smallest = np.argmin(t)
    normal = np.zeros(3)
    normal[smallest] = 1.

    vec = np.cross(tangents[0], normal)
    normals[0] = np.cross(tangents[0], vec)

    all_vec = np.cross(tangents[:-1], tangents[1:])
    all_vec_norm = np.linalg.norm(all_vec, axis=1)

    # Normalise vectors if necessary
    where = all_vec_norm > epsilon
    all_vec[where, :] /= all_vec_norm[where].reshape((sum(where), 1))

    # Precompute inner dot product
    dp = np.sum(tangents[:-1] * tangents[1:], axis=1)
    # Clip
    cl = np.clip(dp, -1, 1)
    # Get theta
    th = np.arccos(cl)

    # Compute normal and binormal vectors along the path
    for i in range(1, len(points)):
        normals[i] = normals[i-1]

        vec_norm = all_vec_norm[i-1]
        vec = all_vec[i-1]
        if vec_norm > epsilon:
            normals[i] = rotate(-np.degrees(th[i-1]),
                                vec)[:3, :3].dot(normals[i])

    binormals = np.cross(tangents, normals)

    return tangents, normals, binormals
