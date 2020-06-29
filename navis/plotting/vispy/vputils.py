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

import collections
import png
import warnings

import numpy as np

from ... import config

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from vispy.gloo.util import _screenshot
    import vispy.scene.visuals as vpvisuals
    from vispy.util.transforms import rotate


def get_viewer():
    """Grab active 3D viewer.

    Returns
    -------
    :class:`~navis.Viewer`

    Examples
    --------
    >>> from vispy import scene
    >>> # Get and plot neuron in 3d
    >>> n = navis.example_neurons(1)
    >>> _ = n.plot3d(color='red')
    >>> # Grab active viewer and add custom text
    >>> viewer = navis.get_viewer()
    >>> text = scene.visuals.Text(text='TEST',
    ...                           pos=(0, 0, 0))
    >>> viewer.add(text)
    >>> # Close viewer
    >>> viewer.close()

    """
    return getattr(config, 'primary_viewer', None)


def clear3d():
    """Clear viewer 3D canvas."""
    viewer = get_viewer()

    if viewer:
        viewer.clear()


def close3d():
    """Close existing vispy 3D canvas (wipes memory)."""
    try:
        viewer = get_viewer()
        viewer.close()
        globals().pop('viewer')
        del viewer
    except BaseException:
        pass


def screenshot(file='screenshot.png', alpha=True):
    """Save a screenshot of active vispy 3D canvas.

    Parameters
    ----------
    file :      str, optional
                Filename
    alpha :     bool, optional
                If True, alpha channel will be saved

    See Also
    --------
    :func:`navis.Viewer.screenshot`
                Take screenshot of specific canvas.

    """
    if alpha:
        mode = 'RGBA'
    else:
        mode = 'RGB'

    im = png.from_array(_screenshot(alpha=alpha), mode=mode)
    im.save(file)

    return


def make_tube(segments, radii=1.0, tube_points=8, use_normals=True):
    """Generate tube mesh (vertices +faces) from lines.

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
