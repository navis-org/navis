#    This script is part of pymaid (http://www.github.com/schlegelp/navis).
#    but has been adapted from vispy (http://www.vispy.org)

import collections
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from vispy.visuals import MeshVisual
    from vispy.color import ColorArray

    from vispy.scene.visuals import create_visual_node
    from vispy.util.transforms import rotate

import numpy as np
from numpy.linalg import norm


class NeuronVisual(MeshVisual):
    """Displays a tube around a piecewise-linear path.
    The tube mesh is corrected following its Frenet curvature and
    torsion such that it varies smoothly along the curve, including if
    the tube is closed.

    Parameters
    ----------
    points : list of arrays
        An list of arrays of (x, y, z) points describing the path along which
        a tube will be extruded.
    radius : list of floats
        The radius of the tube. Defaults to 1.0.
    closed : bool
        Whether the tube should be closed, joining the last point to the
        first. Defaults to False.
    color : Color | ColorArray
        The color(s) to use when drawing the tube. The same color is
        applied to each vertex of the mesh surrounding each point of
        the line. If the input is a ColorArray, the argument will be
        cycled; for instance if 'red' is passed then the entire tube
        will be red, or if ['green', 'blue'] is passed then the points
        will alternate between these colours. Defaults to 'purple'.
    tube_points : int
        The number of points in the circle-approximating polygon of the
        tube's cross section. Defaults to 8.
    shading : str | None
        Same as for the `MeshVisual` class. Defaults to 'smooth'.
    vertex_colors: ndarray | None
        Same as for the `MeshVisual` class.
    face_colors: ndarray | None
        Same as for the `MeshVisual` class.
    mode : str
        Same as for the `MeshVisual` class. Defaults to 'triangles'.
    """

    def __init__(self, segments, radii=1.0,
                 color='purple',
                 tube_points=8,
                 shading='smooth',
                 vertex_colors=None,
                 face_colors=None,
                 use_normals=True,
                 mode='triangles'):

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

            """
            # get the positions of each vertex
            grid = np.zeros((len(points), tube_points, 3))

            for i in range(len(points)):
                pos = points[i]
                if use_normals:
                    normal = normals[i]
                    binormal = binormals[i]
                else:
                    normal = binormal = 1

                r = radius[i]

                # Add a vertex for each point on the circle
                v = np.arange(tube_points,
                              dtype=np.float) / tube_points * 2 * np.pi
                cx = -1. * r * np.cos(v)
                cy = r * np.sin(v)
                grid[i] = (pos + cx[:, np.newaxis] * normal +
                           cy[:, np.newaxis] * binormal)
            """

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

            # Get vertices from grid
            #this_vertices = grid.reshape(grid.shape[0] * grid.shape[1], 3)
            this_vertices = verts

            # Add vertices and faces to total collection
            vertices = np.append(vertices, this_vertices, axis=0)
            indices = np.append(indices, faces, axis=0)

        color = ColorArray(color)
        if vertex_colors is None:
            vertex_colors = np.resize(color.rgba,
                                     (vertices.shape[0], 4))

        MeshVisual.__init__(self, vertices, indices,
                            vertex_colors=vertex_colors,
                            face_colors=face_colors,
                            shading=shading,
                            mode=mode)


class TubeVisual(MeshVisual):
    """Displays a tube around a piecewise-linear path.
    The tube mesh is corrected following its Frenet curvature and
    torsion such that it varies smoothly along the curve, including if
    the tube is closed.

    Parameters
    ----------
    points : ndarray
        An array of (x, y, z) points describing the path along which the
        tube will be extruded.
    radius : float
        The radius of the tube. Defaults to 1.0.
    closed : bool
        Whether the tube should be closed, joining the last point to the
        first. Defaults to False.
    color : Color | ColorArray
        The color(s) to use when drawing the tube. The same color is
        applied to each vertex of the mesh surrounding each point of
        the line. If the input is a ColorArray, the argument will be
        cycled; for instance if 'red' is passed then the entire tube
        will be red, or if ['green', 'blue'] is passed then the points
        will alternate between these colours. Defaults to 'purple'.
    tube_points : int
        The number of points in the circle-approximating polygon of the
        tube's cross section. Defaults to 8.
    shading : str | None
        Same as for the `MeshVisual` class. Defaults to 'smooth'.
    vertex_colors: ndarray | None
        Same as for the `MeshVisual` class.
    face_colors: ndarray | None
        Same as for the `MeshVisual` class.
    mode : str
        Same as for the `MeshVisual` class. Defaults to 'triangles'.
    """

    def __init__(self, points, radius=1.0,
                 closed=False,
                 color='purple',
                 tube_points=8,
                 shading='smooth',
                 vertex_colors=None,
                 face_colors=None,
                 mode='triangles'):

        points = np.array(points)

        tangents, normals, binormals = _frenet_frames(points, closed)

        segments = len(points) - 1

        if not isinstance(radius, collections.Iterable):
            radius = [radius] * len(points)

        # get the positions of each vertex
        grid = np.zeros((len(points), tube_points, 3))
        for i in range(len(points)):
            pos = points[i]
            normal = normals[i]
            binormal = binormals[i]
            r = radius[i]

            # Add a vertex for each point on the circle
            v = np.arange(tube_points,
                          dtype=np.float) / tube_points * 2 * np.pi
            cx = -1. * r * np.cos(v)
            cy = r * np.sin(v)
            grid[i] = (pos + cx[:, np.newaxis] * normal +
                       cy[:, np.newaxis] * binormal)

        # construct the mesh
        indices = []
        for i in range(segments):
            for j in range(tube_points):
                ip = (i + 1) % segments if closed else i + 1
                jp = (j + 1) % tube_points

                index_a = i * tube_points + j
                index_b = ip * tube_points + j
                index_c = ip * tube_points + jp
                index_d = i * tube_points + jp

                indices.append([index_a, index_b, index_d])
                indices.append([index_b, index_c, index_d])

        vertices = grid.reshape(grid.shape[0] * grid.shape[1], 3)

        color = ColorArray(color)
        if vertex_colors is None:
            point_colors = np.resize(color.rgba,
                                     (len(points), 4))
            vertex_colors = np.repeat(point_colors, tube_points, axis=0)

        indices = np.array(indices, dtype=np.uint32)

        MeshVisual.__init__(self, vertices, indices,
                            vertex_colors=vertex_colors,
                            face_colors=face_colors,
                            shading=shading,
                            mode=mode)


def _frenet_frames(points):
    '''Calculates and returns the tangents, normals and binormals for
    the tube.'''
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
    all_vec_norm = norm(all_vec, axis=1)

    # Normalise vectors if necessary
    where = all_vec_norm > epsilon
    all_vec[where, :] /=  all_vec_norm[where].reshape((sum(where), 1))

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


Tube = create_visual_node(TubeVisual)
MeshNeuron = create_visual_node(NeuronVisual)
