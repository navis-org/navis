#    This script is part of navis (http://www.github.com/schlegelp/navis).
#    but has been adapted from vispy (http://www.vispy.org)
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

from vispy.visuals import MeshVisual
from vispy.color import ColorArray
import numpy as np
from vispy.visuals.tube import _frenet_frames
from vispy.scene.visuals import create_visual_node

import collections


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


Tube = create_visual_node(TubeVisual)
