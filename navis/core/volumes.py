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

import csv
import json
import math
import numbers

import numpy as np
import scipy.spatial

from .. import utils, config

# Set up logging
logger = config.logger


class Volume:
    """ Class representing meshes.

    Parameters
    ----------
    vertices :  list | array
                Vertices coordinates. Must be shape (N,3).
    faces :     list | array
                Indexed faceset.
    name :      str, optional
                Name of volume.
    color :     tuple, optional
                RGB color.
    volume_id : int, optional

    Attributes
    ----------
    bbox :      array
                Bounding box of the volume.


    See Also
    --------
    :func:`~navis.example_volume`
        Loads example volume(s).

    """

    def __init__(self, vertices, faces, name=None, color=(1, 1, 1, .1),
                 volume_id=None, **kwargs):
        self.name = name
        self.vertices = vertices
        self.faces = faces
        self.color = color
        self.volume_id = volume_id

    @classmethod
    def from_csv(self, vertices, faces, name=None, color=(1, 1, 1, .1),
                 volume_id=None, **kwargs):
        """ Load volume from csv files containing vertices and faces.

        Parameters
        ----------
        vertices :      filepath | file-like
                        CSV file containing vertices.
        faces :         filepath | file-like
                        CSV file containing faces.
        **kwargs
                        Keyword arguments passed to ``csv.reader``.

        Returns
        -------
        navis.Volume

        """

        with open(vertices, 'r') as f:
            reader = csv.reader(f, **kwargs)
            vertices = np.array([r for r in reader]).astype(float)

        with open(faces, 'r') as f:
            reader = csv.reader(f, **kwargs)
            faces = np.array([r for r in reader]).astype(int)

        return Volume(faces=faces, vertices=vertices, name=name, color=color,
                      volume_id=volume_id)

    def to_csv(self, filename, **kwargs):
        """ Save volume as two separated csv files containing vertices and
        faces.

        Parameters
        ----------
        filename :      str
                        Filename to use. Will get a ``_vertices.csv`` and
                        ``_faces.csv`` suffix.
        **kwargs
                        Keyword arguments passed to ``csv.reader``.
        """

        for data, suffix in zip([self.faces, self.vertices],
                                ['_faces.csv', '_vertices.csv']):
            with open(filename + suffix, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)

    @classmethod
    def from_json(self, filename, name=None, color=(1, 1, 1, .1), **kwargs):
        """ Load volume from json files containing vertices and faces.

        Parameters
        ----------
        filename
        **kwargs
                        Keyword arguments passed to ``json.load``.

        Returns
        -------
        navis.Volume

        """

        with open(filename, 'r') as f:
            data = json.load(f, **kwargs)

        return Volume(faces=data['faces'],
                      vertices=data['vertices'],
                      name=name, color=color)

    def to_json(self, filename):
        """ Save volume as json file.

        Parameters
        ----------
        filename :      str
                        Filename to use.
        """

        with open(filename, 'w') as f:
            json.dump({'vertices': self.vertices.tolist(),
                       'faces': self.faces.tolist()},
                      f)

    @classmethod
    def combine(self, x, name='comb_vol', color=(1, 1, 1, .1)):
        """ Merges multiple volumes into a single object.

        Parameters
        ----------
        x :     list or dict of Volumes
        name :  str, optional
                Name of the combined volume.
        color : tuple, optional
                Color of the combined volume.

        Returns
        -------
        :class:`~navis.Volume`
        """

        if isinstance(x, Volume):
            return x

        if isinstance(x, dict):
            x = list(x.values())

        if not utils.is_iterable(x):
            x = [x]

        if False in [isinstance(v, Volume) for v in x]:
            raise TypeError('Input must be list of volumes')

        vertices = np.empty((0, 3))
        faces = []

        # Reindex faces
        for vol in x:
            offs = len(vertices)
            vertices = np.append(vertices, vol.vertices, axis=0)
            faces += [[f[0] + offs, f[1] + offs, f[2] + offs]
                      for f in vol.faces]

        return Volume(vertices=vertices, faces=faces, name=name, color=color)

    @property
    def bbox(self):
        """ Bounding box of this volume. """
        return np.array([self.vertices.min(axis=0),
                         self.vertices.max(axis=0)]).T

    @property
    def vertices(self):
        return self.__vertices

    @vertices.setter
    def vertices(self, v):
        if not isinstance(v, np.ndarray):
            v = np.array(v)

        if not v.shape[1] == 3:
            raise ValueError('Vertices must be of shape N,3.')

        self.__vertices = v

    @property
    def verts(self):
        """Legacy access to ``.vertices``."""
        return self.vertices

    @verts.setter
    def verts(self, v):
        self.vertices = v

    @property
    def faces(self):
        """Legacy access to ``.vertices``."""
        return self.__faces

    @faces.setter
    def faces(self, v):
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        self.__faces = v

    @property
    def center(self):
        """ Center of mass."""
        return np.mean(self.vertices, axis=0)

    def __deepcopy__(self):
        return self.copy()

    def __copy__(self):
        return self.copy()

    def copy(self):
        """Return copy of this volume. Does not maintain generic values."""
        return Volume(self.vertices, self.faces, self.name,
                      self.color, self.volume_id)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{type(self)} "{self.name}" at {hex(id(self))}: ' \
               f'{self.vertices.shape[0]} vertices, {self.faces.shape[0]} faces'

    def __truediv__(self, other):
        """Implements division for vertex coordinates."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            return self.__mul__(1 / other)
        else:
            return NotImplemented

    def __mul__(self, other):
        """Implements multiplication for vertex coordinates."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            v = self.copy()
            v.vertices *= other
            return v
        else:
            return NotImplemented

    def resize(self, x, inplace=False):
        """ Resize volume by given factor.

        Resize is from center of mass, not origin.

        Parameters
        ----------
        x :         int
                    Resizing factor
        inplace :   bool, optional
                    If False, will return resized copy.

        Returns
        -------
        :class:`navis.Volume`
                    Resized copy of original volume. Only if ``inplace=True``.
        None
                    If ``inplace=False``.
        """
        if not inplace:
            v = self.copy()
        else:
            v = self

        # Get the center
        cn = np.mean(v.vertices, axis=0)

        # Get vector from center to each vertex
        vec = v.vertices - cn

        # Multiply vector by resize factor
        vec *= x

        # Recalculate vertex positions
        v.vertices = vec + cn

        # Make sure to reset any pyoctree data on this volume
        try:
            delattr(v, 'pyoctree')
        except BaseException:
            pass

        if not inplace:
            return v

    def plot3d(self, **kwargs):
        """Plot volume using :func:`navis.plot3d`.

        Parameters
        ----------
        **kwargs
                Keyword arguments. Will be passed to :func:`navis.plot3d`.
                See ``help(navis.plot3d)`` for a list of keywords.

        See Also
        --------
        :func:`navis.plot3d`
                    Function called to generate 3d plot.

        Examples
        --------
        >>> vol = navis.get_volume('v14.LH_R')
        >>> vol.plot3d(color = (255, 0, 0))
        """

        from .. import plotting

        if 'color' in kwargs:
            self.color = kwargs['color']

        return plotting.plot3d(self, **kwargs)

    def to_trimesh(self):
        """ Returns trimesh representation of this volume.

        See Also
        --------
        https://github.com/mikedh/trimesh
                trimesh GitHub page.
        """

        try:
            import trimesh
        except ImportError:
            raise ImportError('Unable to import trimesh. Please make sure it '
                              'is installed properly')

        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    def _outlines_3d(self, view='xy', **kwargs):
        """ Generate 3d outlines along a given view (see ``.to_2d()``).

        Parameters
        ----------
        **kwargs
                    Keyword arguments passed to :func:`~navis.Volume.to_2d`.

        Returns
        -------
        list
                    Coordinates of 2d circumference.
                    e.g. ``[(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), ...]``
                    Third dimension is averaged
        """

        co2d = np.array(self.to_2d(view=view, **kwargs))

        if view in ['xy', 'yx']:
            third = np.repeat(self.center[2], co2d.shape[0])
        elif view in ['xz', 'zx']:
            third = np.repeat(self.center[1], co2d.shape[0])
        elif view in ['yz', 'zy']:
            third = np.repeat(self.center[0], co2d.shape[0])

        return np.append(co2d, third.reshape(co2d.shape[0], 1), axis=1)

    def to_2d(self, alpha=0.00017, view='xy', invert_y=False):
        """ Computes the 2d alpha shape (concave hull) this volume.

        Uses Scipy Delaunay and shapely.

        Parameters
        ----------
        alpha:      float, optional
                    Alpha value to influence the gooeyness of the border.
                    Smaller numbers don't fall inward as much as larger
                    numbers. Too large, and you lose everything!
        view :      'xy' | 'xz' | 'yz', optional
                    Determines if frontal, lateral or top view.

        Returns
        -------
        list
                    Coordinates of 2d circumference
                    e.g. ``[(x1, y1), (x2, y2), (x3, y3), ...]``

        """

        def add_edge(edges, edge_points, coords, i, j):
            """ Add a line between the i-th and j-th points,
            if not in the list already.
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                return
            edges.add((i, j))
            edge_points.append(coords[[i, j]])

        accepted_views = ['xy', 'xz', 'yz']

        try:
            from shapely.ops import cascaded_union, polygonize
            import shapely.geometry as geometry
        except ImportError:
            raise ImportError('This function needs the shapely package.')

        if view in['xy', 'yx']:
            coords = self.vertices[:, [0, 1]]
            if invert_y:
                coords[:, 1] = coords[:, 1] * -1

        elif view in ['xz', 'zx']:
            coords = self.vertices[:, [0, 2]]
        elif view in ['yz', 'zy']:
            coords = self.vertices[:, [1, 2]]
            if invert_y:
                coords[:, 0] = coords[:, 0] * -1
        else:
            raise ValueError(f'View {view} unknown. Please use either: {accepted_views}')

        tri = scipy.spatial.Delaunay(coords)
        edges = set()
        edge_points = []
        # loop over triangles:
        # ia, ib, ic = indices of corner points of the
        # triangle
        for ia, ib, ic in tri.vertices:
            pa = coords[ia]
            pb = coords[ib]
            pc = coords[ic]
            # Lengths of sides of triangle
            a = math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
            b = math.sqrt((pb[0] - pc[0])**2 + (pb[1] - pc[1])**2)
            c = math.sqrt((pc[0] - pa[0])**2 + (pc[1] - pa[1])**2)
            # Semiperimeter of triangle
            s = (a + b + c) / 2.0
            # Area of triangle by Heron's formula
            area = math.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
            # Here's the radius filter.
            if circum_r < 1.0 / alpha:
                add_edge(edges, edge_points, coords, ia, ib)
                add_edge(edges, edge_points, coords, ib, ic)
                add_edge(edges, edge_points, coords, ic, ia)

        m = geometry.MultiLineString(edge_points)
        triangles = list(polygonize(m))
        concave_hull = cascaded_union(triangles)

        # Try with current settings, if this is not successful, try again
        # with lower alpha
        try:
            return list(concave_hull.exterior.coords)
        except AttributeError:
            return self.to_2d(alpha=alpha / 10, view=view, invert_y=invert_y)
        except BaseException:
            raise
