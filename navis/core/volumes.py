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
import os
import uuid

import numpy as np
import scipy.spatial

from typing import Union, Optional, Sequence, List, Dict, Any

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
    id :        int, optional
                If not provided, neuron will be assigned a random UUID
                as ``.id``

    Attributes
    ----------
    bbox :      array
                Bounding box of the volume.

    See Also
    --------
    :func:`~navis.example_volume`
        Loads example volume(s).

    """

    def __init__(self,
                 vertices: Union[list, np.ndarray],
                 faces: Union[list, np.ndarray],
                 name: Optional[str] = None,
                 color: Union[str,
                              Sequence[Union[int, float]]] = (.95, .95, .95, .1),
                 id: Optional[int] = None, **kwargs):
        self.name: Optional[str] = name
        self.vertices: np.ndarray = np.array(vertices)
        self.faces: np.ndarray = np.array(faces)
        self.color: Union[str, Sequence[Union[int, float]]] = color
        self.id: Optional[int] = id if id else uuid.uuid4()

    @classmethod
    def from_csv(self,
                 vertices: str,
                 faces: str,
                 name: Optional[str] = None,
                 color: Union[str,
                              Sequence[Union[int, float]]] = (1, 1, 1, .1),
                 volume_id: Optional[int] = None, **kwargs) -> 'Volume':
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

        if not os.path.isfile(vertices) or not os.path.isfile(faces):
            raise ValueError('File(s) not found.')

        with open(vertices, 'r') as f:
            reader = csv.reader(f, **kwargs)
            vertices = np.array([r for r in reader]).astype(float)

        with open(faces, 'r') as f:
            reader = csv.reader(f, **kwargs)
            faces = np.array([r for r in reader]).astype(int)

        return Volume(faces=faces, vertices=vertices, name=name, color=color,
                      volume_id=volume_id)

    def to_csv(self, filename: str, **kwargs) -> None:
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
    def from_json(self,
                  filename: str,
                  name: Optional[str] = None,
                  color: Union[str,
                               Sequence[Union[int, float]]] = (1, 1, 1, .1),
                  import_kwargs: Dict = {},
                  **init_kwargs) -> 'Volume':
        """ Load volume from json file containing vertices and faces.

        Parameters
        ----------
        filename
        import_kwargs
                        Keyword arguments passed to ``json.load``.
        **init_kwargs
                    Keyword arguments passed to navis.Volume upon
                    initialization.

        Returns
        -------
        navis.Volume

        """

        if not os.path.isfile(filename):
            raise ValueError('File not found.')

        with open(filename, 'r') as f:
            data = json.load(f, **import_kwargs)

        return Volume(faces=data['faces'],
                      vertices=data['vertices'],
                      name=name, color=color, **init_kwargs)

    @classmethod
    def from_object(self,
                    obj: Any,
                    name: Optional[str] = None,
                    color: Union[str,
                                 Sequence[Union[int, float]]] = (1, 1, 1, .1),
                    **init_kwargs) -> 'Volume':
        """ Load volume from generic object that has ``.vertices`` and
        ``.faces`` attributes.

        Parameters
        ----------
        obj
        **init_kwargs
                    Keyword arguments passed to navis.Volume upon
                    initialization.

        Returns
        -------
        navis.Volume

        """

        if not hasattr(obj, 'vertices') or not hasattr(obj, 'faces'):
            raise ValueError('Object must have faces and vertices attributes.')

        return Volume(faces=obj.faces, vertices=obj.vertices,
                      name=name, color=color, **init_kwargs)

    @classmethod
    def from_file(self,
                  filename: str,
                  name: Optional[str] = None,
                  color: Union[str,
                               Sequence[Union[int, float]]] = (1, 1, 1, .1),
                  import_kwargs: Dict = {},
                  **init_kwargs) -> 'Volume':
        """ Load volume from file containing vertices and faces.

        For OBJ and STL files this function requires the optional trimesh
        library.

        Parameters
        ----------
        filename :      str
                        File to load from.
        import_kwargs
                        Keyword arguments passed to importer:
                          - ``json.load`` for JSON file
                          - ``trimesh.load_mesh`` for OBJ and STL files
        **init_kwargs
                    Keyword arguments passed to navis.Volume upon
                    initialization.

        Returns
        -------
        navis.Volume

        """
        if not os.path.isfile(filename):
            raise ValueError('File not found.')

        f, ext = os.path.splitext(filename)

        if ext == '.json':
            return self.from_json(filename=filename, name=name, color=color,
                                  import_kwargs=import_kwargs, **init_kwargs)

        try:
            import trimesh
        except ImportError:
            raise ImportError('Unable to import: trimesh missing - please '
                              'install: "pip install trimesh"')
        except BaseException:
            raise

        tm = trimesh.load_mesh(filename, **import_kwargs)

        return self.from_object(tm, name=name, color=color, **init_kwargs)

    def export(self,
               filename: str,
               **export_kwargs) -> None:
        """ Export volume using trimesh.

        See ``trimesh.export`` for available formats.


        Parameters
        ----------
        filename :      str
                        File to save to. Format will be extracted from
                        file extension.
        export_kwargs
                        Keyword arguments passed to ``trimesh.export``.
        """

        try:
            import trimesh
        except ImportError:
            raise ImportError('Unable to import: trimesh missing - please '
                              'install: "pip install trimesh"')
        except BaseException:
            raise

        tm = trimesh.Trimesh(self.vertices, self.faces)

        tm.export(filename, **export_kwargs)

        return None

    def to_json(self, filename: str) -> None:
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
    def combine(self,
                x: Sequence['Volume'],
                name: str = 'comb_vol',
                color: Union[str,
                             Sequence[Union[int, float]]] = (1, 1, 1, .1)
                ) -> 'Volume':
        """ Merges multiple volumes into a single object.

        Parameters
        ----------
        x :     list or dict of Volumes
        name :  str, optional
                Name of the combined volume.
        color : tuple | str, optional
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
            x = [x]  # type: ignore

        if False in [isinstance(v, Volume) for v in x]:
            raise TypeError('Input must be list of volumes')

        vertices: np.ndarray = np.empty((0, 3))
        faces: List[List[int]] = []

        # Reindex faces
        for vol in x:
            offs = len(vertices)
            vertices = np.append(vertices, vol.vertices, axis=0)
            faces += [[f[0] + offs, f[1] + offs, f[2] + offs]
                      for f in vol.faces]

        return Volume(vertices=vertices, faces=faces, name=name, color=color)

    @property
    def bbox(self) -> np.ndarray:
        """ Bounding box of this volume. """
        return np.array([self.vertices.min(axis=0),
                         self.vertices.max(axis=0)]).T

    @property
    def vertices(self) -> np.ndarray:
        return self.__vertices

    @vertices.setter
    def vertices(self, v):
        if not isinstance(v, np.ndarray):
            v = np.array(v)

        if not v.shape[1] == 3:
            raise ValueError('Vertices must be of shape N,3.')

        self.__vertices = v

    @property
    def verts(self) -> np.ndarray:
        """Legacy access to ``.vertices``."""
        return self.vertices

    @verts.setter
    def verts(self, v):
        self.vertices = v

    @property
    def faces(self) -> np.ndarray:
        """Legacy access to ``.vertices``."""
        return self.__faces

    @faces.setter
    def faces(self, v):
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        self.__faces = v

    @property
    def center(self) -> np.ndarray:
        """ Center of mass."""
        return np.mean(self.vertices, axis=0)

    def __deepcopy__(self):
        return self.copy()

    def __copy__(self):
        return self.copy()

    def copy(self) -> 'Volume':
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

    def resize(self,
               x: Union[float, int],
               inplace: bool = False
               ) -> Optional['Volume']:
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
        return None

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

    def to_trimesh(self) -> 'trimesh.Trimesh':
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

        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)  # type ignore

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

    def to_2d(self,
              alpha: float = 0.00017,
              view: str = 'xy',
              invert_y: bool = False) -> Sequence[Union[float, int]]:
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
            from shapely.ops import cascaded_union, polygonize  # type: ignore
            import shapely.geometry as geometry  # type: ignore
        except ImportError:
            raise ImportError('This function needs the shapely package.')

        coords: np.ndarray

        if view in['xy', 'yx']:
            coords = self.vertices[:, [0, 1]]  # type: ignore
            if invert_y:
                coords[:, 1] = coords[:, 1] * -1  # type: ignore

        elif view in ['xz', 'zx']:
            coords = self.vertices[:, [0, 2]]  # type: ignore
        elif view in ['yz', 'zy']:
            coords = self.vertices[:, [1, 2]]  # type: ignore
            if invert_y:
                coords[:, 0] = coords[:, 0] * -1  # type: ignore
        else:
            raise ValueError(f'View {view} unknown. Please use either: {accepted_views}')

        tri = scipy.spatial.Delaunay(coords)
        edges: set = set()
        edge_points: list = []
        # loop over triangles:
        # ia, ib, ic = indices of corner points of the
        # triangle
        for ia, ib, ic in tri.vertices:
            pa: np.ndarray = coords[ia]  # type: ignore
            pb: np.ndarray = coords[ib]  # type: ignore
            pc: np.ndarray = coords[ic]  # type: ignore
            # Lengths of sides of triangle
            a = math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)  # type: ignore
            b = math.sqrt((pb[0] - pc[0])**2 + (pb[1] - pc[1])**2)  # type: ignore
            c = math.sqrt((pc[0] - pa[0])**2 + (pc[1] - pa[1])**2)  # type: ignore
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
