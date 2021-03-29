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
import functools
import json
import math
import numbers
import os
import uuid

import numpy as np
import scipy.spatial
import trimesh

from typing import Union, Optional, Sequence, List, Dict, Any
from typing_extensions import Literal

from .. import utils, config

# Set up logging
logger = config.logger


class Volume(trimesh.Trimesh):
    """Mesh consisting of vertices and faces.

    Subclass of ``trimesh.Trimesh`` with a few additional methods.

    Parameters
    ----------
    vertices :  list | array
                Vertices coordinates. Must be shape (N,3). Can also be an object
                that has ``.vertices`` and ``.faces`` attributes in which case
                ``faces`` parameter will be ignored.
    faces :     list | array
                Indexed faceset.
    name :      str, optional
                Name of volume.
    color :     tuple, optional
                RGB color.
    id :        int, optional
                If not provided, neuron will be assigned a random UUID
                as ``.id``
    **kwargs
                Keyword arguments passed through to ``trimesh.Trimesh``

    See Also
    --------
    :func:`~navis.example_volume`
        Loads example volume(s).

    """

    def __init__(self,
                 vertices: Union[list, np.ndarray],
                 faces: Union[list, np.ndarray] = None,
                 name: Optional[str] = None,
                 color: Union[str,
                              Sequence[Union[int, float]]] = (.85, .85, .85, .2),
                 id: Optional[int] = None, **kwargs):

        if hasattr(vertices, 'vertices') and hasattr(vertices, 'faces'):
            vertices, faces = vertices.vertices, vertices.faces

        super().__init__(vertices=vertices, faces=faces, **kwargs)

        self.name: Optional[str] = name
        self.color: Union[str, Sequence[Union[int, float]]] = color
        self.id: Optional[int] = id if id else uuid.uuid4()

        # This is very hackish but we want to make sure that parent methods of
        # Trimesh return a navis.Volume instead of a trimesh.Trimesh
        for f in dir(trimesh.Trimesh):
            # Don't mess with magic/private methods
            if f.startswith('_'):
                continue
            # Skip properties
            if not callable(getattr(trimesh.Trimesh, f)):
                continue
            setattr(self, f, _force_volume(getattr(self, f)))

    @property
    def name(self):
        """Name of this volume."""
        return self.metadata.get('name')

    @name.setter
    def name(self, value):
        self.metadata['name'] = value

    @property
    def color(self):
        """Color used for plotting."""
        return self.metadata.get('color')

    @color.setter
    def color(self, value):
        self.metadata['color'] = value

    @property
    def id(self):
        """ID of this volume."""
        return self.metadata.get('id')

    @id.setter
    def id(self, value):
        self.metadata['id'] = value

    @classmethod
    def from_csv(cls,
                 vertices: str,
                 faces: str,
                 name: Optional[str] = None,
                 color: Union[str,
                              Sequence[Union[int, float]]] = (1, 1, 1, .1),
                 volume_id: Optional[int] = None, **kwargs) -> 'Volume':
        """Load volume from csv files containing vertices and faces.

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

        return cls(faces=faces, vertices=vertices, name=name, color=color,
                   volume_id=volume_id)

    def to_csv(self, filename: str, **kwargs) -> None:
        """Save volume as two separated csv files containing vertices and faces.

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
    def from_json(cls,
                  filename: str,
                  import_kwargs: Dict = {},
                  **init_kwargs) -> 'Volume':
        """Load volume from json file containing vertices and faces.

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

        return cls(faces=data['faces'],
                   vertices=data['vertices'], **init_kwargs)

    @classmethod
    def from_object(cls,
                    obj: Any,
                    **init_kwargs) -> 'Volume':
        """Load volume from generic object that has ``.vertices`` and
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

        return cls(faces=obj.faces, vertices=obj.vertices, **init_kwargs)

    @classmethod
    def from_file(cls,
                  filename: str,
                  import_kwargs: Dict = {},
                  **init_kwargs) -> 'Volume':
        """Load volume from file.

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
            return cls.from_json(filename=filename,
                                 import_kwargs=import_kwargs,
                                 **init_kwargs)

        try:
            import trimesh
        except ImportError:
            raise ImportError('Unable to import: trimesh missing - please '
                              'install: "pip install trimesh"')
        except BaseException:
            raise

        tm = trimesh.load_mesh(filename, **import_kwargs)

        return cls.from_object(tm, **init_kwargs)

    def to_json(self, filename: str) -> None:
        """Save volume as json file.

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
    def combine(cls,
                x: Sequence['Volume'],
                name: str = 'comb_vol',
                color: Union[str,
                             Sequence[Union[int, float]]] = (1, 1, 1, .1)
                ) -> 'Volume':
        """Merge multiple volumes into a single object.

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

        return cls(vertices=vertices, faces=faces, name=name, color=color)

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box of this volume."""
        return self.bounds

    @property
    def verts(self) -> np.ndarray:
        """Legacy access to ``.vertices``."""
        return self.vertices

    @verts.setter
    def verts(self, v):
        self.vertices = v

    @property
    def center(self) -> np.ndarray:
        """Center of mass."""
        return np.mean(self.vertices, axis=0)

    def __getstate__(self):
        """Get state (used e.g. for pickling)."""
        return {k: v for k, v in self.__dict__.items() if not callable(v)}

    def __setstate__(self, d):
        """Update state (used e.g. for pickling)."""
        self.__dict__.update(d)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        """Return quick summary of the current geometry.

        Avoids computing properties.

        """
        elements = []
        if hasattr(self, 'name'):
            # for Trimesh
            elements.append(f'name={self.name}')
        if hasattr(self, 'id') and not isinstance(self.id, uuid.UUID):
            # for Trimesh
            elements.append(f'id={self.id}')
        if hasattr(self, 'color'):
            # for Trimesh
            elements.append(f'color={self.color}')
        if hasattr(self, 'vertices'):
            # for Trimesh and PointCloud
            elements.append(f'vertices.shape={self.vertices.shape}')
        if hasattr(self, 'faces'):
            # for Trimesh
            elements.append(f'faces.shape={self.faces.shape}')
        return f'<navis.Volume({", ".join(elements)})>'

    def __truediv__(self, other):
        """Implement division for vertex coordinates."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            return self.__mul__(1 / other)
        else:
            return NotImplemented

    def __mul__(self, other):
        """Implement multiplication for vertex coordinates."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            v = self.copy()
            v.vertices *= other
            return v
        else:
            return NotImplemented

    def resize(self,
               x: Union[float, int],
               method: Union[Literal['center'],
                             Literal['centroid'],
                             Literal['normals'],
                             Literal['origin']] = 'center',
               inplace: bool = False) -> Optional['Volume']:
        """Resize volume.

        Parameters
        ----------
        x :         int | float
                    Resizing factor. For methods "center", "centroid" and
                    "origin" this is the fraction of original size (e.g.
                    ``.5`` for half size). For method "normals", this is
                    is the absolute displacement (e.g. ``-1000`` to shrink
                    volume by that many units)!
        method :    "center" | "centroid" | "normals" | "origin"
                    Point in space to use for resizing.

                    .. list-table::
                        :widths: 15 75
                        :header-rows: 1

                        * - method
                          - explanation
                        * - center
                          - average of all vertices
                        * - centroid
                          - average of the triangle centroids weighted by the
                            area of each triangle.
                        * - origin
                          - resizes relative to origin of coordinate system
                            (0, 0, 0)
                        * - normals
                          - resize using face normals. Note that this method
                            uses absolute displacement for parameter ``x``.

        inplace :   bool, optional
                    If False, will return resized copy.

        Returns
        -------
        :class:`navis.Volume`
                    Resized copy of original volume. Only if ``inplace=False``.
        None
                    If ``inplace=True``.

        """
        assert isinstance(method, str)

        method = method.lower()

        perm_methods = ['center', 'origin', 'normals', 'centroid']
        if method not in perm_methods:
            raise ValueError(f'Unknown method "{method}". Allowed '
                             f'methods: {", ".join(perm_methods)}')

        if not inplace:
            v = self.copy()
        else:
            v = self

        if method == 'normals':
            v.vertices = v.vertices + (v.vertex_normals * x)
        else:
            # Get the center
            if method == 'center':
                cn = np.mean(v.vertices, axis=0)
            elif method == 'centroid':
                cn = v.centroid
            elif method == 'origin':
                cn = np.array([0, 0, 0])

            # Get vector from center to each vertex
            vec = v.vertices - cn

            # Multiply vector by resize factor
            vec *= x

            # Recalculate vertex positions
            v.vertices = vec + cn

        # Make sure to reset any pyoctree data on this volume
        if hasattr(v, 'pyoctree'):
            delattr(v, 'pyoctree')

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
        >>> import navis
        >>> vol = navis.example_volume('LH')
        >>> v = vol.plot3d(color = (255, 0, 0))

        """
        from .. import plotting

        if 'color' in kwargs:
            self.color = kwargs['color']

        return plotting.plot3d(self, **kwargs)

    def show(self, **kwargs):
        """See ``.plot3d``."""
        # This is mostly to override trimesh.Trimesh method
        return self.plot3d(**kwargs)

    def _outlines_3d(self, view='xy', **kwargs):
        """Generate 3d outlines along a given view (see ``.to_2d()``).

        Parameters
        ----------
        **kwargs
                    Keyword arguments passed to :func:`~navis.Volume.to_2d`.

        Returns
        -------
        list
                    Coordinates of 2d circumference.
                    e.g. ``[(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), ...]``
                    Third dimension is averaged.

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
              view: tuple = ('x', 'y'),
              invert_y: bool = False) -> Sequence[Union[float, int]]:
        """Compute the 2d alpha shape (concave hull) this volume.

        Uses Scipy Delaunay and shapely.

        Parameters
        ----------
        alpha:      float, optional
                    Alpha value to influence the gooeyness of the border.
                    Smaller numbers don't fall inward as much as larger
                    numbers. Too large, and you lose everything!
        view :      tuple
                    Determines axis. Can be prefixed with a '-' to invert
                    the axis.

        Returns
        -------
        list
                    Coordinates of 2d circumference
                    e.g. ``[(x1, y1), (x2, y2), (x3, y3), ...]``

        """
        def add_edge(edges, edge_points, coords, i, j):
            """Add line between the i-th and j-th points."""
            if (i, j) in edges or (j, i) in edges:
                # already added
                return
            edges.add((i, j))
            edge_points.append(coords[[i, j]])

        accepted_views = ['x', 'z', 'y', '-x', '-z', '-y']

        for ax in view:
            if ax not in accepted_views:
                raise ValueError(f'Unable to parse "{ax}" view')

        try:
            from shapely.ops import cascaded_union, polygonize  # type: ignore
            import shapely.geometry as geometry  # type: ignore
        except ImportError:
            raise ImportError('This function needs the shapely package.')

        coords: np.ndarray

        map = {'x': 0, 'y': 1, 'z': 2}

        x_ix = map[view[0].replace('-', '').replace('+', '')]
        y_ix = map[view[1].replace('-', '').replace('+', '')]

        xmod = -1 if '-' in view[0] else 1
        ymod = -1 if '-' in view[1] else 1
        coords = self.vertices[:, [x_ix, y_ix]] * np.array([xmod, ymod])

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

    def validate(self):
        """Use trimesh to try and fix issues (holes/normals)."""
        if not self.is_volume:
            logger.info("Mesh not valid, attempting to fix")
            self.fill_holes()
            self.fix_normals()
            if not self.is_volume:
                raise utils.VolumeError("Mesh is not a volume "
                                        "(e.g. not watertight, incorrect "
                                        "winding) and could not be fixed.")


def _force_volume(f):
    """Convert result from ``trimesh.Trimesh`` to ``navis.Volume``."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        if isinstance(res, trimesh.Trimesh):
            res = Volume(res.vertices, res.faces)
        return res
    return wrapper
