#    This script is part of navis (http://www.github.com/navis-org/navis).
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

import copy
import numbers
import os
import pint
import warnings
import scipy

import networkx as nx
import numpy as np
import pandas as pd
import skeletor as sk
import trimesh as tm

from io import BufferedIOBase
from typing import Union, Optional

from .. import utils, config, meshes, conversion, graph
from .base import BaseNeuron
from .skeleton import TreeNeuron
from .core_utils import temp_property


try:
    import xxhash
except ImportError:
    xxhash = None


__all__ = ['MeshNeuron']

# Set up logging
logger = config.get_logger(__name__)

# This is to prevent pint to throw a warning about numpy integration
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


class MeshNeuron(BaseNeuron):
    """Neuron represented as mesh with vertices and faces.

    Parameters
    ----------
    x :             mesh-like | tuple | dictionary | filepath | None
                    Data to construct neuron from:
                     - any object that has ``.vertices`` and ``.faces``
                       properties (e.g. a trimesh.Trimesh)
                     - a tuple ``(vertices, faces)``
                     - a dictionary ``{"vertices": (N, 3), "faces": (M, 3)}``
                     - filepath to a file that can be read by ``trimesh.load``
                     - ``None`` will initialize an empty MeshNeuron
                     - ``skeletor.Skeleton`` will use the mesh and the skeleton
                       (including the vertex to node map)

    units :         str | pint.Units | pint.Quantity
                    Units for coordinates. Defaults to ``None`` (dimensionless).
                    Strings must be parsable by pint: e.g. "nm", "um",
                    "micrometer" or "8 nanometers".
    process :       bool
                    If True (default and highly recommended), will remove NaN
                    and infinite values, and merge duplicate vertices.
    validate :      bool
                    If True, will try to fix some common problems with
                    meshes. See ``navis.fix_mesh`` for details.
    **metadata
                    Any additional data to attach to neuron.

    """

    connectors: Optional[pd.DataFrame]

    vertices: np.ndarray
    faces: np.ndarray

    soma: Optional[Union[list, np.ndarray]]

    #: Attributes used for neuron summary
    SUMMARY_PROPS = ['type', 'name', 'units', 'n_vertices', 'n_faces']

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ['name', 'n_vertices', 'n_faces']

    #: Temporary attributes that need clearing when neuron data changes
    TEMP_ATTR = ['_memory_usage', '_trimesh', '_skeleton', '_igraph', '_graph_nx']

    #: Core data table(s) used to calculate hash
    CORE_DATA = ['vertices', 'faces']

    def __init__(self,
                 x,
                 units: Union[pint.Unit, str] = None,
                 process: bool = True,
                 validate: bool = False,
                 **metadata
                 ):
        """Initialize Mesh Neuron."""
        super().__init__()

        # Lock neuron during initialization
        self._lock = 1
        self._trimesh = None  # this is required to avoid recursion during init

        if isinstance(x, MeshNeuron):
            self.__dict__.update(x.copy().__dict__)
            self.vertices, self.faces = x.vertices, x.faces
        elif hasattr(x, 'faces') and hasattr(x, 'vertices'):
            self.vertices, self.faces = x.vertices, x.faces
        elif isinstance(x, dict):
            if 'faces' not in x or 'vertices' not in x:
                raise ValueError('Dictionary must contain "vertices" and "faces"')
            self.vertices, self.faces = x['vertices'], x['faces']
        elif isinstance(x, str) and os.path.isfile(x):
            m = tm.load(x)
            self.vertices, self.faces = m.vertices, m.faces
        elif isinstance(x, type(None)):
            # Empty neuron
            self.vertices, self.faces = np.zeros((0, 3)), np.zeros((0, 3))
        elif isinstance(x, sk.Skeleton):
            self.vertices, self.faces = x.mesh.vertices, x.mesh.faces
            self._skeleton = TreeNeuron(x)
        elif isinstance(x, tuple):
            if len(x) != 2 or any([not isinstance(v, np.ndarray) for v in x]):
                raise TypeError('Expect tuple to be two arrays: (vertices, faces)')
            self.vertices, self.faces = x[0], x[1]
        else:
            raise utils.ConstructionError(f'Unable to construct MeshNeuron from "{type(x)}"')

        for k, v in metadata.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                raise AttributeError(f"Unable to set neuron's `{k}` attribute.")

        if process and self.vertices.shape[0]:
            # For some reason we can't do self._trimesh at this stage
            _trimesh = tm.Trimesh(self.vertices, self.faces,
                                  process=process,
                                  validate=validate)
            self.vertices = _trimesh.vertices
            self.faces = _trimesh.faces

        self._lock = 0

        if validate:
            self.validate()

        self.units = units

    def __getattr__(self, key):
        """We will use this magic method to calculate some attributes on-demand."""
        # Note that we're mixing @property and __getattr__ which causes problems:
        # if a @property raises an Exception, Python falls back to __getattr__
        # and traceback is lost!

        # Last ditch effort - maybe the base class knows the key?
        return super().__getattr__(key)

    def __getstate__(self):
        """Get state (used e.g. for pickling)."""
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}

        # We don't need the trimesh object
        if '_trimesh' in state:
            _ = state.pop('_trimesh')

        return state

    def __setstate__(self, d):
        """Update state (used e.g. for pickling)."""
        self.__dict__.update(d)

    def __truediv__(self, other, copy=True):
        """Implement division for coordinates (vertices, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self
            _ = np.divide(n.vertices, other, out=n.vertices, casting='unsafe')
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] /= other

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units * other).to_compact()

            self._clear_temp_attr()

            return n
        return NotImplemented

    def __mul__(self, other, copy=True):
        """Implement multiplication for coordinates (vertices, connectors)."""
        if isinstance(other, numbers.Number) or utils.is_iterable(other):
            # If a number, consider this an offset for coordinates
            n = self.copy() if copy else self
            _ = np.multiply(n.vertices, other, out=n.vertices, casting='unsafe')
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] *= other

            # Convert units
            # Note: .to_compact() throws a RuntimeWarning and returns unchanged
            # values  when `units` is a iterable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n.units = (n.units / other).to_compact()

            self._clear_temp_attr()

            return n
        return NotImplemented

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box (includes connectors)."""
        mn = np.min(self.vertices, axis=0)
        mx = np.max(self.vertices, axis=0)

        if self.has_connectors:
            cn_mn = np.min(self.connectors[['x', 'y', 'z']].values, axis=0)
            cn_mx = np.max(self.connectors[['x', 'y', 'z']].values, axis=0)

            mn = np.min(np.vstack((mn, cn_mn)), axis=0)
            mx = np.max(np.vstack((mx, cn_mx)), axis=0)

        return np.vstack((mn, mx)).T

    @property
    def vertices(self):
        """Vertices making up the neuron."""
        return self._vertices

    @vertices.setter
    def vertices(self, verts):
        if not isinstance(verts, np.ndarray):
            raise TypeError(f'Vertices must be numpy array, got "{type(verts)}"')
        if verts.ndim != 2:
            raise ValueError('Vertices must be 2-dimensional array')
        self._vertices = verts
        self._clear_temp_attr()

    @property
    def faces(self):
        """Faces making up the neuron."""
        return self._faces

    @faces.setter
    def faces(self, faces):
        if not isinstance(faces, np.ndarray):
            raise TypeError(f'Faces must be numpy array, got "{type(faces)}"')
        if faces.ndim != 2:
            raise ValueError('Faces must be 2-dimensional array')
        self._faces = faces
        self._clear_temp_attr()

    @temp_property
    def igraph(self) -> 'igraph.Graph':
        """iGraph representation of the vertex connectivity."""
        # If igraph does not exist, create and return
        if not hasattr(self, '_igraph'):
            # This also sets the attribute
            self._igraph = graph.neuron2igraph(self, raise_not_installed=False)
        return self._igraph

    @temp_property
    def graph(self) -> nx.DiGraph:
        """Networkx Graph representation of the vertex connectivity."""
        # If graph does not exist, create and return
        if not hasattr(self, '_graph_nx'):
            # This also sets the attribute
            self._graph_nx = graph.neuron2nx(self)
        return self._graph_nx

    @property
    def sampling_resolution(self) -> float:
        """Average distance between vertices."""
        return float(self.trimesh.edges_unique_length.mean())

    @property
    def volume(self) -> float:
        """Volume of the neuron.

        Calculated from the surface integral. Garbage if neuron is not
        watertight.

        """
        return float(self.trimesh.volume)

    @temp_property
    def skeleton(self) -> 'TreeNeuron':
        """Skeleton representation of this neuron.

        Uses :func:`navis.mesh2skeleton`.

        """
        if not hasattr(self, '_skeleton'):
            self._skeleton = self.skeletonize()
        return self._skeleton

    @skeleton.setter
    def skeleton(self, s):
        """Attach skeleton respresentation for this neuron."""
        if isinstance(s, sk.Skeleton):
            s = TreeNeuron(s, id=self.id, name=self.name)
        elif not isinstance(s, TreeNeuron):
            raise TypeError(f'`.skeleton` must be a TreeNeuron, got "{type(s)}"')
        self._skeleton = s

    @property
    def type(self) -> str:
        """Neuron type."""
        return 'navis.MeshNeuron'

    @temp_property
    def trimesh(self):
        """Trimesh representation of the neuron."""
        if not getattr(self, '_trimesh', None):
            self._trimesh = tm.Trimesh(vertices=self._vertices,
                                       faces=self._faces,
                                       process=False)
        return self._trimesh

    def copy(self) -> 'MeshNeuron':
        """Return a copy of the neuron."""
        no_copy = ['_lock']

        # Generate new neuron
        x = self.__class__(None)
        # Override with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        return x

    def snap(self, locs, to='vertices'):
        """Snap xyz location(s) to closest vertex or synapse.

        Parameters
        ----------
        locs :      (N, 3) array | (3, ) array
                    Either single or multiple XYZ locations.
        to :        "vertices" | "connectors"
                    Whether to snap to vertex or connector.

        Returns
        -------
        ix :        int | list of int
                    Index/indices of the closest vertex/connector.
        dist :      float | list of float
                    Distance(s) to the closest vertex/connector.

        Examples
        --------
        >>> import navis
        >>> n = navis.example_neurons(1, kind='mesh')
        >>> ix, dist = n.snap([0, 0, 0])
        >>> ix
        4134

        """
        locs = np.asarray(locs).astype(self.vertices.dtype)

        is_single = (locs.ndim == 1 and len(locs) == 3)
        is_multi = (locs.ndim == 2 and locs.shape[1] == 3)
        if not is_single and not is_multi:
            raise ValueError('Expected a single (x, y, z) location or a '
                             '(N, 3) array of multiple locations')

        if to not in ('vertices', 'vertex', 'connectors', 'connectors'):
            raise ValueError('`to` must be "vertices" or "connectors", '
                             f'got {to}')

        # Generate tree
        tree = scipy.spatial.cKDTree(data=self.vertices)

        # Find the closest node
        dist, ix = tree.query(locs)

        return ix, dist

    def skeletonize(self, method='wavefront', heal=True, inv_dist=None, **kwargs) -> 'TreeNeuron':
        """Skeletonize mesh.

        See :func:`navis.conversion.mesh2skeleton` for details.

        Parameters
        ----------
        method :    "wavefront" | "teasar"
                    Method to use for skeletonization.
        heal :      bool
                    Whether to heal a fragmented skeleton after skeletonization.
        inv_dist :  int | float
                    Only required for method "teasar": invalidation distance for
                    the traversal. Smaller ``inv_dist`` captures smaller features
                    but is slower and vice versa. A good starting value is around
                    2-5 microns.
        **kwargs
                    Additional keyword are passed through to
                    :func:`navis.conversion.mesh2skeleton`.

        Returns
        -------
        skeleton :  navis.TreeNeuron

        """
        return conversion.mesh2skeleton(self, method=method, heal=heal,
                                        inv_dist=inv_dist, **kwargs)

    def validate(self, inplace=False):
        """Use trimesh to try and fix some common mesh issues.

        See :func:`navis.fix_mesh` for details.

        """
        return meshes.fix_mesh(self, inplace=inplace)
