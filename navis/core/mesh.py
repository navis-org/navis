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

import copy
import numbers
import os
import pint
import uuid
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import trimesh as tm

from io import BufferedIOBase

from typing import Union, Optional

from .. import utils, config, meshes
from .base import BaseNeuron


try:
    import xxhash
except ImportError:
    xxhash = None


__all__ = ['MeshNeuron']

# Set up logging
logger = config.logger

# This is to prevent pint to throw a warning about numpy integration
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


class MeshNeuron(BaseNeuron):
    """Neuron represented as mesh with vertices and faces.

    Parameters
    ----------
    x
                    Data to construct neuron from:
                     - any object that has ``.vertices`` and ``.faces``
                       properties (e.g. a trimesh.Trimesh)
                     - a dictionary ``{"vertices": (N,3), "faces": (M, 3)}``
                     - filepath to a file that can be read by ``trimesh.load``
                     - ``None`` will initialize an empty MeshNeuron

    units :         str | pint.Units | pint.Quantity
                    Units for coordinates. Defaults to ``None`` (dimensionless).
                    Strings must be parsable by pint: e.g. "nm", "um",
                    "micrometer" or "8 nanometers".
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
    TEMP_ATTR = ['trimesh', '_memory_usage']

    def __init__(self,
                 x: Union[pd.DataFrame,
                          BufferedIOBase,
                          str,
                          'TreeNeuron',
                          nx.DiGraph],
                 units: Union[pint.Unit, str] = None,
                 validate: bool = False,
                 **metadata
                 ):
        """Initialize Mesh Neuron."""
        super().__init__()

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
        else:
            raise utils.ConstructionError(f'Unable to construct MeshNeuron from "{type(x)}"')

        for k, v in metadata.items():
            setattr(self, k, v)

        if not getattr(self, 'id', None):
            self.id = uuid.uuid4()

        if validate:
            self.validate()

        self.units = units

    def __getattr__(self, key):
        """We will use this magic method to calculate some attributes on-demand."""
        # Note that we're mixing @property and __getattr__ which causes problems:
        # if a @property raises an Exception, Python falls back to __getattr__
        # and traceback is lost!

        if key == 'trimesh':
            self.trimesh = tm.Trimesh(vertices=self._vertices, faces=self._faces)
            return self.trimesh

        # See if trimesh can help us
        if hasattr(self.trimesh, key):
            return getattr(self.trimesh, key)

        # Last ditch effort - maybe the base class knows the key?
        return super().__getattr__(key)

    def __getstate__(self):
        """Get state (used e.g. for pickling)."""
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}

        # We don't need the trimesh object
        if 'trimesh' in state:
            _ = state.pop('trimesh')

        return state

    def __setstate__(self, d):
        """Update state (used e.g. for pickling)."""
        self.__dict__.update(d)

    def __truediv__(self, other):
        """Implement division for coordinates (vertices, connectors)."""
        if isinstance(other, (numbers.Number, list, np.ndarray)):
            # If a number, consider this an offset for coordinates
            n = self.copy()
            _ = np.divide(n.vertices, other, out=n.vertices, casting='unsafe')
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] /= other

            # Convert units
            # If division is isometric
            if isinstance(other, numbers.Number):
                n.units = (n.units * other).to_compact()
            # If other is iterable but division is still isometric
            elif len(set(other)) == 1:
                n.units = (n.units * other[0]).to_compact()
            # If non-isometric remove units
            else:
                n.units = None

            self._clear_temp_attr()

            return n
        else:
            return NotImplemented

    def __mul__(self, other):
        """Implement multiplication for coordinates (vertices, connectors)."""
        if isinstance(other, (numbers.Number, list, np.ndarray)):
            # If a number, consider this an offset for coordinates
            n = self.copy()
            _ = np.multiply(n.vertices, other, out=n.vertices, casting='unsafe')
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] *= other

            # Convert units
            # If multiplication is isometric
            if isinstance(other, numbers.Number):
                n.units = (n.units / other).to_compact()
            # If other is iterable but multiplication is still isometric
            elif len(set(other)) == 1:
                n.units = (n.units / other[0]).to_compact()
            # If non-isometric remove units
            else:
                n.units = None

            self._clear_temp_attr()

            return n
        else:
            return NotImplemented

    def _clear_temp_attr(self, exclude: list = []) -> None:
        """Clear temporary attributes."""
        for a in [at for at in self.TEMP_ATTR if at not in exclude]:
            try:
                delattr(self, a)
                logger.debug(f'Neuron {id(self)}: {a} cleared')
            except BaseException:
                logger.debug(f'Neuron {id(self)}: Unable to clear temporary attribute "{a}"')
                pass

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

    @property
    def sampling_resolution(self) -> float:
        """Average distance vertices. """
        return float(self.trimesh.edges_unique_length.mean())

    @property
    def type(self) -> str:
        """Neuron type."""
        return 'navis.MeshNeuron'

    def copy(self) -> 'MeshNeuron':
        """Return a copy of the neuron."""
        no_copy = ['_lock']

        # Generate new neuron
        x = self.__class__(None)
        # Override with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        return x

    def validate(self):
        """Use trimesh to try and fix some common mesh issues.

        See :func:`navis.fix_mesh` for details.

        """
        meshes.fix_mesh(self, inplace=True)
