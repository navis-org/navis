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
import functools
import hashlib
import numbers
import os
import pint
import types
import uuid
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import trimesh as tm

from io import BufferedIOBase, StringIO

from typing import Union, Callable, List, Sequence, Optional, Dict, overload, Any, Tuple
from typing_extensions import Literal

from .. import graph, morpho, utils, config, core, sampling, intersection, meshes
from .. import io  # type: ignore # double import

try:
    import xxhash
except ImportError:
    xxhash = None

try:
    from pykdtree.kdtree import KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree

__all__ = ['Neuron', 'TreeNeuron', 'MeshNeuron', 'Dotprops']

# Set up logging
logger = config.logger

# This is to prevent pint to throw a warning about numpy integration
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


def temp_property(func):
    """Check if neuron is stale. Clear cached temporary attributes if it is."""
    @property
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # Do nothing if neurons is locked
        if not self.is_locked:
            if self.is_stale:
                self._clear_temp_attr()
        return func(*args, **kwargs)
    return wrapper


def requires_nodes(func):
    """Return ``None`` if neuron has no nodes."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        # Return 0
        if isinstance(self.nodes, str) and self.nodes == 'NA':
            return 'NA'
        if not isinstance(self.nodes, pd.DataFrame):
            return None
        return func(*args, **kwargs)
    return wrapper


def Neuron(x: Union[nx.DiGraph, str, pd.DataFrame, 'TreeNeuron', 'MeshNeuron'],
           **metadata):
    """Constructor for Neuron objects. Depending on the input, either a
    ``TreeNeuron`` or a ``MeshNeuron`` is returned.

    Parameters
    ----------
    x
                        Anything that can construct a :class:`~navis.TreeNeuron`
                        or :class:`~navis.MeshNeuron`.
    **metadata
                        Any additional data to attach to neuron.

    See Also
    --------
    :func:`navis.read_swc`
                        Gives you more control over how data is extracted from
                        SWC file.
    :func:`navis.example_neurons`
                        Loads some example neurons provided.

    """
    try:
        return TreeNeuron(x, **metadata)
    except utils.ConstructionError:
        try:
            return MeshNeuron(x, **metadata)
        except utils.ConstructionError:
            pass
        except BaseException:
            raise
    except BaseException:
        raise

    raise utils.ConstructionError(f'Unable to construct neuron from "{type(x)}"')


class BaseNeuron:
    """Base Neuron."""

    name: Optional[str]
    id: Union[int, str, uuid.UUID]

    #: Unit space for this neuron. Some functions, like soma detection are
    #: sensitive to units (if provided)
    #: Default = micrometers
    units: Union[pint.Unit, pint.Quantity]

    volume: Union[int, float]

    connectors: Optional[pd.DataFrame]

    #: Attributes used for neuron summary
    SUMMARY_PROPS = ['type', 'name', 'units']

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ['name']

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, key):
        """Get attribute."""
        if key.startswith('has_'):
            key = key[key.index('_') + 1:]
            if hasattr(self, key):
                data = getattr(self, key)
                if isinstance(data, pd.DataFrame):
                    if not data.empty:
                        return True
                    else:
                        return False
                # This is necessary because np.any does not like strings
                elif isinstance(data, str):
                    if data == 'NA' or not data:
                        return False
                    return True
                elif utils.is_iterable(data) and len(data) > 0:
                    return True
                elif data:
                    return True
            return False
        elif key.startswith('n_'):
            key = key[key.index('_') + 1:]
            if hasattr(self, key):
                data = getattr(self, key, None)
                if isinstance(data, pd.DataFrame):
                    return data.shape[0]
                elif utils.is_iterable(data):
                    return len(data)
                elif isinstance(data, str) and data == 'NA':
                    return 'NA'
            return None

        raise AttributeError(f'Attribute "{key}" not found')

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.summary())

    def __copy__(self):
        return self.copy(deepcopy=False)

    def __deepcopy__(self):
        return self.copy(deepcopy=True)

    def __eq__(self, other):
        """Implement neuron comparison."""
        if isinstance(other, TreeNeuron):
            # We will do this sequentially and stop as soon as we find a
            # discrepancy -> this saves tons of time!
            for at in self.EQ_ATTRIBUTES:
                comp = getattr(self, at, None) == getattr(other, at, None)
                if isinstance(comp, np.ndarray) and not all(comp):
                    return False
                elif comp is False:
                    return False
            # If all comparisons have passed, return True
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Generate a hashable value."""
        # We will simply use the neuron's memory address
        return id(self)

    def __add__(self, other):
        """Implement addition."""
        if isinstance(other, TreeNeuron):
            return core.NeuronList([self, other])
        else:
            return NotImplemented

    def _repr_html_(self):
        frame = self.summary().to_frame()
        frame.columns = ['']
        # return self._gen_svg_thumbnail() + frame._repr_html_()
        return frame._repr_html_()

    def _gen_svg_thumbnail(self):
        """Generate 2D plot for thumbnail."""
        import matplotlib.pyplot as plt
        # Store some previous states
        prev_level = logger.getEffectiveLevel()
        prev_pbar = config.pbar_hide
        prev_int = plt.isinteractive()

        plt.ioff()  # turn off interactive mode
        logger.setLevel('WARNING')
        config.pbar_hide = True
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(111)
        fig, ax = self.plot2d(connectors=False, ax=ax)
        output = StringIO()
        fig.savefig(output, format='svg')

        if prev_int:
            plt.ion()  # turn on interactive mode
        logger.setLevel(prev_level)
        config.pbar_hide = prev_pbar
        _ = plt.clf()
        return output.getvalue()

    def _register_attr(self, name, value, summary=True, temporary=False):
        """Set and register attribute.

        Use this if you want an attribute to be used for the summary or cleared
        when temporary attributes are cleared.
        """
        setattr(self, name, value)

        # If this is an easy to summarize attribute, add to summary
        if summary and name not in self.SUMMARY_PROPS:
            if isinstance(value, (numbers.Number, str)):
                self.SUMMARY_PROPS.append(name)
            else:
                logger.error(f'Attributes of type "{type(value)}" can not be '
                             'added to summary')

        if temporary:
            self.TEMP_ATTR.append(name)

    def _unregister_attr(self, name):
        """Remove and unregister attribute."""
        if name in self.SUMMARY_PROPS:
            self.SUMMARY_PROPS.remove(name)

        if name in self.TEMP_ATTR:
            self.TEMP_ATTR.remove(name)

        delattr(self, name)

    @property
    def datatables(self) -> List[str]:
        """Names of all DataFrames attached to this neuron."""
        return [k for k, v in self.__dict__.items() if isinstance(v, pd.DataFrame)]

    @property
    def id(self) -> Any:
        """Hashable ID."""
        return getattr(self, '_id', None)

    @id.setter
    def id(self, value):
        try:
            hash(value)
        except BaseException:
            raise ValueError('id must be hashable')
        self._id = value

    @property
    def label(self) -> str:
        """Label (e.g. for legends)."""
        # If explicitly set return that label
        if getattr(self, '_label', None):
            return self._label

        # If no label set, produce one from name + id (optional)
        name = getattr(self, 'name', None)
        id = getattr(self, 'id', None)

        # If no name, use type
        if not name:
            name = self.type

        label = name

        # Use ID only if not a UUID
        if not isinstance(id, uuid.UUID):
            # And if it can be turned into a string
            try:
                id = str(id)
            except BaseException:
                id = ''

            # Only use ID if it is not the same as name
            if id and name != id:
                label += f' ({id})'

        return label

    @label.setter
    def label(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f'label must be string, got "{type(value)}"')
        self._label = value

    @property
    def name(self) -> str:
        """Neuron name."""
        return getattr(self, '_name', None)

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def connectors(self) -> pd.DataFrame:
        """Connector table. If none, will return ``None``."""
        return getattr(self, '_connectors', None)

    @connectors.setter
    def connectors(self, v):
        if isinstance(v, type(None)):
            self._connectors = None
        else:
            self._connectors = utils.validate_table(v,
                                                    required=['x', 'y', 'z'],
                                                    rename=True,
                                                    restrict=False)

    @property
    def presynapses(self):
        """Table with presynapses.

        Requires a "type" column in connector table. Will look for type labels
        that include "pre" or that equal 0 or "0".
        """
        if not isinstance(getattr(self, 'connectors', None), pd.DataFrame):
            raise ValueError('No connector table found.')
        # Make an educated guess what presynapses are
        types = self.connectors['type'].unique()
        pre = [t for t in types if 'pre' in str(t) or t in [0, "0"]]

        if len(pre) == 0:
            logger.debug(f'Unable to find presynapses in types: {types}')
            return self.connectors.iloc[0:0]  # return empty DataFrame
        elif len(pre) > 1:
            raise ValueError(f'Found ambigous presynapse labels: {pre}')

        return self.connectors[self.connectors['type'] == pre[0]]

    @property
    def postsynapses(self):
        """Table with postsynapses.

        Requires a "type" column in connector table. Will look for type labels
        that include "post" or that equal 1 or "1".
        """
        if not isinstance(getattr(self, 'connectors', None), pd.DataFrame):
            raise ValueError('No connector table found.')
        # Make an educated guess what presynapses are
        types = self.connectors['type'].unique()
        post = [t for t in types if 'post' in str(t) or t in [1, "1"]]

        if len(post) == 0:
            logger.debug(f'Unable to find postsynapses in types: {types}')
            return self.connectors.iloc[0:0]  # return empty DataFrame
        elif len(post) > 1:
            raise ValueError(f'Found ambigous postsynapse labels: {post}')

        return self.connectors[self.connectors['type'] == post[0]]

    @property
    def units(self) -> str:
        """Units for coordinate space."""
        # Note that we are regenerating the pint.Quantity from the string
        # That is to avoid problems with pickling .e.g when using multiprocessing
        return config.ureg(getattr(self, '_unit_str', None))

    @units.setter
    def units(self, v: Union[pint.Unit, pint.Quantity, str, None]):
        # Note that we are storing the string, not the actual pint.Quantity
        # That is to avoid problems with pickling .e.g when using multiprocessing
        if isinstance(v, str):
            self._unit_str = str(config.ureg(v))
        elif isinstance(v, (pint.Unit, pint.Quantity)):
            self._unit_str = str(v)
        elif isinstance(v, type(None)):
            self._unit_str = None
        else:
            raise TypeError(f'Expect str or pint Unit/Quantity, got "{type(v)}"')

    @property
    def is_stale(self):
        """Test if temporary attributes might be outdated."""
        # Always returns False for BaseNeurons
        return False

    @property
    def is_locked(self):
        """Test if neuron is locked."""
        return getattr(self, '_lock', 0) > 0

    @property
    def type(self) -> str:
        """Neuron type."""
        return 'navis.BaseNeuron'

    def convert_units(self,
                      to: Union[pint.Unit, str],
                      inplace: bool = False) -> Optional['BaseNeuron']:
        """Convert coordinates to different unit.

        Only works if neuron's ``.units`` is not dimensionless.

        Parameters
        ----------
        to :        pint.Unit | str
                    Units to convert to. If string, must be parsable by pint.
                    See examples.
        inplace :   bool, optional
                    If True will convert in place. If not will return a
                    copy.

        Examples
        --------
        >>> import navis
        >>> n = navis.example_neurons(1)
        >>> n.units
        <Quantity(8, 'nanometer')>
        >>> n.cable_length
        266476.8
        >>> n2 = n.convert_units('um')
        >>> n2.units
        <Quantity(1.0, 'micrometer')>
        >>> n2.cable_length
        2131.8

        """
        if not isinstance(self.units, (pint.Unit, pint.Quantity)):
            raise ValueError("Unable to convert: neuron has no units set.")

        if inplace:
            n = self
        else:
            n = self.copy()

        # Catch pint's UnitStrippedWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Get factor by which we have to multiply to get to target units
            conv = n.units.to(to).magnitude
            # Multiply by conversion factor
            n *= conv

        n._clear_temp_attr(exclude=['classify_nodes'])

        if not inplace:
            return n

    def copy(self) -> 'BaseNeuron':
        """Return a copy of the neuron."""
        # Attributes not to copy
        no_copy = ['_lock']
        # Generate new empty neuron
        x = self.__class__()
        # Override with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        return x

    def summary(self, add_props=None) -> pd.Series:
        """Get a summary of this neuron."""
        # Do not remove the list -> otherwise we might change the original!
        props = list(self.SUMMARY_PROPS)

        # Add .id to summary if not a generic UUID
        if not isinstance(self.id, uuid.UUID):
            props.insert(2, 'id')

        if add_props:
            props, ix = np.unique(np.append(props, add_props),
                                  return_inverse=True)
            props = props[ix]

        # This is to catch an annoying "UnitStrippedWarning" with pint
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = pd.Series([getattr(self, at, 'NA') for at in props],
                          index=props)

        return s

    def plot2d(self, **kwargs):
        """Plot neuron using :func:`navis.plot2d`.

        Parameters
        ----------
        **kwargs
                Will be passed to :func:`navis.plot2d`.
                See ``help(navis.plot2d)`` for a list of keywords.

        See Also
        --------
        :func:`navis.plot2d`
                    Function called to generate 2d plot.

        """
        from ..plotting import plot2d

        return plot2d(self, **kwargs)

    def plot3d(self, **kwargs):
        """Plot neuron using :func:`navis.plot3d`.

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
        >>> nl = navis.example_neurons()
        >>> #Plot with connectors
        >>> viewer = nl.plot3d(connectors=True)

        """
        from ..plotting import plot3d

        return plot3d(core.NeuronList(self, make_copy=False), **kwargs)

    def memory_usage(self, deep=False, estimate=False):
        """Return estimated memory usage of this neuron.

        Works by going over attached data (numpy arrays and pandas DataFrames
        such as vertices, nodes, etc) and summing up their size in memory.

        Parameters
        ----------
        deep :      bool
                    Pass to pandas DataFrames. If True will inspect data of
                    object type too.
        estimate :  bool
                    If True, we will only estimate the size. This is
                    considerably faster but will slightly underestimate the
                    memory usage.

        Returns
        -------
        int
                    Memory usage in bytes.

        """
        # We will use a very simply caching here
        # We don't check whether neuron is stale because that causes
        # additional overhead and we want this function to be as fast
        # as possible
        if hasattr(self, "_memory_usage"):
            mu = self._memory_usage
            if mu['deep'] == deep and mu['estimate'] == estimate:
                return mu['size']

        size = 0
        if not estimate:
            for k, v in self.__dict__.items():
                if isinstance(v, np.ndarray):
                    size += v.nbytes
                elif isinstance(v, pd.DataFrame):
                    size += v.memory_usage(deep=deep).sum()
                elif isinstance(v, pd.Series):
                    size += v.memory_usage(deep=deep)
        else:
            for k, v in self.__dict__.items():
                if isinstance(v, np.ndarray):
                    size += v.dtype.itemsize * v.size
                elif isinstance(v, pd.DataFrame):
                    for dt in v.dtypes.values:
                        if isinstance(dt, pd.CategoricalDtype):
                            size += len(dt.categories) * dt.itemsize
                        else:
                            size += dt.itemsize * v.shape[0]
                elif isinstance(v, pd.Series):
                    if isinstance(v.dtype, pd.CategoricalDtype):
                        size += len(dt.categories) * dt.itemsize
                    else:
                        size += v.dtype.itemsize * v.shape[0]

        self._memory_usage = {'deep': deep,
                              'estimate': estimate,
                              'size': size}

        return size


class MeshNeuron(BaseNeuron):
    """Neuron represented a by a mesh with vertices and faces.

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


class TreeNeuron(BaseNeuron):
    """Neuron represented as hierarchical tree (i.e. a skeleton).

    Parameters
    ----------
    x
                    Data to construct neuron from:
                     - ``pandas.DataFrame`` is expected to be SWC table
                     - ``pandas.Series`` is expected to have a DataFrame as
                       ``.nodes`` - additional properties will be attached
                       as meta data
                     - ``str`` is treated as SWC file name
                     - ``BufferedIOBase`` e.g. from ``open(filename)``
                     - ``networkx.DiGraph`` parsed by `navis.nx2neuron`
                     - ``None`` will initialize an empty neuron
                     - ``TreeNeuron`` - in this case we will try to copy every
                       attribute
    units :         str | pint.Units | pint.Quantity
                    Units for coordinates. Defaults to ``None`` (dimensionless).
                    Strings must be parsable by pint: e.g. "nm", "um",
                    "micrometer" or "8 nanometers".
    **metadata
                    Any additional data to attach to neuron.

    """

    nodes: pd.DataFrame

    graph: 'nx.DiGraph'
    igraph: 'igraph.Graph'  # type: ignore  # doesn't know iGraph

    n_branches: int
    n_leafs: int
    cable_length: Union[int, float]

    segments: List[list]
    small_segments: List[list]

    root: np.ndarray

    soma: Optional[Union[int, str]]

    #: Minimum radius for soma detection. Set to ``None`` if no tag needed.
    #: Default = 1 micron
    soma_detection_radius: Union[float, int, pint.Quantity] = 1 * config.ureg.um
    #: Label for soma detection. Set to ``None`` if no tag needed. Default = 1.
    soma_detection_label: Union[float, int, str] = 1
    #: Soma radius (e.g. for plotting). If string, must be column in nodes
    #: table. Default = 'radius'.
    soma_radius: Union[float, int, str] = 'radius'
    # Set default function for soma finding. Default = :func:`navis.morpho.find_soma`
    _soma: Union[Callable[['TreeNeuron'], Sequence[int]], int] = morpho.find_soma

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ['n_nodes', 'n_connectors', 'soma', 'root',
                     'n_branches', 'n_leafs', 'cable_length', 'name']

    #: Temporary attributes that need to be regenerated when data changes.
    TEMP_ATTR = ['_igraph', '_graph_nx', '_segments', '_small_segments',
                 '_geodesic_matrix', 'centrality_method', '_simple',
                 '_cable_length', '_memory_usage']

    #: Attributes used for neuron summary
    SUMMARY_PROPS = ['type', 'name', 'n_nodes', 'n_connectors', 'n_branches',
                     'n_leafs', 'cable_length', 'soma', 'units']

    def __init__(self,
                 x: Union[pd.DataFrame,
                          BufferedIOBase,
                          str,
                          'TreeNeuron',
                          nx.DiGraph],
                 units: Union[pint.Unit, str] = None,
                 **metadata
                 ):
        """Initialize Skeleton Neuron."""
        super().__init__()

        # Lock neuron during construction
        self._lock = 1

        if isinstance(x, pd.DataFrame):
            self.nodes = x
        elif isinstance(x, pd.Series):
            if not hasattr(x, 'nodes'):
                raise ValueError('pandas.Series must have `nodes` entry.')
            elif not isinstance(x.nodes, pd.DataFrame):
                raise TypeError(f'Nodes must be pandas DataFrame, got "{type(x.nodes)}"')
            self.nodes = x.nodes
            metadata.update(x.to_dict())
        elif isinstance(x, nx.Graph):
            self.nodes = graph.nx2neuron(x).nodes
        elif isinstance(x, BufferedIOBase) or isinstance(x, str):
            x = io.read_swc(x)  # type: ignore
            self.__dict__.update(x.__dict__)
        elif isinstance(x, TreeNeuron):
            self.__dict__.update(x.copy().__dict__)
            # Try to copy every attribute
            for at in self.__dict__:
                try:
                    setattr(self, at, copy.copy(getattr(self, at)))
                except BaseException:
                    logger.warning(f'Unable to deep-copy attribute "{at}"')
        elif isinstance(x, type(None)):
            # This is a essentially an empty neuron
            pass
        else:
            raise utils.ConstructionError(f'Unable to construct TreeNeuron from "{type(x)}"')

        for k, v in metadata.items():
            setattr(self, k, v)

        if not getattr(self, 'id', None):
            self.id = uuid.uuid4()

        self.units = units
        self._current_md5 = self.core_md5

        self._lock = 0

    def __getattr__(self, key):
        """We will use this magic method to calculate some attributes on-demand."""
        # Note that we're mixing @property and __getattr__ which causes problems:
        # if a @property raises an Exception, Python falls back to __getattr__
        # and traceback is lost!

        # Last ditch effort - maybe the base class knows the key?
        return super().__getattr__(key)

    def __truediv__(self, other):
        """Implement division for coordinates (nodes, connectors)."""
        if isinstance(other, (numbers.Number, list, np.ndarray)):
            if isinstance(other, (list, np.ndarray)) and len(other) != 4:
                raise ValueError('Division by list/array requires divisors '
                                 f'for x/y/z and radius - got {len(other)}')

            # If a number, consider this an offset for coordinates
            n = self.copy()
            n.nodes.loc[:, ['x', 'y', 'z', 'radius']] /= other
            if n.has_connectors:
                if isinstance(other, (list, np.ndarray)):
                    n.connectors.loc[:, ['x', 'y', 'z']] /= other[:3]
                else:
                    n.connectors.loc[:, ['x', 'y', 'z']] /= other

            if hasattr(n, 'soma_radius'):
                if isinstance(n.soma_radius, numbers.Number):
                    n.soma_radius /= other

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

            n._clear_temp_attr(exclude=['classify_nodes'])
            return n
        else:
            return NotImplemented

    def __mul__(self, other):
        """Implement multiplication for coordinates (nodes, connectors)."""
        if isinstance(other, (numbers.Number, list, np.ndarray)):
            if isinstance(other, (list, np.ndarray)) and len(other) != 4:
                raise ValueError('Multiplication by list/array requires multipliers'
                                 f' for x/y/z and radius - got {len(other)}')

            # If a number, consider this an offset for coordinates
            n = self.copy()
            n.nodes.loc[:, ['x', 'y', 'z', 'radius']] *= other
            if n.has_connectors:
                if isinstance(other, (list, np.ndarray)):
                    n.connectors.loc[:, ['x', 'y', 'z']] *= other[:3]
                else:
                    n.connectors.loc[:, ['x', 'y', 'z']] *= other

            if hasattr(n, 'soma_radius'):
                if isinstance(n.soma_radius, numbers.Number):
                    n.soma_radius *= other

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

            n._clear_temp_attr(exclude=['classify_nodes'])
            return n
        else:
            return NotImplemented

    def __getstate__(self):
        """Get state (used e.g. for pickling)."""
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}

        # Pickling the graphs actually takes longer than regenerating them
        # from scratch
        if '_graph_nx' in state:
            _ = state.pop('_graph_nx')
        if '_igraph' in state:
            _ = state.pop('_igraph')

        return state

    @property
    @requires_nodes
    def edges(self) -> np.ndarray:
        """Edges between nodes.

        See Also
        --------
        edge_coords
                Same but with x/y/z coordinates instead of node IDs.

        """
        not_root = self.nodes[self.nodes.parent_id >= 0]
        return not_root[['node_id', 'parent_id']].values

    @property
    def edge_coords(self) -> np.ndarray:
        """Coordinates of edges between nodes.

        See Also
        --------
        edges
                Same but with node IDs instead of x/y/z coordinates.

        """
        locs = self.nodes.set_index('node_id')[['x', 'y', 'z']]
        edges = self.edges
        edges_co = np.zeros((edges.shape[0], 2, 3))
        edges_co[:, 0, :] = locs.loc[edges[:, 0]].values
        edges_co[:, 1, :] = locs.loc[edges[:, 1]].values
        return edges_co

    @temp_property
    def igraph(self) -> 'igraph.Graph':
        """iGraph representation of this neuron."""
        # If igraph does not exist, create and return
        if not hasattr(self, '_igraph'):
            # This also sets the attribute
            return self.get_igraph()
        return self._igraph

    @temp_property
    def graph(self) -> nx.DiGraph:
        """Networkx Graph representation of this neuron."""
        # If graph does not exist, create and return
        if not hasattr(self, '_graph_nx'):
            # This also sets the attribute
            return self.get_graph_nx()
        return self._graph_nx

    @temp_property
    def geodesic_matrix(self):
        """Matrix with geodesic (along-the-arbor) distance between nodes."""
        # If matrix has not yet been generated or needs update
        if not hasattr(self, '_geodesic_matrix'):
            # (Re-)generate matrix
            self._geodesic_matrix = graph.geodesic_matrix(self)

        return self._geodesic_matrix

    @property
    def is_stale(self) -> bool:
        """Test if temporary attributes (e.g. ``.graph``) might be outdated."""
        # If we know we are stale, just return True
        if getattr(self, '_stale', False):
            return True
        else:
            # Only check if we believe we are not stale
            self._stale = self._current_md5 != self.core_md5
        return self._stale

    @property
    def core_md5(self) -> str:
        """MD5 of core information for the neuron.

        Generated from ``nodes`` table.

        Returns
        -------
        md5 :   string
                MD5 of node table. ``None`` if no node data.

        """
        if self.has_nodes:
            data = np.ascontiguousarray(self.nodes[['node_id', 'parent_id',
                                                    'x', 'y', 'z']].values)
            if xxhash:
                return xxhash.xxh128(data).hexdigest()
            return hashlib.md5(data).hexdigest()

    @property
    @requires_nodes
    def leafs(self) -> pd.DataFrame:
        """Leaf node table."""
        return self.nodes[self.nodes['type'] == 'end']

    @property
    @requires_nodes
    def ends(self):
        """End node table (same as leafs)."""
        return self.leafs

    @property
    @requires_nodes
    def branch_points(self):
        """Branch node table."""
        return self.nodes[self.nodes['type'] == 'branch']

    @property
    def nodes(self) -> pd.DataFrame:
        """Node table."""
        return self._get_nodes()

    def _get_nodes(self) -> pd.DataFrame:
        # Redefine this function in subclass to change how nodes are retrieved
        return self._nodes

    @nodes.setter
    def nodes(self, v):
        """Validate and set node table."""
        # We are refering to an extra function to facilitate subclassing:
        # Redefine _set_nodes() to not break property
        self._set_nodes(v)

    def _set_nodes(self, v):
        # Redefine this function in subclass to change validation
        self._nodes = utils.validate_table(v,
                                           required=[('node_id', 'rowId', 'node', 'treenode_id', 'PointNo'),
                                                     ('parent_id', 'link', 'parent', 'Parent'),
                                                     ('x', 'X'),
                                                     ('y', 'Y'),
                                                     ('z', 'Z')],
                                           rename=True,
                                           optional={('radius', 'W'): 0},
                                           restrict=False)
        graph.classify_nodes(self)

    @property
    def n_trees(self) -> int:
        """Count number of connected trees in this neuron."""
        return len(self.subtrees)

    @property
    def is_tree(self) -> bool:
        """Whether neuron is a tree.

        Also returns True if neuron consists of multiple separate trees!

        See also
        --------
        networkx.is_forest()
                    Function used to test whether neuron is a tree.
        :attr:`TreeNeuron.cycles`
                    If your neuron is not a tree, this will help you identify
                    cycles.

        """
        return nx.is_forest(self.graph)

    @property
    def subtrees(self) -> List[List[int]]:
        """List of subtrees. Sorted by size as sets of node IDs."""
        return sorted(graph._connected_components(self),
                      key=lambda x: -len(x))

    @property
    def connectors(self) -> pd.DataFrame:
        """Connector table. If none, will return ``None``."""
        return self._get_connectors()

    def _get_connectors(self) -> pd.DataFrame:
        # Redefine this function in subclass to change how nodes are retrieved
        return getattr(self, '_connectors', None)

    @connectors.setter
    def connectors(self, v):
        """Validate and set connector table."""
        # We are refering to an extra function to facilitate subclassing:
        # Redefine _set_connectors() to not break property
        self._set_connectors(v)

    def _set_connectors(self, v):
        # Redefine this function in subclass to change validation
        if isinstance(v, type(None)):
            self._connectors = None
        else:
            self._connectors = utils.validate_table(v,
                                                    required=[('connector_id', 'id'),
                                                              ('node_id', 'rowId', 'node', 'treenode_id'),
                                                              ('x', 'X'),
                                                              ('y', 'Y'),
                                                              ('z', 'Z'),
                                                              ('type', 'relation', 'label', 'prepost')],
                                                    rename=True,
                                                    restrict=False)

    @property
    @requires_nodes
    def cycles(self) -> Optional[List[int]]:
        """Cycles in neuron if any.

        See also
        --------
        networkx.find_cycles()
                    Function used to find cycles.

        """
        try:
            c = nx.find_cycle(self.graph,
                              source=self.nodes[self.nodes.type == 'end'].node_id.values)
            return c
        except nx.exception.NetworkXNoCycle:
            return None
        except BaseException:
            raise

    @property
    def simple(self) -> 'TreeNeuron':
        """Return simple neuron representation.

        Consists only of root, branch points and leafs.

        """
        if not hasattr(self, '_simple'):
            self._simple = self.downsample(float('inf'),
                                           inplace=False)
        return self._simple

    @property
    def soma(self) -> Optional[Union[str, int]]:
        """Search for soma and return node ID(s).

        ``None`` if no soma. You can assign either a function that accepts a
        TreeNeuron as input or a fix value. The default is
        :func:`navis.utils.find_soma`.

        """
        if callable(self._soma):
            soma = self._soma.__call__()  # type: ignore  # say int not callable
        else:
            soma = self._soma

        # Sanity check to make sure that the soma node actually exists
        if isinstance(soma, type(None)):
            # Return immmediately without expensive checks
            return soma
        elif utils.is_iterable(soma):
            if not any(soma):
                soma = None
            elif not any(self.nodes.node_id.isin(soma)):
                logger.warning(f'Soma(s) {soma} not found in node table.')
                soma = None
        else:
            if soma not in self.nodes.node_id.values:
                logger.warning(f'Soma {soma} not found in node table.')
                soma = None

        return soma

    @soma.setter
    def soma(self, value: Union[Callable, int, None]) -> None:
        """Set soma."""
        if hasattr(value, '__call__'):
            self._soma = types.MethodType(value, self)
        elif isinstance(value, type(None)):
            self._soma = None
        else:
            if value in self.nodes.node_id.values:
                self._soma = value
            else:
                raise ValueError('Soma must be function, None or a valid node ID.')

    @property
    @requires_nodes
    def root(self) -> Sequence:
        """Root node(s)."""
        roots = self.nodes[self.nodes.parent_id < 0].node_id.values
        return roots

    @root.setter
    def root(self, value: Union[int, List[int]]) -> None:
        """Reroot neuron to given node."""
        self.reroot(value, inplace=True)

    @property
    def type(self) -> str:
        """Neuron type."""
        return 'navis.TreeNeuron'

    @property
    @requires_nodes
    def n_branches(self) -> Optional[int]:
        """Number of branch points."""
        return self.nodes[self.nodes.type == 'branch'].shape[0]

    @temp_property
    def cable_length(self) -> Union[int, float]:
        """Cable length."""
        if not hasattr(self, '_cable_length'):
            # Simply sum up edge weight of all graph edges
            if config.use_igraph and self.igraph:
                w = self.igraph.es.get_attribute_values('weight')  # type: ignore # doesn't know iGraph
            else:
                w = nx.get_edge_attributes(self.graph, 'weight').values()
            self._cable_length = np.nansum(list(w))
        return self._cable_length

    @property
    def volume(self) -> float:
        """Radius-based volume."""
        if 'radius' not in self.nodes.columns:
            raise ValueError(f'Neuron {self.id} does not have radius information')

        if any(self.nodes.radius < 0):
            logger.warning(f'Neuron {self.id} has negative radii - volume will not be correct.')

        if any(self.nodes.radius.isnull()):
            logger.warning(f'Neuron {self.id} has NaN radii - volume will not be correct.')

        # Get distance for every child -> parent pair
        dist = morpho.mmetrics.parent_dist(self, root_dist=0)
        # Get cylindric volume for each segment
        vols = (self.nodes.radius ** 2) * dist * np.pi
        # Sum up and return
        return vols.sum()

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box (includes connectors)."""
        mn = np.min(self.nodes[['x', 'y', 'z']].values, axis=0)
        mx = np.max(self.nodes[['x', 'y', 'z']].values, axis=0)

        if self.has_connectors:
            cn_mn = np.min(self.connectors[['x', 'y', 'z']].values, axis=0)
            cn_mx = np.max(self.connectors[['x', 'y', 'z']].values, axis=0)

            mn = np.min(np.vstack((mn, cn_mn)), axis=0)
            mx = np.max(np.vstack((mx, cn_mx)), axis=0)

        return np.vstack((mn, mx)).T

    @property
    def sampling_resolution(self) -> float:
        """Average cable length between 2 nodes. """
        return self.cable_length / self.n_nodes

    @temp_property
    def segments(self) -> List[list]:
        """Neuron broken down into linear segments."""
        # If graph does not exist, create and return
        if not hasattr(self, '_segments'):
            # This also sets the attribute
            self._segments = self._get_segments(how='length')
        return self._segments

    @temp_property
    def small_segments(self) -> List[list]:
        """Neuron broken down into small linear segments."""
        # If graph does not exist, create and return
        if not hasattr(self, '_small_segments'):
            # This also sets the attribute
            self._small_segments = self._get_segments(how='break')
        return self._small_segments

    def _get_segments(self,
                      how: Union[Literal['length'],
                                 Literal['break']] = 'length'
                      ) -> List[list]:
        """Generate segments for neuron."""
        if how == 'length':
            return graph._generate_segments(self)
        elif how == 'break':
            return graph._break_segments(self)
        else:
            raise ValueError(f'Unknown method: "{how}"')

    @property
    def n_skeletons(self) -> int:
        """Return number of seperate skeletons in this neuron."""
        return len(self.root)

    def _clear_temp_attr(self, exclude: list = []) -> None:
        """Clear temporary attributes."""
        # Must set checksum before recalculating e.g. node types
        # -> otherwise we run into a recursive loop
        self._current_md5 = self.core_md5
        self._stale = False

        for a in [at for at in self.TEMP_ATTR if at not in exclude]:
            try:
                delattr(self, a)
                logger.debug(f'Neuron {id(self)}: {a} cleared')
            except BaseException:
                logger.debug(f'Neuron {id(self)}: Unable to clear temporary attribute "{a}"')
                pass

        # Remove temporary node values
        # temp_node_cols = ['flow_centrality', 'strahler_index', 'SI', 'bending_flow']
        # self._nodes.drop(columns=temp_node_cols, errors='ignore', inplace=True)

        # Remove soma if it was manually assigned and is not present anymore
        if not callable(self._soma) and not isinstance(self._soma, type(None)):
            if utils.is_iterable(self._soma):
                exists = np.isin(self._soma, self.nodes.node_id.values)
                self._soma = np.asarray(self._soma)[exists]
                if not np.any(self._soma):
                    self._soma = None
            elif self._soma not in self.nodes.node_id.values:
                self.soma = None

        if 'classify_nodes' not in exclude:
            # Reclassify nodes
            graph.classify_nodes(self, inplace=True)

    def copy(self, deepcopy: bool = False) -> 'TreeNeuron':
        """Return a copy of the neuron.

        Parameters
        ----------
        deepcopy :  bool, optional
                    If False, ``.graph`` (NetworkX DiGraph) will be returned
                    as view - changes to nodes/edges can progagate back!
                    ``.igraph`` (iGraph) - if available - will always be
                    deepcopied.

        Returns
        -------
        TreeNeuron

        """
        no_copy = ['_lock']
        # Generate new empty neuron
        x = self.__class__(None)
        # Populate with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        # Copy graphs only if neuron is not stale
        if not self.is_stale:
            if '_graph_nx' in self.__dict__:
                x._graph_nx = self._graph_nx.copy(as_view=deepcopy is not True)
            if '_igraph' in self.__dict__:
                if self._igraph is not None:
                    # This is pretty cheap, so we will always make a deep copy
                    x._igraph = self._igraph.copy()
        else:
            x._clear_temp_attr()

        return x

    def get_graph_nx(self) -> nx.DiGraph:
        """Calculate and return networkX representation of neuron.

        Once calculated stored as ``.graph``. Call function again to update
        graph.

        See Also
        --------
        :func:`navis.neuron2nx`

        """
        self._graph_nx = graph.neuron2nx(self)
        return self._graph_nx

    def get_igraph(self) -> 'igraph.Graph':  # type: ignore
        """Calculate and return iGraph representation of neuron.

        Once calculated stored as ``.igraph``. Call function again to update
        iGraph.

        Important
        ---------
        Returns ``None`` if igraph is not installed!

        See Also
        --------
        :func:`navis.neuron2igraph`

        """
        self._igraph = graph.neuron2igraph(self, raise_not_installed=False)
        return self._igraph

    @overload
    def resample(self, resample_to: int, inplace: Literal[False]) -> 'TreeNeuron': ...

    @overload
    def resample(self, resample_to: int, inplace: Literal[True]) -> None: ...

    def resample(self, resample_to, inplace=False):
        """Resample neuron to given resolution.

        Parameters
        ----------
        resample_to :           int
                                Resolution to which to resample the neuron.
        inplace :               bool, optional
                                If True, operation will be performed on
                                itself. If False, operation is performed on
                                copy which is then returned.

        See Also
        --------
        :func:`~navis.resample_neuron`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        sampling.resample_neuron(x, resample_to, inplace=True)

        # No need to call this as base function does this for us
        # x._clear_temp_attr()

        if not inplace:
            return x
        return None

    @overload
    def downsample(self,
                   factor: float,
                   inplace: Literal[False],
                   **kwargs) -> 'TreeNeuron': ...

    @overload
    def downsample(self,
                   factor: float,
                   inplace: Literal[True],
                   **kwargs) -> None: ...

    def downsample(self, factor=5, inplace=False, **kwargs):
        """Downsample the neuron by given factor.

        Parameters
        ----------
        factor :                int, optional
                                Factor by which to downsample the neurons.
                                Default = 5.
        inplace :               bool, optional
                                If True, operation will be performed on
                                itself. If False, operation is performed on
                                copy which is then returned.
        **kwargs
                                Additional arguments passed to
                                :func:`~navis.downsample_neuron`.

        See Also
        --------
        :func:`~navis.downsample_neuron`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        sampling.downsample_neuron(x, factor, inplace=True, **kwargs)

        # Delete outdated attributes
        x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def reroot(self,
               new_root: Union[int, str],
               inplace: bool = False) -> Optional['TreeNeuron']:
        """Reroot neuron to given node ID or node tag.

        Parameters
        ----------
        new_root :  int | str
                    Either node ID or node tag.
        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~navis.reroot_neuron`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        graph.reroot_neuron(x, new_root, inplace=True)

        # Clear temporary attributes is done by morpho.reroot_neuron()
        # x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def prune_distal_to(self,
                        node: Union[str, int],
                        inplace: bool = False) -> Optional['TreeNeuron']:
        """Cut off nodes distal to given nodes.

        Parameters
        ----------
        node :      node ID | node tag
                    Provide either node ID(s) or a unique tag(s)
        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~navis.cut_neuron`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        node = utils.make_iterable(node, force_type=None)

        for n in node:
            prox = graph.cut_neuron(x, n, ret='proximal')[0]
            # Reinitialise with proximal data
            x.__init__(prox)  # type: ignore  # Cannot access "__init__" directly
            # Remove potential "left over" attributes (happens if we use a copy)
            x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def prune_proximal_to(self,
                          node: Union[str, int],
                          inplace: bool = False) -> Optional['TreeNeuron']:
        """Remove nodes proximal to given node. Reroots neuron to cut node.

        Parameters
        ----------
        node :      node_id | node tag
                    Provide either a node ID or a (unique) tag
        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~navis.cut_neuron`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        node = utils.make_iterable(node, force_type=None)

        for n in node:
            dist = graph.cut_neuron(x, n, ret='distal')[0]
            # Reinitialise with distal data
            x.__init__(dist)  # type: ignore  # Cannot access "__init__" directly
            # Remove potential "left over" attributes (happens if we use a copy)
            x._clear_temp_attr()

        # Clear temporary attributes is done by cut_neuron
        # x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def prune_by_strahler(self,
                          to_prune: Union[int, List[int], slice],
                          inplace: bool = False) -> Optional['TreeNeuron']:
        """Prune neuron based on `Strahler order
        <https://en.wikipedia.org/wiki/Strahler_number>`_.

        Will reroot neuron to soma if possible.

        Parameters
        ----------
        to_prune :  int | list | range | slice
                    Strahler indices to prune. For example:

                    1. ``to_prune=1`` removes all leaf branches
                    2. ``to_prune=[1, 2]`` removes SI 1 and 2
                    3. ``to_prune=range(1, 4)`` removes SI 1, 2 and 3
                    4. ``to_prune=slice(1, -1)`` removes everything but the
                       highest SI
                    5. ``to_prune=slice(-1, None)`` removes only the highest
                       SI

        inplace :   bool, optional
                    If True, operation will be performed on itself. If False,
                    operation is performed on copy which is then returned.

        See Also
        --------
        :func:`~navis.prune_by_strahler`
            This is the base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy()

        morpho.prune_by_strahler(
            x, to_prune=to_prune, reroot_soma=True, inplace=True)

        # No need to call this as morpho.prune_by_strahler does this already
        # self._clear_temp_attr()

        if not inplace:
            return x
        return None

    def prune_twigs(self,
                    size: float,
                    inplace: bool = False,
                    recursive: Union[int, bool, float] = False
                    ) -> Optional['TreeNeuron']:
        """Prune terminal twigs under a given size.

        Parameters
        ----------
        size :          int | float
                        Twigs shorter than this will be pruned.
        inplace :       bool, optional
                        If False, pruning is performed on copy of original neuron
                        which is then returned.
        recursive :     int | bool | "inf", optional
                        If `int` will undergo that many rounds of recursive
                        pruning. Use ``float("inf")`` to prune until no more
                        twigs under the given size are left.

        See Also
        --------
        :func:`~navis.prune_twigs`
            This is the base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy()

        morpho.prune_twigs(x, size=size, inplace=True)

        if not inplace:
            return x
        return None

    def prune_by_longest_neurite(self,
                                 n: int = 1,
                                 reroot_to_soma: bool = False,
                                 inplace: bool = False,
                                 ) -> Optional['TreeNeuron']:
        """Prune neuron down to the longest neurite.

        Parameters
        ----------
        n :                 int, optional
                            Number of longest neurites to preserve.
        reroot_to_soma :    bool, optional
                            If True, will reroot to soma before pruning.
        inplace :           bool, optional
                            If True, operation will be performed on itself.
                            If False, operation is performed on copy which is
                            then returned.

        See Also
        --------
        :func:`~navis.longest_neurite`
            This is the base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy()

        graph.longest_neurite(
            x, n, inplace=True, reroot_to_soma=reroot_to_soma)

        # Clear temporary attributes
        x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def prune_by_volume(self,
                        v: Union[core.Volume,
                                 List[core.Volume],
                                 Dict[str, core.Volume]],
                        mode: Union[Literal['IN'], Literal['OUT']] = 'IN',
                        prevent_fragments: bool = False,
                        inplace: bool = False
                        ) -> Optional['TreeNeuron']:
        """Prune neuron by intersection with given volume(s).

        Parameters
        ----------
        v :                 str | navis.Volume | list of either
                            Volume(s) to check for intersection
        mode :              'IN' | 'OUT', optional
                            If 'IN', parts of the neuron inside the volume are
                            kept.
        prevent_fragments : bool, optional
                            If True, will add nodes to ``subset`` required to
                            keep neuron from fragmenting.
        inplace :           bool, optional
                            If True, operation will be performed on itself. If
                            False, operation is performed on copy which is then
                            returned.

        See Also
        --------
        :func:`~navis.in_volume`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy()

        intersection.in_volume(x, v, inplace=True,
                               prevent_fragments=prevent_fragments,
                               mode=mode)

        # Clear temporary attributes
        # x._clear_temp_attr()

        if not inplace:
            return x
        return None

    def to_swc(self,
               filename: Optional[str] = None,
               **kwargs) -> None:
        """Generate SWC file from this neuron.

        Parameters
        ----------
        filename :      str | None, optional
                        If ``None``, will use "neuron_{id}.swc".
        kwargs
                        Additional arguments passed to :func:`~navis.write_swc`.

        Returns
        -------
        Nothing

        See Also
        --------
        :func:`~navis.write_swc`
                See this function for further details.

        """
        return io.write_swc(self, filename, **kwargs)  # type: ignore  # double import of "io"

    def reload(self,
               inplace: bool = False,
               ) -> Optional['TreeNeuron']:
        """Reload neuron. Must have filepath as ``.origin`` as attribute.

        Returns
        -------
        TreeNeuron
                If ``inplace=False``.

        """
        if not hasattr(self, 'origin'):
            raise AttributeError('To reload TreeNeuron must have `.origin` '
                                 'attribute')

        if self.origin in ('DataFrame', 'string'):
            raise ValueError('Unable to reload TreeNeuron: it appears to have '
                             'been created from string or DataFrame.')

        kwargs = {}
        if hasattr(self, 'soma_label'):
            kwargs['soma_label'] = self.soma_label
        if hasattr(self, 'connector_labels'):
            kwargs['connector_labels'] = self.connector_labels

        x = io.read_swc(self.origin, **kwargs)

        if inplace:
            self.__dict__.update(x.__dict__)
            self._clear_temp_attr()
        else:
            # This makes sure that we keep any additional data stored after
            # this neuron has been loaded
            x2 = self.copy()
            x2.__dict__.update(x.__dict__)
            x2._clear_temp_attr()
            return x


class Dotprops(BaseNeuron):
    """Neuron represented as dotprops.

    Dotprops consist of points with x/y/z coordinates, a tangent vector and an
    alpha value describing the immediate neighbourhood. See References.

    Typically constructed from a point cloud using :func:`navis.make_dotprops`.

    References
    ----------
    Masse N.Y., Cachero S., Ostrovsky A., and Jefferis G.S.X.E. (2012). A mutual
    information approach to automate identification of neuronal clusters in
    Drosophila brain images. Frontiers in Neuroinformatics 6 (00021).
    doi: 10.3389/fninf.2012.00021

    Parameters
    ----------
    points :        numpy array
                    (N, 3) array of x/y/z coordinates.
    k :             int, optional
                    Number of nearest neighbors for tangent vector calculation.
                    This can be ``None`` or ``0`` but then
    vect :          numpy array, optional
                    (N, 3) array of vectors. If not provided will
                    recalculate both ``vect`` and ``alpha`` using ``k``.
    alpha :         numpy array, optional
                    (N, ) array of alpha values. If not provided will
                    recalculate both ``alpha`` and ``vect`` using ``k``.
    units :         str | pint.Units | pint.Quantity
                    Units for coordinates. Defaults to ``None`` (dimensionless).
                    Strings must be parsable by pint: e.g. "nm", "um",
                    "micrometer" or "8 nanometers".
    **metadata
                    Any additional data to attach to neuron.

    """
    connectors: Optional[pd.DataFrame]

    points: np.ndarray
    alpha: np.ndarray
    vect:  np.ndarray
    k: Optional[int]

    soma: Optional[Union[list, np.ndarray]]

    #: Attributes used for neuron summary
    SUMMARY_PROPS = ['type', 'name', 'k', 'units', 'n_points']

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ['name', 'n_points', 'k']

    #: Temporary attributes that need clearing when neuron data changes
    TEMP_ATTR = ['_memory_usage']

    def __init__(self,
                 points: np.ndarray,
                 k: int,
                 vect: Optional[np.ndarray] = None,
                 alpha: Optional[np.ndarray] = None,
                 units: Union[pint.Unit, str] = None,
                 **metadata
                 ):
        """Initialize Dotprops Neuron."""
        super().__init__()

        self.k = k
        self.points = points
        self.alpha = alpha
        self.vect = vect

        self.soma = None

        for k, v in metadata.items():
            setattr(self, k, v)

        if not getattr(self, 'id', None):
            self.id = uuid.uuid4()

        self.units = units

    def __truediv__(self, other):
        """Implement division for coordinates."""
        if isinstance(other, (numbers.Number, list, np.ndarray)):
            # If a number, consider this an offset for coordinates
            n = self.copy()
            _ = np.divide(n.points, other, out=n.points, casting='unsafe')
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] /= other

            # Force recomputing of KDTree
            if hasattr(n, '_tree'):
                delattr(n, '_tree')

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

            return n
        return NotImplemented

    def __mul__(self, other):
        """Implement multiplication for coordinates."""
        if isinstance(other, (numbers.Number, list, np.ndarray)):
            # If a number, consider this an offset for coordinates
            n = self.copy()
            _ = np.multiply(n.points, other, out=n.points, casting='unsafe')
            if n.has_connectors:
                n.connectors.loc[:, ['x', 'y', 'z']] *= other

            # Force recomputing of KDTree
            if hasattr(n, '_tree'):
                delattr(n, '_tree')

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

            return n
        return NotImplemented

    def __getstate__(self):
        """Get state (used e.g. for pickling)."""
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}

        # The KDTree from pykdtree does not like being pickled
        # We will have to remove it which will force it to be regenerated
        # after unpickling
        if '_tree' in state:
            if 'pykdtree' in str(type(state['_tree'])):
                _ = state.pop('_tree')

        return state

    @property
    def alpha(self):
        """Alpha value for tangent vectors."""
        if isinstance(self._alpha, type(None)):
            if isinstance(self.k, type(None)) or (self.k <= 0):
                raise ValueError('Unable to calculate `alpha` for Dotprops not '
                                 'generated using k-nearest-neighbors.')

            self.recalculate_tangents(self.k, inplace=True)
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, type(None)):
            value = np.asarray(value)
            if value.ndim != 1:
                raise ValueError(f'alpha must be (N, ) array, got {value.shape}')
        self._alpha = value

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box (includes connectors)."""
        mn = np.min(self.points, axis=0)
        mx = np.max(self.points, axis=0)

        if self.has_connectors:
            cn_mn = np.min(self.connectors[['x', 'y', 'z']].values, axis=0)
            cn_mx = np.max(self.connectors[['x', 'y', 'z']].values, axis=0)

            mn = np.min(np.vstack((mn, cn_mn)), axis=0)
            mx = np.max(np.vstack((mx, cn_mx)), axis=0)

        return np.vstack((mn, mx)).T

    @property
    def datatables(self) -> List[str]:
        """Names of all DataFrames attached to this neuron."""
        return [k for k, v in self.__dict__.items() if isinstance(v, pd.DataFrame, np.ndarray)]

    @property
    def kdtree(self):
        """KDTree for points."""
        if not getattr(self, '_tree', None):
            self._tree = KDTree(self.points)
        return self._tree

    @property
    def points(self):
        """Center of tangent vectors."""
        return self._points

    @points.setter
    def points(self, value):
        if isinstance(value, type(None)):
            value = np.zeros((0, 3))
        value = np.asarray(value)
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(f'points must be (N, 3) array, got {value.shape}')
        self._points = value
        # Also reset KDtree
        self._tree = None

    @property
    def vect(self):
        """Tangent vectors."""
        if isinstance(self._vect, type(None)):
            self.recalculate_tangents(self.k, inplace=True)
        return self._vect

    @vect.setter
    def vect(self, value):
        if not isinstance(value, type(None)):
            value = np.asarray(value)
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(f'vectors must be (N, 3) array, got {value.shape}')
        self._vect = value

    @property
    def soma(self) -> Optional[int]:
        """Index of soma point.

        ``None`` if no soma. You can assign either a function that accepts a
        Dotprops as input or a fix value. Default is None.
        """
        if callable(self._soma):
            soma = self._soma.__call__()  # type: ignore  # say int not callable
        else:
            soma = self._soma

        # Sanity check to make sure that the soma node actually exists
        if isinstance(soma, type(None)):
            # Return immmediately without expensive checks
            return soma
        elif utils.is_iterable(soma):
            if not any(soma):
                soma = None
            elif any(np.array(soma) < 0) or any(np.array(soma) > self.points.shape[0]):
                logger.warning(f'Soma(s) {soma} not found in points.')
                soma = None
        else:
            if 0 < soma < self.points.shape[0]:
                logger.warning(f'Soma {soma} not found in node table.')
                soma = None

        return soma

    @soma.setter
    def soma(self, value: Union[Callable, int, None]) -> None:
        """Set soma."""
        if hasattr(value, '__call__'):
            self._soma = types.MethodType(value, self)
        elif isinstance(value, type(None)):
            self._soma = None
        else:
            if 0 < value < self.points.shape[0]:
                self._soma = value
            else:
                raise ValueError('Soma must be function, None or a valid node index.')

    @property
    def type(self) -> str:
        """Neuron type."""
        return 'navis.Dotprops'

    def dist_dots(self,
                  other: 'Dotprops',
                  alpha: bool = False,
                  **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Query this Dotprops against another.

        This function is mainly for ``navis.nblast``.

        Parameters
        ----------
        other :     Dotprops
        alpha :     bool
                    If True, will also return the product of the product
                    of the alpha values of matched points.
        kwargs
                    Keyword arguments are passed to the KDTree's ``query()``
                    method. Note that we are using ``pykdtree.kdtree.KDTree``
                    if available and fall back to ``scipy.spatial.cKDTree`` if
                    pykdtree is not installed.

        Returns
        -------
        dist :          np.ndarray
                        For each point in ``self``, the distance to the closest
                        point in ``other``.
        dotprops :      np.ndarray
                        Dotproduct of each pair of closest points between
                        ``self`` and ``other``.
        alpha_prod :    np.ndarray
                        Dotproduct of each pair of closest points between
                        ``self`` and ``other``. Only returned if ``alpha=True``.

        """
        if not isinstance(other, Dotprops):
            raise TypeError(f'Expected Dotprops, got "{type(other)}"')

        # If we are using pykdtree we need to make sure that self.points is
        # of the same dtype as other.points - not a problem with scipy but
        # it the overhead is typically only a few micro seconds
        points = self.points.astype(other.points.dtype)

        fast_dists, fast_idxs = other.kdtree.query(points, **kwargs)
        fast_dotprods = np.abs((self.vect * other.vect[fast_idxs]).sum(axis=1))

        if not alpha:
            return fast_dists, fast_dotprods
        else:
            fast_alpha = self.alpha * other.alpha[fast_idxs]
            return fast_dists, fast_dotprods, fast_alpha

    def downsample(self, factor=5, inplace=False, **kwargs):
        """Downsample the neuron by given factor.

        Parameters
        ----------
        factor :                int, optional
                                Factor by which to downsample the neurons.
                                Default = 5.
        inplace :               bool, optional
                                If True, operation will be performed on
                                itself. If False, operation is performed on
                                copy which is then returned.
        **kwargs
                                Additional arguments passed to
                                :func:`~navis.downsample_neuron`.

        See Also
        --------
        :func:`~navis.downsample_neuron`
            Base function. See for details and examples.

        """
        if inplace:
            x = self
        else:
            x = self.copy()

        sampling.downsample_neuron(x, factor, inplace=True, **kwargs)

        if not inplace:
            return x
        return None

    def copy(self) -> 'Dotprops':
        """Return a copy of the dotprops.

        Returns
        -------
        Dotprops

        """
        # Don't copy the KDtree - when using pykdtree, copy.copy throws an
        # error and the construction is super fast anyway
        no_copy = ['_lock', '_tree']
        # Generate new empty neuron - note we pass vect and alpha as True to
        # prevent calculation on initialization
        x = self.__class__(points=np.zeros((0, 3)), k=1,
                           vect=np.zeros((0, 3)), alpha=np.zeros(0))
        # Populate with this neuron's data
        x.__dict__.update({k: copy.copy(v) for k, v in self.__dict__.items() if k not in no_copy})

        return x

    def recalculate_tangents(self, k: int, inplace=False) -> None:
        """Recalculate tangent vectors and alpha with a new ``k``.

        Parameters
        ----------
        k :         int
                    Number of nearest neighbours to use for tangent vector
                    calculation.
        inplace :   bool
                    If False, will return a copy and leave the original data
                    unmodified.

        Returns
        -------
        Dotprops
                    Only if ``inplace=False``.

        """
        if not inplace:
            x = self.copy()
        else:
            x = self

        if isinstance(k, type(None)) or k < 1:
            raise ValueError(f'`k` must be integer >= 1, got "{k}"')

        # Checks and balances
        n_points = x.points.shape[0]
        if n_points < k:
            raise ValueError(f"Too few points ({n_points}) to calculate {k} "
                             "nearest-neighbors")

        # Create the KDTree and get the k-nearest neighbors for each point
        dist, ix = self.kdtree.query(x.points, k=k)

        # Get points: array of (N, k, 3)
        pt = x.points[ix]

        # Generate centers for each cloud of k nearest neighbors
        centers = np.mean(pt, axis=1)

        # Generate vector from center
        cpt = pt - centers.reshape((pt.shape[0], 1, 3))

        # Get inertia (N, 3, 3)
        inertia = cpt.transpose((0, 2, 1)) @ cpt

        # Extract vector and alpha
        u, s, vh = np.linalg.svd(inertia)
        x.vect = vh[:, 0, :]
        x.alpha = (s[:, 0] - s[:, 1]) / np.sum(s, axis=1)

        # Keep track of k
        x.k = k

        if not inplace:
            return x

    def to_skeleton(self, scale_vec: float = 1) -> TreeNeuron:
        """Turn dotprops into a skeleton.

        Note that only minimal meta data is carried over.

        Parameters
        ----------
        scale_vec :     float
                        Factor by which to scale each tangent vector when
                        generating the line segments.

        Returns
        -------
        TreeNeuron

        """
        # Prepare segments - this is based on nat:::plot3d.dotprops
        halfvect = self.vect / 2 * scale_vec
        starts = self.points - halfvect
        ends = self.points + halfvect

        # Interweave starts and ends
        segs = np.zeros((starts.shape[0] * 2, 3))
        segs[::2] = starts
        segs[1::2] = ends

        # Generate node table
        nodes = pd.DataFrame(segs, columns=['x', 'y', 'z'])
        nodes['node_id'] = nodes.index
        nodes['parent_id'] = -1
        nodes.loc[1::2, 'parent_id'] = nodes.index.values[::2]

        # Produce a minimal TreeNeuron
        tn = TreeNeuron(nodes, units=self.units, id=self.id)

        # Carry over the label
        if getattr(self, '_label', None):
            tn._label = self._label

        # Add some other relevant attributes directly
        if self.has_connectors:
            tn._connectors = self._connectors
        tn._soma = self._soma

        return tn
