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
import hashlib
import numbers
import pint
import uuid
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from io import StringIO

from typing import Union, List, Optional, Any
from typing_extensions import Literal

from .. import utils, config, core

try:
    import xxhash
except ImportError:
    xxhash = None

__all__ = ['Neuron']

# Set up logging
logger = config.get_logger(__name__)

# This is to prevent pint to throw a warning about numpy integration
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


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
        return core.TreeNeuron(x, **metadata)
    except utils.ConstructionError:
        try:
            return core.MeshNeuron(x, **metadata)
        except utils.ConstructionError:
            pass
        except BaseException:
            raise
    except BaseException:
        raise

    raise utils.ConstructionError(f'Unable to construct neuron from "{type(x)}"')


class BaseNeuron:
    """Base class for all neurons."""

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

    #: Temporary attributes that need clearing when neuron data changes
    TEMP_ATTR = []

    #: Core data table(s) used to calculate hash
    CORE_DATA = []

    def __init__(self, **kwargs):
        # Set a random ID -> may be replaced later
        self.id = uuid.uuid4()

        # Make a copy of summary and temp props so that if we register
        # additional properties we don't change this for every single neuron
        self.SUMMARY_PROPS = self.SUMMARY_PROPS.copy()
        self.TEMP_ATTR = self.TEMP_ATTR.copy()

        self._lock = 0
        for k, v in kwargs.items():
            self._register_attr(name=k, value=v)

        # Base neurons has no data
        self._current_md5 = None

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

    def __deepcopy__(self, memo):
        result = self.copy(deepcopy=True)
        memo[id(self)] = result
        return result

    def __eq__(self, other):
        """Implement neuron comparison."""
        if isinstance(other, BaseNeuron):
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
        if isinstance(other, BaseNeuron):
            return core.NeuronList([self, other])
        else:
            return NotImplemented

    def __imul__(self, other):
        """Multiplication with assignment (*=)."""
        return self.__mul__(other, copy=False)

    def __itruediv__(self, other):
        """Division with assignment (/=)."""
        return self.__truediv__(other, copy=False)

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

    def _clear_temp_attr(self, exclude: list = []) -> None:
        """Clear temporary attributes."""
        # Must set checksum before recalculating e.g. node types
        # -> otherwise we run into a recursive loop
        self._current_md5 = self.core_md5
        self._stale = False

        for a in [at for at in self.TEMP_ATTR if at not in exclude]:
            try:
                delattr(self, a)
                logger.debug(f'Neuron {self.id} {hex(id(self))}: attribute {a} cleared')
            except AttributeError:
                logger.debug(f'Neuron {self.id} at {hex(id(self))}: Unable to clear temporary attribute "{a}"')
            except BaseException:
                raise

    def _register_attr(self, name, value, summary=True, temporary=False):
        """Set and register attribute.

        Use this if you want an attribute to be used for the summary or cleared
        when temporary attributes are cleared.
        """
        setattr(self, name, value)

        # If this is an easy to summarize attribute, add to summary
        if summary and name not in self.SUMMARY_PROPS:
            if isinstance(value, (numbers.Number, str, bool, np.bool_, type(None))):
                self.SUMMARY_PROPS.append(name)
            else:
                logger.error(f'Attribute "{name}" of type "{type(value)}" '
                             'can not be added to summary')

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
    def core_md5(self) -> str:
        """MD5 checksum of core data.

        Generated from ``.CORE_DATA`` properties.

        Returns
        -------
        md5 :   string
                MD5 checksum of core data. ``None`` if no core data.

        """
        hash = ''
        for prop in self.CORE_DATA:
            cols = None
            # See if we need to parse props into property and columns
            # e.g. "nodes:node_id,parent_id,x,y,z"
            if ':' in prop:
                prop, cols = prop.split(':')
                cols = cols.split(',')

            if hasattr(self, prop):
                data = getattr(self, prop)
                if isinstance(data, pd.DataFrame):
                    if cols:
                        data = data[cols]
                    data = data.values

                data = np.ascontiguousarray(data)

                if xxhash:
                    hash += xxhash.xxh128(data).hexdigest()
                else:
                    hash += hashlib.md5(data).hexdigest()

        return hash if hash else None

    @property
    def datatables(self) -> List[str]:
        """Names of all DataFrames attached to this neuron."""
        return [k for k, v in self.__dict__.items() if isinstance(v, pd.DataFrame)]

    @property
    def extents(self) -> np.ndarray:
        """Extents of neuron in x/y/z direction (includes connectors)."""
        if not hasattr(self, 'bbox'):
            raise ValueError('Neuron must implement `.bbox` (bounding box) '
                             'property to calculate extents.')
        bbox = self.bbox
        return bbox[:, 1] - bbox[:, 0]

    @property
    def id(self) -> Any:
        """ID of the neuron.

        Must be hashable. If not set, will assign a random unique identifier.
        Can be indexed by using the ``NeuronList.idx[]`` locator.
        """
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
        """Table with presynapses (filtered from connectors table).

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
        """Table with postsynapses (filtered from connectors table).

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
    def units(self) -> Union[numbers.Number, np.ndarray]:
        """Units for coordinate space."""
        # Note that we are regenerating the pint.Quantity from the string
        # That is to avoid problems with pickling e.g. when using multiprocessing
        unit_str = getattr(self, '_unit_str', None)

        if utils.is_iterable(unit_str):
            values = [config.ureg(u) for u in unit_str]
            conv = [v.to(values[0]).magnitude for v in values]
            return config.ureg.Quantity(np.array(conv), values[0].units)
        else:
            return config.ureg(unit_str)

    @property
    def units_xyz(self) -> np.ndarray:
        """Units for coordinate space. Always returns x/y/z array."""
        units = self.units

        if not utils.is_iterable(units):
            units = config.ureg.Quantity([units.magnitude] * 3, units.units)

        return units

    @units.setter
    def units(self, units: Union[pint.Unit, pint.Quantity, str, None]):
        # Note that we are storing the string, not the actual pint.Quantity
        # That is to avoid problems with pickling e.g. when using multiprocessing

        # Do NOT remove the is_iterable condition - otherwise we might
        # accidentally strip the units from a pint Quantity vector
        if not utils.is_iterable(units):
            units = utils.make_iterable(units)

        if len(units) not in [1, 3]:
            raise ValueError('Must provide either a single unit or one for '
                             'for x, y and z dimension.')

        # Make sure we actually have valid unit(s)
        unit_str = []
        for v in units:
            if isinstance(v, str):
                # This makes sure we have meters (i.e. nm, um, etc) because
                # "microns", for example, produces odd behaviour like
                # "millimicrons" on division
                v = v.replace('microns', 'um').replace('micron', 'um')
                unit_str.append(str(v))
            elif isinstance(v, (pint.Unit, pint.Quantity)):
                unit_str.append(str(v))
            elif isinstance(v, type(None)):
                unit_str.append(None)
            elif isinstance(v, numbers.Number):
                unit_str.append(str(config.ureg(f'{v} dimensionless')))
            else:
                raise TypeError(f'Expect str or pint Unit/Quantity, got "{type(v)}"')

        # Some clean-up
        if len(set(unit_str)) == 1:
            unit_str = unit_str[0]
        else:
            # Check if all base units (e.g. "microns") are the same
            unique_units = set([str(config.ureg(u).units) for u in unit_str])
            if len(unique_units) != 1:
                raise ValueError('Non-isometric units must share the same base,'
                                 f' got: {", ".join(unique_units)}')
            unit_str = tuple(unit_str)

        self._unit_str = unit_str

    @property
    def is_isometric(self):
        """Test if neuron is isometric."""
        u = self.units
        if utils.is_iterable(u) and len(set(u)) > 1:
            return False
        return True

    @property
    def is_stale(self) -> bool:
        """Test if temporary attributes might be outdated."""
        # If we know we are stale, just return True
        if getattr(self, '_stale', False):
            return True
        else:
            # Only check if we believe we are not stale
            self._stale = self._current_md5 != self.core_md5
        return self._stale

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

        n = self.copy() if not inplace else self

        # Catch pint's UnitStrippedWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Get factor by which we have to multiply to get to target units
            conv = n.units.to(to).magnitude
            # Multiply by conversion factor
            n *= conv

        n._clear_temp_attr(exclude=['classify_nodes'])

        return n

    def copy(self, deepcopy=False) -> 'BaseNeuron':
        """Return a copy of the neuron."""
        copy_fn = copy.deepcopy if deepcopy else copy.copy
        # Attributes not to copy
        no_copy = ['_lock']
        # Generate new empty neuron
        x = self.__class__()
        # Override with this neuron's data
        x.__dict__.update({k: copy_fn(v) for k, v in self.__dict__.items() if k not in no_copy})

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

    def map_units(self,
                  units: Union[pint.Unit, str],
                  on_error: Union[Literal['raise'],
                                  Literal['ignore']] = 'raise') -> Union[int, float]:
        """Convert units to match neuron space.

        Only works if neuron's ``.units`` is isometric and not dimensionless.

        Parameters
        ----------
        units :     number | str | pint.Quantity | pint.Units
                    The units to convert to neuron units. Simple numbers are just
                    passed through.
        on_error :  "raise" | "ignore"
                    What to do if an error occurs (e.g. because `neuron` does not
                    have units specified). If "ignore" will simply return ``units``
                    unchanged.

        See Also
        --------
        :func:`navis.core.to_neuron_space`
                    The base function for this method.

        Examples
        --------
        >>> import navis
        >>> # Example neurons are in 8x8x8nm voxel space
        >>> n = navis.example_neurons(1)
        >>> n.map_units('1 nanometer')
        0.125
        >>> # Numbers are passed-through
        >>> n.map_units(1)
        1
        >>> # For neuronlists
        >>> nl = navis.example_neurons(3)
        >>> nl.map_units('1 nanometer')
        [0.125, 0.125, 0.125]

        """
        return core.core_utils.to_neuron_space(units, neuron=self,
                                               on_error=on_error)

    def memory_usage(self, deep=False, estimate=False):
        """Return estimated memory usage of this neuron.

        Works by going over attached data (numpy arrays and pandas DataFrames
        such as vertices, nodes, etc) and summing up their size in memory.

        Parameters
        ----------
        deep :      bool
                    Passed to pandas DataFrames. If True will also inspect
                    memory footprint of `object` dtypes.
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
