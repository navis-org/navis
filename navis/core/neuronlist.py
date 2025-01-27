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

from concurrent.futures import ThreadPoolExecutor
import os
import random
import re
import types
import uuid

import networkx as nx

import numpy as np
import pandas as pd

from pathlib import Path
from typing import (Sequence, Union, Iterable, List,
                    Optional, Callable, Iterator)

from .. import utils, config, core

__all__ = ['NeuronList']

# Set up logging
logger = config.get_logger(__name__)

# Max number of neurons in NeuronList to search for attributes
# This is to avoid searching through a huge list of neurons
# and freezing the interpreter
MAX_SEARCH = 100


class NeuronList:
    """Collection of neurons.

    Gives quick access to neurons' attributes and functions.

    Parameters
    ----------
    x :                 list | array | TreeNeuron | MeshNeuron | Dotprops | NeuronList
                        Data to construct neuronlist from. Can be either:

                        1. Tree/MeshNeuron(s) or Dotprops
                        2. NeuronList(s)
                        3. Anything that constructs a Tree/MeshNeuron
                        4. List of the above

    make_copy :         bool, optional
                        If True, Neurons are deepcopied before being
                        assigned to the NeuronList.
    make_using :        function | class, optional
                        Function or class used to construct neurons from
                        elements in `x` if they aren't already neurons.
                        By default, will use `navis.Neuron` to try to infer
                        what kind of neuron can be constructed.
    parallel :          bool
                        If True, will use parallel threads when initialising the
                        NeuronList. Should be slightly up to a lot faster
                        depending on the numbers of cores and the input data.
    n_cores :           int
                        Number of cores to use for when `parallel=True`.
                        Defaults to half the available cores.
    **kwargs
                        Will be passed to constructor of Tree/MeshNeuron (see
                        `make_using`).

    """

    neurons: List['core.NeuronObject']

    cable_length: Sequence[float]

    soma: Sequence[int]
    root: Sequence[int]

    graph: 'nx.DiGraph'
    igraph: 'igraph.Graph'  # type: ignore  # doesn't know iGraph

    def __init__(self,
                 x: Union[Iterable[Union[core.BaseNeuron,
                                         'NeuronList',
                                         pd.DataFrame]],
                          'NeuronList',
                          core.BaseNeuron,
                          pd.DataFrame],
                 make_copy: bool = False,
                 make_using: Optional[type] = None,
                 parallel: bool = False,
                 n_cores: int = os.cpu_count() // 2,
                 **kwargs):
        # If below parameter is True, most calculations will be parallelized
        # which speeds them up quite a bit. Unfortunately, this uses A TON of
        # memory - for large lists this might make your system run out of
        # memory. In these cases, leave this property at False
        self.parallel = parallel
        self.n_cores = n_cores

        # Determines if subsetting this NeuronList will copy the neurons
        self.copy_on_subset: bool = False

        if isinstance(x, NeuronList):
            # We can't simply say self.neurons = x.neurons b/c that way
            # changes in the list would backpropagate
            self.neurons = [n for n in x.neurons]
        elif utils.is_iterable(x):
            # If x is a list of mixed objects we need to unpack/flatten that
            # E.g. x = [NeuronList, NeuronList, core.TreeNeuron]
            # We need to make sure the order is retained though (important for
            # e.g. plotting)
            self.neurons = []
            for n in x:
                # Unpack neuronlists
                if isinstance(n, NeuronList):
                    self.neurons += n.neurons
                # Everything else is just appended - will throw error later
                else:
                    self.neurons.append(n)
        elif isinstance(x, type(None)):
            # Empty Neuronlist
            self.neurons = []
        else:
            # Any other datatype will simply be assumed to be accepted by
            # core.Neuron() - if not this will throw an error
            self.neurons = [x]  # type: ignore

        # Now convert and/or make copies if necessary
        to_convert = []
        for i, n in enumerate(self.neurons):
            if not isinstance(n, core.BaseNeuron) or make_copy is True:
                # The `i` keeps track of the original index so that after
                # conversion to Neurons, the objects will occupy the same
                # position
                to_convert.append((n, i))

        if to_convert:
            if not make_using:
                make_using = core.Neuron
            elif not isinstance(make_using, type) and not callable(make_using):
                make_using = make_using.__class__

            if self.parallel:
                with ThreadPoolExecutor(max_workers=self.n_cores) as e:
                    futures = e.map(lambda x: make_using(x, **kwargs),
                                    [n[0] for n in to_convert])

                    converted = [n for n in config.tqdm(futures,
                                                        total=len(to_convert),
                                                        desc='Make nrn',
                                                        disable=config.pbar_hide,
                                                        leave=config.pbar_leave)]

                    for i, c in enumerate(to_convert):
                        self.neurons[c[1]] = converted[i]

            else:
                for n in config.tqdm(to_convert, desc='Make nrn',
                                     disable=config.pbar_hide or len(to_convert) == 1,
                                     leave=config.pbar_leave):
                    self.neurons[n[1]] = make_using(n[0], **kwargs)

        # Add ID-based indexer
        self.idx = _IdIndexer(self)

    @property
    def neurons(self):
        """Neurons contained in this NeuronList."""
        return self.__dict__.get('neurons', [])

    @property
    def is_mixed(self):
        """Return True if contains more than one type of neuron."""
        return len(self.types) > 1

    @property
    def is_degenerated(self):
        """Return True if contains neurons with non-unique IDs."""
        return len(set(self.id)) < len(self.neurons)

    @property
    def types(self):
        """Return neuron types present in this list."""
        return tuple(set([type(n) for n in self.neurons]))

    @property
    def shape(self):
        """Shape of NeuronList (N, )."""
        return (self.__len__(),)

    @property
    def bbox(self):
        """Bounding box across all neurons in the list."""
        if self.empty:
            raise ValueError('No bounding box - NeuronList is empty.')

        bboxes = np.hstack([n.bbox for n in self.neurons])
        mn = np.min(bboxes, axis=1)
        mx = np.max(bboxes, axis=1)
        return np.vstack((mn, mx)).T

    @property
    def empty(self):
        """Return True if NeuronList is empty."""
        return len(self.neurons) == 0

    def __reprframe__(self):
        """Return truncated DataFrame for self representation."""
        if self.empty:
            return pd.DataFrame([])
        elif len(self) < 5:
            return self.summary()
        else:
            nl = self[:3] + self[-3:]
            s = nl.summary()
            # Fix index
            s.index = np.append(s.index[:3], np.arange(len(self)-3, len(self)))
            return s

    def __reprheader__(self, html=False):
        """Generate header for representation."""
        if len(self) <= 2000:
            size = utils.sizeof_fmt(self.memory_usage(deep=False, estimate=True))
            head = f'{type(self)} containing {len(self)} neurons ({size})'
        else:
            # For larger lists, extrapolate from sampling 10% of the list
            size = utils.sizeof_fmt(self.memory_usage(deep=False,
                                                      sample=True,
                                                      estimate=True))
            head = f'{type(self)} containing {len(self)} neurons (est. {size})'

        if html:
            head = head.replace('<', '&lt;').replace('>', '&gt;')

        return head

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = self.__reprheader__(html=False)
        if not self.empty:
            with pd.option_context("display.max_rows", 4,
                                   "display.show_dimensions", False):
                string += f'\n{str(self.__reprframe__())}'

        return string

    def _repr_html_(self):
        string = self.__reprheader__(html=True)
        if not self.empty:
            with pd.option_context("display.max_rows", 4,
                                   "display.show_dimensions", False):
                string += self.__reprframe__()._repr_html_()
        return string

    def __iter__(self) -> Iterator['core.NeuronObject']:
        """Iterator instanciates a new class every time it is called.
        This allows the use of nested loops on the same NeuronList object.
        """
        class prange_iter(Iterator['core.NeuronObject']):
            def __init__(self, neurons, start):
                self.iter = start
                self.neurons = neurons

            def __next__(self) -> 'core.NeuronObject':
                if self.iter >= len(self.neurons):
                    raise StopIteration
                to_return = self.neurons[self.iter]
                self.iter += 1
                return to_return

        return prange_iter(self.neurons, 0)

    def __len__(self):
        """Number of neurons in this list."""
        return len(self.neurons)

    def __dir__(self):
        """Custom __dir__ to add some parameters that we want to make searchable."""
        add_attr = set.union(*[set(dir(n)) for n in self.neurons[:MAX_SEARCH]])

        return list(set(super().__dir__() + list(add_attr)))

    def __getattr__(self, key):
        if self.empty:
            raise AttributeError(f'Neuronlist is empty - "{key}" not found')
        # Dynamically check if the requested attribute/function exists in
        # all neurons
        values = [getattr(n, key, NotImplemented) for n in self.neurons]
        is_method = [isinstance(v, types.MethodType) for v in values]
        # is_none = [isinstance(v, type(None)) for v in values]
        is_frame = [isinstance(v, pd.DataFrame) for v in values]
        is_quantity = [isinstance(v, config.ureg.Quantity) for v in values]

        # First check if there is any reason why we can't collect this
        # attribute across all neurons
        if all([isinstance(v, type(NotImplemented)) for v in values]):
            raise AttributeError(f'Attribute "{key}" not found in '
                                 'NeuronList or its neurons')
        elif any([isinstance(v, type(NotImplemented)) for v in values]):
            raise AttributeError(f'Attribute or function "{key}" missing '
                                 'for some neurons')
        elif len(set(is_method)) > 1:
            raise TypeError('Found both methods and attributes with name '
                            f'"{key}" among neurons.')
        # Concatenate if dealing with DataFrame
        elif not all(is_method):
            if any(is_frame):
                df = pd.concat([v for v in values if isinstance(v, pd.DataFrame)],
                               axis=0,
                               ignore_index=True,
                               join='outer',
                               sort=True)

                # For each row label which neuron (id) it belongs to
                df['neuron'] = None
                ix = 0
                for k, v in enumerate(values):
                    if isinstance(v, pd.DataFrame):
                        df.iloc[ix:ix + v.shape[0],
                                df.columns.get_loc('neuron')] = self.neurons[k].id
                        ix += v.shape[0]
                return df
            elif all(is_quantity):
                # See if units are all compatible
                is_compatible = [values[0].is_compatible_with(v) for v in values]
                if all(is_compatible):
                    # Convert all to the same units
                    conv = [v.to(values[0]).magnitude for v in values]
                    # Return pint array
                    return config.ureg.Quantity(np.array(conv), values[0].units)
                else:
                    logger.warning(f'"{key}" contains incompatible units. '
                                   'Returning unitless values.')
                    return np.array([v.magnitude for v in values])
            elif any(is_quantity):
                logger.warning(f'"{key}" contains data with and without '
                               'units. Removing units.')
                return np.array([getattr(v, 'magnitude', v) for v in values])
            else:
                # If the result would be a ragged array specify dtype as object
                # This avoids a depcrecation warning and future issues
                dtype = None
                if any([utils.is_iterable(v) for v in values]):
                    if not all([utils.is_iterable(v) for v in values]):
                        dtype = object
                    elif len(set([len(v) for v in values])) > 1:
                        dtype = object
                return np.array(values, dtype=dtype)
        # If everything is a method
        else:
            # To avoid confusion we will not allow calling of magic methods
            # via the NeuronProcessor as those are generally expected to
            # be methods of the NeuronList itself
            if key.startswith('__') and key.endswith('__'):
                raise AttributeError(f"'NeuronList' object has no attribute '{key}'")

            # Delayed import to avoid circular import
            from .core_utils import NeuronProcessor

            # Return function but wrap it in a function that will show
            # a progress bar. Note that we do not use parallel processing by
            # default to avoid errors with `inplace=True`
            return NeuronProcessor(self,
                                   values,
                                   parallel=False,
                                   desc=key)

    def __setattr__(self, key, value):
        # We have cater for the situation when we want to replace the whole
        # dictionary - e.g. when unpickling (see __setstate__)
        # Below code for setting the dictionary looks complicated and
        # unnecessary but is really complicated and VERY necessary
        if key == '__dict__':
            if not isinstance(value, dict):
                raise TypeError(f'__dict__ must be dict, got {type(value)}')
            self.__dict__.clear()
            for k, v in value.items():
                self.__dict__[k] = v
            return

        # Check if this attribute exists in the neurons
        if any([hasattr(n, key) for n in self.neurons]):
            logger.warning('It looks like you are trying to add a neuron '
                           'attribute to a NeuronList. Setting the attribute '
                           f'"{key}" on the NeuronList will not propagated to '
                           'the neurons it contains! To set neuron attributes '
                           'use the `NeuronList.set_neuron_attributes()` method.')

        self.__dict__[key] = value

    def __getstate__(self):
        """Get state (used e.g. for pickling)."""
        # We have to implement this to make sure that we don't accidentally
        # call __getstate__ of each neuron via the NeuronProcessor
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}
        return state

    def __setstate__(self, d):
        """Set state (used e.g. for unpickling)."""
        # We have to implement this to make sure that we don't accidentally
        # call __setstate__ of each neuron via the NeuronProcessor
        self.__dict__ = d

    def __contains__(self, x):
        return x in self.neurons

    def __copy__(self):
        return self.copy(deepcopy=False)

    def __deepcopy__(self):
        return self.copy(deepcopy=True)

    def __getitem__(self, key):
        if utils.is_iterable(key):
            if all([isinstance(k, (bool, np.bool_)) for k in key]):
                if len(key) != len(self.neurons):
                    raise IndexError('boolean index did not match indexed '
                                     f'NeuronList; dimension is {len(self.neurons)} '
                                     'but corresponding boolean dimension is '
                                     f'{len(key)}')
                subset = [n for i, n in enumerate(self.neurons) if key[i]]
            else:
                subset = [self[i] for i in key]
        elif isinstance(key, str):
            subset = [n for n in self.neurons if re.fullmatch(key, getattr(n, 'name', ''))]

            # For indexing by name, we expect a match
            if not subset:
                raise AttributeError('NeuronList does not contain neuron(s) '
                                     f'with name: "{key}"')

        elif isinstance(key, (int, np.integer, slice)):
            subset = self.neurons[key]
        else:
            raise NotImplementedError(f'Indexing NeuronList by {type(key)} not implemented')

        if isinstance(subset, core.BaseNeuron):
            return subset

        # Make sure we unpack neurons
        subset = utils.unpack_neurons(subset)

        return self.__class__(subset, make_copy=self.copy_on_subset)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if not utils.is_iterable(value):
                for n in self.neurons:
                    setattr(n, key, value)
            elif len(value) == len(self.neurons):
                for n, v in zip(self.neurons, value):
                    setattr(n, key, v)
            else:
                raise ValueError('Length of values does not match number of '
                                 'neurons in NeuronList.')
        else:
            msg = ('Itemsetter can only be used to set attributes of the '
                   'neurons contained in the NeuronList. For example:\n'
                   '  >>> nl = navis.example_neurons(3)\n'
                   '  >>> nl["propertyA"] = 1\n'
                   '  >>> nl[0].propertyA\n'
                   '  1\n'
                   '  >>> nl["propertyB"] = ["a", "b", "c"]\n'
                   '  >>> nl[2].propertyB\n'
                   '  "c"')
            raise NotImplementedError(msg)

    def __missing__(self, key):
        raise AttributeError('No neuron matching the search criteria.')

    def __add__(self, to_add):
        """Implement addition."""
        if isinstance(to_add, core.BaseNeuron):
            return self.__class__(self.neurons + [to_add],
                                  make_copy=self.copy_on_subset)
        elif isinstance(to_add, NeuronList):
            return self.__class__(self.neurons + to_add.neurons,
                                  make_copy=self.copy_on_subset)
        elif utils.is_iterable(to_add):
            if False not in [isinstance(n, core.BaseNeuron) for n in to_add]:
                return self.__class__(self.neurons + list(to_add),
                                      make_copy=self.copy_on_subset)
            else:
                return self.__class__(self.neurons + [core.BaseNeuron[n] for n in to_add],
                                      make_copy=self.copy_on_subset)
        else:
            return NotImplemented

    def __eq__(self, other):
        """Implement equality."""
        if isinstance(other, NeuronList):
            if len(self) != len(other):
                return False
            else:
                return all([n1 == n2 for n1, n2 in zip(self, other)])
        else:
            return NotImplemented

    def __sub__(self, to_sub):
        """Implement substraction."""
        if isinstance(to_sub, core.BaseNeuron):
            return self.__class__([n for n in self.neurons if n != to_sub],
                                  make_copy=self.copy_on_subset)
        elif isinstance(to_sub, NeuronList):
            return self.__class__([n for n in self.neurons if n not in to_sub],
                                  make_copy=self.copy_on_subset)
        else:
            return NotImplemented

    def __truediv__(self, other):
        """Implements division for coordinates (nodes, connectors)."""
        return self.__class__([n / other for n in config.tqdm(self.neurons,
                                                              desc='Dividing',
                                                              disable=config.pbar_hide,
                                                              leave=False)])


    def __mul__(self, other):
        """Implement multiplication for coordinates (nodes, connectors)."""
        return self.__class__([n * other for n in config.tqdm(self.neurons,
                                                              desc='Multiplying',
                                                              disable=config.pbar_hide,
                                                              leave=False)])

    def __and__(self, other):
        """Implement bitwise AND using the & operator."""
        if isinstance(other, core.BaseNeuron):
            return self.__class__([n for n in self.neurons if n == other],
                                  make_copy=self.copy_on_subset)
        elif isinstance(other, NeuronList):
            return self.__class__([n for n in self.neurons if n in other],
                                  make_copy=self.copy_on_subset)
        else:
            return NotImplemented

    def __or__(self, other):
        """Implement bitwise OR using the | operator."""
        if isinstance(other, core.BaseNeuron):
            neurons = self.neurons
            if not any(n == other for n in neurons):
                neurons.append(other)
            return self.__class__(neurons, make_copy=self.copy_on_subset)
        elif isinstance(other, NeuronList):
            neurons = self.neurons + [n for n in other.neurons if n not in self]
            return self.__class__(neurons, make_copy=self.copy_on_subset)
        else:
            return NotImplemented

    def append(self, v):
        """Add neuron(s) to this list.

        Examples
        --------
        >>> import navis
        >>> # This is mostly for doctests
        >>> nl = navis.example_neurons()
        >>> len(nl)
        5
        >>> # Add a single neuron to the list
        >>> nl.append(nl[0])
        >>> len(nl)
        6
        >>> # Add a list of neurons to the list
        >>> nl.append(nl)
        >>> len(nl)
        12

        """
        if isinstance(v, core.BaseNeuron):
            self.neurons.append(v)
        elif isinstance(v, NeuronList):
            self.neurons += v.neurons
        else:
            raise NotImplementedError('Unable to append data of type'
                                      f'{type(v)} to NeuronList')

    def apply(self,
              func: Callable,
              *,
              parallel: bool = False,
              n_cores: int = os.cpu_count() // 2,
              omit_failures: bool = False,
              **kwargs):
        """Apply function across all neurons in this NeuronList.

        Parameters
        ----------
        func :          callable
                        Function to be applied. Must accept
                        [`navis.BaseNeuron`][] as first argument.
        parallel :      bool
                        If True (default) will use multiprocessing. Spawning the
                        processes takes time (and memory). Using `parallel=True`
                        makes only sense if the NeuronList is large or the
                        function takes a long time to run.
        n_cores :       int
                        Number of CPUs to use for multiprocessing. Defaults to
                        half the available cores.
        omit_failures : bool
                        If True, will ignore failures.

        **kwargs
                    Will be passed to function.

        Returns
        -------
        Results

        Examples
        --------
        >>> import navis
        >>> nl = navis.example_neurons()
        >>> # Apply resampling function
        >>> nl_rs = nl.apply(navis.resample_skeleton, resample_to=1000, inplace=False)

        """
        if not callable(func):
            raise TypeError('"func" must be callable')

        # Delayed import to avoid circular import
        from .core_utils import NeuronProcessor
        proc = NeuronProcessor(self,
                               func,
                               parallel=parallel,
                               n_cores=n_cores,
                               omit_failures=omit_failures,
                               desc=f'Apply {func.__name__}')

        return proc(self.neurons, **kwargs)

    def sum(self) -> pd.DataFrame:
        """Return sum numeric and boolean values over all neurons."""
        return self.summary().sum(numeric_only=True)

    def mean(self) -> pd.DataFrame:
        """Return mean numeric and boolean values over all neurons."""
        return self.summary().mean(numeric_only=True)

    def memory_usage(self, deep=False, estimate=False, sample=False):
        """Return estimated size in memory of this NeuronList.

        Works by going over each neuron and summing up their size in memory.

        Parameters
        ----------
        deep :          bool
                        Pass to pandas DataFrames. If True will inspect data of
                        object type too.
        estimate :      bool
                        If True, we will only estimate the size. This is
                        considerably faster but will slightly underestimate the
                        memory usage.
        sample :        bool
                        If True, we will only sample 10% of the neurons
                        contained in the list and extrapolate an estimate from
                        there.

        Returns
        -------
        int
                    Memory usage in bytes.

        """
        if self.empty:
            return 0

        if not sample:
            try:
                return sum([n.memory_usage(deep=deep,
                                           estimate=estimate) for n in self.neurons])
            except BaseException:
                return 0
        else:
            try:
                s = sum([n.memory_usage(deep=deep,
                                        estimate=estimate) for n in self.neurons[::10]])
                return s * (len(self.neurons) / len(self.neurons[::10]))
            except BaseException:
                return 0

    def sample(self, N: Union[int, float] = 1) -> 'NeuronList':
        """Return random subset of neurons."""
        if N < 1 and N > 0:
            N = int(len(self.neurons) * N)

        indices = list(range(len(self.neurons)))
        random.shuffle(indices)
        return self.__class__([n for i, n in enumerate(self.neurons) if i in indices[:N]],
                              make_copy=self.copy_on_subset)

    def plot3d(self, **kwargs):
        """Plot neuron in 3D using [`navis.plot3d`][].

        Parameters
        ----------
        **kwargs
                Keyword arguments will be passed to [`navis.plot3d`][].
                See `help(navis.plot3d)` for a list of keywords.

        See Also
        --------
        [`navis.plot3d`][]
                Base function called to generate 3d plot.

        """
        from ..plotting import plot3d

        return plot3d(self, **kwargs)

    def plot2d(self, **kwargs):
        """Plot neuron in 2D using [`navis.plot2d`][].

        Parameters
        ----------
        **kwargs
                Keyword arguments will be passed to [`navis.plot2d`][].
                See `help(navis.plot2d)` for a list of accepted keywords.

        See Also
        --------
        [`navis.plot2d`][]
                Base function called to generate 2d plot.

        """
        from ..plotting import plot2d

        return plot2d(self, **kwargs)

    def summary(self,
                N: Optional[Union[int, slice]] = None,
                add_props: list = [],
                progress=False
                ) -> pd.DataFrame:
        """Get summary over all neurons in this NeuronList.

        Parameters
        ----------
        N :         int | slice, optional
                    If int, get only first N entries.
        add_props : list, optional
                    Additional properties to add to summary. If attribute not
                    available will return 'NA'.
        progress :  bool
                    Whether to show a progress bar. Can be useful for very
                    large list.

        Returns
        -------
        pandas DataFrame

        """
        if not self.empty:
            # Fetch a union of all summary props (keep order)
            all_props = [p for l in self.SUMMARY_PROPS for p in l]
            props = np.unique(all_props)
            props = sorted(props, key=lambda x: all_props.index(x))
        else:
            props = []

        # Add ID to properties - unless all are generic UUIDs
        if any([not isinstance(n.id, uuid.UUID) for n in self.neurons]):
            # Make sure we don't have two IDs
            if 'id' in props:
                props.remove('id')
            props = np.insert(props, 2, 'id')

        if add_props:
            props = np.append(props, add_props)

        if not isinstance(N, slice):
            N = slice(N)

        summary = pd.DataFrame(
            data=[
                [getattr(n, a, "NA") for a in props]
                for n in config.tqdm(
                    self.neurons[N],
                    desc="Summarizing",
                    leave=False,
                    disable=not progress,
                )
            ],
            columns=props,
        )

        if any((n.is_masked for n in self.neurons[N])):
            summary['masked'] = [n.is_masked for n in self.neurons[N]]

        return summary

    def itertuples(self):
        """Helper to mimic `pandas.DataFrame.itertuples()`."""
        return self.neurons

    def add_metadata(self, meta, id_col='id', neuron_id='id', columns=None, register=False, missing='raise'):
        """Add neuron metadata from a DataFrame.

        Parameters
        ----------
        meta :      pd.DataFrame | str | Path
                    DataFrame or filepath to a CSV file containing metadata.
                    Must contain a column with neuron IDs.
        id_col :    str
                    Name of the column containing neuron IDs.
        neuron_id : str
                    Name of the attribute in the neuron that corresponds to
                    the `id_col`.
        columns :   list, optional
                    List of columns to add. If None, will add all columns except
                    for `id_col`.
        register :  bool
                    If True, will also register the attribute(s) as properties
                    that should show up in the summary.
        missing :   'raise' | 'warn' | 'ignore'
                    What to do if `meta` is missing a value for a neuron.

        See Also
        --------
        navis.NeuronList.set_neuron_attributes
                    Set individual attributes of neurons contained in the NeuronList.

        """
        assert missing in ('raise', 'warn', 'ignore')

        if isinstance(meta, (str, Path)):
            meta = pd.read_csv(meta)

        if not isinstance(meta, pd.DataFrame):
            raise TypeError('`meta` must be a DataFrame or a path to a CSV file, '
                            f'got {type(meta)}')

        if id_col not in meta.columns:
            raise KeyError(f'Column "{id_col}" not found in metadata.')

        # Index meta data by the neuron_id
        neuron_id = getattr(self, neuron_id)
        miss = ~np.isin(neuron_id, meta[id_col].values)
        if any(miss):
            msg = f'Metadata is missing entries for IDs: {neuron_id[miss]}'
            if missing == 'raise':
                raise KeyError(msg)
            elif missing == 'warn':
                logger.warning(msg)

        meta = meta.set_index(id_col).reindex(neuron_id)

        if columns is None:
            columns = meta.columns

        for c in columns:
            if c == id_col:
                continue
            self.set_neuron_attributes(
                meta[c].values.tolist(),
                name=c,
                register=register
                )

    def get_neuron_attributes(self, *args, **kwargs):
        """Get attributes of neurons contained in the NeuronList.

        Parameters
        ----------
        name :      str
                    Name of the property to get.
        default :   any, optional
                    Default value to return if attribute is not found.

        Returns
        -------
        np.ndarray
                    Array of values for the requested attribute.

        """
        return np.array([getattr(n, *args, **kwargs) for n in self.neurons])

    def set_neuron_attributes(self, x, name, register=False, na='raise'):
        """Set attributes of neurons contained in the NeuronList.

        Parameters
        ----------
        x :         any | list | np.ndarray | dict | function
                    Value of the property:
                      - lists and arrays are expected to contain a value for
                        each neuron and hence have to match the length of the
                        NeuronList
                      - dict is expected to map `{neuron.id: value}`
                      - a function is expected to take `neuron.id` as input
                        and return a value
        name :      str
                    Name of the property to set.
        register :  bool
                    If True, will also register the attribute(s) as properties
                    that should show up in the summary.
        na :        'raise' | 'propagate' | 'skip'
                    What to do if `x` is a dictionary and does not contain a
                    value for a neuron:
                     - 'raise' will raise a KeyError
                     - 'propagate' will set the attribute to `None`
                     - 'skip' will not set the attribute

        See Also
        --------
        navis.NeuronList.add_metadata
                    Set metadata from a dataframe.

        Examples
        --------
        >>> import navis
        >>> nl = navis.example_neurons(5)
        >>> # Set a single value
        >>> nl.set_neuron_attributes('some_value', name='my_attr')
        >>> nl[0].my_attr
        'some_value'
        >>> # Set individual values as iterable
        >>> nl.set_neuron_attributes([1, 2, 3, 4, 5], name='my_attr')
        >>> nl[0].my_attr
        1
        >>> nl.my_attr
        array([1, 2, 3, 4, 5])
        >>> # Set individual values using a dictionary
        >>> val_dict = dict(zip(nl.id, ['test', 2, 2.2, 4, 'test2']))
        >>> nl.set_neuron_attributes(val_dict, name='my_attr')
        >>> nl[0].my_attr
        'test'

        """
        utils.eval_param(na, name='na',
                         allowed_values=('raise', 'propagate', 'skip'))
        utils.eval_param(name, name='name', allowed_types=(str, ))

        if isinstance(x, dict):
            if na == 'raise':
                miss = ~np.isin(self.id, list(x))
                if any(miss):
                    raise KeyError('Dictionary `x` is missing entries for IDs: '
                                   f'{self.id[miss]}')
            for n in self.neurons:
                v = x.get(n.id, None)
                if (v is None) and (na == 'skip'):
                    continue
                n._register_attr(name, v, summary=register)
        elif isinstance(x, (list, np.ndarray)):
            if len(x) != len(self):
                raise ValueError(f'Got {len(x)} values for the{len(self)} '
                                 'neurons in the NeuronList.')
            for n, v in zip(self.neurons, x):
                n._register_attr(name, v, summary=register)
        elif callable(x):
            for n in self.neurons:
                n._register_attr(name, x(n.id), summary=register)
        else:
            for n in self.neurons:
                n._register_attr(name, x, summary=register)

    def sort_values(self, key: str, ascending: bool = False):
        """Sort neurons by given key.

        Needs to be an attribute of all neurons: for example `name`.
        Also works with custom attributes.
        """
        self.neurons = sorted(self.neurons,
                              key=lambda x: getattr(x, key),
                              reverse=ascending is False)

    def copy(self, **kwargs) -> 'NeuronList':
        """Return copy of this NeuronList.

        Parameters
        ----------
        **kwargs
                    Keyword arguments passed to neuron's `.copy()` method::

                    deepcopy :  bool, for TreeNeurons only
                                If False, `.graph` (NetworkX DiGraphs) will be
                                returned as views - changes to nodes/edges can
                                progagate back! `.igraph` (iGraph) - if
                                available - will always be deepcopied.

        """
        return self.__class__([n.copy(**kwargs) for n in config.tqdm(self.neurons,
                                                                     desc='Copy',
                                                                     leave=False,
                                                                     disable=config.pbar_hide or len(self) < 20)],
                              make_copy=False)

    def head(self, N: int = 5) -> pd.DataFrame:
        """Return summary for top N neurons."""
        return self.summary(N=N)

    def tail(self, N: int = 5) -> pd.DataFrame:
        """Return summary for bottom N neurons."""
        return self.summary(N=slice(-N, len(self)))

    def remove_duplicates(self,
                          key: str = 'name',
                          keep: str = 'first',
                          inplace: bool = False
                          ) -> Optional['NeuronList']:
        """Remove duplicate neurons from list.

        Parameters
        ----------
        key :       str | list, optional
                    Attribute(s) by which to identify duplicates. In case of
                    multiple, all attributes must match to flag a neuron as
                    duplicate.
        keep :      str
                    Which of the duplicated neurons to keep.
        inplace :   bool, optional
                    If False will return a copy of the original with
                    duplicates removed.

        """
        if inplace:
            x = self
        else:
            x = self.copy()

        key = utils.make_iterable(key)

        # Generate pandas DataFrame
        df = pd.DataFrame([[getattr(n, at) for at in key] for n in x],
                          columns=key)

        # Find out which neurons to keep
        to_keep = ~df.duplicated(keep=keep).values

        # Reassign neurons
        x.neurons = x[to_keep].neurons

        if not inplace:
            return x
        return None

    def unmix(self):
        """Split into NeuronLists of the same neuron type.

        Returns
        -------
        dict
                Dictionary of `{Neurontype: NeuronList}`

        """
        return {t: self.__class__([n for n in self.neurons if isinstance(n, t)])
                for t in self.types}


class _IdIndexer():
    """ID-based indexer for NeuronLists to access their neurons by ID."""

    def __init__(self, neuronlist):
        self.nl = neuronlist

    def __getitem__(self, ids):
        # Track if a single neuron was requested
        if not utils.is_iterable(ids):
            single = True
        else:
            single = False

        # Turn into list and force strings
        ids = utils.make_iterable(ids, force_type=str)

        # Generate a map
        # Note we account for the fact we might have duplicate IDs in the list
        map = {}
        for n in self.nl:
            map[str(n.id)] = map.get(str(n.id), []) + [n]

        # Get selection
        sel = [map.get(i, []) for i in ids]

        # Check for missing IDs
        miss = [i for i, k in zip(ids, sel) if len(k) == 0]
        if miss:
            raise ValueError(f'No neuron(s) found for ID(s): {", ".join(miss)}')

        # Check for duplicate Ids in query IDs or in resulting selection
        dupl = [i for i, k in zip(ids, sel) if len(k) > 1]
        if dupl or len(set(ids)) < len(ids):
            logger.warning('Selection contains duplicate IDs.')

        # Flatten selection
        sel = [n for l in sel for n in l]

        if single and len(sel) == 1:
            return sel[0]
        else:
            return self.nl.__class__(sel)
