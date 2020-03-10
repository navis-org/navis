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

from concurrent.futures import ThreadPoolExecutor
import functools
import multiprocessing as mp
import numbers
import os
import random
import re
import types
import uuid

import networkx as nx

import numpy as np
import pandas as pd

from typing import (ClassVar, Sequence, Union, Iterable, List, Any,
                    Optional, Callable, Iterator)

from .. import utils, config, core

__all__ = ['NeuronList']

# Set up logging
logger = config.logger


class NeuronList:
    """Compilation of :class:`~navis.TreeNeuron`.

    Gives quick access to neurons' attributes and functions.

    Parameters
    ----------
    x :                 list | array | TreeNeuron | NeuronList
                        Data to construct neuronlist from. Can be either:

                        1. TreeNeuron(s)
                        2. NeuronList(s)
                        3. Anything that constructs a TreeNeuron
                        4. List of the above

    make_copy :         bool, optional
                        If True, Neurons are deepcopied before being
                        assigned to the NeuronList.

    Attributes
    ----------
    use_threading :     bool (default=True)
                        If True, will use parallel threads when initialising the
                        NeuronList. Should be slightly up to a lot faster
                        depending on the numbers of cores. Switch off if you
                        experience performance issues.
    n_cores :           int
                        Number of cores to use for threading and parallel
                        processing. Default = ``os.cpu_count()-1``.
    **kwargs
                        Will be passed to constructor of TreeNeuron.

    """
    neurons: List[core.TreeNeuron]

    nodes: pd.DataFrame
    connectors: pd.DataFrame

    n_connectors: Sequence[int]
    n_nodes: Sequence[int]
    n_branch_nodes: Sequence[int]
    n_end_nodes: Sequence[int]

    cable_length: Sequence[int]

    soma: Sequence[int]
    root: Sequence[int]

    graph: 'nx.DiGraph'
    igraph: 'igraph.Graph'  # type: ignore  # doesn't know iGraph

    def __init__(self,
                 x: Union[Iterable[Union[core.TreeNeuron,
                                         'NeuronList',
                                         pd.DataFrame]],
                          'NeuronList',
                          core.TreeNeuron,
                          pd.DataFrame],
                 make_copy: bool = False,
                 **kwargs):
        # Set number of cores
        self.n_cores: int = max(1, os.cpu_count())

        # If below parameter is True, most calculations will be parallelized
        # which speeds them up quite a bit. Unfortunately, this uses A TON of
        # memory - for large lists this might make your system run out of
        # memory. In these cases, leave this property at False
        self.use_threading: bool = True

        # Determines if subsetting this NeuronList will copy the neurons
        self.copy_on_subset: bool = False

        if isinstance(x, NeuronList):
            # We can't simply say self.neurons = x.neurons b/c that way
            # changes in the list would backpropagate
            self.neurons = [n for n in x.neurons]
        elif utils.is_iterable(x):
            # If x is a list of mixed objects we need to unpack/flatten that
            # E.g. x = [NeuronList, NeuronList, core.TreeNeuron]
            to_unpack = [e for e in x if isinstance(e, NeuronList)]
            x = [e for e in x if not isinstance(e, NeuronList)]
            x += [n for ob in to_unpack for n in ob.neurons]

            # We have to convert from numpy ndarray to list
            # Do NOT remove list() here!
            self.neurons = list(x)  # type: ignore
        else:
            # Any other datatype will simply be assumed to be accepted by
            # core.TreeNeuron() - if not this will throw an error
            self.neurons = [x]  # type: ignore

        # Now convert and/or make copies if necessary
        to_convert = []
        for i, n in enumerate(self.neurons):
            if not isinstance(n, core.TreeNeuron) or make_copy is True:
                # The `i` keeps track of the original index so that after
                # conversion to Neurons, the objects will occupy the same
                # position
                to_convert.append((n, i))

        if to_convert:
            if self.use_threading:
                with ThreadPoolExecutor(max_workers=self.n_cores) as e:
                    futures = e.map(lambda x: core.TreeNeuron(x, **kwargs),
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
                    self.neurons[n[2]] = core.TreeNeuron(n[0])

        # Add ID-based indexer
        self.idx = _IdIndexer(self.neurons)

    def _convert_helper(self, x: Any) -> core.TreeNeuron:
        """Helper function to convert x to core.TreeNeuron."""
        return core.TreeNeuron(x[0])

    def summary(self,
                N: Optional[Union[int, slice]] = None,
                add_props: list = []
                ) -> pd.DataFrame:
        """Get summary over all neurons in this NeuronList.

        Parameters
        ----------
        N :         int | slice, optional
                    If int, get only first N entries.
        add_props : list, optional
                    Additional properties to add to summary. If attribute not
                    available will return 'NA'.

        Returns
        -------
        pandas DataFrame

        """
        # Fetch a union of all summary props
        props = list(set.union(*[set(p) for p in self.SUMMARY_PROPS]))

        # Add ID to properties - unless all are generic UUIDs
        if any([not isinstance(n.id, uuid.UUID) for n in self.neurons]):
            props.insert(2, 'id')

        if add_props:
            props = np.append(props, add_props)

        if not isinstance(N, slice):
            N = slice(N)

        return pd.DataFrame(data=[[getattr(n, a, 'NA') for a in props]
                                  for n in self.neurons[N]],
                            columns=props)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{type(self)} of {len(self)} neurons \n {str(self.summary())}'

    def _repr_html_(self):
        return self.summary()._repr_html_()

    def __iter__(self) -> Iterator['core.TreeNeuron']:
        """ Iterator instanciates a new class everytime it is called.
        This allows the use of nested loops on the same neuronlist object.
        """
        class prange_iter(Iterator['core.TreeNeuron']):
            def __init__(self, neurons, start):
                self.iter = start
                self.neurons = neurons

            def __next__(self) -> 'core.TreeNeuron':
                if self.iter >= len(self.neurons):
                    raise StopIteration
                to_return = self.neurons[self.iter]
                self.iter += 1
                return to_return

        return prange_iter(self.neurons, 0)

    def __len__(self):
        """Use skeleton ID here, otherwise this is terribly slow."""
        return len(self.neurons)

    def __dir__(self):
        """ Custom __dir__ to add some parameters that we want to make
        searchable.
        """
        add_attr = set.union(*[set(dir(n)) for n in self.neurons])

        return list(set(super().__dir__() + list(add_attr)))

    def __getattr__(self, key):
        if key == 'shape':
            return (self.__len__(),)
        elif key == 'bbox':
            return self.nodes.describe().loc[['min', 'max'], ['x', 'y', 'z']].values.T
        elif key == 'empty':
            return len(self.neurons) == 0
        else:
            if self.empty:
                raise AttributeError(f'Neuronlist is empty - "{key}" not found')
            # Dynamically check if the requested attribute/function exists in
            # all neurons
            values = [getattr(n, key, NotImplemented) for n in self.neurons]
            is_method = [isinstance(v, types.MethodType) for v in values]
            is_frame = [isinstance(v, (pd.DataFrame, type(None))) for v in values]

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
                if all(is_frame):
                    data = []
                    for i, v in enumerate(values):
                        if isinstance(v, pd.DataFrame):
                            v['neuron'] = i
                            data.append(v)
                    return pd.concat(data,
                                     axis=0,
                                     ignore_index=True,
                                     sort=True)
                else:
                    return np.array(values)
            else:
                # Return function but wrap it in a function that will show
                # a progress bar and use multiprocessing (if applicable)
                return NeuronProcessor(self, values, parallel=False, desc=key)

    def __setattr__(self, key, value):
        # Check if this attribute exists in the neurons
        if any([hasattr(n, key) for n in self.neurons]):
            logger.warning('It looks like you are trying to add a TreeNeuron '
                           f'attribute to a NeuronList. "{key}" will not '
                           'propagated to the neurons it contains!')

        self.__dict__[key] = value

    @property
    def neurons(self):
        return self.__dict__.get('neurons', [])

    def apply(self, func, parallel=False, n_cores=os.cpu_count(), **kwargs):
        """Apply function across all neurons in this NeuronList.

        Parameters
        ----------
        func :      callable
                    Function to be applied. Must accept :class:`~navis.TreeNeuron`
                    as first argument.
        parallel :  bool
                    If True (default) will use multiprocessing. Spawning the
                    processes takes time (and memory). So using ``parallel=True``
                    makes only sense if the NeuronList is large or the function
                    takes a long time.
        n_cores :   int
                    Number of CPUs to use for multiprocessing.

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
        >>> nl_rs = nl.apply(navis.resample_neuron, resample_to=1000, inplace=False)

        """
        if not callable(func):
            raise TypeError('"func" must be callable')

        proc = NeuronProcessor(self,
                               func,
                               parallel=parallel,
                               n_cores=n_cores,
                               desc='Processing')

        return proc(self.neurons, **kwargs)

    def __contains__(self, x):
        return x in self.neurons

    def __getitem__(self, key):
        if utils.is_iterable(key):
            if all([isinstance(k, (bool, np.bool_)) for k in key]):
                subset = [n for i, n in enumerate(self.neurons) if key[i]]
            else:
                subset = [self[i] for i in key]
        elif isinstance(key, str):
            subset = [n for n in self.neurons if re.fullmatch(key, getattr(n, 'name', ''))]
        elif isinstance(key, (int, slice)):
            subset = self.neurons[key]
        else:
            raise NotImplementedError(f'Indexing NeuronList by {type(key)} not implemented')

        if isinstance(subset, core.TreeNeuron):
            return subset

        # Make sure we unpack neurons
        subset = utils.unpack_neurons(subset)
        # Make sure each neuron shows up only once but keep original order
        subset = sorted(set(subset), key=lambda x: subset.index(x))

        if not subset:
            # This will call __missing__
            return self.__missing__(key)

        return NeuronList(subset, make_copy=self.copy_on_subset)

    def __missing__(self, key):
        raise AttributeError('No neuron matching the search critera.')

    def __add__(self, to_add):
        """Implements addition. """
        if isinstance(to_add, core.TreeNeuron):
            return NeuronList(self.neurons + [to_add],
                              make_copy=self.copy_on_subset)
        elif isinstance(to_add, NeuronList):
            return NeuronList(self.neurons + to_add.neurons,
                              make_copy=self.copy_on_subset)
        elif utils.is_iterable(to_add):
            if False not in [isinstance(n, core.TreeNeuron) for n in to_add]:
                return NeuronList(self.neurons + list(to_add),
                                  make_copy=self.copy_on_subset)
            else:
                return NeuronList(self.neurons + [core.TreeNeuron[n] for n in to_add],
                                  make_copy=self.copy_on_subset)
        else:
            return NotImplemented

    def __eq__(self, other):
        """Implements equality. """
        if isinstance(other, NeuronList):
            if len(self) != len(other):
                return False
            else:
                return all([n1 == n2 for n1, n2 in zip(self, other)])
        else:
            return NotImplemented

    def __sub__(self, to_sub):
        """Implements substraction. """
        if isinstance(to_sub, core.TreeNeuron):
            return NeuronList([n for n in self.neurons if n != to_sub],
                              make_copy=self.copy_on_subset)
        elif isinstance(to_sub, NeuronList):
            return NeuronList([n for n in self.neurons if n not in to_sub],
                              make_copy=self.copy_on_subset)
        else:
            return NotImplemented

    def __truediv__(self, other):
        """Implements division for coordinates (nodes, connectors)."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            nl = self.copy()
            for n in nl:
                n.nodes.loc[:, ['x', 'y', 'z', 'radius']] /= other

                if self.has_connectors:
                    n.connectors.loc[:, ['x', 'y', 'z']] /= other

                n._clear_temp_attr(exclude=['classify_nodes'])
            return nl
        else:
            return NotImplemented

    def __mul__(self, other):
        """Implements multiplication for coordinates (nodes, connectors)."""
        if isinstance(other, numbers.Number):
            # If a number, consider this an offset for coordinates
            nl = self.copy()
            for n in nl:
                n.nodes.loc[:, ['x', 'y', 'z', 'radius']] *= other

                if self.has_connectors:
                    n.connectors.loc[:, ['x', 'y', 'z']] *= other

                n._clear_temp_attr(exclude=['classify_nodes'])
            return nl
        else:
            return NotImplemented

    def __and__(self, other):
        """Implements bitwise AND using the & operator. """
        if isinstance(other, core.TreeNeuron):
            return NeuronList([n for n in self.neurons if n == other],
                              make_copy=self.copy_on_subset)
        elif isinstance(other, NeuronList):
            return NeuronList([n for n in self.neurons if n in other],
                              make_copy=self.copy_on_subset)
        else:
            return NotImplemented

    def sum(self) -> pd.DataFrame:
        """Returns sum numeric and boolean values over all neurons. """
        return self.summary().sum(numeric_only=True)

    def mean(self) -> pd.DataFrame:
        """Returns mean numeric and boolean values over all neurons. """
        return self.summary().mean(numeric_only=True)

    def sample(self, N: int = 1) -> 'NeuronList':
        """Returns random subset of neurons."""
        indices = list(range(len(self.neurons)))
        random.shuffle(indices)
        return NeuronList([n for i, n in enumerate(self.neurons) if i in indices[:N]],
                          make_copy=self.copy_on_subset)

    def plot3d(self, **kwargs):
        """Plot neuron in 3D using :func:`~navis.plot3d`.

        Parameters
        ----------
        **kwargs
                Keyword arguments will be passed to :func:`navis.plot3d`.
                See ``help(navis.plot3d)`` for a list of keywords.

        See Also
        --------
        :func:`~navis.plot3d`
                Base function called to generate 3d plot.
        """

        from ..plotting import plot3d

        return plot3d(self, **kwargs)

    def plot2d(self, **kwargs):
        """Plot neuron in 2D using :func:`~navis.plot2d`.

        Parameters
        ----------
        **kwargs
                Keyword arguments will be passed to :func:`navis.plot2d`.
                See ``help(navis.plot2d)`` for a list of accepted keywords.

        See Also
        --------
        :func:`~navis.plot2d`
                Base function called to generate 2d plot.
        """

        from ..plotting import plot2d

        return plot2d(self, **kwargs)

    def itertuples(self):
        """Helper class to mimic ``pandas.DataFrame`` ``itertuples()``."""
        return self.neurons

    def sort_values(self, key: str, ascending: bool = False):
        """Sort neurons by given key.

        Needs to be an attribute of all neurons: for example ``n_nodes``.
        Also works with custom attributes.
        """
        self.neurons = sorted(self.neurons,
                              key=lambda x: getattr(x, key),
                              reverse=ascending is False)

    def __copy__(self):
        return self.copy(deepcopy=False)

    def __deepcopy__(self):
        return self.copy(deepcopy=True)

    def copy(self, deepcopy: bool = False) -> 'NeuronList':
        """Return copy of this NeuronList.

        Parameters
        ----------
        deepcopy :  bool, optional
                    If False, ``.graph`` (NetworkX DiGraphs) will be returned
                    as views - changes to nodes/edges can progagate back!
                    ``.igraph`` (iGraph) - if available - will always be
                    deepcopied.

        """
        return NeuronList([n.copy(deepcopy=deepcopy) for n in config.tqdm(self.neurons,
                                                                          desc='Copy',
                                                                          leave=False,
                                                                          disable=config.pbar_hide | len(self) < 20)],
                          make_copy=False)

    def head(self, N: int = 5) -> pd.DataFrame:
        """Return summary for top N neurons."""
        return self.summary(N=N)

    def tail(self, N: int = 5) -> pd.DataFrame:
        """Return summary for bottom N neurons."""
        return self.summary(N=slice(-N, len(self)))

    def remove_duplicates(self,
                          key: str = 'neuron_name',
                          inplace: bool = False
                          ) -> Optional['NeuronList']:
        """Removes duplicate neurons from list.

        Parameters
        ----------
        key :       str | list, optional
                    Attribute(s) by which to identify duplicates. In case of
                    multiple, all attributes must match to flag a neuron as
                    duplicate.
        inplace :   bool, optional
                    If False will return a copy of the original with
                    duplicates removed.
        """
        if inplace:
            x = self
        else:
            x = self.copy(deepcopy=False)

        key = utils.make_iterable(key)

        # Generate pandas DataFrame
        df = pd.DataFrame([[getattr(n, at) for at in key] for n in x],
                          columns=key)

        # Find out which neurons to keep
        keep = ~df.duplicated(keep='first').values

        # Reassign neurons
        x.neurons = x[keep].neurons

        if not inplace:
            return x
        return None


class NeuronProcessor:
    """ Helper class to allow processing of arbitrary functions of
    all neurons in a neuronlist.
    """

    def __init__(self,
                 nl: NeuronList,
                 funcs: Callable,
                 parallel: bool = False,
                 n_cores: int = os.cpu_count() - 1,
                 desc: Optional[str] = None):
        self.nl = nl
        self.funcs = funcs
        self.desc = None
        self.parallel = parallel
        self.n_cores = n_cores

        # Copy function for each neuron in neuronlist
        if not utils.is_iterable(self.funcs):
            self.funcs = [self.funcs] * len(nl)

        # This makes sure that help and name match the functions being called
        functools.update_wrapper(self, self.funcs[0])

    def __call__(self, *args, **kwargs):
        # We will check for each argument if it matches the number of
        # functions to be run. If they do, we will assume that each value
        # is meant for a single function
        parsed_args = []
        parsed_kwargs = []

        for i in range(len(self.funcs)):
            parsed_args.append([])
            parsed_kwargs.append({})
            for k, a in enumerate(args):
                if not utils.is_iterable(a) or len(a) != len(self.funcs):
                    parsed_args[i].append(a)
                else:
                    parsed_args[i].append(a[i])

            for k, v in kwargs.items():
                if not utils.is_iterable(v) or len(v) != len(self.funcs):
                    parsed_kwargs[i][k] = v
                else:
                    parsed_kwargs[i][k] = v[i]

        # Silence loggers (except Errors)
        level = logger.getEffectiveLevel()

        logger.setLevel('ERROR')
        if self.parallel:
            pool = mp.Pool(self.n_cores)
            combinations = list(zip(self.funcs, parsed_args, parsed_kwargs))
            res = list(config.tqdm(pool.imap(_worker_wrapper,
                                             combinations,
                                             chunksize=10),
                                   total=len(combinations),
                                   desc=self.desc,
                                   disable=config.pbar_hide,
                                   leave=config.pbar_leave))
            pool.close()
            pool.join()
        else:
            res = []
            for i, f in enumerate(config.tqdm(self.funcs, desc=self.desc,
                                              disable=config.pbar_hide,
                                              leave=config.pbar_leave)):
                res.append(f(*parsed_args[i], **parsed_kwargs[i]))

        # Reset logger level to previous state
        logger.setLevel(level)

        if not kwargs.get('inplace', False):
            return NeuronList(res)


def _worker_wrapper(x: Sequence):
    f, args, kwargs = x
    return f(*args, **kwargs)


class _IdIndexer():
    """ID-based indexer for NeuronLists to access their neurons by ID."""

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, ids):
        # Turn into list and force strings
        ids = utils.make_iterable(ids, force_type=str)

        # Get objects that match skid
        sel = [n for n in self.obj if str(n.id) in ids]

        # Reorder to keep in the order requested
        sel = sorted(sel, key=lambda x: np.where(ids == str(x.id))[0][0])

        if len(sel) != len(ids):
            miss = list(set(ids) - set([n.id for n in sel]))
            raise ValueError(f'No neuron(s) with ID(s): {", ".join(miss)}')
        elif len(sel) == 1:
            return sel[0]
        else:
            return NeuronList(sel)
