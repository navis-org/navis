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
import types

import numpy as np
import pandas as pd

from .. import utils, config, core

__all__ = ['NeuronList']

# Set up logging
logger = config.logger


class NeuronList:
    """ Compilation of :class:`~navis.core.TreeNeuron` that allow quick
    access to neurons' attributes/functions. They are designed to work in many
    ways much like a pandas.DataFrames by, for example, supporting
    ``.itertuples()``, ``.empty`` or ``.copy()``.

    Attributes
    ----------
    nodes :             ``pandas.DataFrame``
                        Merged node table.
    connectors :        ``pandas.DataFrame``
                        Merged connector table. This also works for
                        `presynapses`, `postsynapses` and `gap_junctions`.
    graph :             np.array of ``networkx`` graph objects
    igraph :            np.array of ``igraph`` graph objects
    review_status :     np.array of int
    n_connectors :      np.array of int
    n_branch_nodes :    np.array of int
    n_end_nodes :       np.array of int
    n_open_ends :       np.array of int
    cable_length :      np.array of float
                        Cable lengths in micrometers [um].
    soma :              np.array of node_ids
    root :              np.array of node_ids
    n_cores :           int
                        Number of cores to use. Default ``os.cpu_count()-1``.
    use_threading :    bool (default=True)
                        If True, will use parallel threads. Should be slightly
                        up to a lot faster depending on the numbers of cores.
                        Switch off if you experience performance issues.

    """

    def __init__(self, x, make_copy=False, use_parallel=False):
        """ Initialize NeuronList.

        Parameters
        ----------
        x :                 list | array | core.TreeNeuron | NeuronList
                            Data to construct neuronlist from. Can be either:

                            1. core.TreeNeuron(s)
                            2. NeuronList(s)
                            3. Anything that constructs a core.TreeNeuron
                            4. List of the above

        make_copy :         bool, optional
                            If True, Neurons are deepcopied before being
                            assigned to the neuronlist.
        use_parallel :      bool, optional
                            If True, will use parallel processing for
                            operations performed across all neuron in this
                            neuronlist. This is very memory heavy!
        """

        # Set number of cores
        self.n_cores = max(1, os.cpu_count())

        # If below parameter is True, most calculations will be parallelized
        # which speeds them up quite a bit. Unfortunately, this uses A TON of
        # memory - for large lists this might make your system run out of
        # memory. In these cases, leave this property at False
        self.use_parallel = use_parallel
        self.use_threading = True

        # Determines if subsetting this NeuronList will copy the neurons
        self.copy_on_subset = False

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
            self.neurons = list(x)
        else:
            # Any other datatype will simply be assumed to be accepted by
            # core.TreeNeuron() - if not this will throw an error
            self.neurons = [x]

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
                    futures = e.map(core.TreeNeuron, [n[0] for n in to_convert])

                    converted = [n for n in config.tqdm(futures,
                                                        total=len(to_convert),
                                                        desc='Make nrn',
                                                        disable=config.pbar_hide,
                                                        leave=config.pbar_leave)]

                    for i, c in enumerate(to_convert):
                        self.neurons[c[1]] = converted[i]

            else:
                for n in config.tqdm(to_convert, desc='Make nrn',
                                     disable=config.pbar_hide,
                                     leave=config.pbar_leave):
                    self.neurons[n[2]] = core.TreeNeuron(n[0])

    def _convert_helper(self, x):
        """ Helper function to convert x to core.TreeNeuron."""
        return core.TreeNeuron(x[0])

    def summary(self, N=None, add_cols=[]):
        """ Get summary over all neurons in this NeuronList.

        Parameters
        ----------
        N :         int | slice, optional
                    If int, get only first N entries.
        add_cols :  list, optional
                    Additional columns for the summary. If attribute not
                    available will return 'NA'.

        Returns
        -------
        pandas DataFrame

        """
        cols = ['type', 'n_nodes', 'n_connectors', 'n_branches', 'n_leafs',
                'cable_length', 'soma']
        cols += add_cols

        if not isinstance(N, slice):
            N = slice(N)

        return pd.DataFrame(data=[[getattr(n, a, 'NA') for a in cols]
                                  for n in self.neurons[N]],
                            columns=cols)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{type(self)} of {len(self)} neurons \n {str(self.summary())}'

    def _repr_html_(self):
        return self.summary()._repr_html_()

    def __iter__(self):
        """ Iterator instanciates a new class everytime it is called.
        This allows the use of nested loops on the same neuronlist object.
        """
        class prange_iter:
            def __init__(self, neurons, start):
                self.iter = start
                self.neurons = neurons

            def __next__(self):
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
            # Dynamically check if the requested attribute/function exists in
            # all neurons
            values = [getattr(n, key, NotImplemented) for n in self.neurons]
            is_method = [isinstance(v, types.MethodType) for v in values]
            is_frame = [isinstance(v, (pd.DataFrame, type(None))) for v in values]

            # First check if there is any reason why we can't collect this
            # attribute across all neurons
            if all([isinstance(v, type(NotImplemented)) for v in values]):
                raise AttributeError(f'Attribute "{key}" not found in in '
                                     'NeuronList nor in contained neurons')
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
                return NeuronProcessor(self, values, key)

    def __contains__(self, x):
        return x in self.neurons

    def __getitem__(self, key):
        if isinstance(key, np.ndarray) and key.dtype == 'bool':
            subset = [n for i, n in enumerate(self.neurons) if key[i]]
        elif utils.is_iterable(key):
            if False not in [isinstance(k, bool) for k in key]:
                subset = [n for i, n in enumerate(self.neurons) if key[i]]
            else:
                subset = [self.neurons[i] for i in key]
        else:
            subset = self.neurons[key]

        if isinstance(subset, core.TreeNeuron):
            return subset

        return NeuronList(subset, make_copy=self.copy_on_subset)

    def __missing__(self, key):
        logger.error('No neuron matching the search critera.')
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

    def sum(self):
        """Returns sum numeric and boolean values over all neurons. """
        return self.summary().sum(numeric_only=True)

    def mean(self):
        """Returns mean numeric and boolean values over all neurons. """
        return self.summary().mean(numeric_only=True)

    def sample(self, N=1):
        """Returns random subset of neurons."""
        indices = list(range(len(self.neurons)))
        random.shuffle(indices)
        return NeuronList([n for i, n in enumerate(self.neurons) if i in indices[:N]],
                          make_copy=self.copy_on_subset)

    def plot3d(self, **kwargs):
        """Plot neuron in 3D.

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
        """Plot neuron in 2D.

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

    def sort_values(self, key, ascending=False):
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

    def copy(self, deepcopy=False):
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
                          make_copy=False,
                          use_parallel=self.use_parallel)

    def head(self, N=5):
        """Return summary for top N neurons."""
        return self.summary(N=N)

    def tail(self, N=5):
        """Return summary for bottom N neurons."""
        return self.summary(N=slice(-N, len(self)))

    def remove_duplicates(self, key='neuron_name', inplace=False):
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


class NeuronProcessor:
    """ Helper class to allow processing of arbitrary functions of
    all neurons in a neuronlist.
    """
    def __init__(self, nl, funcs, desc=None):
        self.nl = nl
        self.funcs = funcs
        self.desc = None

        # This makes sure that help and name match the functions being called
        functools.update_wrapper(self, funcs[0])

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

        if self.nl.use_parallel:
            pool = mp.Pool(self.nl.n_cores)
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

        if kwargs.get('inplace', True):
            self.nl.neurons = res
            return
        else:
            return NeuronList(res)


def _worker_wrapper(x):
    f, args, kwargs = x
    return f(*args, **kwargs)
