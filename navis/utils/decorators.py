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

"""Module for decorators.

Important: defer importing from other navis modules to avoid circular imports!
Even if this doesn't cause immediate issues it might well break pickling
functions (e.g. for multiprocessing).

"""

import inspect
import re
import os

import numpy as np
import pandas as pd

from functools import wraps
from textwrap import dedent, indent

from typing import Optional, Union, List, Iterable, Dict, Tuple, Any
from typing_extensions import Literal


def map_neuronlist(desc: str = "",
                   can_zip: List[Union[str, int]] = [],
                   must_zip: List[Union[str, int]] = [],
                   allow_parallel: bool = False):
    """Decorate function to run on all neurons in the NeuronList.

    This also updates the docstring.

    Parameters
    ----------
    desc :           str
                     Descriptor to show in the progress bar if run over multiple
                     neurons.
    can_zip/
    must_zip :       list
                     Names of keyword arguments that need to be zipped together
                     with the neurons in the neuronlist. For example:

                       some_function(NeuronList([n1, n2, n3]), [p1, p2, p3])

                     Should be executed as:

                       some_function(n1, p1)
                       some_function(n2, p2)
                       some_function(n3, p3)

                     `can_zip` will be zipped only if the length matches the
                     length of the neuronlist. If a `can_zip` argument has only
                     one value it will be re-used for all neurons.

                     `must_zip` arguments have to have one value for each of the
                     neurons.

                     Single ``None`` values are always just passed through.

                     Note that for this to consistently work the parameters in
                     question have to be keyword-only (*).
    allow_parallel : bool
                     If True and the function is called with `parallel=True`,
                     will use multiple cores to process the neuronlist. Number
                     of cores a can be set using `n_cores` keyword argument.

    """
    # TODO:
    # - make can_zip/must_zip work with positional-only argumens to, i.e. let
    #   it work with integers instead of strings
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            from .. import core
            # Get the function's signature
            sig = inspect.signature(function)

            try:
                fnname = function.__name__
            except BaseException:
                fnname = str(function)

            parallel = kwargs.pop('parallel', False)
            if parallel and not allow_parallel:
                raise ValueError(f'Function {fnname} does not support parallel '
                                 'processing.')

            # First, we need to extract the neuronlist
            if args:
                # If there are positional arguments, the first one is
                # the input neuron(s)
                nl = args[0]
                nl_key = '__args'
            else:
                # If not, we need to look for the name of the first argument
                # in the signature
                nl_key = list(sig.parameters.keys())[0]
                nl = kwargs.get(nl_key, None)

            # Complain if we did not get what we expected
            if isinstance(nl, type(None)):
                raise ValueError('Unable to identify the neurons for call'
                                 f'{fnname}:\n {args}\n {kwargs}')

            # If we have a neuronlist
            if isinstance(nl, core.NeuronList):
                # Pop the neurons from kwargs or args so we don't pass the
                # neurons twice
                if nl_key == '__args':
                    args = args[1:]
                else:
                    _ = kwargs.pop(nl_key)

                # Check "can zip" arguments
                for p in can_zip:
                    # Skip if not present or is None
                    if p not in kwargs or isinstance(kwargs[p], type(None)):
                        continue

                    if is_iterable(kwargs[p]):
                        # If iterable but length does not match: complain
                        le = len(kwargs[p])
                        if le != len(nl):
                            raise ValueError(f'Got {le} values of `{p}` for '
                                             f'{len(nl)} neurons.')

                # Parse "must zip" arguments
                for p in must_zip:
                    # Skip if not present or is None
                    if p not in kwargs or isinstance(kwargs[p], type(None)):
                        continue

                    values = make_iterable(kwargs[p])
                    if len(values) != len(nl):
                        raise ValueError(f'Got {len(values)} values of `{p}` for '
                                         f'{len(nl)} neurons.')

                # If we use parallel processing it makes sense to modify neurons
                # "inplace" since they will be copied into the child processes
                # anyway and that way we can avoid making an additional copy
                if 'inplace' in kwargs:
                    # First check keyword arguments
                    inplace = kwargs['inplace']
                elif 'inplace' in sig.parameters:
                    # Next check signatures default
                    inplace = sig.parameters['inplace'].default
                else:
                    # All things failing assume it's not inplace
                    inplace = False

                if parallel and 'inplace' in sig.parameters:
                    kwargs['inplace'] = True

                # Prepare processor
                n_cores = kwargs.pop('n_cores', os.cpu_count() // 2)
                chunksize = kwargs.pop('chunksize', 1)
                excl = list(kwargs.keys()) + list(range(1, len(args) + 1))
                proc = core.NeuronProcessor(nl, function,
                                            parallel=parallel,
                                            desc=desc,
                                            warn_inplace=False,
                                            progress=kwargs.pop('progress', True),
                                            omit_failures=kwargs.pop('omit_failures', False),
                                            chunksize=chunksize,
                                            exclude_zip=excl,
                                            n_cores=n_cores)
                # Apply function
                res = proc(nl, *args, **kwargs)

                # When using parallel processing, the neurons will not actually
                # have been modified inplace - in that case we will simply
                # replace the neurons in `nl`
                if inplace:
                    nl.neurons = res.neurons
                else:
                    nl = res

                return nl
            else:
                # If single neuron just pass through
                return function(*args, **kwargs)

        # Update the docstring
        wrapper = map_neuronlist_update_docstring(wrapper, allow_parallel)

        return wrapper

    return decorator


def map_neuronlist_df(desc: str = "",
                      id_col: str = "neuron",
                      reset_index: bool = True,
                      allow_parallel: bool = False):
    """Decorate function to run on all neurons in the NeuronList.

    This version of the decorator is meant for functions that return a
    DataFrame. This decorator will add a `neuron` column with the respective
    neuron's ID and will then concatenate the dataframes.

    Parameters
    ----------
    desc :           str
                     Descriptor to show in the progress bar if run over multiple
                     neurons.
    id_col :         str
                     Name of the ID column to be added to the results dataframe.
    reset_index :    bool
                     Whether to reset the index of the dataframe after
                     concatenating.
    allow_parallel : bool
                     If True and the function is called with `parallel=True`,
                     will use multiple cores to process the neuronlist. Number
                     of cores a can be set using `n_cores` keyword argument.

    """
    # TODO:
    # - make can_zip/must_zip work with positional-only argumens to, i.e. let
    #   it work with integers instead of strings
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            # Lazy import to avoid issues with circular imports and pickling
            from .. import core
            # Get the function's signature
            sig = inspect.signature(function)

            try:
                fnname = function.__name__
            except BaseException:
                fnname = str(function)

            parallel = kwargs.pop('parallel', False)
            if parallel and not allow_parallel:
                raise ValueError(f'Function {fnname} does not allow parallel '
                                 'processing.')

            # First, we need to extract the neuronlist
            if args:
                # If there are positional arguments, the first one is
                # the input neuron(s)
                nl = args[0]
                nl_key = '__args'
            else:
                # If not, we need to look for the name of the first argument
                # in the signature
                nl_key = list(sig.parameters.keys())[0]
                nl = kwargs.get(nl_key, None)

            # Complain if we did not get what we expected
            if isinstance(nl, type(None)):
                raise ValueError('Unable to identify the neurons for call'
                                 f'{fnname}:\n {args}\n {kwargs}')

            # If we have a neuronlist
            if isinstance(nl, core.NeuronList):
                # Pop the neurons from kwargs or args so we don't pass the
                # neurons twice
                if nl_key == '__args':
                    args = args[1:]
                else:
                    _ = kwargs.pop(nl_key)

                # Prepare processor
                n_cores = kwargs.pop('n_cores', os.cpu_count() // 2)
                chunksize = kwargs.pop('chunksize', 1)
                excl = list(kwargs.keys()) + list(range(1, len(args) + 1))
                proc = core.NeuronProcessor(nl, function,
                                            parallel=parallel,
                                            desc=desc,
                                            warn_inplace=False,
                                            progress=kwargs.pop('progress', True),
                                            omit_failures=kwargs.pop('omit_failures', False),
                                            chunksize=chunksize,
                                            exclude_zip=excl,
                                            n_cores=n_cores)
                # Apply function
                res = proc(nl, *args, **kwargs)

                for n, df in zip(nl, res):
                    df.insert(0, column=id_col, value=n.id)

                df = pd.concat(res, axis=0)

                if reset_index:
                    df = df.reset_index(drop=True)

            else:
                # If single neuron just pass through
                df = function(*args, **kwargs)
                # df.insert(0, column=id_col, value=nl.id)

            return df

        # Update the docstring
        wrapper = map_neuronlist_update_docstring(wrapper, allow_parallel)

        return wrapper

    return decorator


def map_neuronlist_update_docstring(func, allow_parallel):
    """Add additional parameters to docstring of function."""
    # Parse docstring
    lines = func.__doc__.split('\n')

    # Find a line with a parameter
    pline = [l for l in lines if ' : ' in l][0]
    # Get the leading whitespaces
    wspaces = ' ' * re.search('( *)', pline).end(1)
    # Get the offset for type and description
    offset = re.search('( *: *)', pline).end(1) - len(wspaces)

    # Find index of the last parameters (assuming there is a single empty
    # line between Returns and the last parameter)
    lastp = [i for i, l in enumerate(lines) if ' Returns' in l][0] - 1

    msg = ''
    if allow_parallel:
        msg += dedent(f"""\
        parallel :{" " * (offset - 10)}bool
                  {" " * (offset - 10)}If True and input is NeuronList, use parallel
                  {" " * (offset - 10)}processing. Requires `pathos`.
        n_cores : {" " * (offset - 10)}int, optional
                  {" " * (offset - 10)}Numbers of cores to use if ``parallel=True``.
                  {" " * (offset - 10)}Defaults to half the available cores.
        """)

    msg += dedent(f"""\
    progress :{" " * (offset - 10)}bool
              {" " * (offset - 10)}Whether to show a progress bar. Overruled by
              {" " * (offset - 10)}``navis.set_pbars``.
    omit_failures :{" " * (offset - 15)}bool
                   {" " * (offset - 15)}If True will omit failures instead of raising
                   {" " * (offset - 15)}an exception. Ignored if input is single neuron.
    """)

    # Insert new docstring
    lines.insert(lastp, indent(msg, wspaces))

    # Update docstring
    func.__doc__ = '\n'.join(lines)

    return func


def lock_neuron(function):
    """Lock neuron while function is executed.

    This makes sure that temporary attributes aren't re-calculated as changes
    are being made.

    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        # Lazy import to avoid issues with circular imports and pickling
        from .. import core

        # Lock if first argument is a neuron
        if isinstance(args[0], core.BaseNeuron):
            args[0]._lock = getattr(args[0], '_lock', 0) + 1
        try:
            # Execute function
            res = function(*args, **kwargs)
        except BaseException:
            raise
        finally:
            # Unlock neuron
            if isinstance(args[0], core.BaseNeuron):
                args[0]._lock -= 1
        # Return result
        return res
    return wrapper


def meshneuron_skeleton(method: Union[Literal['subset'],
                                      Literal['split'],
                                      Literal['node_properties'],
                                      Literal['node_to_vertex'],
                                      Literal['pass_through']],
                        include_connectors: bool = False,
                        copy_properties: list = [],
                        disallowed_kwargs: dict = {},
                        node_props: list = [],
                        reroot_soma: bool = False,
                        heal: bool = False):
    """Decorate function such that MeshNeurons are automatically skeletonized,
    the function is run on the skeleton and changes are propagated
    back to the meshe.

    Parameters
    ----------
    method :    str
                What to do with the results:
                  - 'subset': subset MeshNeuron to what's left of the skeleton
                  - 'split': split MeshNeuron following the skeleton's splits
                  - 'node_to_vertex': map the returned node ID to the vertex IDs
                  - 'node_properties' map node properties to vertices (requires
                    `node_props` parameter)
                  - 'pass_through' simply passes through the return value
    include_connectors : bool
                If True, will try to make sure that if the MeshNeuron has
                connectors, they will be carried over to the skeleton.
    copy_properties : list
                Any additional properties that need to be copied from the
                skeleton to the mesh.
    disallowed_kwargs : dict
                Keyword arguments (name + value) that are not permitted when
                input is MeshNeuron.
    node_props : list
                For method 'node_properties'. String must be column names in
                node table of skeleton.
    reroot_soma :  bool
                If True and neuron has a soma (.soma_pos), will reroot to
                that soma.
    heal :      bool
                Whether or not to heal the skeleton if the mesh is fragmented.

    """
    assert isinstance(copy_properties, list)
    assert isinstance(disallowed_kwargs, dict)
    assert isinstance(node_props, list)

    allowed_methods = ('subset', 'node_to_vertex', 'split', 'node_properties',
                       'pass_through')
    if method not in allowed_methods:
        raise ValueError(f'Unknown method "{method}"')

    if method == 'node_properties' and not node_props:
        raise ValueError('Must provide `node_props` for method "node_properties"')

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            # Get the function's signature
            sig = inspect.signature(function)

            try:
                fnname = function.__name__
            except BaseException:
                fnname = str(function)

            # First, we need to extract the neuron from args and kwargs
            if args:
                # If there are positional arguments, the first one is assumed to
                # be the input neuron
                x = args[0]
                args = args[1:]
                x_key = '__args'
            else:
                # If not, we need to look for the name of the first argument
                # in the signature
                x_key = list(sig.parameters.keys())[0]
                x = kwargs.pop(x_key, None)

            # Complain if we did not get what we expected
            if isinstance(x, type(None)):
                raise ValueError('Unable to identify the neurons for call'
                                 f'{fnname}:\n {args}\n {kwargs}')

            # If input not a MeshNeuron, just pass through
            # Note delayed import to avoid circular imports and IMPORTANTLY
            # funky interactions with pickle/dill
            from .. import core
            if not isinstance(x, core.MeshNeuron):
                return function(x, *args, **kwargs)

            # Check for disallowed kwargs
            for k, v in disallowed_kwargs.items():
                if k in kwargs and kwargs[k] == v:
                    raise ValueError(f'{k}={v} is not allowed when input is '
                                     'MeshNeuron(s).')

            # See if this is meant to be done inplace
            if 'inplace' in kwargs:
                # First check keyword arguments
                inplace = kwargs['inplace']
            elif 'inplace' in sig.parameters:
                # Next check signatures default
                inplace = sig.parameters['inplace'].default
            else:
                # All things failing assume it's not inplace
                inplace = False

            # Now skeletonize
            sk = x.skeleton

            # Delayed import to avoid circular imports
            # Note that this HAS to be in the inner function otherwise
            # we get a weird error when pickling for parallel processing
            from .. import morpho

            if heal:
                sk = morpho.heal_skeleton(sk, method='LEAFS')

            if reroot_soma and sk.has_soma:
                sk = sk.reroot(sk.soma)

            if include_connectors and x.has_connectors and not sk.has_connectors:
                sk._connectors = x.connectors.copy()
                sk._connectors['node_id'] = sk.snap(sk.connectors[['x', 'y', 'z']].values)[0]

            # Apply function
            res = function(sk, *args, **kwargs)

            if method == 'subset':
                # See which vertices we need to keep
                keep = np.isin(sk.vertex_map, res.nodes.node_id.values)

                x = morpho.subset_neuron(x, keep, inplace=inplace)

                for p in copy_properties:
                    setattr(x, p, getattr(sk, p, None))
            elif method == 'split':
                meshes = []
                for n in res:
                    # See which vertices we need to keep
                    keep = np.isin(sk.vertex_map, n.nodes.node_id.values)

                    meshes.append(morpho.subset_neuron(x, keep, inplace=False))

                    for p in copy_properties:
                        setattr(meshes[-1], p, getattr(n, p, None))
                x = core.NeuronList(meshes)
            elif method == 'node_to_vertex':
                x = np.where(sk.vertex_map == res)[0]
            elif method == 'node_properties':
                for p in node_props:
                    node_map = sk.nodes.set_index('node_id')[p].to_dict()
                    vertex_props = np.array([node_map[n] for n in sk.vertex_map])
                    setattr(x, p, vertex_props)
            elif method == 'pass_through':
                return res

            return x

        return wrapper

    return decorator
