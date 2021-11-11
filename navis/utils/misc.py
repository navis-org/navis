
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

import inspect
import math
import os
import re
import requests
import urllib


import numpy as np
import pandas as pd

from functools import wraps
from textwrap import dedent, indent
from typing import Optional, Union, List, Iterable, Dict, Tuple, Any
from typing_extensions import Literal

from .. import config, core
from .eval import is_mesh
from .iterables import is_iterable, make_iterable
from ..transforms.templates import TemplateBrain

# Set up logging
logger = config.logger


def round_smart(num: Union[int, float], prec: int = 8) -> float:
    """Round number intelligently to produce Human-readable numbers.

    This functions rounds to the Nth decimal, where N is `precision` minus the
    number of digits before the decimal. The general idea is that the bigger
    the number, the less we care about decimals - and vice versa.

    Parameters
    ----------
    num :       float | int
                A number.
    prec :      int
                The precision we are aiming for.

    Examples
    --------
    >>> import navis
    >>> navis.utils.round_smart(0.00999)
    0.00999
    >>> navis.utils.round_smart(10000000.00999)
    10000000.0

    """
    # Number of digits before decimal
    lg = math.log10(num)
    if lg < 0:
        N = 0
    else:
        N = int(lg)

    return round(num, max(prec - N, 0))


def sizeof_fmt(num, suffix='B'):
    """Bytes to Human readable."""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def make_volume(x: Any) -> 'core.Volume':
    """Try making a navis.Volume from input object."""
    if isinstance(x, core.Volume):
        return x
    if is_mesh(x):
        inits = dict(vertices=x.vertices, faces=x.faces)
        for p in ['name', 'id', 'color']:
            if hasattr(x, p):
                inits[p] = getattr(x, p, None)
        return core.Volume(**inits)

    raise TypeError(f'Unable to coerce input of type "{type(x)}" to navis.Volume')


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
        lines = wrapper.__doc__.split('\n')

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
        wrapper.__doc__ = '\n'.join(lines)

        return wrapper

    return decorator


def meshneuron_skeleton(method: Union[Literal['subset'],
                                      Literal['split'],
                                      Literal['node_properties'],
                                      Literal['node_to_vertex']],
                        include_connectors: bool = False,
                        copy_properties: list = [],
                        disallowed_kwargs: dict = {},
                        node_props: list = [],
                        reroot_soma: bool = False,
                        heal: bool = False):
    """Decorate function such that MeshNeurons are automatically skeletonized,
    the function is run function on the skeletons and changes are propagated
    back to the meshes.

    Parameters
    ----------
    method :    "subset" | "split" | "node_to_vertex" | "node_properties"
                What to do with the results:
                  - 'subset': subset MeshNeuron to what's left of the skeleton
                  - 'split': split MeshNeuron following the skeleton's splits
                  - 'node_to_vertex': map the returned node ID to the vertex IDs
                  - 'node_properties' map node properties to vertices (requires
                    `node_props` parameter)
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

    allowed_methods = ('subset', 'node_to_vertex', 'split', 'node_properties')
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

            # delayed import to avoid circular imports
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

            return x

        return wrapper

    return decorator


def lock_neuron(function):
    """Lock neuron while function is executed.

    This makes sure that temporary attributes aren't re-calculated as changes
    are being made.

    """
    @wraps(function)
    def wrapper(*args, **kwargs):
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


def is_url(x: str) -> bool:
    """Return True if str is URL.

    Examples
    --------
    >>> from navis.utils import is_url
    >>> is_url('www.google.com')
    False
    >>> is_url('http://www.google.com')
    True

    """
    parsed = urllib.parse.urlparse(x)

    if parsed.netloc and parsed.scheme:
        return True
    else:
        return False


def _type_of_script() -> str:
    """Return context (terminal, jupyter, colab, iPython) in which navis is run."""
    try:
        ipy_str = str(type(get_ipython()))  # type: ignore
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        elif 'colab' in ipy_str:
            return 'colab'
        else:  # if 'terminal' in ipy_str:
            return 'ipython'
    except BaseException:
        return 'terminal'


def is_jupyter() -> bool:
    """Test if navis is run in a Jupyter notebook.

    Also returns True if inside Google colaboratory!

    Examples
    --------
    >>> from navis.utils import is_jupyter
    >>> # If run outside a Jupyter environment
    >>> is_jupyter()
    False

    """
    return _type_of_script() in ('jupyter', 'colab')


def set_loggers(level: str = 'INFO'):
    """Set levels for all associated module loggers.

    Examples
    --------
    >>> from navis.utils import set_loggers
    >>> from navis import config
    >>> # Get current level
    >>> lvl = config.logger.level
    >>> # Set new level
    >>> set_loggers('INFO')
    >>> # Revert to old level
    >>> set_loggers(lvl)

    """
    config.logger.setLevel(level)


def set_pbars(hide: Optional[bool] = None,
              leave: Optional[bool] = None,
              jupyter: Optional[bool] = None) -> None:
    """Set global progress bar behaviors.

    Parameters
    ----------
    hide :      bool, optional
                Set to True to hide all progress bars.
    leave :     bool, optional
                Set to False to clear progress bars after they have finished.
    jupyter :   bool, optional
                Set to False to force using of classic tqdm even if in
                Jupyter environment.

    Returns
    -------
    Nothing

    Examples
    --------
    >>> from navis.utils import set_pbars
    >>> # Hide progress bars after finishing
    >>> set_pbars(leave=False)
    >>> # Never show progress bars
    >>> set_pbars(hide=True)
    >>> # Never use Jupyter widget progress bars
    >>> set_pbars(jupyter=False)

    """
    if isinstance(hide, bool):
        config.pbar_hide = hide

    if isinstance(leave, bool):
        config.pbar_leave = leave

    if isinstance(jupyter, bool):
        if jupyter:
            if not is_jupyter():
                logger.error('No Jupyter environment detected.')
            else:
                config.tqdm = config.tqdm_notebook
                config.trange = config.trange_notebook
        else:
            config.tqdm = config.tqdm_classic
            config.trange = config.trange_classic

    return


def unpack_neurons(x: Union[Iterable, 'core.NeuronList', 'core.NeuronObject'],
                   raise_on_error: bool = True
                   ) -> List['core.NeuronObject']:
    """Unpack neurons and returns a list of individual neurons.

    Examples
    --------
    This is mostly for doc tests:

    >>> from navis.utils import unpack_neurons
    >>> from navis.data import example_neurons
    >>> nl = example_neurons(3)
    >>> type(nl)
    <class 'navis.core.neuronlist.NeuronList'>
    >>> # Unpack list of neuronlists
    >>> unpacked = unpack_neurons([nl, nl])
    >>> type(unpacked)
    <class 'list'>
    >>> type(unpacked[0])
    <class 'navis.core.skeleton.TreeNeuron'>
    >>> len(unpacked)
    6

    """
    neurons: list = []

    if isinstance(x, (list, np.ndarray, tuple)):
        for l in x:
            neurons += unpack_neurons(l)
    elif isinstance(x, core.BaseNeuron):
        neurons.append(x)
    elif isinstance(x, core.NeuronList):
        neurons += x.neurons
    elif raise_on_error:
        raise TypeError(f'Unknown neuron format: "{type(x)}"')

    return neurons


def set_default_connector_colors(x: Union[List[tuple], Dict[str, tuple]]
                                 ) -> None:
    """Set/update default connector colors.

    Parameters
    ----------
    x :         dict
                New default connector colors. Can be::

                   {'cn_label': (r, g, b), ..}
                   {'cn_label': {'color': (r, g, b)}, ..}

    """
    if not isinstance(x, dict):
        raise TypeError(f'Expect dict, got "{type(x)}"')

    for k, v in x.items():
        if isinstance(v, dict):
            config.default_connector_colors[k].update(v)
        else:
            config.default_connector_colors[k]['color'] = v

    return


def parse_objects(x) -> Tuple['core.NeuronList',
                              List['core.Volume'],
                              List[np.ndarray],
                              List]:
    """Categorize objects e.g. for plotting.

    Returns
    -------
    Neurons :       navis.NeuronList
    Volume :        list of navis.Volume (trimesh.Trimesh will be converted)
    Points :        list of arrays
    Visuals :       list of vispy visuals

    Examples
    --------
    This is mostly for doc tests:

    >>> from navis.utils import parse_objects
    >>> from navis.data import example_neurons, example_volume
    >>> import numpy as np
    >>> nl = example_neurons(3)
    >>> v = example_volume('LH')
    >>> p = nl[0].nodes[['x', 'y', 'z']].values
    >>> n, vols, points, vis = parse_objects([nl, v, p])
    >>> type(n), len(n)
    (<class 'navis.core.neuronlist.NeuronList'>, 3)
    >>> type(vols), len(vols)
    (<class 'list'>, 1)
    >>> type(vols[0])
    <class 'navis.core.volumes.Volume'>
    >>> type(points), len(points)
    (<class 'list'>, 1)
    >>> type(points[0])
    <class 'numpy.ndarray'>
    >>> type(vis), len(points)
    (<class 'list'>, 1)

    """
    # Make sure this is a list.
    if not isinstance(x, list):
        x = [x]

    # If any list in x, flatten first
    if any([isinstance(i, list) for i in x]):
        # We need to be careful to preserve order because of colors
        y = []
        for i in x:
            y += i if isinstance(i, list) else [i]
        x = y

    # Collect neuron objects, make a single NeuronList and split into types
    neurons = core.NeuronList([ob for ob in x if isinstance(ob,
                                                            (core.BaseNeuron,
                                                             core.NeuronList))],
                              make_copy=False)

    # Collect visuals
    visuals = [ob for ob in x if 'vispy' in str(type(ob))]

    # Collect and parse volumes
    volumes = [ob for ob in x if not isinstance(ob, (core.BaseNeuron,
                                                     core.NeuronList))
                                 and is_mesh(ob)]
    # Add templatebrains
    volumes += [ob.mesh for ob in x if isinstance(ob, TemplateBrain)]
    # Converts any non-navis meshes into Volumes
    volumes = [core.Volume(v) if not isinstance(v, core.Volume) else v for v in volumes]

    # Collect dataframes with X/Y/Z coordinates
    dataframes = [ob for ob in x if isinstance(ob, pd.DataFrame)]
    if [d for d in dataframes if False in np.isin(['x', 'y', 'z'], d.columns)]:
        logger.warning('DataFrames must have x, y and z columns.')
    # Filter to and extract x/y/z coordinates
    dataframes = [d for d in dataframes if False not in [c in d.columns for c in ['x', 'y', 'z']]]
    dataframes = [d[['x', 'y', 'z']].values for d in dataframes]

    # Collect arrays
    arrays = [ob.copy() for ob in x if isinstance(ob, np.ndarray)]
    # Remove arrays with wrong dimensions
    if [ob for ob in arrays if ob.shape[1] != 3 and ob.shape[0] != 2]:
        logger.warning('Arrays need to be of shape (N, 3) for scatter or (2, N)'
                       ' for line plots.')
    arrays = [ob for ob in arrays if any(np.isin(ob.shape, [2, 3]))]

    points = dataframes + arrays

    return neurons, volumes, points, visuals


def make_url(baseurl, *args: str, **GET) -> str:
    """Generate URL.

    Parameters
    ----------
    *args
                Will be turned into the URL. For example::

                    >>> make_url('http://neuromorpho.org', 'neuron', 'fields')
                    'http://neuromorpho.org/neuron/fields'

    **GET
                Keyword arguments are assumed to be GET request queries
                and will be encoded in the url. For example::

                    >>> make_url('http://neuromorpho.org', 'neuron', 'fields',
                    ...          page=1)
                    'http://neuromorpho.org/neuron/fields?page=1'

    Returns
    -------
    url :       str


    Examples
    --------
    >>> from navis.utils import is_url, make_url
    >>> url = make_url('http://www.google.com', 'test', query='test')
    >>> url
    'http://www.google.com/test?query=test'
    >>> is_url(url)
    True

    """
    url = baseurl
    # Generate the URL
    for arg in args:
        arg_str = str(arg)
        joiner = '' if url.endswith('/') else '/'
        relative = arg_str[1:] if arg_str.startswith('/') else arg_str
        url = requests.compat.urljoin(url + joiner, relative)
    if GET:
        url += f'?{urllib.parse.urlencode(GET)}'
    return url
