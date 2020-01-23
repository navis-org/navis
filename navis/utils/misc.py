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

import pandas as pd
import numpy as np
import requests
import urllib

from typing import Optional, Union, List, Iterable, Dict, Tuple

from .. import config, core

# Set up logging
logger = config.logger


def is_url(x: str) -> bool:
    """ Returns True if str is URL.

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
    """ Returns context in which navis is run. """
    try:
        ipy_str = str(type(get_ipython()))  # type: ignore
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        else:  # if 'terminal' in ipy_str:
            return 'ipython'
    except BaseException:
        return 'terminal'


def is_jupyter() -> bool:
    """ Test if navis is run in a Jupyter notebook.

    Examples
    --------
    >>> from navis.utils import is_jupyter
    >>> # If run outside a Jupyter environment
    >>> is_jupyter()
    False
    """
    return _type_of_script() == 'jupyter'


def set_loggers(level: str = 'INFO'):
    """Helper function to set levels for all associated module loggers.

    Examples
    --------
    >>> from navis.utils import set_loggers
    >>> from navis import config
    >>> # Get current level
    >>> lvl = config.logger.getLevel()
    >>> # Set new level
    >>> set_loggers('INFO')
    >>> # Revert to old level
    >>> set_loggers(lvl)
    """
    config.logger.setLevel(level)


def set_pbars(hide: Optional[bool] = None,
              leave: Optional[bool] = None,
              jupyter: Optional[bool] = None) -> None:
    """ Set global progress bar behaviors.

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
    >>> set_pbars(juyter=False)
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
                config.trange = config.tnrange
        else:
            config.tqdm = config.tqdm_classic
            config.trange = config.trange_classic

    return


def unpack_neurons(x: Union[Iterable, 'core.NeuronList', 'core.TreeNeuron'],
                   raise_on_error: bool = True
                   ) -> List['core.TreeNeuron']:
    """ Unpacks neurons and returns a list of individual neurons.

    Examples
    --------
    This is mostly for doc tests:

    >>> from navis.utils import unpack_neurons
    >>> from navis.data import example_neurons
    >>> nl = example_neurons(3)
    >>> type(nl)
    navis.core.neuronlist.NeuronList
    >>> # Unpack list of neuronlists
    >>> unpacked = unpack_neurons([nl, nl])
    >>> type(unpacked)
    list
    >>> type(unpacked[0])
    navis.core.neurons.TreeNeuron
    >>> len(unpacked)
    6
    """

    neurons: list = []

    if isinstance(x, (list, np.ndarray, tuple)):
        for l in x:
            neurons += unpack_neurons(l)
    elif isinstance(x, core.TreeNeuron):
        neurons.append(x)
    elif isinstance(x, core.NeuronList):
        neurons += x.neurons
    elif raise_on_error:
        raise TypeError(f'Unknown neuron format: "{type(x)}"')

    return neurons


def set_default_connector_colors(x: Union[List[tuple], Dict[str, tuple]]
                                 ) -> None:
    """ Set default connector colors.

    Parameters
    ----------
    x :         list-like | dict
                New default connector colors. Can be::

                   list : [(r, g, b), (r, g, b), ..]
                   dict : {'cn_label': (r, g, b), ..}
    """

    if not isinstance(x, (dict, list, np.ndarray)):
        raise TypeError(f'Expect dict, list or numpy array, got "{type(x)}"')

    config.default_connector_colors = x

    return


def parse_objects(x) -> Tuple['core.NeuronList',
                              pd.DataFrame,
                              List['core.Volume'],
                              List[np.ndarray],
                              List]:
    """ Helper class to categorize objects e.g. for plotting.

    Returns
    -------
    TreeNeurons :   navis.NeuronList
    Dotprops :      pd.DataFrame
    Volumes :       list
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
    >>> n, dps, vols, points, vis = parse_objects([nl, v, p])
    >>> type(n), len(n)
    (navis.core.neuronlist.NeuronList, 3)
    >>> type(dps), len(dps)
    (navis.core.dotprops.Dotprops, 0)
    >>> type(vols), len(vols)
    (list, 1)
    >>> type(vols[0])
    navis.core.volumes.Volume
    >>> type(points), len(points)
    (list, 1)
    >>> type(points[0])
    numpy.ndarray
    >>> type(vis), len(points)
    (list, 1)
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

    # Collect neuron objects and collate to single Neuronlist
    neuron_obj = [ob for ob in x if isinstance(ob,
                                               (core.TreeNeuron,
                                                core.NeuronList))]
    skdata = core.NeuronList(neuron_obj, make_copy=False)

    # Collect visuals
    visuals = [ob for ob in x if 'vispy' in str(type(ob))]

    # Collect dotprops
    dps = [ob for ob in x if isinstance(ob, core.Dotprops)]

    if len(dps) == 1:
        dotprops = dps[0]
    elif len(dps) == 0:
        dotprops = core.Dotprops()
        dotprops['gene_name'] = []
    else:
        dotprops = pd.concat(dps)

    # Collect and parse volumes
    volumes = [ob for ob in x if isinstance(ob, core.Volume)]

    # Collect dataframes with X/Y/Z coordinates
    # Note: dotprops and volumes are instances of pd.DataFrames
    dataframes = [ob for ob in x if isinstance(ob, pd.DataFrame)
                  and not isinstance(ob, (core.Dotprops, core.Volume))]
    if [d for d in dataframes if False in [c in d.columns for c in ['x', 'y', 'z']]]:
        logger.warning('DataFrames must have x, y and z columns.')
    # Filter to and extract x/y/z coordinates
    dataframes = [d for d in dataframes if False not in [c in d.columns for c in ['x', 'y', 'z']]]
    dataframes = [d[['x', 'y', 'z']].values for d in dataframes]

    # Collect arrays
    arrays = [ob.copy() for ob in x if isinstance(ob, np.ndarray)]
    # Remove arrays with wrong dimensions
    if [ob for ob in arrays if ob.shape[1] != 3]:
        logger.warning('Point objects need to be of shape (n,3).')
    arrays = [ob for ob in arrays if ob.shape[1] == 3]

    points = dataframes + arrays

    return skdata, dotprops, volumes, points, visuals


def make_url(baseurl, *args: str, **GET) -> str:
    """Generate URL.

    Parameters
    ----------
    *args
                Will be turned into the URL. For example::

                    >>> make_url('http://neuromorpho.org', 'neuron', 'fields')
                    'http://neuromorpho.org/api/neuron/fields'

    **GET
                Keyword arguments are assumed to be GET request queries
                and will be encoded in the url. For example::

                    >>> make_url('http://neuromorpho.org', 'neuron', 'fields',
                    ...          page = 1)
                    'http://neuromorpho.org/api/neuron/fields?page=1'

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
        url += '?{}'.format(urllib.parse.urlencode(GET))
    return url
