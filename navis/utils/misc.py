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
#
#    You should have received a copy of the GNU General Public License
#    along


import pandas as pd
import numpy as np

from .. import core, config

# Set up logging
logger = config.logger


def _type_of_script():
    """ Returns context in which navis is run. """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except BaseException:
        return 'terminal'


def is_jupyter():
    """ Test if navis is run in a Jupyter notebook."""
    return _type_of_script() == 'jupyter'


def set_loggers(level='INFO'):
    """Helper function to set levels for all associated module loggers."""
    config.logger.setLevel(level)


def set_pbars(hide=None, leave=None, jupyter=None):
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


def unpack_neurons(x, raise_on_error=True):
    """ Unpacks neurons and returns a list of individual neurons.
    """

    neurons = []

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


def set_default_connector_colors(x):
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


def parse_objects(x):
    """ Helper class to categorize objects e.g. for plotting.

    Returns
    -------
    TreeNeurons :   navis.NeuronList
    Dotprops :      pd.DataFrame
    Volumes :       list
    Points :        list of arrays
    Visuals :       list of vispy visuals
    """

    # Make sure this is a list.
    if not isinstance(x, list):
        x = [x]

    # Collect neuron objects and collate to single Neuronlist
    neuron_obj = [ob for ob in x if isinstance(ob,
                                               (core.TreeNeuron,
                                                core.NeuronList))]
    skdata = core.NeuronList(neuron_obj, make_copy=False)

    # Collect visuals
    visuals = [ob for ob in x if 'vispy' in str(type(ob))]

    # Collect dotprops
    dotprops = [ob for ob in x if isinstance(ob, core.Dotprops)]

    if len(dotprops) == 1:
        dotprops = dotprops[0]
    elif len(dotprops) == 0:
        dotprops = core.Dotprops()
        dotprops['gene_name'] = []
    elif len(dotprops) > 1:
        dotprops = pd.concat(dotprops)

    # Collect and parse volumes
    volumes = [ob for ob in x if isinstance(ob, (core.Volume, str))]

    # Collect dataframes with X/Y/Z coordinates
    # Note: dotprops and volumes are instances of pd.DataFrames
    dataframes = [ob for ob in x if isinstance(ob, pd.DataFrame) and
                  not isinstance(ob, (core.Dotprops, core.Volume))]
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
