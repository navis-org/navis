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

"""This module contains functions to read R data (.rda) files."""

import rdata
import warnings

import numpy as np
import pandas as pd

from typing import Any, Mapping, Union

from .. import config, utils, core

__all__ = ['read_rda']

# Set up logging
logger = config.get_logger(__name__)


def read_rda(f: str,
             combine: bool = True,
             neurons_only: bool = True,
             **kwargs) -> 'core.NeuronList':
    """Read objects from nat R data (.rda) file.

    Currently supports parsing neurons, dotprops and mesh3d. Note that this is
    rather slow and I do not recommend doing this for large collections of
    neurons. For large scale conversion I recommend using the R interface
    (``navis.interfaces.r``, see online tutorials) via ``rpy2``.

    Parameters
    ----------
    f :                 str
                        Filepath.
    combined :          bool
                        What to do if there are multiple neuronlists contained
                        in the RDA files. By default, we will combine them into
                        a single NeuronList but you can also choose to keep them
                        as separate neuronlists.
    neurons_only :      bool
                        Whether to only parse and return neurons and dotprops
                        found in the RDA file.
    **kwargs
                        Keyword arguments passed to the construction of
                        `Tree/MeshNeuron/Dotprops`. You can use this to e.g. set
                        meta data.

    Returns
    -------
    navis.NeuronList
                        If ``combine=True`` and ``neurons_only=True`` returns
                        a single NeuronList with the parsed neurons.
    dict
                        If ``combine=False`` or ``neurons_only=False`` returns
                        a dictionary with the original R object name as key and
                        the parsed object as value.

    """
    # Parse the file
    parsed = rdata.parser.parse_file(f)

    # Now convert to Python objects
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        converted = rdata.conversion.convert(parsed, CLASS_MAP_EXT)

    # Some clean-up
    for k, v in converted.items():
        # Convert single neurons to neuronlist
        if isinstance(v, core.BaseNeuron):
            converted[k] = core.NeuronList(v)
        # Give volumes a name
        elif isinstance(v, core.Volume):
            converted[k].name = k

    if combine:
        nl = core.NeuronList([n for n in converted.values() if isinstance(n, core.NeuronList)])
        if nl:
            converted = {k: v for k, v in converted.items() if not isinstance(v, core.NeuronList)}
            converted['neurons'] = nl

    if neurons_only:
        if combine:
            converted = converted['neurons']
        else:
            converted = {k: v for k, v in converted.items() if isinstance(v, core.NeuronList)}

    return converted


def neuronlist_constructor(obj: Any,
                           attrs: Mapping[Union[str, bytes], Any],
                           ) -> 'core.NeuronList':
    """Convert nat neuronlists to navis NeuronLists."""
    # Set IDs
    neurons = []
    for k, n in obj.items():
        if isinstance(n, (core.BaseNeuron, core.NeuronList)):
            n.id = k
            neurons.append(n)
        else:
            logger.warning(f'Unexpected object in neuronlist: {type(n)}. '
                           'Possible parsing error.')

    # Turn into NeuronList
    nl = core.NeuronList(neurons)

    # Now parse extra attributes DataFrame
    df = attrs.get('df', None)
    if isinstance(df, pd.DataFrame):
        # Make sure we have still the correct order
        nl = nl.idx[attrs['names']]

        for col in df:
            # Skip non-string columns
            if not isinstance(col, str):
                continue

            # Skip some columns
            if col.lower() in ['type', 'idx']:
                continue
            if col.lower() in nl[0].__dict__.keys():
                continue

            for n, v in zip(nl, df[col].values):
                # Register
                n._register_attr(col.lower(), v)

    return nl


def dotprops_constructor(obj: Any,
                         attrs: Mapping[Union[str, bytes], Any],
                         ) -> 'core.Dotprops':
    """Convert nat dotprops to navis Dotprops."""
    pts = np.asarray(obj.pop('points'))
    vect = np.asarray(obj.pop('vect'))
    alpha = np.asarray(obj.pop('alpha'))
    k = int(attrs.get('k', 1)[0])
    file = attrs.get('file', [None])[0]

    return core.Dotprops(points=pts, k=k, alpha=alpha, vect=vect, file=file)


def volume_constructor(obj: Any,
                       attrs: Mapping[Union[str, bytes], Any],
                       ) -> 'core.Volume':
    """Convert e.g. mesh3d to navis Volume."""
    if 'vb' in obj and 'it' in obj:
        verts = np.asarray(obj.pop('vb'))[:3, :].T
        faces = np.asarray(obj.pop('it')).T - 1
        return core.Volume(vertices=verts, faces=faces)
    elif 'Vertices' in obj and "Regions" in obj:
        verts = obj['Vertices'][['X', 'Y', 'Z']].values

        # If only one region
        if len(obj['Regions']) == 1:
            region = list(obj['Regions'].keys())[0]
            faces = obj['Regions'][region][['V1', 'V2', 'V3']].values - 1
            return core.Volume(vertices=verts, faces=faces)
        else:
            volumes = []
            for r in obj['Regions']:
                faces = obj['Regions'][r][['V1', 'V2', 'V3']].values - 1
                volumes.append(core.Volume(vertices=verts, faces=faces, name=r))
            return volumes
    else:
        logger.warning('Unable to construct Volume from R object of type '
                       f'"{attrs["class"]}". Returning raw data')
        return obj


def neuron_constructor(obj: Any,
                       attrs: Mapping[Union[str, bytes], Any],
                       ) -> 'core.TreeNeuron':
    """Convert nat neuron/catmaidneuron to navis TreeNeuron."""
    # Data to skip
    DO_NOT_USE = ['nTrees', 'SegList', 'NumPoints', 'StartPoint', 'EndPoints',
                  'BranchPoints', 'NumSegs']

    # Construct neuron from just the nodes
    n = core.TreeNeuron(obj.pop('d'))

    # R uses diameter, not radius - let's fix that
    if 'radius' in n.nodes.columns:
        has_rad = n.nodes.radius.fillna(0) > 0
        n.nodes.loc[has_rad, 'radius'] = n.nodes.loc[has_rad, 'radius'] / 2

    # If this is a CATMAID neuron, we assume it's in nanometers
    if 'catmaidneuron' in attrs.get('class', []):
        n.units = 'nm'

    # Reuse ID
    if 'skid' in obj:
        skid = obj.pop('skid')
        if utils.is_iterable(skid):
            n.id = skid[0]
        else:
            n.id = skid

    # Try attaching other data
    for k, v in obj.items():
        if k in DO_NOT_USE:
            continue
        try:
            setattr(n, k, v)
        except BaseException:
            pass

    return n


CLASS_MAP_EXT = {**rdata.conversion.DEFAULT_CLASS_MAP,
                 "neuronlist": neuronlist_constructor,
                 "neuron": neuron_constructor,
                 "mesh3d": volume_constructor,
                 "hxsurf": volume_constructor,
                 "dotprops": dotprops_constructor}
