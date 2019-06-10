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

import json

import pandas as pd

from .. import config, core

# Set up logging
logger = config.logger


def neuron2json(x: 'core.NeuronObject', **kwargs) -> str:
    """ Generate JSON formatted ``str`` respresentation of TreeNeuron/List.

    Nodes and connectors are serialised using pandas' ``to_json()``. Most
    other items in the neuron's __dict__ are serialised using
    ``json.dumps()``. Properties not serialised: `.graph`, `.igraph`.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList
    **kwargs
                Parameters passed to ``json.dumps()`` and
                ``pandas.DataFrame.to_json()``.

    Returns
    -------
    str

    See Also
    --------
    :func:`~navis.json2neuron`
                Read json back into navis neurons.

    """

    if not isinstance(x, (core.TreeNeuron, core.NeuronList)):
        raise TypeError('Unable to convert data of type "{0}"'.format(type(x)))

    if isinstance(x, core.TreeNeuron):
        x = core.NeuronList([x])

    data = []
    for n in x:
        this_data = {'skeleton_id': n.skeleton_id}

        if 'nodes' in n.__dict__:
            this_data['nodes'] = n.nodes.to_json()

        if 'connectors' in n.__dict__:
            this_data['connectors'] = n.connectors.to_json(**kwargs)

        for k in n.__dict__:
            if k in ['nodes', 'connectors', 'graph', 'igraph', '_remote_instance',
                     'segments', 'small_segments', 'nodes_geodesic_distance_matrix',
                     'dps', 'simple']:
                continue
            try:
                this_data[k] = n.__dict__[k]
            except BaseException:
                logger.error('Lost attribute "{0}"'.format(k))

        data.append(this_data)

    return json.dumps(data, **kwargs)


def json2neuron(s: str, **kwargs) -> 'core.NeuronList':
    """ Load neuron from JSON string.

    Parameters
    ----------
    s :         str
                JSON-formatted string.
    **kwargs
                Parameters passed to ``json.loads()`` and
                ``pandas.DataFrame.read_json()``.

    Returns
    -------
    :class:`~navis.NeuronList`

    See Also
    --------
    :func:`~navis.neuron2json`
                Turn neuron into json.

    """

    if not isinstance(s, str):
        raise TypeError('Need str, got "{0}"'.format(type(s)))

    data = json.loads(s, **kwargs)

    nl = core.NeuronList([])

    for n in data:
        # Make sure we have all we need
        REQUIRED = ['skeleton_id']

        missing = [p for p in REQUIRED if p not in n]

        if missing:
            raise ValueError('Missing data: {0}'.format(','.join(missing)))

        cn = core.TreeNeuron(int(n['skeleton_id']))

        if 'nodes' in n:
            cn.nodes = pd.read_json(n['nodes'])
            cn.connectors = pd.read_json(n['connectors'])

        for key in n:
            if key in ['skeleton_id', 'nodes', 'connectors']:
                continue
            setattr(cn, key, n[key])

        nl += cn

    return nl
