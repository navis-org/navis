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
import numpy as np

from .. import config, core

# Set up logging
logger = config.logger


def neuron2json(x: 'core.NeuronObject', **kwargs) -> str:
    """Generate JSON formatted ``str`` respresentation of TreeNeuron/List.

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
        raise TypeError(f'Unable to convert data of type "{type(x)}"')

    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList([x])

    data = []
    for n in x:
        this_data = {'id': n.id}
        for k, v in n.__dict__.items():
            if not isinstance(k, str):
                continue
            if k.startswith('_') and k not in ['_nodes', '_connectors']:
                continue

            if isinstance(v, pd.DataFrame):
                this_data[k] = v.to_json()
            elif isinstance(v, np.ndarray):
                this_data[k] = v.tolist()
            else:
                this_data[k] = v

        data.append(this_data)

    return json.dumps(data, **kwargs)


def json2neuron(s: str, **kwargs) -> 'core.NeuronList':
    """Load neuron from JSON string.

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

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> js = navis.neuron2json(n)
    >>> n2 = navis.json2neuron(js)

    """
    if not isinstance(s, str):
        raise TypeError(f'Expected str, got "{type(s)}"')

    data = json.loads(s, **kwargs)

    nl = core.NeuronList([])

    for n in data:
        cn = core.TreeNeuron(None)

        if '_nodes' in n:
            try:
                cn._nodes = pd.read_json(n['_nodes'])
            except ValueError:
                cn._connectors = None

        if '_connectors' in n:
            try:
                cn._connectors = pd.read_json(n['_connectors'])
            except ValueError:
                cn._connectors = None

        for key in n:
            if key in ['_nodes', '_connectors']:
                continue
            setattr(cn, key, n[key])

        nl += cn

    return nl
