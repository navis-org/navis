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


from .. import core, config

# Set up logging
logger = config.logger


def find_soma(x):
    """ Tries finding a neuron's soma.

    Will use the ``.soma_detection_radius`` and ``.soma_label`` attribute of
    a neuron to search for the soma in the node table. If attributes don't
    exists, will fallback to defaults: ``None`` and ``1``, respectively.

    Parameters
    ----------
    x :         Neuron

    Returns
    -------
    node ID of potential soma(s)
    """

    if not isinstance(x, core.TreeNeuron):
        raise TypeError('Input must be neuron, not "{}"'.format(type(x)))

    soma_radius = getattr(x, 'soma_detection_radius', None)
    soma_label = getattr(x, 'soma_detection_label', 1)

    soma_nodes = x.nodes

    if not isinstance(soma_radius, type(None)):
        soma_nodes = soma_nodes[soma_nodes.radius >= soma_radius]

    if not isinstance(soma_label, type(None)) and 'label' in soma_nodes.columns:
        soma_nodes = soma_nodes[soma_nodes.label == soma_label]

    return soma_nodes.node_id.values
