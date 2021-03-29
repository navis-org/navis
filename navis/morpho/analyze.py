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

import pint
import warnings

import numpy as np

from .. import config, core

from typing import Sequence

# Set up logging
logger = config.logger

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


def find_soma(x: 'core.TreeNeuron') -> Sequence[int]:
    """Try finding a neuron's soma.

    Will use the ``.soma_detection_radius`` and ``.soma_detection_label``
    attribute of a neuron to search for the soma in the node table.

    If attributes don't exists, will fallback to defaults: ``None`` and
    ``1``, respectively.

    Parameters
    ----------
    x :         Neuron

    Returns
    -------
    Node ID(s) of potential somata.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> navis.find_soma(n)
    array([4177], dtype=int32)

    """
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Input must be neuron, not "{type(x)}"')

    soma_radius = getattr(x, 'soma_detection_radius', None)
    soma_label = getattr(x, 'soma_detection_label', 1)

    soma_nodes = x.nodes

    if not isinstance(soma_radius, type(None)):
        # Drop nodes that don't have a radius
        soma_nodes = soma_nodes.loc[~soma_nodes.radius.isnull()]

        # Filter further to nodes that have a large enough radius
        if not soma_nodes.empty:
            if isinstance(soma_radius, pint.Quantity):
                if isinstance(x.units, (pint.Quantity, pint.Unit)) and \
                   not x.units.dimensionless:
                    # Do NOT remove the .values here -> otherwise conversion to units won't work
                    is_large = soma_nodes.radius.values * x.units >= soma_radius
                else:
                    # If neurons has no units, assume they are the same as the soma radius
                    is_large = soma_nodes.radius.values * soma_radius.units >= soma_radius
            else:
                is_large = soma_nodes.radius >= soma_radius

            soma_nodes = soma_nodes[is_large]

    if not isinstance(soma_label, type(None)) and 'label' in soma_nodes.columns:
        soma_nodes = soma_nodes[soma_nodes.label.astype(str) == str(soma_label)]

    return soma_nodes.node_id.values
