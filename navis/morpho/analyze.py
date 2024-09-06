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

import pint
import warnings

import numpy as np

from .. import config, core

from typing import Sequence

# Set up logging
logger = config.get_logger(__name__)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


def find_soma(x: 'core.TreeNeuron') -> Sequence[int]:
    """Try finding a neuron's soma.

    Will use the `.soma_detection_radius` and `.soma_detection_label`
    attribute of a neuron to search for the soma in the node table.

    If attributes don't exists, will fallback to defaults: `None` and
    `1`, respectively.

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
        raise TypeError(f'Input must be TreeNeuron, not "{type(x)}"')

    soma_radius = getattr(x, 'soma_detection_radius', None)
    soma_label = getattr(x, 'soma_detection_label', None)

    check_labels = not isinstance(soma_label, type(None)) and 'label' in x.nodes.columns
    check_radius = not isinstance(soma_radius, type(None))

    # If no label or radius is given, return empty array
    if not check_labels and not check_radius:
        return np.array([], dtype=x.nodes.node_id.values.dtype)

    # Note to self: I've optimised the s**t out of this function
    # The reason reason why we're using a mask and this somewhat
    # convoluted logic is to avoid having to subset the node table
    # because that's really slow.

    # Start with a mask that includes all nodes
    mask = np.ones(len(x.nodes), dtype=bool)

    if check_radius:
        # When checking for radii, we use an empty mask and fill it
        # with nodes that have a large enough radius
        mask[:] = False

        # Drop nodes that don't have a radius
        radii = x.nodes.radius.values
        has_radius = ~np.isnan(radii)

        # Filter further to nodes that have a large enough radius
        if has_radius.any():
            if isinstance(soma_radius, pint.Quantity):
                if isinstance(x.units, (pint.Quantity, pint.Unit)) and \
                   not x.units.dimensionless and \
                   not isinstance(x.units._magnitude, np.ndarray) \
                   and x.units != soma_radius:  # only convert if units are different
                    # Do NOT remove the .values here -> otherwise conversion to units won't work
                    is_large = radii * x.units >= soma_radius
                else:
                    # If neurons has no units or if units are non-isotropic,
                    # assume they are the same as the soma radius
                    is_large = radii >= soma_radius._magnitude
            else:
                is_large = radii >= soma_radius

            # Mark nodes that have a large enough radius
            mask[is_large] = True

    # See if we (also) need to check for a specific label
    if check_labels:
        # Important: we need to use np.asarray here because the `label` column
        # can be categorical in which case a `soma_nodes.label.astype(str)` might
        # throw annoying runtime warnings
        soma_node_ids = x.nodes.node_id.values[mask]
        soma_node_labels = np.asarray(x.nodes.label.values[mask]).astype(str)

        return soma_node_ids[soma_node_labels == str(soma_label)]
    # If no labels to check we can return the mask directly
    else:
        return x.nodes.node_id.values[mask]

