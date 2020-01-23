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
import os

import networkx as nx

from typing import Union, Optional
from typing_extensions import Literal

from ..core.volumes import Volume
from ..core.neuronlist import NeuronList
from ..core.neurons import TreeNeuron
from ..io import *
from ..graph import *

__all__ = ['example_neurons', 'example_volume']

fp = os.path.dirname(__file__)

gml_path = os.path.join(fp, 'gml')
swc_path = os.path.join(fp, 'swc')
vols_path = os.path.join(fp, 'volumes')

gml = [f for f in os.listdir(gml_path) if f.endswith('.gml')]
swc = [f for f in os.listdir(swc_path) if f.endswith('.swc')]
vols = [f for f in os.listdir(vols_path) if f.endswith('.json')]

NeuronObject = Union[TreeNeuron, NeuronList]


def example_neurons(n: Optional[int] = None,
                    source: Union[Literal['swc'],
                                  Literal['gml']] = 'swc'
                    ) -> NeuronObject:
    """ Load example neuron(s).

    These example neurons are olfactory projection neurons from the DA2
    glomerulus that have been manually reconstructed from an EM volume of an
    entire fruit fly brain (see Zheng et al., Cell 2018).

    Parameters
    ----------
    n :         int, None, optional
                Number of neurons to return. If None, will return all example
                neurons.
    source :    'swc' | 'gml', optional
                Neurons can be generated from SWC files or GML graphs. This
                is only relevant for testing.

    Returns
    -------
    TreeNeuron
                If ``n=1``.
    NeuronList
                If ``n>1``.


    Examples
    --------
    Load a single neuron
    >>> import navis
    >>> n = navis.example_neurons(n=1)

    Load all example neurons
    >>> nl = navis.example_neurons()

    """

    if not isinstance(n, (int, type(None))):
        raise TypeError(f'Expected int or None, got "{type(n)}"')

    if isinstance(n, int) and n < 1:
        raise ValueError("Unable to return less than 1 neuron.")

    if source == 'gml':
        graphs = [nx.read_gml(os.path.join(gml_path, g)) for g in gml[:n]]
        nl = NeuronList([nx2neuron(g) for g in graphs], units='nm')
    elif source == 'swc':
        nl = NeuronList([from_swc(os.path.join(swc_path, f), units='nm') for f in swc[:n]])
    else:
        raise ValueError(f'Source must be "swc" or "gml", not "{source}"')

    if n == 1:
        return nl[0]
    return nl


def example_volume(name: str) -> Volume:
    """ Load an example volume.

    Parameters
    ----------
    name :      str
                Name of available volume. Currently available::
                  "LH" = lateral horn in FAFB v14 space
                  'neuropil' = neuropil of FAFB v14 volume


    Returns
    -------
    navis.Volume

    Examples
    --------
    Load LH volume
    >>> import navis
    >>> lh = navis.example_volume('LH')
    """
    if not isinstance(name, str):
        raise TypeError(f'Expected string, got "{type(name)}"')

    # Force lower case
    name = name.lower()

    # Make sure extension is correct
    if not name.endswith('.json'):
        name += '.json'

    if name not in vols:
        raise ValueError(f'No volume named "{name}". Available volumes: {",".join(vols)}')

    with open(os.path.join(vols_path, name), 'r') as f:
        data = json.load(f)
        return Volume(**data)
