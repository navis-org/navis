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

import os

import networkx as nx
import pandas as pd

from typing import Union, Optional
from typing_extensions import Literal

from ..core.volumes import Volume
from ..core.neuronlist import NeuronList
from ..core.neurons import TreeNeuron, MeshNeuron
from ..io import read_swc
from ..graph import nx2neuron

__all__ = ['example_neurons', 'example_volume']

fp = os.path.dirname(__file__)

gml_path = os.path.join(fp, 'gml')
swc_path = os.path.join(fp, 'swc')
vols_path = os.path.join(fp, 'volumes')
obj_path = os.path.join(fp, 'obj')
syn_path = os.path.join(fp, 'synapses')

gml = sorted([f for f in os.listdir(gml_path) if f.endswith('.gml')])
swc = sorted([f for f in os.listdir(swc_path) if f.endswith('.swc')])
vols = sorted([f for f in os.listdir(vols_path) if f.endswith('.obj')])
obj = sorted([f for f in os.listdir(obj_path) if f.endswith('.obj')])
syn = sorted([f for f in os.listdir(syn_path) if f.endswith('.csv')])

NeuronObject = Union[TreeNeuron, MeshNeuron, NeuronList]


def example_neurons(n: Optional[int] = None,
                    kind:  Union[Literal['mesh'],
                                 Literal['skeleton'],
                                 Literal['mix']] = 'skeleton',
                    synapses: bool = True,
                    source: Union[Literal['swc'],
                                  Literal['gml']] = 'swc',
                    ) -> NeuronObject:
    """Load example neuron(s).

    These example neurons are skeletons and meshes of the same olfactory
    projection neurons from the DA1 glomerulus which have been automatically
    segmented in the Janelia hemibrain data set [1]. See also
    `https://neuprint.janelia.org`.

    Coordinates are in voxels which equal 8 x 8 x 8 nanometers.

    Parameters
    ----------
    n :         int | None, optional
                Number of neurons to return. If None, will return all available
                example neurons. Can never return more than the maximum number
                of example neurons.
    kind :      "skeleton" | "mesh" | "mix"
                Example neurons What kind of neurons to return.
    synapses :  bool,
                If True, will also load synapses.
    source :    'swc' | 'gml', optional
                Only relevant for skeletons. Skeletons can be generated from SWC
                files or GML graphs (this is really only used for testing).

    Returns
    -------
    TreeNeuron
                If ``n=1`` and ``kind='skeleton'``.
    MeshNeuron
                If ``n=1`` and ``kind='mesh'``.
    NeuronList
                List of the above neuron types if ``n>1``.

    References
    ----------
    [1] Louis K. Scheffer et al., bioRxiv. 2020. doi: https://doi.org/10.1101/2020.04.07.030213
    A Connectome and Analysis of the Adult Drosophila Central Brain.

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

    if kind not in ['skeleton', 'mesh', 'mix']:
        raise ValueError(f'Unknown value for `kind`: "{kind}"')

    if kind == 'mix':
        n_mesh = round(n/2)
        n_skel = n - n_mesh
    else:
        n_mesh = n_skel = n

    nl = []
    if kind in ['skeleton', 'mix']:
        if source == 'gml':
            graphs = [nx.read_gml(os.path.join(gml_path, g)) for g in gml[:n_skel]]
            nl += [nx2neuron(g,
                             units='8 nm',
                             id=int(f.split('.')[0])) for f, g in zip(gml, graphs)]
        elif source == 'swc':
            nl += [read_swc(os.path.join(swc_path, f),
                            units='8 nm',
                            id=int(f.split('.')[0])) for f in swc[:n_skel]]
        else:
            raise ValueError(f'Source must be "swc" or "gml", not "{source}"')

    if kind in ['mesh', 'mix']:
        files = [os.path.join(obj_path, f) for f in obj[:n_mesh]]
        nl += [MeshNeuron(fp,
                          units='8 nm',
                          name=f.split('.')[0],
                          id=int(f.split('.')[0])) for f, fp in zip(obj, files)]

    if synapses:
        for n in nl:
            n.connectors = pd.read_csv(os.path.join(syn_path, f'{n.id}.csv'))

            if isinstance(n, MeshNeuron):
                n._connectors.drop('node_id', axis=1, inplace=True)

    if len(nl) == 1:
        return nl[0]
    return NeuronList(nl)


def example_volume(name: str) -> Volume:
    """Load an example volume.

    Volumes are in hemibrain space which means coordinates are in voxels
    at 8 x 8 x 8 nanometers/voxel.

    Parameters
    ----------
    name :      str
                Name of available volume. Currently available::

                  "LH" = lateral horn in hemibrain space
                  "neuropil" = neuropil in hemibrain space

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
    if not name.endswith('.obj'):
        name += '.obj'

    if name not in vols:
        raise ValueError(f'No volume named "{name}". Available volumes: {",".join(vols)}')

    vol = Volume.from_file(os.path.join(vols_path, name),
                           name=name.split('.')[0])

    return vol
