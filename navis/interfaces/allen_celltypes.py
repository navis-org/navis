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

"""Interface with Allen cell type atlas: https://celltypes.brain-map.org/."""

from textwrap import dedent

try:
    import allensdk
    from allensdk.core.cell_types_cache import CellTypesCache
except ImportError:
    msg = dedent("""
          allensdk library not found. Please install using pip:

                pip install allensdk --no-deps

          """)
    raise ImportError(msg)
except BaseException:
    raise

import numpy as np
import pandas as pd

from .. import config, utils
from ..core import TreeNeuron, NeuronList

logger = config.get_logger(__name__)
dataset = None

DTYPES = {
    'node_id': np.int32,
    'parent_id': np.int32,
    'compartment': 'category',
    'x': np.float32,
    'y': np.float32,
    'z': np.float32,
    'radius': np.float32,
}
COMPS = {
    1: 'soma',
    2: 'axon',
    3: 'dendrites',
    4: 'apical dendrites'
}
SWC_FILE_TYPE = '3DNeuronReconstruction'

ctc = CellTypesCache()


__all__ = ['fetch_neurons']


def fetch_neurons(ids):
    """Fetch given neurons.

    Parameters
    ----------
    ids :   int | iterable
            IDs of the neurons to fetch skeletons for.

    Returns
    -------
    navis.NeuronList

    """
    ids = utils.make_iterable(ids, force_type=int)

    neurons = []
    for id in config.tqdm(ids,
                          desc='Fetching',
                          disable=config.pbar_hide,
                          leave=config.pbar_leave
                          ):
        morphology = ctc.get_reconstruction(id)
        neurons.append(_parse_morphology(morphology))
        neurons[-1].id = id

    return NeuronList(neurons)


def _parse_morphology(morphology):
    """Convert allensdk morphology to TreeNeuron."""
    assert isinstance(morphology, allensdk.core.swc.Morphology)

    nodes = []
    for n in morphology.compartment_list:
        nodes.append([n[k] for k in ['id', 'x', 'y', 'z', 'radius', 'parent', 'type']])
    nodes = pd.DataFrame(nodes, columns=['node_id', 'x', 'y', 'z', 'radius', 'parent_id', 'compartment'])
    nodes['compartment'] = nodes.compartment.map(COMPS).fillna('na')
    nodes = nodes.astype(DTYPES)

    # I'm guessing these are all in microns
    n = TreeNeuron(nodes, units='1 um')

    if getattr(morphology, 'soma', None):
        n.soma = morphology.soma['id']

    return n
