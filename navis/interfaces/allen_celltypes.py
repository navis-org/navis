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

import numpy as np
import pandas as pd

from .. import config, utils
from ..core import Skeleton, NeuronList
from .base import optional_import

_swc = optional_import("allensdk.core.swc", pip="allensdk --no-deps")
_cell_types_cache = optional_import(
    "allensdk.core.cell_types_cache", pip="allensdk --no-deps"
)

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

__all__ = ['fetch_neurons']

_ctc = None


def _get_cache():
    """Return the (lazily built) allensdk cache.

    Deliberately not built at import time: `CellTypesCache()` writes a manifest
    into the current working directory, and merely importing navis must not
    leave files behind.
    """
    global _ctc
    if _ctc is None:
        _ctc = _cell_types_cache.CellTypesCache()
    return _ctc


def __getattr__(name):
    # `ctc` used to be a module-level global. Keep it resolvable, but built on
    # first access rather than on import.
    if name == "ctc":
        return _get_cache()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        morphology = _get_cache().get_reconstruction(id)
        neurons.append(_parse_morphology(morphology))
        neurons[-1].id = id

    return NeuronList(neurons)


def _parse_morphology(morphology):
    """Convert allensdk morphology to Skeleton."""
    assert isinstance(morphology, _swc.Morphology)

    nodes = []
    for n in morphology.compartment_list:
        nodes.append([n[k] for k in ['id', 'x', 'y', 'z', 'radius', 'parent', 'type']])
    nodes = pd.DataFrame(nodes, columns=['node_id', 'x', 'y', 'z', 'radius', 'parent_id', 'compartment'])
    nodes['compartment'] = nodes.compartment.map(COMPS).fillna('na')
    nodes = nodes.astype(DTYPES)

    # I'm guessing these are all in microns
    n = Skeleton(nodes, units='1 um')

    if getattr(morphology, 'soma', None):
        n.soma = morphology.soma['id']

    return n
