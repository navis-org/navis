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

""" Interface with neuprint. This module is wrapping neuprint-python
(https://github.com/connectome-neuprint/neuprint-python) and adding some
navis-specific functions.
"""

from textwrap import dedent

try:
    from neuprint import *
    from neuprint.client import inject_client
except ImportError:
    msg = dedent("""
          neuprint library not found. Please install using pip:

                pip install neuprint-python

          """)
    raise ImportError(msg)
except BaseException:
    raise

import io

from tqdm import tqdm

import numpy as np
import pandas as pd

from ..core import Volume, TreeNeuron, NeuronList
from ..graph import neuron2KDTree

# Define some integer types
int_types = (int, np.int32, np.int64, np.int, np.int0)


@inject_client
def fetch_roi(roi, *, client=None):
    """Fetch given ROI.

    Requires `trimesh <https://trimsh.org/>`_::

        pip3 install navis trimesh

    Parameters
    ----------
    roi :           str
                    Name of an ROI.
    client :        neuprint.Client, optional
                    If ``None`` will try using global client.

    Returns
    -------
    navis.Volume

    """
    if not isinstance(roi, str):
        raise TypeError(f'Expect ROI name as string, got "{type(x)}"')

    try:
        import trimesh
    except ImportError:
        msg = """
              Unable to find trimesh (https://trimsh.org) library.
              To install using pip:

                pip3 install trimesh

              """
        raise ImportError(msg)

    # Fetch data
    data = client.fetch_roi_mesh(roi, export_path=None)

    # Turn into file-like object
    f = io.StringIO(data.decode())

    # Parse with trimesh
    ob = trimesh.load_mesh(f, file_type='obj')

    return Volume.from_object(ob, name=roi)


@inject_client
def fetch_skeletons(x, with_synapses=True, *, client=None):
    """Construct navis.TreeNeuron/List from neuprint neurons.

    Notes
    -----
    Synapses will be attached to the closest node in the skeleton.

    Parameters
    ----------
    x :             str | int | list-like | pandas.DataFrame | SegmentCriteria
                    Body ID(s). Multiple Ids can be provided as list-like or
                    DataFrame with "bodyId" column.
    with_synapses : bool, optional
                    If True will also attach synapses as ``.connectors``.
    client :        neuprint.Client, optional
                    If ``None`` will try using global client.

    Returns
    -------
    navis.Neuronlist

    """
    if isinstance(x, pd.DataFrame):
        if 'bodyId' in x.columns:
            x = x['bodyId'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    if isinstance(x, SegmentCriteria):
        query = x
    else:
        query = SegmentCriteria(bodyId=x)

    # Fetch names, etc
    meta, roi_info = fetch_neurons(query, client=client)

    # Make sure there is a somaLocation and somaRadius column
    if 'somaLocation' not in meta.columns:
        meta['somaLocation'] = None
        meta['somaRadius'] = None

    # Add data to skeletons
    nl = []
    for r in tqdm(meta.itertuples(), desc='Fetching', total=meta.shape[0],
                  leave=False, disable=meta.shape[0] == 1):

        # Fetch skeleton SWC
        data = client.fetch_skeleton(r.bodyId, format='pandas')

        # Convert from raw to nanometers
        # TODO!!!

        # Generate neuron
        n = TreeNeuron(data, units='nm')

        # Add some missing meta data
        n.id = r.bodyId

        if hasattr(r, 'instance'):
            n.name = r.instance

        n.n_voxels = r.size
        n.status = r.status

        # Make KDE tree for NN
        if r.somaLocation or with_synapses:
            tree = neuron2KDTree(n, data='nodes')

        # Set soma
        if r.somaLocation:
            d, i = tree.query([r.somaLocation])
            n.soma = int(n.nodes.iloc[i[0]].node_id)
            n.soma_radius = r.somaRadius

        if with_synapses:
            # Fetch synapses
            syn = fetch_synapses(r.bodyId, client=client)

            if not syn.empty:
                # Process synapses
                syn['connector_id'] = syn.index.values
                locs = syn[['x', 'y', 'z']].values
                d, i = tree.query(locs)

                syn['node_id'] = n.nodes.iloc[i].node_id.values
                syn['x'] = locs[:, 0]
                syn['y'] = locs[:, 1]
                syn['z'] = locs[:, 2]

                # Keep only relevant columns
                syn = syn[['connector_id', 'node_id', 'type',
                           'x', 'y', 'z', 'roi', 'confidence']]

                n.connectors = syn

        nl.append(n)

    return NeuronList(nl)
