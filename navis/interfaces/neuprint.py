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

import trimesh

from textwrap import dedent

from concurrent.futures import ThreadPoolExecutor, as_completed

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

from requests.exceptions import HTTPError

import numpy as np
import pandas as pd

from .. import config

from ..core import Volume, TreeNeuron, NeuronList
from ..graph import neuron2KDTree
from ..morpho import heal_fragmented_neuron

logger = config.logger

# Define some integer types
int_types = (int, np.int32, np.int64, np.int, np.int0)


@inject_client
def fetch_roi(roi, *, client=None):
    """Fetch given ROI.

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
        raise TypeError(f'Expect ROI name as string, got "{type(roi)}"')

    # Fetch data
    data = client.fetch_roi_mesh(roi, export_path=None)

    # Turn into file-like object
    f = io.StringIO(data.decode())

    # Parse with trimesh
    ob = trimesh.load_mesh(f, file_type='obj')

    return Volume.from_object(ob, name=roi)


@inject_client
def fetch_skeletons(x, with_synapses=True, *, heal=False, missing_swc='raise',
                    parallel=True, max_threads=5, client=None):
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
    heal :          bool, optional
                    If True, will automatically heal fragmented skeletons using
                    neuprint-python's heal function.
    missing_swc :   'raise' | 'warn' | 'skip'
                    What to do if no skeleton is found for a given body ID::

                        "raise" (default) will raise an exception
                        "warn" will throw a warning but continue
                        "skip" will skip without any message

    parallel :      bool
                    If True, will use parallel threads to fetch data.
    max_threads :   int
                    Max number of parallel threads to use.
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

    if meta.empty:
        raise ValueError('No neurons matching the given criteria found!')

    # Make sure there is a somaLocation and somaRadius column
    if 'somaLocation' not in meta.columns:
        meta['somaLocation'] = None
        meta['somaRadius'] = None

    nl = []
    with ThreadPoolExecutor(max_workers=1 if not parallel else max_threads) as executor:
        futures = {}
        for r in meta.itertuples():
            f = executor.submit(__fetch_skeleton,
                                r,
                                client=client,
                                with_synapses=with_synapses,
                                missing_swc=missing_swc,
                                heal=heal)
            futures[f] = r.bodyId

        with config.tqdm(desc='Fetching',
                         total=meta.shape[0],
                         leave=config.pbar_leave,
                         disable=meta.shape[0] == 1 or config.pbar_hide) as pbar:
            for f in as_completed(futures):
                bodyId = futures[f]
                pbar.update(1)
                try:
                    nl.append(f.result())
                except Exception as exc:
                    print(f'{bodyId} generated an exception:', exc)

    nl = NeuronList(nl)

    """
    if heal:
        # max_dist of 1000 corresponds to 8um
        heal_fragmented_neuron(nl, max_dist=1000, inplace=True)
    """
    return nl


def __fetch_skeleton(r, client, with_synapses=True, missing_swc='raise', heal=False):
    """Fetch a single skeleton + synapses and turn into CATMAID neuron."""
    # Fetch skeleton SWC
    try:
        data = client.fetch_skeleton(r.bodyId, format='pandas', heal=heal)
    except HTTPError as err:
        if err.response.status_code == 400:
            if missing_swc in ['warn', 'skip']:
                if missing_swc == 'warn':
                    logger.warning(f'No SWC found for {r.bodyId}')
            else:
                raise
        else:
            raise

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

    return n
