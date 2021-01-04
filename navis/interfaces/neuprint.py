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

from ..core import Volume, TreeNeuron, MeshNeuron, NeuronList
from ..graph import neuron2KDTree, subset_neuron

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
def fetch_mesh_neuron(x, *, lod=1, with_synapses=True, missing_mesh='raise',
                      parallel=True, max_threads=5, seg_source=None,
                      client=None, **kwargs):
    """Fetch mesh neuron.

    Requires `cloudvolume <https://github.com/seung-lab/cloud-volume>`_::

        pip3 install cloud-volume

    Notes
    -----
    Synapses will be attached to the closest node on the mesh.

    Parameters
    ----------
    x :             str | int | list-like | pandas.DataFrame | SegmentCriteria
                    Body ID(s). Multiple Ids can be provided as list-like or
                    DataFrame with "bodyId" or "bodyid" column.
    lod :           int
                    Level of detail. Higher ``lod`` = coarser.
    with_synapses : bool, optional
                    If True will also attach synapses as ``.connectors``.
    missing_mesh :  'raise' | 'warn' | 'skip'
                    What to do if no mesh is found for a given body ID::

                        "raise" (default) will raise an exception
                        "warn" will throw a warning but continue
                        "skip" will skip without any message

    parallel :      bool
                    If True, will use parallel threads to fetch data.
    max_threads :   int
                    Max number of parallel threads to use.
    seg_source :    str | cloudvolume.CloudVolume
                    Segmentation source that can be parsed by
                    ``cloudvolume.CloudVolume`` or a ``CloudVolume`` itself. If
                    not provided will try to extract from neuprint ``client``
                    meta data.
    client :        neuprint.Client, optional
                    If ``None`` will try using global client.
    **kwargs
                    Will be passed to ``cloudvolume.CloudVolume``.

    Returns
    -------
    navis.Neuronlist
                    Containing :class:`navis.MeshNeuron`. Note that meshes are
                    resized to raw voxel size to match other spatial data from
                    neuprint (synapses, skeletons, etc).

    """
    try:
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError('``navis.fetch_mesh_neuron`` requires the CloudVolume'
                          ' library. Please install using e.g. pip:'
                          '   pip3 install cloud-volume')

    if isinstance(x, pd.DataFrame):
        if 'bodyId' in x.columns:
            x = x['bodyId'].values
        elif 'bodyid' in x.columns:
            x = x['bodyid'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    # Extract source
    if not seg_source:
        segs = [s for s in client.meta['neuroglancerMeta'] if s.get('dataType') == 'segmentation']
        if len(segs) < 1:
            raise ValueError('Segmentation source could not be automatically '
                             'determined. Please provide via ``seg_source``.')

        seg_source = segs[0]['source']
        if len(segs) > 1:
            logger.warning(f'{len(segs)} segmentation sources found. Using the '
                           f'first entry: "{seg_source}"')

    # Initialize volume
    if isinstance(seg_source, CloudVolume):
        vol = seg_source
    else:
        defaults = dict(use_https=True)
        defaults.update(kwargs)
        vol = CloudVolume(seg_source, **defaults)

    if isinstance(x, NeuronCriteria):
        query = x
    else:
        query = NeuronCriteria(bodyId=x)

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
            f = executor.submit(__fetch_mesh,
                                r,
                                vol=vol,
                                lod=lod,
                                client=client,
                                with_synapses=with_synapses,
                                missing_mesh=missing_mesh)
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

    return nl


def __fetch_mesh(r, *, vol, lod, client, with_synapses=True, missing_mesh='raise'):
    """Fetch a single mesh (+ synapses) and construct navis MeshNeuron."""
    # Fetch mesh
    import cloudvolume
    try:
        mesh = vol.mesh.get(r.bodyId, lod=lod)[r.bodyId]
    except cloudvolume.exceptions.MeshDecodeError as err:
        if 'not found' in str(err):
            if missing_mesh in ['warn', 'skip']:
                if missing_mesh == 'warn':
                    logger.warning(f'No SWC found for {r.bodyId}')
            else:
                raise
        else:
            raise

    # Meshes come out in units (e.g. nanometers) but most other data (synapses,
    # skeletons, etc) come out in voxels, we will therefore scale meshes down
    mesh.vertices /= np.array(client.meta['voxelSize']).reshape(1, 3)

    # Generate neuron
    n = MeshNeuron(mesh,
                   units=f'{client.meta["voxelSize"][0]} {client.meta["voxelUnits"]}')

    # Add some missing meta data
    n.id = r.bodyId

    if hasattr(r, 'instance'):
        n.name = r.instance

    n.n_voxels = r.size
    n.status = r.status
    n.statusLabel = r.statusLabel
    n.lod = lod

    # Set soma
    n.somaLocation = r.somaLocation
    if r.somaLocation:
        n.soma_radius = r.somaRadius / n.units.to('nm').magnitude
        n.soma = r.somaLocation

    if with_synapses:
        # Fetch synapses
        syn = fetch_synapses(r.bodyId,
                             synapse_criteria=SynapseCriteria(primary_only=True),
                             client=client)

        if not syn.empty:
            # Keep only relevant columns
            syn = syn[['type', 'x', 'y', 'z', 'roi', 'confidence']]
            n.connectors = syn

    return n


@inject_client
def fetch_skeletons(x, *, with_synapses=True, heal=False, missing_swc='raise',
                    parallel=True, max_threads=5, client=None):
    """Construct navis.TreeNeuron/List from neuprint neurons.

    Notes
    -----
    Synapses will be attached to the closest node in the skeleton.

    Parameters
    ----------
    x :             str | int | list-like | pandas.DataFrame | SegmentCriteria
                    Body ID(s). Multiple Ids can be provided as list-like or
                    DataFrame with "bodyId"  or "bodyid" column.
    with_synapses : bool, optional
                    If True will also attach synapses as ``.connectors``.
    heal :          bool | int | float, optional
                    If True, will automatically heal fragmented skeletons using
                    neuprint-python's ``heal_skeleton`` function. Pass a float
                    or an int to limit the max distance at which nodes are
                    allowed to be re-connected (requires neuprint-python >= 0.4.11).
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
        elif 'bodyid' in x.columns:
            x = x['bodyid'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    if isinstance(x, NeuronCriteria):
        query = x
    else:
        query = NeuronCriteria(bodyId=x)

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

    return nl


def __fetch_skeleton(r, client, with_synapses=True, missing_swc='raise',
                     heal=False, max_distance=None):
    """Fetch a single skeleton + synapses and construct navis TreeNeuron."""
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

    # Generate neuron
    # Note that we are currently assuming that the x/y/z data is isotropic
    n = TreeNeuron(data,
                   units=f'{client.meta["voxelSize"][0]} {client.meta["voxelUnits"]}')

    # Reduce precision
    n._nodes = n._nodes.astype({'node_id': np.int32, 'parent_id': np.int32,
                                'x': np.float32, 'y': np.float32,
                                'z': np.float32, 'radius': np.float32})

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
    else:
        n.soma = None

    if with_synapses:
        # Fetch synapses
        syn = fetch_synapses(r.bodyId,
                             synapse_criteria=SynapseCriteria(primary_only=True),
                             client=client)

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

            # Manually make the "roi" column of the synapse table into a 
            # categorical to save some memory
            syn['roi'] = syn.roi.astype('category')

            n.connectors = syn

    return n


def remove_soma_hairball(x: 'core.TreeNeuron',
                         radius: float = 500,
                         inplace: bool = False):
    """Remove hairball around soma.

    Parameters
    ----------
    x :         core.TreeNeuron
    radius :    float
                Radius around the soma to check for hairball

    Returns
    -------
    TreeNeuron
                If inplace=False.
    """
    if not inplace:
        x = x.copy()
    if not x.soma:
        if not inplace:
            return x
        return
    # Get all nodes within given radius of soma nodes
    soma_loc = x.nodes.set_index('node_id').loc[[x.soma],
                                                ['x', 'y', 'z']].values
    tree = neuron2KDTree(x)
    dist, ix = tree.query(soma_loc, k=x.n_nodes, distance_upper_bound=radius)

    # Subset to nodes within range
    to_check = set(list(ix[0, dist[0, :] <= radius]))

    # Get the segments that have nodes in the soma
    segs = [seg for seg in x.segments if set(seg) & to_check]

    # Unless these segments end in a root node, we will keep the last node
    # (which will be a branch point)
    segs = [s[:-1] if s[-1] not in x.root else s for s in segs]

    # This is already sorted by length -> we will keep the first (i.e. longest)
    # segment and remove the rest
    to_remove = [n for s in segs[1:] for n in s]

    to_keep = x.nodes.loc[~x.nodes.node_id.isin(to_remove), 'node_id'].values

    # Move soma if required
    if x.soma in to_remove:
        x.soma = list(to_check & set(to_keep))[0]

    subset_neuron(x, to_keep, inplace=True)

    if not inplace:
        return x
