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

"""Interface with MICrONS datasets: https://www.microns-explorer.org/."""

from ..core import MeshNeuron, NeuronList
from .. import config, utils
import pandas as pd
import numpy as np
import warnings

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from textwrap import dedent

err_msg = dedent("""
      Failed to import `caveclient` library. Please install using pip:

            pip install caveclient -U

      """)

try:
    from caveclient import CAVEclient
    import cloudvolume as cv
except ImportError:
    logger.error(err_msg)
    CAVEclient = None
    cv = None
except BaseException:
    raise


logger = config.get_logger(__name__)
dataset = None

CAVE_DATASTACKS = {
    'cortex65': 'minnie65_public_v117',
    'cortex35': 'minnie35_public_v0',
    'layer 2/3': 'pinky100_public_flat_v185'
}

SEG_URLS = {
    'cortex65': 'precomputed://gs://iarpa_microns/minnie/minnie65/seg',
    'cortex35': 'precomputed://gs://iarpa_microns/minnie/minnie35/seg',
    'layer 2/3': 'precomputed://gs://microns_public_datasets/pinky100_v185/seg'
}


@lru_cache(None)
def get_cave_client(datastack='cortex65'):
    """Get caveclient for given datastack.

    Parameters
    ----------
    datastack :     "cortex65" | "cortex35" | "layer 2/3"
                    Name of the dataset to use.

    """
    if not CAVEclient:
        raise ImportError(err_msg)

    # Try mapping, else pass-through
    datastack = CAVE_DATASTACKS.get(datastack, datastack)
    return CAVEclient(datastack)


@lru_cache(None)
def get_cloudvol(url, cache=True):
    """Get (cached) CloudVolume for given segmentation.

    Parameters
    ----------
    url :     str

    """
    if not cv:
        raise ImportError(err_msg)

    return cv.CloudVolume(url, cache=cache, use_https=True,
                          progress=False, fill_missing=True)


def get_somas(root_ids, table='nucleus_neuron_svm', datastack='cortex65'):
    """Fetch somas based on nuclei segmentation for given neuron(s).

    Since this is a nucleus detection you will find that some neurons do
    not have an entry despite clearly having a soma. This is due to the
    "avocado problem" where the nucleus is separate from the rest of the
    soma.

    Important
    ---------
    This data currently only exists for the 'cortex65' datastack (i.e.
    "minnie65_public_v117").

    Parameters
    ----------
    root_ids  :         int | list of ints | None
                        Root ID(s) for which to fetch soma infos. Use
                        ``None`` to fetch complete list of annotated nuclei.
    table :             str
                        Which table to use for nucleus annotations.
    datastack :         "cortex65" | "cortex35" | "layer 2/3"
                        Which dataset to use. Internally these are mapped to the
                        corresponding sources (e.g. "minnie65_public_v117" for
                        "cortex65").

    Returns
    -------
    DataFrame
                        Pandas DataFrame with nuclei (see Examples). Root IDs
                        without a nucleus will simply not have an entry in the
                        table.

    """
    if datastack != 'cortex65':
        warnings.warn('To our knowledge there is no nucleus segmentation '
                      f'for "{datastack}". If that has changed please '
                      'get in touch on navis Github.')

    # Get/Initialize the CAVE client
    client = get_cave_client(datastack)

    filter_in_dict = None
    if not isinstance(root_ids, type(None)):
        root_ids = utils.make_iterable(root_ids)
        filter_in_dict = {'pt_root_id': root_ids}

    return client.materialize.query_table(table, filter_in_dict=filter_in_dict)


def fetch_neurons(x, *, lod=2,
                  with_synapses=True,
                  datastack='cortex65',
                  parallel=True,
                  max_threads=4,
                  **kwargs):
    """Fetch neuron meshes.

    Notes
    -----
    Synapses will be attached to the closest vertex on the mesh.

    Parameters
    ----------
    x :             str | int | list-like
                    Segment ID(s). Multiple Ids can be provided as list-like.
    lod :           int
                    Level of detail. Higher ``lod`` = coarser. This parameter
                    is ignored if the data source does not support multi-level
                    meshes.
    with_synapses : bool, optional
                    If True will also attach synapses as ``.connectors``.
    datastack :     "cortex65" | "cortex35" | "layer 2/3"
                    Which dataset to use. Internally these are mapped to the
                    corresponding sources (e.g. "minnie65_public_v117" for
                    "cortex65").
    parallel :      bool
                    If True, will use parallel threads to fetch data.
    max_threads :   int
                    Max number of parallel threads to use.
    **kwargs
                    Keyword arguments are passed through to the initialization
                    of the ``navis.MeshNeurons``.

    Returns
    -------
    navis.Neuronlist
                    Containing :class:`navis.MeshNeuron`.

    """
    x = utils.make_iterable(x, force_type=int)
    client = get_cave_client(datastack)

    if datastack in SEG_URLS:
        url = SEG_URLS[datastack]
    else:
        url = client.info.get_datastack_info()['segmentation_source']
    vol = get_cloudvol(url)

    if datastack == 'cortex65':
        try:
            somas = get_somas(x, datastack=datastack)
            soma_pos = somas.set_index('pt_root_id').pt_position.to_dict()
        except BaseException as e:
            logger.warning('Failed to fetch somas via nucleus segmentation'
                           f'(){e})')
            soma_pos = {}        
    else:
        soma_pos = {}

    nl = []
    with ThreadPoolExecutor(max_workers=1 if not parallel else max_threads) as executor:
        futures = {}
        for id in x:
            f = executor.submit(_fetch_single_neuron,
                                id,
                                vol=vol,
                                lod=lod,
                                client=client,
                                with_synapses=with_synapses,
                                source=datastack,
                                **kwargs
                                )
            futures[f] = id

        with config.tqdm(desc='Fetching',
                         total=len(x),
                         leave=config.pbar_leave,
                         disable=len(x) == 1 or config.pbar_hide) as pbar:
            for f in as_completed(futures):
                id = futures[f]
                pbar.update(1)
                try:
                    nl.append(f.result())
                except Exception as exc:
                    print(f'{id} generated an exception:', exc)

    nl = NeuronList(nl)

    for n in nl:
        if n.id in soma_pos:
            n.soma_pos = np.array(soma_pos[n.id]) * [4, 4, 40]
        else:
            n.soma_pos = None

    return nl


def _fetch_single_neuron(id, lod, vol, client, with_synapses=False, **kwargs):
    """Fetch a single neuron."""
    # Make sure we only use `lod` if that's actually supported by the source
    if 'MultiLevel' in str(type(vol.mesh)):
        mesh = vol.mesh.get(id, lod=lod, progress=False)[id]
    else:
        mesh = vol.mesh.get(id, progress=False)[id]

    n = MeshNeuron(mesh, id=id, units='nm', **kwargs)

    if with_synapses:
        pre = client.materialize.synapse_query(pre_ids=id)
        post = client.materialize.synapse_query(post_ids=id)

        syn_table = client.materialize.synapse_table
        syn_info = client.materialize.get_table_metadata(syn_table)
        vxl_size = np.array(syn_info['voxel_resolution']).astype(int)

        to_concat = []
        if not pre.empty:
            pre['type'] = 'pre'
            locs = np.vstack(pre['pre_pt_position'].values)
            locs = locs * vxl_size
            pre['x'], pre['y'], pre['z'] = locs[:, 0], locs[:, 1], locs[:, 2]
            pre = pre[['id', 'x', 'y', 'z', 'type', 'size']].copy()
            to_concat.append(pre)
        if not post.empty:
            post['type'] = 'post'
            locs = np.vstack(post['post_pt_position'].values)
            locs = locs * vxl_size
            post['x'], post['y'], post['z'] = locs[:, 0], locs[:, 1], locs[:, 2]
            post = post[['id', 'x', 'y', 'z', 'type', 'size']].copy()
            to_concat.append(post)

        if len(to_concat) == 1:
            n.connectors = to_concat[0]
        elif len(to_concat) == 2:
            n.connectors = pd.concat(to_concat, axis=0).reset_index(drop=True)

    return n


def get_voxels(x, mip=0, bounds=None, datastack='cortex65'):
    """Fetch voxels making a up given root ID.

    Parameters
    ----------
    x :             int
                    A single root ID.
    mip :           int
                    Scale at which to fetch voxels.
    bounds :        list, optional
                    Bounding box [xmin, xmax, ymin, ymax, zmin, zmax] in voxel
                    space. For example, the voxel resolution for mip 0
                    segmentation is 8 x 8 x 40 nm.
    datastack :     "cortex65" | "cortex35" | "layer 2/3"
                    Which dataset to use. Internally these are mapped to the
                    corresponding sources (e.g. "minnie65_public_v117" for
                    "cortex65").

    Returns
    -------
    voxels :        (N, 3) np.ndarray
                    In voxel space according to `mip`.

    """
    client = get_cave_client(datastack)
    # Need to get the graphene (not the precomputed) version of the data
    vol_graphene = cv.CloudVolume(client.chunkedgraph.cloudvolume_path,
                                  use_https=True, progress=False)
    if datastack in SEG_URLS:
        url = SEG_URLS[datastack]
    else:
        url = client.info.get_datastack_info()['segmentation_source']
    vol_prec = get_cloudvol(url)

    # Get L2 chunks making up this neuron
    l2_ids = client.chunkedgraph.get_leaves(x, stop_layer=2)

    # Turn l2_ids into chunk indices
    l2_ix = [np.array(vol_graphene.mesh.meta.meta.decode_chunk_position(l)) for l in l2_ids]
    l2_ix = np.unique(l2_ix, axis=0)

    # Convert to nm
    l2_nm = np.asarray(_chunks_to_nm(l2_ix, vol=vol_graphene))

    # Convert back to voxel space (according to mip)
    l2_vxl = l2_nm // vol_prec.meta.scales[mip]["resolution"]

    voxels = []
    ch_size = np.array(vol_graphene.mesh.meta.meta.graph_chunk_size)
    ch_size = ch_size // (vol_prec.mip_resolution(mip) / vol_prec.mip_resolution(0))
    ch_size = np.asarray(ch_size).astype(int)
    old_mip = vol_prec.mip

    if not isinstance(bounds, type(None)):
        bounds = np.asarray(bounds)
        if not bounds.ndim == 1 or len(bounds) != 6:
            raise ValueError('`bounds` must be [xmin, xmax, ymin, ymax, zmin, zmax]')
        l2_vxl = l2_vxl[np.all(l2_vxl >= bounds[::2], axis=1)]
        l2_vxl = l2_vxl[np.all(l2_vxl < bounds[1::2] + ch_size, axis=1)]

    try:
        vol_prec.mip = mip
        for ch in config.tqdm(l2_vxl, desc='Loading'):
            ct = vol_prec[ch[0]:ch[0] + ch_size[0],
                          ch[1]:ch[1] + ch_size[1],
                          ch[2]:ch[2] + ch_size[2]][:, :, :, 0]
            this_vxl = np.dstack(np.where(ct == x))[0]
            this_vxl = this_vxl + ch
            voxels.append(this_vxl)
    except BaseException:
        raise
    finally:
        vol_prec.mip = old_mip
    voxels = np.vstack(voxels)

    if not isinstance(bounds, type(None)):
        voxels = voxels[np.all(voxels >= bounds[::2], axis=1)]
        voxels = voxels[np.all(voxels < bounds[1::2], axis=1)]

    return voxels


def _chunks_to_nm(xyz_ch, vol, voxel_resolution=[4, 4, 40]):
    """Map a chunk location to Euclidean space.

    Parameters
    ----------
    xyz_ch :            array-like
                        (N, 3) array of chunk indices.
    vol :               cloudvolume.CloudVolume
                        CloudVolume object associated with the chunked space.
    voxel_resolution :  list, optional
                        Voxel resolution.

    Returns
    -------
    np.array
                        (N, 3) array of spatial points.

    """
    mip_scaling = vol.mip_resolution(0) // np.array(voxel_resolution, dtype=int)

    x_vox = np.atleast_2d(xyz_ch) * vol.mesh.meta.meta.graph_chunk_size
    return (
        (x_vox + np.array(vol.mesh.meta.meta.voxel_offset(0)))
        * voxel_resolution
        * mip_scaling
    )
