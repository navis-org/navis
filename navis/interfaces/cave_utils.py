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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.


from ..core import MeshNeuron, NeuronList
from .. import config, utils
import pandas as pd
import numpy as np

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
    config.logger.error(err_msg)
    CAVEclient = None
    cv = None
except BaseException:
    raise


logger = config.get_logger(__name__)
dataset = None
SILENCE_FIND_MAT_VERSION = False


@lru_cache(None)
def get_cave_client(datastack="cortex65", server_address=None):
    """Get caveclient for given datastack.

    Parameters
    ----------
    datastack :     "cortex65" | "cortex35" | "layer 2/3" | "h01_c3_flat" | str
                    Which dataset to query. "cortex65", "cortex35" and "layer 2/3"
                    are internally mapped to the corresponding sources: for example,
                    "minnie65_public_vXXX" for "cortex65" where XXX is always the
                    most recent version).

    """
    if not CAVEclient:
        raise ImportError(err_msg)

    return CAVEclient(datastack, server_address)


@lru_cache(None)
def _get_cloudvol(url, cache=True):
    """Get (cached) CloudVolume for given segmentation.

    Parameters
    ----------
    url :     str

    """
    if not cv:
        raise ImportError(err_msg)

    return cv.CloudVolume(
        url, cache=cache, use_https=True, parallel=10, progress=False, fill_missing=True
    )


def _get_somas(root_ids, client, materialization="auto"):
    """Fetch somas based on nuclei segmentation for given neuron(s).

    Since this is a nucleus detection you will find that some neurons do
    not have an entry despite clearly having a soma. This is due to the
    "avocado problem" where the nucleus is separate from the rest of the
    soma.

    Important
    ---------
    This data currently only exists for the 'cortex65' datastack (i.e.
    "minnie65_public_vXXX").

    Parameters
    ----------
    root_ids  :     int | list of ints | None
                    Root ID(s) for which to fetch soma infos. Use
                    `None` to fetch complete list of annotated nuclei.
    table :         str
                    Which table to use for nucleus annotations. Also
                    see the `microns.list_annotation_tables()` function.
    datastack :     "cortex65" | "cortex35" | "layer 2/3" | "h01_c3_flat" | str
                    Which dataset to query. "cortex65", "cortex35" and "layer 2/3"
                    are internally mapped to the corresponding sources: for example,
                    "minnie65_public_vXXX" for "cortex65" where XXX is always the
                    most recent version).

    Returns
    -------
    DataFrame
                    Pandas DataFrame with nuclei (see Examples). Root IDs
                    without a nucleus will simply not have an entry in the
                    table.

    """
    # This property should be set by the dispatching function
    table = client.materialize.nucleus_table

    if materialization == "auto":
        if root_ids is not None:
            materialization = roots_to_mat(root_ids, client, verbose=False)
        else:
            # Use the most recent materialization
            materialization = None

    filter_in_dict = None
    if not isinstance(root_ids, type(None)):
        root_ids = utils.make_iterable(root_ids)
        filter_in_dict = {"pt_root_id": root_ids}

    return client.materialize.query_table(
        table, filter_in_dict=filter_in_dict, materialization_version=materialization
    )


def fetch_neurons(
    x,
    lod,
    with_synapses,
    client,
    parallel,
    max_threads,
    materialization="auto",
    **kwargs,
):
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
    client :        CAVEclient
                    The CAVEclient with which to interact.
    parallel :      bool
                    If True, will use parallel threads to fetch data.
    max_threads :   int
                    Max number of parallel threads to use.
    materialization : "auto" | int
                    Which materialization version to use to look up somas and synapses
                    (if applicable). If "auto" (default) will try to find the most
                    recent version that contains the given root IDs. If an
                    integer is provided will use that version.
    **kwargs
                    Keyword arguments are passed through to the initialization
                    of the ``navis.MeshNeurons``.

    Returns
    -------
    navis.Neuronlist
                    Containing :class:`navis.MeshNeuron`.

    """
    x = utils.make_iterable(x, force_type=int)

    vol = _get_cloudvol(client.info.segmentation_source())  # this is cached

    try:
        somas = _get_somas(x, client=client, materialization=materialization)
        soma_pos = somas.set_index("pt_root_id").pt_position.to_dict()
    except BaseException as e:
        logger.warning("Failed to fetch somas via nucleus segmentation" f"(){e})")
        soma_pos = {}

    nl = []
    if max_threads > 1 and parallel:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {}
            for id in x:
                f = executor.submit(
                    _fetch_single_neuron,
                    id,
                    vol=vol,
                    lod=lod,
                    client=client,
                    with_synapses=with_synapses,
                    materialization=materialization,
                    **kwargs,
                )
                futures[f] = id

            with config.tqdm(
                desc="Fetching",
                total=len(x),
                leave=config.pbar_leave,
                disable=len(x) == 1 or config.pbar_hide,
            ) as pbar:
                for f in as_completed(futures):
                    id = futures[f]
                    pbar.update(1)
                    try:
                        nl.append(f.result())
                    except Exception as exc:
                        print(f"{id} generated an exception:", exc)
    else:
        for id in config.tqdm(
            x,
            desc="Fetching",
            leave=config.pbar_leave,
            disable=len(x) == 1 or config.pbar_hide,
        ):
            n = _fetch_single_neuron(
                id,
                vol=vol,
                lod=lod,
                client=client,
                with_synapses=with_synapses,
                materialization=materialization,
                **kwargs,
            )
            nl.append(n)

    nl = NeuronList(nl)

    for n in nl:
        if n.id in soma_pos:
            # For VoxelResolution see client.materialize.get_table_metadata('nucleus_detection_v0')
            # (attached to df as 'table_voxel_resolution')
            n.soma_pos = (
                np.array(soma_pos[n.id]) * somas.attrs["table_voxel_resolution"]
            )
        else:
            n.soma_pos = None

    return nl


def _fetch_single_neuron(
    id, lod, vol, client, with_synapses=False, materialization="auto", **kwargs
):
    """Fetch a single neuron."""
    # Make sure we only use `lod` if that's actually supported by the source
    if "MultiLevel" in str(type(vol.mesh)):
        mesh = vol.mesh.get(id, lod=lod)[id]
    else:
        mesh = vol.mesh.get(id, deduplicate_chunk_boundaries=False)[id]

    n = MeshNeuron(mesh, id=id, units="nm", **kwargs)

    if materialization == "auto":
        materialization = roots_to_mat(id, client, verbose=False)

    if with_synapses:
        pre = client.materialize.synapse_query(
            pre_ids=id, materialization_version=materialization
        )
        post = client.materialize.synapse_query(
            post_ids=id, materialization_version=materialization
        )

        syn_table = client.materialize.synapse_table
        syn_info = client.materialize.get_table_metadata(syn_table)
        vxl_size = np.array(syn_info["voxel_resolution"]).astype(int)

        to_concat = []
        if not pre.empty:
            pre["type"] = "pre"
            locs = np.vstack(pre["pre_pt_position"].values)
            locs = locs * vxl_size
            pre["x"], pre["y"], pre["z"] = locs[:, 0], locs[:, 1], locs[:, 2]
            pre = pre[["id", "x", "y", "z", "type", "size"]].copy()
            to_concat.append(pre)
        if not post.empty:
            post["type"] = "post"
            locs = np.vstack(post["post_pt_position"].values)
            locs = locs * vxl_size
            post["x"], post["y"], post["z"] = locs[:, 0], locs[:, 1], locs[:, 2]
            post = post[["id", "x", "y", "z", "type", "size"]].copy()
            to_concat.append(post)

        if len(to_concat) == 1:
            n.connectors = to_concat[0]
        elif len(to_concat) == 2:
            n.connectors = pd.concat(to_concat, axis=0).reset_index(drop=True)

    return n


def get_voxels(x, mip, bounds, client):
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
    client :        CAVEclient
                    The CAVEclient with which to interact.

    Returns
    -------
    voxels :        (N, 3) np.ndarray
                    In voxel space according to `mip`.

    """
    # Need to get the graphene (not the precomputed) version of the data
    vol_graphene = cv.CloudVolume(
        client.chunkedgraph.cloudvolume_path, use_https=True, progress=False
    )
    url = client.info.get_datastack_info()["segmentation_source"]
    vol_prec = _get_cloudvol(url)

    # Get L2 chunks making up this neuron
    l2_ids = client.chunkedgraph.get_leaves(x, stop_layer=2)

    # Turn l2_ids into chunk indices
    l2_ix = [
        np.array(vol_graphene.mesh.meta.meta.decode_chunk_position(l2)) for l2 in l2_ids
    ]
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
            raise ValueError("`bounds` must be [xmin, xmax, ymin, ymax, zmin, zmax]")
        l2_vxl = l2_vxl[np.all(l2_vxl >= bounds[::2], axis=1)]
        l2_vxl = l2_vxl[np.all(l2_vxl < bounds[1::2] + ch_size, axis=1)]

    try:
        vol_prec.mip = mip
        for ch in config.tqdm(l2_vxl, desc="Loading"):
            ct = vol_prec[
                ch[0] : ch[0] + ch_size[0],
                ch[1] : ch[1] + ch_size[1],
                ch[2] : ch[2] + ch_size[2],
            ][:, :, :, 0]
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


def roots_to_mat(
    ids,
    client,
    verbose=True,
    allow_multiple=False,
    raise_missing=True,
):
    """Find a materialization version (or live) for given root ID(s).

    Parameters
    ----------
    ids :           int | iterable
                    Root ID(s) to check.
    client :        CAVEclient
                    The CAVEclient with which to interact.
    verbose :       bool
                    Whether to print results of search.
    allow_multiple : bool
                    If True, will track if IDs can be found spread across multiple
                    materialization versions if there is no single one containing
                    all.
    raise_missing : bool
                    Only relevant if `allow_multiple=True`. If False, will return
                    versions even if some IDs could not be found.

    Returns
    -------
    version :       int | "live"
                    A single version (including "live") that contains all given
                    root IDs.
    versions :      np.ndarray
                    If no single version was found and `allow_multiple=True` will
                    return a vector of `len(ids)` with the latest version at which
                    the respective ID can be found.
                    Important: "live" version will be return as -1!
                    If `raise_missing=False` and one or more root IDs could not
                    be found in any of the available materialization versions
                    these IDs will be return as version 0.

    """
    ids = utils.make_iterable(ids)

    # For each ID track the most recent valid version
    latest_valid = np.zeros(len(ids), dtype=np.int32)

    # Get the meta data for the available materialization versions
    # This is a list of dicts where each dict has a "time_stamp" key
    vmeta = client.materialize.get_versions_metadata()

    # Sort by "time_stamp"
    vmeta = sorted(vmeta, key=lambda x: x["time_stamp"], reverse=True)

    # Go over each version (start with the most recent)
    for i, mat in enumerate(vmeta):
        ts_m = mat["time_stamp"]
        version = mat["version"]

        # Check which root IDs were valid at the time
        is_valid = client.chunkedgraph.is_latest_roots(ids, timestamp=ts_m)

        # Update latest valid versions
        latest_valid[(latest_valid == 0) & is_valid] = version

        if all(is_valid):
            if verbose and not SILENCE_FIND_MAT_VERSION:
                print(f"Using materialization version {version}.")
            return version

    # If no single materialized version can be found, see if we can get
    # by with the live materialization
    is_latest = client.chunkedgraph.is_latest_roots(ids, timestamp=None)
    latest_valid[(latest_valid == 0) & is_latest] = -1  # track "live" as -1
    if all(is_latest) and dataset != "public":  # public does not have live
        if verbose:
            print("Using live materialization")
        return "live"

    if allow_multiple and any(latest_valid != 0):
        if all(latest_valid != 0):
            if verbose and not SILENCE_FIND_MAT_VERSION:
                print(
                    f"Found root IDs spread across {len(np.unique(latest_valid))} "
                    "materialization versions."
                )
            return latest_valid

        msg = (
            f"Found root IDs spread across {len(np.unique(latest_valid)) - 1} "
            f"materialization versions but {(latest_valid == 0).sum()} IDs "
            "do not exist in any of the materialized tables."
        )

        if not raise_missing:
            if verbose and not SILENCE_FIND_MAT_VERSION:
                print(msg)
            return latest_valid
        else:
            raise MaterializationMatchError(msg)

    if dataset not in ("public, "):
        raise MaterializationMatchError(
            "Given root IDs do not (co-)exist in any of the available "
            "materialization versions (including live). Try updating "
            "root IDs and rerun your query."
        )
    else:
        raise MaterializationMatchError(
            "Given root IDs do not (co-)exist in any of the available "
            "public materialization versions. Please make sure that "
            "the root IDs do exist and rerun your query."
        )


class MaterializationMatchError(Exception):
    pass
