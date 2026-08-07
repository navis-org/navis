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

"""Interface with neuprint. This module is wrapping neuprint-python
(https://github.com/connectome-neuprint/neuprint-python) and adding some
navis-specific functions.
"""

import copy
import threading
import trimesh

from urllib.parse import urlparse
from textwrap import dedent

try:
    from neuprint import *

    # remove neuprint's own fetch_skeleton function to avoid confusion
    try:
        del fetch_skeleton  # noqa
    except NameError:
        pass
    from neuprint.client import inject_client

    # Keep a reference to neuprint's own (de-duplicating) `fetch_synapses`. Our
    # own `fetch_synapses` (defined further down) shadows it but delegates back
    # to this for the default (de-duplicating) behaviour.
    from neuprint import fetch_synapses as _neuprint_fetch_synapses
    from neuprint.queries.neuroncriteria import neuroncriteria_args
    from neuprint.utils import iter_batches
except ModuleNotFoundError:
    msg = dedent("""
          neuprint library not found. Please install using pip:

                pip install neuprint-python

          """)
    raise ModuleNotFoundError(msg)
except BaseException:
    raise

import io

from requests.exceptions import HTTPError

import numpy as np
import pandas as pd

from .. import config, utils

from ..core import Volume, Skeleton, Mesh, NeuronList
from ..graph import neuron2KDTree
from ..morpho import subset_neuron
from .base import fetch_parallel

logger = config.get_logger(__name__)

# Define some integer types
int_types = (int, np.integer)

# Holds a per-thread copy of the neuprint client (see `_init_worker`)
_thread_local = threading.local()


def _init_worker(client):
    """Give this worker thread its own copy of the client.

    `requests.Session` is not thread-safe which is why neuprint-python hands out
    a separate deepcopy of the client to each thread (see
    `neuprint.client.default_client`). Passing an explicit client into a thread
    bypasses that mechanism, so we have to do the copying ourselves.

    Note that `Session`/`HTTPAdapter` implement `__getstate__`/`__setstate__`,
    so the copy gets a fresh connection pool rather than aliasing the original's.
    `Client.__init__` is not re-run, i.e. this costs no extra requests.
    """
    _thread_local.client = copy.deepcopy(client)


@inject_client
@neuroncriteria_args("neuron_criteria")
def fetch_synapses(
    neuron_criteria,
    synapse_criteria=None,
    batch_size=10,
    *,
    dedup=True,
    nt=None,
    client=None,
):
    """Fetch synapses for a neuron or selection of neurons.

    This is a thin wrapper around `neuprint.fetch_synapses` that adds the
    option to *not* de-duplicate presynapses (see `dedup`).

    In insect connectomes synapses are typically polyadic, i.e. a single
    presynapse connects to multiple postsynapses. By default - and as done by
    `neuprint.fetch_synapses` - each presynapse is reported only once. With
    `dedup=False` a presynapse is instead reported once for each postsynaptic
    site it connects to. This is useful if you want presynapse counts to
    reflect the number of downstream connections.

    Parameters
    ----------
    neuron_criteria : bodyId(s) | type/instance | NeuronCriteria
                    Determines which bodies to fetch synapses for.
    synapse_criteria : SynapseCriteria, optional
                    Filter synapses by roi, type or confidence. See
                    `neuprint.SynapseCriteria` for details.
    batch_size :    int
                    To avoid timeouts, synapses are fetched in batches of this
                    many bodies.
    dedup :         bool
                    If True (default), presynapses that appear in more than one
                    SynapseSet are de-duplicated - this matches the behaviour of
                    `neuprint.fetch_synapses`. If False, each presynapse is
                    reported once per postsynaptic site it connects to.
    nt :            None | 'max' | 'all', optional
                    Whether/how to fetch neurotransmitter predictions for each
                    "pre" synapse. See `neuprint.fetch_synapses` for details.
                    Only supported with `dedup=True`.
    client :        neuprint.Client, optional
                    If `None` will try using global client.

    Returns
    -------
    pandas.DataFrame
                    Each row represents a single synapse.

    """
    # De-duplicating is the default neuprint behaviour - simply delegate.
    # (`nt` is passed through explicitly rather than via `**kwargs` because
    # neuprint's `@neuroncriteria_args` decorator does not compose with a
    # `**kwargs` parameter.)
    if dedup:
        return _neuprint_fetch_synapses(
            neuron_criteria,
            synapse_criteria=synapse_criteria,
            batch_size=batch_size,
            nt=nt,
            client=client,
        )

    # The no-dedup path runs our own query and does not replicate neuprint's
    # neurotransmitter handling.
    if nt is not None:
        raise ValueError(
            "Fetching neurotransmitters (`nt`) is only supported with "
            "`dedup=True` (it is handled by `neuprint.fetch_synapses`)."
        )

    # No-dedup path: mirror neuprint's batching (see `neuprint.fetch_synapses`)
    # but run a query that omits the `WITH DISTINCT n, s` step.
    neuron_criteria.matchvar = "n"
    q = f"""
        {neuron_criteria.global_with(prefix=8)}
        MATCH (n:{neuron_criteria.label})
        {neuron_criteria.all_conditions(prefix=8)}
        RETURN n.bodyId as bodyId
    """
    bodies = client.fetch_custom(q)["bodyId"].values

    batch_dfs = []
    for batch_bodies in iter_batches(bodies, batch_size):
        batch_criteria = copy.copy(neuron_criteria)
        batch_criteria.bodyId = batch_bodies
        batch_df = _fetch_synapses_no_dedup(batch_criteria, synapse_criteria, client)
        if len(batch_df) > 0:
            batch_dfs.append(batch_df)

    if batch_dfs:
        return pd.concat(batch_dfs, ignore_index=True)

    # Return empty results, but with correct dtypes (matches neuprint)
    dtypes = {
        "bodyId": np.dtype("int64"),
        "type": pd.CategoricalDtype(categories=["pre", "post"], ordered=False),
        "roi": pd.Series([""]).dtype,
        "x": np.dtype("int32"),
        "y": np.dtype("int32"),
        "z": np.dtype("int32"),
        "confidence": np.dtype("float32"),
    }
    return pd.DataFrame([], columns=dtypes.keys()).astype(dtypes)


def _fetch_synapses_no_dedup(neuron_criteria, synapse_criteria, client):
    """Fetch synapses for given neurons WITHOUT de-duplicating presynapses.

    This mirrors neuprint's private `_fetch_synapses` but omits the
    `WITH DISTINCT n, s` step, so that each presynapse is reported once per
    postsynaptic site (i.e. once per SynapseSet it is contained in). We do not
    replicate its neurotransmitter (`nt`) handling - that path goes through
    `neuprint.fetch_synapses` (see `fetch_synapses` with `dedup=True`).

    TODO: drop this copy once neuprint-python exposes a `dedup`/`distinct`
    toggle on `fetch_synapses`; until then this is coupled to the (private)
    query/output shape of `neuprint.queries.synapses._fetch_synapses`.
    """
    neuron_criteria.matchvar = "n"

    if synapse_criteria is None:
        synapse_criteria = SynapseCriteria(client=client)

    if synapse_criteria.primary_only:
        return_rois = {*client.primary_rois}
    else:
        return_rois = {*client.all_rois}

    # If the user specified rois to filter synapses by, but hasn't specified rois
    # in the NeuronCriteria, add them to the NeuronCriteria to speed up the query.
    if synapse_criteria.rois and not neuron_criteria.rois:
        neuron_criteria.rois = {*synapse_criteria.rois}
        neuron_criteria.roi_req = "any"

    # Fetch results. Note the absence of a `WITH DISTINCT n, s` step (compared
    # to `neuprint.fetch_synapses`) which is what keeps presynapses duplicated.
    cypher = dedent(f"""\
        {neuron_criteria.global_with(prefix=8)}
        MATCH (n:{neuron_criteria.label})
        {neuron_criteria.all_conditions('n', prefix=8)}

        MATCH (n)-[:Contains]->(ss:SynapseSet),
              (ss)-[:Contains]->(s:Synapse)

        {synapse_criteria.condition('n', 's', prefix=8)}

        RETURN n.bodyId as bodyId,
               s.type as type,
               s.confidence as confidence,
               s.location.x as x,
               s.location.y as y,
               s.location.z as z,
               apoc.map.removeKeys(s, ['location', 'confidence', 'type']) as syn_info
    """)

    data = client.fetch_custom(cypher, format="json")["data"]

    # Assemble DataFrame
    syn_table = []
    for body, syn_type, conf, x, y, z, syn_info in data:
        # Exclude non-primary ROIs if necessary
        syn_rois = return_rois & {*syn_info.keys()}
        for roi in syn_rois:
            syn_table.append((body, syn_type, roi, x, y, z, conf))

        if not syn_rois:
            syn_table.append((body, syn_type, None, x, y, z, conf))

    syn_df = pd.DataFrame(
        syn_table, columns=["bodyId", "type", "roi", "x", "y", "z", "confidence"]
    )

    # Save RAM with smaller dtypes and interned strings
    syn_df["type"] = pd.Categorical(syn_df["type"], ["pre", "post"])
    syn_df["x"] = syn_df["x"].astype(np.int32)
    syn_df["y"] = syn_df["y"].astype(np.int32)
    syn_df["z"] = syn_df["z"].astype(np.int32)
    syn_df["confidence"] = syn_df["confidence"].astype(np.float32)

    return syn_df


@inject_client
def fetch_roi(roi, *, client=None):
    """Fetch given ROI.

    Parameters
    ----------
    roi :           str
                    Name of an ROI.
    client :        neuprint.Client, optional
                    If `None` will try using global client.

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
    ob = trimesh.load_mesh(f, file_type="obj")

    return Volume.from_object(ob, name=roi)


@inject_client
def fetch_mesh_neuron(
    x,
    *,
    lod=1,
    with_synapses=False,
    dedup=True,
    missing_mesh="raise",
    parallel=True,
    max_threads=5,
    errors=None,
    seg_source=None,
    client=None,
    **kwargs,
):
    """Fetch neuron meshes as navis.Mesh.

    Requires additional packages depending on the mesh source.

    For DVID you need [`dvid-tools`](https://github.com/flyconnectome/dvid_tools):

        ``` shell
        pip3 install dvidtools
        ```

    For everything else you need [cloudvolume](https://github.com/seung-lab/cloud-volume):

        ``` shell
        pip3 install cloud-volume
        ```


    Parameters
    ----------
    x :             str | int | list-like | pandas.DataFrame | SegmentCriteria
                    Body ID(s). Multiple IDs can be provided as list-like or
                    DataFrame with "bodyId" or "bodyid" column.
    lod :           int
                    Level of detail. Higher `lod` = coarser. Ignored if mesh
                    source does not support LODs (e.g. for DVID).
    with_synapses : bool, optional
                    If True will download and attach synapses as `.connectors`.
    dedup :         bool, optional
                    Only relevant if `with_synapses=True`. In insect connectomes
                    synapses are polyadic (a presynapse connects to multiple
                    postsynapses). If True (default), presynapses are
                    de-duplicated. If False, each presynapse is reported once
                    per postsynaptic site it connects to.
    missing_mesh :  'raise' | 'warn' | 'skip'
                    What to do if no mesh is found for a given body ID:

                        "raise" (default) will raise an exception
                        "warn" will throw a warning but continue
                        "skip" will skip without any message

    parallel :      bool
                    If True, will use parallel threads to fetch data.
    max_threads :   int
                    Max number of parallel threads to use.
    errors :        "raise" | "log" | "ignore", optional
                    What to do if an individual neuron fails to fetch. Defaults
                    to "log", or to "raise" under `navis.config.strict`. Note
                    this governs *failures*; a body that simply has no mesh is
                    governed by `missing_mesh`.
    seg_source :    str | cloudvolume.CloudVolume, optional
                    Use this to override the segmentation source specified by
                    neuPrint.
    client :        neuprint.Client, optional
                    If `None` will try using global client.
    **kwargs
                    Will be passed to `cloudvolume.CloudVolume`.

    Returns
    -------
    navis.Neuronlist
                    Containing [`navis.Mesh`][]. Note that meshes are
                    resized to raw voxel size to match other spatial data from
                    neuprint (synapses, skeletons, etc).

    """
    if isinstance(x, pd.DataFrame):
        if "bodyId" in x.columns:
            x = x["bodyId"].values
        elif "bodyid" in x.columns:
            x = x["bodyid"].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    # Extract source
    if not seg_source:
        seg_source = get_seg_source(client=client)

    if not seg_source:
        raise ValueError(
            "Segmentation source could not be automatically "
            "determined. Please provide via `seg_source`."
        )

    if isinstance(seg_source, str) and seg_source.startswith("dvid"):
        try:
            import dvid as dv
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "This looks like a DVID mesh source. For this we "
                "need the `dvid-tools` library:\n"
                "  pip3 install dvidtools -U"
            )
        o = urlparse(seg_source.replace("dvid://", ""))
        server = f"{o.scheme}://{o.netloc}"
        node = o.path.split("/")[1]

        if lod is not None:
            logger.warning(
                "This dataset does not support LODs. "
                "Will ignore the `lod` argument. "
                "You can silence this warning by setting `lod=None`."
            )
            lod = None
    else:
        try:
            from cloudvolume import CloudVolume
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "You need to install the `cloudvolume` library"
                "to fetch meshes from this mesh source:\n"
                "  pip3 install cloud-volume -U"
            )
        # Initialize volume
        if isinstance(seg_source, CloudVolume):
            vol = seg_source
        else:
            defaults = dict(use_https=True, progress=False)
            defaults.update(kwargs)
            vol = CloudVolume(seg_source, **defaults)

            # Check if vol.mesh.get has a lod argument
            if lod is not None and "lod" not in vol.mesh.get.__code__.co_varnames:
                logger.warning(
                    "This dataset does not have multi-resolution meshes and "
                    "the `lod` parameter will be ignored. "
                    "You can silence this warning by setting `lod=None`."
                )
                lod = None

    if isinstance(x, NeuronCriteria):
        query = x
        wanted_ids = None
    else:
        query = NeuronCriteria(bodyId=x, client=client)
        wanted_ids = utils.make_iterable(x)

    # Fetch names, etc
    meta = fetch_neurons(query, client=client, omit_rois=True)

    if meta.empty:
        raise ValueError("No neurons matching the given criteria found!")
    elif not isinstance(wanted_ids, type(None)):
        miss = wanted_ids[~np.isin(wanted_ids, meta.bodyId.values)]
        if len(miss):
            logger.warning(
                f"Skipping {len(miss)} body IDs that were not found: "
                f'{", ".join(miss.astype(str))}'
            )

    # Apply a small number of potential fixes to the meta data
    meta = _fix_meta(meta)

    if isinstance(seg_source, str) and seg_source.startswith("dvid"):
        # Fetch the meshes
        nl = dv.get_meshes(
            meta.bodyId.values,
            on_error=missing_mesh,
            output="navis",
            progress=meta.shape[0] > 1 and not config.pbar_hide,
            max_threads=1 if not parallel else max_threads,
            server=server,
            node=node,
        )
    else:
        # `None` here is either a body that has no mesh (see `missing_mesh`) or
        # one that failed outright (see `errors`) - neither belongs in the list.
        nl = [
            n
            for n in fetch_parallel(
                __fetch_mesh,
                meta.bodyId.values,
                errors=errors,
                parallel=parallel,
                max_threads=max_threads,
                vol=vol,
                lod=lod,
                missing_mesh=missing_mesh,
            )
            if n is not None
        ]

    nl = NeuronList(nl)

    # Add meta data
    for col in ("instance", "size", "status", "statusLabel", "somaLocation", "somaRadius"):
        if col not in meta.columns:
            meta[col] = None
    instances = meta.set_index("bodyId").instance.to_dict()
    sizes = meta.set_index("bodyId")["size"].to_dict()
    status = meta.set_index("bodyId").status.to_dict()
    statuslabel = meta.set_index("bodyId").statusLabel.to_dict()
    somalocs = meta.set_index("bodyId").somaLocation.to_dict()
    radii = meta.set_index("bodyId").somaRadius.to_dict()

    for n in nl:
        n.name = instances[n.id]
        n.status = status[n.id]
        n.statusLabel = statuslabel[n.id]
        n.n_voxels = sizes[n.id]
        n.somaLocation = somalocs[n.id]

        # Meshes come out in units (e.g. nanometers) but most other data (synapses,
        # skeletons, etc) come out in voxels, we will therefore scale meshes to voxels
        n.vertices /= np.array(client.meta["voxelSize"]).reshape(1, 3)
        n.units = _voxel_size_to_units(client)

        if n.somaLocation:
            if radii[n.id]:
                n.soma_radius = radii[n.id] / n.units.to("nm").magnitude
            else:
                n.soma_radius = None
            n.soma_pos = n.somaLocation

    if with_synapses:
        # Fetch synapses
        syn = fetch_synapses(
            meta.bodyId.values,
            synapse_criteria=SynapseCriteria(primary_only=True, client=client),
            dedup=dedup,
            client=client,
        )

        for n in nl:
            this_syn = syn[syn.bodyId == n.id]
            if not this_syn.empty:
                # Keep only relevant columns
                n.connectors = this_syn[
                    ["type", "x", "y", "z", "roi", "confidence"]
                ].copy()

    # Make an effort to retain the original order
    if not isinstance(x, NeuronCriteria) and not nl.empty:
        nl = nl.idx[np.asarray(x)[np.isin(x, nl.id)]]

    return nl


def __fetch_mesh(bodyId, *, vol, lod, missing_mesh="raise"):
    """Fetch a single mesh (+ synapses) and construct navis Mesh."""
    # Fetch mesh
    import cloudvolume

    try:
        if lod is None:
            mesh = vol.mesh.get(bodyId)
        else:
            mesh = vol.mesh.get(bodyId, lod=lod)
    except cloudvolume.exceptions.MeshDecodeError as err:
        if "not found" in str(err):
            if missing_mesh in ["warn", "skip"]:
                if missing_mesh == "warn":
                    logger.warning(f"No mesh found for {bodyId}")
                return
            else:
                raise
        else:
            raise

    # Make sure we don't pass through a {bodyId: MeshObject} dictionary
    if isinstance(mesh, dict):
        mesh = mesh[bodyId]

    n = Mesh(mesh)
    n.lod = lod
    n.id = bodyId

    return n


@inject_client
def fetch_skeletons(
    x,
    *,
    with_synapses=False,
    dedup=True,
    heal=False,
    missing_swc="raise",
    parallel=True,
    max_threads=5,
    errors=None,
    client=None,
):
    """Fetch neuron skeletons as navis.Skeletons.

    Notes
    -----
    Synapses will be attached to the closest node in the skeleton.

    Parameters
    ----------
    x :             str | int | list-like | pandas.DataFrame | SegmentCriteria
                    Body ID(s). Multiple Ids can be provided as list-like or
                    DataFrame with "bodyId"  or "bodyid" column.
    with_synapses : bool, optional
                    If True will also attach synapses as `.connectors`.
    dedup :         bool, optional
                    Only relevant if `with_synapses=True`. In insect connectomes
                    synapses are polyadic (a presynapse connects to multiple
                    postsynapses). If True (default), presynapses are
                    de-duplicated. If False, each presynapse is reported once
                    per postsynaptic site it connects to.
    heal :          bool | int | float, optional
                    If True, will automatically heal fragmented skeletons using
                    neuprint-python's `heal_skeleton` function. Pass a float
                    or an int to limit the max distance at which nodes are
                    allowed to be re-connected (requires neuprint-python >= 0.4.11).
    missing_swc :   'raise' | 'warn' | 'skip'
                    What to do if no skeleton is found for a given body ID:

                      - "raise" (default) will raise an exception
                      - "warn" will throw a warning but continue
                      - "skip" will skip without any message

    parallel :      bool
                    If True, will use parallel threads to fetch data.
    max_threads :   int
                    Max number of parallel threads to use.
    errors :        "raise" | "log" | "ignore", optional
                    What to do if an individual neuron fails to fetch. Defaults
                    to "log", or to "raise" under `navis.config.strict`. Note
                    this governs *failures*; a body that simply has no SWC is
                    governed by `missing_swc`.
    client :        neuprint.Client, optional
                    If `None` will try using global client.

    Returns
    -------
    navis.Neuronlist

    """
    if isinstance(x, pd.DataFrame):
        if "bodyId" in x.columns:
            x = x["bodyId"].values
        elif "bodyid" in x.columns:
            x = x["bodyid"].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')

    if isinstance(x, NeuronCriteria):
        query = x
        wanted_ids = None
    else:
        query = NeuronCriteria(bodyId=x, client=client)
        wanted_ids = utils.make_iterable(x)

    # Fetch names, etc
    meta = fetch_neurons(query, client=client, omit_rois=True)

    if meta.empty:
        raise ValueError("No neurons matching the given criteria found!")
    elif not isinstance(wanted_ids, type(None)):
        miss = wanted_ids[~np.isin(wanted_ids, meta.bodyId.values)]
        if len(miss):
            logger.warning(
                f"Skipping {len(miss)} body IDs that were not found: "
                f'{", ".join(miss.astype(str))}'
            )

    # Apply a small number of potential fixes to the meta data
    meta = _fix_meta(meta)

    # `None` here is either a body that has no SWC (see `missing_swc`) or one
    # that failed outright (see `errors`) - neither belongs in the list.
    nl = NeuronList(
        [
            n
            for n in fetch_parallel(
                __fetch_skeleton,
                list(meta.itertuples()),
                labels=meta.bodyId.values,
                errors=errors,
                parallel=parallel,
                max_threads=max_threads,
                initializer=_init_worker,
                initargs=(client,),
                client=client,
                with_synapses=with_synapses,
                dedup=dedup,
                missing_swc=missing_swc,
                heal=heal,
            )
            if n is not None
        ]
    )

    # Make an effort to retain the original order
    if not isinstance(x, NeuronCriteria) and not nl.empty:
        nl = nl.idx[np.asarray(x)[np.isin(x, nl.id)]]

    return nl


def __fetch_skeleton(
    r, client, with_synapses=True, dedup=True, missing_swc="raise", heal=False, max_distance=None
):
    """Fetch a single skeleton + synapses and construct navis Skeleton."""
    # Use this thread's own client (and hence session) if we have one - see
    # `_init_worker`. Falls back to the passed client when called directly.
    client = getattr(_thread_local, "client", client)

    # Fetch skeleton SWC
    try:
        data = client.fetch_skeleton(r.bodyId, format="pandas", heal=heal)
    except HTTPError as err:
        if err.response.status_code == 400:
            if missing_swc in ["warn", "skip"]:
                if missing_swc == "warn":
                    logger.warning(f"No SWC found for {r.bodyId}")
                return
            else:
                raise
        else:
            raise

    # Generate neuron
    n = Skeleton(
        data, units=_voxel_size_to_units(client)
    )

    # Reduce precision
    n._nodes = n._nodes.astype(
        {
            "node_id": np.int32,
            "parent_id": np.int32,
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
            "radius": np.float32,
        }
    )

    # Add some missing meta data
    n.id = r.bodyId

    if hasattr(r, "instance"):
        n.name = r.instance

    n.n_voxels = r.size
    n.status = r.status

    # Make KDE tree for NN
    if r.somaLocation or with_synapses:
        tree = neuron2KDTree(n, data="nodes")

    # Set soma
    if r.somaLocation:
        d, i = tree.query([r.somaLocation])
        n.soma = int(n.nodes.iloc[i[0]].node_id)
        n.soma_radius = r.somaRadius if r.somaRadius else "radius"
    else:
        n.soma = None

    if with_synapses:
        # Fetch synapses
        syn = fetch_synapses(
            r.bodyId,
            synapse_criteria=SynapseCriteria(primary_only=True, client=client),
            dedup=dedup,
            client=client,
        )

        if not syn.empty:
            # Process synapses
            syn["connector_id"] = syn.index.values
            locs = syn[["x", "y", "z"]].values
            d, i = tree.query(locs)

            syn["node_id"] = n.nodes.iloc[i].node_id.values
            syn["x"] = locs[:, 0]
            syn["y"] = locs[:, 1]
            syn["z"] = locs[:, 2]

            # Keep only relevant columns
            syn = syn[
                ["connector_id", "node_id", "type", "x", "y", "z", "roi", "confidence"]
            ]

            # Manually make the "roi" column of the synapse table into a
            # categorical to save some memory
            syn["roi"] = syn.roi.astype("category")

            n.connectors = syn

    return n


def remove_soma_hairball(
    x: "core.Skeleton", radius: float = 500, inplace: bool = False
):
    """Remove hairball around soma.

    Parameters
    ----------
    x :         core.Skeleton
    radius :    float
                Radius around the soma to check for hairball

    Returns
    -------
    Skeleton
                If inplace=False.
    """
    if not inplace:
        x = x.copy()
    if not x.soma:
        if not inplace:
            return x
        return
    # Get all nodes within given radius of soma nodes
    soma_loc = x.nodes.set_index("node_id").loc[[x.soma], ["x", "y", "z"]].values
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

    to_keep = x.nodes.loc[~x.nodes.node_id.isin(to_remove), "node_id"].values

    # Move soma if required
    if x.soma in to_remove:
        x.soma = list(to_check & set(to_keep))[0]

    subset_neuron(x, to_keep, inplace=True)

    if not inplace:
        return x


@inject_client
def get_seg_source(*, client=None):
    """Get segmentation source for given client+dataset."""
    # First try to fetch the scene for the neuroglancer
    url = f"{client.server}/api/npexplorer/nglayers/{client.dataset}.json"

    r = client.session.get(url)
    try:
        r.raise_for_status()
        scene = r.json()
        segs = [s for s in scene["layers"] if s.get("type") == "segmentation"]
    except BaseException:
        segs = []

    # If we didn't find a `dataset.json`, will check the client's meta data for a seg source
    if not segs:
        segs = [
            s
            for s in client.meta["neuroglancerMeta"]
            if s.get("dataType") == "segmentation"
        ]

    if not len(segs):
        return None

    # Check if any segmentation source matches our dataset exactly
    named_segs = [s for s in segs if s.get("name") == client.dataset]
    if len(named_segs):
        segs = named_segs

    # If there are multiple segmentation layers, select the first entry
    seg_source = segs[0]["source"]

    # If there are multiple segmentation sources for
    # the layer we picked, select the first source.
    if isinstance(seg_source, list):
        seg_source = seg_source[0]

    # If it's a dict like {'source': url, 'subsources'...},
    # select the url.
    if isinstance(seg_source, dict):
        seg_source = seg_source["url"]

    if not isinstance(seg_source, str):
        e = f"Could not understand segmentation source: {seg_source}"
        raise RuntimeError(e)

    if len(segs) > 1:
        logger.warning(
            f"{len(segs)} segmentation sources found. Using the "
            f'first entry: "{seg_source}"'
        )

    return seg_source


def _fix_meta(meta):
    """Fix a number of potential issues with neuprint metadata."""
    # Make sure there is a somaLocation and somaRadius column
    if "somaLocation" not in meta.columns:
        meta["somaLocation"] = None
    if "somaRadius" not in meta.columns:
        meta["somaRadius"] = None
    # Backfill from tosomaLocation if available
    if "tosomaLocation" in meta.columns:
        meta["somaLocation"] = meta.somaLocation.fillna(meta.tosomaLocation)

    # Fix coordinate columns
    for col in ("somaLocation", "tosomaLocation"):
        if col not in meta.columns:
            continue

        # Prevent an issue when coordinates are returned as string:
        # "Point{SpatialRefId=9157, X=48556.000000, Y=43018.000000, Z=37620.000000}"
        if isinstance(meta[col].values[0], str):
            if meta[col].values[0].startswith("Point"):
                meta[col] = (
                    meta[col]
                    .str.extract(r"X=(\d+.\d+), Y=(\d+.\d+), Z=(\d+.\d+)")
                    .values.astype(float)
                    .tolist()
                )

        # Prevent an issue when coordinates are returned as dict
        if isinstance(meta[col].values[0], dict):
            if "coordinates" in meta[col].values[0]:
                meta[col] = meta[col].apply(lambda x: x["coordinates"])

    return meta


def _voxel_size_to_units(client):
    """Extract voxel size from client and convert to pint units."""
    units = client.meta["voxelUnits"]
    voxel_size = np.asarray(client.meta["voxelSize"])

    if (voxel_size == voxel_size[0]).all():
        return f"{voxel_size[0]} {units}"
    else:
        return [f"{vs} {units}" for vs in voxel_size]
