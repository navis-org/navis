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

import numpy as np
import pandas as pd
import sparsecubes

from typing import Optional, Union, List

from .. import config, graph, core, utils, meshes
from ..core import schema
from .utils import sample_points_uniform

# Set up logging
logger = config.get_logger(__name__)

__all__ = ["downsample_neuron"]


@utils.map_neuronlist(desc="Downsampling", allow_parallel=True)
def downsample_neuron(
    x: "core.NeuronObject",
    downsampling_factor: Union[int, float],
    inplace: bool = False,
    preserve_nodes: Optional[List[int]] = None,
    method: str = "simple",
) -> Optional["core.NeuronObject"]:
    """Downsample neuron(s) by a given factor.

    For skeletons: preserves root, leafs, branchpoints by default. Preservation
    of nodes with synapses can be toggled - see `preserve_nodes` parameter.
    Use `downsampling_factor=float('inf')` to get a skeleton consisting only
    of root, branch and end points. Connectors and tags sitting on a node that
    does not survive are moved onto the geodesically nearest node that does.

    Parameters
    ----------
    x :                     single neuron | NeuronList
                            Neuron(s) to downsample. Note that for Meshes
                            we use the first available backend.
    downsampling_factor :   int | float('inf')
                            Factor by which downsample. For Skeleton, Dotprops
                            and Meshes this reduces the node, point
                            and face count, respectively. For Voxels it
                            reduces the dimensions by given factor.
    preserve_nodes :        str | list, optional
                            Can be either list of node IDs to exclude from
                            downsampling or a string to a DataFrame attached
                            to the neuron (e.g. "connectors"). DataFrame must
                            have `node_id` column. Only relevant for
                            Skeletons.
    method :                "simple" | "uniform" | "fps" | "decimate"
                            How to pick which points to keep. Only relevant for
                            Dotprops (ignored for all other neuron types):
                              - "simple" (default): take every N-th point. Fast
                                but not spatially uniform.
                              - "uniform": draw a uniform sample, automatically
                                choosing farthest-point sampling or decimation
                                depending on size.
                              - "fps": force farthest-point sampling (most
                                uniform coverage but O(N * size)).
                              - "decimate": force decimation (fast, scales to
                                large point clouds).
    inplace :               bool, optional
                            If True, will modify original neuron. If False, we
                            will operate and return o a copy.

    Returns
    -------
    Skeleton/Dotprops/Voxels/NeuronList
                            Same datatype as input.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n_ds = navis.downsample_neuron(n,
    ...                                downsampling_factor=5,
    ...                                inplace=False)
    >>> n.n_nodes > n_ds.n_nodes
    True

    See Also
    --------
    [`navis.resample_skeleton`][]
                             This function resamples a neuron to given
                             resolution. This will change node IDs!
    [`navis.simplify_mesh`][]
                             This is the function used for `Meshes`. Use
                             directly for more control of the simplification.

    """
    if downsampling_factor <= 1:
        raise ValueError("Downsampling factor must be greater than 1.")

    if not inplace:
        x = x.copy()

    if isinstance(x, core.Skeleton):
        _ = _downsample_treeneuron(
            x, downsampling_factor=downsampling_factor, preserve_nodes=preserve_nodes
        )
    elif isinstance(x, core.Dotprops):
        _ = _downsample_dotprops(
            x, downsampling_factor=downsampling_factor, method=method
        )
    elif isinstance(x, core.Voxels):
        _ = _downsample_voxels(x, downsampling_factor=downsampling_factor)
    elif isinstance(x, core.Mesh):
        _ = meshes.simplify_mesh(x, F=1 / downsampling_factor, inplace=True)
    else:
        raise TypeError(f'Unable to downsample data of type "{type(x)}"')

    return x


def _downsample_voxels(x, downsampling_factor, agg="max"):
    """Downsample voxels.

    Pools voxels into `downsampling_factor`-sized cells straight off the sparse
    voxels. Note this never allocates the dense grid - which matters because a
    neuron sparse enough to be worth downsampling is exactly the kind whose grid
    may not fit in memory (see `navis.config.max_grid_size`).
    """
    assert isinstance(x, core.Voxels)

    # Pooling voxels into coarse cells only makes sense for whole factors.
    # Note we must use the *same* integer for the units below - scaling them by
    # the requested factor while pooling by a different one would resize the
    # neuron.
    factor = int(round(downsampling_factor))
    if factor != downsampling_factor:
        logger.warning(
            f"Voxels can only be downsampled by whole factors - rounding "
            f"{downsampling_factor} to {factor}."
        )

    voxels, values = sparsecubes.downsample(
        x.voxels, factor, values=x.values, agg=agg
    )

    # The coarse grid's canvas shrinks by the same factor
    shape = np.ceil(np.array(x.shape) / factor).astype(int)

    x._replace_voxels(voxels, values, inplace=True)
    x._canvas_shape = tuple(shape)

    # Voxels are now bigger, so the units have to grow with them
    x.units = x.units_xyz * factor


def _downsample_dotprops(x, downsampling_factor, method="simple"):
    """Downsample Dotprops.

    Parameters
    ----------
    method :    "simple" | "uniform" | "fps" | "decimate"
                How to pick the points to keep:
                  - "simple": take every N-th point. Fast but not spatially
                    uniform (order-dependent).
                  - "uniform": draw a uniform sample, automatically using
                    farthest-point sampling or decimation depending on size.
                  - "fps": force farthest-point sampling (most uniform coverage
                    but O(N * size)).
                  - "decimate": force decimation (fast and scales to large
                    point clouds).
                See [`navis.sampling.utils.sample_points_uniform`][] for details
                on the last three.
    """
    assert isinstance(x, core.Dotprops)

    # Can't downsample if no points
    if isinstance(x._points, type(None)):
        return

    # If not enough points
    if x._points.shape[0] <= downsampling_factor:
        return

    # Generate a mask
    if method == "simple":
        mask = np.arange(0, x._points.shape[0], int(downsampling_factor))
    elif method in ("uniform", "fps", "decimate"):
        # "uniform" lets sample_points_uniform pick the strategy; "fps"/"decimate"
        # force it.
        sampler = "auto" if method == "uniform" else method
        mask = sample_points_uniform(
            x._points,
            int(x._points.shape[0] // downsampling_factor),
            output="mask",
            method=sampler,
        )
    else:
        raise ValueError(f"Unknown (down-)sampling method: {method}")

    # Make sure the tangent vectors exist before we select. This also triggers
    # re-calculation which is necessary for two reasons:
    # 1. Vectors will change dramatically if they have to be recalculated from
    #    the downsampled dotprops.
    # 2. There might not be enough points left after downsampling given the
    #    original k.
    if isinstance(x._vect, type(None)) and x.k:
        x.recalculate_tangents(k=x.k, inplace=True)

    # Taking points away is a selection, not a rebuild - so the schema carries
    # the vectors, the alphas, anything attached and the soma's point index, and
    # takes the connectors along with the points they sit on.
    axis = schema.get_axis(x, "points")
    schema.apply_selection(x, axis, schema.resolve_selection(x, axis, mask))


@utils.rebuilds("nodes")
def _downsample_treeneuron(x, downsampling_factor, preserve_nodes):
    """Downsample Skeleton.

    A rebuild rather than a selection, in the one sense that matters: the nodes
    that survive are genuinely the old ones, but the ones that go do not take
    anything with them - thinning a slab does not remove that stretch of the
    neuron, so a connector sitting on it still means something. Hence the
    `Rebuild` handed back, which says both where a reference should now point
    and - because here we really can - which new nodes are which old ones.
    """
    assert isinstance(x, core.Skeleton)

    if not isinstance(preserve_nodes, type(None)):
        if isinstance(preserve_nodes, str):
            table = getattr(x, preserve_nodes)
            if not isinstance(table, pd.DataFrame):
                raise TypeError(
                    f'Expected "{preserve_nodes}" to be a '
                    f"DataFrame - got {type(table)}"
                )
            if "node_id" not in table.columns:
                raise IndexError(
                    f'DataFrame {preserve_nodes} has no "node_id"' " column."
                )

            preserve_nodes = table["node_id"].values

        if not isinstance(preserve_nodes, (list, set, np.ndarray)):
            raise TypeError(
                'Expected "preserve_nodes" to be list-like, got '
                f'"{type(preserve_nodes)}"'
            )

    if x.nodes.shape[0] <= 1:
        logger.warning(f"Neuron {x.id} has no nodes. Skipping.")
        return x, schema.Rebuild()

    list_of_parents = {
        n: p for n, p in zip(x.nodes.node_id.values, x.nodes.parent_id.values)
    }
    list_of_parents[-1] = -1  # type: ignore  # doesn't know that node_id is int

    if "type" not in x.nodes:
        graph.classify_nodes(x)

    selection = x.nodes.type != "slab"

    if utils.is_iterable(preserve_nodes):
        selection = selection | x.nodes.node_id.isin(preserve_nodes)  # type: ignore

    fix_points = x.nodes[selection].node_id.values

    # Membership of `fix_points` is tested for every node on every walk below;
    # a set makes that O(1) instead of a linear scan over the array.
    fix_set = set(fix_points.tolist())

    # Add soma node(s)
    if not isinstance(x.soma, type(None)):
        soma = utils.make_iterable(x.soma)
        for s in soma:
            if s not in fix_set:
                fix_points = np.append(fix_points, s)
                fix_set.add(s)

    # Walk from all fix points to the root - jump N nodes on the way
    new_parents = {}

    for en in fix_points:
        this_node = en

        while True:
            stop = False
            new_p = list_of_parents[this_node]
            if new_p >= 0:
                i = 0
                while i < downsampling_factor:
                    if new_p < 0 or new_p in fix_set:
                        new_parents[this_node] = new_p
                        stop = True
                        break
                    new_p = list_of_parents[new_p]
                    i += 1

                if stop is True:
                    break
                else:
                    new_parents[this_node] = new_p
                    this_node = new_p
            else:
                new_parents[this_node] = -1  # type: ignore
                break

    # Subset to kept nodes
    new_nodes = x.nodes[x.nodes.node_id.isin(list(new_parents.keys()))].copy()

    # Assign new parent IDs
    new_nodes["parent_id"] = new_nodes.node_id.map(new_parents).astype(int)

    logger.debug(f"Nodes before/after: {len(x.nodes)}/{len(new_nodes)}")

    # Where a reference to a node that is about to go should now point. Worked
    # out first - the search needs the full topology, which assigning the
    # thinned table is exactly what ends.
    snap = _snap_dropped(x, kept=set(new_parents), axis=schema.get_axis(x, "nodes"))

    x.nodes = new_nodes

    # This is essential -> otherwise e.g. graph.neuron2graph will fail
    x.nodes.reset_index(inplace=True, drop=True)

    x._clear_temp_attr()

    # Every surviving node is one of the old ones, row for row, so identity can
    # be claimed here - which is what lets attached data be carried.
    return x, schema.Rebuild(snap=snap, kept=x.nodes.node_id.values)


def _snap_dropped(x, kept, axis):
    """Where each node downsampling is about to remove sends its references.

    Connectors, tags and the soma refer to nodes by ID, so without this they
    would keep pointing at nodes that are no longer in the table - a neuron that
    looks fine until something tries to look one of them up.

    They are moved rather than dropped because downsampling only thins slabs: it
    does not remove any part of the neuron, so the geodesically nearest survivor
    is the same stretch of the same branch, just sampled more coarsely. Use
    `preserve_nodes="connectors"` to pin connectors exactly instead.

    Only the nodes something actually names are searched for: the graph search
    is priced by how many places we ask about, and a skeleton with no connectors,
    tags or soma has nothing to ask.

    Must run *before* the thinned node table is assigned - finding the nearest
    survivor needs the original topology.

    """
    referenced = schema.referenced_values(x, axis)
    dropped = referenced[~np.isin(referenced, list(kept))]
    if not len(dropped):
        return None

    nearest, _ = graph._geodesic_nearest(
        x,
        targets=np.array(list(kept)),
        query=dropped,
        weight="weight",
        directed=False,
    )

    # Every fragment keeps its root, so there is always something to snap to and
    # `-1` (unreachable) should not happen - but if it somehow does, leaving the
    # ID alone beats silently moving the connector to an arbitrary node.
    reachable = nearest >= 0
    return dropped[reachable], nearest[reachable]
