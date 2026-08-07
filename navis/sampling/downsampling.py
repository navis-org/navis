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

from .. import config, core, utils, meshes
from ..core import schema
from .utils import sample_points_uniform

# Set up logging
logger = config.get_logger(__name__)

__all__ = ["downsample_neuron"]

# The methods that read `downsampling_factor` as a distance rather than a count.
_GEOMETRIC = ("rdp", "vw")


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
    Connectors and tags sitting on a node that does not survive are moved onto
    the geodesically nearest node that does. Skeletons can also be thinned by
    *shape* rather than by counting - see the `method` parameter.

    Parameters
    ----------
    x :                     single neuron | NeuronList
                            Neuron(s) to downsample. Note that for Meshes
                            we use the first available backend.
    downsampling_factor :   int | float | str
                            How much to downsample. What this means depends on
                            `method`:
                              - the counting methods read it as a **factor**
                                > 1: for Skeletons, Dotprops and Meshes it
                                divides the node, point and face count,
                                respectively, and for Voxels it divides the
                                dimensions. `float('inf')` reduces a Skeleton
                                to just its root, branch and end points.
                              - the shape-aware Skeleton methods read it as a
                                **distance tolerance** in the neuron's own
                                units - roughly "how far the simplified neuron
                                may stray from this one". There is no lower
                                bound: `0` is a legitimate (if barely
                                simplifying) request. If the neuron has
                                `.units` you can pass a string such as
                                "1 micron".
    preserve_nodes :        str | list, optional
                            Can be either list of node IDs to exclude from
                            downsampling or a string to a DataFrame attached
                            to the neuron (e.g. "connectors"). DataFrame must
                            have `node_id` column. Only relevant for
                            Skeletons.
    method :                str
                            How to pick what to keep. Which values are allowed
                            depends on the type of neuron; passing one that
                            does not apply is an error rather than ignored.

                            For all neurons:
                              - "simple" (default): take every N-th node/point/
                                face. Fast but pays no attention to geometry.

                            For Skeletons, thinning by shape instead:
                              - "rdp": Ramer-Douglas-Peucker. Drops a node
                                unless removing it would move the traced path by
                                more than the tolerance, so long straight
                                stretches collapse to their two ends while a
                                tight curve keeps every node it needs.
                              - "vw": Visvalingam-Whyatt. Repeatedly removes
                                whichever node adds least area to the path.
                                Sheds detail more evenly than "rdp" under
                                aggressive simplification, which keeps a neurite
                                looking like itself where "rdp" will happily
                                keep one spike and flatten everything around it.

                            For Dotprops, sampling the point cloud instead:
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

    Notes
    -----
    Downsampling a skeleton **shortens it**. The nodes that survive keep their
    original coordinates, so the edges that replace a dropped chain cut its
    corners, and `.cable_length` drops accordingly - by around 5-6% on the
    example neuron at any of the settings below, and by more the harder you
    simplify. Use [`navis.resample_skeleton`][] if you need a node count you
    choose *and* the cable length left intact.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n_ds = navis.downsample_neuron(n,
    ...                                downsampling_factor=5,
    ...                                inplace=False)
    >>> n.n_nodes > n_ds.n_nodes
    True

    Thinning by shape rather than by count. The example neurons are in 8nm
    voxels, so this lets the skeleton stray by up to half a micron:

    >>> n_rdp = navis.downsample_neuron(n, "0.5 micron", method="rdp")
    >>> n_rdp.n_nodes < n_ds.n_nodes
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
    _check_method(x, method)

    # `downsampling_factor` means different things per method, so it is checked
    # per method rather than up front: a *factor* below 1 asks for more nodes
    # than there were, while a *tolerance* below 1 is an ordinary request.
    if method in _GEOMETRIC:
        downsampling_factor = x.map_units(downsampling_factor, on_error="raise")
        # Checked here rather than left to fastcore, which aborts a Rust worker
        # thread on a negative tolerance instead of raising.
        if not np.isfinite(downsampling_factor) or downsampling_factor < 0:
            raise ValueError(
                f'`downsampling_factor` is read as a distance tolerance for '
                f'method "{method}" and must be finite and >= 0, got '
                f"{downsampling_factor}."
            )
    elif downsampling_factor <= 1:
        raise ValueError("Downsampling factor must be greater than 1.")

    if not inplace:
        x = x.copy()

    if isinstance(x, core.Skeleton):
        _ = _downsample_treeneuron(
            x,
            downsampling_factor=downsampling_factor,
            preserve_nodes=preserve_nodes,
            method=method,
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


def _check_method(x, method):
    """Reject a `method` the given neuron has no way of honouring.

    `method` used to be read only by `Dotprops` and silently ignored by
    everything else, which was harmless while it named a way of picking points.
    It is not harmless now that it also names a way of picking *nodes*: quietly
    ignoring `method="rdp"` on a mesh would hand back something simplified by
    face count and call it shape-aware.

    "simple" - take every N-th of whatever the neuron is made of - is the only
    one that means the same thing for a skeleton, a point cloud, a voxel grid
    and a mesh, so it is the only one they all have. The rest are specific to
    how the data is arranged: you cannot ask a point cloud where it bends, and
    a skeleton has no use for farthest-point sampling.
    """
    if isinstance(x, core.Skeleton):
        allowed = ("simple", "rdp", "vw")
    elif isinstance(x, core.Dotprops):
        allowed = ("simple", "uniform", "fps", "decimate")
    elif isinstance(x, (core.Voxels, core.Mesh)):
        allowed = ("simple",)
    else:
        # Not something we can downsample at all. Let the dispatch say so with
        # its `TypeError` rather than complaining here about `method` - naming
        # the methods a type does not have would only imply it had the others.
        return

    if method not in allowed:
        raise ValueError(
            f'Unknown (down-)sampling method "{method}" for '
            f"{type(x).__name__}. Valid options: "
            f"{', '.join(repr(m) for m in allowed)}."
        )


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
def _downsample_treeneuron(x, downsampling_factor, preserve_nodes, method="simple"):
    """Downsample Skeleton.

    A rebuild rather than a selection, in the one sense that matters: the nodes
    that survive are genuinely the old ones, but the ones that go do not take
    anything with them - thinning a slab does not remove that stretch of the
    neuron, so a connector sitting on it still means something. Hence the
    `Rebuild` handed back, which says both where a reference should now point
    and - because here we really can - which new nodes are which old ones.

    All three methods run in `navis-fastcore` and share one contract: roots,
    branch points and leafs always survive, so whatever `method` and however
    hard it is pushed the result is still the same neuron - only its unbranched
    stretches are sampled more coarsely.
    """
    assert isinstance(x, core.Skeleton)

    node_ids = x.nodes.node_id.values
    parent_ids = x.nodes.parent_id.values
    coords = x.nodes[["x", "y", "z"]].values

    # Real edge lengths rather than hop counts. These do not change which nodes
    # survive - only which survivor each dropped node's data goes to, i.e. the
    # `merged` map below. Measured in hops that answer is wrong wherever a
    # skeleton is unevenly sampled, which is most skeletons, and it is the
    # answer connectors and tags ride on.
    weights = utils.fastcore.dag.parent_dist(
        node_ids, parent_ids, coords, root_dist=0
    )

    preserve = _preserve_ids(x, preserve_nodes)

    if method == "simple":
        # `float('inf')` is documented as "just the fix points" and fastcore
        # takes an int, so it has to be brought down to one. Any factor at least
        # as large as the node count already means the same thing - no chain is
        # longer than the whole skeleton - so this is exact rather than an
        # approximation. `max(..., 1)` covers the empty neuron, where the clamp
        # would otherwise land on the one factor fastcore rejects.
        factor = int(min(downsampling_factor, max(len(node_ids), 1)))
        new_ids, new_parents, _, node_map = utils.fastcore.downsample_skeleton(
            node_ids, parent_ids, factor, preserve=preserve, weights=weights
        )
    elif method == "rdp":
        new_ids, new_parents, _, node_map = utils.fastcore.simplify_rdp(
            node_ids,
            parent_ids,
            coords,
            downsampling_factor,
            preserve=preserve,
            weights=weights,
        )
    elif method == "vw":
        # fastcore's Visvalingam-Whyatt threshold is an *area*, i.e. in squared
        # coordinate units. navis takes the same distance as "rdp" and squares
        # it here so that `method` can be swapped without also rescaling the
        # number - and so that "1 micron" means one thing rather than two.
        new_ids, new_parents, _, node_map = utils.fastcore.simplify_vw(
            node_ids,
            parent_ids,
            coords,
            downsampling_factor**2,
            preserve=preserve,
            weights=weights,
        )
    else:
        # `_check_method` should have caught this, but the list it checks
        # against lives 300 lines up. Falling through to whichever branch
        # happens to be last is how a new method silently gets someone else's
        # algorithm - the very thing `_check_method` exists to prevent.
        raise ValueError(f"Unknown (down-)sampling method for Skeleton: {method}")

    # Every surviving node is one of the old ones, so this is a row selection
    # and only the parents change. Rows are taken in fastcore's order rather
    # than by a mask so that they line up with `new_ids` - which is what makes
    # the `kept` below a claim about *this* row and not merely about this ID.
    # `.take` rather than `.iloc[...].copy()`: it already hands back a fresh
    # frame, and without the `_is_copy` flag that would make the next line warn.
    new_nodes = x.nodes.take(pd.Index(node_ids).get_indexer(new_ids))
    new_nodes["parent_id"] = new_parents.astype(x.nodes.parent_id.dtype)

    logger.debug(f"Nodes before/after: {len(x.nodes)}/{len(new_nodes)}")

    x.nodes = new_nodes

    # This is essential -> otherwise e.g. graph.neuron2graph will fail
    x.nodes.reset_index(inplace=True, drop=True)

    x._clear_temp_attr()

    # Two claims, and fastcore can make both. `kept`: every surviving node is
    # one of the old ones, row for row, which is what lets attached data be
    # carried. `merged`: where each dropped node's *references* go - the nearer
    # end of the chain it sat on, measured in `weights`, i.e. the same stretch
    # of the same branch just sampled more coarsely. Use
    # `preserve_nodes="connectors"` to pin connectors exactly instead.
    return x, schema.Rebuild(kept=new_ids, merged=node_map)


def _preserve_ids(x, preserve_nodes):
    """The nodes downsampling must not drop, over and above the fix points.

    fastcore keeps roots, branch points and leafs on its own account; the soma
    is navis' own idea and has to be named, as does whatever the caller asked
    for.

    Returns `None` - fastcore's "nothing extra" - rather than an empty array,
    which it would otherwise have to build a mask for.
    """
    # Collected as arrays rather than appended to one list: `preserve_nodes` is
    # routinely a whole connector table's worth of IDs, and boxing those into
    # Python ints only to unbox them again costs more than the rest of this.
    parts = []

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

        # `np.asarray` of a set gives a 0-d object array, so that one case does
        # have to go through a list.
        parts.append(
            np.asarray(
                list(preserve_nodes)
                if isinstance(preserve_nodes, set)
                else preserve_nodes
            )
        )

    # Read once: `.soma` is not a cached property - it may be a *rule* for
    # finding the soma, and running `find_soma` twice over a large node table
    # costs more than everything else here put together.
    soma = x.soma
    if not isinstance(soma, type(None)):
        parts.append(utils.make_iterable(soma))

    if not parts:
        return None

    # fastcore raises on an ID it has never heard of, where navis' own
    # `isin`-based selection quietly ignored one. Keep quietly ignoring: a soma
    # or a connector's node can outlive the node table naming it (a hand-edited
    # neuron, a table swapped out from under one), and refusing to downsample at
    # all over a stale reference helps nobody.
    ids = np.unique(np.concatenate(parts))
    return ids[np.isin(ids, x.nodes.node_id.values)]
