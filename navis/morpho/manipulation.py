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


"""This module contains functions to analyse and manipulate neuron morphology."""

import warnings

import pandas as pd
import numpy as np
import networkx as nx
import trimesh as tm

from itertools import combinations
from typing import Union, Optional, Sequence, List, Set, Tuple, Callable
from typing_extensions import Literal

try:
    from pykdtree.kdtree import KDTree
except ModuleNotFoundError:
    from scipy.spatial import cKDTree as KDTree

import scipy.spatial

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse.csgraph import connected_components as _sparse_cc

from . import mmetrics, subset
from .. import graph, utils, config, core

# Set up logging
logger = config.get_logger(__name__)

# Knobs for the nearest-neighbour search in `_stitch_edges`:
# - the largest `k` we are willing to escalate to before falling back to an
#   exclusion query (see `_resolve_by_exclusion`)
# - the largest (nodes x k) neighbour matrix we allow ourselves to materialise;
#   queries are chunked to stay below this, which keeps memory flat in `k`
_KNN_MAX_K = 128
_KNN_MAX_ELEMENTS = 2_000_000

__all__ = sorted(
    [
        "prune_by_strahler",
        "stitch_skeletons",
        "split_axon_dendrite",
        "split_axon_dendrite_prop",
        "average_skeletons",
        "despike_skeleton",
        "guess_radius",
        "smooth_skeleton",
        "heal_skeleton",
        "cell_body_fiber",
        "break_fragments",
        "prune_twigs",
        "prune_at_depth",
        "drop_fluff",
        "combine_neurons",
    ]
)

NeuronObject = Union["core.NeuronList", "core.TreeNeuron"]


# Default colors for compartments (used in split_axon_dendrite)
_COMP_COLORS = {
    "axon": (178, 34, 34),
    "dendrite": (0, 0, 255),
    "cellbodyfiber": (50, 50, 50),
    "linker": (150, 150, 150),
}


@utils.map_neuronlist(desc="Pruning", allow_parallel=True)
@utils.meshneuron_skeleton(method="subset")
def cell_body_fiber(
    x: NeuronObject,
    method: Union[Literal["longest_neurite"], Literal["betweenness"]] = "betweenness",
    reroot_soma: bool = True,
    heal: bool = True,
    threshold: float = 0.95,
    inverse: bool = False,
    inplace: bool = False,
):
    """Prune neuron to its cell body fiber.

    Here, "cell body fiber" (CBF) refers to the tract connecting the soma to the
    backbone in unipolar neuron (common in e.g. insects). This function works
    best for typical neurons with clean skeletons.

    Parameters
    ----------
    x :             TreeNeuron | MeshNeuron | NeuronList
    method :        "longest_neurite" | "betweenness"
                    The method to use:
                      - "longest_neurite" assumes that the main branch point
                        is where the two largest branches converge
                      - "betweenness" uses centrality to determine the point
                        which most shortest paths traverse
    reroot_soma :   bool
                    If True (recommended) and neuron has a soma, it will be
                    rerooted to its soma.
    heal :          bool
                    If True (recommended), will heal fragmented neurons.
                    Fragmented neurons are not guaranteed to have correct CBFs.
    threshold :     float [0-1]
                    For method "betweenness" only: threshold at which to cut the
                    cell body fiber. Lower thresholds produce longer CBFs.
    inverse :       bool
                    If True, will instead *remove* the cell body fiber.
    inplace :       bool, optional
                    If False, pruning is performed on copy of original neuron
                    which is then returned.

    Returns
    -------
    TreeNeuron/List
                    Pruned neuron(s). Neurons without branches (i.e. w/ a single
                    long segment) will be returned unaltered.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> cbf = navis.cell_body_fiber(n, inplace=False)
    >>> # Neuron now has only a single segment from the soma to the main fork
    >>> len(cbf.segments)
    1

    See Also
    --------
    [`navis.find_main_branchpoint`][]
                    Find the main branch point.

    [`navis.betweeness_centrality`][]
                    Calculate the per-node betweeness centrality. This is used
                    under the hood for `method='betweeness'`.

    """
    utils.eval_param(
        method, "method", allowed_values=("longest_neurite", "betweenness")
    )

    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f"Expected TreeNeuron(s), got {type(x)}")

    if not inplace:
        x = x.copy()

    if x.n_trees > 1 and heal:
        _ = heal_skeleton(x, method="LEAFS", inplace=True)

    # If no branches, just return the neuron
    if "branch" not in x.nodes.type.values:
        return x

    if reroot_soma and not isinstance(x.soma, type(None)):
        x.reroot(x.soma, inplace=True)

    # Find main branch point
    cut = graph.find_main_branchpoint(
        x, method=method, threshold=threshold, reroot_soma=False
    )

    # Find the path to root (and account for multiple roots)
    for r in x.root:
        try:
            path = nx.shortest_path(x.graph, target=r, source=cut)
            break
        except nx.NetworkXNoPath:
            continue
        except BaseException:
            raise

    if not inverse:
        keep = path
    else:
        keep = x.nodes.node_id.values[~np.isin(x.nodes.node_id, path)]

    _ = subset.subset_neuron(x, keep, inplace=True)

    return x


@utils.map_neuronlist(desc="Pruning", allow_parallel=True)
@utils.meshneuron_skeleton(method="subset")
def prune_by_strahler(
    x: NeuronObject,
    to_prune: Union[int, List[int], range, slice],
    inplace: bool = False,
    reroot_soma: bool = True,
    force_strahler_update: bool = False,
    relocate_connectors: bool = False,
) -> NeuronObject:
    """Prune neuron based on [Strahler order](https://en.wikipedia.org/wiki/Strahler_number).

    Parameters
    ----------
    x :             TreeNeuron | MeshNeuron | NeuronList
                    Neuron(s) to prune.
    to_prune :      int | list | range | slice
                    Strahler indices (SI) to prune. For example:
                      1. `to_prune=1` removes all leaf branches
                      2. `to_prune=[1, 2]` removes SI 1 and 2
                      3. `to_prune=range(1, 4)` removes SI 1, 2 and 3
                      4. `to_prune=slice(0, -1)` removes everything but the
                         highest SI
                      5. `to_prune=slice(-1, None)` removes only the highest
                         SI
    reroot_soma :   bool, optional
                    If True, neuron will be rerooted to its soma.
    inplace :       bool, optional
                    If False, pruning is performed on copy of original neuron
                    which is then returned.
    force_strahler_update : bool, optional
                            If True, will force update of Strahler order even
                            if already exists in node table.
    relocate_connectors : bool, optional
                          If True, connectors on removed nodes will be
                          reconnected to the closest still existing node.
                          Works only in child->parent direction.

    Returns
    -------
    TreeNeuron/List
                    Pruned neuron(s).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n_pr = navis.prune_by_strahler(n, to_prune=1, inplace=False)
    >>> n.n_nodes > n_pr.n_nodes
    True

    """
    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f"Expected TreeNeuron(s), got {type(x)}")

    # Make a copy if necessary before making any changes
    neuron = x
    if not inplace:
        neuron = neuron.copy()

    if reroot_soma and not isinstance(neuron.soma, type(None)):
        neuron.reroot(neuron.soma, inplace=True)

    if "strahler_index" not in neuron.nodes or force_strahler_update:
        mmetrics.strahler_index(neuron)

    # Prepare indices
    if isinstance(to_prune, int) and to_prune < 0:
        to_prune = range(1, int(neuron.nodes.strahler_index.max() + (to_prune + 1)))

    if isinstance(to_prune, int):
        if to_prune < 1:
            raise ValueError(
                "SI to prune must be positive. Please see docs for additional options."
            )
        to_prune = [to_prune]
    elif isinstance(to_prune, range):
        to_prune = list(to_prune)
    elif isinstance(to_prune, slice):
        SI_range = range(1, int(neuron.nodes.strahler_index.max() + 1))
        to_prune = list(SI_range)[to_prune]

    # Prepare parent dict if needed later
    if relocate_connectors:
        parent_dict = {tn.node_id: tn.parent_id for tn in neuron.nodes.itertuples()}

    # Avoid setting the nodes as this potentiall triggers a regeneration
    # of the graph which in turn will raise an error because some nodes might
    # still have parents that don't exist anymore
    neuron._nodes = neuron._nodes[
        ~neuron._nodes.strahler_index.isin(to_prune)
    ].reset_index(drop=True, inplace=False)

    if neuron.has_connectors:
        if not relocate_connectors:
            neuron._connectors = neuron._connectors[
                neuron._connectors.node_id.isin(neuron._nodes.node_id.values)
            ].reset_index(drop=True, inplace=False)
        else:
            remaining_tns = set(neuron._nodes.node_id.values)
            for cn in neuron._connectors[
                ~neuron.connectors.node_id.isin(neuron._nodes.node_id.values)
            ].itertuples():
                this_tn = parent_dict[cn.node_id]
                while True:
                    if this_tn in remaining_tns:
                        break
                    this_tn = parent_dict[this_tn]
                neuron._connectors.loc[cn.Index, "node_id"] = this_tn

    # Reset indices of node and connector tables (important for igraph!)
    neuron._nodes.reset_index(inplace=True, drop=True)

    if neuron.has_connectors:
        neuron._connectors.reset_index(inplace=True, drop=True)

    # Theoretically we can end up with disconnected pieces, i.e. with more
    # than 1 root node -> we have to fix the nodes that lost their parents
    neuron._nodes.loc[
        ~neuron._nodes.parent_id.isin(neuron._nodes.node_id.values), "parent_id"
    ] = -1

    # Remove temporary attributes
    neuron._clear_temp_attr()

    return neuron


@utils.map_neuronlist(desc="Pruning", allow_parallel=True)
@utils.meshneuron_skeleton(method="subset")
def prune_twigs(
    x: NeuronObject,
    size: Union[float, str],
    exact: bool = False,
    mask: Optional[Union[Sequence[int], Callable]] = None,
    inplace: bool = False,
    recursive: Union[int, bool, float] = False,
) -> NeuronObject:
    """Prune terminal twigs under a given size.

    By default this function will simply drop all terminal twigs shorter than
    `size`. This is very fast but rather stupid: for example, if a twig is
    just 1 nanometer longer than `size` it will not be touched at all. If you
    require precision, set `exact=True` which will prune *exactly* `size`
    off the terminals at a few times the cost.

    Parameters
    ----------
    x :             TreeNeuron | MeshNeuron | NeuronList
    size :          int | float | str
                    Twigs shorter than this will be pruned. If the neuron has
                    its `.units` set, you can also pass a string including the
                    units, e.g. '5 microns'.
    exact:          bool
                    See notes above.
    mask :          iterable | callable, optional
                    Either a boolean mask, a list of node IDs or a callable taking
                    a neuron as input and returning one of the former. If provided,
                    only nodes that are in the mask will be considered for pruning.
    inplace :       bool, optional
                    If False, pruning is performed on copy of original neuron
                    which is then returned.
    recursive :     int | bool, optional
                    If `int` will undergo that many rounds of recursive
                    pruning. If True will prune iteratively until no more
                    terminal twigs under the given size are left. Only
                    relevant if `exact=False`.

    Returns
    -------
    TreeNeuron/List
                    Pruned neuron(s).

    Examples
    --------
    Simple pruning

    >>> import navis
    >>> n = navis.example_neurons(2)
    >>> # Prune twigs smaller than 5 microns
    >>> # (example neuron are in 8x8x8nm units)
    >>> n_pr = navis.prune_twigs(n,
    ...                          size=5000 / 8,
    ...                          recursive=float('inf'),
    ...                          inplace=False)
    >>> all(n.n_nodes > n_pr.n_nodes)
    True

    Exact pruning

    >>> n = navis.example_neurons(1)
    >>> # Prune twigs by exactly 5 microns
    >>> # (example neuron are in 8x8x8nm units)
    >>> n_pr = navis.prune_twigs(n,
    ...                          size=5000 / 8,
    ...                          exact=True,
    ...                          inplace=False)
    >>> n.n_nodes > n_pr.n_nodes
    True

    Prune using units

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> # Example neurons are in 8x8x8nm units...
    >>> n.units
    <Quantity(8, 'nanometer')>
    >>> # ... therefore we can use units for `size`
    >>> n_pr = navis.prune_twigs(n,
    ...                          size='5 microns',
    ...                          inplace=False)
    >>> n.n_nodes > n_pr.n_nodes
    True

    """
    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f"Expected TreeNeuron(s), got {type(x)}")

    # Convert to neuron units - numbers will be passed through
    size = x.map_units(size, on_error="raise")

    if not exact:
        return _prune_twigs_simple(
            x, size=size, inplace=inplace, recursive=recursive, mask=mask
        )
    else:
        return _prune_twigs_precise(x, size=size, mask=mask, inplace=inplace)


def _prune_twigs_simple(
    neuron: "core.TreeNeuron",
    size: float,
    inplace: bool = False,
    mask: Optional[Union[Sequence[int], Callable]] = None,
    recursive: Union[int, bool, float] = False,
) -> Optional[NeuronObject]:
    """Prune twigs using simple method."""
    if not isinstance(neuron, core.TreeNeuron):
        raise TypeError(f"Expected Neuron/List, got {type(neuron)}")

    # If people set recursive=True, assume that they mean float("inf")
    if isinstance(recursive, bool) and recursive:
        recursive = float("inf")

    # Make a copy if necessary before making any changes
    if not inplace:
        neuron = neuron.copy()

    if callable(mask):
        mask = mask(neuron)

    if mask is not None:
        mask = np.asarray(mask)

        if mask.dtype == bool:
            if len(mask) != neuron.n_nodes:
                raise ValueError("Mask length must match number of nodes")
            mask_nodes = neuron.nodes.node_id.values[mask]
        elif mask.dtype in (int, np.int32, np.int64):
            mask_nodes = mask
        else:
            raise TypeError(
                f"Mask must be boolean or list of node IDs, got {mask.dtype}"
            )
    else:
        mask_nodes = None

    if utils.fastcore:
        nodes_to_keep = utils.fastcore.prune_twigs(
            neuron.nodes.node_id.values,
            neuron.nodes.parent_id.values,
            threshold=size,
            weights=utils.fastcore.dag.parent_dist(
                neuron.nodes.node_id.values,
                neuron.nodes.parent_id.values,
                neuron.nodes[["x", "y", "z"]].values,
            ),
            mask=(
                neuron.nodes.node_id.isin(mask_nodes).values
                if mask_nodes is not None
                else None
            ),
        )

        if len(nodes_to_keep) < neuron.n_nodes:
            subset.subset_neuron(neuron, nodes_to_keep, inplace=True)

            if recursive:
                _prune_twigs_simple(
                    neuron,
                    size=size,
                    inplace=True,
                    recursive=recursive - 1,
                    mask=mask_nodes,
                )
    else:
        # Find terminal nodes
        leafs = neuron.nodes[neuron.nodes.type == "end"].node_id.values

        # Find terminal segments
        segs = graph._break_segments(neuron)
        segs = np.array([s for s in segs if s[0] in leafs], dtype=object)

        # Get segment lengths
        seg_lengths = np.array([graph.segment_length(neuron, s) for s in segs])

        # Find out which to delete
        segs_to_delete = segs[seg_lengths <= size]

        # If mask is given, only consider nodes in mask
        if mask_nodes is not None:
            segs_to_delete = [s for s in segs_to_delete if s[0] in mask_nodes]

        if len(segs_to_delete):
            # Unravel the into list of node IDs -> skip the last parent
            nodes_to_delete = [n for s in segs_to_delete for n in s[:-1]]

            # Subset neuron
            nodes_to_keep = neuron.nodes[
                ~neuron.nodes.node_id.isin(nodes_to_delete)
            ].node_id.values
            subset.subset_neuron(neuron, nodes_to_keep, inplace=True)

            # Go recursive
            if recursive:
                prune_twigs(
                    neuron,
                    size=size,
                    inplace=True,
                    recursive=recursive - 1,
                    mask=mask_nodes,
                )

    return neuron


def _subtree_height(neuron: "core.TreeNeuron", weight: str = "weight") -> pd.Series:
    """Geodesic distance from each node down to the furthest leaf below it.

    Returns a Series of heights indexed by node ID (leafs have height 0).

    A leaf `l` that is distal to `v` always sits below `v`, so the path between
    them runs straight down and `dist(v -> l) == depth(l) - depth(v)`. That turns
    "how far is the furthest leaf under v" into "what is the deepest leaf under v",
    which one upward sweep answers.
    """
    nid = neuron.nodes.node_id.values
    pid = neuron.nodes.parent_id.values

    ix = {n: i for i, n in enumerate(nid.tolist())}
    parent_ix = np.array([ix.get(p, -1) for p in pid.tolist()], dtype=np.int64)

    d = graph.dist_to_root(neuron, weight=weight)
    depth = np.array([d[n] for n in nid.tolist()], dtype=float)

    # Leafs are the nodes that are nobody's parent
    leaf_ix = np.where(~np.isin(nid, pid))[0]

    # Walk up from each leaf, deepest first. Once we hit a node already carrying a
    # value >= ours, every ancestor above it does too and we can stop - so each
    # node is written exactly once and the whole sweep is O(N).
    deepest = np.full(len(nid), -np.inf)
    for i in leaf_ix[np.argsort(-depth[leaf_ix])]:
        dl = depth[i]
        v = i
        while v != -1 and deepest[v] < dl:
            deepest[v] = dl
            v = parent_ix[v]

    return pd.Series(deepest - depth, index=nid)


def _prune_twigs_precise(
    neuron: "core.TreeNeuron",
    size: float,
    inplace: bool = False,
    mask: Optional[Union[Sequence[int], Callable]] = None,
    recursive: Union[int, bool, float] = False,
) -> Optional[NeuronObject]:
    """Prune twigs using precise method."""
    if not isinstance(neuron, core.TreeNeuron):
        raise TypeError(f"Expected Neuron/List, got {type(neuron)}")

    if size <= 0:
        raise ValueError("`length` must be > 0")

    # Make a copy if necessary before making any changes
    if not inplace:
        neuron = neuron.copy()

    # Find all nodes that could possibly be within distance to a leaf. Note we ask
    # each node for its *nearest* leaf rather than asking each leaf for every node
    # within `size`: the latter materialises one hit per (leaf, node) pair - easily
    # millions - only to collapse them down to at most one entry per node.
    leaf_tree = scipy.spatial.cKDTree(neuron.leafs[["x", "y", "z"]].values)
    dist, _ = leaf_tree.query(
        neuron.nodes[["x", "y", "z"]].values, k=1, distance_upper_bound=size
    )
    candidates = neuron.nodes.node_id.values[np.isfinite(dist)]

    if callable(mask):
        mask = mask(neuron)

    if mask is not None:
        if mask.dtype == bool:
            if len(mask) != neuron.n_nodes:
                raise ValueError("Mask length must match number of nodes")
            mask_nodes = neuron.nodes.node_id.values[mask]
        elif mask.dtype in (int, np.int32, np.int64):
            mask_nodes = mask
        else:
            raise TypeError(
                f"Mask must be boolean or list of node IDs, got {mask.dtype}"
            )

        candidates = np.intersect1d(candidates, mask_nodes)

    if not len(candidates):
        return neuron

    # A node may be pruned only if *every* leaf distal to it lies within `size` -
    # i.e. iff the height of its sub-tree (the geodesic distance down to its
    # furthest distal leaf) is <= size. `_subtree_height` gets that in a single
    # O(N) sweep; this used to run an all-pairs Dijkstra to answer the same
    # question.
    # N.B. computed on the *unpruned* neuron - we still need these heights after
    # the subset below.
    height = _subtree_height(neuron)

    node_ids = neuron.nodes.node_id.values
    in_range = np.intersect1d(node_ids[height.values <= size], candidates)

    # For a node to be deleted its PARENT has to be within
    # `length` to ALL edges that are distal do it
    nodes_to_keep = node_ids[~np.isin(neuron.nodes.parent_id.values, in_range)]

    if len(nodes_to_keep) < neuron.n_nodes:
        # Subset neuron
        subset.subset_neuron(neuron, nodes_to_keep, inplace=True)

    # For each of the new leafs check their shortest distance to the
    # original leafs to get the remainder
    is_new_leaf = (neuron.nodes.type == "end").values

    # If there is a mask, we have to exclude old leafs which would not have
    # been in the mask
    if mask is not None:
        is_new_leaf = is_new_leaf & np.isin(neuron.nodes.node_id, mask_nodes)

    new_leafs = neuron.nodes[is_new_leaf].node_id.values

    # Distance from each new leaf down to the furthest leaf it used to carry. That
    # is exactly the sub-tree height we already computed on the unpruned neuron.
    max_len = height.loc[new_leafs].values

    # For each of the new leafs check how much we need to take of the existing
    # edge
    len_to_prune = size - max_len

    # Get vectors from leafs to their parents
    nodes = neuron.nodes.set_index("node_id")
    parents = nodes.loc[new_leafs, "parent_id"].values
    loc1 = nodes.loc[new_leafs, ["x", "y", "z"]].values
    loc2 = nodes.loc[parents, ["x", "y", "z"]].values
    vec = loc1 - loc2
    vec_len = np.linalg.norm(vec, axis=1)
    vec_norm = vec / vec_len.reshape(-1, 1)

    # If `vec_len` is greater than the remaining pruning length we just remove
    # that node, if it is lower, we will move the node the given distance
    to_remove = vec_len < len_to_prune

    # First move nodes -> we can safely move all leaf nodes as the others
    # will be deleted anyway
    if not all(to_remove):
        new_loc = loc1 - vec_norm * len_to_prune.reshape(-1, 1)
        neuron.nodes.loc[is_new_leaf, ["x", "y", "z"]] = new_loc.astype(
            neuron.nodes.x.dtype, copy=False
        )

    if any(to_remove):
        leafs_to_remove = new_leafs[to_remove]
        nodes_to_keep = neuron.nodes.loc[
            ~neuron.nodes.node_id.isin(leafs_to_remove), "node_id"
        ].values
        # Subset neuron
        subset.subset_neuron(neuron, nodes_to_keep, inplace=True)

    return neuron


@utils.map_neuronlist(desc="Splitting", allow_parallel=True)
@utils.meshneuron_skeleton(
    method="split",
    include_connectors=True,
    copy_properties=["color", "compartment"],
    disallowed_kwargs={"label_only": True},
    heal=True,
)
def split_axon_dendrite(
    x: NeuronObject,
    metric: Union[
        Literal["synapse_flow_centrality"],
        Literal["flow_centrality"],
        Literal["bending_flow"],
        Literal["segregation_index"],
    ] = "synapse_flow_centrality",
    flow_thresh: float = 0.9,
    split: Union[Literal["prepost"], Literal["distance"]] = "prepost",
    cellbodyfiber: Union[Literal["soma"], Literal["root"], bool] = False,
    reroot_soma: bool = True,
    label_only: bool = False,
) -> "core.NeuronList":
    """Split a neuron into axon and dendrite using flow metrics.

    The result is highly dependent on the method and on your neuron's morphology and works best for "typical" neurons.
    See [`navis.split_axon_dendrite_prop`][] for a slightly different approach.

    Parameters
    ----------
    x :                 TreeNeuron | MeshNeuron | NeuronList
                        Neuron(s) to split into axon, dendrite (and cell body fiber if possible).
                        Always operates on the skeleton. For MeshNeurons, we will use the
                        `.skeleton` attribute if it exists or generate the skeleton on the fly.
    metric :            'synapse_flow_centrality' | 'bending_flow' | 'segregation_index' | "flow_centrality"
                        Defines which flow metric we will try to maximize when
                        splitting the neuron(s). There are four flavors:

                         - 'synapse_flow_centrality' via [`navis.synapse_flow_centrality`][]
                           (note that this metric was previously called just "flow_centrality")
                         - 'bending_flow' via [`navis.bending_flow`][]
                         - 'segregation_index' via [`navis.arbor_segregation_index`][]
                         - 'flow_centrality' via [`navis.flow_centrality`][]

                        Will try using existing columns in the node table. If
                        not present, will invoke the respective functions with
                        default parameters. All but `flow_centrality` require
                        the neuron to have connectors.
    flow_thresh :       float [0-1]
                        The "linker" between axon and dendrites will be the part
                        of the neuron with the highest flow (see metric). We
                        define it by `max(flow) * flow_thresh`. You might have
                        to decrease this value for atypical or not well
                        segregated neurons.
    split :             'prepost' | 'distance'
                        Method for determining which compartment is axon and
                        which is the dendrites:

                            - 'prepost' uses number of in- vs. outputs. By default,
                              a ratio of >1 (more out- than inputs) is considered
                              axon and vice versa. You can provide a custom threshold
                              by setting `split='prepost:0.5'` for example. Values
                              above 1.0 will bias towards dendrites and below 1.0
                              towards axon.
                            - 'distance' assumes the compartment proximal to the
                              soma is the dendrites.

    cellbodyfiber :     "soma" | "root" | False
                        Determines whether we will try to find a cell body
                        fiber (CBF).

                            - "soma" will try finding the CBF only if the neuron
                              has a soma
                            - "root" will consider the root to be the source
                              of the CBF as fallback if there is no soma
                            - `False` will not attempt to extract the CBF

                        A CBF is something typically found in insect neurons
                        which are not bipolar unlike most vertebrate neurons but
                        rather have a passive soma some distance away from
                        axon/dendrites.
    reroot_soma :       bool,
                        If True and neuron has a soma, will make sure the neuron
                        is rooted to its soma.
    label_only :        bool,
                        If True, will not split the neuron but rather add a
                        "compartment" column to the node and connector table of
                        the input neuron.

    Returns
    -------
    NeuronList
                        Axon, dendrite, linker and CBF (the latter two aren't
                        guaranteed). Fragments will have a new property
                        `compartment` (see example).

    Examples
    --------
    >>> import navis
    >>> x = navis.example_neurons(1)
    >>> split = navis.split_axon_dendrite(x, metric='synapse_flow_centrality',
    ...                                   reroot_soma=True)
    >>> split                                                   # doctest: +SKIP
    <class 'navis.NeuronList'> of 3 neurons
                          neuron_name  id  n_nodes  n_connectors  compartment
    0                  neuron 123457   16      148             0         axon
    1                  neuron 123457   16     9682          1766       linker
    2                  neuron 123457   16     2892           113     dendrite
    >>> # For convenience, split_axon_dendrite assigns colors to the resulting
    >>> # fragments: axon = red, dendrites = blue, CBF = green
    >>> _ = split.plot3d(color=split.color)

    Alternatively just label the compartments

    >>> x = navis.split_axon_dendrite(x, label_only=True)
    >>> x.nodes[~x.nodes.compartment.isnull()].head()           # doctest: +SKIP
             node_id label        x        y        z     radius  parent_id  type compartment
    110      111     0  17024.0  33790.0  26602.0  72.462097        110  slab      linker
    111      112     0  17104.0  33670.0  26682.0  72.462097        111  slab      linker
    112      113     0  17184.0  33450.0  26782.0  70.000000        112  slab      linker
    113      114     0  17244.0  33270.0  26822.0  70.000000        113  slab      linker
    114      115     0  17324.0  33150.0  26882.0  74.852798        114  slab      linker

    See Also
    --------
    [`navis.split_axon_dendrite_prop`][]
            Same function but instead of flow metrics, uses label propagation to determine
            the split. Will also give you probabilities.
    [`navis.heal_skeleton`][]
            Axon/dendrite split works only on neurons consisting of a single
            tree. Use this function to heal fragmented neurons before trying
            the axon/dendrite split.

    """
    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Can only process TreeNeurons, got "{type(x)}"')

    if not x.has_connectors:
        if metric != "flow_centrality":
            raise ValueError("Neuron must have connectors.")
        elif split == "prepost":
            raise ValueError(
                'Set `split="distance"` when trying to split neurons '
                "without connectors."
            )

    split_val = 1
    if isinstance(split, str) and ":" in split:
        split, split_val = split.split(":")
        split_val = float(split_val)

    _METRIC = (
        "synapse_flow_centrality",
        "bending_flow",
        "segregation_index",
        "flow_centrality",
    )
    utils.eval_param(metric, "metric", allowed_values=_METRIC)
    utils.eval_param(split, "split", allowed_values=("prepost", "distance"))
    utils.eval_param(
        cellbodyfiber, "cellbodyfiber", allowed_values=("soma", "root", False)
    )

    if len(x.root) > 1:
        raise ValueError(
            f"Unable to split neuron {x.id}: multiple roots. "
            "Try `navis.heal_skeleton(x)` to merged "
            "disconnected fragments."
        )

    # Make copy, so that we don't screw things up
    original = x
    x = x.copy()

    if np.any(x.soma) and not np.all(np.isin(x.soma, x.root)) and reroot_soma:
        x.reroot(x.soma, inplace=True)

    FUNCS = {
        "bending_flow": mmetrics.bending_flow,
        "synapse_flow_centrality": mmetrics.synapse_flow_centrality,
        "flow_centrality": mmetrics.flow_centrality,
        "segregation_index": mmetrics.arbor_segregation_index,
    }

    if metric not in FUNCS:
        raise ValueError(f'Unknown `metric`: "{metric}"')

    # Add metric if not already present
    if metric not in x.nodes.columns:
        _ = FUNCS[metric](x)

    # We can lock this neuron indefinitely since we are not returning it
    x._lock = 1

    # Make sure we have a metric for every single node
    if np.any(np.isnan(x.nodes[metric].values)):
        raise ValueError(f'NaN values encountered in "{metric}"')

    # The first step is to remove the linker -> that's the bit that connects
    # the axon and dendrite
    is_linker = x.nodes[metric] >= x.nodes[metric].max() * flow_thresh
    linker = set(x.nodes.loc[is_linker, "node_id"].values)

    # We try to perform processing on the graph to avoid overhead from
    # (re-)generating neurons
    g = x.graph.to_undirected()

    # Drop linker nodes
    g.remove_nodes_from(linker)

    # Break into connected components
    cc = list(nx.connected_components(g))

    # Figure out which one is which
    axon = set()
    if split == "prepost":
        # Collect # of pre- and postsynapses on each of the connected components
        sm = pd.DataFrame()
        sm["n_nodes"] = [len(c) for c in cc]
        pre = x.presynapses
        post = x.postsynapses
        sm["n_pre"] = [pre[pre.node_id.isin(c)].shape[0] for c in cc]
        sm["n_post"] = [post[post.node_id.isin(c)].shape[0] for c in cc]
        sm["prepost_ratio"] = sm.n_pre / sm.n_post
        sm["frac_post"] = sm.n_post / sm.n_post.sum()
        sm["frac_pre"] = sm.n_pre / sm.n_pre.sum()

        # In theory, we can encounter neurons with either no pre- or no
        # postsynapses (e.g. sensory neurons).
        # For those n_pre/post.sum() would cause a division by 0 which in turn
        # causes frac_pre/post to be NaN. By filling, we make sure that the
        # split doesn't fail further down but they might end up missing either
        # an axon or a dendrite (which may actually be OK?).
        sm["frac_post"] = sm["frac_post"].fillna(0)
        sm["frac_pre"] = sm["frac_pre"].fillna(0)

        # Produce the ratio of pre- to postsynapses
        sm["frac_prepost"] = sm.frac_pre / sm.frac_post

        # Some small side branches might have either no pre- or no postsynapses.
        # Even if they have synapses: if the total count is low they might be
        # incorrectly assigned to a compartment. Here, we will make sure that
        # they are disregarded for now to avoid introducing noise. Instead we
        # will connect them onto their parent compartment later.
        sm.loc[
            sm[["frac_pre", "frac_post"]].max(axis=1) < 0.01,
            ["prepost_ratio", "frac_prepost"],
        ] = np.nan

        # Each fragment is considered separately as either giver or recipient
        # of flow:
        # - prepost < 1 = dendritic
        # - prepost > 1 = axonic
        dendrite = [cc[i] for i in sm[sm.frac_prepost < split_val].index.values]
        if len(dendrite):
            dendrite = set.union(*dendrite)
        axon = [cc[i] for i in sm[sm.frac_prepost >= split_val].index.values]
        if len(axon):
            axon = set.union(*axon)
    else:
        for c in cc:
            # If original root present assume it's the proximal dendrites
            if x.root[0] in c:
                dendrite = c
            else:
                axon = axon | c

    # Now that we have in principle figured out what's what we need to do some
    # clean-up
    # First: it is quite likely that the axon(s) and/or the dendrites fragmented
    # and we need to stitch them back together using linker but not dendrites!
    g = x.graph.subgraph(np.append(list(axon), list(linker)))
    axon = set(graph.connected_subgraph(g, axon)[0])

    # Remove nodes that were re-assigned to axon from linker
    linker = linker - axon

    g = x.graph.subgraph(np.append(list(dendrite), list(linker)))
    dendrite = set(graph.connected_subgraph(g, dendrite)[0])

    # Remove nodes that were re-assigned to axon from linker
    linker = linker - set(dendrite)

    # Next up: finding the CBF
    # The CBF is defined as the part of the neuron between the soma (or root)
    # and the first branch point with sizeable synapse flow
    cbf = set()
    if cellbodyfiber and (np.any(x.soma) or cellbodyfiber == "root"):
        # To excise the CBF, we subset the neuron to those parts with
        # no/hardly any flow and find the part that contains the soma
        no_flow = x.nodes[x.nodes[metric] <= x.nodes[metric].max() * 0.05]
        g = x.graph.subgraph(no_flow.node_id.values)

        # Find the connected component containing the soma
        for c in nx.connected_components(g.to_undirected()):
            if x.root[0] in c:
                cbf = set(c)
                dendrite = dendrite - cbf
                axon = axon - cbf
                linker = linker - cbf
                break

    # See if we lost any nodes on the way
    miss = set(original.nodes.node_id.values) - linker - axon - dendrite - cbf
    miss = np.array(list(miss))

    # From hereon we can use lists
    linker = list(linker)
    axon = list(axon)
    cbf = list(cbf)
    dendrite = list(dendrite)

    # If we have, assign these nodes to the closest node with a compartment
    if len(miss):
        # Nodes that already have a compartment
        nodes_w_comp = original.nodes.node_id.values[
            ~np.isin(original.nodes.node_id.values, miss)
        ]

        # For each orphan node, find the geodesically nearest labeled node.
        # We use a memory-efficient nearest-target search instead of building a
        # full (n_miss, n_nodes) geodesic matrix, which would run out of memory
        # on large neurons.
        closest_id, _ = graph._geodesic_nearest(
            original, targets=nodes_w_comp, query=miss, weight=None, directed=False
        )

        # Orphans in a fragment without any labeled node are unreachable (-1)
        # and simply stay unassigned.
        valid = closest_id >= 0
        linker += miss[valid & np.isin(closest_id, linker)].tolist()
        axon += miss[valid & np.isin(closest_id, axon)].tolist()
        dendrite += miss[valid & np.isin(closest_id, dendrite)].tolist()
        cbf += miss[valid & np.isin(closest_id, cbf)].tolist()

    # Add labels
    if label_only:
        nodes = original.nodes
        nodes["compartment"] = None
        is_linker = nodes.node_id.isin(linker)
        is_axon = nodes.node_id.isin(axon)
        is_dend = nodes.node_id.isin(dendrite)
        is_cbf = nodes.node_id.isin(cbf)
        nodes.loc[is_linker, "compartment"] = "linker"
        nodes.loc[is_dend, "compartment"] = "dendrite"
        nodes.loc[is_axon, "compartment"] = "axon"
        nodes.loc[is_cbf, "compartment"] = "cellbodyfiber"

        # Set connector compartments
        cmp_map = original.nodes.set_index("node_id").compartment.to_dict()
        original.connectors["compartment"] = original.connectors.node_id.map(cmp_map)

        # Turn into categorical data
        original.nodes["compartment"] = original.nodes.compartment.astype("category")
        original.connectors["compartment"] = original.connectors.compartment.astype(
            "category"
        )

        return original

    # Generate the actual splits
    nl = []
    for label, nodes in zip(
        ["cellbodyfiber", "dendrite", "linker", "axon"], [cbf, dendrite, linker, axon]
    ):
        if not len(nodes):
            continue
        n = subset.subset_neuron(original, nodes)
        n.color = _COMP_COLORS.get(label, (100, 100, 100))
        n._register_attr("compartment", label)
        nl.append(n)

    return core.NeuronList(nl)


@utils.map_neuronlist(desc="Labeling", allow_parallel=True)
def split_axon_dendrite_prop(
    x: NeuronObject,
    use_weights: Union[bool, str] = "balance",
    alpha: float = 1,
    max_iter: int = 10_000,
    tol: float = 1,
    label_only: bool = False,
    verbose: bool = False,
) -> "core.NeuronList":
    """Split a neuron into axon and dendrite using label propagation.

    In contrast to [`navis.split_axon_dendrite`][] this function will
    only label the predicted compartments. It also only distringuishes
    between axon and dendrite (i.e. no linker or CBF). On the plus
    side, you get probabilities for each node/vertex and the method is
    not dependent on flow metrics which can be noisy for some neurons.

    A note of caution: on neurons where axon/dendrites are not well
    segregated (e.g. 50/50 mix of pre/postsynapses on the dendrite) or
    where one synapse type is very dominant, label propagation might
    return garbage results. A warning sign is if the propagation takes a
    long time to converge or doesn't converge at all, or if the resulting
    compartment probabilities are close to 0.5 for large chunks of
    the neuron.

    Parameters
    ----------
    x :                 TreeNeuron | MeshNeuron | NeuronList
                        Neuron(s) to split into axon and dendrite.
    use_weights :       bool or "balance"
                        Whether to use weights for the label propagation:
                         - If `True`, we will use the number of pre- vs.
                           postsynapses as weights, i.e. if a given
                           node/vertex has only marginally more pre- than
                           postsynapses, it's weight will be lowered.
                         - If "balance" (default), we will balance the
                           weights to avoid that the split is dominated by
                           the compartment with more synapses.
    alpha :             float [0-1]
                        Clamping factor for label propagation:
                         - 0 means that the initial labels (i.e. pre or
                           post) will be fully clamped and not changed
                           during propagation.
                         - 1 means that the original labels will not be
                           clamped at all and can be fully overridden during
                           propagation (default).
                         - values between 0 and 1 will softly clamp the
                           original labels.
    max_iter/tol :      int / float | "hard"
                        Stop conditions for the label propagation:
                         - `max_iter` is the maximum number of iterations
                         - `tol` is the stop condition for convergence
                        We stop when either of those two is reached. See
                        `navis.propagate_labels` for details.
    label_only :        bool,
                        If True, will not split the neuron but rather compartments
                        and probabilities to the neuron. See Returns for details!
    verbose :           bool
                        Whether to print out convergence information during label
                        propagation.

    Returns
    -------
    Neuron/List
                        If `label_only=False` (default), will return a `NeuronList`
                        with the neuron(s) split axon and dendrite. These fragments
                        will have a `.compartment` property telling you whether
                        they are the axon or the dendrite.

                        If `label_only=True`, will return the original neuron(s)
                        but with compartments (axon/dendrite) annotated.
                        For TreeNeurons, there will be two new columns in their node
                        table: `compartment` and `compartment_prob`.
                        For MeshNeurons, there will be a new `compartment` and
                        `compartment_prob` property.
                        Important: any existing compartment annotations will be
                        quietly overwritten!

    Examples
    --------
    >>> import navis
    >>> x = navis.example_neurons(1)
    >>> x = navis.split_axon_dendrite_prop(x)

    See Also
    --------
    [`navis.split_axon_dendrite_prop`][]
            Similar function but instead of label propagation it uses flow metrics to determine
            a split. Can distringuish between axon, dendrite, linker and CBF.
    [`navis.heal_skeleton`][]
            Axon/dendrite split really only works correctly on neurons consisting of a single
            tree. Use this function to heal fragmented neurons before trying
            the axon/dendrite split.
    [`navis.propagate_labels`][]
            Functions for propagating arbitrary labels on a neuron.`split_axon_dendrite_prop` is
            a specific application of this function.

    """
    assert use_weights in (
        True,
        False,
        "balance",
    ), f'`use_weights` must be True, False or "balance", got {use_weights}'

    def _calc_pre_post(x, balance_weights):
        tree = graph.neuron2KDTree(x)
        if len(post_co := x.postsynapses[["x", "y", "z"]].values):
            post_nodes = tree.query(post_co)[1]
            # post_nodes = x.snap(post_co)[0]
        else:
            post_nodes = np.array([], dtype=int)

        if len(pre_co := x.presynapses[["x", "y", "z"]].values):
            pre_nodes = tree.query(pre_co)[1]
            # pre_nodes = x.snap(pre_co)[0]
        else:
            pre_nodes = np.array([], dtype=int)

        pre_nodes_unique, pre_count = np.unique(pre_nodes, return_counts=True)
        post_nodes_unique, post_count = np.unique(post_nodes, return_counts=True)

        n = len(x.nodes) if isinstance(x, core.TreeNeuron) else len(x.vertices)
        pre_weights = np.zeros(n, dtype=np.float32)
        post_weights = np.zeros(n, dtype=np.float32)

        pre_weights[pre_nodes_unique] = pre_count
        post_weights[post_nodes_unique] = post_count

        # Create label array
        prepost = np.full(n, np.nan, dtype=object)
        prepost[pre_weights < post_weights] = "post"
        prepost[pre_weights > post_weights] = "pre"

        # Simple weights based on pre vs. post counts
        prepost_weight = np.abs(post_weights - pre_weights)

        if not balance_weights:
            return prepost, prepost_weight

        # Normalised weight - in case we have very different numbers of pre- vs. postsynapses
        # we don't want the split to be dominated by the compartment with more synapses
        ratio = 1  # doesn't matter if we have no pre- or no postsynapses
        if len(pre_nodes) and len(post_nodes):
            ratio = post_nodes.shape[0] / pre_nodes.shape[0]

        prepost_weight_balanced = prepost_weight.copy()
        prepost_weight_balanced[prepost == "pre"] *= ratio

        return prepost, prepost_weight_balanced

    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, (core.TreeNeuron, core.MeshNeuron)):
        raise TypeError(f'Can only process Tree- and MeshNeurons, got "{type(x)}"')

    if not x.has_connectors:
        raise ValueError("Neuron must have connectors.")

    prepost, prepost_weight = _calc_pre_post(
        x, balance_weights=use_weights == "balance"
    )

    if alpha == 0:
        clamping = True
    elif alpha == 1:
        clamping = False
    else:
        clamping = f"soft:{alpha}"

    pred, probs, _ = graph.graph_utils.propagate_labels(
        x,
        labels=prepost,
        weights=prepost_weight if use_weights else None,
        clamping=clamping,  # original labels can be overridden
        max_iter=max_iter,
        tol=tol,
        return_probs="raw",
        verbose=verbose,
    )
    # Make sure probs are sorted as expected (post vs. pre)
    if _[1] == "post":
        probs = probs[:, ::-1]

    probs_norm = probs.copy()
    probs_norm[probs_norm.max(axis=1) == 0] = np.nan  #  unvisited nodes/vertices have prob 0 in both columns
    is_nan = np.isnan(probs_norm).any(axis=1)
    probs_norm[~is_nan] /= probs[~is_nan].sum(axis=1).reshape(-1, 1)

    diff = probs_norm[:, 1] - probs_norm[:, 0]
    diff[is_nan] = np.nan

    comp = np.full(len(diff), np.nan, dtype=object)
    comp[diff > 0] = "axon"
    comp[diff < 0] = "dendrite"

    if label_only:
        if isinstance(x, core.TreeNeuron):
            x.nodes["compartment"] = comp
            x.nodes["compartment_prob"] = probs_norm.max(axis=1)
        else:
            x.compartment = comp
            x.compartment_prob = probs_norm.max(axis=1)

        return x

    if pd.isnull(comp).any():
        logger.warning(
            f"Parts of neuron {x.id} did not receive a label (disconnected or not reached). "
            "These parts will not be parts of the split."
        )

    axon = subset.subset_neuron(x, comp == "axon")
    dendrite = subset.subset_neuron(x, comp == "dendrite")

    axon.color = _COMP_COLORS.get("axon", (100, 100, 100))
    axon._register_attr("compartment", "axon")

    dendrite.color = _COMP_COLORS.get("dendrite", (100, 100, 100))
    dendrite._register_attr("compartment", "dendrite")

    return core.NeuronList([dendrite, axon])


def combine_neurons(
    *x: Union[Sequence[NeuronObject], "core.NeuronList"],
) -> "core.NeuronObject":
    """Combine multiple neurons into one.

    Parameters
    ----------
    x :                 NeuronList | Neuron/List
                        Neurons to combine. Must all be of the same type. Does
                        not yet work with VoxelNeurons. The combined neuron will
                        inherit its name, id, units, etc. from the first neuron
                        in `x`.

    Returns
    -------
    Neuron
                        Combined neuron.

    See Also
    --------
    [`navis.stitch_skeletons`][]
                        Stitches multiple skeletons together to create one
                        continuous neuron.

    Examples
    --------
    Combine skeletons:

    >>> import navis
    >>> nl = navis.example_neurons(3)
    >>> comb = navis.combine_neurons(nl)

    Combine meshes:

    >>> import navis
    >>> nl = navis.example_neurons(3, kind='mesh')
    >>> comb = navis.combine_neurons(nl)

    Combine dotprops:

    >>> import navis
    >>> nl = navis.example_neurons(3)
    >>> dp = navis.make_dotprops(nl)
    >>> comb = navis.combine_neurons(dp)

    """
    # Compile list of individual neurons
    nl = utils.unpack_neurons(x)
    nl = core.NeuronList(nl)

    # Check that neurons are all of the same type
    if len(nl.types) > 1:
        raise TypeError("Unable to combine neurons of different types")

    if isinstance(nl[0], core.TreeNeuron):
        x = stitch_skeletons(*nl, method="NONE", master="FIRST")
    elif isinstance(nl[0], core.MeshNeuron):
        x = nl[0].copy()
        comb = tm.util.concatenate([n.trimesh for n in nl])
        x._vertices = comb.vertices
        x._faces = comb.faces

        if any(nl.has_connectors):
            x._connectors = pd.concat(
                [n.connectors for n in nl],  # type: ignore  # no stubs for concat
                ignore_index=True,
            )
    elif isinstance(nl[0], core.Dotprops):
        x = nl[0].copy()
        x._points = np.vstack(nl._points)

        x._vect = np.vstack(nl.vect)

        if not any([isinstance(n._alpha, type(None)) for n in nl]):
            x._alpha = np.hstack(nl.alpha)

        if any(nl.has_connectors):
            x._connectors = pd.concat(
                [n.connectors for n in nl],  # type: ignore  # no stubs for concat
                ignore_index=True,
            )
    elif isinstance(nl[0], core.VoxelNeuron):
        raise TypeError("Combining VoxelNeuron not (yet) supported")
    else:
        raise TypeError(f"Unable to combine {type(nl[0])}")

    return x


def stitch_skeletons(
    *x: Union[Sequence[NeuronObject], "core.NeuronList"],
    method: Union[
        Literal["LEAFS"], Literal["ALL"], Literal["NONE"], Sequence[int]
    ] = "ALL",
    master: Union[Literal["SOMA"], Literal["LARGEST"], Literal["FIRST"]] = "SOMA",
    max_dist: Optional[float] = None,
) -> "core.TreeNeuron":
    """Stitch multiple skeletons together.

    Uses minimum spanning tree to determine a way to connect all fragments
    while minimizing length (Euclidean distance) of the new edges. Nodes
    that have been stitched will get a "stitched" tag.

    Important
    ---------
    If duplicate node IDs are found across the fragments to stitch they will
    be remapped to new unique values!

    Parameters
    ----------
    x :                 NeuronList | list of TreeNeuron/List
                        Neurons to stitch (see examples).
    method :            'LEAFS' | 'ALL' | 'NONE' | list of node IDs
                        Set stitching method:
                            (1) 'LEAFS': Only leaf (including root) nodes will
                                be allowed to make new edges.
                            (2) 'ALL': All nodes are considered.
                            (3) 'NONE': Node and connector tables will simply
                                be combined without generating any new edges.
                                The resulting neuron will have multiple roots.
                            (4) List of node IDs that are allowed to be used.
                                Note that if these nodes are insufficient
                                the resulting neuron will not be fully
                                connected.

    master :            'SOMA' | 'LARGEST' | 'FIRST', optional
                        Sets the master neuron:
                            (1) 'SOMA': The largest fragment with a soma
                                becomes the master neuron. If no neuron with
                                soma, will pick the largest (option 2).
                            (2) 'LARGEST': The largest (by number of nodes)
                                fragment becomes the master neuron.
                            (3) 'FIRST': The first fragment provided becomes
                                the master neuron.
    max_dist :          float,  optional
                        Max distance at which to stitch nodes. This can result
                        in a neuron with multiple roots.

    Returns
    -------
    TreeNeuron
                        Stitched neuron.

    See Also
    --------
    [`navis.combine_neurons`][]
                        Combines multiple neurons of the same type into one
                        without stitching. Works on TreeNeurons, MeshNeurons
                        and Dotprops.

    Examples
    --------
    Stitching neuronlist by simply combining data tables:

    >>> import navis
    >>> nl = navis.example_neurons(2)
    >>> stitched = navis.stitch_skeletons(nl, method='NONE')

    Stitching fragmented neurons:

    >>> a = navis.example_neurons(1)
    >>> fragments = navis.cut_skeleton(a, 100)
    >>> stitched = navis.stitch_skeletons(fragments, method='LEAFS')

    """
    master = str(master).upper()
    ALLOWED_MASTER = ("SOMA", "LARGEST", "FIRST")
    utils.eval_param(master, "master", allowed_values=ALLOWED_MASTER)

    # Compile list of individual neurons
    neurons = utils.unpack_neurons(x)

    # Use copies of the original neurons!
    nl = core.NeuronList(neurons).copy()

    if len(nl) < 2:
        logger.warning(f"Need at least 2 neurons to stitch, found {len(nl)}")
        return nl[0]

    # If no soma, switch to largest
    if master == "SOMA" and not any(nl.has_soma):
        master = "LARGEST"

    # First find master
    if master == "SOMA":
        # Pick the first neuron with a soma
        m_ix = [i for i, n in enumerate(nl) if n.has_soma][0]
    elif master == "LARGEST":
        # Pick the largest neuron
        m_ix = sorted(list(range(len(nl))), key=lambda x: nl[x].n_nodes, reverse=True)[
            0
        ]
    else:
        # Pick the first neuron
        m_ix = 0
    m = nl[m_ix]

    # Check if we need to make any node IDs unique
    if nl.nodes.duplicated(subset="node_id").sum() > 0:
        # Master neuron will not be changed
        seen_tn: Set[int] = set(m.nodes.node_id)
        for i, n in enumerate(nl):
            # Skip the master neuron
            # Note we're using the index in case we have two neurons that are
            # equal (by our definition) - happens e.g. if a neuron has been
            # mirrored
            if i == m_ix:
                continue

            # Grab nodes
            this_tn = set(n.nodes.node_id)

            # Get duplicate node IDs
            non_unique = seen_tn & this_tn

            # Add this neuron's existing nodes to seen
            seen_tn = seen_tn | this_tn
            if non_unique:
                # Generate new, unique node IDs
                new_tn = np.arange(0, len(non_unique)) + max(seen_tn) + 1

                # Generate new map
                new_map = dict(zip(non_unique, new_tn))

                # Remap node IDs - if no new value, keep the old
                n.nodes["node_id"] = n.nodes.node_id.map(lambda x: new_map.get(x, x))

                if n.has_connectors:
                    n.connectors["node_id"] = n.connectors.node_id.map(
                        lambda x: new_map.get(x, x)
                    )

                if getattr(n, "tags", None) is not None:
                    n.tags = {new_map.get(k, k): v for k, v in n.tags.items()}  # type: ignore

                # Remap parent IDs
                new_map[None] = -1  # type: ignore
                n.nodes["parent_id"] = n.nodes.parent_id.map(
                    lambda x: new_map.get(x, x)
                ).astype(int)

                # Add new nodes to seen
                seen_tn = seen_tn | set(new_tn)

                # Make sure the graph is updated
                n._clear_temp_attr()

    # We will start by simply merging all neurons into one
    m._nodes = pd.concat(
        [n.nodes for n in nl],  # type: ignore  # no stubs for concat
        ignore_index=True,
    )

    if any(nl.has_connectors):
        m._connectors = pd.concat(
            [n.connectors for n in nl],  # type: ignore  # no stubs for concat
            ignore_index=True,
        )

    if not m.has_tags or not isinstance(m.tags, dict):
        m.tags = {}  # type: ignore  # TreeNeuron has no tags

    for n in nl:
        for k, v in (getattr(n, "tags", None) or {}).items():
            m.tags[k] = m.tags.get(k, []) + list(utils.make_iterable(v))

    # Reset temporary attributes of our final neuron
    m._clear_temp_attr()

    # If this is all we meant to do, return this neuron
    if not utils.is_iterable(method) and (method == "NONE" or method is None):
        return m

    return _stitch_mst(m, nodes=method, inplace=False, max_dist=max_dist)




def average_skeletons(
    x: "core.NeuronList",
    limit: Union[int, str] = 10,
    base_neuron: Optional[Union[int, "core.TreeNeuron"]] = None,
) -> "core.TreeNeuron":
    """Compute an average from a list of skeletons.

    This is a very simple implementation which may give odd results if used
    on complex neurons. Works fine on e.g. backbones or tracts.

    Parameters
    ----------
    x :             NeuronList
                    Neurons to be averaged.
    limit :         int | str
                    Max distance for nearest neighbour search. If the neurons
                    have `.units` set, you can also pass a string such as e.g.
                    "2 microns".
    base_neuron :   neuron id | TreeNeuron, optional
                    Neuron to use as template for averaging. If not provided,
                    the first neuron in the list is used as template!

    Returns
    -------
    TreeNeuron

    Examples
    --------
    >>> # Get a bunch of neurons
    >>> import navis
    >>> da2 = navis.example_neurons()
    >>> # Prune down to longest neurite
    >>> for n in da2:
    ...     if n.has_soma:
    ...         n.reroot(n.soma, inplace=True)
    >>> da2_pr = da2.prune_by_longest_neurite(inplace=False)
    >>> # Make average
    >>> da2_avg = navis.average_skeletons(da2_pr, limit=10e3)
    >>> # Plot
    >>> da2.plot3d() # doctest: +SKIP
    >>> da2_avg.plot3d() # doctest: +SKIP

    """
    if not isinstance(x, core.NeuronList):
        raise TypeError(f'Need NeuronList, got "{type(x)}"')

    if len(x) < 2:
        raise ValueError("Need at least 2 neurons to average!")

    # Map limit into unit space, if applicable
    limit = x[0].map_units(limit, on_error="raise")

    # Generate KDTrees for each neuron
    for n in x:
        n.tree = graph.neuron2KDTree(n, tree_type="c", data="nodes")  # type: ignore  # TreeNeuron has no tree

    # Set base for average: we will use this neurons nodes to query
    # the KDTrees
    if isinstance(base_neuron, core.TreeNeuron):
        bn = base_neuron.copy()
    elif isinstance(base_neuron, int):
        bn = x[base_neuron].copy()
    elif isinstance(base_neuron, type(None)):
        bn = x[0].copy()
    else:
        raise ValueError(
            f'Unable to interpret base_neuron of type "{type(base_neuron)}"'
        )

    base_nodes = bn.nodes[["x", "y", "z"]].values
    other_neurons = x[[n != bn for n in x]]

    # Make sure these stay 2-dimensional arrays -> will add a colum for each
    # "other" neuron
    base_x = base_nodes[:, 0:1]
    base_y = base_nodes[:, 1:2]
    base_z = base_nodes[:, 2:3]

    # For each "other" neuron, collect nearest neighbour coordinates
    for n in other_neurons:
        nn_dist, nn_ix = n.tree.query(base_nodes, k=1, distance_upper_bound=limit)

        # Translate indices into coordinates
        # First, make empty array
        this_coords = np.zeros((len(nn_dist), 3))
        # Set coords without a nearest neighbour within distances to "None"
        this_coords[nn_dist == float("inf")] = None
        # Fill in coords of nearest neighbours
        this_coords[nn_dist != float("inf")] = n.tree.data[
            nn_ix[nn_dist != float("inf")]
        ]
        # Add coords to base coords
        base_x = np.append(base_x, this_coords[:, 0:1], axis=1)
        base_y = np.append(base_y, this_coords[:, 1:2], axis=1)
        base_z = np.append(base_z, this_coords[:, 2:3], axis=1)

    # Calculate means
    mean_x = np.mean(base_x, axis=1)
    mean_y = np.mean(base_y, axis=1)
    mean_z = np.mean(base_z, axis=1)

    # If any of the base coords has NO nearest neighbour within limit
    # whatsoever, the average of that row will be "NaN" -> in this case we
    # will fall back to the base coordinate
    mean_x[np.isnan(mean_x)] = base_nodes[np.isnan(mean_x), 0]
    mean_y[np.isnan(mean_y)] = base_nodes[np.isnan(mean_y), 1]
    mean_z[np.isnan(mean_z)] = base_nodes[np.isnan(mean_z), 2]

    # Change coordinates accordingly
    bn.nodes["x"] = mean_x
    bn.nodes["y"] = mean_y
    bn.nodes["z"] = mean_z

    return bn


@utils.map_neuronlist(desc="Despiking", allow_parallel=True)
def despike_skeleton(
    x: NeuronObject,
    sigma: int = 5,
    max_spike_length: int = 1,
    inplace: bool = False,
    reverse: bool = False,
) -> Optional[NeuronObject]:
    r"""Remove spikes in skeleton (e.g. from jumps in image data).

    For each node A, the Euclidean distance to its next successor (parent)
    B and that node's successor C (i.e A->B->C) is computed. If
    $\frac{dist(A,B)}{dist(A,C)}>sigma$, node B is considered a spike
    and realigned between A and C.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
                        Neuron(s) to be processed.
    sigma :             float | int, optional
                        Threshold for spike detection. Smaller sigma = more
                        aggressive spike detection.
    max_spike_length :  int, optional
                        Determines how long (# of nodes) a spike can be.
    inplace :           bool, optional
                        If False, a copy of the neuron is returned.
    reverse :           bool, optional
                        If True, will **also** walk the segments from proximal
                        to distal. Use this to catch spikes on e.g. terminal
                        nodes.

    Returns
    -------
    TreeNeuron/List
                Despiked neuron(s). Only if `inplace=False`.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> despiked = navis.despike_skeleton(n)

    """
    # TODO:
    # - flattening all segments first before Spike detection should speed up
    #   quite a lot
    # -> as intermediate step: assign all new positions at once

    # The decorator makes sure that we have single neurons at this point
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f"Can only process TreeNeurons, not {type(x)}")

    if not inplace:
        x = x.copy()

    # Index nodes table by node ID
    this_nodes = x.nodes.set_index("node_id", inplace=False)

    segs_to_walk = x.segments

    if reverse:
        segs_to_walk += x.segments[::-1]

    # For each spike length do -> do this in reverse to correct the long
    # spikes first
    for l in list(range(1, max_spike_length + 1))[::-1]:
        # Go over all segments
        for seg in segs_to_walk:
            # Get nodes A, B and C of this segment
            this_A = this_nodes.loc[seg[: -l - 1]]
            this_B = this_nodes.loc[seg[l:-1]]
            this_C = this_nodes.loc[seg[l + 1 :]]

            # Get coordinates
            A = this_A[["x", "y", "z"]].values
            B = this_B[["x", "y", "z"]].values
            C = this_C[["x", "y", "z"]].values

            # Calculate euclidean distances A->B and A->C
            dist_AB = np.linalg.norm(A - B, axis=1)
            dist_AC = np.linalg.norm(A - C, axis=1)

            # Get the spikes
            spikes_ix = np.where(
                np.divide(dist_AB, dist_AC, where=dist_AC != 0) > sigma
            )[0]
            spikes = this_B.iloc[spikes_ix]

            if not spikes.empty:
                # Interpolate new position(s) between A and C
                new_positions = A[spikes_ix] + (C[spikes_ix] - A[spikes_ix]) / 2

                this_nodes.loc[spikes.index, ["x", "y", "z"]] = new_positions

    # Reassign node table
    x.nodes = this_nodes.reset_index(drop=False, inplace=False)

    # The weights in the graph have changed, we need to update that
    x._clear_temp_attr(exclude=["segments", "small_segments", "classify_nodes"])

    return x


@utils.map_neuronlist(desc="Guessing", allow_parallel=True)
def guess_radius(
    x: NeuronObject,
    method: str = "linear",
    limit: Optional[int] = None,
    smooth: bool = True,
    inplace: bool = False,
) -> Optional[NeuronObject]:
    """Guess radii for skeleton nodes.

    Uses distance between connectors and nodes to guess radii. Interpolate for
    nodes without connectors. Fills in `radius` column in node table.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
                    Neuron(s) to be processed.
    method :        str, optional
                    Method to be used to interpolate unknown radii. See
                    `pandas.DataFrame.interpolate` for details.
    limit :         int, optional
                    Maximum number of consecutive missing radii to fill.
                    Must be greater than 0.
    smooth :        bool | int, optional
                    If True, will smooth radii after interpolation using a
                    rolling window. If `int`, will use to define size of
                    window.
    inplace :       bool, optional
                    If False, will use and return copy of original neuron(s).

    Returns
    -------
    TreeNeuron/List

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(2)
    >>> nl_radius = navis.guess_radius(nl)

    """
    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f"Can only process TreeNeurons, not {type(x)}")

    if not hasattr(x, "connectors") or x.connectors.empty:
        raise ValueError("Neuron must have connectors!")

    if not inplace:
        x = x.copy()

    # Set default rolling window size
    if isinstance(smooth, bool) and smooth:
        smooth = 5

    # We will be using the index as distance to interpolate. For this we have
    # to change method 'linear' to 'index'
    method = "index" if method == "linear" else method

    # Collect connectors and calc distances
    cn = x.connectors.copy()

    # Prepare nodes (add parent_dist for later, set index)
    x.nodes["parent_dist"] = mmetrics.parent_dist(x, root_dist=0)
    nodes = x.nodes.set_index("node_id", inplace=False)

    # For each connector (pre and post), get the X/Y distance to its node
    cn_locs = cn[["x", "y"]].values
    tn_locs = nodes.loc[cn.node_id.values, ["x", "y"]].values
    dist = np.sqrt(np.sum((tn_locs - cn_locs) ** 2, axis=1).astype(int))
    cn["dist"] = dist

    # Get max distance per node (in case of multiple connectors per
    # node)
    cn_grouped = cn.groupby("node_id").dist.max()

    # Set undefined radii to None so that they are ignored for interpolation
    nodes.loc[nodes.radius <= 0, "radius"] = None

    # Assign radii to nodes
    nodes.loc[cn_grouped.index, "radius"] = cn_grouped.values.astype(
        nodes.radius.dtype, copy=False
    )

    # Go over each segment and interpolate radii
    for s in config.tqdm(
        x.segments, desc="Interp.", disable=config.pbar_hide, leave=config.pbar_leave
    ):
        # Get this segments radii and parent dist
        this_radii = nodes.loc[s, ["radius", "parent_dist"]]
        this_radii["parent_dist_cum"] = this_radii.parent_dist.cumsum()

        # Set cumulative distance as index and drop parent_dist
        this_radii = this_radii.set_index("parent_dist_cum", drop=True).drop(
            "parent_dist", axis=1
        )

        # Interpolate missing radii
        interp = this_radii.interpolate(
            method=method, limit_direction="both", limit=limit
        )

        if smooth:
            interp = interp.rolling(smooth, min_periods=1).max()

        nodes.loc[s, "radius"] = interp.values

    # Set non-interpolated radii back to -1
    nodes.loc[nodes.radius.isnull(), "radius"] = -1

    # Reassign nodes
    x.nodes = nodes.reset_index(drop=False, inplace=False)

    return x


@utils.map_neuronlist(desc="Smoothing", allow_parallel=True)
def smooth_skeleton(
    x: NeuronObject,
    window: int = 5,
    to_smooth: list = ["x", "y", "z"],
    inplace: bool = False,
) -> NeuronObject:
    """Smooth skeleton(s) using rolling windows.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
                    Neuron(s) to be processed.
    window :        int, optional
                    Size (N observations) of the rolling window in number of
                    nodes.
    to_smooth :     list
                    Columns of the node table to smooth. Should work with any
                    numeric column (e.g. 'radius').
    inplace :       bool, optional
                    If False, will use and return copy of original neuron(s).

    Returns
    -------
    TreeNeuron/List
                    Smoothed neuron(s).

    Examples
    --------
    Smooth x/y/z locations (default):

    >>> import navis
    >>> nl = navis.example_neurons(2)
    >>> smoothed = navis.smooth_skeleton(nl, window=5)

    Smooth only radii:

    >>> rad_smoothed = navis.smooth_skeleton(nl, to_smooth='radius')

    See Also
    --------
    [`navis.smooth_mesh`][]
                    For smoothing MeshNeurons and other mesh-likes.
    [`navis.smooth_voxels`][]
                    For smoothing VoxelNeurons.

    """
    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f"Can only process TreeNeurons, not {type(x)}")

    if not inplace:
        x = x.copy()

    # Prepare nodes (add parent_dist for later, set index)
    # mmetrics.parent_dist(x, root_dist=0)
    nodes = x.nodes.set_index("node_id", inplace=False).copy()

    to_smooth = utils.make_iterable(to_smooth)

    miss = to_smooth[~np.isin(to_smooth, nodes.columns)]
    if len(miss):
        raise ValueError(f"Column(s) not found in node table: {miss}")

    # Go over each segment and smooth
    for s in config.tqdm(
        x.segments[::-1],
        desc="Smoothing",
        disable=config.pbar_hide,
        leave=config.pbar_leave,
    ):
        # Get this segment's parent distances and get cumsum
        this_co = nodes.loc[s, to_smooth]

        interp = this_co.rolling(window, min_periods=1).mean()

        for i, c in enumerate(to_smooth):
            nodes.loc[s, c] = interp.iloc[:, i].values.astype(
                nodes[c].dtype, copy=False
            )

    # Reassign nodes
    x.nodes = nodes.reset_index(drop=False, inplace=False)

    x._clear_temp_attr()

    return x


def break_fragments(
    x: Union["core.TreeNeuron", "core.MeshNeuron"],
    labels_only: bool = False,
    min_size: Optional[int] = None,
) -> "core.NeuronList":
    """Break neuron into its connected components.

    Neurons can consists of several disconnected fragments. This function
    turns these fragments into separate neurons.

    Parameters
    ----------
    x :             TreeNeuron | MeshNeuron
                    Fragmented neuron.
    labels_only :   bool
                    If True, will only label each node/vertex by which
                    fragment it belongs to. For TreeNeurons, this adds a
                    `"fragment"` column and for MeshNeurons, it adds a
                    `.fragments` property.
    min_size :      int, optional
                    Fragments smaller than this (# of nodes/vertices) will be
                    dropped. Ignored if `labels_only=True`.

    Returns
    -------
    NeuronList

    See Also
    --------
    [`navis.heal_skeleton`][]
                Use to heal fragmentation instead of breaking it up.


    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> # Artifically disconnect parts of the neuron
    >>> n.nodes.loc[100, 'parent_id'] = -1
    >>> # Break into fragments
    >>> frags = navis.break_fragments(n)
    >>> len(frags)
    2

    """
    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]

    if not isinstance(x, (core.TreeNeuron, core.MeshNeuron)):
        raise TypeError(f'Expected Tree- or MeshNeuron, got "{type(x)}"')

    # Get connected components
    comp = graph._connected_components(x)
    # Sort so that the first component is the largest
    comp = sorted(comp, key=len, reverse=True)

    if labels_only:
        cc_id = {n: i for i, cc in enumerate(comp) for n in cc}
        if isinstance(x, core.TreeNeuron):
            x.nodes["fragment"] = x.nodes.node_id.map(cc_id).astype(str)
        elif isinstance(x, core.MeshNeuron):
            x.fragments = np.array([cc_id[i] for i in range(x.n_vertices)]).astype(str)
        return x

    if min_size:
        comp = [cc for cc in comp if len(cc) >= min_size]

    return core.NeuronList(
        [
            subset.subset_neuron(x, list(ss), inplace=False)
            for ss in config.tqdm(
                comp, desc="Breaking", disable=config.pbar_hide, leave=config.pbar_leave
            )
        ]
    )


@utils.map_neuronlist(desc="Healing", allow_parallel=True)
def heal_skeleton(
    x: "core.NeuronList",
    method: Union[Literal["LEAFS"], Literal["ALL"]] = "ALL",
    max_dist: Optional[float] = None,
    min_size: Optional[float] = None,
    use_radius: Union[bool, float] = False,
    drop_disc: float = False,
    mask: Optional[Sequence] = None,
    inplace: bool = False,
) -> Optional[NeuronObject]:
    """Heal fragmented skeleton(s).

    Tries to heal a fragmented skeleton (i.e. a neuron with multiple roots)
    using a minimum spanning tree.

    Parameters
    ----------
    x :         TreeNeuron/List
                Fragmented skeleton(s).
    method :    'LEAFS' | 'ALL', optional
                Method used to heal fragments:
                 - 'LEAFS': Only leaf (including root) nodes will be used to
                   heal gaps. This can be much faster depending on the size of
                   the neuron
                 - 'ALL': All nodes can be used to reconnect fragments.
    max_dist :  float | str, optional
                This effectively sets the max length for newly added edges. Use
                it to prevent far away fragments to be forcefully connected.
                If the neurons have `.units` set, you can also pass a string
                such as e.g. "2 microns".
    min_size :  int, optional
                Minimum size in nodes for fragments to be reattached. Fragments
                smaller than `min_size` will be ignored during stitching and
                hence remain disconnected.
    use_radius :    bool | float, optional
                If True, will use the radius of nodes when calculating distances
                between them. This effectively prioritizes connecting nodes with
                similar radii. If float, will use that value as weight for radius:
                higher values = more importance for radius.
    drop_disc : bool
                If True and the neuron remains fragmented after healing (i.e.
                `max_dist` or `min_size` prevented a full connect), we will
                keep only the largest (by number of nodes) connected component
                and discard all other fragments.
    mask :      list-like, optional
                Either a boolean mask or a list of node IDs. If provided will
                only heal breaks between these nodes.
    inplace :   bool, optional
                If False, will perform healing on and return a copy.

    Returns
    -------
    None
                If `inplace=True`.
    CatmaidNeuron/List
                If `inplace=False`.


    See Also
    --------
    [`navis.stitch_skeletons`][]
                Use to stitch multiple skeletons together.
    [`navis.break_fragments`][]
                Use to produce individual neurons from disconnected fragments.


    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> # Disconnect parts of the neuron
    >>> n.nodes.loc[100, 'parent_id'] = -1
    >>> len(n.root)
    2
    >>> # Heal neuron
    >>> healed = navis.heal_skeleton(n)
    >>> len(healed.root)
    1

    """
    method = str(method).upper()

    if method not in ("LEAFS", "ALL"):
        raise ValueError(f'Unknown method "{method}"')

    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron(s), got "{type(x)}"')

    if not isinstance(max_dist, type(None)):
        max_dist = x.map_units(max_dist, on_error="raise")

    if not inplace:
        x = x.copy()

    _ = _stitch_mst(
        x,
        nodes=method,
        max_dist=max_dist,
        min_size=min_size,
        use_radius=use_radius,
        mask=mask,
        inplace=True,
    )

    # See if we need to drop remaining disconnected fragments
    if drop_disc:
        # Compute this property only once
        trees = x.subtrees
        if len(trees) > 1:
            # Tree is sorted such that the largest component is the first
            _ = subset.subset_neuron(x, subset=trees[0], inplace=True)

    return x


def _stitch_mst(
    x: "core.TreeNeuron",
    nodes: Union[Literal["LEAFS"], Literal["ALL"], list] = "ALL",
    max_dist: Optional[float] = np.inf,
    min_size: Optional[float] = None,
    use_radius: Union[bool, float] = False,
    mask: Optional[Sequence] = None,
    progress: bool = False,
    inplace: bool = False,
) -> Optional["core.TreeNeuron"]:
    """Stitch disconnected neuron using a minimum spanning tree.

    Parameters
    ----------
    x :             TreeNeuron
                    Neuron to stitch.
    nodes :         "ALL" | "LEAFS" | list of IDs
                    Nodes that can be used to stitch the neuron. Can be "ALL"
                    nodes, just "LEAFS".
    max_dist :      int | float | str
                    If given, will only connect fragments if they are within
                    `max_distance`. Use this to prevent the creation of
                    unrealistic edges.
    min_size :      int, optional
                    Minimum size in nodes for fragments to be reattached.
                    Fragments smaller than `min_size` will be ignored during
                    stitching and hence remain disconnected.
    use_radius :    bool | float
                    Whether to use the radius of nodes when calculating distances.
                    That way, nodes with similar radii will be preferred for stitching.
                    If float, will be used as a scaling factor for the radius:
                    higher values increase the influence of radius on distance calculation.
                    To make this more robust, we use the average radius of the segment each
                    node belongs to.
    mask :          list-like, optional
                    Either a boolean mask or a list of node IDs. If provided
                    will only heal breaks between these nodes.
    progress :      bool
                    If True, show a progress bar tracking the fragments as they
                    are merged.
    inplace :       bool
                    If True, will stitch the original neuron in place.

    Return
    ------
    TreeNeuron
                    Only if `inplace=True`.

    """
    assert isinstance(x, core.TreeNeuron)
    assert nodes in ("ALL", "LEAFS")
    if max_dist is True or not max_dist:
        max_dist = np.inf

    if not isinstance(mask, type(None)):
        mask = np.asarray(mask)
        if mask.dtype == bool:
            if len(mask) != len(x.nodes):
                raise ValueError(
                    "Length of boolean mask must match number of nodes in the neuron"
                )
            mask = x.nodes.node_id.values[mask]

    if use_radius and "radius" not in x.nodes.columns:
        raise ValueError(
            "Neuron must have a `radius` column for the `use_radius` option"
        )

    # Fast path: fastcore's stitcher prunes the nearest-neighbour search by
    # fragment, which our KDTree can't do (see `_stitch_edges`). That makes it
    # orders of magnitude faster whenever the fragments are separated by an
    # actual gap - which is the norm for real, fragmented skeletons.
    if utils.fastcore is not None and hasattr(utils.fastcore, "heal_skeleton"):
        return _stitch_fastcore(
            x,
            nodes=nodes,
            max_dist=max_dist,
            min_size=min_size,
            use_radius=use_radius,
            mask=mask,
            inplace=inplace,
        )

    # Get the fragment each node belongs to (as a positionally aligned array)
    cc, n_comp = _component_labels(x)
    if n_comp == 1:
        # There's only one component -- no healing necessary
        return x

    to_use = x.nodes

    # Collect the restrictions on which nodes may be used to bridge fragments in
    # a single mask - subsetting the node table once is much cheaper than doing
    # it for each restriction in turn.
    keep = np.ones(len(to_use), dtype=bool)

    # Drop fragments smaller than threshold
    if not isinstance(min_size, type(None)):
        sizes = np.bincount(cc, minlength=n_comp)
        keep &= sizes[cc] >= min_size

    # Filter to leaf nodes if applicable
    if nodes == "LEAFS":
        keep &= to_use["type"].isin(["end", "root"]).values

    # If mask, drop everything that is not in the mask
    if not isinstance(mask, type(None)):
        keep &= to_use.node_id.isin(mask).values

    if not keep.all():
        to_use = to_use.iloc[keep]
        cc = cc[keep]

    # Prepare the data
    cols_to_use = ["x", "y", "z"]
    if use_radius:
        to_use = to_use.copy()
        to_use["radius_seg"] = _segment_radii(x, to_use, use_radius)
        cols_to_use.append("radius_seg")

    # Find the bridges that connect the fragments with minimal added cable
    to_add = _stitch_edges(to_use, cc, cols_to_use, max_dist, progress=progress)

    # Rewire the neuron: add the bridges to the existing edges and re-root
    return _rewire_from_edges(x, to_add, inplace=inplace)


def _stitch_fastcore(
    x: "core.TreeNeuron",
    nodes: Union[Literal["LEAFS"], Literal["ALL"]] = "ALL",
    max_dist: Optional[float] = np.inf,
    min_size: Optional[int] = None,
    use_radius: Union[bool, float] = False,
    mask: Optional[Sequence] = None,
    inplace: bool = False,
) -> "core.TreeNeuron":
    """Stitch a fragmented neuron using fastcore.

    This is the fast path for [`_stitch_mst`][]: `fastcore.heal_skeleton` finds
    the same minimum spanning tree over the fragments but prunes its search by
    fragment, which a KDTree cannot do (see `_stitch_edges` for why that matters).

    Note that where several bridges are of exactly the same length the two
    implementations may pick different ones. Both trees are then valid minimum
    spanning trees of identical total length - the MST simply isn't unique - but
    the resulting `parent_id` column may differ.

    Parameters
    ----------
    x :             TreeNeuron
    nodes :         "ALL" | "LEAFS"
    max_dist :      float
                    Use `np.inf` for no limit.
    min_size :      int, optional
    use_radius :    bool | float
    mask :          array of node IDs, optional
    inplace :       bool

    Returns
    -------
    TreeNeuron

    """
    fastcore = utils.fastcore

    node_ids = x.nodes.node_id.values
    parent_ids = x.nodes.parent_id.values

    # Bail if there is only one fragment. Note we deliberately return `x`
    # untouched (rather than a rewired copy) to match `_stitch_mst`.
    if len(np.unique(fastcore.connected_components(node_ids, parent_ids))) == 1:
        return x

    new_parents = fastcore.heal_skeleton(
        node_ids=node_ids,
        parent_ids=parent_ids,
        coords=x.nodes[["x", "y", "z"]].values,
        method=nodes,
        # fastcore takes `None` rather than infinity for "no limit"
        max_dist=None if not np.isfinite(max_dist) else float(max_dist),
        min_size=min_size,
        mask=None if mask is None else np.isin(node_ids, mask),
        radius=x.nodes.radius.values if use_radius else None,
        use_radius=use_radius,
    )

    if not inplace:
        x = x.copy()

    x.nodes["parent_id"] = new_parents

    x._clear_temp_attr()

    return x


def _component_labels(x: "core.TreeNeuron") -> Tuple[np.ndarray, int]:
    """Label each node with the connected component (fragment) it belongs to.

    Unlike [`navis.graph._connected_components`][] this returns a plain integer
    array positionally aligned with the node table rather than a list of sets.
    That matters here: turning the sets back into a per-node label used to go via
    a `{node_id: component}` dict and a `.map()` over it, which is a Python-level
    pass over every node and dominated the runtime of `_stitch_mst` on large
    skeletons.

    Parameters
    ----------
    x :         TreeNeuron

    Returns
    -------
    labels :    (N, ) int array
                Component label for each node, in node table order. Labels are
                contiguous (`0 .. n_components - 1`) but otherwise arbitrary.
    n :         int
                Number of connected components.

    """
    node_ids = x.nodes.node_id.values
    parent_ids = x.nodes.parent_id.values
    N = len(node_ids)

    if not N:
        return np.zeros(0, dtype=np.int64), 0

    # Existing child -> parent edges (roots have no parent and are dropped)
    id2ix = pd.Series(np.arange(N), index=node_ids)
    par_ix = id2ix.reindex(parent_ids).values
    has_parent = ~np.isnan(par_ix)

    adj = coo_matrix(
        (
            np.ones(int(has_parent.sum()), dtype=np.int8),
            (np.arange(N)[has_parent], par_ix[has_parent].astype(np.int64)),
        ),
        shape=(N, N),
    ).tocsr()

    n, labels = _sparse_cc(adj, directed=False)

    return labels, int(n)


def _segment_radii(
    x: "core.TreeNeuron",
    to_use: pd.DataFrame,
    use_radius: Union[bool, float],
) -> pd.Series:
    """Radius to use as an extra "spatial" dimension when stitching.

    For each node this is the mean radius of the segment it belongs to, scaled by
    `use_radius`. Using the segment mean rather than the node's own radius makes
    the value more robust to noisy per-node radii.

    Isolated nodes (a root without children) belong to no segment and therefore
    have no segment radius; they fall back to their own radius, and to 0 if that
    is missing too.

    Parameters
    ----------
    x :             TreeNeuron
                    The (fragmented) neuron. Segments are taken from here, i.e.
                    from *all* nodes - not just the candidates.
    to_use :        DataFrame
                    Node table of the candidate nodes to return radii for.
    use_radius :    bool | float
                    Scaling factor for the radius: higher values give the radius
                    more influence over which nodes get bridged.

    Returns
    -------
    pandas.Series
                    Scaled radius for each node in `to_use` (same index).

    """
    if "radius" not in x.nodes.columns:
        raise ValueError(
            "Neuron must have a `radius` column for the `use_radius` option"
        )

    rad = dict(zip(x.nodes.node_id.values, x.nodes.radius.values))

    # Mean radius of each node's segment
    seg_rad = {}
    for seg in x.small_segments:
        rad_mean = np.mean([rad[node] for node in seg]) * float(use_radius)
        seg_rad.update({node: rad_mean for node in seg})

    # Note that the fallback must be mapped through `node_id`: passing the `rad`
    # dict straight to `.fillna()` would key it off the DataFrame's *index*
    # instead and hand isolated nodes some other node's radius. It also has to be
    # scaled, just like the segment radii above.
    own_rad = to_use.node_id.map(rad) * float(use_radius)

    return to_use.node_id.map(seg_rad).fillna(own_rad).fillna(0)


def _rewire_from_edges(
    x: "core.TreeNeuron",
    new_edges: Sequence,
    inplace: bool = False,
) -> "core.TreeNeuron":
    """Regenerate a neuron's `parent_id` column after adding new edges.

    This is the counterpart to `_stitch_edges`: it takes the neuron's existing
    child->parent edges plus a set of new (undirected) edges and re-derives a
    valid rooted forest via a single breadth-first search.

    Note that we deliberately do *not* compute a minimum spanning tree here (as
    e.g. [`navis.rewire_skeleton`][] does): the bridges produced by
    `_stitch_edges` each merge two *distinct* fragments, so the resulting graph
    is guaranteed to be acyclic already. A plain BFS is both sufficient and much
    faster.

    Parameters
    ----------
    x :         TreeNeuron
                Neuron to rewire.
    new_edges : (M, 2) array-like
                Pairs of node IDs to connect. Can be empty.
    inplace :   bool
                If True, will rewire `x` in place.

    Returns
    -------
    TreeNeuron

    """
    if not inplace:
        x = x.copy()

    node_ids = x.nodes.node_id.values
    parent_ids = x.nodes.parent_id.values
    N = len(node_ids)

    if not N:
        return x

    # We map node IDs to positional indices which requires unique IDs. This is a
    # given for TreeNeurons but let's fail loudly rather than silently produce
    # garbage if it ever isn't.
    id2ix = pd.Series(np.arange(N), index=node_ids)
    if not id2ix.index.is_unique:
        raise ValueError("Unable to rewire neuron with duplicate node IDs.")

    # Existing child -> parent edges (roots map to NaN and are dropped)
    par_ix = id2ix.reindex(parent_ids).values
    has_parent = ~np.isnan(par_ix)
    sources = np.arange(N)[has_parent]
    targets = par_ix[has_parent].astype(np.int64)

    # Add the new edges
    new_edges = np.asarray(new_edges).reshape(-1, 2)
    if len(new_edges):
        sources = np.concatenate([sources, id2ix.loc[new_edges[:, 0]].values])
        targets = np.concatenate([targets, id2ix.loc[new_edges[:, 1]].values])

    # Pick a seed (i.e. the new root) for each connected component:
    # 1. The neuron's current root if the component contains it
    # 2. Else any other pre-existing root in the component
    # 3. Else the lowest-index node in the component
    # This preserves the neuron's orientation wherever possible.
    adj = coo_matrix(
        (np.ones(len(sources), dtype=np.int8), (sources, targets)), shape=(N, N)
    ).tocsr()
    adj = adj + adj.T
    _, labels = _sparse_cc(adj, directed=False)

    seeds = np.full(labels.max() + 1, -1, dtype=np.int64)
    # Fill in lowest-index node per component (np.unique returns the first
    # occurrence which, since labels are in node order, is the lowest index)
    comps, first = np.unique(labels, return_index=True)
    seeds[comps] = first
    # Pre-existing roots take precedence (iterate in reverse so that the *first*
    # root in table order wins), and the current root trumps all.
    roots = np.where(parent_ids < 0)[0]
    for r in roots[::-1]:
        seeds[labels[r]] = r

    # Rather than running one BFS per component we attach a virtual super-root
    # (index N) to every seed and run a single BFS from it. Each component is
    # only reachable via its own seed, so this yields exactly the same forest.
    sources = np.concatenate([sources, np.full(len(seeds), N, dtype=np.int64)])
    targets = np.concatenate([targets, seeds.astype(np.int64)])
    adj = coo_matrix(
        (np.ones(len(sources), dtype=np.int8), (sources, targets)), shape=(N + 1, N + 1)
    ).tocsr()
    adj = adj + adj.T

    _, preds = breadth_first_order(
        adj, N, directed=False, return_predecessors=True
    )
    par_ix = preds[:N]

    # Seeds have the super-root as predecessor; unreachable nodes get scipy's
    # null marker (-9999). Both become roots (-1).
    par_ix = np.where(par_ix >= N, -1, par_ix)
    par_ix = np.where(par_ix < 0, -1, par_ix)

    new_parents = np.full(N, -1, dtype=node_ids.dtype)
    is_child = par_ix >= 0
    new_parents[is_child] = node_ids[par_ix[is_child]]

    x.nodes["parent_id"] = new_parents

    x._clear_temp_attr()

    return x


def _resolve_by_exclusion(
    tree,
    coords: np.ndarray,
    nodes: np.ndarray,
    super_of: np.ndarray,
    bound: np.ndarray,
    best_dist: np.ndarray,
    best_target: np.ndarray,
    max_dist: float,
) -> None:
    """Find each node's nearest neighbour outside its own component.

    This is the fallback for nodes that the `k`-nearest-neighbour search in
    `_stitch_edges` could not resolve - i.e. nodes sitting so deep inside a large,
    well-separated fragment that none of their `_KNN_MAX_K` nearest neighbours is
    in another component.

    Rather than escalating `k` (which would cost O(n * fragment size) in both time
    and memory) we ask the tree for the nearest neighbour while excluding the
    node's own component. `pykdtree` can do this directly via its `mask` argument;
    with scipy we have to build a temporary tree over the remaining nodes.

    Updates `best_dist`, `best_target` and `bound` in place.
    """
    if not len(nodes):
        return

    has_mask = "pykdtree" in type(tree).__module__
    n = len(coords)

    # One query per component that still has unresolved nodes
    for comp in np.unique(super_of[nodes]):
        in_comp = nodes[super_of[nodes] == comp]
        is_self = super_of == comp

        # We only care about edges that could still beat this component's best
        upper = min(bound[comp], max_dist)

        if has_mask:
            dist, ix = tree.query(
                coords[in_comp], k=1, distance_upper_bound=upper, mask=is_self
            )
            # pykdtree signals "nothing found" with an out-of-range index
            found = ix < n
        else:
            others = np.where(~is_self)[0]
            if not len(others):
                continue
            dist, ix = KDTree(coords[others]).query(
                coords[in_comp], k=1, distance_upper_bound=upper
            )
            found = ix < len(others)
            ix = others[np.where(found, ix, 0)]

        dist = np.atleast_1d(dist)
        ix = np.atleast_1d(ix)
        found = np.atleast_1d(found) & (dist <= max_dist)

        if not found.any():
            continue

        best_dist[in_comp[found]] = dist[found]
        best_target[in_comp[found]] = ix[found]
        bound[comp] = min(bound[comp], dist[found].min())


def _stitch_edges(
    to_use: pd.DataFrame,
    cc: np.ndarray,
    cols_to_use: List[str],
    max_dist: float,
    progress: bool = False,
) -> List[List[int]]:
    """Find the minimal-length set of edges connecting a neuron's fragments.

    This is Borůvka's algorithm over the fragments: in each round every
    (super-)component contributes its single cheapest outgoing edge, those edges
    are added and the components they connect are merged. The number of
    components at least halves each round, so only O(log F) rounds are needed.
    The result is a true minimum spanning tree over the fragments.

    The expensive part is finding, for every node, its nearest neighbour in a
    *different* component. We do this by querying a single KDTree (built once)
    for each node's `k` nearest neighbours and taking the first one that sits in
    another component. Nodes for which none of the `k` neighbours qualify - i.e.
    those sitting deep inside a large fragment - escalate to a larger `k`.

    The escalation is kept in check by a per-component bound: the length of the
    best cross-component edge any of the component's nodes has found so far this
    round. Once a node's k-th neighbour is farther away than that bound, the node
    cannot possibly supply the component's minimum and is dropped. This keeps the
    result exact while confining the deep-inside nodes - which would otherwise
    have to search across their entire fragment - to a small ball.

    Parameters
    ----------
    to_use :        DataFrame
                    Node table of the candidate nodes.
    cc :            (N, ) array
                    Component label for each node in `to_use` (positionally
                    aligned). Labels need not be contiguous.
    cols_to_use :   list of str
                    Columns of `to_use` to use as coordinates. Note that this can
                    be more than three (see the `use_radius` option).
    max_dist :      float
                    Maximum length for any single new edge.
    progress :      bool
                    If True, show a progress bar tracking the fragments as they
                    are merged.

    Returns
    -------
    list
                    List of `[node_a, node_b]` node ID pairs to connect. At most
                    `(#fragments - 1)` long; fewer if `max_dist` prevented some
                    fragments from being connected.

    """
    coords = np.ascontiguousarray(to_use[cols_to_use].values, dtype=np.float64)
    node_ids = to_use.node_id.values
    n = len(node_ids)

    # Compress the (potentially non-contiguous) component labels to 0..F-1 so we
    # can use them as array indices
    _, frag_ix = np.unique(np.asarray(cc), return_inverse=True)
    n_frags = int(frag_ix.max()) + 1 if n else 0

    if n_frags < 2:
        return []

    tree = KDTree(coords)

    # Union-find over the *fragments* (not the nodes!) with path halving
    uf = np.arange(n_frags, dtype=np.int64)

    def _find(a):
        root = a
        while uf[root] != root:
            root = uf[root]
        while uf[a] != root:
            uf[a], a = root, uf[a]
        return root

    edges: List[List[int]] = []
    n_sets = n_frags

    pbar = config.tqdm(
        total=n_frags - 1,
        desc="Stitching",
        disable=config.pbar_hide or not progress,
        leave=config.pbar_leave,
    )

    while n_sets > 1:
        # Map each node onto the super-component it currently belongs to
        root_of_frag = np.fromiter(
            (_find(f) for f in range(n_frags)), dtype=np.int64, count=n_frags
        )
        super_of = root_of_frag[frag_ix]

        best_dist = np.full(n, np.inf)
        best_target = np.full(n, -1, dtype=np.int64)
        # Per-super-component pruning bound (see docstring)
        bound = np.full(n_frags, max_dist, dtype=np.float64)
        active = np.ones(n, dtype=bool)

        k = 8
        while active.any() and k <= _KNN_MAX_K:
            kk = min(k, n)
            # Process the query in chunks: the (chunk, k) neighbour matrix is by
            # far the largest allocation here, so this is what keeps the memory
            # footprint flat instead of scaling with `k`.
            chunk = max(1, _KNN_MAX_ELEMENTS // kk)
            for query_ix in np.array_split(
                np.where(active)[0], max(1, int(np.ceil(active.sum() / chunk)))
            ):
                if not len(query_ix):
                    continue

                dist, ix = tree.query(coords[query_ix], k=kk)
                if dist.ndim == 1:  # k=1 returns flat arrays
                    dist, ix = dist[:, None], ix[:, None]

                my_comp = super_of[query_ix]
                # For each node: which of its k neighbours are in another component?
                is_cross = super_of[ix] != my_comp[:, None]
                has_cross = is_cross.any(axis=1)
                # Neighbours are sorted by distance, so the first cross-component
                # hit is that node's nearest cross-component neighbour
                first = np.argmax(is_cross, axis=1)
                rows = np.arange(len(query_ix))
                hit_dist, hit_ix = dist[rows, first], ix[rows, first]

                found = has_cross & (hit_dist <= max_dist)
                best_dist[query_ix[found]] = hit_dist[found]
                best_target[query_ix[found]] = hit_ix[found]

                # Tighten each component's bound with what we just found
                np.minimum.at(bound, my_comp[found], hit_dist[found])

                # Retire nodes that either found their neighbour or can no longer
                # beat their component's bound
                exhausted = dist[:, -1] >= bound[my_comp]
                active[query_ix[has_cross | exhausted]] = False

            if kk >= n:
                break
            k *= 4

        # Whatever is still active sits so deep inside its fragment that none of
        # its `_KNN_MAX_K` nearest neighbours belongs to another component - which
        # happens when fragments are large and well separated. Escalating `k`
        # further would cost O(n * fragment size) time and memory, so we instead
        # ask the tree directly for the nearest neighbour *outside* the node's own
        # component. That is O(n) in memory and keeps the result exact.
        _resolve_by_exclusion(
            tree, coords, np.where(active)[0], super_of, bound,
            best_dist, best_target, max_dist,
        )

        candidates = np.where(best_target >= 0)[0]
        if not len(candidates):
            # No fragment can reach another within `max_dist`
            break

        # Reduce to the single cheapest outgoing edge per super-component and add
        # them cheapest-first (which keeps the result deterministic)
        order = candidates[np.lexsort((candidates, best_dist[candidates]))]
        seen: Set[int] = set()
        cheapest = []
        for node in order:
            comp = super_of[node]
            if comp not in seen:
                seen.add(comp)
                cheapest.append(node)

        merged = 0
        for node in cheapest:
            target = int(best_target[node])
            ra, rb = _find(frag_ix[node]), _find(frag_ix[target])
            if ra == rb:
                # Already merged earlier this round
                continue
            uf[ra] = rb
            n_sets -= 1
            merged += 1
            edges.append([node_ids[node], node_ids[target]])

        pbar.update(merged)

        if not merged:
            break

    pbar.close()

    return edges


@utils.map_neuronlist(desc="Pruning", must_zip=["source"], allow_parallel=True)
@utils.meshneuron_skeleton(method="subset")
def prune_at_depth(
    x: NeuronObject,
    depth: Union[float, int],
    *,
    source: Optional[int] = None,
    inplace: bool = False,
) -> Optional[NeuronObject]:
    """Prune all neurites past a given distance from a source.

    Parameters
    ----------
    x :             TreeNeuron | MeshNeuron | NeuronList
    depth :         int | float | str
                    Distance from source at which to start pruning. If neuron
                    has its `.units` set, you can also pass this as a string such
                    as "50 microns".
    source :        int, optional
                    Source node for depth calculation. If `None`, will use
                    root (first root if multiple). If `x` is a
                    list of neurons then must provide a source for each neuron.
    inplace :       bool, optional
                    If False, pruning is performed on copy of original neuron
                    which is then returned.

    Returns
    -------
    TreeNeuron/List
                    Pruned neuron(s).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(2)
    >>> # Reroot to soma
    >>> n.reroot(n.soma, inplace=True)
    >>> # Prune all twigs farther from the root than 100 microns
    >>> # (example neuron are in 8x8x8nm units)
    >>> n_pr = navis.prune_at_depth(n,
    ...                             depth=100e3 / 8,
    ...                             inplace=False)
    >>> all(n.n_nodes > n_pr.n_nodes)
    True

    """
    # The decorator makes sure that at this point we only have single neurons
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f"Expected TreeNeuron, got {type(x)}")

    depth = x.map_units(depth, on_error="raise")
    if depth < 0:
        raise ValueError(f'`depth` must be > 0, got "{depth}"')

    if isinstance(source, type(None)):
        source = x.root[0]
    elif source not in x.nodes.node_id.values:
        raise ValueError(f'Source "{source}" not among nodes')

    # Get distance from source
    dist = graph.geodesic_matrix(x, from_=source, directed=False, limit=depth)
    keep = dist.columns[dist.values[0] < np.inf]

    if not inplace:
        x = x.copy()

    _ = subset.subset_neuron(x, subset=keep, inplace=True)

    return x


@utils.map_neuronlist(desc="Removing fluff", allow_parallel=True)
def drop_fluff(
    x: Union["core.TreeNeuron", "core.MeshNeuron", "core.NeuronList"],
    keep_size: Optional[float] = None,
    n_largest: Optional[int] = None,
    epsilon: Optional[float] = None,
    inplace: bool = False,
):
    """Remove small disconnected pieces of "fluff".

    By default, this function will remove all but the largest connected
    component from the neuron. You can change that behavior using the
    `keep_size` and `n_largest` parameters. Connectors (if present) will
    be remapped to the closest surviving vertex/node.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | Dotprops | NeuronList
                The neuron(s) to remove fluff from.
    keep_size : float, optional
                Use this to set a size (in number of nodes/vertices) for small
                bits to keep. If `keep_size` < 1 it will be intepreted as
                fraction of total nodes/vertices/points.
    n_largest : int, optional
                If set, will keep the `n_largest` connected components. Note:
                if provided, `keep_size` will be applied first!
    epsilon :   float, optional
                For Dotprops: distance at which to consider two points to be
                connected. If `None`, will use the default value of 5 times
                the average node distance (`x.sampling_resolution`).
    inplace :   bool, optional
                If False, pruning is performed on copy of original neuron
                which is then returned.

    Returns
    -------
    Neuron/List
                Neuron(s) without fluff.

    Examples
    --------
    >>> import navis
    >>> m = navis.example_neurons(1, kind='mesh')
    >>> m.n_vertices
    6309
    >>> # Remove all but the largest connected component
    >>> top = navis.drop_fluff(m)
    >>> top.n_vertices
    5951
    >>> # Keep the ten largest connected components
    >>> two = navis.drop_fluff(m, n_largest=10)
    >>> two.n_vertices
    6069
    >>> # Keep all fragments with at least 100 vertices
    >>> clean = navis.drop_fluff(m, keep_size=100)
    >>> clean.n_vertices
    5951
    >>> # Keep the two largest fragments with at least 50 vertices each
    >>> # (for this neuron the result is just the largest fragment)
    >>> clean2 = navis.drop_fluff(m, keep_size=50, n_largest=2)
    >>> clean2.n_vertices
    6037

    """
    utils.eval_param(
        x, name="x", allowed_types=(core.TreeNeuron, core.MeshNeuron, core.Dotprops)
    )

    # This function uses navis_fastcore if available
    cc = sorted(graph.graph_utils._connected_components(x), key=lambda x: len(x), reverse=True)

    # Translate keep_size to number of nodes
    if keep_size and keep_size < 1:
        keep_size = sum(len(c) for c in cc) * keep_size

    if keep_size:
        cc = [c for c in cc if len(c) >= keep_size]
        if not n_largest:
            keep = [i for c in cc for i in c]
        else:
            keep = [i for c in cc[:n_largest] for i in c]
    elif n_largest:
        keep = [i for c in cc[:n_largest] for i in c]
    else:
        keep = cc[0]

    # Subset neuron
    x = subset.subset_neuron(x, subset=keep, inplace=inplace, keep_disc_cn=True)

    # See if we need to/can re-attach any connectors
    if x.has_connectors:
        id_col = [
            c for c in ("node_id", "vertex_id", "point_id") if c in x.connectors.columns
        ]
        if id_col:
            id_col = id_col[0]
            disc = ~x.connectors[id_col].isin(x.graph.nodes).values
            if any(disc):
                xyz = x.connectors.loc[disc, ["x", "y", "z"]].values
                x.connectors.loc[disc, id_col] = x.snap(xyz)[0]

    return x
