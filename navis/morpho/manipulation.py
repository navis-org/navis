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


""" This module contains functions to analyse and manipulate neuron morphology.
"""
import pandas as pd
import numpy as np
import networkx as nx

from collections import namedtuple
from itertools import combinations
from scipy.spatial import cKDTree
from typing import Union, Optional, Sequence, overload, List, Set
from typing_extensions import Literal

from .. import graph, utils, config, core
from . import mmetrics

# Set up logging
logger = config.logger

__all__ = sorted(['prune_by_strahler', 'stitch_neurons', 'split_axon_dendrite',
                  'average_neurons', 'despike_neuron', 'guess_radius',
                  'smooth_neuron', 'heal_fragmented_neuron',
                  'break_fragments', 'prune_twigs'])

NeuronObject = Union['core.NeuronList', 'core.TreeNeuron']


@overload
def prune_by_strahler(x: NeuronObject,
                      to_prune: Union[int, List[int], range, slice],
                      inplace: Literal[False],
                      reroot_soma: bool = True,
                      force_strahler_update: bool = False,
                      relocate_connectors: bool = False) -> NeuronObject: ...


@overload
def prune_by_strahler(x: NeuronObject,
                      to_prune: Union[int, List[int], range, slice],
                      inplace: Literal[True],
                      reroot_soma: bool = True,
                      force_strahler_update: bool = False,
                      relocate_connectors: bool = False) -> None: ...


@overload
def prune_by_strahler(x: NeuronObject,
                      to_prune: Union[int, List[int], range, slice],
                      inplace: bool = False,
                      reroot_soma: bool = True,
                      force_strahler_update: bool = False,
                      relocate_connectors: bool = False) -> Optional[NeuronObject]: ...


def prune_by_strahler(x: NeuronObject,
                      to_prune: Union[int, List[int], range, slice],
                      inplace: bool = False,
                      reroot_soma: bool = True,
                      force_strahler_update: bool = False,
                      relocate_connectors: bool = False) -> Optional[NeuronObject]:
    """Prune neuron based on `Strahler order <https://en.wikipedia.org/wiki/Strahler_number>`_.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
    to_prune :      int | list | range | slice
                    Strahler indices (SI) to prune. For example:

                    1. ``to_prune=1`` removes all leaf branches
                    2. ``to_prune=[1, 2]`` removes SI 1 and 2
                    3. ``to_prune=range(1, 4)`` removes SI 1, 2 and 3
                    4. ``to_prune=slice(0, -1)`` removes everything but the
                       highest SI
                    5. ``to_prune=slice(-1, None)`` removes only the highest
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
    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()

        [prune_by_strahler(n,
                           to_prune=to_prune,
                           inplace=True,
                           reroot_soma=reroot_soma,
                           force_strahler_update=force_strahler_update,
                           relocate_connectors=relocate_connectors
                           ) for n in config.tqdm(x,
                                                  desc='Pruning',
                                                  disable=config.pbar_hide,
                                                  leave=config.pbar_leave)]
        if not inplace:
            return x
        else:
            return None
    elif isinstance(x, core.TreeNeuron):
        neuron = x
    else:
        raise TypeError(f'Expected Neuron/List, got {type(x)}')

    # Make a copy if necessary before making any changes
    if not inplace:
        neuron = neuron.copy()

    if reroot_soma and np.any(neuron.soma):
        neuron.reroot(neuron.soma, inplace=True)

    if 'strahler_index' not in neuron.nodes or force_strahler_update:
        mmetrics.strahler_index(neuron)

    # Prepare indices
    if isinstance(to_prune, int) and to_prune < 0:
        to_prune = range(1, int(neuron.nodes.strahler_index.max() + (to_prune + 1)))

    if isinstance(to_prune, int):
        if to_prune < 1:
            raise ValueError('SI to prune must be positive. Please see help'
                             'for additional options.')
        to_prune = [to_prune]
    elif isinstance(to_prune, range):
        to_prune = list(to_prune)
    elif isinstance(to_prune, slice):
        SI_range = range(1, int(neuron.nodes.strahler_index.max() + 1))
        to_prune = list(SI_range)[to_prune]

    # Prepare parent dict if needed later
    if relocate_connectors:
        parent_dict = {
            tn.node_id: tn.parent_id for tn in neuron.nodes.itertuples()}

    # Avoid setting the nodes as this potentiall triggers a regeneration
    # of the graph which in turn will raise an error because some nodes might
    # still have parents that don't exist anymore
    neuron._nodes = neuron._nodes[~neuron._nodes.strahler_index.isin(to_prune)].reset_index(drop=True, inplace=False)

    if neuron.has_connectors:
        if not relocate_connectors:
            neuron._connectors = neuron._connectors[neuron._connectors.node_id.isin(neuron._nodes.node_id.values)].reset_index(drop=True, inplace=False)
        else:
            remaining_tns = set(neuron._nodes.node_id.values)
            for cn in neuron._connectors[~neuron.connectors.node_id.isin(neuron._nodes.node_id.values)].itertuples():
                this_tn = parent_dict[cn.node_id]
                while True:
                    if this_tn in remaining_tns:
                        break
                    this_tn = parent_dict[this_tn]
                neuron._connectors.loc[cn.Index, 'node_id'] = this_tn

    # Reset indices of node and connector tables (important for igraph!)
    neuron._nodes.reset_index(inplace=True, drop=True)

    if neuron.has_connectors:
        neuron._connectors.reset_index(inplace=True, drop=True)

    # Theoretically we can end up with disconnected pieces, i.e. with more
    # than 1 root node -> we have to fix the nodes that lost their parents
    neuron._nodes.loc[~neuron._nodes.parent_id.isin(neuron._nodes.node_id.values), 'parent_id'] = -1

    # Remove temporary attributes
    neuron._clear_temp_attr()

    if not inplace:
        return neuron
    else:
        return None


@overload
def prune_twigs(x: NeuronObject,
                size: float,
                exact: bool,
                inplace: Literal[True],
                recursive: Union[int, bool, float] = False
                ) -> None: ...


@overload
def prune_twigs(x: NeuronObject,
                size: float,
                exact: bool,
                inplace: Literal[False],
                recursive: Union[int, bool, float] = False
                ) -> NeuronObject: ...


@overload
def prune_twigs(x: NeuronObject,
                size: float,
                exact: bool,
                inplace: bool = False,
                recursive: Union[int, bool, float] = False
                ) -> Optional[NeuronObject]: ...


def prune_twigs(x: NeuronObject,
                size: float,
                exact: bool = False,
                inplace: bool = False,
                recursive: Union[int, bool, float] = False
                ) -> Optional[NeuronObject]:
    """Prune terminal twigs under a given size.

    By default this function will simply drop all terminal twigs shorter than
    ``size``. This is very fast but rather stupid: for example, if a twig is
    just 1 nanometer longer than ``size`` it will not be touched at all. If you
    require precision, set ``exact=True`` which will prune *exactly* ``size``
    off the terminals but is about an order of magnitude slower.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
    size :          int | float
                    Twigs shorter than this will be pruned.
    exact:          bool
                    See notes above.
    inplace :       bool, optional
                    If False, pruning is performed on copy of original neuron
                    which is then returned.
    recursive :     int | bool, optional
                    If `int` will undergo that many rounds of recursive
                    pruning. If True will prune iteratively until no more
                    terminal twigs under the given size are left. Only
                    relevant if ``exact=False``.

    Returns
    -------
    TreeNeuron/List
                    Pruned neuron(s).

    Examples
    --------
    Simple pruning

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> # Prune twigs smaller than 5 microns
    >>> # (example neuron are in 8x8x8nm units)
    >>> n_pr = navis.prune_twigs(n,
    ...                          size=5000 / 8,
    ...                          recursive=float('inf'),
    ...                          inplace=False)
    >>> n.n_nodes > n_pr.n_nodes
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

    """
    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()

        [prune_twigs(n,
                     size=size,
                     exact=exact,
                     inplace=True,
                     recursive=recursive) for n in config.tqdm(x,
                                                               desc='Pruning',
                                                               disable=config.pbar_hide,
                                                               leave=config.pbar_leave)]

        if not inplace:
            return x
        else:
            return None
    elif isinstance(x, core.TreeNeuron):
        neuron = x
    else:
        raise TypeError(f'Expected Neuron/List, got {type(x)}')

    if not exact:
        return _prune_twigs_simple(neuron,
                                   size=size,
                                   inplace=inplace,
                                   recursive=recursive)
    else:
        return _prune_twigs_precise(neuron,
                                    size=size,
                                    inplace=inplace)


def _prune_twigs_simple(neuron: 'core.TreeNeuron',
                        size: float,
                        inplace: bool = False,
                        recursive: Union[int, bool, float] = False
                        ) -> Optional[NeuronObject]:
    """Prune twigs using simple method."""
    if not isinstance(neuron, core.TreeNeuron):
        raise TypeError(f'Expected Neuron/List, got {type(neuron)}')

    # If people set recursive=True, assume that they mean float("inf")
    if isinstance(recursive, bool) and recursive:
        recursive = float('inf')

    # Make a copy if necessary before making any changes
    if not inplace:
        neuron = neuron.copy()

    # Find terminal nodes
    leafs = neuron.nodes[neuron.nodes.type == 'end'].node_id.values

    # Find terminal segments
    segs = graph._break_segments(neuron)
    segs = np.array([s for s in segs if s[0] in leafs], dtype=object)

    # Get segment lengths
    seg_lengths = np.array([graph.segment_length(neuron, s) for s in segs])

    # Find out which to delete
    segs_to_delete = segs[seg_lengths <= size]

    if segs_to_delete.any():
        # Unravel the into list of node IDs -> skip the last parent
        nodes_to_delete = [n for s in segs_to_delete for n in s[:-1]]

        # Subset neuron
        nodes_to_keep = neuron.nodes[~neuron.nodes.node_id.isin(nodes_to_delete)].node_id.values
        graph.subset_neuron(neuron,
                            nodes_to_keep,
                            inplace=True)

        # Go recursive
        if recursive:
            recursive -= 1
            prune_twigs(neuron, size=size, inplace=True, recursive=recursive)

    if not inplace:
        return neuron
    else:
        return None


def _prune_twigs_precise(neuron: 'core.TreeNeuron',
                         size: float,
                         inplace: bool = False,
                         recursive: Union[int, bool, float] = False
                         ) -> Optional[NeuronObject]:
    """Prune twigs using precise method."""
    if not isinstance(neuron, core.TreeNeuron):
        raise TypeError(f'Expected Neuron/List, got {type(neuron)}')

    if size <= 0:
        raise ValueError('`length` must be > 0')

    # Make a copy if necessary before making any changes
    if not inplace:
        neuron = neuron.copy()

    # Find terminal nodes
    leafs = neuron.leafs.node_id.values

    # Find all nodes that could possibly be within distance to a leaf
    tree = graph.neuron2KDTree(neuron)
    res = tree.query_ball_point(neuron.leafs[['x', 'y', 'z']].values,
                                r=size)
    candidates = neuron.nodes.node_id.values[np.unique(np.concatenate(res))]

    # For each node in neuron find out which leafs are directly distal to it
    # `distal` is a matrix with all nodes in columns and leafs in rows
    distal = graph.distal_to(neuron, a=leafs, b=candidates)
    # Turn matrix into dictionary {'node': [leafs, distal, to, it]}
    melted = distal.reset_index(drop=False).melt(id_vars='index')
    melted = melted[melted.value]
    melted.groupby('variable')['index'].apply(list)
    # `distal` is now a dictionary for {'node_id': [leaf1, leaf2, ..], ..}
    distal = melted.groupby('variable')['index'].apply(list).to_dict()

    # For each node find the distance to any leaf - note we are using `length`
    # as cutoff here
    # `path_len` is a dict mapping {nodeA: {nodeB: length, ...}, ...}
    # if nodeB is not in dictionary, it's not within reach
    path_len = dict(nx.all_pairs_dijkstra_path_length(neuron.graph.reverse(),
                                                      cutoff=size, weight='weight'))

    # For each leaf in `distal` check if it's within length
    not_in_length = {k: set(v) - set(path_len[k]) for k, v in distal.items()}

    # For a node to be deleted its PARENT has to be within
    # `length` to ALL edges that are distal do it
    in_range = {k for k, v in not_in_length.items() if not any(v)}
    nodes_to_keep = neuron.nodes.loc[~neuron.nodes.parent_id.isin(in_range),
                                     'node_id'].values

    if len(nodes_to_keep) < neuron.n_nodes:
        # Subset neuron
        graph.subset_neuron(neuron,
                            nodes_to_keep,
                            inplace=True)

    # For each of the new leafs check their shortest distance to the
    # original leafs to get the remainder
    is_new_leaf = (neuron.nodes.type == 'end').values
    new_leafs = neuron.nodes[is_new_leaf].node_id.values
    max_len = [max([path_len[l1][l2] for l2 in distal[l1]]) for l1 in new_leafs]

    # For each of the new leafs check how much we need to take of the existing
    # edge
    len_to_prune = size - np.array(max_len)

    # Get vectors from leafs to their parents
    nodes = neuron.nodes.set_index('node_id')
    parents = nodes.loc[new_leafs, 'parent_id'].values
    loc1 = neuron.leafs[['x', 'y', 'z']].values
    loc2 = nodes.loc[parents, ['x', 'y', 'z']].values
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
        neuron.nodes.loc[is_new_leaf, ['x', 'y', 'z']] = new_loc

    if any(to_remove):
        leafs_to_remove = new_leafs[to_remove]
        nodes_to_keep = neuron.nodes.loc[~neuron.nodes.node_id.isin(leafs_to_remove),
                                         'node_id'].values
        # Subset neuron
        graph.subset_neuron(neuron,
                            nodes_to_keep,
                            inplace=True)

    if not inplace:
        return neuron
    else:
        return None


def split_axon_dendrite(x: NeuronObject,
                        metric: Union[Literal['flow_centrality'],
                                      Literal['bending_flow'],
                                      Literal['segregation_index']] = 'flow_centrality',
                        flow_thresh: float = .9,
                        split: Union[Literal['prepost'],
                                     Literal['distance']] = 'prepost',
                        cellbodyfiber: Union[Literal['soma'],
                                             Literal['root'],
                                             bool] = 'soma',
                        reroot_soma: bool = True,
                        labels: Union[Literal['only'],
                                      bool] = False
                        ) -> 'core.NeuronList':
    """Split a neuron into axon, dendrite, linker and cell body fiber.

    The result is highly dependent on the method and on your neuron's
    morphology and works best for "typical" insect neurons, i.e. those where the
    cell body fiber branches into axon and dendrites.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
                        Neuron(s) to split into axon, dendrite (and cell body
                        fiber if possible). MUST HAVE CONNECTORS.
    metric :            'flow_centrality' | 'bending_flow' | 'segregation_index', optional
                        Defines which flow metric we will try to maximize when
                        splitting the neuron(s). There are five flavors:

                         - `flow_centrality` in :func:`~navis.flow_centrality`
                         - 'bending_flow' uses to :func:`~navis.bending_flow`
                         - 'segregation_index' uses :func:`~navis.arbor_segregation_index`

                        Will try using existing columns in the node table. If
                        not present, will invoke the respective functions with
                        default parameters.
    flow_thresh :       float [0-1]
                        The "linker" between axon and dendrites will be the part
                        of the neuron with the highest flow (see metric). We
                        define it by ``max(flow) * flow_thresh``. You might have
                        to decrease this value for atypical or not well
                        segregated neurons.
    split :             'prepost' | 'distance'
                        Method for determining which compartment is axon and
                        which is the dendrites:

                            - 'prepost' uses number of in- vs. outputs
                            - 'distance' assumes the compartment proximal to the
                              soma is the dendrites

    cellbodyfiber :     "soma" | "root" | False
                        Determines whether we will try to find the cell body
                        fiber (CBF):

                            - "soma" will try finding the CBF only if the neuron
                              has a soma
                            - "root" will consider the root to be the source
                              of the CBF as fallback if there is no soma
                            - `False` will not attempt to extract the CBF

    reroot_soma :       bool,
                        If True and neuron has a soma, will make sure the neuron
                        is rooted to its soma.
    labels :            bool | "only",
                        If True, will add a "compartment" column to the node
                        table of the input neuron. If "only" will only add that
                        column and not return the split.

    Returns
    -------
    NeuronList
                        Axon, dendrite, linker and CBF (the latter two aren't
                        guaranteed). Fragments will have a new property
                        ``compartment`` (see example).

    Examples
    --------
    >>> import navis
    >>> x = navis.example_neurons(1)
    >>> split = navis.split_axon_dendrite(x, metric='flow_centrality',
    ...                                   reroot_soma=True)
    >>> split                                                   # doctest: +SKIP
    <class 'navis.NeuronList'> of 3 neurons
                          neuron_name  id  n_nodes  n_connectors
    0                  neuron 123457   16      148             0
    1                  neuron 123457   16     9682          1766
    2                  neuron 123457   16     2892           113
    >>> # For convenience, split_axon_dendrite assigns colors to the resulting
    >>> # fragments: axon = red, dendrites = blue, CBF = green
    >>> _ = split.plot3d(color=split.color)

    See Also
    --------
    :func:`navis.heal_fragmented_neuron`
            Axon/dendrite split works only on neurons consisting of a single
            tree. Use this function to heal fragmented neurons before trying
            the axon/dendrite split.

    """
    COLORS = {'axon': (255, 0, 0),
              'dendrite': (0, 0, 255),
              'cellbodyfiber': (50, 50, 50),
              'linker': (150, 150, 150)}

    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]
    elif isinstance(x, core.NeuronList):
        nl = []
        for n in config.tqdm(x, desc='Splitting', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            nl.append(split_axon_dendrite(n,
                                          metric=metric,
                                          split=split,
                                          flow_thresh=flow_thresh,
                                          cellbodyfiber=cellbodyfiber,
                                          reroot_soma=reroot_soma,
                                          labels=labels))
        return core.NeuronList([n for l in nl for n in l])

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Can only process TreeNeuron, got "{type(x)}"')

    if not x.has_connectors:
        raise ValueError('Neuron must have connectors.')

    _METRIC = ('flow_centrality', 'bending_flow', 'segregation_index')
    utils.eval_param(metric, 'metric', allowed_values=_METRIC)
    utils.eval_param(split, 'split', allowed_values=('prepost', 'distance'))
    utils.eval_param(cellbodyfiber, 'cellbodyfiber',
                     allowed_values=('soma', 'root'))

    if len(x.root) > 1:
        raise ValueError(f'Unable to split neuron {x.id}: multiple roots. '
                         'Try `navis.heal_fragmented_neuron(x)` to merged '
                         'disconnected fragments.')

    # Make copy, so that we don't screw things up
    original = x
    x = x.copy()

    if np.any(x.soma) and not np.all(np.isin(x.soma, x.root)) and reroot_soma:
        x.reroot(x.soma, inplace=True)

    if metric == 'bending_flow':
        if 'bending_flow' not in x.nodes.columns:
            mmetrics.bending_flow(x)
        col = 'bending_flow'
    elif metric == 'flow_centrality':
        if 'flow_centrality' not in x.nodes.columns:
            mmetrics.flow_centrality(x)
        col = 'flow_centrality'
    elif metric == 'segregation':
        if 'SI' not in x.nodes.columns:
            mmetrics.arbor_segregation_index(x)
        col = 'segregation_index'

    # We can lock this neuron indefinitely since we are not returning it
    x._lock = 1

    # Make sure we have a metric for every single node
    if np.any(np.isnan(x.nodes[col].values)):
        raise ValueError(f'NaN values encountered in "{col}"')

    # The first step is to remove the linker -> that's the bit that connects
    # the axon and dendrite
    is_linker = x.nodes[col] >= x.nodes[col].max() * flow_thresh
    linker = set(x.nodes.loc[is_linker, 'node_id'].values)

    # We try to perform processing on the graph to avoid overhead from
    # (re-)generating neurons
    g = x.graph.copy()

    # Drop linker nodes
    g.remove_nodes_from(linker)

    # Break into connected components
    cc = list(nx.connected_components(g.to_undirected()))

    # Figure out which one is which
    axon = set()
    if split == 'prepost':
        # Collect # of pre- and postsynapses on each of the connected components
        sm = pd.DataFrame()
        sm['n_nodes'] = [len(c) for c in cc]
        pre = x.presynapses
        post = x.postsynapses
        sm['n_pre'] = [pre[pre.node_id.isin(c)].shape[0] for c in cc]
        sm['n_post'] = [post[post.node_id.isin(c)].shape[0] for c in cc]
        sm['prepost'] = (sm.n_pre / sm.n_post)
        sm['frac_post'] = sm.n_post / sm.n_post.sum()  # this is for debugging
        sm['frac_pre'] = sm.n_pre / sm.n_pre.sum()
        sm['frac_prepost'] = (sm.frac_pre / sm.frac_post)

        sm.loc[sm[['frac_pre', 'frac_post']].max(axis=1) < 0.01,
               ['prepost', 'frac_prepost']] = np.nan
        # Above code makes it so that prepost is NaN if there are either no pre-
        # OR postsynapses on a given fragment or the fragment is small.
        # These fragments are typically small sidebranches of the linker.
        # We will disregard them for now because they just introduce noise
        # and will connect them onto their parent compartment later.
        logger.debug(sm)

        # Each fragment is considered separately as either giver or recipient
        # of flow:
        # - prepost < 1 = dendritic
        # - prepost > 1 = axonic
        dendrite = set.union(*[cc[i] for i in sm[sm.frac_prepost < 1].index.values])
        axon = set.union(*[cc[i] for i in sm[sm.frac_prepost >= 1].index.values])
    else:
        for c in cc:
            # If original root present assume it's the proximal dendrites
            if x.root[0] in c:
                dendrite = c
            else:
                axon = axon | c

    # Now that we have in princple figured out what's what we need to do some
    # clean-up
    # First: it is quite likely that the axon(s) and or the dendrites fragmented
    # and we need to stitch them back together using linker but not dendrites!
    g = x.graph.copy()
    g.remove_nodes_from(dendrite)
    axon = graph.connected_subgraph(g, axon)[0]

    # Remove nodes that were re-assigned to axon from linker
    linker = linker - set(axon)

    g = x.graph.copy()
    g.remove_nodes_from(axon)
    dendrite = graph.connected_subgraph(g, dendrite)[0]

    # Remove nodes that were re-assigned to axon from linker
    linker = linker - set(dendrite)

    # Next up: finding the CBF
    # The CBF is defined as the part of the neuron between the soma (or root)
    # and the first branch point with sizeable synapse flow
    cbf = set()
    if cellbodyfiber and (np.any(x.soma) or cellbodyfiber == 'root'):
        # To excise the CBF, we subset the neuron to those parts with
        # no/hardly any flow and find the part that contains the soma
        no_flow = x.nodes[x.nodes[col] <= x.nodes[col].max() * 0.05]
        g = x.graph.subgraph(no_flow.node_id.values)

        # Find the connected component containing the soma
        for c in nx.connected_components(g.to_undirected()):
            if x.root[0] in c:
                cbf = c
                dendrite = set(dendrite) - set(cbf)
                axon = set(axon) - set(cbf)
                linker = set(linker) - set(cbf)
                break

    # Add labels
    if labels:
        original.nodes['compartment'] = None
        is_linker = original.nodes.node_id.isin(linker)
        is_axon = original.nodes.node_id.isin(axon)
        is_dend = original.nodes.node_id.isin(dendrite)
        is_cbf = original.nodes.node_id.isin(cbf)
        original.nodes.loc[is_linker, 'compartment'] = 'linker'
        original.nodes.loc[is_dend, 'compartment'] = 'dendrite'
        original.nodes.loc[is_axon, 'compartment'] = 'axon'
        original.nodes.loc[is_cbf, 'compartment'] = 'cellbodyfiber'

        # There might be small branches w/o synapses (i.e. without flow) that
        # have not yet been assigned. We will attribute them to whatever
        # they connect to
        to_fix = original.nodes.loc[original.nodes.compartment.isnull()]

        # Find branch points proximal to these that have a compartment
        has_type = ~original.nodes.compartment.isnull()
        bps = original.nodes.loc[(original.nodes['type'] == 'branch') & has_type,
                                 'node_id'].values
        to_fix_branch_child = to_fix[to_fix.parent_id.isin(bps)]
        parent_type = original.nodes.set_index('node_id').loc[to_fix_branch_child.parent_id.values,
                                                              'compartment'].values

        g = original.graph
        for source, cmp in zip(to_fix_branch_child.node_id.values, parent_type):
            nodes = nx.bfs_tree(g, source=source, reverse=True).nodes
            original.nodes.loc[original.nodes.node_id.isin(nodes),
                               'compartment'] = cmp

        # For some reason the original root sometimes does not get a compartment
        # and above code won't fix it because we are looking at upstream branch
        # points. In that case, we need to look for its childs
        root_cmp = original.nodes.loc[original.nodes.parent_id == original.root[0],
                                      'compartment'].values[0]
        original.nodes.loc[original.nodes.compartment.isnull() & (original.nodes.parent_id < 0),
                           'compartment'] = root_cmp

        # Turn into categorical data
        original.nodes['compartment'] = original.nodes.compartment.astype('category')

        if labels == 'only':
            return

    # Generate the actual splits
    nl = []
    for label, nodes in zip(['cellbodyfiber', 'dendrite', 'linker', 'axon'],
                            [cbf, dendrite, linker, axon]):
        n = graph.subset_neuron(original, nodes)
        n.color = COLORS.get(label, (100, 100, 100))
        n._register_attr('compartment', label)
        nl.append(n)

    return core.NeuronList(nl)


def stitch_neurons(*x: Union[Sequence[NeuronObject], 'core.NeuronList'],
                   method: Union[Literal['LEAFS'],
                                 Literal['ALL'],
                                 Literal['NONE'],
                                 Sequence[int]] = 'ALL',
                   master: Union[Literal['SOMA'],
                                 Literal['LARGEST'],
                                 Literal['FIRST']] = 'SOMA',
                   max_dist: Optional[float] = None,
                   ) -> 'core.TreeNeuron':
    """Stitch multiple neurons together.

    Uses minimum spanning tree to determine a way to connect all fragments
    while minimizing length (eucledian distance) of the new edges. Nodes
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

    Examples
    --------
    Stitching neuronlist by simply combining data tables:

    >>> import navis
    >>> nl = navis.example_neurons(2)
    >>> stitched = navis.stitch_neurons(nl, method='NONE')

    Stitching fragmented neurons:

    >>> a = navis.example_neurons(1)
    >>> fragments = navis.cut_neuron(a, 100)
    >>> stitched = navis.stitch_neurons(fragments, method='LEAFS')

    """
    master = str(master).upper()
    ALLOWED_MASTER = ('SOMA', 'LARGEST', 'FIRST')
    utils.eval_param(master, 'master', allowed_values=ALLOWED_MASTER)

    # Compile list of individual neurons
    neurons = utils.unpack_neurons(x)

    # Use copies of the original neurons!
    nl = core.NeuronList(neurons).copy()

    if len(nl) < 2:
        logger.warning(f'Need at least 2 neurons to stitch, found {len(nl)}')
        return nl[0]

    # First find master
    if master == 'SOMA':
        has_soma = [n for n in nl if not isinstance(n.soma, type(None))]
        if len(has_soma) > 0:
            m = has_soma[0]
        else:
            m = sorted(nl.neurons,
                       key=lambda x: x.n_nodes,
                       reverse=True)[0]
    elif master == 'LARGEST':
        m = sorted(nl.neurons,
                   key=lambda x: x.n_nodes,
                   reverse=True)[0]
    else:
        # Simply pick the first neuron
        m = nl[0]

    # Check if we need to make any node IDs unique
    if nl.nodes.duplicated(subset='node_id').sum() > 0:
        # Master neuron will not be changed
        seen_tn: Set[int] = set(m.nodes.node_id)
        for n in nl:
            # Skip the master neuron
            if n == m:
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
                n.nodes['node_id'] = n.nodes.node_id.map(lambda x: new_map.get(x, x))

                if n.has_connectors:
                    n.connectors['node_id'] = n.connectors.node_id.map(lambda x: new_map.get(x, x))

                if hasattr(n, 'tags'):
                    n.tags = {new_map.get(k, k): v for k, v in n.tags.items()}  # type: ignore

                # Remap parent IDs
                new_map[None] = -1  # type: ignore
                n.nodes['parent_id'] = n.nodes.parent_id.map(lambda x: new_map.get(x, x)).astype(int)

                # Add new nodes to seen
                seen_tn = seen_tn | set(new_tn)

                # Make sure the graph is updated
                n._clear_temp_attr()

    # We will start by simply merging all neurons into one
    m._nodes = pd.concat([n.nodes for n in nl],  # type: ignore  # no stubs for concat
                         ignore_index=True)

    if any(nl.has_connectors):
        m._connectors = pd.concat([n.connectors for n in nl],  # type: ignore  # no stubs for concat
                                  ignore_index=True)

    m.tags = {}  # type: ignore  # TreeNeuron has no tags
    for n in nl:
        for k, v in getattr(n, 'tags', {}).items():
            m.tags[k] = m.tags.get(k, []) + list(utils.make_iterable(v))

    # Reset temporary attributes of our final neuron
    m._clear_temp_attr()

    # If this is all we meant to do, return this neuron
    if not utils.is_iterable(method) and (method == 'NONE' or method is None):
        return m

    return _stitch_mst(m, nodes=method, inplace=False, max_dist=max_dist)


def _mst_igraph(nl: 'core.NeuronList',
                new_edges: pd.DataFrame) -> List[List[int]]:
    """Compute edges necessary to connect a fragmented neuron using igraph."""
    # Generate a union of all graphs
    g = nl[0].igraph.disjoint_union(nl[1:].igraph)

    # We have to manually set the node IDs again
    nids = np.concatenate([n.igraph.vs['node_id'] for n in nl])
    g.vs['node_id'] = nids

    # Set existing edges to zero weight to make sure they have priority when
    # calculating the minimum spanning tree
    g.es['weight'] = 0

    # If two nodes occupy the same position (e.g. after if fragments are the
    # result of cutting), they will have a distance of 0. Hence, we won't be
    # able to simply filter by distance
    g.es['new'] = False

    # Convert node IDs in new_edges to vertex IDs and add to graph
    name2ix = dict(zip(g.vs['node_id'], range(len(g.vs))))
    new_edges['source_ix'] = new_edges.source.map(name2ix)
    new_edges['target_ix'] = new_edges.target.map(name2ix)

    # Add new edges
    g.add_edges(new_edges[['source_ix', 'target_ix']].values.tolist())

    # Add edge weight to new edges
    g.es[-new_edges.shape[0]:]['weight'] = new_edges.weight.values

    # Keep track of new edges
    g.es[-new_edges.shape[0]:]['new'] = True

    # Compute the minimum spanning tree
    mst = g.spanning_tree(weights='weight')

    # Extract the new edges
    to_add = mst.es.select(new=True)

    # Convert to node IDs
    to_add = [(g.vs[e.source]['node_id'],
               g.vs[e.target]['node_id'],
               {'weight': e['weight']})
              for e in to_add]

    return to_add


def _mst_nx(nl: 'core.NeuronList',
            new_edges: pd.DataFrame) -> List[List[int]]:
    """Compute edges necessary to connect a fragmented neuron using networkX."""
    # Generate a union of all graphs
    g = nx.union_all([n.graph for n in nl]).to_undirected()

    # Set existing edges to zero weight to make sure they have priority when
    # calculating the minimum spanning tree
    nx.set_edge_attributes(g, 0, 'weight')

    # If two nodes occupy the same position (e.g. after if fragments are the
    # result of cutting), they will have a distance of 0. Hence, we won't be
    # able to simply filter by distance
    nx.set_edge_attributes(g, False, 'new')

    # Convert new edges in the right format
    edges_nx = [(r.source, r.target, {'weight': r.weight, 'new': True})
                for r in new_edges.itertuples()]

    # Add edges to union graph
    g.add_edges_from(edges_nx)

    # Get minimum spanning tree
    edges = nx.minimum_spanning_edges(g)

    # Edges that need adding are those that were newly added
    to_add = [e for e in edges if e[2]['new']]

    return to_add


def average_neurons(x: 'core.NeuronList',
                    limit: int = 10,
                    base_neuron: Optional[Union[int, 'core.TreeNeuron']] = None
                    ) -> 'core.TreeNeuron':
    """Compute an average from a list of neurons.

    This is a very simple implementation which may give odd results if used
    on complex neurons. Works fine on e.g. backbones or tracts.

    Parameters
    ----------
    x :             NeuronList
                    Neurons to be averaged.
    limit :         int, optional
                    Max distance for nearest neighbour search.
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
    >>> da2_avg = navis.average_neurons(da2_pr, limit=10e3)
    >>> # Plot
    >>> da2.plot3d() # doctest: +SKIP
    >>> da2_avg.plot3d() # doctest: +SKIP

    """
    if not isinstance(x, core.NeuronList):
        raise TypeError(f'Need NeuronList, got "{type(x)}"')

    if len(x) < 2:
        raise ValueError('Need at least 2 neurons to average!')

    # Generate KDTrees for each neuron
    for n in x:
        n.tree = graph.neuron2KDTree(n, tree_type='c', data='nodes')  # type: ignore  # TreeNeuron has no tree

    # Set base for average: we will use this neurons nodes to query
    # the KDTrees
    if isinstance(base_neuron, core.TreeNeuron):
        bn = base_neuron.copy()
    elif isinstance(base_neuron, int):
        bn = x[base_neuron].copy()
    elif isinstance(base_neuron, type(None)):
        bn = x[0].copy()
    else:
        raise ValueError(f'Unable to interpret base_neuron of type "{type(base_neuron)}"')

    base_nodes = bn.nodes[['x', 'y', 'z']].values
    other_neurons = x[[n != bn for n in x]]

    # Make sure these stay 2-dimensional arrays -> will add a colum for each
    # "other" neuron
    base_x = base_nodes[:, 0:1]
    base_y = base_nodes[:, 1:2]
    base_z = base_nodes[:, 2:3]

    # For each "other" neuron, collect nearest neighbour coordinates
    for n in other_neurons:
        nn_dist, nn_ix = n.tree.query(base_nodes,
                                      k=1,
                                      distance_upper_bound=limit)

        # Translate indices into coordinates
        # First, make empty array
        this_coords = np.zeros((len(nn_dist), 3))
        # Set coords without a nearest neighbour within distances to "None"
        this_coords[nn_dist == float('inf')] = None
        # Fill in coords of nearest neighbours
        this_coords[nn_dist != float(
            'inf')] = n.tree.data[nn_ix[nn_dist != float('inf')]]
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
    bn.nodes.loc[:, 'x'] = mean_x
    bn.nodes.loc[:, 'y'] = mean_y
    bn.nodes.loc[:, 'z'] = mean_z

    return bn


def despike_neuron(x: NeuronObject,
                   sigma: int = 5,
                   max_spike_length: int = 1,
                   inplace: bool = False,
                   reverse: bool = False) -> Optional[NeuronObject]:
    """Remove spikes in skeleton (e.g. from jumps in image data).

    For each node A, the euclidean distance to its next successor (parent)
    B and that node's successor C (i.e A->B->C) is computed. If
    :math:`\\frac{dist(A,B)}{dist(A,C)}>sigma`, node B is considered a spike
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
                Despiked neuron(s). Only if ``inplace=False``.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> despiked = navis.despike_neuron(n)

    """
    # TODO:
    # - flattening all segments first before Spike detection should speed up
    #   quite a lot
    # -> as intermediate step: assign all new positions at once

    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()

        for n in config.tqdm(x, desc='Despiking', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            despike_neuron(n, sigma=sigma, inplace=True)

        if not inplace:
            return x
        return None
    elif not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Can only process TreeNeuron or NeuronList, not {type(x)}')

    if not inplace:
        x = x.copy()

    # Index nodes table by node ID
    this_nodes = x.nodes.set_index('node_id', inplace=False)

    segs_to_walk = x.segments

    if reverse:
        segs_to_walk += x.segments[::-1]

    # For each spike length do -> do this in reverse to correct the long
    # spikes first
    for l in list(range(1, max_spike_length + 1))[::-1]:
        # Go over all segments
        for seg in x.segments:
            # Get nodes A, B and C of this segment
            this_A = this_nodes.loc[seg[:-l - 1]]
            this_B = this_nodes.loc[seg[l:-1]]
            this_C = this_nodes.loc[seg[l + 1:]]

            # Get coordinates
            A = this_A[['x', 'y', 'z']].values
            B = this_B[['x', 'y', 'z']].values
            C = this_C[['x', 'y', 'z']].values

            # Calculate euclidian distances A->B and A->C
            dist_AB = np.linalg.norm(A - B, axis=1)
            dist_AC = np.linalg.norm(A - C, axis=1)

            # Get the spikes
            spikes_ix = np.where((dist_AB / dist_AC) > sigma)[0]
            spikes = this_B.iloc[spikes_ix]

            if not spikes.empty:
                # Interpolate new position(s) between A and C
                new_positions = A[spikes_ix] + (C[spikes_ix] - A[spikes_ix]) / 2

                this_nodes.loc[spikes.index, ['x', 'y', 'z']] = new_positions

    # Reassign node table
    x.nodes = this_nodes.reset_index(drop=False, inplace=False)

    # The weights in the graph have changed, we need to update that
    x._clear_temp_attr(exclude=['segments', 'small_segments',
                                'classify_nodes'])

    if not inplace:
        return x
    else:
        return None


def guess_radius(x: NeuronObject,
                 method: str = 'linear',
                 limit: Optional[int] = None,
                 smooth: bool = True,
                 inplace: bool = False) -> Optional[NeuronObject]:
    """Guess radii for all nodes.

    Uses distance between connectors and nodes and interpolate for all
    nodes. Fills in ``radius`` column in node table.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
                    Neuron(s) to be processed.
    method :        str, optional
                    Method to be used to interpolate unknown radii. See
                    ``pandas.DataFrame.interpolate`` for details.
    limit :         int, optional
                    Maximum number of consecutive missing radii to fill.
                    Must be greater than 0.
    smooth :        bool | int, optional
                    If True, will smooth radii after interpolation using a
                    rolling window. If ``int``, will use to define size of
                    window.
    inplace :       bool, optional
                    If False, will use and return copy of original neuron(s).

    Returns
    -------
    TreeNeuron/List
                    If ``inplace=False``.

    """
    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()

        for n in config.tqdm(x, desc='Guessing', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            guess_radius(n, method=method, limit=limit, smooth=smooth,
                         inplace=True)

        if not inplace:
            return x
        return None

    elif not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Can only process TreeNeuron or NeuronList, not {type(x)}')

    if not hasattr(x, 'connectors') or x.connectors.empty:
        raise ValueError('Neuron must have connectors!')

    if not inplace:
        x = x.copy()

    # Set default rolling window size
    if isinstance(smooth, bool) and smooth:
        smooth = 5

    # We will be using the index as distance to interpolate. For this we have
    # to change method 'linear' to 'index'
    method = 'index' if method == 'linear' else method

    # Collect connectors and calc distances
    cn = x.connectors.copy()

    # Prepare nodes (add parent_dist for later, set index)
    x.nodes['parent_dist'] = mmetrics.parent_dist(x, root_dist=0)
    nodes = x.nodes.set_index('node_id', inplace=False)

    # For each connector (pre and post), get the X/Y distance to its node
    cn_locs = cn[['x', 'y']].values
    tn_locs = nodes.loc[cn.node_id.values,
                        ['x', 'y']].values
    dist = np.sqrt(np.sum((tn_locs - cn_locs) ** 2, axis=1).astype(int))
    cn['dist'] = dist

    # Get max distance per node (in case of multiple connectors per
    # node)
    cn_grouped = cn.groupby('node_id').max()

    # Set undefined radii to None
    nodes.loc[nodes.radius <= 0, 'radius'] = None

    # Assign radii to nodes
    nodes.loc[cn_grouped.index, 'radius'] = cn_grouped.dist.values

    # Go over each segment and interpolate radii
    for s in config.tqdm(x.segments, desc='Interp.', disable=config.pbar_hide,
                         leave=config.pbar_leave):

        # Get this segments radii and parent dist
        this_radii = nodes.loc[s, ['radius', 'parent_dist']]
        this_radii['parent_dist_cum'] = this_radii.parent_dist.cumsum()

        # Set cumulative distance as index and drop parent_dist
        this_radii = this_radii.set_index('parent_dist_cum',
                                          drop=True).drop('parent_dist',
                                                          axis=1)

        # Interpolate missing radii
        interp = this_radii.interpolate(method=method, limit_direction='both',
                                        limit=limit)

        if smooth:
            interp = interp.rolling(smooth,
                                    min_periods=1).max()

        nodes.loc[s, 'radius'] = interp.values

    # Set non-interpolated radii back to -1
    nodes.loc[nodes.radius.isnull(), 'radius'] = -1

    # Reassign nodes
    x.nodes = nodes.reset_index(drop=False, inplace=False)

    if not inplace:
        return x
    else:
        return None


def smooth_neuron(x: NeuronObject,
                  window: int = 5,
                  inplace: bool = False) -> Optional[NeuronObject]:
    """Smooth neuron using rolling windows.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
                    Neuron(s) to be processed.
    window :        int, optional
                    Size (n observations) of the rolling window in number of
                    nodes.
    inplace :       bool, optional
                    If False, will use and return copy of original neuron(s).

    Returns
    -------
    TreeNeuron/List
                    Smoothed neuron(s). If ``inplace=False``.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> smoothed = navis.smooth_neuron(n, window=10)

    """
    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()

        for n in config.tqdm(x, desc='Smoothing', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            smooth_neuron(n, window=window, inplace=True)

        if not inplace:
            return x
        else:
            return None

    elif not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Can only process TreeNeuron or NeuronList, not {type(x)}')

    if not inplace:
        x = x.copy()

    # Prepare nodes (add parent_dist for later, set index)
    mmetrics.parent_dist(x, root_dist=0)
    nodes = x.nodes.set_index('node_id', inplace=False).copy()

    # Go over each segment and smooth
    for s in config.tqdm(x.segments, desc='Smoothing',
                         disable=config.pbar_hide,
                         leave=config.pbar_leave):

        # Get this segment's parent distances and get cumsum
        this_co = nodes.loc[s, ['x', 'y', 'z']]

        interp = this_co.rolling(window, min_periods=1).mean()

        nodes.loc[s, ['x', 'y', 'z']] = interp.values

    # Reassign nodes
    x.nodes = nodes.reset_index(drop=False, inplace=False)

    x._clear_temp_attr()

    if not inplace:
        return x
    else:
        return None


def break_fragments(x: 'core.TreeNeuron') -> 'core.NeuronList':
    """Break neuron into continuous fragments.

    Neurons can consists of several disconnected fragments. This function
    turn these fragments into separate neurons.

    Parameters
    ----------
    x :         TreeNeuron
                Fragmented neuron.

    Returns
    -------
    NeuronList

    See Also
    --------
    :func:`navis.heal_fragmented_neuron`
                Use to heal fragmentation instead of breaking it up.


    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> # Disconnect parts of the neuron
    >>> n.nodes.loc[100, 'parent_id'] = -1
    >>> # Break into fragments
    >>> frags = navis.break_fragments(n)
    >>> len(frags)
    2

    """
    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected Neuron/List, got "{type(x)}"')

    # Don't do anything if not actually fragmented
    if x.n_skeletons > 1:
        # Get connected components
        comp = graph._connected_components(x)
        # Sort so that the first component is the largest
        comp = sorted(comp, key=len, reverse=True)

        return core.NeuronList([graph.subset_neuron(x,
                                                    list(ss),
                                                    inplace=False) for ss in comp])
    else:
        return core.NeuronList(x.copy())


def heal_fragmented_neuron(x: 'core.NeuronList',
                           method: Union[Literal['LEAFS'],
                                         Literal['ALL']] = 'ALL',
                           max_dist: Optional[float] = None,
                           min_size: Optional[float] = None,
                           drop_disc: float = False,
                           inplace: bool = False) -> Optional[NeuronObject]:
    """Heal fragmented neuron(s).

    Tries to heal a fragmented neuron (i.e. a neuron with multiple roots)
    using a minimum spanning tree.

    Parameters
    ----------
    x :         TreeNeuron/List
                Fragmented neuron(s).
    method :    'LEAFS' | 'ALL', optional
                Method used to heal fragments:
                        (1) 'LEAFS': Only leaf (including root) nodes will
                            be used to heal gaps.
                        (2) 'ALL': All nodes can be used to reconnect
                            fragments.
    max_dist :  float, optional
                This effectively sets the max length for newly added edges. Use
                it to prevent far away fragments to be forcefully connected.
    min_size :  int, optional
                Minimum size in nodes for fragments to be reattached. Fragments
                smaller than ``min_size`` will be ignored during stitching and
                hence remain disconnected.
    drop_disc : bool
                If True and the neuron remains fragmented after healing (i.e.
                ``max_dist`` or ``min_size`` prevented a full connect), we will
                keep only the largest (by number of nodes) connected component
                and discard all other fragments.
    inplace :   bool, optional
                If False, will perform healing on and return a copy.

    Returns
    -------
    None
                If ``inplace=True``.
    CatmaidNeuron/List
                If ``inplace=False``.


    See Also
    --------
    :func:`navis.stitch_neurons`
                Use to stitch multiple neurons together.
    :func:`navis.break_fragments`
                Use to produce individual neurons from disconnected fragments.


    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> # Disconnect parts of the neuron
    >>> n.nodes.loc[100, 'parent_id'] = -1
    >>> len(n.root)
    2
    >>> # Heal neuron
    >>> healed = navis.heal_fragmented_neuron(n)
    >>> len(healed.root)
    1

    """
    method = str(method).upper()

    if method not in ['LEAFS', 'ALL']:
        raise ValueError(f'Unknown method "{method}"')

    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()
        _ = [heal_fragmented_neuron(n,
                                    min_size=min_size,
                                    method=method,
                                    max_dist=max_dist,
                                    inplace=True)
             for n in config.tqdm(x,
                                  desc='Healing',
                                  disable=config.pbar_hide,
                                  leave=config.pbar_leave)]
        if not inplace:
            return x
        else:
            return None

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected CatmaidNeuron/List, got "{type(x)}"')

    if not inplace:
        x = x.copy()

    _ = _stitch_mst(x,
                    nodes=method,
                    max_dist=max_dist,
                    min_size=min_size,
                    inplace=True)

    # See if we need to drop remaining disconnected fragments
    if drop_disc:
        # Compute this property only once
        trees = x.subtrees
        if len(trees) > 1:
            # Tree is sorted such that the largest component is the first
            _ = graph.subset_neuron(x, subset=trees[0], inplace=True)

    if not inplace:
        return x


def _stitch_mst(x: 'core.TreeNeuron',
                nodes:  Union[Literal['LEAFS'],
                              Literal['ALL'],
                              list] = 'ALL',
                max_dist: Optional[float] = np.inf,
                min_size: Optional[float] = None,
                inplace: bool = False) -> Optional['core.TreeNeuron']:
    """Stitch disconnected neuron using a minimum spanning tree.

    Parameters
    ----------
    x :             TreeNeuron
                    Neuron to stitch.
    nodes :         "ALL" | "LEAFS" | list of IDs
                    Nodes that can be used to stitch the neuron. Can be "ALL"
                    nodes, just "LEAFS" or a list of node IDs.
    max_dist :      int | float
                    If given, will only connect fragments if they are within
                    ``max_distance``. Use this to prevent the creation of
                    unrealistic edges.
    min_size :      int, optional
                    Minimum size in nodes for fragments to be reattached.
                    Fragments smaller than ``min_size`` will be ignored during
                    stitching and hence remain disconnected.
    inplace :       bool
                    If True, will stitch the original neuron in place.

    Return
    ------
    TreeNeuron
                    Only if ``inplace=True``.

    """
    assert isinstance(x, core.TreeNeuron)
    # Code modified from neuprint-python:
    # https://github.com/connectome-neuprint/neuprint-python
    if max_dist is True or not max_dist:
        max_dist = np.inf

    g = x.graph.to_undirected()

    # Extract each fragment's rows and construct a KD-Tree
    Fragment = namedtuple('Fragment', ['frag_id', 'df', 'kd'])
    fragments = []
    for frag_id, cc in enumerate(nx.connected_components(g)):
        if len(cc) == len(g.nodes):
            # There's only one component -- no healing necessary
            return x

        # Skip if fragment is smaller than threshold
        if not isinstance(min_size, type(None)):
            if len(cc) < min_size:
                continue

        df = x.nodes.query('node_id in @cc')

        # Filter to leaf nodes if applicable
        if isinstance(nodes, str) and nodes == 'LEAFS':
            df = df[df['type'].isin(['end', 'root'])]
        if utils.is_iterable(nodes):
            df = df[df['node_id'].isin(nodes)]

        if not df.empty:
            kd = cKDTree(df[[*'xyz']].values)
            fragments.append(Fragment(frag_id, df, kd))

    # Sort from big-to-small, so the calculations below use a
    # KD tree for the larger point set in every fragment pair.
    fragments = sorted(fragments, key=lambda frag: -len(frag.df))

    # We could use the full graph and connect all
    # fragment pairs at their nearest neighbors,
    # but it's faster to treat each fragment as a
    # single node and run MST on that quotient graph,
    # which is tiny.
    frag_graph = nx.Graph()
    for frag_a, frag_b in combinations(fragments, 2):
        coords_b = frag_b.kd.data
        distances, indexes = frag_a.kd.query(coords_b)

        index_b = np.argmin(distances)
        index_a = indexes[index_b]

        node_a = frag_a.df['node_id'].iloc[index_a]
        node_b = frag_b.df['node_id'].iloc[index_b]
        dist_ab = distances[index_b]

        # Add edge from one fragment to another,
        # but keep track of which fine-grained skeleton
        # nodes were used to calculate distance.
        frag_graph.add_edge(frag_a.frag_id, frag_b.frag_id,
                            node_a=node_a, node_b=node_b,
                            distance=dist_ab)

    # Compute inter-fragment MST edges
    frag_edges = nx.minimum_spanning_edges(frag_graph, weight='distance', data=True)

    # For each inter-fragment edge, add the corresponding
    # fine-grained edge between skeleton nodes in the original graph.
    to_add = [[e[2]['node_a'], e[2]['node_b']] for e in frag_edges if e[2]['distance'] <= max_dist]
    g.add_edges_from(to_add)

    # Rewire based on graph
    return graph.rewire_neuron(x, g, inplace=inplace)
