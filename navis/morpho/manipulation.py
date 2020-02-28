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

import itertools

import pandas as pd
import numpy as np
import scipy.spatial.distance
import networkx as nx

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
    """ Prune neuron based on `Strahler order
    <https://en.wikipedia.org/wiki/Strahler_number>`_.

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
                           ) for n in x]
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

    if reroot_soma and neuron.soma:
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

    neuron.nodes = neuron.nodes[~neuron.nodes.strahler_index.isin(to_prune)].reset_index(drop=True, inplace=False)

    if neuron.has_connectors:
        if not relocate_connectors:
            neuron.connectors = neuron.connectors[neuron.connectors.node_id.isin(neuron.nodes.node_id.values)].reset_index(drop=True, inplace=False)
        else:
            remaining_tns = set(neuron.nodes.node_id.values)
            for cn in neuron.connectors[~neuron.connectors.node_id.isin(neuron.nodes.node_id.values)].itertuples():
                this_tn = parent_dict[cn.node_id]
                while True:
                    if this_tn in remaining_tns:
                        break
                    this_tn = parent_dict[this_tn]
                neuron.connectors.loc[cn.Index, 'node_id'] = this_tn

    # Reset indices of node and connector tables (important for igraph!)
    neuron.nodes.reset_index(inplace=True, drop=True)

    if neuron.has_connectors:
        neuron.connectors.reset_index(inplace=True, drop=True)

    # Theoretically we can end up with disconnected pieces, i.e. with more
    # than 1 root node -> we have to fix the nodes that lost their parents
    neuron.nodes.loc[~neuron.nodes.parent_id.isin(neuron.nodes.node_id.values), 'parent_id'] = -1

    # Remove temporary attributes
    neuron._clear_temp_attr()

    if not inplace:
        return neuron
    else:
        return None


@overload
def prune_twigs(x: NeuronObject,
                size: float,
                inplace: Literal[True],
                recursive: Union[int, bool, float] = False
                ) -> None: ...


@overload
def prune_twigs(x: NeuronObject,
                size: float,
                inplace: Literal[False],
                recursive: Union[int, bool, float] = False
                ) -> NeuronObject: ...


@overload
def prune_twigs(x: NeuronObject,
                size: float,
                inplace: bool = False,
                recursive: Union[int, bool, float] = False
                ) -> Optional[NeuronObject]: ...


def prune_twigs(x: NeuronObject,
                size: float,
                inplace: bool = False,
                recursive: Union[int, bool, float] = False
                ) -> Optional[NeuronObject]:
    """ Prune terminal twigs under a given size.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
    size :          int | float
                    Twigs shorter than this will be pruned.
    inplace :       bool, optional
                    If False, pruning is performed on copy of original neuron
                    which is then returned.
    recursive :     int | bool | float("inf"), optional
                    If `int` will undergo that many rounds of recursive
                    pruning. Use `float(`"inf")`` to prune until no more twigs
                    under the given size are left.

    Returns
    -------
    TreeNeuron/List
                    Pruned neuron(s).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> # Prune twigs smaller than 5 microns (example neurons are in nm)
    >>> n_pr = navis.prune_twigs(n,
    ...                          size=5000,
    ...                          recursive=float('inf'),
    ...                          inplace=False)
    >>> n.n_nodes > n_pr.n_nodes
    True

    """

    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()

        [prune_twigs(n,
                     size=size,
                     inplace=True,
                     recursive=recursive) for n in tqdm(x,
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
    segs = np.array([s for s in segs if s[0] in leafs])

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


def split_axon_dendrite(x: NeuronObject,
                        method: Union[Literal['centrifugal'],
                                      Literal['centripetal'],
                                      Literal['sum'],
                                      Literal['bending']] = 'bending',
                        split: Union[Literal['prepost'],
                                     Literal['distance']] = 'prepost',
                        reroot_soma: bool = True,
                        return_point: bool = False) -> 'core.NeuronList':
    """Split a neuron into axon, dendrite and primary neurite.

    The result is highly dependent on the method and on your neuron's
    morphology and works best for "typical" neurons, i.e. those where the
    primary neurite branches into axon and dendrites.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
                        Neuron(s) to split into axon, dendrite (and primary
                        neurite if possible). MUST HAVE CONNECTORS.
    method :            'centrifugal' | 'centripetal' | 'sum' | 'bending', optional
                        Type of flow centrality to use to split the neuron.
                        There are four flavors: the first three refer to
                        :func:`~navis.flow_centrality`, the last
                        refers to :func:`~navis.bending_flow`.

                        Will try using stored centrality, if possible.
    split :             "prepost" | "distance"
                        Method for determining which compartment is axon and
                        which is the dendrites:

                            - "prepost" uses number of in- vs. outputs
                            - "distance" assumes the compartment closer to the
                              soma is the dendrites


    reroot_soma :       bool, optional
                        If True, will make sure neuron is rooted to soma if at
                        all possible.
    return_point :      bool, optional
                        If True, will only return node ID of the node at
                        which to split the neuron.

    Returns
    -------
    NeuronList
                        Axon, dendrite and primary neurite. Fragments will
                        have a new property ``compartment`` (see example).

    Examples
    --------
    >>> import navis
    >>> x = navis.example_neurons(1)
    >>> split = navis.split_axon_dendrite(x, method='centrifugal',
    ...                                   reroot_soma=True)
    >>> split
    <class 'navis.NeuronList'> of 3 neurons
                          neuron_name skeleton_id  n_nodes  n_connectors
    0                  neuron 123457          16      148             0
    1                  neuron 123457          16     9682          1766
    2                  neuron 123457          16     2892           113
    >>> # For convenience, split_axon_dendrite assigns colors to the resulting
    >>> # fragments: axon = red, dendrites = blue, primary neurite = green
    >>> split.plot3d(color=split.color)

    See Also
    --------
    :func:`navis.heal_fragmented_neuron`
            Axon/dendrite split works only on neurons consisting of a single
            tree. Use this function to heal fragmented neurons.

    """
    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]
    elif isinstance(x, core.NeuronList):
        nl = []
        for n in config.tqdm(x, desc='Splitting', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            nl.append(split_axon_dendrite(n,
                                          method=method,
                                          split=split,
                                          reroot_soma=reroot_soma,
                                          return_point=return_point))
        return core.NeuronList([n for l in nl for n in l])

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Can only process TreeNeuron, got "{type(x)}"')

    if not x.has_connectors:
        raise ValueError('Neuron must have connectors.')

    if method not in ['centrifugal', 'centripetal', 'sum', 'bending']:
        raise ValueError(f'"{method}" not allowed for parameter `method`.')

    if split not in ['prepost', 'distance']:
        raise ValueError(f'"{split}" not allowed for parameter `split`.')

    if len(x.root) > 1:
        raise ValueError(f'Unable to split neuron {x.id}: multiple roots. '
                         'Try navis.heal_fragmented_neuron(x) to merged '
                         'disconnected fragments.')

    if x.soma and x.soma not in x.root and reroot_soma:
        x.reroot(x.soma, inplace=True)

    # Calculate flow centrality if necessary
    last_method = getattr(x, 'centrality_method', None)

    if last_method != method:
        if method == 'bending':
            mmetrics.bending_flow(x)
        elif method in ['centripetal', 'centrifugal', 'sum']:
            # At this point method is not "bending"
            method: Union[Literal['centripetal'],
                          Literal['centrifugal'],
                          Literal['sum']]
            mmetrics.flow_centrality(x, mode=method)

    # Make copy, so that we don't screw things up
    x = x.copy()

    # Now get the node point with the highest flow centrality.
    cut = x.nodes[x.nodes.flow_centrality
                  == x.nodes.flow_centrality.max()].node_id.values

    # If more than one point we need to get one closest to the soma (root)
    if len(cut) > 1:
        cut = sorted(cut, key=lambda y: graph.dist_between(x.graph, y, x.root[0]))[0]
    else:
        cut = cut[0]

    if return_point:
        return cut

    # If cut node is a branch point, we will try cutting off main neurite
    if x.graph.degree(cut) > 2:
        # First make sure that there are no other branch points with flow
        # between this one and the soma
        path_to_root = nx.shortest_path(x.graph, cut, x.root[0])

        # Get flow centrality along the path
        flows = x.nodes.set_index('node_id', inplace=False).loc[path_to_root]

        # Subset to those that are branches (exclude mere synapses)
        flows = flows[flows.type == 'branch']

        # Find the first branch point from the soma with no flow (fillna is
        # important!)
        last_with_flow = np.where(flows.flow_centrality.fillna(0).values > 0)[0][-1]

        if method != 'bending':
            last_with_flow += 1

        to_cut = flows.iloc[last_with_flow].name

        # Cut off primary neurite
        rest, primary_neurite = graph.cut_neuron(x, to_cut)

        if method == 'bending':
            # The new cut node has to be a child of the original cut node
            cut = next(x.graph.predecessors(cut))

        # Change compartment and color
        primary_neurite.color = (0, 255, 0)  # type: ignore
        primary_neurite._register_attr('compartment', 'primary_neurite')  # type: ignore
    else:
        rest = x
        primary_neurite = None

    # Next, cut the rest into axon and dendrite
    a, b = graph.cut_neuron(rest, cut)

    # Figure out which one is which
    if split == 'prepost' or split == 'distance':
        a_inout = a.n_postsynapses / a.n_presynapses if a.n_presynapses else float('inf')
        b_inout = b.n_postsynapses / b.n_presynapses if b.n_presynapses else float('inf')
        if a_inout > b_inout:
            dendrite, axon = a, b
        else:
            dendrite, axon = b, a

    # Add compartment property
    axon._register_attr('compartment', 'axon')  # type: ignore
    dendrite._register_attr('compartment', 'dendrite')  # type: ignore

    # Change colors
    axon.color = (255, 0, 0)  # type: ignore
    dendrite.color = (0, 0, 255)  # type: ignore

    if primary_neurite:
        return core.NeuronList([primary_neurite, axon, dendrite])
    else:
        return core.NeuronList([axon, dendrite])


def stitch_neurons(*x: Union[Sequence[NeuronObject], 'core.NeuronList'],
                   method: Union[Literal['LEAFS'],
                                 Literal['ALL'],
                                 Literal['NONE']] = 'LEAFS',
                   master: Union[Literal['SOMA'],
                                 Literal['LARGEST'],
                                 Literal['FIRST']] = 'SOMA',
                   tn_to_stitch: Optional[Sequence[int]] = None,
                   suggest_only: bool = False,
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
    method :            'LEAFS' | 'ALL' | 'NONE', optional
                        Set stitching method:
                            (1) 'LEAFS': Only leaf (including root) nodes will
                                be allowed to make new edges.
                            (2) 'ALL': All nodes are considered.
                            (3) 'NONE': Node and connector tables will simply
                                be combined without generating any new edges.
                                The resulting neuron will have multiple roots.
    master :            'SOMA' | 'LARGEST' | 'FIRST', optional
                        Sets the master neuron:
                            (1) 'SOMA': The largest fragment with a soma
                                becomes the master neuron. If no neuron with
                                soma, will pick the largest (option 2).
                            (2) 'LARGEST': The largest fragment becomes the
                                master neuron.
                            (3) 'FIRST': The first fragment provided becomes
                                the master neuron.
    tn_to_stitch :      List of node IDs, optional
                        If provided, these nodes will be preferentially
                        used to stitch neurons together. Overrides methods
                        ``'ALL'`` or ``'LEAFS'``.
    suggest_only :      bool, optional
                        If True, will only return list of edges to add instead
                        of actually stitching the neuron.
    max_dist :          float,  optional
                        Max distance at which to stitch nodes. Setting this can
                        drastically speed up the process but can also lead to
                        failed stitching.

    Returns
    -------
    TreeNeuron
                        Stitched neuron.

    Examples
    --------
    Stitching neuronlist by simply combining data tables:

    >>> nl = navis.example_neurons(2)
    >>> stitched = navis.stitch_neurons(nl, method='NONE')

    Stitching fragmented neurons:
    >>> a = navis.example_neurons(1)
    >>> fragments = navis.cut_neuron(a, 100)
    >>> stitched = navis.stitch_neurons(frag, method='LEAFS')

    """
    method = str(method).upper()
    master = str(master).upper()

    if method not in ['LEAFS', 'ALL', 'NONE']:
        raise ValueError(f'Unknown method: "{method}"')

    if master not in ['SOMA', 'LARGEST', 'FIRST']:
        raise ValueError(f'Unknown master: "{master}"')

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
                       key=lambda x: list(nl.cable_length),
                       reverse=True)[0]
    elif master == 'LARGEST':
        m = sorted(nl.neurons,
                   key=lambda x: list(nl.cable_length),
                   reverse=True)[0]
    else:
        # Simply pick the first neuron
        m = nl[0]

    # Check if we need to make any node IDs unique
    if nl.nodes.duplicated(subset='node_id').sum() > 0:
        seen_tn: Set[int] = set(m.nodes.node_id)
        for n in [n for n in nl if n != m]:
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
                n.nodes.node_id = n.nodes.node_id.map(lambda x: new_map.get(x, x))

                if n.has_connectors:
                    n.connectors.node_id = n.connectors.node_id.map(lambda x: new_map.get(x, x))

                if hasattr(n, 'tags'):
                    n.tags = {new_map.get(k, k): v for k, v in n.tags.items()}  # type: ignore

                # Remap parent IDs
                new_map[None] = -1  # type: ignore
                n.nodes.parent_id = n.nodes.parent_id.map(lambda x: new_map.get(x, x)).astype(int)

                # Add new nodes to seen
                seen_tn = seen_tn | set(new_tn)

                # Make sure the graph is updated
                n._clear_temp_attr()

    # If method is none, we can just merge the data tables
    if method == 'NONE' or method is None:
        m.nodes = pd.concat([n.nodes for n in nl],  # type: ignore  # no stubs for concat
                            ignore_index=True)

        if any(nl.has_connectors):
            m.connectors = pd.concat([n.connectors for n in nl],  # type: ignore  # no stubs for concat
                                     ignore_index=True)

        m.tags = {}  # type: ignore  # TreeNeuron has no tags
        for n in nl:
            for k, v in getattr(n, 'tags', {}):
                m.tags[k] = m.tags.get(k, []) + list(utils.make_iterable(v))

        # Reset temporary attributes of our final neuron
        m._clear_temp_attr()

        return m

    # Fix potential problems with tn_to_stitch
    if not isinstance(tn_to_stitch, type(None)):
        if not isinstance(tn_to_stitch, (list, np.ndarray)):
            tn_to_stitch = [tn_to_stitch]

        # Make sure we're working with integers
        tn_to_stitch = [int(tn) for tn in tn_to_stitch]

    # Generate a list of potential new edges
    # Collect relevant nodes
    if not isinstance(tn_to_stitch, type(None)):
        tn = nl.nodes.loc[nl.nodes.node_id.isin(tn_to_stitch)]
    elif method == 'LEAFS':
        tn = nl.nodes.loc[nl.nodes['type'].isin(['end', 'root'])]
    else:
        tn = nl.nodes

    # Get pairwise distance between nodes
    # cdist
    d = scipy.spatial.distance.cdist(tn[['x', 'y', 'z']].values,
                                     tn[['x', 'y', 'z']].values,
                                     metric='euclidean')
    d = pd.DataFrame(d, index=tn.node_id.values, columns=tn.node_id.values)

    # Mask for the upper triangle
    mask = np.triu(np.ones(d.shape)).astype(np.bool)

    # Set lower triangle to NaN
    d = d.where(mask)

    # Stack
    new_edges = d.stack().reset_index(drop=False)
    new_edges.columns = ['source', 'target', 'weight']

    # Kick stuff that's too far away
    if max_dist:
        new_edges = new_edges[new_edges.weight <= max_dist]

    # Remove self edges
    new_edges = new_edges[new_edges.source != new_edges.target]

    # Remove edges that are within the same neuron
    id2ix = tn.set_index('node_id').neuron
    new_edges = new_edges[new_edges.source.map(id2ix) != new_edges.target.map(id2ix)]

    # Extract edges that need to be added to connect fragments
    if config.use_igraph and all([n.igraph for n in nl]):
        to_add = _mst_igraph(nl, new_edges)
    else:
        to_add = _mst_nx(nl, new_edges)

    if suggest_only:
        return to_add

    # Keep track of original master root
    master_root = m.root[0]

    # Generate one big neuron
    m.nodes = nl.nodes

    if any(nl.has_connectors):
        m.connectors = nl.connectors

    if any([hasattr(n, 'tags') for n in nl]):
        m.tags = {}  # type: ignore  # TreeNeuron has no tags
        for n in nl:
            m.tags.update(getattr(n, 'tags', {}))

    # Clear temporary attributes
    m._clear_temp_attr()

    for e in to_add:
        # Reroot to one of the nodes in the edge
        m.reroot(e[0], inplace=True)

        # Connect the nodes
        m.nodes.loc[m.nodes.node_id == e[0], 'parent_id'] = e[1]

        # Add edge to graphs
        if config.use_igraph and m.igraph:
            m.igraph.add_edge(m.igraph.vs.find(node_id=e[0]),
                              m.igraph.vs.find(node_id=e[1]),
                              **e[2])
        # We only really need to update this graph if we need it for reroot
        else:
            m.graph.add_edge(e[0], e[1], **e[2])

        # Add node tags
        m.tags = getattr(m, 'tags', {})  # type: ignore  # TreeNeuron has no tags
        m.tags['stitched'] = m.tags.get('stitched', []) + [e[0], e[1]]

        # We don't need this because reroot is taking care of this already
        # m._clear_temp_attr(exclude=['graph', 'igraph'])

    # Reroot to original root
    m.reroot(master_root, inplace=True)

    return m


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
    """ Computes an average from a list of neurons.

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
    >>> da2 = navis.example_neurons()
    >>> # Prune down to longest neurite
    >>> da2.reroot(da2.soma)
    >>> da2_pr = da2.prune_by_longest_neurite(inplace=False)
    >>> # Make average
    >>> da2_avg = navis.average_neurons(da2_pr, limit=10e3)
    >>> # Plot
    >>> da2.plot3d()
    >>> da2_avg.plot3d()

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
        raise ValueError('Unable to interpret base_neuron of '
                         'type "{0}"'.format(type(base_neuron)))

    base_nodes = bn.nodes[['x', 'y', 'z']].values
    other_neurons = x[1:]

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
    """ Removes spikes in neuron traces (e.g. from jumps in image data).

    For each node A, the euclidean distance to its next successor (parent)
    B and that node's successor is computed. If
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
    """ Tries guessing radii for all nodes.

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
    mmetrics.parent_dist(x, root_dist=0)
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
    """ Smooth neuron using rolling windows.

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
                           min_size: int = 0,
                           method: Union[Literal['LEAFS'],
                                         Literal['ALL']] = 'LEAFS',
                           max_dist: Optional[float] = None,
                           inplace: bool = False) -> Optional[NeuronObject]:
    """ Heal fragmented neuron(s).

    Tries to heal a fragmented neuron (i.e. a neuron with multiple roots)
    using a minimum spanning tree.

    Parameters
    ----------
    x :         TreeNeuron/List
                Fragmented neuron(s).
    min_size :  int, optional
                Minimum size in nodes for fragments to be reattached.
    method :    'LEAFS' | 'ALL', optional
                Method used to heal fragments:
                        (1) 'LEAFS': Only leaf (including root) nodes will
                            be used to heal gaps.
                        (2) 'ALL': All nodes can be used to reconnect
                            fragments.
    max_dist :  float, optional
                Max distance at which to merge fragments. Setting this to a
                reasonable value can dramatically speed things up. If too low,
                this can lead to failed healing.
    inplace :   bool, optional
                If False, will perform healing on and return a copy.

    Returns
    -------
    None
                If ``inplace=True``
    CatmaidNeuron/List
                If ``inplace=False``


    See Also
    --------
    :func:`navis.stitch_neurons`
                Function used by ``heal_fragmented_neuron`` to stitch
                fragments.
    :func:`navis.break_fragments`
                Use to break a fragmented neuron into disconnected pieces.


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
        healed = [heal_fragmented_neuron(n,
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

    # Only process if actually fragmented
    if x.n_skeletons > 1:
        frags = break_fragments(x)
        healed = stitch_neurons(*[f for f in frags if f.n_nodes > min_size],
                                max_dist=max_dist,
                                method=method)
        if not inplace:
            return healed
        else:
            x.nodes = healed.nodes  # update nodes
            x.tags = healed.tags  # update tags
            x._clear_temp_attr()
            return None
    else:
        if not inplace:
            return x
        else:
            return None
