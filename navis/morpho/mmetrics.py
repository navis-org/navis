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


""" This module contains functions to analyse and manipulate neuron morphology.
"""

import math
import itertools
import scipy

import pandas as pd
import numpy as np

from typing import Union, Optional, Sequence, List, Dict, overload
from typing_extensions import Literal

from .. import config, graph, sampling, core, utils

# Set up logging
logger = config.logger

__all__ = sorted(['strahler_index', 'bending_flow', 'sholl_analysis',
                  'flow_centrality', 'segregation_index', 'tortuosity'])


def parent_dist(x: Union['core.TreeNeuron', pd.DataFrame],
                root_dist: Optional[int] = None) -> None:
    """Get child->parent distances for skeleton nodes.

    Parameters
    ----------
    x :         TreeNeuron | node table
    root_dist : int | None
                ``parent_dist`` for the root's row. Set to ``None``, to leave
                at ``NaN`` or e.g. to ``0`` to set to 0.

    Returns
    -------
    np.ndarray
                Array with distances in same order and size as node table.

    """
    if isinstance(x, core.TreeNeuron):
        nodes = x.nodes
    elif isinstance(x, pd.DataFrame):
        nodes = x
    else:
        raise TypeError(f'Need TreeNeuron or DataFrame, got "{type(x)}"')

    # Extract node coordinates
    tn_coords = nodes[['x', 'y', 'z']].values

    # Get parent coordinates
    parent_coords = nodes.set_index('node_id').reindex(nodes.parent_id.values)[['x', 'y', 'z']].values

    # Calculate distances between nodes and their parents
    w = np.sqrt(np.sum((tn_coords - parent_coords) ** 2, axis=1))

    # Replace root dist (nan by default)
    w[np.isnan(w)] = root_dist

    return w


@utils.map_neuronlist(desc='Calc. SI', allow_parallel=True)
@utils.meshneuron_skeleton(method='node_properties',
                           reroot_soma=True,
                           node_props=['strahler_index'])
def strahler_index(x: 'core.NeuronObject',
                   method: Union[Literal['standard'],
                                 Literal['greedy']] = 'standard',
                   to_ignore: list = [],
                   min_twig_size: Optional[int] = None
                   ) -> 'core.NeuronObject':
    """Calculate Strahler Index (SI).

    Starts with SI of 1 at each leaf and walks to root. At forks with different
    incoming SIs, the highest index is continued. At forks with the same
    incoming SI, highest index + 1 is continued.

    Parameters
    ----------
    x :                 TreeNeuron | MeshNeuron | NeuronList
    method :            'standard' | 'greedy', optional
                        Method used to calculate Strahler indices: 'standard'
                        will use the method described above; 'greedy' will
                        always increase the index at converging branches
                        whether these branches have the same index or not.
    to_ignore :         iterable, optional
                        List of node IDs to ignore. Must be the FIRST node
                        of the branch. Excluded branches will not contribute
                        to Strahler index calculations and instead be assigned
                        the SI of their parent branch.
    min_twig_size :     int, optional
                        If provided, will ignore twigs with fewer nodes than
                        this. Instead, they will be assigned the SI of their
                        parent branch.

    Returns
    -------
    neuron
                Adds "strahler_index" as column in the node table (for
                TreeNeurons) or as `."strahler_index` property
                (for MeshNeurons).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(2, kind='skeleton')
    >>> n.reroot(n.soma, inplace=True)
    >>> _ = navis.strahler_index(n)
    >>> n[0].nodes.strahler_index.max()
    6
    >>> m = navis.example_neurons(1, kind='mesh')
    >>> _ = navis.strahler_index(m)
    >>> m.strahler_index.max()
    5

    """
    utils.eval_param(x, name='x', allowed_types=(core.TreeNeuron, ))

    # Find branch, root and end nodes
    if 'type' not in x.nodes:
        graph.classify_nodes(x)

    end_nodes = x.nodes[x.nodes.type == 'end'].node_id.values
    branch_nodes = x.nodes[x.nodes.type == 'branch'].node_id.values
    root = x.nodes[x.nodes.type == 'root'].node_id.values

    end_nodes = set(end_nodes)
    branch_nodes = set(branch_nodes)
    root = set(root)

    if min_twig_size:
        to_ignore = np.append(to_ignore,
                              [seg[0] for seg in x.small_segments if seg[0]
                               in end_nodes and len(seg) < min_twig_size])

    # Generate dicts for childs and parents
    list_of_childs = graph.generate_list_of_childs(x)

    # Get a node ID -> parent ID dictionary for FAST lookups
    parents = x.nodes.set_index('node_id').parent_id.to_dict()

    # Do NOT name any parameter `strahler_index` - this overwrites the function!
    SI: Dict[int, int] = {}

    starting_points = end_nodes
    seen = set()
    while starting_points:
        logger.debug(f'New starting point. Remaining: {len(starting_points)}')
        this_node = starting_points.pop()

        # Get upstream indices for this branch
        previous_indices = [SI.get(c, None) for c in list_of_childs[this_node]]

        # If this is a not-a-branch branch
        if this_node in to_ignore:
            this_branch_index = None
        # If this is an end node: start at 1
        elif len(previous_indices) == 0:
            this_branch_index = 1
        # If this is a slab: assign SI of predecessor
        elif len(previous_indices) == 1:
            this_branch_index = previous_indices[0]
        # If this is a branch point at which similar indices collide: +1
        elif method == 'greedy' or previous_indices.count(max(previous_indices)) >= 2:
            this_branch_index = max(previous_indices) + 1
        # If just a branch point: continue max SI
        else:
            this_branch_index = max(previous_indices)

        # Keep track of that this node has been processed
        seen.add(this_node)

        # Now walk down this segment
        # Find parent
        segment = [this_node]
        parent_node = parents[this_node]
        while parent_node >= 0 and parent_node not in branch_nodes:
            this_node = parent_node
            parent_node = parents[this_node]
            segment.append(this_node)
            seen.add(this_node)

        # Update indices for the entire segment
        SI.update({n: this_branch_index for n in segment})

        # The last `this_node` is either a branch node or the root
        # If a branch point: check, if all its childs have already been
        # processed
        if parent_node > 0:
            node_ready = True
            for child in list_of_childs[parent_node]:
                if child not in seen:
                    node_ready = False
                    break

            if node_ready is True:
                starting_points.add(parent_node)

    # Fix branches that were ignored
    if to_ignore:
        # Go over all terminal branches with the tag
        for tn in x.nodes[(x.nodes.type == 'end') & x.nodes.node_id.isin(to_ignore)].node_id.values:
            # Get this terminal's segment
            this_seg = [s for s in x.small_segments if s[0] == tn][0]
            # Get strahler index of parent branch
            this_SI = SI.get(this_seg[-1], 1)
            SI.update({n: this_SI for n in this_seg})

    # Disconnected single nodes (e.g. after pruning) will end up w/o an entry
    # --> we will give them an SI of 1
    x.nodes['strahler_index'] = x.nodes.node_id.map(lambda x: SI.get(x, 1))

    # Set correct data type
    x.nodes['strahler_index'] = x.nodes.strahler_index.astype(np.int16)

    return x


def segregation_index(x: Union['core.NeuronObject', dict]) -> float:
    """Calculate segregation index (SI).

    The segregation index as established by Schneider-Mizell et al. (eLife,
    2016) is a measure for how polarized a neuron is. SI of 1 indicates total
    segregation of inputs and outputs into dendrites and axon, respectively.
    SI of 0 indicates homogeneous distribution.

    Parameters
    ----------
    x :                 NeuronList | list
                        Neuron to calculate segregation index (SI) for. If a
                        NeuronList, will assume that it contains
                        fragments (e.g. from axon/ dendrite splits) of a
                        single neuron. If list, must be records containing
                        number of pre- and postsynapses for each fragment::

                            [{'presynapses': 10, 'postsynapses': 320},
                             {'presynapses': 103, 'postsynapses': 21}]

    Notes
    -----
    From Schneider-Mizell et al. (2016): "Note that even a modest amount of
    mixture (e.g. axo-axonic inputs) corresponds to values near H = 0.5–0.6
    (Figure 7—figure supplement 1). We consider an unsegregated neuron
    (H ¡ 0.05) to be purely dendritic due to their anatomical similarity with
    the dendritic domains of those segregated neurons that have dendritic
    outputs."

    Returns
    -------
    H :                 float
                        Segregation Index (SI).

    """
    if not isinstance(x, (core.NeuronList, list)):
        raise ValueError(f'Expected NeuronList or list got "{type(x)}"')

    if isinstance(x, core.NeuronList) and len(x) <= 1:
        raise ValueError(f'Expected multiple neurons, got {len(x)}')

    # Turn NeuronList into records
    if isinstance(x, core.NeuronList):
        x = [{'presynapses': n.n_presynapses, 'postsynapses': n.n_postsynapses}
             for n in x]

    # Extract the total number of pre- and postsynapses
    total_pre = sum([n['presynapses'] for n in x])
    total_post = sum([n['postsynapses'] for n in x])
    total_syn = total_pre + total_post

    # Calculate entropy for each fragment
    entropy = []
    for n in x:
        n['total_syn'] = n['postsynapses'] + n['presynapses']

        # This is to avoid warnings
        if n['total_syn']:
            p = n['postsynapses'] / n['total_syn']
        else:
            p = float('inf')

        if 0 < p < 1:
            S = - (p * math.log(p) + (1 - p) * math.log(1 - p))
        else:
            S = 0

        entropy.append(S)

    # Calc entropy between fragments
    S = 1 / total_syn * sum([e * n['total_syn'] for n, e in zip(x, entropy)])

    # Normalize to entropy in whole neuron
    p_norm = total_post / total_syn
    if 0 < p_norm < 1:
        S_norm = -(p_norm * math.log(p_norm) + (1 - p_norm) * math.log(1 - p_norm))
        H = 1 - S / S_norm
    else:
        S_norm = 0
        H = 0

    return H


@utils.map_neuronlist(desc='Calc. seg.', allow_parallel=True)
@utils.meshneuron_skeleton(method='node_properties',
                           node_props=['segregation_index'])
def arbor_segregation_index(x: 'core.NeuronObject') -> 'core.NeuronObject':
    """Per arbor seggregation index (SI).

    The segregation index (SI) as established by Schneider-Mizell et al. (eLife,
    2016) is a measure for how polarized a neuron is. SI of 1 indicates total
    segregation of inputs and outputs into dendrites and axon, respectively.
    SI of 0 indicates homogeneous distribution. Here, we apply this to
    each arbour within a neuron by asking "If we were to cut a neuron at this
    node, what would the SI of the two resulting fragments be?"

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | NeuronList
                Neuron(s) to calculate segregation indices for. Must have
                connectors!

    See Also
    --------
    :func:`~navis.segregation_index`
            Calculate segregation score (polarity) between two fragments of
            a neuron.
    :func:`~navis.flow_centrality`
            Calculate synapse flow centrality after Schneider-Mizell et al.
    :func:`~navis.bending_flow`
            Variation on the Schneider-Mizell et al. synapse flow.
    :func:`~navis.split_axon_dendrite`
            Split the neuron into axon, dendrite and primary neurite.

    Returns
    -------
    neuron
                Adds "segregation_index" as column in the node table (for
                TreeNeurons) or as `.segregation_index` property
                (for MeshNeurons).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n.reroot(n.soma, inplace=True)
    >>> _ = navis.arbor_segregation_index(n)
    >>> n.nodes.segregation_index.max().round(3)
    0.277

    """
    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Expected TreeNeuron(s), got "{type(x)}"')

    if not x.has_connectors:
        raise ValueError('Neuron must have connectors.')

    # Figure out how connector types are labeled
    cn_types = x.connectors.type.unique()
    if all(np.isin(['pre', 'post'], cn_types)):
        pre, post = 'pre', 'post'
    elif all(np.isin([0, 1], cn_types)):
        pre, post = 0, 1
    else:
        raise ValueError(f'Unable to parse connector types for neuron {x.id}')

    # Get list of nodes with pre/postsynapses
    pre_node_ids = x.connectors[x.connectors.type == pre].node_id.values
    post_node_ids = x.connectors[x.connectors.type == post].node_id.values

    # Get list of points to calculate SI for:
    # branches points and their children plus nodes with connectors
    is_bp = x.nodes['type'].isin(['branch', 'root'])
    is_bp_child = x.nodes.parent_id.isin(x.nodes.loc[is_bp, 'node_id'].values)
    is_cn = x.nodes.node_id.isin(x.connectors.node_id)
    calc_node_ids = x.nodes[is_bp | is_bp_child | is_cn].node_id.values

    # We will be processing a super downsampled version of the neuron to speed
    # up calculations
    current_level = logger.level
    logger.setLevel('ERROR')
    y = x.downsample(factor=float('inf'),
                     preserve_nodes=calc_node_ids,
                     inplace=False)
    logger.setLevel(current_level)

    # Get number of pre/postsynapses distal to each branch's childs
    distal = graph.distal_to(y, np.append(pre_node_ids, post_node_ids),
                             calc_node_ids)

    # Since nodes can have multiple pre-/postsynapses but they show up only
    # once in distal, we have to reindex to reflect the correct number of synapes
    distal_pre = distal.loc[pre_node_ids]
    distal_post = distal.loc[post_node_ids]

    # Sum up columns: now each row represents the number of pre/postsynapses
    # distal to that node
    distal_pre_sum = distal_pre.sum(axis=0)
    distal_post_sum = distal_post.sum(axis=0)

    # Now go over all branch points and check flow between branches
    # (centrifugal) vs flow from branches to root (centripetal)
    SI = {}
    total_pre = pre_node_ids.shape[0]
    total_post = post_node_ids.shape[0]
    for n in calc_node_ids:
        # Get the SI if we were to cut at this point
        post = distal_post_sum[n]
        pre = distal_pre_sum[n]
        n_syn = [{'presynapses': pre, 'postsynapses': post},
                 {'presynapses': total_pre - pre, 'postsynapses': total_post - post}]
        SI[n] = segregation_index(n_syn)

    # At this point there are only segregation indices for branch points and
    # their childs. Let's complete that mapping by adding SI for the nodes
    # between branch points.
    for s in x.small_segments:
        # Segments' orientation goes from distal -> proximal
        # Each segment will have at least its last (branch point) and
        # second last (branch point's child) node mapped

        # Drop first (distal) node if it is not a leaf
        if s[0] in SI:
            s = s[1:]

        # If shorter than 3 nodes all nodes should already have an SI
        if len(s) <= 2:
            continue

        # Update remaining nodes with the SI of the first child
        this_SI = SI[s[-2]]
        SI.update({n: this_SI for n in s[:-2]})

    # Add segregation index to node table
    x.nodes['segregation_index'] = x.nodes.node_id.map(SI)

    return x


@utils.map_neuronlist(desc='Calc. flow', allow_parallel=True)
@utils.meshneuron_skeleton(method='node_properties',
                           include_connectors=True,
                           heal=True,
                           node_props=['bending_flow'])
def bending_flow(x: 'core.NeuronObject') -> 'core.NeuronObject':
    """Calculate bending flow.

    This is a variation of the algorithm for calculating synapse flow from
    Schneider-Mizell et al. (eLife, 2016).

    The way this implementation works is by iterating over each branch point
    and counting the number of pre->post synapse paths that "flow" from one
    child branch to the other(s).

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | NeuronList
                Neuron(s) to calculate bending flow for. Must have connectors!

    Notes
    -----
    This is algorithm appears to be more reliable than synapse flow
    centrality for identifying the main branch point for neurons that have
    incompletely annotated synapses.

    See Also
    --------
    :func:`~navis.flow_centrality`
            Calculate synapse flow centrality after Schneider-Mizell et al.
    :func:`~navis.segregation_index`
            Calculate segregation score (polarity).
    :func:`~navis.arbor_segregation_index`
            Calculate the a by-arbor segregation index.
    :func:`~navis.split_axon_dendrite`
            Split the neuron into axon, dendrite and primary neurite.

    Returns
    -------
    neuron
                Adds "bending_flow" as column in the node table (for
                TreeNeurons) or as `.bending_flow` property
                (for MeshNeurons).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n.reroot(n.soma, inplace=True)
    >>> _ = navis.bending_flow(n)
    >>> n.nodes.bending_flow.max()
    785645

    """
    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Expected TreeNeuron(s), got "{type(x)}"')

    if not x.has_connectors:
        raise ValueError('Neuron must have connectors.')

    if np.any(x.soma) and not np.all(np.isin(x.soma, x.root)):
        logger.warning(f'Neuron {x.id} is not rooted to its soma!')

    # We will be processing a super downsampled version of the neuron to speed
    # up calculations
    current_level = logger.level
    logger.setLevel('ERROR')
    y = x.downsample(factor=float('inf'),
                     preserve_nodes='connectors',
                     inplace=False)
    logger.setLevel(current_level)

    # Figure out how connector types are labeled
    cn_types = y.connectors.type.unique()
    if all(np.isin(['pre', 'post'], cn_types)):
        pre, post = 'pre', 'post'
    elif all(np.isin([0, 1], cn_types)):
        pre, post = 0, 1
    else:
        raise ValueError(f'Unable to parse connector types for neuron {y.id}')

    # Get list of nodes with pre/postsynapses
    pre_node_ids = y.connectors[y.connectors.type == pre].node_id.values
    post_node_ids = y.connectors[y.connectors.type == post].node_id.values

    # Get list of branch_points
    bp_node_ids = y.nodes[y.nodes.type == 'branch'].node_id.values.tolist()
    # Add root if it is also a branch point
    for root in y.root:
        if y.graph.degree(root) > 1:
            bp_node_ids += [root]

    # Get a list of childs of each branch point
    bp_childs = {t: [e[0] for e in y.graph.in_edges(t)] for t in bp_node_ids}
    childs = [tn for l in bp_childs.values() for tn in l]

    # Get number of pre/postsynapses distal to each branch's childs
    distal = graph.distal_to(y,
                             np.append(pre_node_ids, post_node_ids),
                             childs)

    # Since nodes can have multiple pre-/postsynapses but they show up only
    # once in distal, we have to reindex to reflect the correct
    # number of synapes
    distal_pre = distal.loc[pre_node_ids]
    distal_post = distal.loc[post_node_ids]

    # Sum up columns: now each row represents the number of pre/postsynapses
    # distal to that node
    distal_pre_sum = distal_pre.sum(axis=0)
    distal_post_sum = distal_post.sum(axis=0)

    # Now go over all branch points and check flow between branches
    # (centrifugal) vs flow from branches to root (centripetal)
    flow = {bp: 0 for bp in bp_childs}
    for bp in bp_childs:
        # We will use left/right to label the different branches here
        # (even if there is more than two)
        for left, right in itertools.permutations(bp_childs[bp], r=2):
            flow[bp] += distal_post_sum.loc[left] * distal_pre_sum.loc[right]

    # At this point there are only flows for the childs of branch points.
    # Let's complete that mapping by adding flow for the nodes
    # between branch points.
    for s in x.small_segments:
        # Segments' orientation goes from distal -> proximal
        # Drop first (distal) node if it is not a leaf
        if s[0] in flow:
            s = s[1:]

        # Update remaining nodes with the flow of the first child
        this_flow = flow.get(s[-1], 0)
        flow.update({n: this_flow for n in s})

    # Set flow centrality to None for all nodes
    x.nodes['bending_flow'] = x.nodes.node_id.map(flow)

    return x


@utils.map_neuronlist(desc='Calc. flow', allow_parallel=True)
@utils.meshneuron_skeleton(method='node_properties',
                           include_connectors=True,
                           heal=True,
                           node_props=['flow_centrality'])
def flow_centrality(x: 'core.NeuronObject',
                    mode: Union[Literal['centrifugal'],
                                Literal['centripetal'],
                                Literal['sum']] = 'sum'
                    ) -> 'core.NeuronObject':
    """Calculate synapse flow centrality (SFC).

    From Schneider-Mizell et al. (2016): "We use flow centrality for
    four purposes. First, to split an arbor into axon and dendrite at the
    maximum centrifugal SFC, which is a preliminary step for computing the
    segregation index, for expressing all kinds of connectivity edges (e.g.
    axo-axonic, dendro-dendritic) in the wiring diagram, or for rendering the
    arbor in 3d with differently colored regions. Second, to quantitatively
    estimate the cable distance between the axon terminals and dendritic arbor
    by measuring the amount of cable with the maximum centrifugal SFC value.
    Third, to measure the cable length of the main dendritic shafts using
    centripetal SFC, which applies only to insect neurons with at least one
    output synapse in their dendritic arbor. And fourth, to weigh the color
    of each skeleton node in a 3d view, providing a characteristic signature of
    the arbor that enables subjective evaluation of its identity."

    Losely based on Alex Bates' implemention in `catnat
    <https://github.com/alexanderbates/catnat>`_.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | NeuronList
                Neuron(s) to calculate flow centrality for. Must have
                connectors!
    mode :      'centrifugal' | 'centripetal' | 'sum', optional
                Type of flow centrality to calculate. There are three flavors::
                (1) centrifugal counts paths from proximal inputs to distal outputs
                (2) centripetal counts paths from distal inputs to proximal outputs
                (3) the sum of both - this is the original implementation

    See Also
    --------
    :func:`~navis.bending_flow`
            Variation of flow centrality: calculates bending flow.
    :func:`~navis.arbor_segregation_index`
            By-arbor segregation index.
    :func:`~navis.segregation_index`
            Calculates segregation score (polarity) of a neuron.
    :func:`~navis.split_axon_dendrite`
            Tries splitting a neuron into axon and dendrite.

    Returns
    -------
    neuron
                Adds "flow_centrality" as column in the node table (for
                TreeNeurons) or as `.flow_centrality` property
                (for MeshNeurons).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(2)
    >>> n.reroot(n.soma, inplace=True)
    >>> _ = navis.flow_centrality(n)
    >>> n[0].nodes.flow_centrality.max()
    786341

    """
    if mode not in ['centrifugal', 'centripetal', 'sum']:
        raise ValueError(f'Unknown "mode" parameter: {mode}')

    if not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Expected TreeNeuron(s), got "{type(x)}"')

    if not x.has_connectors:
        raise ValueError('Neuron must have connectors.')

    if np.any(x.soma) and not np.all(np.isin(x.soma, x.root)):
        logger.warning(f'Neuron {x.id} is not rooted to its soma!')

    # Figure out how connector types are labeled
    cn_types = x.connectors.type.unique()
    if any(np.isin(['pre', 'post'], cn_types)):
        pre, post = 'pre', 'post'
    elif any(np.isin([0, 1], cn_types)):
        pre, post = 0, 1
    else:
        raise ValueError(f'Unable to parse connector types "{cn_types}" for neuron {x.id}')

    # Get list of nodes with pre/postsynapses
    pre_node_ids = x.connectors[x.connectors.type == pre].node_id.values
    post_node_ids = x.connectors[x.connectors.type == post].node_id.values
    total_post = len(post_node_ids)
    total_pre = len(pre_node_ids)

    # Get list of points to calculate flow centrality for:
    # branches and and their children
    is_bp = x.nodes['type'] == 'branch'
    is_cn = x.nodes.node_id.isin(x.connectors.node_id)
    calc_node_ids = x.nodes[is_bp | is_cn].node_id.values

    # We will be processing a super downsampled version of the neuron to
    # speed up calculations
    current_level = logger.level
    current_state = config.pbar_hide
    logger.setLevel('ERROR')
    config.pbar_hide = True
    y = sampling.downsample_neuron(x=x,
                                   downsampling_factor=float('inf'),
                                   inplace=False,
                                   preserve_nodes=calc_node_ids)
    logger.setLevel(current_level)
    config.pbar_hide = current_state

    # Get number of pre/postsynapses distal to each branch's childs
    distal = graph.distal_to(y,
                             np.append(pre_node_ids, post_node_ids),
                             calc_node_ids)

    # Since nodes can have multiple pre-/postsynapses but they show up only
    # once in distal, we have to reindex to reflect the correct number of synapes
    distal_pre = distal.loc[pre_node_ids]
    distal_post = distal.loc[post_node_ids]

    # Sum up axis - now each row represents the number of pre/postsynapses
    # that are distal to that node
    distal_pre = distal_pre.sum(axis=0)
    distal_post = distal_post.sum(axis=0)

    if mode != 'centripetal':
        # Centrifugal is the flow from all proximal postsynapses to all
        # distal presynapses
        centrifugal = {n: (total_post - distal_post[n]) * distal_pre[n] for n in calc_node_ids}

    if mode != 'centrifugal':
        # Centripetal is the flow from all distal postsynapses to all
        # non-distal presynapses
        centripetal = {n: distal_post[n] * (total_pre - distal_pre[n]) for n in calc_node_ids}

    # Now map this onto our neuron
    if mode == 'centrifugal':
        flow = centrifugal
    elif mode == 'centripetal':
        flow = centripetal
    elif mode == 'sum':
        flow = {n: centrifugal[n] + centripetal[n] for n in centrifugal}

    # At this point there is only flow for branch points and
    # their childs. Let's complete that mapping by adding flow
    # for the nodes between branch points.
    for s in x.small_segments:
        # Segments' orientation goes from distal -> proximal
        # Each non-terminal segment will have its first node mapped
        flow.update({n: flow.get(s[0], 0) for n in s[:-1]})

    x.nodes['flow_centrality'] = x.nodes.node_id.map(lambda x: flow.get(x, 0))

    # Add info on method/mode used for flow centrality
    x.centrality_method = mode  # type: ignore

    return x


def tortuosity(x: 'core.NeuronObject',
               seg_length: Union[int, float, str,
                                 Sequence[Union[int, float, str]]] = 10
               ) -> Union[float,
                          Sequence[float],
                          pd.DataFrame]:
    """Calculate tortuosity of a neuron.

    See Stepanyants et al., Neuron (2004) for detailed explanation. Briefly,
    tortuosity index `T` is defined as the ratio of the branch segment length
    `L` (``seg_length``) to the Euclidian distance `R` between its ends.

    The way this is implemented in `navis`:
     1. Each linear stretch (i.e. between branch points or branch points to a
        leaf node) is divided into segments of exactly ``seg_length``
        geodesic length. Any remainder is skipped.
     2. For each of these segments we divide its geodesic length `L`
        (i.e. `seg_length`) by the Eucledian distance `R` between its start and
        its end.
     3. The final tortuosity is the mean of `L / R` across all segments.

    Note
    ----
    If you want to make sure that segments are as close to length `L` as
    possible, consider resampling the neuron using :func:`navis.resample_skeleton`.

    Parameters
    ----------
    x :                 TreeNeuron | MeshNeuron | NeuronList
                        Neuron to analyze. If MeshNeuron, will generate and
                        use a skeleton representation.
    seg_length :        int | float | str | list thereof, optional
                        Target segment length(s) `L`. If neuron(s) have their
                        ``.units`` set, you can also pass a string such as
                        "1 micron". ``seg_length`` must be larger than the
                        current sampling resolution of the neuron.

    Returns
    -------
    tortuosity :        float | np.array | pandas.DataFrame
                        If x is NeuronList, will return DataFrame.
                        If x is single TreeNeuron, will return either a
                        single float (if single seg_length is queried) or a
                        DataFrame (if multiple seg_lengths are queried).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> # Calculate tortuosity with 1 micron seg lengths
    >>> T = navis.tortuosity(n, seg_length='1 micron')
    >>> round(T, 3)
    1.054

    """
    if isinstance(x, core.NeuronList):
        if not isinstance(seg_length, (list, np.ndarray, tuple)):
            seg_length = [seg_length]  # type: ignore
        df = pd.DataFrame([tortuosity(n,
                                      seg_length=seg_length) for n in config.tqdm(x,
                                                                                  desc='Tortuosity',
                                                                                  disable=config.pbar_hide,
                                                                                  leave=config.pbar_leave)],
                          index=x.id, columns=seg_length).T
        df.index.name = 'seg_length'
        return df

    if isinstance(x, core.MeshNeuron):
        x = x.skeleton
    elif not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron(s), got {type(x)}')

    if isinstance(seg_length, (list, np.ndarray)):
        return [tortuosity(x, l) for l in seg_length]

    # From here on out seg length is single value
    seg_length: float = x.map_units(seg_length, on_error='raise')

    if seg_length <= 0:
        raise ValueError('`seg_length` must be > 0.')
    res = x.sampling_resolution
    if seg_length <= res:
        raise ValueError('`seg_length` must not be smaller than the sampling '
                         f'resolution of the neuron ({res:.2f}).')

    # Iterate over segments
    locs = x.nodes.set_index('node_id')[['x', 'y', 'z']].astype(float)
    T_all = []
    for i, seg in enumerate(x.small_segments):
        # Get coordinates
        coords = locs.loc[seg].values

        # Vecs between subsequently measured points
        vecs = np.diff(coords.T)

        # path: cum distance along points (norm from first to Nth point)
        dist = np.cumsum(np.linalg.norm(vecs, axis=0))
        dist = np.insert(dist, 0, 0)

        # Skip if segment too short
        if dist[-1] <= seg_length:
            continue

        # New partitions
        new_dist = np.arange(0, dist[-1], seg_length)

        try:
            sampleX = scipy.interpolate.interp1d(dist, coords[:, 0],
                                                 kind='linear')
            sampleY = scipy.interpolate.interp1d(dist, coords[:, 1],
                                                 kind='linear')
            sampleZ = scipy.interpolate.interp1d(dist, coords[:, 2],
                                                 kind='linear')
        except ValueError:
            continue

        # Sample each dim
        xnew = sampleX(new_dist)
        ynew = sampleY(new_dist)
        znew = sampleZ(new_dist)

        # We know that each child -> parent pair was originally exactly
        # `seg_length` geodesic distance apart. Now we need to find out
        # how far they are apart in Eucledian distance
        new_coords = np.array([xnew, ynew, znew]).T

        R = np.linalg.norm(new_coords[:-1] - new_coords[1:], axis=1)
        T = seg_length / R
        T_all = np.append(T_all, T)

    return T_all.mean()


@utils.map_neuronlist(desc='Sholl analysis', allow_parallel=True)
def sholl_analysis(x: 'core.NeuronObject',
                   radii: Union[int, list] = 10,
                   center: Union[Literal['root'],
                                 Literal['soma'],
                                 list,
                                 int] = 'centermass',
                   geodesic=False,
                  ) -> Union[float,
                             Sequence[float],
                             pd.DataFrame]:
    """Run Sholl analysis for given neuron(s).

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | NeuronList
                Neuron to analyze. If MeshNeuron, will generate and
                use a skeleton representation.
    radii :     int | list-like
                If integer, will produce N evenly space radii covering the
                distance between the center and the most distal node.
                Alternatively, you can also provide a list of radii to check.
                If `x` is multiple neurons, must provide a list of ``radii``!
    center :    "centermass" | "root" | "soma" | int | list-like
                The center to use for Sholl analysis:
                    - "centermass" (default) uses the mean across nodes positions
                    - "root" uses the current root of the skeleton
                    - "soma" uses the neuron's soma (will raise error if no soma)
                    - int is interpreted as a node ID
                    - (3, ) list-like is interpreted as x/y/z coordinate
    geodesic :  bool
                If True, will use geodesic (along-the-arbor) instead of
                Euclidean distances. This does not work if center is an x/y/z
                coordinate.

    Returns
    -------
    results :   pd.DataFrame
                Results contain, for each spherical bin, the number of
                intersections, cable length and number of branch points.

    References
    ----------
    See the `Wikipedia <https://en.wikipedia.org/wiki/Sholl_analysis>`_ entry
    for a brief explanation.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> # Sholl analysis
    >>> sha = navis.sholl_analysis(n, radii=100, center='root')
    >>> # Plot distributions
    >>> ax = sha.plot()                                         # doctest: +SKIP
    >>> # Sholl analysis but using geodesic distance
    >>> sha = navis.sholl_analysis(n, radii=100, center='root', geodesic=True)

    """
    # Use MeshNeuron's skeleton
    if isinstance(x, core.MeshNeuron):
        x = x.skeleton

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron or MeshNeuron(s), got {type(x)}')

    if geodesic and len(x.root) > 1:
        raise ValueError('Unable to use `geodesic=True` with fragmented '
                         'neurons. Use `navis.heal_fragmented_neuron` first.')

    if center == 'soma' and not x.has_soma:
        raise ValueError(f'Neuron {x.id} has no soma.')
    elif utils.is_iterable(center):
        center = np.asarray(center)
        if center.ndim != 1 or len(center) != 3:
            raise ValueError('`center` must be (3, ) list-like when providing '
                            f'a coordinate. Got {center.shape}')
        if geodesic:
            raise ValueError('Must not provide a `center` as coordinate when '
                             'geodesic=True')
    elif center == 'root' and len(x.root) > 1:
        raise ValueError(f'Neuron {x.id} has multiple roots. Please specify '
                         'which node/coordinate to use as center.')

    if center == 'centermass':
        center = x.nodes[['x', 'y', 'z']].mean(axis=0).values

    # Calculate distances for each node
    nodes = x.nodes.set_index('node_id').copy()
    if not geodesic:
        if isinstance(center, int):
            if center not in nodes.index.values:
                raise ValueError(f'{center} is not a valid node ID.')

            center = nodes.loc[center, ['x', 'y', 'z']].values
        elif center == 'soma':
            center = nodes.loc[utils.make_iterable(x.soma)[0], ['x', 'y', 'z']].values
        elif center == 'root':
            center = nodes.loc[utils.make_iterable(x.root)[0], ['x', 'y', 'z']].values
        center = center.astype(float)

        nodes['dist'] = np.sqrt(((x.nodes[['x', 'y', 'z']].values - center)**2).sum(axis=1))
    else:
        if center == 'soma':
            center = x.soma[0]
        elif center == 'root':
            center = x.root[0]

        nodes['dist'] = graph.geodesic_matrix(x, from_=center)[x.nodes.node_id.values].values[0]

    not_root = nodes.parent_id >= 0
    dists = nodes.loc[not_root, 'dist'].values
    pdists = nodes.loc[nodes[not_root].parent_id.values, 'dist'].values
    le = parent_dist(x)[not_root]
    ty = nodes.loc[not_root, 'type'].values

    # Generate radii for the Sholl spheres
    if isinstance(radii, int):
        radii = np.linspace(0, dists.max(), radii + 1)
    else:
        if radii[0] != 0:
            radii = np.insert(radii, 0, 0)

    data = []
    for i in range(1, len(radii)):
        # Find the number of crossings
        crossings = ((dists <= radii[i]) & (pdists > radii[i])).sum()

        # Get the (approximate) cable length in this sphere
        this_sphere = (dists > radii[i - 1]) & (dists < radii[i])
        cable = le[this_sphere].sum()

        # The number of branch points in this sphere
        n_branchpoints = (ty[this_sphere] == 'branch').sum()

        data.append([radii[i], crossings, cable, n_branchpoints])

    return pd.DataFrame(data,
                        columns=['radius', 'intersections',
                                 'cable_length', 'branch_points']).set_index('radius')
