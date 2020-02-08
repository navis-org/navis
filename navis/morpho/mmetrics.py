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

import math
import itertools

import pandas as pd
import numpy as np

from typing import Union, Optional, Sequence, List, Dict, overload
from typing_extensions import Literal

from .. import config, graph, sampling, core
from .manipulation import split_axon_dendrite

# Set up logging
logger = config.logger

__all__ = sorted(['strahler_index', 'bending_flow',
                  'flow_centrality', 'segregation_index', 'tortuosity'])


def parent_dist(x: Union['core.TreeNeuron', pd.DataFrame],
                root_dist: Optional[int] = None) -> None:
    """Add ``parent_dist`` [nm] column to the treenode table.

    Parameters
    ----------
    x :         TreeNeuron | treenode table
    root_dist : int | None
                ``parent_dist`` for the root's row. Set to ``None``, to leave
                at ``NaN`` or e.g. to ``0`` to set to 0.

    Returns
    -------
    Nothing

    """

    if isinstance(x, core.TreeNeuron):
        nodes = x.nodes
    elif isinstance(x, pd.DataFrame):
        nodes = x
    else:
        raise TypeError('Need TreeNeuron or DataFrame, got "{}"'.format(type(x)))

    # Calculate distance to parent for each node
    not_root = nodes[nodes.parent_id >= 0]
    tn_coords = not_root[['x', 'y', 'z']].values

    # Ready treenode table to be indexes by node_id
    this_tn = nodes.set_index('node_id', inplace=False)
    parent_coords = this_tn.loc[not_root.parent_id.values,
                                ['x', 'y', 'z']].values

    # Calculate distances between nodes and their parents
    w = np.sqrt(np.sum((tn_coords - parent_coords) ** 2, axis=1))

    nodes['parent_dist'] = root_dist
    nodes.loc[nodes.parent_id >= 0, 'parent_dist'] = w

    return None


@overload
def strahler_index(x: 'core.NeuronObject',
                   inplace: Literal[False],
                   method: Union[Literal['standard'],
                                 Literal['greedy']] = 'standard',
                   to_ignore: list = [],
                   min_twig_size: Optional[int] = None
                   ) -> 'core.NeuronObject':
    pass


@overload
def strahler_index(x: 'core.NeuronObject',
                   inplace: Literal[True],
                   method: Union[Literal['standard'],
                                 Literal['greedy']] = 'standard',
                   to_ignore: list = [],
                   min_twig_size: Optional[int] = None
                   ) -> None:
    pass


def strahler_index(x: 'core.NeuronObject',
                   inplace: bool = True,
                   method: Union[Literal['standard'],
                                 Literal['greedy']] = 'standard',
                   to_ignore: list = [],
                   min_twig_size: Optional[int] = None
                   ) -> Optional['core.NeuronObject']:
    """ Calculates Strahler Index (SI).

    Starts with SI of 1 at each leaf and walks to root. At forks with different
    incoming SIs, the highest index is continued. At forks with the same
    incoming SI, highest index + 1 is continued.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
    inplace :           bool, optional
                        If False, a copy of original skdata is returned.
    method :            'standard' | 'greedy', optional
                        Method used to calculate strahler indices: 'standard'
                        will use the method described above; 'greedy' will
                        always increase the index at converging branches
                        whether these branches have the same index or not.
                        This is useful e.g. if you want to cut the neuron at
                        the first branch point.
    to_ignore :         iterable, optional
                        List of node IDs to ignore. Must be the FIRST node
                        of the branch. Excluded branches will not contribute
                        to Strahler index calculations and instead be assigned
                        the SI of their parent branch.
    min_twig_size :     int, optional
                        If provided, will ignore terminal (!) twigs with
                        fewer nodes. Instead, they will be assigned the SI of
                        their parent branch.

    Returns
    -------
    if ``inplace=False``
                        Returns nothing but adds new column ``strahler_index``
                        to neuron.nodes.
    if ``inplace=True``
                        Returns copy of original neuron with new column
                        ``strahler_index``.

    """

    if isinstance(x, core.NeuronList):
        if x.shape[0] == 1:
            x = x[0]
        else:
            res = []
            for n in config.tqdm(x):
                res.append(strahler_index(n, inplace=inplace, method=method))  # type: ignore

            if not inplace:
                return core.NeuronList(res)
            else:
                return None

    if not inplace:
        x = x.copy()

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

    # Reindex according to node_id
    this_tn = x.nodes.set_index('node_id', inplace=False)

    # Do NOT name anything strahler_index - this overwrites the function!
    SI: Dict[int, int] = {}

    starting_points = end_nodes

    nodes_processed = []

    while starting_points:
        logger.debug(f'New starting point. Remaining: {len(starting_points)}')
        new_starting_points = []
        starting_points_done = []

        for i, this_node in enumerate(starting_points):
            logger.debug(f'{i} of {len(starting_points)}')

            # Calculate index for this branch
            previous_indices = []
            for child in list_of_childs[this_node]:
                if SI.get(child, None):
                    previous_indices.append(SI[child])

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
            elif previous_indices.count(max(previous_indices)) >= 2 or method == 'greedy':
                this_branch_index = max(previous_indices) + 1
            # If just a branch point: continue max SI
            else:
                this_branch_index = max(previous_indices)

            nodes_processed.append(this_node)
            starting_points_done.append(this_node)

            # Now walk down this spine
            # Find parent
            spine = [this_node]

            parent_node = this_tn.loc[this_node, 'parent_id']

            while parent_node not in branch_nodes and parent_node is not None:
                this_node = parent_node
                parent_node = None

                spine.append(this_node)
                nodes_processed.append(this_node)

                # Find next parent
                try:
                    parent_node = this_tn.loc[this_node, 'parent_id']
                except BaseException:
                    # Will fail if at root (no parent)
                    break

            SI.update({n: this_branch_index for n in spine})

            # The last this_node is either a branch node or the root
            # If a branch point: check, if all its childs have already been
            # processed
            if parent_node is not None:
                node_ready = True
                for child in list_of_childs[parent_node]:
                    if child not in nodes_processed:
                        node_ready = False

                if node_ready is True and parent_node is not None:
                    new_starting_points.append(parent_node)

        # Remove those starting_points that were successfully processed in this
        # run before the next iteration
        for node in starting_points_done:
            starting_points.remove(node)

        # Add new starting points
        starting_points = starting_points | set(new_starting_points)

    # Disconnected single nodes (e.g. after pruning) will end up w/o an entry
    # --> we will give them an SI of 1
    x.nodes['strahler_index'] = [SI.get(n, 1)
                                 for n in x.nodes.node_id.values]

    # Fix branches that were ignored
    if to_ignore:
        this_tn = x.nodes.set_index('node_id', inplace=False)
        # Go over all terminal branches with the tag
        for tn in x.nodes[(x.nodes.type == 'end') & (x.nodes.node_id.isin(to_ignore))].node_id.values:
            # Get this terminal's segment
            this_seg = [s for s in x.small_segments if s[0] == tn][0]
            # Get strahler index of parent branch
            new_SI = this_tn.loc[this_seg[-1]].strahler_index
            # Set these nodes strahler index to that of the last branch point
            x.nodes.loc[x.nodes.node_id.isin(this_seg), 'strahler_index'] = new_SI

    if not inplace:
        return x

    return None


def segregation_index(x: 'core.NeuronObject',
                      centrality_method: Union[Literal['centrifugal'],
                                               Literal['centripetal'],
                                               Literal['bending'],
                                               Literal['sum']] = 'centrifugal',
                      ) -> float:
    """ Calculates segregation index (SI).

    The segregation index as established by Schneider-Mizell et al. (eLife,
    2016) is a measure for how polarized a neuron is. SI of 1 indicates total
    segregation of inputs and outputs into dendrites and axon, respectively.
    SI of 0 indicates homogeneous distribution.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
                        Neuron to calculate segregation index (SI). If a
                        NeuronList is provided, will assume that it contains
                        fragments (e.g. from axon/ dendrite splits) of a
                        single neuron.
    centrality_method : 'centrifugal' | 'centripetal' | 'sum' | 'bending'
                        Type of flow centrality to use to split into axon +
                        dendrite of ``x`` is only a single neuron.
                        There are four flavors:
                            - for the first three, see :func:`~navis.flow_centrality`
                            - for `bending`, see :func:`~navis.bending_flow`

                        Will try using stored centrality, if possible.

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

    if not isinstance(x, (core.TreeNeuron, core.NeuronList)):
        raise ValueError('Must pass TreeNeuron or NeuronList, '
                         'not {0}'.format(type(x)))

    if isinstance(x, core.NeuronList) and x.shape[0] == 1:
        x = x[0]

    if not isinstance(x, core.NeuronList):
        # Get the branch point with highest flow centrality
        split_point = split_axon_dendrite(
            x, reroot_soma=True, return_point=True)

        # Now make a virtual split (downsampled neuron to speed things up)
        temp = x.copy()
        temp.downsample(factor=float('inf'), inplace=True)

        # Get one of its children
        child = temp.nodes[temp.nodes.parent_id == split_point].node_id.values[0]

        # This will leave the proximal split with the primary neurite but
        # since that should not have synapses, we don't care at this point.
        x = core.NeuronList(list(graph.cut_neuron(temp, child)))

    # Calculate entropy for each fragment
    entropy = []
    for n in x:
        p = n.n_postsynapses / n.n_connectors

        if 0 < p < 1:
            S = - (p * math.log(p) + (1 - p) * math.log(1 - p))
        else:
            S = 0

        entropy.append(S)

    # Calc entropy between fragments
    S = 1 / sum(x.n_connectors) * \
        sum([e * x[i].n_connectors for i, e in enumerate(entropy)])

    # Normalize to entropy in whole neuron
    p_norm = sum(x.n_postsynapses) / sum(x.n_connectors)
    if 0 < p_norm < 1:
        S_norm = - (p_norm * math.log(p_norm) +
                    (1 - p_norm) * math.log(1 - p_norm))
        H = 1 - S / S_norm
    else:
        S_norm = 0
        H = 0

    return H


def bending_flow(x: 'core.NeuronObject') -> None:
    """Variation of the algorithm for calculating synapse flow from
    Schneider-Mizell et al. (eLife, 2016).

    The way this implementation works is by iterating over each branch point
    and counting the number of pre->post synapse paths that "flow" from one
    child branch to the other(s).

    Parameters
    ----------
    x :         TreeNeuron | NeuronList
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
    :func:`~navis.segregation_score`
            Uses flow centrality to calculate segregation score (polarity).
    :func:`~navis.split_axon_dendrite`
            Split the neuron into axon, dendrite and primary neurite.

    Returns
    -------
    Adds a new column ``'flow_centrality'`` to the nodes table (branch points
    only).

    """
    if not isinstance(x, (core.TreeNeuron, core.NeuronList)):
        raise ValueError(f'Expected TreeNeuron or NeuronList, got "{type(x)}"')

    if isinstance(x, core.NeuronList):
        [bending_flow(n) for n in x]  # type: ignore
        return

    if not x.has_connectors:
        raise ValueError('Neuron must have connectors.')

    if x.soma and x.soma not in x.root:
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

    # Get list of childs of each branch point
    bp_childs = {t: [e[0] for e in y.graph.in_edges(t)] for t in bp_node_ids}
    childs = [tn for l in bp_childs.values() for tn in l]

    # Get number of pre/postsynapses distal to each branch's childs
    distal_pre = graph.distal_to(y, pre_node_ids, childs)
    distal_post = graph.distal_to(y, post_node_ids, childs)

    # Sum up axis - now each row represents the number of pre/postsynapses
    # distal to that node
    distal_pre_sum = distal_pre.T.sum(axis=1)
    distal_post_sum = distal_post.T.sum(axis=1)

    # Now go over all branch points and check flow between branches
    # (centrifugal) vs flow from branches to root (centripetal)
    flow = {bp: 0 for bp in bp_childs}
    for bp in bp_childs:
        # We will use left/right to label the different branches here
        # (even if there is more than two)
        for left, right in itertools.permutations(bp_childs[bp], r=2):
            flow[bp] += distal_post_sum.loc[left] * distal_pre_sum.loc[right]

    # Set flow centrality to None for all nodes
    x.nodes['flow_centrality'] = None

    # Change index to node_id
    x.nodes.set_index('node_id', inplace=True)

    # Add flow (make sure we use igraph of y to get node ids!)
    x.nodes.loc[list(flow.keys()), 'flow_centrality'] = list(flow.values())

    # Add little info on method used for flow centrality
    x.centrality_method = 'bending'  # type: ignore

    x.nodes.reset_index(inplace=True)

    return None


def flow_centrality(x: 'core.NeuronObject',
                    mode: Union[Literal['centrifugal'],
                                Literal['centripetal'],
                                Literal['sum']] = 'centrifugal'
                    ) -> None:
    """ Calculates synapse flow centrality (SFC).

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
    output syn- apse in their dendritic arbor. And fourth, to weigh the color
    of each skeleton node in a 3d view, providing a characteristic signature of
    the arbor that enables subjective evaluation of its identity."

    Losely based on Alex Bate's implemention in `catnat
    <https://github.com/alexanderbates/catnat>`_.

    Catmaid uses the equivalent of ``mode='sum'`` and ``polypre=True``.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList
                Neuron(s) to calculate flow centrality for. Must have
                connectors!
    mode :      'centrifugal' | 'centripetal' | 'sum', optional
                Type of flow centrality to calculate. There are three flavors::
                (1) centrifugal, counts paths from proximal inputs to distal outputs
                (2) centripetal, counts paths from distal inputs to proximal outputs
                (3) the sum of both

    See Also
    --------
    :func:`~navis.bending_flow`
            Variation of flow centrality: calculates bending flow.
    :func:`~navis.segregation_index`
            Calculates segregation score (polarity) of a neuron
    :func:`~navis.flow_centrality_split`
            Tries splitting a neuron into axon, dendrite and primary neurite.


    Returns
    -------
    Adds a new column 'flow_centrality' to nodes table (only processes
    branchpoints and synapse-holding nodes).

    """

    if mode not in ['centrifugal', 'centripetal', 'sum']:
        raise ValueError('Unknown parameter for mode: {0}'.format(mode))

    if not isinstance(x, (core.TreeNeuron, core.NeuronList)):
        raise ValueError(f'Expected TreeNeuron or NeuronList, got "{type(x)}"')

    if not x.has_connectors:
        raise ValueError('Neuron must have connectors.')

    if isinstance(x, core.NeuronList):
        _ = [flow_centrality(n, mode=mode) for n in x]  # type: ignore  # does not like assigment
        return

    if x.soma and x.soma not in x.root:
        logger.warning(f'Neuron {x.id} is not rooted to its soma!')

    # We will be processing a super downsampled version of the neuron to
    # speed up calculations
    current_level = logger.level
    current_state = config.pbar_hide
    logger.setLevel('ERROR')
    config.pbar_hide = True
    y = sampling.downsample_neuron(x=x,
                                   downsampling_factor=float('inf'),
                                   inplace=False,
                                   preserve_nodes=x.connectors.node_id.values)
    logger.setLevel(current_level)
    config.pbar_hide = current_state

    # Figure out how connector types are labeled
    cn_types = y.connectors.type.unique()
    if all(np.isin(['pre', 'post'], cn_types)):
        pre, post = 'pre', 'post'
    elif all(np.isin([0, 1], cn_types)):
        pre, post = 0, 1
    else:
        raise ValueError(f'Unable to parse connector types "{cn_types} for neuron {y.id}')

    # Get list of nodes with pre/postsynapses
    pre_node_ids = y.connectors[y.connectors.type == pre].node_id.unique()
    post_node_ids = y.connectors[y.connectors.type == post].node_id.unique()
    total_post = len(post_node_ids)

    # Get list of points to calculate flow centrality for:
    # branches and nodes with synapses
    is_bp = y.nodes.type == 'branch'
    is_cn = y.nodes.node_id.isin(y.connectors.node_id)
    calc_node_ids = y.nodes[is_bp | is_cn].node_id.values

    # Get number of pre/postsynapses distal to each branch's childs
    distal_pre = graph.distal_to(y, pre_node_ids, calc_node_ids)
    distal_post = graph.distal_to(y, post_node_ids, calc_node_ids)

    # Sum up axis - now each row represents the number of pre/postsynapses
    # that are distal to that node
    distal_pre = distal_pre.T.sum(axis=1)
    distal_post = distal_post.T.sum(axis=1)

    if mode != 'centripetal':
        # Centrifugal is the flow from all non-distal postsynapses to all
        # distal presynapses
        centrifugal = {n: (total_post - distal_post[n]) * distal_pre[n] for n in calc_node_ids}

    if mode != 'centrifugal':
        # Centripetal is the flow from all distal postsynapses to all
        # non-distal presynapses
        centripetal = {n: distal_post[n] * (total_post - distal_pre[n]) for n in calc_node_ids}

    # Now map this onto our neuron
    x.nodes['flow_centrality'] = None
    if mode == 'centrifugal':
        x.nodes['flow_centrality'] = x.nodes.node_id.map(centrifugal)
    elif mode == 'centripetal':
        x.nodes['flow_centrality'] = x.nodes.node_id.map(centripetal)
    elif mode == 'sum':
        combined = {n: centrifugal[n] + centripetal[n] for n in centrifugal}
        x.nodes['flow_centrality'] = x.nodes.node_id.map(combined)

    # Add info on method/mode used for flow centrality
    x.centrality_method = mode  # type: ignore

    return None


def tortuosity(x: 'core.NeuronObject',
               seg_length: Union[int, float, Sequence[Union[int, float]]] = 10,
               skip_remainder: bool = False
               ) -> Union[float,
                          Sequence[float],
                          pd.DataFrame]:
    """ Calculates tortuosity for a neurons.

    See Stepanyants et al., Neuron (2004) for detailed explanation. Briefly,
    tortuosity index `T` is defined as the ratio of the branch segment length
    `L` (``seg_length``) to the eucledian distance `R` between its ends.

    Note
    ----
    If you want to make sure that segments are as close to length `L` as
    possible, consider resampling the neuron using :func:`navis.resample`.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
    seg_length :        int | float | list, optional
                        Target segment length(s) L. Will try resampling neuron
                        to this resolution. Please note that the final segment
                        length is restricted by the neuron's original
                        resolution.
    skip_remainder :    bool, optional
                        Segments can turn out to be smaller than desired if a
                        branch point or end point is hit before `seg_length`
                        is reached. If ``skip_remainder`` is True, these will
                        be ignored.

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
    >>> T = navis.tortuosity(n, seg_length=1e3)
    >>> round(T, 3)
    1.163

    """

    # TODO:
    # - try as angles between dotproduct vectors
    #

    if isinstance(x, core.NeuronList):
        if not isinstance(seg_length, (list, np.ndarray, tuple)):
            seg_length = [seg_length]  # type: ignore
        df = pd.DataFrame([tortuosity(n, seg_length) for n in config.tqdm(x, desc='Tortuosity', disable=config.pbar_hide, leave=config.pbar_leave)],
                          index=x.id, columns=seg_length).T
        df.index.name = 'seg_length'
        return df

    if not isinstance(x, core.TreeNeuron):
        raise TypeError('Need TreeNeuron, got {0}'.format(type(x)))

    if isinstance(seg_length, (list, np.ndarray)):
        return [tortuosity(x, l) for l in seg_length]  # type: ignore  # would need to overload to fix this

    # From here on out seg length is single value
    seg_length: float

    if seg_length <= 0:
        raise ValueError('Segment length must be >0.')

    # We will collect coordinates and do distance calculations later
    start_tn: List[int] = []
    end_tn: List[int] = []
    L: List[Union[int, float]] = []

    # Go over all segments
    for seg in x.small_segments:
        # Collect distances between treenodes (in microns)
        dist = np.array([x.graph.edges[(c, p)]['weight']
                         for c, p in zip(seg[:-1], seg[1:])])
        # Walk the segment, collect stretches of length `seg_length`
        cut_ix = [0]
        for i, tn in enumerate(seg):
            if sum(dist[cut_ix[-1]:i]) > seg_length:
                cut_ix.append(i)

        # If the last node is not a cut node
        if cut_ix[-1] < i and not skip_remainder:
            cut_ix.append(i)

        # Translate into treenode IDs
        if len(cut_ix) > 1:
            L += [sum(dist[s:e]) for s, e in zip(cut_ix[:-1], cut_ix[1:])]
            start_tn += [seg[n] for n in cut_ix[:-1]]
            end_tn += [seg[n] for n in cut_ix[1:]]

    # Now calculate euclidean distances
    tn_table = x.nodes.set_index('node_id', inplace=False)
    start_co = tn_table.loc[start_tn, ['x', 'y', 'z']].values
    end_co = tn_table.loc[end_tn, ['x', 'y', 'z']].values
    R = np.linalg.norm(start_co - end_co, axis=1)

    # Get tortousity
    T = np.array(L) / R

    return T.mean()
