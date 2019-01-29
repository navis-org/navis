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
import scipy.spatial.distance
import networkx as nx

from .. import core, graph, utils, config, sampling
from . import metrics

# Set up logging
logger = config.logger

__all__ = sorted(['prune_by_strahler', 'stitch_neurons', 'split_axon_dendrite',
                  'average_neurons', 'despike_neuron', 'guess_radius',
                  'smooth_neuron'])


def prune_by_strahler(x, to_prune, reroot_soma=True, inplace=False,
                      force_strahler_update=False, relocate_connectors=False):
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
    relocate_connectors : bool, optional
                          If True, connectors on removed treenodes will be
                          reconnected to the closest still existing treenode.
                          Works only in child->parent direction.

    Returns
    -------
    TreeNeuron/List
                    Pruned neuron(s).

    """

    if isinstance(x, core.TreeNeuron):
        neuron = x
    elif isinstance(x, core.NeuronList):
        temp = [prune_by_strahler(
            n, to_prune=to_prune, inplace=inplace) for n in x]
        if not inplace:
            return core.NeuronList(temp, x._remote_instance)
        else:
            return

    # Make a copy if necessary before making any changes
    if not inplace:
        neuron = neuron.copy()

    if reroot_soma and neuron.soma:
        neuron.reroot(neuron.soma)

    if 'strahler_index' not in neuron.nodes or force_strahler_update:
        strahler_index(neuron)

    # Prepare indices
    if isinstance(to_prune, int) and to_prune < 0:
        to_prune = range(1, neuron.nodes.strahler_index.max() + (to_prune + 1))

    if isinstance(to_prune, int):
        if to_prune < 1:
            raise ValueError('SI to prune must be positive. Please see help'
                             'for additional options.')
        to_prune = [to_prune]
    elif isinstance(to_prune, range):
        to_prune = list(to_prune)
    elif isinstance(to_prune, slice):
        SI_range = range(1, neuron.nodes.strahler_index.max() + 1)
        to_prune = list(SI_range)[to_prune]

    # Prepare parent dict if needed later
    if relocate_connectors:
        parent_dict = {
            tn.treenode_id: tn.parent_id for tn in neuron.nodes.itertuples()}

    neuron.nodes = neuron.nodes[
        ~neuron.nodes.strahler_index.isin(to_prune)].reset_index(drop=True)

    if not relocate_connectors:
        neuron.connectors = neuron.connectors[neuron.connectors.treenode_id.isin(
            neuron.nodes.treenode_id.values)].reset_index(drop=True)
    else:
        remaining_tns = set(neuron.nodes.treenode_id.values)
        for cn in neuron.connectors[~neuron.connectors.treenode_id.isin(neuron.nodes.treenode_id.values)].itertuples():
            this_tn = parent_dict[cn.treenode_id]
            while True:
                if this_tn in remaining_tns:
                    break
                this_tn = parent_dict[this_tn]
            neuron.connectors.loc[cn.Index, 'treenode_id'] = this_tn

    # Reset indices of node and connector tables (important for igraph!)
    neuron.nodes.reset_index(inplace=True, drop=True)
    neuron.connectors.reset_index(inplace=True, drop=True)

    # Theoretically we can end up with disconnected pieces, i.e. with more
    # than 1 root node -> we have to fix the nodes that lost their parents
    neuron.nodes.loc[~neuron.nodes.parent_id.isin(
        neuron.nodes.treenode_id.values), 'parent_id'] = None

    # Remove temporary attributes
    neuron._clear_temp_attr()

    if not inplace:
        return neuron
    else:
        return


def split_axon_dendrite(x, method='bending', primary_neurite=True,
                        reroot_soma=True, return_point=False):
    """ Split a neuron into axon, dendrite and primary neurite.

    The result is highly dependent on the method and on your neuron's
    morphology and works best for "typical" neurons, i.e. those where the
    primary neurite branches into axon and dendrites.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
                        Neuron(s) to split into axon, dendrite (and primary
                        neurite).
    method :            'centrifugal' | 'centripetal' | 'sum' | 'bending', optional
                        Type of flow centrality to use to split the neuron.
                        There are four flavors: the first three refer to
                        :func:`~pymaid.flow_centrality`, the last
                        refers to :func:`~pymaid.bending_flow`.

                        Will try using stored centrality, if possible.
    primary_neurite :   bool, optional
                        If True and the split point is at a branch point, will
                        try splittig into axon, dendrite and primary neurite.
                        Works only with ``method=bending``!
    reroot_soma :       bool, optional
                        If True, will make sure neuron is rooted to soma if at
                        all possible.
    return_point :      bool, optional
                        If True, will only return treenode ID of the node at
                        which to split the neuron.

    Returns
    -------
    NeuronList
                        Axon, dendrite and primary neurite.

    Examples
    --------
    >>> x = pymaid.get_neuron(123456)
    >>> split = pymaid.split_axon_dendrite(x, method='centrifugal',
    ...                                    reroot_soma=True)
    >>> split
    <class 'pymaid.NeuronList'> of 3 neurons
                          neuron_name skeleton_id  n_nodes  n_connectors
    0  neuron 123457_primary_neurite          16      148             0
    1             neuron 123457_axon          16     9682          1766
    2         neuron 123457_dendrite          16     2892           113
    >>> # For convenience, split_axon_dendrite assigns colors to the resulting
    >>> # fragments: axon = red, dendrites = blue, primary neurite = green
    >>> split.plot3d(color=split.color)

    """

    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]
    elif isinstance(x, core.NeuronList):
        nl = []
        for n in config.tqdm(x, desc='Splitting', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            nl.append(split_axon_dendrite(n,
                                          method=method,
                                          primary_neurite=primary_neurite,
                                          reroot_soma=reroot_soma,
                                          return_point=return_point))
        return core.NeuronList([n for l in nl for n in l])

    if not isinstance(x, core.TreeNeuron):
        raise TypeError('Can only process TreeNeuron, '
                        'got "{}"'.format(type(x)))

    if method not in ['centrifugal', 'centripetal', 'sum', 'bending']:
        raise ValueError('Unknown parameter for mode: {0}'.format(method))

    if primary_neurite and method != 'bending':
        logger.warning('Primary neurite splits only works well with '
                       'method "bending"')

    if x.soma and x.soma not in x.root and reroot_soma:
        x.reroot(x.soma)

    # Calculate flow centrality if necessary
    try:
        last_method = x.centrality_method
    except BaseException:
        last_method = None

    if last_method != method:
        if method == 'bending':
            _ = bending_flow(x)
        elif method in ['centripetal', 'centrifugal', 'sum']:
            _ = flow_centrality(x, mode=method)
        else:
            raise ValueError('Unknown method "{}"'.format(method))

    # Make copy, so that we don't screw things up
    x = x.copy()

    # Now get the node point with the highest flow centrality.
    cut = x.nodes[x.nodes.flow_centrality ==
                  x.nodes.flow_centrality.max()].treenode_id.values

    # If there is more than one point we need to get one closest to the soma
    # (root)
    if len(cut) > 1:
        cut = sorted(cut, key=lambda y: graph_utils.dist_between(
            x.graph, y, x.root[0]))[0]
    else:
        cut = cut[0]

    if return_point:
        return cut

    # If cut node is a branch point, we will try cutting off main neurite
    if x.graph.degree(cut) > 2 and primary_neurite:
        # First make sure that there are no other branch points with flow
        # between this one and the soma
        path_to_root = nx.shortest_path(x.graph, cut, x.root[0])

        # Get flow centrality along the path
        flows = x.nodes.set_index('treenode_id').loc[path_to_root]

        # Subset to those that are branches (exclude mere synapses)
        flows = flows[flows.type == 'branch']

        # Find the first branch point from the soma with no flow (fillna is
        # important!)
        last_with_flow = np.where(flows.flow_centrality.fillna(0).values > 0)[0][-1]

        if method != 'bending':
            last_with_flow += 1

        to_cut = flows.iloc[last_with_flow].name

        # Cut off primary neurite
        rest, primary_neurite = graph_utils.cut_neuron(x, to_cut)

        if method == 'bending':
            # The new cut node has to be a child of the original cut node
            cut = next(x.graph.predecessors(cut))

        # Change name and color
        primary_neurite.neuron_name = x.neuron_name + '_primary_neurite'
        primary_neurite.color = (0, 255, 0)
        primary_neurite.type = 'primary_neurite'
    else:
        rest = x
        primary_neurite = None

    # Next, cut the rest into axon and dendrite
    a, b = graph_utils.cut_neuron(rest, cut)

    # Figure out which one is which by comparing fraction of in- to outputs
    a_inout = a.n_postsynapses/a.n_presynapses if a.n_presynapses else float('inf')
    b_inout = b.n_postsynapses/b.n_presynapses if b.n_presynapses else float('inf')
    if a_inout > b_inout:
        dendrite, axon = a, b
    else:
        dendrite, axon = b, a

    axon.neuron_name = x.neuron_name + '_axon'
    dendrite.neuron_name = x.neuron_name + '_dendrite'

    axon.type = 'axon'
    dendrite.type = 'dendrite'

    # Change colors
    axon.color = (255, 0, 0)
    dendrite.color = (0, 0, 255)

    if primary_neurite:
        return core.NeuronList([primary_neurite, axon, dendrite])
    else:
        return core.NeuronList([axon, dendrite])


def stitch_neurons(*x, method='NONE', tn_to_stitch=None):
    """ Stitch multiple neurons together.

    The first neuron provided will be the master neuron. Unless node IDs
    are provided via ``tn_to_stitch``, neurons will be stitched at the
    closest point.

    Important
    ---------
    This will change node IDs!


    Parameters
    ----------
    x :                 Neuron | NeuronList | list of either
                        Neurons to stitch (see examples).
    method :            'LEAFS' | 'ALL' | 'NONE', optional
                        Set stitching method:
                            (1) 'LEAFS': Only leaf (including root) nodes will
                                be considered for stitching.
                            (2) 'ALL': All treenodes are considered.
                            (3) 'NONE': Node and connector tables will simply
                                be combined. Use this if your neurons consist
                                of fragments with multiple roots. Ignores
                                ``tn_to_stitch``.
    tn_to_stitch :      List of node IDs, optional
                        If provided, these nodes will be preferentially
                        used to stitch neurons together. Overrides methods
                        ``'ALL'`` or ``'LEAFS'``. If there are more
                        than two possible nodes for a single stitching
                        operation, the two closest are used.

    Returns
    -------
    core.TreeNeuron
                        Stitched neuron.

    Examples
    --------
    Stitching using a neuron list:

    >>> nl = pymaid.get_neuron('annotation:glomerulus DA1 right')
    >>> stitched = pymaid.stitch_neurons(nl, method='NONE')

    Stitching using individual neurons:

    >>> a = pymaid.get_neuron(16)
    >>> b = pymaid.get_neuron(2863104)
    >>> stitched = pymaid.stitch_neurons(a, b, method='NONE')
    >>> # Or alternatively:
    >>> stitched = pymaid.stitch_neurons([a, b], method='NONE')

    """
    method = method.upper() if isinstance(method, str) else method

    if method not in ['LEAFS', 'ALL', 'NONE', None]:
        raise ValueError('Unknown method: %s' % str(method))

    # Compile list of individual neurons
    neurons = utils._unpack_neurons(x)

    # Use copies of the original neurons!
    neurons = [n.copy() for n in neurons]

    if len(neurons) < 2:
        raise ValueError('Need at least 2 neurons to stitch, found {}'.format(len(neurons)))    

    stitched_n = neurons[0].copy()

    # We have to make sure node IDs are unique

    # If method is none, we can just merge the data tables
    if method == 'NONE' or not method:
        all_nodes = []
        all_cn = []

        for n in neurons:
            max_node_id = max([n.node_id.max() for n in all_nodes] + [0])
            this_nodes = n.nodes.copy()
            this_nodes.loc[:, ['node_id', 'parent_id']] += max_node_id
            all_nodes.append(this_nodes)

            if n.has_connectors:
                max_cn_id = max([n.connector_id.max() for n in all_nodes] + [0])
                this_cn = n.connectors.copy()
                this_cn.loc[:, 'node_id'] += max_node_id
                this_cn.loc[:, 'connector_id'] += max_cn_id
                all_nodes.append(this_cn)              

        stitched_n.nodes = pd.concat(all_nodes, ignore_index=True)

        if any(all_cn):            
            stitched_n.connectors = pd.concat(all_cn, ignore_index=True)

        # Reset temporary attributes of our final neuron
        stitched_n._clear_temp_attr()

        return stitched_n

    # Fix potential problems with tn_to_stitch
    if not isinstance(tn_to_stitch, type(None)):
        if not isinstance(tn_to_stitch, (list, np.ndarray)):
            tn_to_stitch = [tn_to_stitch]

        # Make sure we're working with integers
        tn_to_stitch = [int(tn) for tn in tn_to_stitch]

    for i, nB in enumerate(neurons[1:]):
        # First find treenodes to connect
        if not isinstance(tn_to_stitch, type(None)):
            if set(tn_to_stitch) & set(stitched_n.nodes.node_id):
                treenodesA = stitched_n.nodes.set_index(
                    'node_id').loc[tn_to_stitch].reset_index()
            else:
                logger.warning('None of the nodes in tn_to_stitch were found '
                               'in the first {0} stitched neurons. Falling '
                               'back to all nodes!'.format(i + 1))
                treenodesA = stitched_n.nodes

            if set(tn_to_stitch) & set(nB.nodes.node_id):
                treenodesB = nB.nodes.set_index(
                    'node_id').loc[tn_to_stitch].reset_index()
            else:
                logger.warning('None of the nodes in tn_to_stitch were found '
                               'in neuron #{0}. Falling back to all nodes!'.format(nB.skeleton_id))
                treenodesB = nB.nodes
        elif method == 'LEAFS':
            treenodesA = stitched_n.nodes[stitched_n.nodes.type.isin(
                ['end', 'root'])].reset_index()
            treenodesB = nB.nodes[nB.nodes.type.isin(
                ['end', 'root'])].reset_index()
        else:
            treenodesA = stitched_n.nodes
            treenodesB = nB.nodes

        # Calculate pairwise distances
        dist = scipy.spatial.distance.cdist(treenodesA[['x', 'y', 'z']].values,
                                            treenodesB[['x', 'y', 'z']].values,
                                            metric='euclidean')

        # Get the closest treenodes
        tnA = treenodesA.iloc[dist.argmin(axis=0)[0]].node_id
        tnB = treenodesB.iloc[dist.argmin(axis=1)[0]].node_id

        # Reroot neuronB onto the node that will be stitched
        nB.reroot(tnB)

        # Change neuronA root node's parent to treenode of neuron B
        nB.nodes.loc[nB.nodes.parent_id.isnull(), 'parent_id'] = tnA

        # Add nodes, connectors and tags onto the stitched neuron
        stitched_n.nodes = pd.concat(
            [stitched_n.nodes, nB.nodes], ignore_index=True)
        if any([n.has_connectors for n in neurons]):
            stitched_n.connectors = pd.concat(
                [stitched_n.connectors, nB.connectors], ignore_index=True)        

    # Reset temporary attributes of our final neuron
    stitched_n._clear_temp_attr()

    return stitched_n


def average_neurons(x, limit=10, base_neuron=None):
    """ Computes an average from a list of neurons.

    This is a very simple implementation which may give odd results if used
    on complex neurons. Works fine on e.g. backbones or tracts.

    Parameters
    ----------
    x :             NeuronList
                    Neurons to be averaged.
    limit :         int, optional
                    Max distance for nearest neighbour search. In microns.
    base_neuron :   skeleton_ID | TreeNeuron, optional
                    Neuron to use as template for averaging. If not provided,
                    the first neuron in the list is used as template!

    Returns
    -------
    TreeNeuron

    Examples
    --------
    >>> # Get a bunch of neurons
    >>> da1 = pymaid.get_neurons('annotation:glomerulus DA1 right')
    >>> # Prune down to longest neurite
    >>> da1.reroot(da1.soma)
    >>> da1_pr = da1.prune_by_longest_neurite(inplace=False)
    >>> # Make average
    >>> da1_avg = pymaid.average_neurons(da1_pr)
    >>> # Plot
    >>> da1.plot3d()
    >>> da1_avg.plot3d()

    """

    if not isinstance(x, core.NeuronList):
        raise TypeError('Need NeuronList, got "{0}"'.format(type(x)))

    if len(x) < 2:
        raise ValueError('Need at least 2 neurons to average!')

    # Generate KDTrees for each neuron
    for n in x:
        n.tree = graph.neuron2KDTree(n, tree_type='c', data='treenodes')

    # Set base for average: we will use this neurons treenodes to query
    # the KDTrees
    if isinstance(base_neuron, core.TreeNeuron):
        base_neuron = base_neuron.copy()
    elif isinstance(base_neuron, int):
        base_neuron = x.skid[base_neuron].copy
    elif isinstance(base_neuron, type(None)):
        base_neuron = x[0].copy()
    else:
        raise ValueError('Unable to interpret base_neuron of '
                         'type "{0}"'.format(type(base_neuron)))

    base_nodes = base_neuron.nodes[['x', 'y', 'z']].values
    other_neurons = x[1:]

    # Make sure these stay 2-dimensional arrays -> will add a colum for each
    # "other" neuron
    base_x = base_nodes[:, 0:1]
    base_y = base_nodes[:, 1:2]
    base_z = base_nodes[:, 2:3]

    # For each "other" neuron, collect nearest neighbour coordinates
    for n in other_neurons:
        nn_dist, nn_ix = n.tree.query(
            base_nodes, k=1, distance_upper_bound=limit * 1000)

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
    base_neuron.nodes.loc[:, 'x'] = mean_x
    base_neuron.nodes.loc[:, 'y'] = mean_y
    base_neuron.nodes.loc[:, 'z'] = mean_z

    return base_neuron


def despike_neuron(x, sigma=5, max_spike_length=1, inplace=False,
                   reverse=False):
    """ Removes spikes in neuron traces (e.g. from jumps in image data).

    For each treenode A, the euclidean distance to its next successor (parent)
    B and the second next successor is computed. If
    :math:`\\frac{dist(A,B)}{dist(A,C)}>sigma`. node B is considered a spike
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
        return
    elif not isinstance(x, core.TreeNeuron):
        raise TypeError('Can only process TreeNeuron or NeuronList, '
                        'not "{0}"'.format(type(x)))

    if not inplace:
        x = x.copy()

    # Index treenodes table by treenode ID
    this_treenodes = x.nodes.set_index('treenode_id')

    segs_to_walk = x.segments

    if reverse:
        segs_to_walk += x.segments[::-1]

    # For each spike length do -> do this in reverse to correct the long
    # spikes first
    for l in list(range(1, max_spike_length + 1))[::-1]:
        # Go over all segments
        for seg in x.segments:
            # Get nodes A, B and C of this segment
            this_A = this_treenodes.loc[seg[:-l - 1]]
            this_B = this_treenodes.loc[seg[l:-1]]
            this_C = this_treenodes.loc[seg[l + 1:]]

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

                this_treenodes.loc[spikes.index, ['x', 'y', 'z']] = new_positions

    # Reassign treenode table
    x.nodes = this_treenodes.reset_index(drop=False)

    # The weights in the graph have changed, we need to update that
    x._clear_temp_attr(exclude=['segments', 'small_segments',
                                'classify_nodes'])

    if not inplace:
        return x


def guess_radius(x, method='linear', limit=None, smooth=True, inplace=False):
    """ Tries guessing radii for all treenodes.

    Uses distance between connectors and treenodes and interpolate for all
    treenodes. Fills in ``radius`` column in treenode table.

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
        return

    elif not isinstance(x, core.TreeNeuron):
        raise TypeError('Can only process TreeNeuron or NeuronList, '
                        'not "{0}"'.format(type(x)))

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
    metrics.parent_dist(x, root_dist=0)
    nodes = x.nodes.set_index('treenode_id')

    # For each connector (pre and post), get the X/Y distance to its treenode
    cn_locs = cn[['x', 'y']].values
    tn_locs = nodes.loc[cn.treenode_id.values,
                        ['x', 'y']].values
    dist = np.sqrt(np.sum((tn_locs - cn_locs) ** 2, axis=1).astype(int))
    cn['dist'] = dist

    # Get max distance per treenode (in case of multiple connectors per
    # treenode)
    cn_grouped = cn.groupby('treenode_id').max()

    # Set undefined radii to None
    nodes.loc[nodes.radius <= 0, 'radius'] = None

    # Assign radii to treenodes
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
    x.nodes = nodes.reset_index(drop=False)

    if not inplace:
        return x


def smooth_neuron(x, window=5, inplace=False):
    """ Smooth neuron using rolling windows.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
                    Neuron(s) to be processed.
    window :        int, optional
                    Size of the rolling window in number of nodes.
    inplace :       bool, optional
                    If False, will use and return copy of original neuron(s).

    Returns
    -------
    TreeNeuron/List
                    Smoothed neuron(s). If ``inplace=False``.

    """

    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()

        for n in config.tqdm(x, desc='Smoothing', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            smooth_neuron(n, window=window, inplace=True)

        if not inplace:
            return x
        return

    elif not isinstance(x, core.TreeNeuron):
        raise TypeError('Can only process TreeNeuron or NeuronList, '
                        'not "{0}"'.format(type(x)))

    if not inplace:
        x = x.copy()

    # Prepare nodes (add parent_dist for later, set index)
    metrics.parent_dist(x, root_dist=0)
    nodes = x.nodes.set_index('treenode_id')

    # Go over each segment and interpolate radii
    for s in config.tqdm(x.segments, desc='Smoothing',
                         disable=config.pbar_hide,
                         leave=config.pbar_leave):

        # Get this segments radii and parent dist
        this_co = nodes.loc[s, ['x', 'y', 'z', 'parent_dist']]
        this_co['parent_dist_cum'] = this_co.parent_dist.cumsum()

        # Set cumulative distance as index and drop parent_dist
        this_co = this_co.set_index('parent_dist_cum',
                                    drop=True).drop('parent_dist', axis=1)

        interp = this_co.rolling(window, min_periods=1).mean()

        nodes.loc[s, ['x', 'y', 'z']] = interp.values

    # Reassign nodes
    x.nodes = nodes.reset_index(drop=False)

    x._clear_temp_attr()

    if not inplace:
        return x
