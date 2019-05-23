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


""" This module contains functions to analyse connectivity.
"""

import pandas as pd
import numpy as np
import scipy.spatial

from .. import core, intersection, utils, config, graph

# Set up logging
logger = config.logger


def cable_overlap(a, b, dist=2, method='min'):
    """ Calculates the amount of cable of neuron A within distance of neuron B.

    Uses dotproduct representation of a neuron!

    Parameters
    ----------
    a,b :       TreeNeuron | NeuronList
                Neuron(s) for which to compute cable within distance.
    dist :      int, optional
                Maximum distance.
    method :    'min' | 'max' | 'avg'
                Method by which to calculate the overlapping cable between
                two cables. Assuming that neurons A and B have 300 and 150
                um of cable within given distances, respectively:
                    1. 'min' returns 150
                    2. 'max' returns 300
                    3. 'avg' returns 225

    Returns
    -------
    pandas.DataFrame
            Matrix in which neurons A are rows, neurons B are columns. Cable
            within distance is given in microns::

                        skidB1   skidB2  skidB3  ...
                skidA1    5        1        0
                skidA2    10       20       5
                skidA3    4        3        15
                ...

    """

    if not isinstance(a, (core.TreeNeuron, core.NeuronList)) \
       or not isinstance(b, (core.TreeNeuron, core.NeuronList)):
        raise TypeError('Need to pass CatmaidNeurons')

    if isinstance(a, core.NeuronList):
        a = core.NeuronList(a)

    if isinstance(b, core.NeuronList):
        b = core.NeuronList(b)

    allowed_methods = ['min', 'max', 'avg']
    if method not in allowed_methods:
        raise ValueError('Unknown method "{0}". Allowed methods: "{0}"'.format(
            method, ','.join(allowed_methods)))

    matrix = pd.DataFrame(np.zeros((a.shape[0], b.shape[0])),
                          index=a.name, columns=b.name)

    with config.tqdm(total=len(a), desc='Calc. overlap',
                     disable=config.pbar_hide,
                     leave=config.pbar_leave) as pbar:
        # Keep track of KDtrees
        trees = {}
        for nA in a:
            # Get cKDTree for nA
            tA = trees.get(nA.name, None)
            if not tA:
                trees[nA.name] = tA = scipy.spatial.cKDTree(
                    np.vstack(nA.dps.point), leafsize=10)

            for nB in b:
                # Get cKDTree for nB
                tB = trees.get(nB.name, None)
                if not tB:
                    trees[nB.name] = tB = scipy.spatial.cKDTree(
                        np.vstack(nB.dps.point), leafsize=10)

                # Query nB -> nA
                distA, ixA = tA.query(np.vstack(nB.dps.point),
                                      k=1,
                                      distance_upper_bound=dist,
                                      n_jobs=-1
                                      )
                # Query nA -> nB
                distB, ixB = tB.query(np.vstack(nA.dps.point),
                                      k=1,
                                      distance_upper_bound=dist,
                                      n_jobs=-1
                                      )

                nA_in_dist = nA.dps.loc[ixA[distA != float('inf')]]
                nB_in_dist = nB.dps.loc[ixB[distB != float('inf')]]

                if nA_in_dist.empty:
                    overlap = 0
                elif method == 'avg':
                    overlap = (nA_in_dist.vec_length.sum() +
                               nB_in_dist.vec_length.sum()) / 2
                elif method == 'max':
                    overlap = max(nA_in_dist.vec_length.sum(),
                                  nB_in_dist.vec_length.sum())
                elif method == 'min':
                    overlap = min(nA_in_dist.vec_length.sum(),
                                  nB_in_dist.vec_length.sum())

                matrix.at[nA.name, nB.name] = overlap

            pbar.update(1)

    return matrix


def predict_connectivity(source, target, method='possible_contacts', **kwargs):
    """ Calculates potential synapses from source onto target neurons based
    on distance.

    Based on a concept by Alexander Bates.

    Parameters
    ----------
    source,target : CatmaidNeuron | CatmaidNeuronList
                    Neuron(s) for which to compute potential connectivity.
                    This is unidirectional: source -> target.
    method :        'possible_contacts'
                    Method to use for calculations. See Notes.
    **kwargs
                    1. For method 'possible_contacts':
                        - ``dist`` to set distance between connectors and
                          treenodes manually.
                        - ``stdev`` to set number of standard-deviations of
                          average distance. Default = 2.

    Notes
    -----
    Method ``possible_contacts``:
        1. Calculating mean distance ``d`` (connector->treenode) at which
           connections between neurons A and neurons B occur.
        2. For all presynapses of neurons A, check if they are within ``stdev``
           (default=2) standard deviations of ``d`` of a neurons B treenode.


    Returns
    -------
    pandas.DataFrame
            Matrix holding possible synaptic contacts. Sources are rows,
            targets are columns::

                         target1  target2  target3  ...
                source1    5        1        0
                source2    10       20       5
                source3    4        3        15
                ...

    """

    for _ in [source, target]:
        if not isinstance(_, (core.TreeNeuron, core.NeuronList)):
            raise TypeError('Need CatmaidNeuron/List, got '
                            '"{}"'.format(type(_)))

    if isinstance(source, core.CatmaidNeuron):
        source = core.CatmaidNeuronList(source)

    if isinstance(target, core.CatmaidNeuron):
        target = core.CatmaidNeuronList(target)

    allowed_methods = ['possible_contacts']
    if method not in allowed_methods:
        raise ValueError('Unknown method "{0}". Allowed methods: "{0}"'.format(
            method, ','.join(allowed_methods)))

    matrix = pd.DataFrame(np.zeros((source.shape[0], target.shape[0])),
                          index=source.skeleton_id,
                          columns=target.skeleton_id)

    # First let's calculate at what distance synapses are being made
    cn_between = fetch.get_connectors_between(source, target,
                                              remote_instance=remote_instance)

    if kwargs.get('dist', None):
        distances = kwargs.get('dist')
    elif cn_between.shape[0] > 0:
        logger.warning('No ')
        cn_locs = np.vstack(cn_between.connector_loc.values)
        tn_locs = np.vstack(cn_between.treenode2_loc.values)

        distances = np.sqrt(np.sum((cn_locs - tn_locs) ** 2, axis=1))

        logger.info('Average connector->treenode distances: '
                    '{:.2f} +/- {:.2f} nm'.format(distances.mean(),
                                                  distances.std()))
    else:
        logger.warning('No existing connectors to calculate average'
                       'connector->treenode distance found. Falling'
                       'back to default of 1um. Use <stdev> argument'
                       'to set manually.')
        distances = 1000

    # Calculate distances threshold
    n_std = kwargs.get('n_std', 2)
    dist_threshold = np.mean(distances) + n_std * np.std(distances)

    with config.tqdm(total=len(target), desc='Predicting',
                     disable=config.pbar_hide,
                     leave=config.pbar_leave) as pbar:
        for t in target:
            # Create cKDTree for target
            tree = scipy.spatial.cKDTree(
                t.nodes[['x', 'y', 'z']].values, leafsize=10)
            for s in source:
                # Query against presynapses
                dist, ix = tree.query(s.presynapses[['x', 'y', 'z']].values,
                                      k=1,
                                      distance_upper_bound=dist_threshold,
                                      n_jobs=-1
                                      )

                # Calculate possible contacts
                possible_contacts = sum(dist != float('inf'))

                matrix.at[s.skeleton_id, t.skeleton_id] = possible_contacts

            pbar.update(1)

    return matrix.astype(int)
