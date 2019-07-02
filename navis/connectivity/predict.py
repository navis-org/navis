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

from typing import Union, Dict
from typing_extensions import Literal

from ..core.neurons import TreeNeuron
from ..core.neuronlist import NeuronList
from .. import config

# Set up logging
logger = config.logger

NeuronObject = Union[TreeNeuron, NeuronList]


def cable_overlap(a: NeuronObject,
                  b: NeuronObject,
                  dist: float = 2,
                  method: Union[Literal['min'], Literal['max'], Literal['avg']] = 'min'
                  ) -> pd.DataFrame:
    """ Calculates the amount of cable of neuron A within distance of neuron B.

    Uses dotproduct representation of a neuron! It is recommended to
    resample neurons first.

    Parameters
    ----------
    a,b :       TreeNeuron | NeuronList
                Neuron(s) for which to compute cable within distance.
    dist :      int | float, optional
                Maximum distance.
    method :    'min' | 'max' | 'avg'
                Method by which to calculate the overlapping cable between
                two cables::

                  Assuming that neurons A and B have 300 and 150 um of cable
                  within given distances:

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

    See Also
    --------
    :func:`navis.resample_neuron`
                Use to resample neurons before calculating overlap.

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(2)
    >>> # Resample to 1 micron
    >>> nl.resample(1000, inplace=True)
    >>> # Get overlapping cable within 2 microns
    >>> ol = navis.cable_overlap(nl, nl, dist=2000)

    """

    if not isinstance(a, (TreeNeuron, NeuronList)) \
       or not isinstance(b, (TreeNeuron, NeuronList)):
        raise TypeError('Need to pass CatmaidNeurons')

    if not isinstance(a, NeuronList):
        a = NeuronList(a)

    if not isinstance(b, NeuronList):
        b = NeuronList(b)

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
        trees: Dict[str, scipy.spatial.cKDTree] = {}
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
