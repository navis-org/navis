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
from .. import config, graph

# Set up logging
logger = config.logger

NeuronObject = Union[TreeNeuron, NeuronList]


def cable_overlap(a: NeuronObject,
                  b: NeuronObject,
                  dist: float = 2,
                  method: Union[Literal['min'], Literal['max'], Literal['avg']] = 'min'
                  ) -> pd.DataFrame:
    """Calculate the amount of cable of neuron A within distance of neuron B.

    Parameters
    ----------
    a,b :       TreeNeuron | NeuronList
                Neuron(s) for which to compute cable within distance. It is
                highly recommended to resample neurons to guarantee an even
                sampling rate. Also note that neurons need to have unique IDs.
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
    >>> nl = navis.example_neurons(4)
    >>> # Resample to 1 micron (example data is in 8x8x8nm units)
    >>> nl_res = nl.resample(1000/8, inplace=False)
    >>> # Get overlapping cable within 2 microns
    >>> ol = navis.cable_overlap(nl_res[:2], nl_res[2:], dist=2000/8)

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
        raise ValueError(f'Unknown method "{method}". Allowed methods: '
                         f'"{", ".join(allowed_methods)}"')

    if a.is_degenerated or b.is_degenerated:
        raise ValueError('Input neurons must have unique IDs.')

    matrix = pd.DataFrame(np.zeros((a.shape[0], b.shape[0])),
                          index=a.id, columns=b.id)

    with config.tqdm(total=len(a), desc='Calc. overlap',
                     disable=config.pbar_hide,
                     leave=config.pbar_leave) as pbar:
        # Keep track of KDtrees
        trees: Dict[str, scipy.spatial.cKDTree] = {}
        lengths: Dict[str, float] = {}
        for nA in a:
            # Get cKDTree for nA
            tA = trees.get(nA.id, None)
            if not tA:
                points, vect, length = graph.neuron2tangents(nA)
                trees[nA.id] = tA = scipy.spatial.cKDTree(points)
                lengths[nA.id] = length

            for nB in b:
                # Get cKDTree for nB
                tB = trees.get(nB.id, None)
                if not tB:
                    points, vect, length = graph.neuron2tangents(nB)
                    trees[nB.id] = tB = scipy.spatial.cKDTree(points)
                    lengths[nB.id] = length

                # Query nB -> nA
                distA, ixA = tA.query(tB.data,
                                      k=1,
                                      distance_upper_bound=dist,
                                      workers=-1
                                      )
                # Query nA -> nB
                distB, ixB = tB.query(tA.data,
                                      k=1,
                                      distance_upper_bound=dist,
                                      workers=-1
                                      )

                nA_lengths = lengths[nA.id][ixA[distA != float('inf')]]
                nB_lengths = lengths[nB.id][ixB[distB != float('inf')]]

                if method == 'avg':
                    overlap = (nA_lengths.sum() + nB_lengths.sum()) / 2
                elif method == 'max':
                    overlap = max(nA_lengths.sum(), nB_lengths.sum())
                elif method == 'min':
                    overlap = min(nA_lengths.sum(), nB_lengths.sum())

                matrix.at[nA.id, nB.id] = overlap

            pbar.update(1)

    return matrix
