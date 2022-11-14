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

"""This module contains functions to analyse connectivity."""

import pandas as pd
import numpy as np
import scipy.spatial

from typing import Union
from typing_extensions import Literal

from ..core import TreeNeuron, NeuronList
from .. import config, graph

# Set up logging
logger = config.get_logger(__name__)

NeuronObject = Union[TreeNeuron, NeuronList]


def cable_overlap(a: NeuronObject,
                  b: NeuronObject,
                  dist: Union[float, str] = 2,
                  method: Union[Literal['min'], Literal['max'], Literal['mean'],
                                Literal['foward'], Literal['reverse']] = 'min'
                  ) -> pd.DataFrame:
    """Calculate the amount of cable of neuron A within distance of neuron B.

    Parameters
    ----------
    a,b :       TreeNeuron | NeuronList
                Neuron(s) for which to compute cable within distance. It is
                highly recommended to resample neurons to guarantee an even
                sampling rate.
    dist :      int | float, optional
                Maximum distance. If the neurons have their `.units` set, you
                can also provides this as a string such as "2 microns".
    method :    'min' | 'max' | 'mean' | 'forward' | 'reverse'
                Method by which to calculate the overlapping cable between
                two cables::

                  Assuming that neurons A and B have 300 and 150 um of cable
                  within given distances, respectively:

                    1. 'min' returns 150
                    2. 'max' returns 300
                    3. 'mean' returns 225
                    4. 'forward' returns 300 (i.e. A->B)
                    5. 'reverse' returns 150 (i.e. B->A)

    Returns
    -------
    pandas.DataFrame
            Matrix in which neurons A are rows, neurons B are columns. Cable
            within distance is given in the neuron's native units::

                          neuronD  neuronE   neuronF  ...
                neuronA         5        1         0
                neuronB        10       20         5
                neuronC         4        3        15
                ...

    See Also
    --------
    :func:`navis.resample_skeleton`
                Use to resample neurons before calculating overlap.

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(4)
    >>> # Cable overlap is given in the neurons' units
    >>> # Converting the example neurons from 8x8x8 voxel space into microns
    >>> # make the results easier to interpret
    >>> nl = nl.convert_units('um')
    >>> # Resample to half a micron
    >>> nl_res = nl.resample('.5 micron', inplace=False)
    >>> # Get overlapping cable within 2 microns
    >>> ol = navis.cable_overlap(nl_res[:2], nl_res[2:], dist='2 microns')

    """
    if not isinstance(a, (TreeNeuron, NeuronList)) \
       or not isinstance(b, (TreeNeuron, NeuronList)):
        raise TypeError(f'Expected `TreeNeurons`, got "{type(a)}" and "{type(b)}"')

    if not isinstance(a, NeuronList):
        a = NeuronList(a)

    if not isinstance(b, NeuronList):
        b = NeuronList(b)

    # Make sure neurons have the same units
    # Do not use np.unique here because unit_str can be `None`
    units = set(np.append(a._unit_str, b._unit_str))
    units = np.array(list(units)).astype(str)
    if len(units) > 1:
        logger.warning('Neurons appear to have different units: '
                       f'{", ".join(units)}. If that is the case, cable '
                       'matrix overlap results will be garbage.')

    allowed_methods = ['min', 'max', 'mean', 'forward', 'reverse']
    if method not in allowed_methods:
        raise ValueError(f'Unknown method "{method}". Allowed methods: '
                         f'"{", ".join(allowed_methods)}"')

    dist = a[0].map_units(dist, on_error='raise')

    matrix = pd.DataFrame(np.zeros((a.shape[0], b.shape[0])),
                          index=a.id, columns=b.id)

    # Compute required props
    treesA = []
    lengthsA = []
    for nA in a:
        points, vect, length = graph.neuron2tangents(nA)
        treesA.append(scipy.spatial.cKDTree(points))
        lengthsA.append(length)

    treesB = []
    lengthsB = []
    for nB in b:
        points, vect, length = graph.neuron2tangents(nB)
        treesB.append(scipy.spatial.cKDTree(points))
        lengthsB.append(length)

    with config.tqdm(total=len(a), desc='Calc. overlap',
                     disable=config.pbar_hide,
                     leave=config.pbar_leave) as pbar:
        for i, nA in enumerate(a):
            # Get cKDTree for nA
            tA = treesA[i]

            for k, nB in enumerate(b):
                # Get cKDTree for nB
                tB = treesB[k]

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

                nA_lengths = lengthsA[i][ixA[distA != float('inf')]]
                nB_lengths = lengthsB[k][ixB[distB != float('inf')]]

                if method == 'mean':
                    overlap = (nA_lengths.sum() + nB_lengths.sum()) / 2
                elif method == 'max':
                    overlap = max(nA_lengths.sum(), nB_lengths.sum())
                elif method == 'min':
                    overlap = min(nA_lengths.sum(), nB_lengths.sum())
                elif method == 'forward':
                    overlap = nA_lengths.sum()
                elif method == 'reverse':
                    overlap = nB_lengths.sum()

                matrix.iloc[i, k] = overlap

            pbar.update(1)

    return matrix
