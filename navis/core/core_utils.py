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

import numbers

import pandas as pd
import numpy as np

from scipy.spatial import cKDTree
from typing import Union, Optional

from .neurons import TreeNeuron, MeshNeuron, Dotprops
from .neuronlist import NeuronList
from .. import config

__all__ = ['make_dotprops']

# Set up logging
logger = config.logger


def make_dotprops(x: Union[pd.DataFrame, np.ndarray, 'core.TreeNeuron', 'core.MeshNeuron'],
                  k: int = 20,
                  sample: Optional[float] = None) -> Dotprops:
    """Produce dotprops from x/y/z points.

    This is following the implementation in R's nat library.

    Parameters
    ----------
    x :         pandas.DataFrame | numpy.ndarray | TreeNeuron | MeshNeuron
                Data/object to generate dotprops from. DataFrame must have
                 'x', 'y' and 'z' columns.
    k :         int
                Number of nearest neighbours to use for tangent vector
                calculation.
    sample :    float, optional
                If provided will evenly sample only fraction of data points.

    Returns
    -------
    navis.Dotprops

    """
    if isinstance(x, NeuronList):
        res = []
        for n in config.tqdm(x, desc='Dotprops',
                             leave=config.pbar_leave,
                             disable=config.pbar_hide):
            res.append(make_dotprops(n, k=k, sample=sample))
        return NeuronList(res)

    properties = {'k': k}
    if isinstance(x, pd.DataFrame):
        if not all(np.isin(['x', 'y', 'z'], x.columns)):
            raise ValueError('DataFrame must contain "x", "y" and "z" columns.')
        x = x[['x', 'y', 'z']].values
    elif isinstance(x, TreeNeuron):
        properties.update({'units': x.units, 'name': x.name, 'id': x.id})
        x = x.nodes[['x', 'y', 'z']].values
    elif isinstance(x, MeshNeuron):
        properties.update({'units': x.units, 'name': x.name, 'id': x.id})
        x = x.vertices
    elif not isinstance(x, np.ndarray):
        raise TypeError(f'Unable to generate dotprops from data of type "{type(x)}"')

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f'Expected input of shape (N, 3), got {x.shape}')

    # Drop rows with NAs
    x = x[~np.any(np.isnan(x), axis=1)]

    # Sample
    if sample:
        if not isinstance(sample, numbers.Number) or (0 >= sample > 1):
            raise ValueError('If provided, `sample` must be between 0 and 1.')
        ix = np.arange(0, x.shape[0], int(x.shape[0] * sample))
        x = x[ix]

    # Checks and balances
    n_points = x.shape[0]
    if n_points < k:
        raise ValueError(f"Too few points ({n_points}) to calculate properties.")

    # Create the KDTree and get the k-nearest neighbors for each point
    tree = cKDTree(x)
    dist, ix = tree.query(x, k=k)

    # Get points: array of (N, k, 3)
    pt = x[ix]

    # Generate centers for each cloud of k nearest neighbors
    centers = np.mean(pt, axis=1)

    # Generate vector from center
    cpt = pt - centers.reshape((pt.shape[0], 1, 3))

    # Get innertia (N, 3, 3)
    inertia = cpt.transpose((0, 2, 1)) @ cpt

    # Extract vector and alpha
    u, s, vh = np.linalg.svd(inertia)
    vect = vh[:, 0, :]
    alpha = (s[:, 0] - s[:, 1]) / np.sum(s, axis=1)

    return Dotprops(centers, alpha, vect, **properties)
