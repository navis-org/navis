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
import trimesh as tm

from scipy.spatial import cKDTree
from typing import Union

from .neurons import TreeNeuron, MeshNeuron, Dotprops
from .neuronlist import NeuronList
from .. import config, graph

__all__ = ['make_dotprops']

# Set up logging
logger = config.logger


def make_dotprops(x: Union[pd.DataFrame, np.ndarray, 'core.TreeNeuron', 'core.MeshNeuron'],
                  k: int = 20,
                  resample: Union[float, int, bool] = False) -> Dotprops:
    """Produce dotprops from x/y/z points.

    This is following the implementation in R's nat library.

    Parameters
    ----------
    x :         pandas.DataFrame | numpy.ndarray | TreeNeuron | MeshNeuron
                Data/object to generate dotprops from. DataFrame must have
                'x', 'y' and 'z' columns.
    k :         int, optional
                Number of nearest neighbours to use for tangent vector
                calculation. ``k=0`` or ``k=None`` is possible but only for
                ``TreeNeurons``: then we use child->parent connections
                to define points (midpoint) and their vectors. Also note that
                ``k`` is only guaranteed if the input has at least ``k`` points.
    resample :  float | int, optional
                If provided will resample neurons to the given resolution. For
                ``MeshNeurons``, we are using ``trimesh.points.remove_close`` to
                remove surface vertices closer than the given resolution. Note
                that this is only approximate and it also means that
                ``MeshNeurons`` can not be up-sampled!

    Returns
    -------
    navis.Dotprops

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> dp = navis.make_dotprops(n)
    >>> dp
    type        navis.Dotprops
    name            1734350788
    id              1734350788
    k                       20
    units          8 nanometer
    n_points              4465
    dtype: object

    """
    if resample:
        if not isinstance(resample, numbers.Number):
            raise TypeError(f'`resample` must be None, False or a Number, got "{type(resample)}"')

    if isinstance(x, NeuronList):
        res = []
        for n in config.tqdm(x, desc='Dotprops',
                             leave=config.pbar_leave,
                             disable=config.pbar_hide):
            res.append(make_dotprops(n, k=k, resample=resample))
        return NeuronList(res)

    properties = {}
    if isinstance(x, pd.DataFrame):
        if not all(np.isin(['x', 'y', 'z'], x.columns)):
            raise ValueError('DataFrame must contain "x", "y" and "z" columns.')
        x = x[['x', 'y', 'z']].values
    elif isinstance(x, TreeNeuron):
        if resample:
            x = x.resample(resample_to=resample, inplace=False)
        properties.update({'units': x.units, 'name': x.name, 'id': x.id})

        if isinstance(k, type(None)) or k <= 0:
            points, vect, length = graph.neuron2tangents(x)
            return Dotprops(points=points, vect=vect, length=length, alpha=None,
                            k=None, **properties)

        x = x.nodes[['x', 'y', 'z']].values
    elif isinstance(x, MeshNeuron):
        properties.update({'units': x.units, 'name': x.name, 'id': x.id})
        x = x.vertices
        if resample:
            x, _ = tm.points.remove_close(x, resample)
    elif not isinstance(x, np.ndarray):
        raise TypeError(f'Unable to generate dotprops from data of type "{type(x)}"')

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f'Expected input of shape (N, 3), got {x.shape}')

    if isinstance(k, type(None)) or k <= 0:
        raise ValueError('`k` must be > 0 when converting non-TreeNeurons to '
                         'Dotprops.')

    # Drop rows with NAs
    x = x[~np.any(np.isnan(x), axis=1)]

    # Checks and balances
    n_points = x.shape[0]

    # Make sure we don't ask for more nearest neighbors than we have points
    k = min(n_points, k)

    properties['k'] = k

    # Create the KDTree and get the k-nearest neighbors for each point
    tree = cKDTree(x)
    dist, ix = tree.query(x, k=k)

    # Get points: array of (N, k, 3)
    pt = x[ix]

    # Generate centers for each cloud of k nearest neighbors
    centers = np.mean(pt, axis=1)

    # Generate vector from center
    cpt = pt - centers.reshape((pt.shape[0], 1, 3))

    # Get inertia (N, 3, 3)
    inertia = cpt.transpose((0, 2, 1)) @ cpt

    # Extract vector and alpha
    u, s, vh = np.linalg.svd(inertia)
    vect = vh[:, 0, :]
    alpha = (s[:, 0] - s[:, 1]) / np.sum(s, axis=1)

    return Dotprops(points=x, alpha=alpha, vect=vect, **properties)
