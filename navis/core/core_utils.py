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
import pint

import pandas as pd
import numpy as np
import trimesh as tm

from scipy.spatial import cKDTree
from typing import Union
from typing_extensions import Literal

from .neurons import TreeNeuron, MeshNeuron, Dotprops, BaseNeuron
from .neuronlist import NeuronList
from .. import config, graph, utils

__all__ = ['make_dotprops', 'to_neuron_space']

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


def to_neuron_space(units: Union[int, float, pint.Quantity, pint.Unit],
                    neuron: BaseNeuron,
                    on_error: Union[Literal['ignore'],
                                    Literal['raise']] = 'raise'):
    """Convert units to match neuron space.

    Parameters
    ----------
    units :     number | str | pint.Quantity | pint.Units
                The units to convert to neuron units. Simple numbers are just
                passed through.
    neuron :    Neuron
                A single neuron.
    on_error :  "raise" | "ignore"
                What to do if an error occurs (e.g. because `neuron` does not
                have units specified). If "ignore" will simply return ``units``
                unchanged.

    Returns
    -------
    float
                The units in neuron space. Note that this number may be rounded
                to avoid ugly floating point precision issues such as
                0.124999999999999 instead of 0.125.

    Examples
    --------
    >>> import navis
    >>> # Example neurons are in 8x8x8nm voxel space
    >>> n = navis.example_neurons(1)
    >>> navis.core.to_neuron_space('1 nm', n)
    0.125
    >>> # Alternatively use the neuron method
    >>> n.map_units('1 nm')
    0.125
    >>> # Numbers are passed-through
    >>> n.map_units(1)
    1
    >>> # For neuronlists
    >>> nl = navis.example_neurons(3)
    >>> nl.map_units(1)
    [1, 1, 1]
    >>> nl.map_units('1 nanometer')
    [0.125, 0.125, 0.125]

    """
    utils.eval_param(on_error, name='on_error',
                     allowed_values=('ignore', 'raise'))
    utils.eval_param(neuron, name='neuron', allowed_types=(BaseNeuron, ))

    # If string, convert to units
    if isinstance(units, str):
        units = pint.Quantity(units)
    # If not a pint object (i.e. just a number)
    elif not isinstance(units, (pint.Quantity, pint.Unit)):
        return units

    if neuron.units.dimensionless:
        if on_error == 'raise':
            raise ValueError('Neuron units unknown or dimensionless - unable '
                             f'to convert "{str(units)}"')
        else:
            return units

    # If input was e.g. `units="1"`
    if units.dimensionless:
        return units.magnitude

    # First convert to same unit as neuron units
    units = units.to(neuron.units)

    # Now convert magnitude
    mag = units.magnitude / neuron.units.magnitude

    # Rounding may not be exactly kosher but it avoids floating point issues
    # like 124.9999999999999 instead of 125
    # I hope that in practice it won't screw things up:
    # even if asking for
    return utils.round_smart(mag)
