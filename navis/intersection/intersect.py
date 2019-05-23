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


""" This module contains functions for intersections.
"""

import pandas as pd
import numpy as np


from .. import core, config, graph

from .ray import *
from .convex import *

# Set up logging -> has to be before try statement!
logger = config.logger

try:
    from pyoctree import pyoctree
except ImportError:
    pyoctree = None
    logger.warning("Module pyoctree not found. Falling back to scipy's \
                            ConvexHull for intersection calculations.")

__all__ = sorted(['in_volume', 'intersection_matrix'])


def in_volume(x, volume, inplace=False, mode='IN', method='FAST',
              prevent_fragments=False):
    """ Test if points/neurons are within a given volume.

    Important
    ---------
    This function requires `pyoctree <https://github.com/mhogg/pyoctree>`_
    which is only an optional dependency of navis. If pyoctree is not
    installed, we will fall back to using scipy ConvexHull instead. This is
    slower and may give wrong positives for concave meshes!

    Parameters
    ----------
    x :                 list of tuples | numpy.array | pandas.DataFrame | TreeNeuron | NeuronList

                        - list/numpy.array is treated as list of x/y/z
                          coordinates. Needs to be shape (N,3): e.g.
                          ``[[x1, y1, z1], [x2, y2, z2], ..]``
                        - ``pandas.DataFrame`` needs to have ``x, y, z``
                          columns

    volume :            navis.Volume | dict or list of navis.Volume
                        :class:`navis.Volume` to test. Multiple volumes can
                        be given as list (``[volume1, volume2, ...]``) or dict
                        (``{'label1': volume1, ...}``).
    inplace :           bool, optional
                        If False, a copy of the original DataFrames/Neuron is
                        returned. Does only apply to TreeNeuron or
                        NeuronList objects. Does apply if multiple
                        volumes are provided.
    mode :              'IN' | 'OUT', optional
                        If 'IN', parts of the neuron that are within the volume
                        are kept.
    method :            'FAST' | 'SAFE', optional
                        Method used for raycasting. "FAST" will cast only a
                        single ray to check for intersections. If you
                        experience problems, set method to "SAFE" to use
                        multiple rays (slower).
    prevent_fragments : bool, optional
                        Only relevant if input is Neuron/List: if True,
                        will add nodes required to keep neuron from
                        fragmenting.

    Returns
    -------
    TreeNeuron
                      If input is TreeNeuron or NeuronList, will
                      return subset of the neuron(s) (nodes and connectors)
                      that are within given volume.
    list of bools
                      If input is a set of coordinates, returns boolean:
                      ``True`` if in volume, ``False`` if not in order.
    dict
                      If multiple volumes are provided, results will be
                      returned in dictionary with volumes as keys::

                        {'volume1': in_volume(x, volume1),
                         'volume2': in_volume(x, volume2),
                         ... }

    """

    # If we are given multiple volumes
    if isinstance(volume, (list, dict, np.ndarray)):
        # Force into dict
        if not isinstance(volume, dict):
            # Make sure all Volumes can be uniquely indexed
            vnames = set([v.name for v in volume if isinstance(v, core.Volume)])
            dupli = [v for v in set(vnames) if vnames.count(v) > 1]
            if dupli:
                raise ValueError('Duplicate Volume names detected: '
                                 f'{",".join(dupli)}. Volume.name must be '
                                 'unique.')

            temp = {v: v for v in volume if isinstance(v, str)}
            temp.update({v.name: v for v in volume if isinstance(v, core.Volume)})
            volume = temp

        data = dict()
        for v in config.tqdm(volume, desc='Volumes', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            data[v] = in_volume(x, volume[v], inplace=False, mode=mode,
                                method=method)
        return data

    # Make copy if necessary
    if isinstance(x, (core.NeuronList, core.TreeNeuron)):
        if inplace is False:
            x = x.copy()

    if isinstance(x, pd.DataFrame):
        points = x[['x', 'y', 'z']].values
    elif isinstance(x, core.TreeNeuron):
        in_v = in_volume(x.nodes[['x', 'y', 'z']].values, volume,
                         method=method)

        # If mode is OUT, invert selection
        if mode == 'OUT':
            in_v = ~np.array(in_v)

        x = graph.subset_neuron(x, x.nodes[in_v].node_id.values,
                                inplace=True,
                                prevent_fragments=prevent_fragments)

        if inplace is False:
            return x
        return
    elif isinstance(x, core.NeuronList):
        for n in x:
            _ = in_volume(n, volume, inplace=True, mode=mode, method=method,
                          prevent_fragments=prevent_fragments)

        if inplace is False:
            return x
        return
    else:
        points = x

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Points must be array of shape (N,3).')

    if pyoctree:
        return ray_in_volume(points, volume,
                             multi_ray=method.upper() == 'SAFE')
    else:
        logger.warning(
            'Package pyoctree not found. Falling back to ConvexHull.')
        return in_volume_convex(points, volume, approximate=False)


def intersection_matrix(x, volumes, attr=None, method='FAST'):
    """ Computes intersection matrix between a set of neurons and a set of
    volumes.

    Parameters
    ----------
    x :               navis.NeuronList | navis.TreeNeuron
                      Neurons to intersect.
    volume :          list or dict of navis.Volume
    attr :            str | None, optional
                      Attribute to return for intersected neurons (e.g.
                      'cable_length'). If None, will return TreeNeuron.
    method :          'FAST' | 'SAFE', optional
                      See :func:`navis.intersect.in_volume`.

    Returns
    -------
    pandas DataFrame
    """

    if isinstance(x, core.TreeNeuron):
        x = core.NeuronList(x)

    if not isinstance(x, core.NeuronList):
        raise TypeError(f'x must be Neuron/List, not "{type(x)}"')

    if isinstance(volumes, list):
        volumes = {v.name: v for v in volumes}

    if not isinstance(volumes, (list, dict)):
        raise TypeError('Volumes must be given as list or dict, not '
                        f'"{type(volumes)}"')

    for v in volumes.values():
        if not isinstance(v, core.Volume):
            raise TypeError(f'Wrong data type found in volumes: "{type(v)}"')

    data = in_volume(x, volumes, inplace=False, mode='IN', method=method)

    if not attr:
        df = pd.DataFrame([[n for n in data[v]] for v in data],
                          index=list(data.keys()),
                          columns=x.skeleton_id)
    else:
        df = pd.DataFrame([[getattr(n, attr) for n in data[v]] for v in data],
                          index=list(data.keys()),
                          columns=x.skeleton_id)

    return df
