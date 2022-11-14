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

import numpy as np
import pandas as pd

from scipy import ndimage
from typing import Optional, Union, List

from .. import config, graph, core, utils, meshes

# Set up logging
logger = config.get_logger(__name__)

__all__ = ['downsample_neuron']


@utils.map_neuronlist(desc='Downsampling', allow_parallel=True)
def downsample_neuron(x: 'core.NeuronObject',
                      downsampling_factor: Union[int, float],
                      inplace: bool = False,
                      preserve_nodes: Optional[List[int]] = None
                      ) -> Optional['core.NeuronObject']:
    """Downsample neuron(s) by a given factor.

    For skeletons: preserves root, leafs, branchpoints by default. Preservation
    of nodes with synapses can be toggled - see ``preserve_nodes`` parameter.
    Use ``downsampling_factor=float('inf')`` to get a skeleton consisting only
    of root, branch and end points.

    Parameters
    ----------
    x :                     single neuron | NeuronList
                            Neuron(s) to downsample. Note that for MeshNeurons
                            we use the first available backend.
    downsampling_factor :   int | float('inf')
                            Factor by which downsample. For TreeNeuron, Dotprops
                            and MeshNeurons this reduces the node, point
                            and face count, respectively. For VoxelNeurons it
                            reduces the dimensions by given factor.
    preserve_nodes :        str | list, optional
                            Can be either list of node IDs to exclude from
                            downsampling or a string to a DataFrame attached
                            to the neuron (e.g. "connectors"). DataFrame must
                            have `node_id`` column. Only relevant for
                            TreeNeurons.
    inplace :               bool, optional
                            If True, will modify original neuron. If False, we
                            will operate and return o a copy.

    Returns
    -------
    TreeNeuron/Dotprops/VoxelNeurons/NeuronList
                            Same datatype as input.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n_ds = navis.downsample_neuron(n,
    ...                                downsampling_factor=5,
    ...                                inplace=False)
    >>> n.n_nodes > n_ds.n_nodes
    True

    See Also
    --------
    :func:`navis.resample_skeleton`
                             This function resamples a neuron to given
                             resolution. This will change node IDs!
    :func:`navis.simplify_mesh`
                             This is the function used for ``MeshNeurons``. Use
                             directly for more control of the simplification.

    """
    if downsampling_factor <= 1:
        raise ValueError('Downsampling factor must be greater than 1.')

    if not inplace:
        x = x.copy()

    if isinstance(x, core.TreeNeuron):
        _ = _downsample_treeneuron(x,
                                   downsampling_factor=downsampling_factor,
                                   preserve_nodes=preserve_nodes)
    elif isinstance(x, core.Dotprops):
        _ = _downsample_dotprops(x,
                                 downsampling_factor=downsampling_factor)
    elif isinstance(x, core.VoxelNeuron):
        _ = _downsample_voxels(x,
                               downsampling_factor=downsampling_factor)
    elif isinstance(x, core.MeshNeuron):
        _ = meshes.simplify_mesh(x,
                                 F=1/downsampling_factor,
                                 inplace=True)
    else:
        raise TypeError(f'Unable to downsample data of type "{type(x)}"')

    return x


def _downsample_voxels(x, downsampling_factor, order=1):
    """Downsample voxels."""
    assert isinstance(x, core.VoxelNeuron)

    zoom_factor = 1 / downsampling_factor

    # order=1 means linear interpolation
    x._data = ndimage.zoom(x.grid, zoom_factor, order=order)

    # We have to change the units here too
    x.units *= downsampling_factor


def _downsample_dotprops(x, downsampling_factor):
    """Downsample Dotprops."""
    assert isinstance(x, core.Dotprops)

    # Can't downsample if no points
    if isinstance(x._points, type(None)):
        return

    # If not enough points
    if x._points.shape[0] <= downsampling_factor:
        return

    # Generate a mask
    mask = np.arange(0, x._points.shape[0], int(downsampling_factor))

    # Mask vectors
    # This will also trigger re-calculation which is necessary for two reasons:
    # 1. Vectors will change dramatically if they have to be recalculated from
    #    the downsampled dotprops.
    # 2. There might not be enough points left after downsampling given the
    #    original k.
    if isinstance(x._vect, type(None)) and x.k:
        x.recalculate_tangents(k=x.k, inplace=True)
    x._vect = x._vect[mask]

    # Mask alphas if exists
    if not isinstance(x._alpha, type(None)):
        x._alpha = x._alpha[mask]

    # Finally mask points
    x._points = x._points[mask]


def _downsample_treeneuron(x, downsampling_factor, preserve_nodes):
    """Downsample TreeNeuron."""
    assert isinstance(x, core.TreeNeuron)

    if not isinstance(preserve_nodes, type(None)):
        if isinstance(preserve_nodes, str):
            table = getattr(x, preserve_nodes)
            if not isinstance(table, pd.DataFrame):
                raise TypeError(f'Expected "{preserve_nodes}" to be a '
                                f'DataFrame - got {type(table)}')
            if 'node_id' not in table.columns:
                raise IndexError(f'DataFrame {preserve_nodes} has no "node_id"'
                                 ' column.')

            preserve_nodes = table['node_id'].values

        if not isinstance(preserve_nodes, (list, set, np.ndarray)):
            raise TypeError('Expected "preserve_nodes" to be list-like, got '
                            f'"{type(preserve_nodes)}"')

    if x.nodes.shape[0] <= 1:
        logger.warning(f'Neuron {x.id} has no nodes. Skipping.')
        return

    list_of_parents = {n: p for n, p in zip(x.nodes.node_id.values,
                                            x.nodes.parent_id.values)}
    list_of_parents[-1] = -1  # type: ignore  # doesn't know that node_id is int

    if 'type' not in x.nodes:
        graph.classify_nodes(x)

    selection = x.nodes.type != 'slab'

    if utils.is_iterable(preserve_nodes):
        selection = selection | x.nodes.node_id.isin(preserve_nodes)  # type: ignore

    fix_points = x.nodes[selection].node_id.values

    # Add soma node(s)
    if not isinstance(x.soma, type(None)):
        soma = utils.make_iterable(x.soma)
        for s in soma:
            if s not in fix_points:
                fix_points = np.append(fix_points, s)

    # Walk from all fix points to the root - jump N nodes on the way
    new_parents = {}

    for en in fix_points:
        this_node = en

        while True:
            stop = False
            new_p = list_of_parents[this_node]
            if new_p >= 0:
                i = 0
                while i < downsampling_factor:
                    if new_p in fix_points or new_p < 0:
                        new_parents[this_node] = new_p
                        stop = True
                        break
                    new_p = list_of_parents[new_p]
                    i += 1

                if stop is True:
                    break
                else:
                    new_parents[this_node] = new_p
                    this_node = new_p
            else:
                new_parents[this_node] = -1  # type: ignore
                break

    # Subset to kept nodes
    new_nodes = x.nodes[x.nodes.node_id.isin(list(new_parents.keys()))].copy()

    # Assign new parent IDs
    new_nodes['parent_id'] = new_nodes.node_id.map(new_parents).astype(int)

    logger.debug(f'Nodes before/after: {len(x.nodes)}/{len(new_nodes)}')

    x.nodes = new_nodes

    # This is essential -> otherwise e.g. graph.neuron2graph will fail
    x.nodes.reset_index(inplace=True, drop=True)

    x._clear_temp_attr()
