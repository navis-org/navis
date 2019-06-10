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

import numpy as np

from typing import Optional, overload, Union
from typing_extensions import Literal

from .. import config, graph, core

# Set up logging
logger = config.logger

__all__ = ['downsample_neuron']


@overload
def downsample_neuron(x: 'core.NeuronObject',
                      downsampling_factor: float,
                      inplace: Literal[True],
                      preserve_cn_treenodes: bool = True,
                      preserve_tag_treenodes: bool = False
                      ) -> None: ...


@overload
def downsample_neuron(x: 'core.NeuronObject',
                      downsampling_factor: float,
                      inplace: Literal[False],
                      preserve_cn_treenodes: bool = True,
                      preserve_tag_treenodes: bool = False
                      ) -> 'core.NeuronObject': ...


@overload
def downsample_neuron(x: 'core.NeuronObject',
                      downsampling_factor: float,
                      inplace: bool = False,
                      preserve_cn_treenodes: bool = True,
                      preserve_tag_treenodes: bool = False
                      ) -> 'core.NeuronObject': ...


def downsample_neuron(x: 'core.NeuronObject',
                      downsampling_factor: Union[int, float],
                      inplace: bool = False,
                      preserve_cn_treenodes: bool = True,
                      preserve_tag_treenodes: bool = False,
                      ) -> Optional['core.NeuronObject']:
    """ Downsamples neuron(s) by a given factor.

    Preserves root, leafs, branchpoints by default. Preservation of treenodes
    with synapses can be toggled.

    Parameters
    ----------
    x :                      Neuron(s) to downsample.
    downsampling_factor :
                             Factor by which to reduce the node count.
    preserve_cn_treenodes :
                             If True, node that have connectors are
                             excluded from downsampling.
    inplace :
                             If True, will modify original neuron. If False, a
                             downsampled copy is returned.

    Notes
    -----
    Use ``downsampling_factor=float('inf')`` and ``preserve_cn_treenodes=False``
    to get a neuron consisting only of root, branch and end points.

    See Also
    --------
    :func:`navis.resample_neuron`
                             This function resamples a neuron to given
                             resolution. This will not preserve treenode IDs!

    """
    if isinstance(x, core.NeuronList):
        res = core.NeuronList([downsample_neuron(n,
                                                 downsampling_factor=downsampling_factor,
                                                 preserve_cn_treenodes=preserve_cn_treenodes,
                                                 preserve_tag_treenodes=preserve_tag_treenodes,
                                                 inplace=inplace) for n in x])
        if not inplace:
            return res
        else:
            return None
    elif isinstance(x, core.TreeNeuron):
        if not inplace:
            x = x.copy()
    else:
        raise TypeError(f'Unable to downsample data of type "{type(x)}"')

    # If no resampling, simply return neuron
    if downsampling_factor <= 1:
        raise ValueError('Downsampling factor must be greater than 1.')

    if x.nodes.shape[0] <= 1:
        logger.warning(f'No nodes in neuron {x.uuid}. Skipping.')
        if not inplace:
            return x
        else:
            return None

    list_of_parents = {n.node_id: n.parent_id for n in x.nodes.itertuples()}
    list_of_parents[-1] = None  # type: ignore  # doesn't know that node_id is int

    if 'type' not in x.nodes:
        graph.classify_nodes(x)

    selection = x.nodes.type != 'slab'

    if preserve_cn_treenodes and x.has_connectors:
        selection = selection | x.nodes.node_id.isin(x.connectors.node_id)  # type: ignore

    if preserve_tag_treenodes and x.has_tags:
        with_tags = [t for l in x.tags.values() for t in l]
        selection = selection | x.nodes.node_id.isin(with_tags)  # type: ignore

    fix_points = x.nodes[selection].node_id.values

    # Add soma node
    if not isinstance(x.soma, type(None)) and x.soma not in fix_points:
        fix_points = np.append(fix_points, x.soma)

    # Walk from all fix points to the root - jump N nodes on the way
    new_parents = {}

    for en in fix_points:
        this_node = en

        while True:
            stop = False
            new_p = list_of_parents[this_node]
            if new_p:
                i = 0
                while i < downsampling_factor:
                    if new_p in fix_points or not new_p:
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
                new_parents[this_node] = None  # type: ignore
                break

    new_nodes = x.nodes[x.nodes.node_id.isin(list(new_parents.keys()))].copy()
    new_nodes.loc[:, 'parent_id'] = [new_parents[tn]
                                     for tn in new_nodes.node_id.values]

    # Assign new parent IDs
    new_nodes.loc[:, 'parent_id'] = new_nodes.parent_id.values.astype(int)

    logger.debug(f'Nodes before/after: {len(x.nodes)}/{len(new_nodes)}')

    x.nodes = new_nodes

    # This is essential -> otherwise e.g. graph.neuron2graph will fail
    x.nodes.reset_index(inplace=True, drop=True)

    x._clear_temp_attr()

    if not inplace:
        return x
    return None
