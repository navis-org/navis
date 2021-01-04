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

""" Module contains functions to plot neurons in 1D.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

from typing import Optional, Union, Dict, Tuple, Any

from .. import core, config, graph

logger = config.logger

colortype = Union[str,
                  Tuple[float, float, float],
                  Tuple[float, float, float, float]]

__all__ = ['plot1d']


def plot1d(x: 'core.NeuronObject',
           ax: Optional[mpl.axes.Axes] = None,
           color: Optional[Union['str',
                                 colortype,
                                 Dict[Any, colortype],
                                 ]
                           ] = None,
           **kwargs) -> mpl.axes.Axes:
    """Plot neuron topology in 1D according to Cuntz et al. (2010).

    This function breaks a neurons into segments between branch points.
    See Cuntz et al., PLoS Computational Biology (2010) for detailed
    explanation. For very complex neurons, this neuron "barcode" can get
    fairly complicated - make sure to zoom in.

    Parameters
    ----------
    x :         TreeNeuron | NeuronList
                Neuron(s) to plot.
    ax :        matplotlib.ax, optional
    cmap :      tuple | dict
                Color. If dict must map neuron UUID to color.
    **kwargs
                Will be passed to ``matplotlib.patches.Rectangle``.

    Returns
    -------
    matplotlib.ax

    Examples
    --------
    >>> import navis
    >>> import matplotlib.pyplot as plt
    >>> n = navis.example_neurons(2)
    >>> ax = navis.plot1d(n)
    >>> plt.show() # doctest: +SKIP

    To close figure(s)
    
    >>> plt.close('all')

    """
    if isinstance(x, core.NeuronList):
        if x.is_mixed:
            raise TypeError('NeuronList contains MeshNeuron(s). Unable to plot1d.')
    elif isinstance(x, core.TreeNeuron):
        x = core.NeuronList(x)
    else:
        raise TypeError(f'Unable plot1d data of type "{type(x)}"')

    if isinstance(color, type(None)):
        color = (0.56, 0.86, 0.34)

    if not ax:
        fig, ax = plt.subplots(figsize=(8, len(x) / 3))

    # Add some default parameters for the plotting to kwargs
    kwargs.update({'lw': kwargs.get('lw', .1),
                   'ec': kwargs.get('ec', (1, 1, 1)),
                   })

    max_x = []
    for ix, n in enumerate(config.tqdm(x, desc='Processing',
                                       disable=config.pbar_hide,
                                       leave=config.pbar_leave)):
        if isinstance(color, dict):
            this_c = color[n.id]
        else:
            this_c = color

        # Get topological sort (root -> terminals)
        topology = graph.node_label_sorting(n)

        # Get terminals and branch points
        bp = n.nodes[n.nodes.type == 'branch'].node_id.values
        term = n.nodes[n.nodes.type == 'end'].node_id.values

        # Order this neuron's segments by topology
        breaks = [topology[0]] + \
            [n for i, n in enumerate(topology) if n in bp or n in term]
        segs = [([s for s in n.small_segments if s[0] == end][0][-1], end)
                for end in breaks[1:]]

        # Now get distances for each segment
        if 'nodes_geodesic_distance_matrix' in n.__dict__:
            # If available, use geodesic distance matrix
            dist_mat = n.nodes_geodesic_distance_matrix
        else:
            # If not, compute matrix for subset of nodes
            dist_mat = graph.geodesic_matrix(n, tn_ids=breaks, directed=False)

        dist = np.array([dist_mat.loc[s[0], s[1]] for s in segs]) / 1000
        max_x.append(sum(dist))

        # Plot
        curr_dist = 0
        for k, d in enumerate(dist):
            if segs[k][1] in term:
                c = tuple(np.array(this_c) / 2)
            else:
                c = color

            p = mpatches.Rectangle((curr_dist, ix), d, 1, fc=c, **kwargs)
            ax.add_patch(p)
            curr_dist += d

    ax.set_xlim(0, max(max_x))
    ax.set_ylim(0, len(x))

    ax.set_yticks(np.array(range(0, len(x))) + .5)
    ax.set_yticklabels(x.name)

    ax.set_xlabel('distance [um]')

    ax.set_frame_on(False)

    try:
        plt.tight_layout()
    except BaseException:
        pass

    return ax
