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

""" Module contains functions to plot neurons in 1D.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

from typing import Optional, Union, Dict, Tuple, Any

from .. import core, config, graph
from .colors import prepare_colormap, vertex_colors

logger = config.get_logger(__name__)

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
           color_by: Optional[Union[str, np.ndarray]] = None,
           palette: Optional[str] = None,
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
    color :     tuple | dict
                Color. If dict must map neuron UUID to color.
    palette :   str | array | list of arrays, default=None
                Name of a matplotlib or seaborn palette. If ``color`` is
                not specified will pick colors from this palette.
    color_by :  str | array | list of arrays, default = None
                Can be the name of a column in the node table of
                ``TreeNeurons`` or an array of (numerical or categorical)
                values for each node. Numerical values will be normalized.
                You can control the normalization by passing a ``vmin``
                and/or ``vmax`` parameter.
    **kwargs
                Will be passed to ``matplotlib.patches.Rectangle``.

    Returns
    -------
    matplotlib.ax

    Examples
    --------

    .. plot::
       :context: close-figs

        >>> import navis
        >>> import matplotlib.pyplot as plt
        >>> n = navis.example_neurons(2)
        >>> ax = navis.plot1d(n)
        >>> plt.show() # doctest: +SKIP

    Close figures (only relevant for doctests)

    >>> plt.close('all')

    See the :ref:`plotting tutorial <plot_intro>` for more examples.

    """
    if isinstance(x, core.NeuronList):
        if x.is_mixed:
            raise TypeError('NeuronList contains MeshNeuron(s). Unable to plot1d.')
    elif isinstance(x, core.TreeNeuron):
        x = core.NeuronList(x)
    else:
        raise TypeError(f'Unable plot1d data of type "{type(x)}"')

    if isinstance(color, type(None)) and isinstance(palette, type(None)):
        color = (0.56, 0.86, 0.34)

    color, _ =  prepare_colormap(color,
                                 neurons=x,
                                 palette=palette,
                                 color_range=1)

    if not isinstance(color_by, type(None)):
        if not palette:
            raise ValueError('Must provide `palette` (e.g. "viridis") argument '
                             'if using `color_by`')

        vertex_map = vertex_colors(x,
                                   by=color_by,
                                   use_alpha=False,
                                   palette=palette,
                                   vmin=kwargs.pop('vmin', None),
                                   vmax=kwargs.pop('vmax', None),
                                   na=kwargs.pop('na', 'raise'),
                                   color_range=1)
    else:
        vertex_map = None

    if not ax:
        fig, ax = plt.subplots(figsize=(8, len(x)))
        # Make background transparent (nicer for dark themes)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

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
            this_c = color[ix]

        # Get topological sort (root -> terminals)
        topology = graph.node_label_sorting(n, weighted=True)

        # Get terminals and branch points
        roots = n.nodes[n.nodes.type == 'root'].node_id.values
        bp = n.nodes[n.nodes.type == 'branch'].node_id.values
        term = n.nodes[n.nodes.type == 'end'].node_id.values
        breaks = np.concatenate((bp, term, roots))

        # Order this neuron's segments by topology (remember that segments are
        # sorted child -> parent, i.e. distal to proximal)
        topo_ix = dict(zip(topology, range(len(topology))))
        segs = sorted(n.small_segments, key=lambda x: topo_ix[x[0]])

        # Keep only the first and the last node in each segment
        segs = [[s[0], s[1]] for s in segs]

        # Now get distances for each segment
        if 'nodes_geodesic_distance_matrix' in n.__dict__:
            # If available, use geodesic distance matrix
            dist_mat = n.nodes_geodesic_distance_matrix
        else:
            # If not, compute matrix for subset of nodes
            dist_mat = graph.geodesic_matrix(n, from_=breaks, directed=False)

        # Get length of each segment
        lengths = np.array([dist_mat.loc[s[0], s[1]] for s in segs])
        max_x.append(sum(lengths))

        # Plot
        curr_dist = 0
        id2ix = dict(zip(n.nodes.node_id.values, range(n.n_nodes)))
        for k, l in enumerate(lengths):
            if isinstance(vertex_map, type(None)):
                c = this_c

                if segs[k][0] in term:
                    c = tuple(np.array(c) / 2)
            else:
                # Get this segments vertex colors
                node_ix = [id2ix[i] for i in segs[k]]
                vc = vertex_map[ix][node_ix]
                c = vc[-1]

            p = mpatches.Rectangle((curr_dist, ix), l, 1, fc=c, **kwargs)
            ax.add_patch(p)
            curr_dist += l

    ax.set_xlim(0, max(max_x))
    ax.set_ylim(0, len(x))

    ax.set_yticks(np.array(range(0, len(x))) + .5)
    ax.set_yticklabels(x.name)

    dstring = 'distance'
    ax.set_xlabel(dstring)

    ax.set_frame_on(False)

    return ax
