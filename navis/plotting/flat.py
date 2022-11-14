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
"""Module contains functions to plot neurons as flat structures."""

import math
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import pandas as pd
import numpy as np
import networkx as nx

from typing import Optional, Union, Any, List
from typing_extensions import Literal

from .. import core, config, utils

from ..morpho.mmetrics import parent_dist

from .colors import prepare_connector_cmap

logger = config.get_logger(__name__)

__all__ = ['plot_flat']

_DEFAULTS = dict(origin=(0, 0),  # Origin in coordinate system
                 start_angle=0,  # Start angle (0 -> to right)
                 angle_change=45,  # Angle between branch and its child
                 angle_decrease=0,  # Angle decrease with each branch point
                 syn_marker_size=.5,  # Length of orthogonal synapse markers/size of scatter
                 switch_dist=1,  # Distance threshold for inverting angle (i.e. flip branch direction)
                 syn_linewidth=1.5,  # Line width for connectors
                 syn_highlight_color=(1, 0, 0),  # Color for highlighted connectors
                 force_nx=False,  # Force using networkx over igraph
                 color=(0.1, 0.1, 0.1)  # Color for neurites
                 )


def plot_flat(x,
              layout: Union[Literal['subway'],
                            Literal['dot'],
                            Literal['neato'],
                            Literal['fpd'],
                            Literal['sfpd'],
                            Literal['twopi'],
                            Literal['circo'],
                            ] = 'subway',
              connectors: bool = False,
              highlight_connectors: Optional[List[int]] = None,
              shade_by_length: bool = False,
              normalize_distance: bool = False,
              reroot_soma: bool = False,
              ax: Optional[Any] = None,
              **kwargs):
    """Plot neuron as flat diagrams.

    Parameters
    ----------
    x :                     TreeNeuron
                            A single neuron to plot.
    layout :                'subway' | 'dot' | 'neato' | 'fdp' | 'sfpd' | 'twopi' | 'circo'
                            Layout to use. All but 'subway' require graphviz to
                            be installed. For the 'fdp' and 'neato' it is highly
                            recommended to downsample the neuron first.
    connectors :            bool
                            If True and neuron has connectors, will plot
                            connectors.
    highlight_connectors :  list of connector IDs, optional
                            Will highlight these connector IDs.
    ax :                    matplotlib.ax, optional
                            Ax to plot on. Will create new one if not provided.
    shade_by_length :       bool, optional
                            Change shade of branch with length. For layout
                            "subway" only.
    normalize_distance :    bool, optional
                            If True, will normalise all distances to the longest
                            neurite. For layout "subway" only.
    **kwargs
                            Keyword argument passed on to the respective
                            plotting functions.

    Returns
    -------
    ax :                    matplotlib.ax
    pos :                   dict
                            (X, Y) positions for each node: ``{node_id: (x, y)}``.


    Examples
    --------
    Plot neuron in "subway" layout:

    .. plot::
       :context: close-figs

       >>> import navis
       >>> n = navis.example_neurons(1).convert_units('nm')
       >>> ax, pos = navis.plot_flat(n, layout='subway',
       ...                           figsize=(12, 2),
       ...                           connectors=True)
       >>> _ = ax.set_xlabel('distance [nm]')
       >>> plt.show() # doctest: +SKIP

    Plot neuron in "dot" layout (requires pygraphviz and graphviz):

    .. plot::
       :context: close-figs

        >>> # First downsample to speed up processing
        >>> ds = navis.downsample_neuron(n, 10, preserve_nodes='connectors')
        >>> ax, pos = navis.plot_flat(ds, layout='dot', connectors=True) # doctest: +SKIP
        >>> plt.show()                                                   # doctest: +SKIP

    To close all figures (only for doctests)

    >>> plt.close('all')

    See the :ref:`plotting tutorial <plot_intro>` for more examples.

    """
    if isinstance(x, core.NeuronList) and len(x) == 1:
        x = x[0]

    utils.eval_param(x, name='x', allowed_types=(core.TreeNeuron,))
    utils.eval_param(layout, name='layout',
                     allowed_values=('subway', 'dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo'))

    # Work on the copy of the neuron
    x = x.copy()

    # Reroot to soma (if applicable)
    if reroot_soma and x.soma:
        x.reroot(x.soma, inplace=True)

    if layout == 'subway':
        return _plot_subway(x,
                            connectors=connectors,
                            highlight_connectors=highlight_connectors,
                            shade_by_length=shade_by_length,
                            normalize_distance=normalize_distance,
                            ax=ax, **kwargs)
    else:
        return _plot_force(x,
                           prog=layout,
                           connectors=connectors,
                           highlight_connectors=highlight_connectors,
                           ax=ax, **kwargs)


def _plot_subway(x, connectors=False, highlight_connectors=[],
                 shade_by_length=False, normalize_distance=False, ax=None,
                 **kwargs):
    """Plot neuron as dendrogram. Preserves distances along branches."""
    DEFAULTS = _DEFAULTS.copy()
    DEFAULTS.update(kwargs)
    if len(x.root) > 1:
        raise ValueError('Unable to plot neuron with multiple roots. Use '
                         '`navis.heal_skeleton` to merge the fragments.')

    # Change scale of markers if we normalise to max neurite length
    if normalize_distance:
        DEFAULTS['syn_marker_len'] /= 1000
        DEFAULTS['switch_dist'] /= 1000

    if not ax:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))
        # Make background transparent (nicer for dark themes)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    # For each node get the distance to its root
    if 'parent_dist' not in x.nodes.columns:
        x.nodes['parent_dist'] = parent_dist(x, 0)

    # First collect leafs, branches and root
    leaf_nodes = x.leafs.node_id.values
    root_nodes = x.root
    branch_nodes = set(x.branch_points.node_id.values)

    # Use igraph if possible:
    if x.igraph and not DEFAULTS['force_nx']:
        # Convert node IDs to igraph vertex indices
        leaf_vs = x.igraph.vs.select(node_id_in=leaf_nodes)
        root_vs = x.igraph.vs.select(node_id_in=root_nodes)

        # Now get paths from all tips to the root
        paths = x.igraph.get_shortest_paths(root_vs[0], leaf_vs, mode='ALL')

        # Translate indices back into node ids
        ids = np.array(x.igraph.vs.get_attribute_values('node_id'))
        paths_tn = [ids[p] for p in paths]
    else:
        # Fall back to networkX
        iterator = nx.shortest_path(x.graph, target=root_nodes[0])
        paths_tn = [iterator[l][::-1] for l in leaf_nodes]

    # Generate DataFrame with all the info
    nodes = x.nodes.set_index('node_id')
    path_df = pd.DataFrame()
    path_df['path'] = paths_tn
    pdist = nodes.parent_dist.to_dict()
    path_df['distances'] = path_df.path.map(lambda x: np.array([pdist[n] for n in x]))
    path_df['cable'] = path_df.distances.map(lambda x: sum(x))

    # Sort DataFrame by cable length
    path_df.sort_values('cable', inplace=True, ascending=False)
    path_df.reset_index(inplace=True)

    # Prepare for plotting by finding starts points and defining angles
    positions = {x.root[0]: DEFAULTS['origin']}
    angles = {x.root[0]: DEFAULTS['start_angle']}

    seen = {x.root[0]}
    for k, path in enumerate(path_df.path.values):
        # Because the paths are always from tip to root, we have to find out
        # which of the nodes have already been plotted and at which branch point
        # we should add this path to the dendrogram
        path = np.asarray(path)
        exists = path[np.isin(path, list(seen))]
        n_branch_points = len(branch_nodes & set(exists))

        # Prune path to the bit that does not yet exist
        is_new = ~np.isin(path, list(seen))  # numpy.isin doesn't like sets
        path = path[is_new]
        start_point = positions[exists[-1]]
        last_angle = angles[exists[-1]]

        # Get distance of the remaining path
        distances = path_df.iloc[k].distances[is_new]
        distances[0] = 0
        distances = np.cumsum(distances)

        # Normalise distances
        if normalize_distance:
            longest_dist = path_df.iloc[0].cable
            distances /= longest_dist

        # Make sure the longest neurite goes horizontally
        # (or whatever START_ANGLE is)
        if k != 0:
            angle = DEFAULTS['angle_change'] - (DEFAULTS['angle_decrease'] * n_branch_points)
            angle = max(angle, DEFAULTS['angle_decrease'])
        else:
            angle = DEFAULTS['start_angle']

        # Invert angle depending on odd or even branch points
        # (only to this if major branch -> SWITCH_DIST)
        if n_branch_points % 2 != 0 and distances[-1] >= DEFAULTS['switch_dist']:
            angle *= - 1

        # Angle to radians
        angle *= math.pi/180

        # Add to last angle
        angle += last_angle

        # Calc x/y positions
        y_coords = np.array([math.sin(angle) * v for v in distances])
        x_coords = np.array([math.cos(angle) * v for v in distances])

        # Offset by starting point
        y_coords += start_point[1]
        x_coords += start_point[0]

        # Apply shade
        color = DEFAULTS['color']
        if shade_by_length:
            a = .8 - .8 * distances[-1] / path_df.cable.max()
            color = mcl.to_rgba(color, alpha=a)

        # Change linewidths with path length
        lw = 1 * distances[-1] / path_df.cable.max() + .5

        # Plot
        ax.plot(x_coords, y_coords, color=color, zorder=path_df.shape[0]-k, linewidth=lw)

        # Keep track of positions for each treenode and angle of each path
        for i, coords in enumerate(zip(x_coords, y_coords)):
            positions[path[i]] = coords

            if path[i] not in angles:
                angles[path[i]] = angle

        seen = seen | set(path)

    # Plot connectors
    if connectors and x.has_connectors:
        # Get centers for each connector
        centers = np.vstack(x.connectors.node_id.map(positions))
        # Angle of the branch they belong to
        angles = (x.connectors.node_id.map(angles) + 90 * (math.pi / 180)).values

        # Create lines orthogonal to parent branch
        y_coords = np.sin(angles) * DEFAULTS['syn_marker_size']
        y_coords = np.dstack((y_coords + centers[:, 1],
                              -y_coords + centers[:, 1],
                              [None] * len(y_coords))
                             ).flatten()
        x_coords = np.cos(angles) * DEFAULTS['syn_marker_size']
        x_coords = np.dstack((x_coords + centers[:, 0],
                              -x_coords + centers[:, 0],
                              [None] * len(x_coords))
                             ).flatten()

        cn_cmap = prepare_connector_cmap(x)
        for ty in x.connectors.type.unique():
            is_type = x.connectors.type == ty
            is_type = np.dstack((is_type, is_type, is_type)).flatten()
            ax.plot(x_coords[is_type],
                    y_coords[is_type],
                    color=cn_cmap[ty]['color'],
                    zorder=path_df.shape[0] + 1,
                    linewidth=DEFAULTS['syn_linewidth'])

    # Plot highlighted connectors
    if not isinstance(highlight_connectors, type(None)) and x.has_connectors:
        this = x.connectors[x.connectors.connector_id.isin(highlight_connectors)]
        # Get centers for each connector
        centers = np.vstack(this.node_id.map(positions))
        # Angle of the branch they belong to
        angles = (this.node_id.map(angles) + 90 * (math.pi / 180)).values

        # Create lines orthogonal to parent branch
        y_coords = np.sin(angles) * DEFAULTS['syn_marker_size']
        y_coords = np.dstack((y_coords + centers[:, 1],
                              -y_coords + centers[:, 1],
                              [None] * len(y_coords))
                             ).flatten()
        x_coords = np.cos(angles) * DEFAULTS['syn_marker_size']
        x_coords = np.dstack((x_coords + centers[:, 0],
                              -x_coords + centers[:, 0],
                              [None] * len(x_coords))
                             ).flatten()

        ax.plot(x_coords,
                y_coords,
                color=DEFAULTS['syn_highlight_color'],
                zorder=path_df.shape[0] + 2,
                linewidth=DEFAULTS['syn_linewidth'])

    # Plot soma
    if x.has_soma:
        soma = utils.make_iterable(x.soma)[0]
        soma_pos = positions[soma]
        ax.scatter([soma_pos[0]], [soma_pos[1]],
                   s=40, color=(.1, .1, .1))

    # Make sure x/y axis are equal
    ax.set_aspect('equal')

    # Return axis
    return ax, positions


def _plot_force(x, connectors=False, highlight_connectors=None, prog='dot',
                ax=None, **kwargs):
    """Plot neurons as dendrograms using graphviz layouts."""
    DEFAULTS = _DEFAULTS.copy()
    DEFAULTS.update(kwargs)
    # Save start time
    start = time.time()

    # Generate and populate networkX graph representation of the neuron
    G = x.graph.copy()
    # graphviz needs "len" not "weight"
    nx.set_edge_attributes(G, nx.get_edge_attributes(G, 'weight'), name='len')

    # Calculate layout
    logger.info('Calculating node positions.')
    positions = nx.nx_agraph.graphviz_layout(G, prog=prog,
                                             root=utils.make_iterable(x.soma)[0] if x.has_soma else None)

    # Plot tree with above layout
    logger.info('Plotting tree.')
    if not ax:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 6)))
        # Make background transparent (nicer for dark themes)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    nx.draw(G, positions,
            node_size=0,
            arrows=False,
            edge_color=DEFAULTS['color'],
            ax=ax)

    # Add soma
    if x.has_soma:
        for s in utils.make_iterable(x.soma):
            ax.scatter([positions[s][0]], [positions[s][1]],
                       s=40, color=DEFAULTS['color'],
                       zorder=1)

    if connectors and x.has_connectors:
        cn_cmap = prepare_connector_cmap(x)
        for ty in x.connectors.type.unique():
            this = x.connectors[x.connectors.type == ty]
            coords = np.vstack(this.node_id.map(positions))
            ax.scatter(coords[:, 0], coords[:, 1],
                       color=cn_cmap[ty]['color'],
                       zorder=2,
                       s=DEFAULTS['syn_marker_size'] * 10)

    if not isinstance(highlight_connectors, type(None)) and x.has_connectors:
        this = x.connectors[x.connectors.connector_id.isin(highlight_connectors)]
        coords = np.vstack(this.node_id.map(positions))
        ax.scatter(coords[:, 0], coords[:, 1],
                   color=DEFAULTS['syn_highlight_color'],
                   zorder=3,
                   s=DEFAULTS['syn_marker_size'] * 10)

    logger.debug(f'Done in {time.time()-start}s')

    return ax, positions
