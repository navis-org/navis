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

""" Module contains functions to plot neurons in 2D/2.5D.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import mpl_toolkits
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.collections import LineCollection

import numpy as np

from .. import utils, config
from .colors import prepare_colormap
from .plot_utils import segments_to_coords, tn_pairs_to_coords

__all__ = ['plot2d']

logger = config.logger


def plot2d(x, method='2d', **kwargs):
    """ Generate 2D plots of neurons and neuropils.

    The main advantage of this is that you can save plot as vector graphics.

    Important
    ---------
    This function uses matplotlib which "fakes" 3D as it has only very limited
    control over layers. Therefore neurites aren't necessarily plotted in the
    right Z order which becomes especially troublesome when plotting a complex
    scene with lots of neurons criss-crossing. See the ``method`` parameter
    for details. All methods use orthogonal projection.

    Parameters
    ----------
    x :               skeleton IDs | TreeNeuron | NeuronList | CatmaidVolume | Dotprops | np.ndarray
                      Objects to plot::

                        - int is intepreted as skeleton ID(s)
                        - str is intepreted as volume name(s)
                        - multiple objects can be passed as list (see examples)
                        - numpy array of shape (n,3) is intepreted as scatter
    method :          '2d' | '3d' | '3d_complex'
                      Method used to generate plot. Comes in three flavours:
                        1. '2d' uses normal matplotlib. Neurons are plotted in
                           the order their are provided. Well behaved when
                           plotting neuropils and connectors. Always gives
                           frontal view.
                        2. '3d' uses matplotlib's 3D axis. Here, matplotlib
                           decide the order of plotting. Can chance perspective
                           either interacively or by code (see examples).
                        3. '3d_complex' same as 3d but each neuron segment is
                           added individually. This allows for more complex
                           crossing patterns to be rendered correctly. Slows
                           down rendering though.
    remote_instance : CatmaidInstance, optional
                      Need this too if you are passing only skids
    **kwargs
                      See Notes for permissible keyword arguments.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # 1. Plot two neurons from skeleton IDs:
    >>> fig, ax = navis.plot2d( [12345, 45567] )
    >>> # 2. Manually download a neuron, prune it and plot it:
    >>> neuron = navis.get_neuron( [12345], rm )
    >>> neuron.prune_distal_to( 4567 )
    >>> fig, ax = navis.plot2d( neuron )
    >>> matplotlib.pyplot.show()
    >>> # 3. Plots neuropil in grey, and mushroom body in red:
    >>> neurop = navis.get_volume('v14.neuropil')
    >>> neurop.color = (.8,.8,.8)
    >>> mb = navis.get_volume('v14.MB_whole')
    >>> mb.color = (.8,0,0)
    >>> fig, ax = navis.plot2d(  [ 12346, neurop, mb ] )
    >>> matplotlib.pyplot.show()
    >>> # Change perspective
    >>> fig, ax = navis.plot2d( neuron, method='3d_complex' )
    >>> # Change view to lateral
    >>> ax.azim = 0
    >>> ax.elev = 0
    >>> # Change view to top
    >>> ax.azim = -90
    >>> ax.elev = 90
    >>> # Tilted top view
    >>> ax.azim = -135
    >>> ax.elev = 45
    >>> # Move camera closer (will make image bigger)
    >>> ax.dist = 5

    Returns
    --------
    fig, ax :      matplotlib figure and axis object

    Notes
    -----

    Optional keyword arguments:

    ``connectors`` (boolean, default = True)
       Plot connectors (synapses, gap junctions, abutting)

    ``connectors_only`` (boolean, default = False)
       Plot only connectors, not the neuron.

    ``cn_size`` (int | float, default = 1)
      Size of connectors.

    ``linewidth``/``lw`` (int | float, default = .5)
      Width of neurites.

    ``linestyle``/``ls`` (str, default = '-')
      Line style of neurites.

    ``autoscale`` (bool, default=True)
       If True, will scale the axes to fit the data.

    ``scalebar`` (int | float, default=False)
       Adds scale bar. Provide integer/float to set size of scalebar in um.
       For methods '3d' and '3d_complex', this will create an axis object.

    ``ax`` (matplotlib ax, default=None)
       Pass an ax object if you want to plot on an existing canvas.

    ``figsize`` (tuple, default = (8, 8))
      Size of figure.

    ``color`` (tuple | list | str | dict)
      Tuples/lists (r,g,b) and str (color name) are interpreted as a single
      colors that will be applied to all neurons. Dicts will be mapped onto
      neurons by skeleton ID.

    ``alpha`` (float [0-1], default = .9)
      Alpha value for neurons. Overriden if alpha is provided as fourth value
      in ``color``.

    ``use_neuron_color`` (bool, default = False)
      If True, will attempt to use ``.color`` attribute of neurons.

    ``depth_coloring`` (bool, default = False)
      If True, will color encode depth (Z). Overrides ``color``. Does not work
      with ``method = '3d_complex'``.

    ``depth_scale`` (bool, default = True)
      If True and ``depth_coloring=True`` will plot a scale.

    ``cn_mesh_colors`` (bool, default = False)
      If True, will use the neuron's color for its connectors too.

    ``group_neurons`` (bool, default = False)
      If True, neurons will be grouped. Works with SVG export (not PDF).
      Does NOT work with ``method='3d_complex'``.

    ``scatter_kws`` (dict, default = {})
      Parameters to be used when plotting points. Accepted keywords are:
      ``size`` and ``color``.

    ``view`` (tuple, default = ("x", "y"))
      Sets view for ``method='2d'``.


    See Also
    --------
    :func:`navis.plot3d`
            Use this if you want interactive, perspectively correct renders
            and if you don't need vector graphics as outputs.

    """

    # Filter kwargs
    _ACCEPTED_KWARGS = ['connectors', 'connectors_only',
                        'ax', 'color', 'colors', 'c', 'view', 'scalebar',
                        'cn_mesh_colors', 'linewidth', 'cn_size',
                        'group_neurons', 'scatter_kws', 'figsize', 'linestyle',
                        'alpha', 'depth_coloring', 'autoscale', 'depth_scale',
                        'use_neuron_color', 'ls', 'lw']
    wrong_kwargs = [a for a in kwargs if a not in _ACCEPTED_KWARGS]
    if wrong_kwargs:
        raise KeyError(f'Unknown kwarg(s): {",".join(wrong_kwargs)}. '
                       f'Currently accepted: {",".join(_ACCEPTED_KWARGS)}')

    _METHOD_OPTIONS = ['2d', '3d', '3d_complex']
    if method not in _METHOD_OPTIONS:
        raise ValueError(f'Unknown method "{method}". Please use either: '
                         f'{",".join(_METHOD_OPTIONS)}')

    # Set axis to plot for method '2d'
    axis1, axis2 = kwargs.get('view', ('x', 'y'))

    connectors = kwargs.get('connectors', True)
    connectors_only = kwargs.get('connectors_only', False)
    cn_mesh_colors = kwargs.get('cn_mesh_colors', False)
    use_neuron_color = kwargs.get('use_neuron_color', False)
    ax = kwargs.get('ax', None)
    color = kwargs.get('color',
                       kwargs.get('c',
                                  kwargs.get('colors', None)))
    scalebar = kwargs.get('scalebar', None)
    group_neurons = kwargs.get('group_neurons', False)

    # This is overwritten if color specifies alphas
    alpha = kwargs.get('alpha', .9)

    # Depth coloring
    depth_coloring = kwargs.get('depth_coloring', False)
    depth_scale = kwargs.get('depth_scale', True)

    scatter_kws = kwargs.get('scatter_kws', {})

    linewidth = kwargs.get('linewidth', kwargs.get('lw', .5))
    cn_size = kwargs.get('cn_size', 1)
    linestyle = kwargs.get('linestyle', kwargs.get('ls', '-'))
    autoscale = kwargs.get('autoscale', True)

    # Keep track of limits if necessary
    lim = []

    # Parse objects
    skdata, dotprops, volumes, points, visuals = utils.parse_objects(x)

    # Generate the colormaps
    neuron_cmap, dotprop_cmap, volumes_cmap = \
                        prepare_colormap(color,
                                         skdata,
                                         dotprops,
                                         volumes,
                                         use_neuron_color=use_neuron_color,
                                         color_range=1)

    # Make sure axes are projected orthogonally
    if method in ['3d', '3d_complex']:
        proj3d.persp_transformation = _orthogonal_proj

    # Generate axes
    if not ax:
        if method == '2d':
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 8)))
            ax.set_aspect('equal')
        elif method in ['3d', '3d_complex']:
            fig = plt.figure(figsize=kwargs.get('figsize',
                                                plt.figaspect(1) * 1.5))
            ax = fig.gca(projection='3d')

            # This sets front view
            ax.azim = -90
            ax.elev = 0
            ax.dist = 7
            # Disallowed for 3D in matplotlib 3.1.0
            # ax.set_aspect('equal')
    # Check if correct axis were provided
    else:
        if not isinstance(ax, mpl.axes.Axes):
            raise TypeError('Ax must be of type "mpl.axes.Axes", '
                            f'not "{type(ax)}"')
        fig = ax.get_figure()
        if method in ['3d', '3d_complex']:
            if ax.name != '3d':
                raise TypeError('Axis must be 3d.')

            # Add existing limits to ax
            lim += np.array((ax.get_xlim3d(),
                             ax.get_zlim3d(),
                             ax.get_ylim3d())
                            ).T.tolist()
        elif method == '2d':
            if ax.name == '3d':
                raise TypeError('Axis must be 2d.')

    # Prepare some stuff for depth coloring
    if depth_coloring and method == '3d_complex':
        raise Exception(f'Depth coloring unavailable for method "{method}"')
    elif depth_coloring and method == '2d':
        all_co = skdata.nodes[['x', 'y', 'z']]
        norm = plt.Normalize(vmin=all_co.z.min(), vmax=all_co.z.max())

    # Plot volumes first
    if volumes:
        for i, v in enumerate(volumes):
            c = volumes_cmap[i]

            if method == '2d':
                vpatch = mpatches.Polygon(v.to_2d(view=f'{axis1}{axis2}',
                                                  invert_y=True),
                                          closed=True,
                                          lw=0,
                                          fill=True,
                                          fc=c,
                                          alpha=1)
                ax.add_patch(vpatch)
            elif method in ['3d', '3d_complex']:
                verts = np.vstack(v.vertices)

                # Invert y-axis
                verts[:, 1] *= -1

                # Add alpha
                if len(c) == 3:
                    c = (c[0], c[1], c[2], .1)

                ts = ax.plot_trisurf(verts[:, 0],
                                     verts[:, 2],
                                     v.faces,
                                     verts[:, 1],
                                     label=v.name,
                                     color=c)
                ts.set_gid(v.name)

                # Keep track of limits
                lim.append(verts.max(axis=0))
                lim.append(verts.min(axis=0))

    # Create lines from segments
    line3D_collections = []
    surf3D_collections = []
    for i, neuron in enumerate(config.tqdm(skdata.itertuples(),
                                           desc='Plot neurons',
                                           total=skdata.shape[0],
                                           leave=False,
                                           disable=config.pbar_hide | len(dotprops) == 0)):
        this_color = neuron_cmap[i]

        if neuron.nodes.empty:
            logger.warning(f'Skipping neuron w/o nodes: {neuron.uuid}')
            continue

        if not connectors_only:
            soma = neuron.nodes[neuron.nodes.radius > 1]

            # Now make traces (invert y axis)
            coords = segments_to_coords(neuron,
                                        neuron.segments,
                                        modifier=(1, -1, 1))

            if method == '2d':
                if not depth_coloring:
                    # We have to add (None, None, None) to the end of each
                    # slab to make that line discontinuous there
                    coords = np.vstack(
                        [np.append(t, [[None] * 3], axis=0) for t in coords])

                    this_line = mlines.Line2D(coords[:, 0], coords[:, 1],
                                              lw=linewidth, ls=linestyle,
                                              alpha=alpha, color=this_color,
                                              label=f'{getattr(neuron, "name", "NA")} - #{neuron.uuid}')
                    ax.add_line(this_line)
                else:
                    coords = tn_pairs_to_coords(neuron, modifier=(1, -1, 1))
                    lc = LineCollection(coords[:, :, [0, 1]],
                                        cmap='jet',
                                        norm=norm)
                    lc.set_array(neuron.nodes.loc[neuron.nodes.parent_id >= 0,
                                                  'z'].values)
                    lc.set_linewidth(linewidth)
                    lc.set_alpha(alpha)
                    lc.set_linestyle(linestyle)
                    lc.set_label(f'{getattr(neuron, "name", "NA")} - #{neuron.uuid}')
                    line = ax.add_collection(lc)

                for n in soma.itertuples():
                    if depth_coloring:
                        this_color = mpl.cm.jet(norm(n.z))

                    s = mpatches.Circle((int(n.x), int(-n.y)), radius=n.radius,
                                        alpha=alpha, fill=True, fc=this_color,
                                        zorder=4, edgecolor='none')
                    ax.add_patch(s)

            elif method in ['3d', '3d_complex']:
                cmap = mpl.cm.jet if depth_coloring else None

                # For simple scenes, add whole neurons at a time -> will speed
                # up rendering
                if method == '3d':
                    if depth_coloring:
                        this_coords = tn_pairs_to_coords(neuron,
                                                         modifier=(1, -1, 1))[:, :, [0, 2, 1]]
                    else:
                        this_coords = [c[:, [0, 2, 1]] for c in coords]

                    lc = Line3DCollection(this_coords,
                                          color=this_color,
                                          label=neuron.uuid,
                                          alpha=alpha,
                                          cmap=cmap,
                                          lw=linewidth,
                                          linestyle=linestyle)
                    if group_neurons:
                        lc.set_gid(neuron.uuid)
                    ax.add_collection3d(lc)
                    line3D_collections.append(lc)

                # For complex scenes, add each segment as a single collection
                # -> help preventing Z-order errors
                elif method == '3d_complex':
                    for c in coords:
                        lc = Line3DCollection([c[:, [0, 2, 1]]],
                                              color=this_color,
                                              lw=linewidth,
                                              alpha=alpha,
                                              linestyle=linestyle)
                        if group_neurons:
                            lc.set_gid(neuron.uuid)
                        ax.add_collection3d(lc)

                coords = np.vstack(coords)
                lim.append(coords.max(axis=0))
                lim.append(coords.min(axis=0))

                surf3D_collections.append([])
                for n in soma.itertuples():
                    resolution = 20
                    u = np.linspace(0, 2 * np.pi, resolution)
                    v = np.linspace(0, np.pi, resolution)
                    x = n.radius * np.outer(np.cos(u), np.sin(v)) + n.x
                    y = n.radius * np.outer(np.sin(u), np.sin(v)) - n.y
                    z = n.radius * \
                        np.outer(np.ones(np.size(u)), np.cos(v)) + n.z
                    surf = ax.plot_surface(
                        x, z, y, color=this_color, shade=False, alpha=alpha)
                    if group_neurons:
                        surf.set_gid(neuron.uuid)

                    surf3D_collections[-1].append(surf)

        if (connectors or connectors_only) and neuron.has_connectors:
            if not cn_mesh_colors:
                cn_types = {0: 'red', 1: 'blue', 2: 'green', 3: 'magenta'}
            else:
                cn_types = {0: this_color, 1: this_color,
                            2: this_color, 3: this_color}
            if method == '2d':
                for c in cn_types:
                    this_cn = neuron.connectors[neuron.connectors.relation == c]
                    ax.scatter(this_cn.x.values,
                               (-this_cn.y).values,
                               c=cn_types[c], alpha=alpha, zorder=4,
                               edgecolor='none', s=cn_size)
                    ax.get_children(
                    )[-1].set_gid(f'CN_{neuron.uuid}')
            elif method in ['3d', '3d_complex']:
                all_cn = neuron.connectors
                c = [cn_types[i] for i in all_cn.relation.values]
                ax.scatter(all_cn.x.values, all_cn.z.values, -all_cn.y.values,
                           c=c, s=cn_size, depthshade=False, edgecolor='none',
                           alpha=alpha)
                ax.get_children(
                )[-1].set_gid(f'CN_{neuron.uuid}')

            coords = neuron.connectors[['x', 'y', 'z']].values
            coords[:, 1] *= -1
            lim.append(coords.max(axis=0))
            lim.append(coords.min(axis=0))

    for i, neuron in enumerate(config.tqdm(dotprops.itertuples(),
                                           desc='Plt dotprops',
                                           total=dotprops.shape[0],
                                           leave=False,
                                           disable=config.pbar_hide | len(dotprops) == 0)):
        # Prepare lines - this is based on nat:::plot3d.dotprops
        halfvect = neuron.points[
            ['x_vec', 'y_vec', 'z_vec']] / 2

        starts = neuron.points[['x', 'y', 'z']
                               ].values - halfvect.values
        ends = neuron.points[['x', 'y', 'z']
                             ].values + halfvect.values

        try:
            this_color = dotprop_cmap[i]
        except BaseException:
            this_color = (.1, .1, .1)

        if method == '2d':
            # Add None between segments
            x_coords = [n for sublist in zip(
                starts[:, 0], ends[:, 0], [None] * starts.shape[0]) for n in sublist]
            y_coords = [n for sublist in zip(
                starts[:, 1] * -1, ends[:, 1] * -1, [None] * starts.shape[0]) for n in sublist]

            """
            z_coords = [n for sublist in zip(
                starts[:, 2], ends[:, 2], [None] * starts.shape[0]) for n in sublist]
            """

            this_line = mlines.Line2D(x_coords, y_coords,
                                      lw=linewidth, ls=linestyle,
                                      alpha=alpha, color=this_color,
                                      label='%s' % (neuron.gene_name))

            ax.add_line(this_line)

            # Add soma
            s = mpatches.Circle((neuron.X, -neuron.Y), radius=2,
                                alpha=alpha, fill=True, fc=this_color,
                                zorder=4, edgecolor='none')
            ax.add_patch(s)
        elif method in ['3d', '3d_complex']:
                # Combine coords by weaving starts and ends together
            coords = np.empty((starts.shape[0] * 2, 3), dtype=starts.dtype)
            coords[0::2] = starts
            coords[1::2] = ends

            # Invert y-axis
            coords[:, 1] *= -1

            # For simple scenes, add whole neurons at a time
            # -> will speed up rendering
            if method == '3d':
                lc = Line3DCollection(np.split(coords[:, [0, 2, 1]],
                                               starts.shape[0]),
                                      color=this_color,
                                      label=neuron.gene_name,
                                      lw=linewidth,
                                      alpha=alpha,
                                      linestyle=linestyle)
                if group_neurons:
                    lc.set_gid(neuron.gene_name)
                ax.add_collection3d(lc)

            # For complex scenes, add each segment as a single collection
            # -> help preventing Z-order errors
            elif method == '3d_complex':
                for c in np.split(coords[:, [0, 2, 1]], starts.shape[0]):
                    lc = Line3DCollection([c],
                                          color=this_color,
                                          lw=linewidth,
                                          alpha=alpha,
                                          linestyle=linestyle)
                    if group_neurons:
                        lc.set_gid(neuron.gene_name)
                    ax.add_collection3d(lc)

            lim.append(coords.max(axis=0))
            lim.append(coords.min(axis=0))

            resolution = 20
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            x = 2 * np.outer(np.cos(u), np.sin(v)) + neuron.X
            y = 2 * np.outer(np.sin(u), np.sin(v)) - neuron.Y
            z = 2 * np.outer(np.ones(np.size(u)), np.cos(v)) + neuron.Z
            surf = ax.plot_surface(
                x, z, y, color=this_color, shade=False, alpha=alpha)
            if group_neurons:
                surf.set_gid(neuron.gene_name)

    if points:
        for p in points:
            if method == '2d':
                default_settings = dict(
                    c='black',
                    zorder=4,
                    edgecolor='none',
                    s=1
                )
                default_settings.update(scatter_kws)
                default_settings = _fix_default_dict(default_settings)

                ax.scatter(p[:, 0],
                           p[:, 1] * -1,
                           **default_settings)
            elif method in ['3d', '3d_complex']:
                default_settings = dict(
                    c='black',
                    s=1,
                    depthshade=False,
                    edgecolor='none'
                )
                default_settings.update(scatter_kws)
                default_settings = _fix_default_dict(default_settings)

                ax.scatter(p[:, 0], p[:, 2], p[:, 1] * -1,
                           **default_settings
                           )

            coords = p
            coords[:, 1] *= -1
            lim.append(coords.max(axis=0))
            lim.append(coords.min(axis=0))

    if autoscale:
        if method == '2d':
            ax.autoscale()
        elif method in ['3d', '3d_complex']:
            lim = np.vstack(lim)
            lim_min = lim.min(axis=0)
            lim_max = lim.max(axis=0)

            center = lim_min + (lim_max - lim_min) / 2
            max_dim = (lim_max - lim_min).max()

            new_min = center - max_dim / 2
            new_max = center + max_dim / 2

            ax.set_xlim(new_min[0], new_max[0])
            ax.set_ylim(new_min[2], new_max[2])
            ax.set_zlim(new_min[1], new_max[1])

    if scalebar is not None:
        # Convert sc size to nm
        sc_size = scalebar * 1000

        # Hard-coded offset from figure boundaries
        ax_offset = 1000

        if method == '2d':
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            coords = np.array([[xlim[0] + ax_offset, ylim[0] + ax_offset],
                               [xlim[0] + ax_offset + sc_size, ylim[0] + ax_offset]
                               ])

            sbar = mlines.Line2D(
                coords[:, 0], coords[:, 1], lw=3, alpha=.9, color='black')
            sbar.set_gid(f'{scalebar}_um')

            ax.add_line(sbar)
        elif method in ['3d', '3d_complex']:
            left = lim_min[0] + ax_offset
            bottom = lim_min[1] + ax_offset
            front = lim_min[2] + ax_offset

            sbar = [np.array([[left, front, bottom],
                              [left, front, bottom]]),
                    np.array([[left, front, bottom],
                              [left, front, bottom]]),
                    np.array([[left, front, bottom],
                              [left, front, bottom]])]
            sbar[0][1][0] += sc_size
            sbar[1][1][1] += sc_size
            sbar[2][1][2] += sc_size

            lc = Line3DCollection(sbar, color='black', lw=1)
            lc.set_gid(f'{scalebar}_um')
            ax.add_collection3d(lc)

    def set_depth():
        """Sets depth information for neurons according to camera position."""

        # Modifier for soma coordinates
        modifier = np.array([1, 1, -1])

        # Get all coordinates
        all_co = np.concatenate([lc._segments3d[:, 0, :] for lc in line3D_collections],
                                axis=0)

        # Get projected coordinates
        proj_co = mpl_toolkits.mplot3d.proj3d.proj_points(all_co, ax.get_proj())

        # Get min and max of z coordinates
        z_min, z_max = min(proj_co[:, 2]), max(proj_co[:, 2])

        # Generate a new normaliser
        norm = plt.Normalize(vmin=z_min, vmax=z_max)

        # Go over all neurons and update Z information
        for neuron, lc, surf in zip(skdata,
                                    line3D_collections,
                                    surf3D_collections):
            # Get this neurons coordinates
            this_co = lc._segments3d[:, 0, :]

            # Get projected coordinates
            this_proj = mpl_toolkits.mplot3d.proj3d.proj_points(this_co,
                                                                ax.get_proj())

            # Normalise z coordinates
            ns = norm(this_proj[:, 2]).data

            # Set array
            lc.set_array(ns)

            # No need for normaliser - already happened
            lc.set_norm(None)

            # Get depth of soma(s)
            soma_co = neuron.nodes[neuron.nodes.radius > 1][['x', 'z', 'y']].values
            soma_proj = mpl_toolkits.mplot3d.proj3d.proj_points(soma_co * modifier,
                                                                ax.get_proj())
            soma_cs = norm(soma_proj[:, 2]).data

            # Set soma color
            for cs, s in zip(soma_cs, surf):
                s.set_color(cmap(cs))

    def Update(event):
        set_depth()

    if depth_coloring:
        if method == '2d' and depth_scale:
            fig.colorbar(line, ax=ax, fraction=.075, shrink=.5, label='Depth')
        elif method == '3d':
            fig.canvas.mpl_connect('draw_event', Update)
            set_depth()

    plt.axis('off')

    logger.debug('Done. Use matplotlib.pyplot.show() to show plot.')

    return fig, ax


def _fix_default_dict(x):
    """ Consolidates duplicate settings in e.g. scatter kwargs when 'c' and
    'color' is provided.
    """

    # The first entry is the "survivor"
    duplicates = [['color', 'c'], ['size', 's']]

    for dupl in duplicates:
        if sum([v in x for v in dupl]) > 1:
            to_delete = [v for v in dupl if v in x][1:]
            _ = [x.pop(v) for v in to_delete]

    return x


def _orthogonal_proj(zfront, zback):
    """ Function to get matplotlib to use orthogonal instead of perspective
    view.

    Usage:
    proj3d.persp_transformation = _orthogonal_proj
    """
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    # -0.0001 added for numerical stability as suggested in:
    # http://stackoverflow.com/questions/23840756
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, a, b],
                     [0, 0, -0.0001, zback]])
