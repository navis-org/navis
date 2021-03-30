
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
from mpl_toolkits.mplot3d.art3d import (Line3DCollection, Poly3DCollection,
                                        Path3DCollection, Patch3DCollection)
from mpl_toolkits.mplot3d import proj3d
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.cm import ScalarMappable

import numpy as np

import pint
import warnings

from typing import Union, List, Tuple
from typing_extensions import Literal

from .. import utils, config, core
from .colors import prepare_colormap, vertex_colors
from .plot_utils import segments_to_coords, tn_pairs_to_coords

__all__ = ['plot2d']

logger = config.logger

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


def plot2d(x: Union[core.NeuronObject,
                    core.Volume,
                    np.ndarray,
                    List[Union[core.NeuronObject, np.ndarray, core.Volume]]
                    ],
           method: Union[Literal['2d'],
                         Literal['3d'],
                         Literal['3d_complex']] = '3d',
           **kwargs) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Generate 2D plots of neurons and neuropils.

    The main advantage of this is that you can save plot as vector graphics.

    Important
    ---------
    This function uses matplotlib which "fakes" 3D as it has only very limited
    control over layering objects in 3D. Therefore neurites are not necessarily
    plotted in the right Z order. This becomes especially troublesome when
    plotting a complex scene with lots of neurons criss-crossing. See the
    ``method`` parameter for details. All methods use orthogonal projection.

    Parameters
    ----------
    x :                 TreeNeuron | MeshNeuron | NeuronList | Volume | Dotprops | np.ndarray
                        Objects to plot:

                        - multiple objects can be passed as list (see examples)
                        - numpy array of shape (n,3) is intepreted as points for
                          scatter plots
    method :            '2d' | '3d' (default) | '3d_complex'
                        Method used to generate plot. Comes in three flavours:

                        1. '2d' uses normal matplotlib. Neurons are plotted on
                           top of one another in the order their are passed to
                           the function. Use the ``view`` parameter (below) to
                           set the view (default = xy).
                        2. '3d' uses matplotlib's 3D axis. Here, matplotlib
                           decide the depth order (zorder) of plotting. Can
                           change perspective either interacively or by code
                           (see examples).
                        3. '3d_complex' same as 3d but each neuron segment is
                           added individually. This allows for more complex
                           zorders to be rendered correctly. Slows down
                           rendering though.
    soma :              bool, default=True
                        Plot soma if one exists. Size of the soma is determined
                        by the neuron's ``.soma_radius`` property which defaults
                        to the "radius" column for ``TreeNeurons``.
    connectors :        bool, default=True
                        Plot connectors.
    connectors_only :   boolean, default=False
                        Plot only connectors, not the neuron.
    cn_size :           int | float, default = 1
                        Size of connectors.
    linewidth :         int | float, default=.5
                        Width of neurites. Also accepts alias ``lw``.
    linestyle :         str, default='-'
                        Line style of neurites. Also accepts alias ``ls``.
    autoscale :         bool, default=True
                        If True, will scale the axes to fit the data.
    scalebar :          int | float | str | pint.Quantity, default=False
                        Adds scale bar. Provide integer, float or str to set
                        size of scalebar. Int|float are assumed to be in same
                        units as data. You can specify units in as string:
                        e.g. "1 um". For methods '3d' and '3d_complex', this
                        will create an axis object.
    ax :                matplotlib ax, default=None
                        Pass an ax object if you want to plot on an existing
                        canvas. Must match ``method`` - i.e. 2D or 3D axis.
    figsize :           tuple, default=(8, 8)
                        Size of figure.
    color :             None | str | tuple | list | dict, default=None
                        Use single str (e.g. ``'red'``) or ``(r, g, b)`` tuple
                        to give all neurons the same color. Use ``list`` of
                        colors to assign colors: ``['red', (1, 0, 1), ...].
                        Use ``dict`` to map colors to neuron IDs:
                        ``{id: (r, g, b), ...}``.
    palette :           str | array | list of arrays, default=None
                        Name of a matplotlib or seaborn palette. If ``color`` is
                        not specified will pick colors from this palette.
    color_by :          str | array | list of arrays, default = None
                        Can be the name of a column in the node table of
                        ``TreeNeurons`` or an array of (numerical or
                        categorical) values for each node. Numerical values will
                        be normalized. You can control the normalization by
                        passing a ``vmin`` and/or ``vmax`` parameter.
    shade_by :          str | array | list of arrays, default=None
                        Similar to ``color_by`` but will affect only the alpha
                        channel of the color. If ``shade_by='strahler'`` will
                        compute Strahler order if not already part of the node
                        table (TreeNeurons only). Numerical values will be
                        normalized. You can control the normalization by passing
                        a ``smin`` and/or ``smax`` parameter.
    alpha :             float [0-1], default=.9
                        Alpha value for neurons. Overriden if alpha is provided
                        as fourth value in ``color`` (rgb*a*). You can override
                        alpha value for connectors by using ``cn_alpha``.
    depth_coloring :    bool, default=False
                        If True, will color encode depth (Z). Overrides
                        ``color``. Does not work with ``method = '3d_complex'``.
    depth_scale :       bool, default=True
                        If True and ``depth_coloring=True`` will plot a scale.
    cn_mesh_colors :    bool, default=False
                        If True, will use the neuron's color for its connectors.
    group_neurons :     bool, default=False
                        If True, neurons will be grouped. Works with SVG export
                        (not PDF). Does NOT work with ``method='3d_complex'``.
    scatter_kws :       dict, default={}
                        Parameters to be used when plotting points. Accepted
                        keywords are: ``size`` and ``color``.
    view :              tuple, default = ("x", "y")
                        Sets view for ``method='2d'``.
    orthogonal :        bool, default=True
                        Whether to use orthogonal or perspective view for
                        methods '3d' and '3d_complex'.
    volume_outlines :   bool, default=True
                        If True will plot volume outline with no fill.
    dps_scale_vec :     float
                        Scale vector for dotprops.
    rasterize :         bool, default=False
                        Neurons produce rather complex vector graphics which can
                        lead to large files when saving to SVG, PDF or PS. Use
                        this parameter to rasterize neurons and meshes/volumes
                        (but not axes or labels) to reduce file size.

    Examples
    --------
    >>> import navis
    >>> import matplotlib.pyplot as plt

    Plot list of neurons as simple 2d

    >>> nl = navis.example_neurons()
    >>> fig, ax = navis.plot2d(nl, method='2d')
    >>> plt.show() # doctest: +SKIP

    Add a volume

    >>> vol = navis.example_volume('LH')
    >>> fig, ax = navis.plot2d([nl, vol], method='2d')
    >>> plt.show() # doctest: +SKIP

    Change neuron colors

    >>> fig, ax = navis.plot2d(nl,
    ...                        method='2d',
    ...                        color=['r', 'g', 'b', 'm', 'c', 'y'])
    >>> plt.show() # doctest: +SKIP

    Plot in "fake" 3D

    >>> fig, ax = navis.plot2d(nl, method='3d')
    >>> plt.show() # doctest: +SKIP
    >>> # Now try dragging the plot to rotate

    Plot in "fake" 3D and change perspective

    >>> fig, ax = navis.plot2d(nl, method='3d')
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
    >>> plt.show() # doctest: +SKIP

    Plot using depth-coloring

    >>> fig, ax = navis.plot2d(nl, method='3d', depth_coloring=True)
    >>> plt.show() # doctest: +SKIP

    To close all figures

    >>> plt.close('all')


    Returns
    -------
    fig, ax :      matplotlib figure and axis object

    See Also
    --------
    :func:`navis.plot3d`
            Use this if you want interactive, perspectively correct renders
            and if you don't need vector graphics as outputs.
    :func:`navis.plot1d`
            A nifty way to visualise neurons in a single dimension.

    """
    # Filter kwargs
    _ACCEPTED_KWARGS = ['soma', 'connectors', 'connectors_only',
                        'ax', 'color', 'colors', 'c', 'view', 'scalebar',
                        'cn_mesh_colors', 'linewidth', 'cn_size', 'cn_alpha',
                        'orthogonal', 'group_neurons', 'scatter_kws', 'figsize',
                        'linestyle', 'rasterize',
                        'alpha', 'depth_coloring', 'autoscale', 'depth_scale',
                        'ls', 'lw', 'volume_outlines',
                        'dps_scale_vec', 'palette', 'color_by', 'shade_by',
                        'vmin', 'vmax', 'smin', 'smax', 'norm_global']
    wrong_kwargs = [a for a in kwargs if a not in _ACCEPTED_KWARGS]
    if wrong_kwargs:
        raise KeyError(f'Unknown kwarg(s): {", ".join(wrong_kwargs)}. '
                       f'Currently accepted: {", ".join(_ACCEPTED_KWARGS)}')

    _METHOD_OPTIONS = ['2d', '3d', '3d_complex']
    if method not in _METHOD_OPTIONS:
        raise ValueError(f'Unknown method "{method}". Please use either: '
                         f'{",".join(_METHOD_OPTIONS)}')

    connectors = kwargs.get('connectors', False)
    connectors_only = kwargs.get('connectors_only', False)
    ax = kwargs.pop('ax', None)
    scalebar = kwargs.get('scalebar', None)

    # Depth coloring
    depth_coloring = kwargs.get('depth_coloring', False)
    depth_scale = kwargs.get('depth_scale', True)

    scatter_kws = kwargs.get('scatter_kws', {})
    autoscale = kwargs.get('autoscale', True)

    # Parse objects
    (neurons, volumes, points, _) = utils.parse_objects(x)

    # Generate colors
    colors = kwargs.pop('color',
                        kwargs.pop('c',
                                   kwargs.pop('colors', None)))
    palette = kwargs.get('palette', None)
    color_by = kwargs.get('color_by', None)
    shade_by = kwargs.get('shade_by', None)

    # Generate the colormaps
    (neuron_cmap,
     volumes_cmap) = prepare_colormap(colors,
                                      neurons=neurons,
                                      volumes=volumes,
                                      palette=palette,
                                      alpha=kwargs.get('alpha', None),
                                      color_range=1)

    if not isinstance(color_by, type(None)):
        if not palette:
            raise ValueError('Must provide `palette` (e.g. "viridis") argument '
                             'if using `color_by`')
        neuron_cmap = vertex_colors(neurons,
                                    by=color_by,
                                    alpha=False,
                                    palette=palette,
                                    norm_global=kwargs.get('norm_global', True),
                                    vmin=kwargs.get('vmin', None),
                                    vmax=kwargs.get('vmax', None),
                                    na=kwargs.get('na', 'raise'),
                                    color_range=1)

    if not isinstance(shade_by, type(None)):
        alphamap = vertex_colors(neurons,
                                 by=shade_by,
                                 alpha=True,
                                 palette='viridis',  # palette is irrelevant here
                                 norm_global=kwargs.get('norm_global', True),
                                 vmin=kwargs.get('smin', None),
                                 vmax=kwargs.get('smax', None),
                                 na=kwargs.get('na', 'raise'),
                                 color_range=1)

        new_colormap = []
        for c, a in zip(neuron_cmap, alphamap):
            if not (isinstance(c, np.ndarray) and c.ndim == 2):
                c = np.tile(c, (a.shape[0],  1))

            if c.shape[1] == 4:
                c[:, 3] = a[:, 3]
            else:
                c = np.insert(c, 3, a[:, 3], axis=1)

            new_colormap.append(c)
        neuron_cmap = new_colormap

    # Set axis projection
    if method in ['3d', '3d_complex']:
        if kwargs.get('orthogonal', True):
            proj3d.persp_transformation = _orthogonal_proj
        else:
            proj3d.persp_transformation = _perspective_proj

    # Generate axes
    if not ax:
        if method == '2d':
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 8)))
            ax.set_aspect('equal')
        elif method in ['3d', '3d_complex']:
            fig = plt.figure(figsize=kwargs.get('figsize',
                                                plt.figaspect(1) * 1.5))
            ax = fig.add_subplot(111, projection='3d')

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
        elif method == '2d':
            if ax.name == '3d':
                raise TypeError('Axis must be 2d.')

    ax.had_data = ax.has_data()

    # Prepare some stuff for depth coloring
    if depth_coloring and not neurons.empty:
        if method == '3d_complex':
            raise Exception(f'Depth coloring unavailable for method "{method}"')
        elif method == '2d':
            bbox = neurons.bbox
            # Add to kwargs
            xy = [v.replace('-', '').replace('+', '') for v in kwargs.get('view', ('x', 'y'))]
            z_ix = [v[1] for v in [('x', 0), ('y', 1), ('z', 2)] if v[0] not in xy]

            kwargs['norm'] = plt.Normalize(vmin=bbox[z_ix, 0], vmax=bbox[z_ix, 1])

    # Plot volumes first
    if volumes:
        for i, v in enumerate(volumes):
            _ = _plot_volume(v,
                             volumes_cmap[i],
                             method,
                             ax,
                             **kwargs)

    # Create lines from segments
    visuals = {}
    for i, neuron in enumerate(config.tqdm(neurons,
                                           desc='Plot neurons',
                                           leave=False,
                                           disable=config.pbar_hide | len(neurons) < 2)):
        if not connectors_only:
            if isinstance(neuron, core.TreeNeuron) and neuron.nodes.empty:
                logger.warning(f'Skipping TreeNeuron w/o nodes: {neuron.id}')
            elif isinstance(neuron, core.MeshNeuron) and neuron.faces.size == 0:
                logger.warning(f'Skipping MeshNeuron w/o faces: {neuron.id}')
            elif isinstance(neuron, core.Dotprops) and neuron.points.size == 0:
                logger.warning(f'Skipping Dotprops w/o points: {neuron.id}')
            elif isinstance(neuron, core.TreeNeuron):
                lc, sc = _plot_skeleton(neuron, neuron_cmap[i], method, ax, **kwargs)
                # Keep track of visuals related to this neuron
                visuals[neuron] = {'skeleton': lc, 'somata': sc}
            elif isinstance(neuron, core.MeshNeuron):
                m = _plot_mesh(neuron, neuron_cmap[i], method, ax, **kwargs)
                visuals[neuron] = {'mesh': m}
            elif isinstance(neuron, core.Dotprops):
                dp = _plot_dotprops(neuron, neuron_cmap[i], method, ax, **kwargs)
                visuals[neuron] = {'dotprop': dp}
            else:
                raise TypeError(f"Don't know how to plot neuron of type '{type(neuron)}' ")

        if (connectors or connectors_only) and neuron.has_connectors:
            _ = _plot_connectors(neuron, neuron_cmap[i], method, ax, **kwargs)

    for p in points:
        _ = _plot_scatter(p, method, ax, kwargs, **scatter_kws)

    if autoscale:
        if method == '2d':
            ax.autoscale(tight=True)
        elif method in ['3d', '3d_complex']:
            # Make sure data lims are set correctly
            _update_axes3d_bounds(ax)
            # First autoscale
            ax.autoscale()
            # Now we need to set equal aspect manually
            lim = np.array([ax.get_xlim(),
                            ax.get_ylim(),
                            ax.get_zlim()])
            dim = lim[:, 1] - lim[:, 0]
            center = lim[:, 0] + dim / 2
            max_dim = dim.max()

            new_min = center - max_dim / 2
            new_max = center + max_dim / 2

            ax.set_xlim(new_min[0], new_max[0])
            ax.set_ylim(new_min[1], new_max[1])
            ax.set_zlim(new_min[2], new_max[2])

    if scalebar is not None:
        _ = _add_scalebar(scalebar, neurons, method, ax)

    def set_depth():
        """Set depth information for neurons according to camera position."""
        # Get projected coordinates
        proj_co = mpl_toolkits.mplot3d.proj3d.proj_points(all_co, ax.get_proj())

        # Get min and max of z coordinates
        z_min, z_max = min(proj_co[:, 2]), max(proj_co[:, 2])

        # Generate a new normaliser
        norm = plt.Normalize(vmin=z_min, vmax=z_max)

        # Go over all neurons and update Z information
        for neuron in visuals:
            # Get this neurons colletion and coordinates
            if 'skeleton' in visuals[neuron]:
                c = visuals[neuron]['skeleton']
                this_co = c._segments3d[:, 0, :]
            elif 'mesh' in visuals[neuron]:
                c = visuals[neuron]['mesh']
                # Note that we only get every third position -> that's because
                # these vectors actually represent faces, i.e. each vertex
                this_co = c._vec.T[::3, [0, 1, 2]]
            else:
                raise ValueError(f'Neither mesh nor skeleton found for neuron {neuron.id}')

            # Get projected coordinates
            this_proj = mpl_toolkits.mplot3d.proj3d.proj_points(this_co,
                                                                ax.get_proj())

            # Normalise z coordinates
            ns = norm(this_proj[:, 2]).data

            # Set array
            c.set_array(ns)

            # No need for normaliser - already happened
            c.set_norm(None)

            if (isinstance(neuron, core.TreeNeuron)
                and not isinstance(getattr(neuron, 'soma', None), type(None))):
                # Get depth of soma(s)
                soma = utils.make_iterable(neuron.soma)
                soma_co = neuron.nodes.set_index('node_id').loc[soma][['x', 'y', 'z']].values
                soma_proj = mpl_toolkits.mplot3d.proj3d.proj_points(soma_co,
                                                                    ax.get_proj())
                soma_cs = norm(soma_proj[:, 2]).data

                # Set soma color
                for cs, s in zip(soma_cs, visuals[neuron]['somata']):
                    s.set_color(cmap(cs))

    def Update(event):
        set_depth()

    if depth_coloring:
        cmap = mpl.cm.jet
        if method == '2d' and depth_scale:
            sm = ScalarMappable(norm=kwargs['norm'], cmap=cmap)
            fig.colorbar(sm, ax=ax, fraction=.075, shrink=.5, label='Depth')
        elif method == '3d':
            # Collect all coordinates
            all_co = []
            for n in visuals:
                if 'skeleton' in visuals[n]:
                    all_co.append(visuals[n]['skeleton']._segments3d[:, 0, :])
                if 'mesh' in visuals[n]:
                    all_co.append(visuals[n]['mesh']._vec.T[:, [0, 1, 2]])

            all_co = np.concatenate(all_co, axis=0)
            fig.canvas.mpl_connect('draw_event', Update)
            set_depth()

    plt.axis('off')

    return fig, ax


def _add_scalebar(scalebar, neurons, method, ax):
    """Add scalebar."""
    if isinstance(scalebar, bool):
        scalebar = '1 um'

    if isinstance(scalebar, str):
        scalebar = config.ureg(scalebar)

    if isinstance(scalebar, pint.Quantity):
        # If we have neurons as points of reference convert
        if neurons:
            scalebar = scalebar.to(neurons[0].units).magnitude
        # If no reference, use assume it's the same units
        else:
            scalebar = scalebar.magnitude

    # Hard-coded offset from figure boundaries
    ax_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 100 * 5

    if method == '2d':
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        coords = np.array([[xlim[0] + ax_offset, ylim[0] + ax_offset],
                           [xlim[0] + ax_offset + scalebar, ylim[0] + ax_offset]
                           ])

        sbar = mlines.Line2D(
            coords[:, 0], coords[:, 1], lw=3, alpha=.9, color='black')
        sbar.set_gid(f'{scalebar}_scalebar')

        ax.add_line(sbar)
    elif method in ['3d', '3d_complex']:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        left = xlim[0] + ax_offset
        bottom = zlim[0] + ax_offset
        front = ylim[0] + ax_offset

        sbar = [np.array([[left, front, bottom],
                          [left, front, bottom]]),
                np.array([[left, front, bottom],
                          [left, front, bottom]]),
                np.array([[left, front, bottom],
                          [left, front, bottom]])]
        sbar[0][1][0] += scalebar
        sbar[1][1][1] += scalebar
        sbar[2][1][2] += scalebar

        lc = Line3DCollection(sbar, color='black', lw=1)
        lc.set_gid(f'{scalebar}_scalebar')
        ax.add_collection3d(lc)


def _plot_scatter(points, method, ax, kwargs, **scatter_kws):
    """Plot dotprops."""
    if method == '2d':
        default_settings = dict(
            c='black',
            zorder=4,
            edgecolor='none',
            s=1
        )
        default_settings.update(scatter_kws)
        default_settings = _fix_default_dict(default_settings)

        view = kwargs.get('view', ('x', 'y'))
        x, y = _parse_view2d(points, view)

        ax.scatter(x, y, **default_settings)
    elif method in ['3d', '3d_complex']:
        default_settings = dict(
            c='black',
            s=1,
            depthshade=False,
            edgecolor='none'
        )
        default_settings.update(scatter_kws)
        default_settings = _fix_default_dict(default_settings)

        ax.scatter(points[:, 0],
                   points[:, 1],
                   points[:, 2],
                   **default_settings
                   )


def _plot_dotprops(dp, color, method, ax, **kwargs):
    """Plot dotprops."""
    # Here, we will effectively cheat and turn the dotprops into a skeleton
    # which we can then pass to _plot_skeleton
    tn = dp.to_skeleton(scale_vec=kwargs.get('dps_scale_vec', 1))

    return _plot_skeleton(tn, color, method, ax, **kwargs)


def _plot_connectors(neuron, color, method, ax, **kwargs):
    """Plot connectors."""
    cn_alpha = kwargs.get('cn_alpha', kwargs.get('alpha', .9))
    cn_size = kwargs.get('cn_size', .9)
    view = kwargs.get('view', ('x', 'y'))

    if not kwargs.get('cn_mesh_colors', False):
        cn_lay = config.default_connector_colors
    else:
        cn_lay = {{'name': c, 'color': color}
                  for c in neuron.connectors.type.unique()}

    if method == '2d':
        for c in neuron.connectors.type.unique():
            this_cn = neuron.connectors[neuron.connectors.type == c]

            x, y = _parse_view2d(this_cn[['x', 'y', 'z']].values, view)

            ax.scatter(x, y,
                       c=cn_lay[c]['color'],
                       alpha=cn_alpha,
                       zorder=4,
                       edgecolor='none',
                       s=cn_size)
            ax.get_children()[-1].set_gid(f'CN_{neuron.id}')
    elif method in ['3d', '3d_complex']:
        all_cn = neuron.connectors
        c = [cn_lay[i]['color'] for i in all_cn.type.values]
        ax.scatter(all_cn.x.values, all_cn.y.values, all_cn.z.values,
                   c=c, s=cn_size, depthshade=False, edgecolor='none',
                   alpha=cn_alpha)
        ax.get_children()[-1].set_gid(f'CN_{neuron.id}')


def _plot_mesh(neuron, color, method, ax, **kwargs):
    """Plot mesh (i.e. MeshNeuron)."""
    name = getattr(neuron, 'name')
    depth_coloring = kwargs.get('depth_coloring', False)
    alpha = kwargs.get('alpha', None)
    group_neurons = kwargs.get('group_neurons', False)
    view = kwargs.get('view', ('x', 'y'))
    rasterize = kwargs.get('rasterize', False)

    # Add alpha
    if alpha:
        color = (color[0], color[1], color[2], alpha)

    ts = None
    if method == '2d':
        # Generate 2d representation
        xy = np.dstack(_parse_view2d(neuron.vertices, view))[0]

        # Generate a patch for each face
        patches = []
        for f in neuron.faces:
            p = mpatches.Polygon(xy[f], closed=True, fill=color)
            patches.append(p)
        pc = PatchCollection(patches, linewidth=0, facecolor=color,
                             rasterized=rasterize,
                             edgecolor='none', alpha=alpha)
        ax.add_collection(pc)
    else:
        ts = ax.plot_trisurf(neuron.vertices[:, 0],
                             neuron.vertices[:, 1],
                             neuron.faces,
                             neuron.vertices[:, 2],
                             label=name,
                             rasterized=rasterize,
                             cmap=mpl.cm.jet if depth_coloring else None,
                             color=color)

        if group_neurons:
            ts.set_gid(neuron.id)
    return ts


def _get_depth_axis(view):
    """Return index of axis which is not used for x/y."""
    view = [v.replace('-', '').replace('+', '') for v in view]
    depth = [ax for ax in ['x', 'y', 'z']][0]
    map = {'x': 0, 'y': 1, 'z': 2}
    return map[depth]


def _parse_view2d(co, view):
    """Parse view parameter and returns x/y parameter."""
    if not isinstance(co, np.ndarray):
        co = np.array(co)

    map = {'x': 0, 'y': 1, 'z': 2}

    x_ix = map[view[0].replace('-', '').replace('+', '')]
    y_ix = map[view[1].replace('-', '').replace('+', '')]

    x_mod = -1 if '-' in view[0] else 1
    y_mod = -1 if '-' in view[1] else 1

    if co.ndim == 2:
        x = co[:, x_ix]
        y = co[:, y_ix]

        # Multiply only where co is not None
        x = np.multiply(x, x_mod, where=x != None, subok=False)
        y = np.multiply(y, y_mod, where=y != None, subok=False)

        # Do NOT remove the list() here - for some reason the multiplication
        # above causes issues in matplotlib
        return (list(x), list(y))
    elif co.ndim == 3:
        xy = co[:, :, [x_ix, y_ix]] * [x_mod, y_mod]
        return xy
    else:
        raise ValueError(f'Expect coordinates to have 2 or 3 dimensions, got {co.ndim}')


def _plot_skeleton(neuron, color, method, ax, **kwargs):
    """Plot skeleton."""
    depth_coloring = kwargs.get('depth_coloring', False)
    linewidth = kwargs.get('linewidth', kwargs.get('lw', .5))
    linestyle = kwargs.get('linestyle', kwargs.get('ls', '-'))
    alpha = kwargs.get('alpha', None)
    norm = kwargs.get('norm')
    plot_soma = kwargs.get('soma', True)
    group_neurons = kwargs.get('group_neurons', False)
    view = kwargs.get('view', ('x', 'y'))
    rasterize = kwargs.get('rasterize', False)

    if method == '2d':
        if not depth_coloring and not (isinstance(color, np.ndarray) and color.ndim == 2):
            # Generate by-segment coordinates
            coords = segments_to_coords(neuron,
                                        neuron.segments,
                                        modifier=(1, 1, 1))

            # We have to add (None, None, None) to the end of each
            # slab to make that line discontinuous there
            coords = np.vstack([np.append(t, [[None] * 3],
                                          axis=0) for t in coords])

            x, y = _parse_view2d(coords, view)
            this_line = mlines.Line2D(x, y,
                                      lw=linewidth, ls=linestyle,
                                      alpha=alpha, color=color,
                                      rasterized=rasterize,
                                      label=f'{getattr(neuron, "name", "NA")} - #{neuron.id}')
            ax.add_line(this_line)
        else:
            coords = tn_pairs_to_coords(neuron, modifier=(1, 1, 1))

            xy = _parse_view2d(coords, view)
            lc = LineCollection(xy,
                                cmap='jet' if depth_coloring else None,
                                norm=norm if depth_coloring else None,
                                rasterized=rasterize,
                                joinstyle='round')

            lc.set_linewidth(linewidth)
            lc.set_linestyle(linestyle)
            lc.set_label(f'{getattr(neuron, "name", "NA")} - #{neuron.id}')

            if depth_coloring:
                lc.set_alpha(alpha)
                lc.set_array(neuron.nodes.loc[neuron.nodes.parent_id >= 0, 'z'].values)
            elif (isinstance(color, np.ndarray) and color.ndim == 2):
                # If we have a color for each node, we need to drop the roots
                if color.shape[1] != coords.shape[0]:
                    lc.set_color(color[neuron.nodes.parent_id.values >= 0])
                else:
                    lc.set_color(color)

            ax.add_collection(lc)

        if plot_soma and np.any(neuron.soma):
            soma = utils.make_iterable(neuron.soma)
            # If soma detection is messed up we might end up producing
            # dozens of soma which will freeze the kernel
            if len(soma) >= 10:
                logger.warning(f'{neuron.id} - {len(soma)} somas found.')
            for s in soma:
                if isinstance(color, np.ndarray) and color.ndim > 1:
                    s_ix = np.where(neuron.nodes.node_id == s)[0][0]
                    soma_color = color[s_ix]
                else:
                    soma_color = color

                n = neuron.nodes.set_index('node_id').loc[s]
                r = getattr(n, neuron.soma_radius) if isinstance(neuron.soma_radius, str) else neuron.soma_radius

                if depth_coloring:
                    d = [n.x, n.y, n.z][_get_depth_axis(view)]
                    soma_color = mpl.cm.jet(norm(d))

                sx, sy = _parse_view2d(np.array([[n.x, n.y, n.z]]), view)
                c = mpatches.Circle((sx[0], sy[0]), radius=r,
                                    alpha=alpha, fill=True, fc=soma_color,
                                    rasterized=rasterize,
                                    zorder=4, edgecolor='none')
                ax.add_patch(c)
        return None, None

    elif method in ['3d', '3d_complex']:
        # For simple scenes, add whole neurons at a time to speed up rendering
        if method == '3d':
            if (isinstance(color, np.ndarray) and color.ndim == 2) or depth_coloring:
                coords = tn_pairs_to_coords(neuron,
                                            modifier=(1, 1, 1))
                # If we have a color for each node, we need to drop the roots
                if isinstance(color, np.ndarray) and color.shape[1] != coords.shape[0]:
                    line_color = color[neuron.nodes.parent_id.values >= 0]
                else:
                    line_color = color
            else:
                # Generate by-segment coordinates
                coords = segments_to_coords(neuron,
                                            neuron.segments,
                                            modifier=(1, 1, 1))
                line_color = color

            lc = Line3DCollection(coords,
                                  color=line_color,
                                  label=neuron.id,
                                  alpha=alpha,
                                  cmap=mpl.cm.jet if depth_coloring else None,
                                  lw=linewidth,
                                  joinstyle='round',
                                  rasterized=rasterize,
                                  linestyle=linestyle)
            if group_neurons:
                lc.set_gid(neuron.id)
            # Need to get this before adding data
            line3D_collection = lc
            ax.add_collection3d(lc)

        # For complex scenes, add each segment as a single collection
        # -> helps reducing Z-order errors
        elif method == '3d_complex':
            # Generate by-segment coordinates
            coords = segments_to_coords(neuron,
                                        neuron.segments,
                                        modifier=(1, 1, 1))
            for c in coords:
                lc = Line3DCollection([c],
                                      color=color,
                                      lw=linewidth,
                                      alpha=alpha,
                                      rasterized=rasterize,
                                      linestyle=linestyle)
                if group_neurons:
                    lc.set_gid(neuron.id)
                ax.add_collection3d(lc)
            line3D_collection = None

        surf3D_collections = []
        if plot_soma and not isinstance(getattr(neuron, 'soma', None),
                                        type(None)):
            soma = utils.make_iterable(neuron.soma)
            # If soma detection is messed up we might end up producing
            # dozens of soma which will freeze the kernel
            if len(soma) >= 5:
                logger.warning(f'Neuron {neuron.id} appears to have {len(soma)}'
                               ' somas. Skipping plotting its somas.')
            else:
                for s in soma:
                    if isinstance(color, np.ndarray) and color.ndim > 1:
                        s_ix = np.where(neuron.nodes.node_id == s)[0][0]
                        soma_color = color[s_ix]
                    else:
                        soma_color = color

                    n = neuron.nodes.set_index('node_id').loc[s]
                    r = getattr(n, neuron.soma_radius) if isinstance(neuron.soma_radius, str) else neuron.soma_radius

                    resolution = 20
                    u = np.linspace(0, 2 * np.pi, resolution)
                    v = np.linspace(0, np.pi, resolution)
                    x = r * np.outer(np.cos(u), np.sin(v)) + n.x
                    y = r * np.outer(np.sin(u), np.sin(v)) + n.y
                    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + n.z
                    surf = ax.plot_surface(x, y, z,
                                           color=soma_color,
                                           shade=False,
                                           rasterized=rasterize,
                                           alpha=alpha)
                    if group_neurons:
                        surf.set_gid(neuron.id)

                    surf3D_collections.append(surf)

        return line3D_collection, surf3D_collections


def _plot_volume(volume, color, method, ax, **kwargs):
    """Plot volume."""
    name = getattr(volume, 'name')
    rasterize = kwargs.get('rasterize', False)

    if len(color) == 4:
        this_alpha = color[3]
    else:
        this_alpha = 1

    if kwargs.get('volume_outlines', False):
        fill, lw, fc, ec = False, 1, 'none', color
    else:
        fill, lw, fc, ec = True, 0, color, 'none'

    if method == '2d':
        view = kwargs.get('view', ('x', 'y'))

        if not kwargs.get('volume_outlines', False):
            # Generate 2d representation
            xy = np.dstack(_parse_view2d(volume.verts, view))[0]

            # Generate a patch for each face
            patches = []
            for f in volume.faces:
                p = mpatches.Polygon(xy[f], closed=True, fill=fill)
                patches.append(p)
            pc = PatchCollection(patches, linewidth=lw, facecolor=fc,
                                 rasterized=rasterize,
                                 edgecolor=ec, alpha=this_alpha, zorder=0)
            ax.add_collection(pc)
        else:
            verts = volume.to_2d(view=view)
            vpatch = mpatches.Polygon(verts, closed=True, lw=lw, fill=fill,
                                      rasterized=rasterize,
                                      fc=fc, ec=ec, alpha=this_alpha, zorder=0)
            ax.add_patch(vpatch)

    elif method in ['3d', '3d_complex']:
        verts = np.vstack(volume.vertices)

        # Add alpha
        if len(color) == 3:
            color = (color[0], color[1], color[2], .1)

        ts = ax.plot_trisurf(verts[:, 0],
                             verts[:, 1],
                             volume.faces,
                             verts[:, 2],
                             label=name,
                             rasterized=rasterize,
                             color=color)
        ts.set_gid(name)


def _update_axes3d_bounds(ax):
    """Update axis bounds and remove default points (0,0,0) and (1,1,1)."""
    # Collect data points present in the figure
    points = []
    for c in ax.collections:
        if isinstance(c, Line3DCollection):
            for s in c._segments3d:
                points.append(s)
        elif isinstance(c, Poly3DCollection):
            points.append(c._vec[:3, :].T)
        elif isinstance(c, (Path3DCollection, Patch3DCollection)):
            points.append(np.array(c._offsets3d).T)
    points = np.vstack(points)

    # If this is the first set of points, we need to overwrite the defaults
    # That should happen automatically but for some reason doesn't for 3d axes
    if not getattr(ax, 'had_data', False):
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        new_xybounds = np.array([[mn[0], mn[1]],
                                 [mx[0], mx[1]]])
        new_zzbounds = np.array([[mn[2], mn[2]],
                                 [mx[2], mx[2]]])
        ax.xy_dataLim.set_points(new_xybounds)
        ax.zz_dataLim.set_points(new_zzbounds)
        ax.had_data = True
    else:
        ax.auto_scale_xyz(points[:, 0].tolist(),
                          points[:, 1].tolist(),
                          points[:, 2].tolist(),
                          had_data=True)


def __old__update_axes3d_bounds(ax, points):
    """Update axis bounds and remove default points (0,0,0) and (1,1,1)."""
    if not isinstance(points, np.ndarray):
        points = np.ndarray(points)

    # If this is the first set of points, we need to overwrite the defaults
    # That should happen automatically but for some reason doesn't for 3d axes
    if not getattr(ax, 'had_data', False):
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        new_xybounds = np.array([[mn[0], mn[1]],
                                 [mx[0], mx[1]]])
        new_zzbounds = np.array([[mn[2], mn[2]],
                                 [mx[2], mx[2]]])
        ax.xy_dataLim.set_points(new_xybounds)
        ax.zz_dataLim.set_points(new_zzbounds)
        ax.had_data = True
    else:
        ax.auto_scale_xyz(points[:, 0].tolist(),
                          points[:, 1].tolist(),
                          points[:, 2].tolist(),
                          had_data=True)


def _fix_default_dict(x):
    """Consolidate duplicate settings.

    E.g. scatter kwargs when 'c' and 'color' is provided.

    """
    # The first entry is the "survivor"
    duplicates = [['color', 'c'], ['size', 's'], ['alpha', 'a']]

    for dupl in duplicates:
        if sum([v in x for v in dupl]) > 1:
            to_delete = [v for v in dupl if v in x][1:]
            _ = [x.pop(v) for v in to_delete]

    return x


def _perspective_proj(zfront, zback):
    """Copy of the original matplotlib perspective projection."""
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, a, b],
                     [0, 0, -1, 0]])


def _orthogonal_proj(zfront, zback):
    """Get matplotlib to use orthogonal instead of perspective view.

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
