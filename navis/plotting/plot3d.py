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

""" Module contains functions to plot neurons in 2D and 3D.
"""

import numbers
import warnings

import pandas as pd
import numpy as np

import plotly.offline

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import vispy

from .. import morpho, core, utils, config
from .vispy.viewer import Viewer
from .plotly.graph_objs import *
from .colors import prepare_colormap

try:
    # Try setting vispy backend to PyQt5
    vispy.use(app='PyQt5')
except BaseException:
    pass

__all__ = ['plot3d']

logger = config.logger

if utils.is_jupyter():
    plotly.offline.init_notebook_mode(connected=True)


def plot3d(x, **kwargs):
    """ Generate 3D plot.

    Uses either `vispy <http://vispy.org>`_ (default) or
    `plotly <http://plot.ly>`_.

    Parameters
    ----------

    x :               TreeNeuron/List| navis.Dotprops | navis.Volume | numpy.array
                        - ``numpy.array (N,3)`` is plotted as scatter plot
                        - multiple objects can be passed as list (see examples)
    backend :         'auto' | 'vispy' | 'plotly', default='auto'
                        - ``auto`` selects backend based on context: ``vispy``
                          for terminal and ``plotly`` for Jupyter notebooks.
                        - ``vispy`` uses OpenGL to generate high-performance
                          3D plots. Works in terminals.
                        - ``plotly`` generates 3D plots using WebGL. Works in
                          Jupyter notebooks.
    connectors :      bool, default=False
                      Plot connectors (e.g. synapses) if available.
    by_strahler :     bool, default=False
                      Will shade neuron(s) by strahler index.
    by_confidence :   bool, default=False
                      Will shade neuron(s) by arbor confidence.
    cn_mesh_colors :  bool, default=False
                      Plot connectors using mesh colors.
    clear3d :         bool, default=False
                      If True, canvas is cleared before plotting (only for
                      vispy).
    color :           None | str | tuple | list | dict, default=None (random)
                      Use single str (e.g. ``'red'``) or ``(r, g, b)`` tuple
                      to give all neurons the same color. Use ``list`` of
                      colors to assign colors: ``['red', (1, 0, 1), ...].
                      Use ``dict`` to map colors to neurons:
                      ``{skid: (r, g, b), ...}``. RGB must be 0-255.
    use_neuron_color : bool, default=False
                      If True, will try using the ``.color`` attribute of
                      each neuron.
    width/height :    int, default=600
                      Use to define figure/window size.
    title :           str, default=None
                      For plotly only! Change plot title.
    fig_autosize :    bool, default=False
                      For plotly only! Autoscale figure size.
                      Attention: autoscale overrides width and height
    scatter_kws :     dict, optional
                      Use to modify scatter plots. Accepted parameters are
                        - ``size`` to adjust size of dots
                        - ``color`` to adjust color
    plotly_inline :   bool, default=True
                      If True and you are in an Jupyter environment, will
                      render plotly plots inline.

    Returns
    --------
    If ``backend='vispy'``

       Opens a 3D window and returns :class:`navis.Viewer`.

    If ``backend='plotly'``

        Returns either ``None`` if you are in a Jupyter notebook (see
        ``plotly_inline`` parameter) or a ``fig`` dictionary to generate
        plotly 3D figure (see examples).

    See Also
    --------
    :class:`navis.Viewer`
        Interactive vispy 3D viewer. Makes it easy to add/remove/select
        objects.


    Examples
    --------
    In a Jupyter notebook using plotly as backend.

    >>> import plotly.offline
    >>> plotly.offline.init_notebook_mode()
    >>> nl = navis.get_neuron(16)
    >>> # Backend is automatically chosen but we can set it explicitly
    >>> # Plot inline
    >>> nl.plot3d(backend='plotly')
    >>> # Plot as separate html in a new window
    >>> fig = nl.plot3d(backend='plotly', plotly_inline=False)
    >>> plotly.offline.plot(fig)

    In a terminal using vispy as backend.

    >>> # Plot single neuron
    >>> nl = navis.get_neuron(16)
    >>> v = navis.plot3d(nl, backend='vispy')
    >>> # Clear canvas
    >>> navis.clear3d()

    Some more advanced examples (using vispy here but also works with plotly).

    >>> # plot3d() can deal with combinations of objects
    >>> nl2 = navis.get_neuron('annotation:glomerulus DA1')
    >>> vol = navis.get_volume('v13.LH_R')
    >>> vol.color = (255, 0, 0, .5)
    >>> # This plots two neuronlists, two volumes and a single neuron
    >>> navis.plot3d([nl1, nl2, vol, 'v13.AL_R', 233007])
    >>> # Pass kwargs
    >>> navis.plot3d(nl1, connectors=True, clear3d=True)

    """

    """
    TO-DO:
    - allow generic "plot_{}" arguments:
        - e.g. plot_nodes (default=True) or plot_connectors
        - either autodetect how to plot stuff (x, y, z required)
        - accept "{}_layout" parameter:
            {'size': either int/float or str (str=column)
             'as': 'scatter' | 'lineplot',
             'color': either color | str (column name)
             'colormap': if color is column, map to

              }
    """

    # Backend
    backend = kwargs.pop('backend', 'auto')
    allowed_backends = ['auto', 'vispy', 'plotly']
    if backend.lower() == 'auto':
        if utils.is_jupyter():
            backend = 'plotly'
        else:
            backend = 'vispy'
    elif backend.lower() not in allowed_backends:
        raise ValueError('Unknown backend "{}". '
                         'Permitted: {}.'.format(','.join(backend)))

    if backend == 'vispy':
        return plot3d_vispy(x, **kwargs)
    else:
        return plot3d_plotly(x, **kwargs)


def plot3d_vispy(x, **kwargs):
    """ Plot3d() helper function to generate vispy 3D plots. This is just to
    improve readability. It's only purpose is to find the existing viewer
    or generate a new one.
    """

    # Parse objects to plot
    skdata, dotprops, volumes, points, visual = utils.parse_objects(x)

    # Check for allowed static parameters
    allowed = {'color', 'colors', 'by_strahler', 'by_confidence',
               'cn_mesh_colors', 'linewidth', 'scatter_kws', 'synapse_layout',
               'dps_scale_vec', 'title', 'width', 'height', 'fig_autosize',
               'auto_limits', 'autolimits', 'plotly_inline', 'viewer',
               'clear', 'clear3d'}

    # Check if any of these parameters are dynamic (i.e. attached data tables)
    notallowed = set(kwargs.keys()) - allowed

    # Parameters for neurons
    color = kwargs.get('color',
                       kwargs.get('c',
                                  kwargs.get('colors', None)))
    connectors = kwargs.get('connectors', False)
    by_strahler = kwargs.get('by_strahler', False)
    by_confidence = kwargs.get('by_confidence', False)
    cn_mesh_colors = kwargs.get('cn_mesh_colors', False)
    linewidth = kwargs.get('linewidth', 2)
    connectors_only = kwargs.get('connectors_only', False)
    scatter_kws = kwargs.pop('scatter_kws', {})
    syn_lay_new = kwargs.get('synapse_layout', {})
    syn_lay = {0: {'name': 'Presynapses',
                   'color': (255, 0, 0)},
               1: {'name': 'Postsynapses',
                   'color': (0, 0, 255)},
               2: {'name': 'Gap junctions',
                   'color': (0, 255, 0)},
               'display': 'lines'  # 'circles'
               }
    syn_lay.update(syn_lay_new)

    # Parameters for dotprops
    scale_vect = kwargs.get('dps_scale_vec', 1)

    # Parameters for figure
    pl_title = kwargs.get('title', None)
    width = kwargs.get('width', 1000)
    height = kwargs.get('height', 600)
    fig_autosize = kwargs.get('fig_autosize', False)
    auto_limits = kwargs.get('auto_limits', True)
    auto_limits = kwargs.get('autolimits', auto_limits)
    plotly_inline = kwargs.get('plotly_inline', True)

    if 'viewer' not in kwargs:
        # If does not exists yet, initialise a canvas object and make global
        if not getattr(config, 'primary_viewer', None):
            viewer = config.primary_viewer = Viewer()
        else:
            viewer = getattr(config, 'primary_viewer', None)
    else:
        viewer = kwargs.pop('viewer', getattr(config, 'primary_viewer'))

    # Make sure viewer is visible
    viewer.show()

    if kwargs.get('clear3d', False) or kwargs.get('clear', False):
        viewer.clear()

    if skdata:
        viewer.add(skdata, **kwargs)
    if not dotprops.empty:
        viewer.add(dotprops, **kwargs)
    if volumes:
        viewer.add(volumes, **kwargs)
    if points:
        viewer.add(points, scatter_kws=scatter_kws)

    return viewer


def plot3d_plotly(x, **kwargs):
    """
    Plot3d() helper function to generate plotly 3D plots. This is just to
    improve readability and structure of the code.
    """

    # Parse objects to plot
    skdata, dotprops, volumes, points, visual = utils.parse_objects(x)

    trace_data = []

    if skdata:
        trace_data += neuron2plotly(skdata, **kwargs)
    if not dotprops.empty:
        trace_data += dotprops2plotly(dotprops, **kwargs)
    if volumes:
        trace_data += volume2plotly(volumes, **kwargs)
    if points:
        trace_data += scatter2plotly(points,
                                     scatter_kws=kwargs.get('scatter_kws', {}))

    layout = layout2plotly(**kwargs)

    fig = dict(data=trace_data, layout=layout)

    if kwargs.get('plotly_inline', True) and utils.is_jupyter():
        plotly.offline.iplot(fig)
        return
    else:
        logger.info('Use plotly.offline.plot(fig, filename="3d_plot.html")'
                    ' to plot. Optimized for Google Chrome.')
        return fig

"""
def plot3d_plotly(x, **kwargs):

    # Parse objects to plot
    skdata, dotprops, volumes, points, visual = utils.parse_objects(x)

    # Generate sphere for somas
    fib_points = _fibonacci_sphere(samples=30)

    # Generate the colormaps
    neuron_cmap, dotprop_cmap = prepare_colormap(kwargs.get('color',
                                                            kwargs.get('colors', None)),,
                                                 skdata, dotprops,
                                                 use_neuron_color=kwargs.get('use_neuron_color', False),
                                                 color_range=255)

    trace_data = []
    for i, neuron in enumerate(skdata.itertuples()):
        neuron_name = str(neuron.uuid)
        skid = neuron.uuid

        if not connectors_only:
            if by_strahler:
                s_index = morpho.strahler_index(neuron, return_dict=True)

            soma = neuron.nodes[neuron.nodes.radius > 1]

            coords = _segments_to_coords(
                neuron, neuron.segments, modifier=(-1, -1, -1))

            # We have to add (None,None,None) to the end of each slab to
            # make that line discontinuous there
            coords = np.vstack(
                [np.append(t, [[None] * 3], axis=0) for t in coords])

            if by_strahler:
                c = []
                for k, s in enumerate(coords):
                    this_c = 'rgba(%i,%i,%i,%f)' % (neuron_cmap[i][0],
                                                    neuron_cmap[i][1],
                                                    neuron_cmap[i][2],
                                                    s_index[s[0]] / max(s_index.values()))
                    # Slabs are separated by a <None> coordinate -> this is
                    # why we need one more color entry
                    c += [this_c] * (len(s) + 1)
            else:
                try:
                    c = 'rgb{}'.format(neuron_cmap[i])
                except BaseException:
                    c = 'rgb(10,10,10)'

            trace_data.append(go.Scatter3d(x=coords[:, 0],
                                           # y and z are switched
                                           y=coords[:, 2],
                                           z=coords[:, 1],
                                           mode='lines',
                                           line=dict(color=c,
                                                     width=linewidth),
                                           name=neuron_name,
                                           legendgroup=neuron_name,
                                           showlegend=True,
                                           hoverinfo='none'
                                           ))

            # Add soma(s):
            for n in soma.itertuples():
                try:
                    c = 'rgb{}'.format(neuron_cmap[i])
                except BaseException:
                    c = 'rgb(10,10,10)'
                trace_data.append(go.Mesh3d(
                    x=[(v[0] * n.radius / 2) - n.x for v in fib_points],
                    # y and z are switched
                    y=[(v[1] * n.radius / 2) - n.z for v in fib_points],
                    z=[(v[2] * n.radius / 2) - n.y for v in fib_points],

                    alphahull=.5,
                    color=c,
                    name=neuron_name,
                    legendgroup=neuron_name,
                    showlegend=False,
                    hoverinfo='name'))

        if connectors or connectors_only:
            for j in [0, 1, 2]:
                if cn_mesh_colors:
                    try:
                        c = neuron_cmap[i]
                    except BaseException:
                        c = (10, 10, 10)
                else:
                    c = syn_lay[j]['color']

                this_cn = neuron.connectors[
                    neuron.connectors.relation == j]

                if syn_lay['display'] == 'circles':
                    trace_data.append(go.Scatter3d(
                        x=this_cn.x.values * -1,
                        y=this_cn.z.values * -1,  # y and z are switched
                        z=this_cn.y.values * -1,
                        mode='markers',
                        marker=dict(
                            color='rgb%s' % str(c),
                            size=2
                        ),
                        name=syn_lay[j]['name'] + ' of ' + neuron_name,
                        showlegend=True,
                        hoverinfo='none'
                    ))
                elif syn_lay['display'] == 'lines':
                    # Find associated treenode
                    tn = neuron.nodes.set_index('node_id').ix[this_cn.node_id.values]
                    x_coords = [n for sublist in zip(this_cn.x.values * -1, tn.x.values * -1, [None] * this_cn.shape[0]) for n in sublist]
                    y_coords = [n for sublist in zip(this_cn.y.values * -1, tn.y.values * -1, [None] * this_cn.shape[0]) for n in sublist]
                    z_coords = [n for sublist in zip(this_cn.z.values * -1, tn.z.values * -1, [None] * this_cn.shape[0]) for n in sublist]

                    trace_data.append(go.Scatter3d(
                        x=x_coords,
                        y=z_coords,  # y and z are switched
                        z=y_coords,
                        mode='lines',
                        line=dict(
                            color='rgb%s' % str(c),
                            width=5
                        ),
                        name=syn_lay[j]['name'] + ' of ' + neuron_name,
                        showlegend=True,
                        hoverinfo='none'
                    ))

    for i, neuron in enumerate(dotprops.itertuples()):
        # Prepare lines - this is based on nat:::plot3d.dotprops
        halfvect = neuron.points[
            ['x_vec', 'y_vec', 'z_vec']] / 2 * scale_vect

        starts = neuron.points[['x', 'y', 'z']
                               ].values - halfvect.values
        ends = neuron.points[['x', 'y', 'z']
                             ].values + halfvect.values

        x_coords = [n for sublist in zip(
            starts[:, 0] * -1, ends[:, 0] * -1, [None] * starts.shape[0]) for n in sublist]
        y_coords = [n for sublist in zip(
            starts[:, 1] * -1, ends[:, 1] * -1, [None] * starts.shape[0]) for n in sublist]
        z_coords = [n for sublist in zip(
            starts[:, 2] * -1, ends[:, 2] * -1, [None] * starts.shape[0]) for n in sublist]

        try:
            c = 'rgb{}'.format(dotprop_cmap[i])
        except BaseException:
            c = 'rgb(10,10,10)'

        trace_data.append(go.Scatter3d(x=x_coords, y=z_coords, z=y_coords,
                                       mode='lines',
                                       line=dict(
                                           color=c,
                                           width=5
                                       ),
                                       name=neuron.gene_name,
                                       legendgroup=neuron.gene_name,
                                       showlegend=True,
                                       hoverinfo='none'
                                       ))

        # Add soma
        rad = 4
        trace_data.append(go.Mesh3d(
            x=[(v[0] * rad / 2) - neuron.X for v in fib_points],
            # y and z are switched
            y=[(v[1] * rad / 2) - neuron.Z for v in fib_points],
            z=[(v[2] * rad / 2) - neuron.Y for v in fib_points],

            alphahull=.5,

            color=c,
            name=neuron.gene_name,
            legendgroup=neuron.gene_name,
            showlegend=False,
            hoverinfo='name'
        )
        )

    # Now add neuropils:
    for v in volumes_data:
        # Skip empty data
        if isinstance(volumes_data[v]['verts'], np.ndarray):
            if not volumes_data[v]['verts'].any():
                continue
        elif not volumes_data[v]['verts']:
            continue
        trace_data.append(go.Mesh3d(
            x=[-v[0] for v in volumes_data[v]['verts']],
            # y and z are switched
            y=[-v[2] for v in volumes_data[v]['verts']],
            z=[-v[1] for v in volumes_data[v]['verts']],

            i=[f[0] for f in volumes_data[v]['faces']],
            j=[f[1] for f in volumes_data[v]['faces']],
            k=[f[2] for f in volumes_data[v]['faces']],

            opacity=.5,
            color='rgb' + str(volumes_data[v]['color']),
            name=v,
            showlegend=True,
            hoverinfo='none'
        )
        )

    # Add scatter plots
    for p in points:
        trace_data.append(go.Scatter3d(x=-p[:, 0],
                                       y=-p[:, 2], # Z and Y are swapped
                                       z=-p[:, 1],
                                       mode='markers',
                                       marker=dict(
                                                   size=scatter_kws.get('size',3),
                                                   color='rgb' + str(scatter_kws.get('color',(0, 0, 0))),
                                                   opacity=scatter_kws.get('opacity',1))
                                       )
                         )

    layout = dict(
        width=width,
        height=height,
        autosize=fig_autosize,
        title=pl_title,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(240, 240, 240)',
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(240, 240, 240)',
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(240, 240, 240)',
            ),
            camera=dict(up=dict(x=0,
                                y=0,
                                z=1
                                ),
                        eye=dict(x=-1.7428,
                                 y=1.0707,
                                 z=0.7100,
                                 )
                        ),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='data'
        ),
    )

    # Need to remove width and height to make autosize actually matter
    if fig_autosize:
        layout.pop('width')
        layout.pop('height')

    fig = dict(data=trace_data, layout=layout)

    logger.debug('Done. Plotted %i nodes and %i connectors' % (sum([n.nodes.shape[0] for n in skdata.itertuples() if not connectors_only] + [
        n.points.shape[0] for n in dotprops.itertuples()]), sum([n.connectors.shape[0] for n in skdata.itertuples() if connectors or connectors_only])))

    if plotly_inline and utils.is_jupyter():
        plotly.offline.iplot(fig)
        return
    else:
        logger.info('Use plotly.offline.plot(fig, filename="3d_plot.html")'
                    ' to plot. Optimized for Google Chrome.')
        return fig


    skdata, dotprops, volumes, points, visual = utils.parse_objects(x)



    # Parameters for neurons
    color = kwargs.get('color',
                       kwargs.get('c',
                                  kwargs.get('colors', None)))
    downsampling = kwargs.get('downsampling', 1)
    connectors = kwargs.get('connectors', False)
    by_strahler = kwargs.get('by_strahler', False)
    by_confidence = kwargs.get('by_confidence', False)
    cn_mesh_colors = kwargs.get('cn_mesh_colors', False)
    linewidth = kwargs.get('linewidth', 2)
    connectors_only = kwargs.get('connectors_only', False)
    use_neuron_color = kwargs.get('use_neuron_color', False)
    scatter_kws = kwargs.get('scatter_kws', {})
    syn_lay_new = kwargs.get('synapse_layout', {})
    syn_lay = {0: {'name': 'Presynapses',
                   'color': (255, 0, 0)},
               1: {'name': 'Postsynapses',
                   'color': (0, 0, 255)},
               2: {'name': 'Gap junctions',
                   'color': (0, 255, 0)},
               'display': 'lines'  # 'circles'
               }
    syn_lay.update(syn_lay_new)

    # Parameters for dotprops
    scale_vect = kwargs.get('scale_vect', 1)

    # Parameters for figure
    pl_title = kwargs.get('title', None)
    width = kwargs.get('width', 1000)
    height = kwargs.get('height', 600)
    fig_autosize = kwargs.get('fig_autosize', False)
    auto_limits = kwargs.get('auto_limits', True)
    auto_limits = kwargs.get('autolimits', auto_limits)
    plotly_inline = kwargs.get('plotly_inline', True)


    # Prepare volumes
    volumes_data = {}
    for v in volumes:
        volumes_data[v.name] = {'verts': v.vertices,
                                'faces': v.faces,
                                'color': v.color}

    # First downsample neurons
    if downsampling > 1 and not connectors_only and not skdata.empty:
        logger.setLevel('ERROR')
        skdata.downsample(downsampling, inplace=True)
        logger.setLevel('INFO')

    if backend == 'plotly':
        return _plot3d_plotly()
    else:
        return _plot3d_vispy()
"""
