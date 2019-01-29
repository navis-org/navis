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

import plotly.graph_objs as go
import plotly.offline

from .. import config

from .colors import *
from .plot_utils import *

__all__ = ['plot3d_plotly', 'plot2d', 'plot1d']

# Generate sphere for somas
fib_points = fibonacci_sphere(samples=30)


def plot3d_plotly():
    """
    Plot3d() helper function to generate plotly 3D plots. This is just to
    improve readability and structure of the code.
    """
    trace_data = []

    # Generate the colormaps
    neuron_cmap, dotprop_cmap = _prepare_colormap(color,
                                                  skdata, dotprops,
                                                  use_neuron_color=use_neuron_color,
                                                  color_range=255)

    fig = dict(data=trace_data, layout=layout)

    if plotly_inline and utils.is_jupyter():
        plotly.offline.iplot(fig)
        return
    else:
        logger.info('Use plotly.offline.plot(fig, filename="3d_plot.html")'
                    ' to plot. Optimized for Google Chrome.')
        return fig


def neuron2plotly(x, **kwargs):
    """ Converts TreeNeuron to plotly objects."""

    name = str(getattr(x.name, None))
    color = kwargs.get('color', config.default_color)
    linewidth = kwargs.get('linewidth', 1)

    if not kwargs.get('connectors_only', False):
        coords = segments_to_coords(x, x.segments)

        # We have to add (None,None,None) to the end of each slab to
        # make that line discontinuous there
        coords = np.vstack([np.append(t, [[None] * 3], axis=0) for t in coords])

        if kwargs.get('by_strahler', False):
            s_index = morpho.strahler_index(x, return_dict=True)
            c = []
            for k, s in enumerate(coords):
                this_c = 'rgba(%i,%i,%i,%f)' % (color[0],
                                                color[1],
                                                color[2],
                                                s_index[s[0]] / max(s_index.values()))
                # Slabs are separated by a <None> coordinate -> this is
                # why we need one more color entry
                c += [this_c] * (len(s) + 1)
        else:
            try:
                c = 'rgb{}'.format(color)
            except BaseException:
                c = 'rgb(10,10,10)'

        trace_data.append(go.Scatter3d(x=coords[:, 0],
                                       y=coords[:, 1],
                                       z=coords[:, 2],
                                       mode='lines',
                                       line=dict(color=c,
                                                 width=linewidth),
                                       name=name,
                                       legendgroup=name,
                                       showlegend=True,
                                       hoverinfo='none'
                                       ))

        # Add soma(s):
        if not isinstance(x.soma, type(None)):
            for s in x.soma:
                n = x.nodes.set_index('node_id').loc[s]
                try:
                    c = 'rgb{}'.format(color)
                except BaseException:
                    c = 'rgb(10,10,10)'
                trace_data.append(go.Mesh3d(
                    x=[(v[0] * n.radius / 2) - n.x for v in fib_points],
                    # y and z are switched
                    y=[(v[1] * n.radius / 2) - n.y for v in fib_points],
                    z=[(v[2] * n.radius / 2) - n.z for v in fib_points],

                    alphahull=.5,
                    color=c,
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    hoverinfo='name'))

    return trace_data


def scatter2plotly(x, **kwargs):
    """ Converts DataFrame with x,y,z columns to plotly scatter plot."""

    c = kwargs.get('color', (10, 10, 10))
    s = kwargs.get('size', 2)
    name = kwargs.get('name', None)

    trace_data = []
    trace_data.append(go.Scatter3d(
                                    x=x.x.values,
                                    y=x.y.values,
                                    z=x.z.values,
                                    mode='markers',
                                    marker=dict(
                                                color='rgb%s' % str(c),
                                                size=s
                                    ),
                                    name=name,
                                    showlegend=True,
                                    hoverinfo='none'
                                ))
    return trace_data


def lines2plotly(x, **kwargs):
    """ Convert DataFrame with x, y, z, x1, y2, y3 columns to line plots"""

    s = kwargs.get('size', 2)
    name = kwargs.get('name', None)
    c = kwargs.get('color', (10, 10, 10))

    x_coords = [n for sublist in zip(this_cn.x.values * -1, tn.x.values * -1, [None] * this_cn.shape[0]) for n in sublist]
    y_coords = [n for sublist in zip(this_cn.y.values * -1, tn.y.values * -1, [None] * this_cn.shape[0]) for n in sublist]
    z_coords = [n for sublist in zip(this_cn.z.values * -1, tn.z.values * -1, [None] * this_cn.shape[0]) for n in sublist]

    trace_data = []
    trace_data.append(go.Scatter3d(
                                    x=x_coords,
                                    y=y_coords,  # y and z are switched
                                    z=z_coords,
                                    mode='lines',
                                    line=dict(
                                        color='rgb%s' % str(c),
                                        width=5
                                    ),
                                    name=syn_lay[j]['name'] + ' of ' + neuron_name,
                                    showlegend=True,
                                    hoverinfo='none'
                    ))

    return trace_data


def dotprops2plotly(x, **kwargs):
    """ Converts dotprops to plotly objects."""

    linewidth = kwargs.get('linewidth', 5)
    name = getattr(x, 'name', getattr(x, 'gene_name', None))
    c = kwargs.get('color', (10, 10, 10))

    # Prepare lines - this is based on nat:::plot3d.dotprops
    halfvect = x.points[['x_vec', 'y_vec', 'z_vec']] / 2 * scale_vect

    starts = x.points[['x', 'y', 'z']].values - halfvect.values
    ends = x.points[['x', 'y', 'z']].values + halfvect.values

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
                                       width=linewidth
                                   ),
                                   name=name,
                                   legendgroup=name,
                                   showlegend=True,
                                   hoverinfo='none'
                                   ))

    # Add soma
    rad = 4
    trace_data.append(go.Mesh3d(
        x=[(v[0] * rad / 2) - x.X for v in fib_points],
        # y and z are switched
        y=[(v[1] * rad / 2) - x.Z for v in fib_points],
        z=[(v[2] * rad / 2) - x.Y for v in fib_points],

        alphahull=.5,

        color=c,
        name=name,
        legendgroup=name,
        showlegend=False,
        hoverinfo='name'
    ))

    return trace_data


def volume2plotly(x, **kwargs):
    """ Convert Volumes to plotly objects. """

    name = getattr(x, 'name', None)
    c = kwargs.get('color', (.5, .5, .5))

    # Skip empty data
    if isinstance(x.vertices, np.ndarray):
        if not x.vertices.any():
            continue
    elif not x.vertices:
        continue

    trace_data = []
    trace_data.append(go.Mesh3d(
                                x = v.vertices[:, 0],
                                y = v.vertices[:, 1],
                                z = v.vertices[:, 2],
                                i = v.faces[:, 0],
                                j = v.faces[:, 1],
                                k = v.faces[:, 2],

                                opacity = .5,
                                color = 'rgb' + str(c),
                                name = name,
                                showlegend = True,
                                hoverinfo = 'none'
                            ))

    return trace_data


def layout2plotly(x, **kwargs):
    """ Generate layout for plotly figures."""
    layout = dict(
                width=kwargs.get('width', 1000),
                height=kwargs.get('height', 600),
                autosize=kwargs.get('fig_autosize', False),
                title=kwargs.get('pl_title', None),
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
    if kwargs.get('fig_autosize', False):
        layout.pop('width')
        layout.pop('height')

    return layout
