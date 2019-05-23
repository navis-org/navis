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
import pandas as pd

import plotly.graph_objs as go

from ..colors import *
from ..plot_utils import *
from ... import core

__all__ = ['neuron2plotly', 'scatter2plotly', 'dotprops2plotly',
           'volume2plotly', 'layout2plotly']

# Generate sphere for somas
fib_points = fibonacci_sphere(samples=30)


def neuron2plotly(x, **kwargs):
    """ Converts TreeNeurons to plotly objects."""

    if isinstance(x, core.TreeNeuron):
        x = core.NeuronList(x)
    elif isinstance(x, core.NeuronList):
        pass
    else:
        raise TypeError('Unable to process data of type "{}"'.format(type(x)))

    colors = kwargs.get('color',
                        kwargs.get('c',
                                   kwargs.get('colors', None)))

    colormap, _, _ = prepare_colormap(colors,
                                      x, None,
                                      use_neuron_color=kwargs.get('use_neuron_color', False),
                                      color_range=255)

    linewidth = kwargs.get('linewidth', 1)

    syn_lay = {
        0: {
            'name': 'Presynapses',
            'color': (1, 0, 0)
        },
        1: {
            'name': 'Postsynapses',
            'color': (0, 0, 1)
        },
        2: {
            'name': 'Gap junctions',
            'color': (0, 1, 0)
        },
        'display': 'lines'  # 'circle'
    }
    syn_lay.update(kwargs.get('synapse_layout', {}))

    trace_data = []
    for i, neuron in enumerate(x):
        color = colormap[i]
        name = str(getattr(neuron, 'name', None))

        if not kwargs.get('connectors_only', False):
            coords = segments_to_coords(neuron, neuron.segments)

            # We have to add (None, None, None) to the end of each slab to
            # make that line discontinuous there
            coords = np.vstack([np.append(t, [[None] * 3], axis=0) for t in coords])

            if kwargs.get('by_strahler', False):
                s_index = morpho.strahler_index(neuron, return_dict=True)
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
            if not isinstance(neuron.soma, type(None)):
                for s in neuron.soma:
                    n = neuron.nodes.set_index('node_id').loc[s]
                    try:
                        c = 'rgb{}'.format(color)
                    except BaseException:
                        c = 'rgb(10,10,10)'
                    trace_data.append(go.Mesh3d(
                        x=[(v[0] * n.radius) + n.x for v in fib_points],
                        # y and z are switched
                        y=[(v[1] * n.radius) + n.y for v in fib_points],
                        z=[(v[2] * n.radius) + n.z for v in fib_points],

                        alphahull=.5,
                        color=c,
                        name=name,
                        hoverlabel='name'))

        # Add connectors
        if kwargs.get('connectors', False) \
        or kwargs.get('connectors_only', False):
            for j in neuron.connectors.relation.unique():
                if cn_mesh_colors:
                    c = color
                else:
                    c = syn_lay.get(j, {'color': (10, 10, 10)})['color']

                this_cn = neuron.connectors[neuron.connectors.relation == j]

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
                        name=syn_lay.get(j, {'name': 'connector'})['name'] + ' of ' + name,
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
                        name=syn_lay.get(j, {'name': 'connector'})['name'] + ' of ' + name,
                        showlegend=True,
                        hoverinfo='none'
                    ))
                else:
                    raise ValueError('Unknown display type for connectors "{}"'.format(syn_lay['display']))

    return trace_data


def scatter2plotly(x, **kwargs):
    """ Converts DataFrame with x,y,z columns to plotly scatter plot."""

    c = eval_colors(kwargs.get('color', (10, 10, 10)), color_range=255)
    s = kwargs.get('size', 2)
    name = kwargs.get('name', None)

    trace_data = []
    for scatter in x:
        if isinstance(scatter, pd.DataFrame):
            if not all([c in scatter.columns for c in ['x', 'y', 'z']]):
                raise ValueError('DataFrame must have x, y and z columns')
            scatter = [['x', 'y', 'z']].values

        if not isinstance(scatter, np.ndarray):
            scatter = np.array(scatter)

        trace_data.append(go.Scatter3d(x=scatter[:, 0],
                                       y=scatter[:, 1],
                                       z=scatter[:, 2],
                                       mode='markers',
                                       marker=dict(color='rgb%s' % str(c),
                                                   size=s,
                                                   opacity=kwargs.get('opacity', 1)),
                                       name=name,
                                       showlegend=True,
                                       hoverinfo='none'))
    return trace_data


def lines2plotly(x, **kwargs):
    """ Convert DataFrame with x, y, z, x1, y2, y3 columns to line plots"""

    s = kwargs.get('size', 2)
    name = kwargs.get('name', None)
    c = kwargs.get('color', (10, 10, 10))

    x_coords = [n for sublist in zip(this_cn.x.values * -1, tn.x.values * -1,
                                [None] * this_cn.shape[0]) for n in sublist]
    y_coords = [n for sublist in zip(this_cn.y.values * -1, tn.y.values * -1,
                                [None] * this_cn.shape[0]) for n in sublist]
    z_coords = [n for sublist in zip(this_cn.z.values * -1, tn.z.values * -1,
                                [None] * this_cn.shape[0]) for n in sublist]

    trace_data = []
    trace_data.append(go.Scatter3d(x=x_coords,
                                   y=y_coords,  # y and z are switched
                                   z=z_coords,
                                   mode='lines',
                                   line=dict(
                                        color='rgb%s' % str(c),
                                        width=5
                                   ),
                                   name=syn_lay[j]['name'] + ' of ' + neuron_name,
                                   showlegend=True,
                                   hoverinfo='none'))

    return trace_data


def dotprops2plotly(x, **kwargs):
    """ Converts dotprops to plotly objects."""

    linewidth = kwargs.get('linewidth', 5)

    colors = kwargs.get('color',
                        kwargs.get('c',
                                   kwargs.get('colors', None)))

    _, colormap, _ = prepare_colormap(colors,
                                      x, None,
                                      use_neuron_color=kwargs.get('use_neuron_color', False),
                                      color_range=255)

    for i, dp in enumerate(x):
        # Get Name
        name = getattr(dp, 'name', getattr(dp, 'gene_name', None))
        c = colormap[i]

        # Prepare lines - this is based on nat:::plot3d.dotprops
        halfvect = dp.points[['x_vec', 'y_vec', 'z_vec']] / 2 * scale_vect

        starts = dp.points[['x', 'y', 'z']].values - halfvect.values
        ends = dp.points[['x', 'y', 'z']].values + halfvect.values

        x_coords = [n for sublist in zip(
            starts[:, 0] * -1, ends[:, 0] * -1, [None] * starts.shape[0]) for n in sublist]
        y_coords = [n for sublist in zip(
            starts[:, 1] * -1, ends[:, 1] * -1, [None] * starts.shape[0]) for n in sublist]
        z_coords = [n for sublist in zip(
            starts[:, 2] * -1, ends[:, 2] * -1, [None] * starts.shape[0]) for n in sublist]

        try:
            c = 'rgb{}'.format(c)
        except BaseException:
            c = 'rgb(10,10,10)'

        trace_data.append(go.Scatter3d(x=x_coords,
                                       y=z_coords,
                                       z=y_coords,
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

    return trace_data


def volume2plotly(x, **kwargs):
    """ Convert Volumes to plotly objects. """

    colors = kwargs.get('color',
                        kwargs.get('c',
                                   kwargs.get('colors', None)))

    _, _, colormap = prepare_colormap(colors,
                                      x, None,
                                      use_neuron_color=kwargs.get('use_neuron_color', False),
                                      color_range=255)

    trace_data = []
    for i, v in enumerate(x):
        name = getattr(v, 'name', None)
        c = colormap[i]
        # Skip empty data
        if isinstance(v.vertices, np.ndarray):
            if not v.vertices.any():
                continue
        elif not v.vertices:
            continue

        trace_data.append(go.Mesh3d(x=v.vertices[:, 0],
                                    y=v.vertices[:, 1],
                                    z=v.vertices[:, 2],
                                    i=v.faces[:, 0],
                                    j=v.faces[:, 1],
                                    k=v.faces[:, 2],

                                    opacity=.5,
                                    color='rgb' + str(c),
                                    name=name,
                                    hoverinfo='none'))

    return trace_data


def layout2plotly(**kwargs):
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
