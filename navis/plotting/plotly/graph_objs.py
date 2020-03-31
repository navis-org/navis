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
from ... import core, utils

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
            soma = utils.make_iterable(neuron.soma)
            if any(soma):
                for s in soma:
                    n = neuron.nodes.set_index('node_id').loc[s]
                    r = getattr(n, neuron.soma_radius) if isinstance(neuron.soma_radius, str) else neuron.soma_radius
                    try:
                        c = 'rgb{}'.format(color)
                    except BaseException:
                        c = 'rgb(10,10,10)'
                    trace_data.append(go.Mesh3d(
                        x=[(v[0] * r) + n.x for v in fib_points],
                        # y and z are switched
                        y=[(v[1] * r) + n.y for v in fib_points],
                        z=[(v[2] * r) + n.z for v in fib_points],

                        alphahull=.5,
                        color=c,
                        name=name,
                        hoverinfo='name'))

        # Add connectors
        if kwargs.get('connectors', False) or \
           kwargs.get('connectors_only', False):
            for j in neuron.connectors.type.unique():
                if kwargs.get('cn_mesh_colors', False):
                    c = color
                else:
                    c = syn_lay.get(j, {'color': (10, 10, 10)})['color']

                this_cn = neuron.connectors[neuron.connectors.type == j]

                if syn_lay['display'] == 'circles':
                    trace_data.append(go.Scatter3d(
                        x=this_cn.x.values,
                        y=this_cn.zy.values,
                        z=this_cn.z.values,
                        mode='markers',
                        marker=dict(color='rgb%s' % str(c), size=2),
                        name=syn_lay.get(j, {'name': 'connector'})['name'] + ' of ' + name,
                        showlegend=True,
                        hoverinfo='none'
                    ))
                elif syn_lay['display'] == 'lines':
                    # Find associated treenode
                    tn = neuron.nodes.set_index('node_id').loc[this_cn.node_id.values]
                    x_coords = [n for sublist in zip(this_cn.x.values, tn.x.values, [None] * this_cn.shape[0]) for n in sublist]
                    y_coords = [n for sublist in zip(this_cn.y.values, tn.y.values, [None] * this_cn.shape[0]) for n in sublist]
                    z_coords = [n for sublist in zip(this_cn.z.values, tn.z.values, [None] * this_cn.shape[0]) for n in sublist]

                    trace_data.append(go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
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

    c = eval_color(kwargs.get('color', (10, 10, 10)), color_range=255)
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
    """Convert DataFrame with x, y, z, x1, y1, z1 columns to line plots."""
    name = kwargs.get('name', None)
    c = kwargs.get('color', (10, 10, 10))

    x_coords = [n for sublist in zip(x.x.values, x.x1.values,
                                     [None] * x.shape[0]) for n in sublist]
    y_coords = [n for sublist in zip(x.y.values, x.y1.values,
                                     [None] * x.shape[0]) for n in sublist]
    z_coords = [n for sublist in zip(x.z.values, x.z1.values,
                                     [None] * x.shape[0]) for n in sublist]

    trace_data = []
    trace_data.append(go.Scatter3d(x=x_coords,
                                   y=y_coords,
                                   z=z_coords,
                                   mode='lines',
                                   line=dict(
                                        color=f'rgb{str(c)}',
                                        width=5
                                   ),
                                   name=name,
                                   showlegend=True,
                                   hoverinfo='none'))

    return trace_data


def dotprops2plotly(x, **kwargs):
    """ Converts dotprops to plotly objects."""

    linewidth = kwargs.get('linewidth', 5)

    colors = kwargs.get('color',
                        kwargs.get('c',
                                   kwargs.get('colors', None)))

    scale_vect = kwargs.get('dps_scale_vec', 1)

    _, colormap, _ = prepare_colormap(colors,
                                      dotprops=x,
                                      use_neuron_color=kwargs.get('use_neuron_color', False),
                                      color_range=255)

    trace_data = []
    for i, dp in enumerate(x.itertuples()):
        # Get Name
        name = getattr(dp, 'name', getattr(dp, 'gene_name', None))
        c = colormap[i]

        # Prepare lines - this is based on nat:::plot3d.dotprops
        halfvect = dp.points[['x_vec', 'y_vec', 'z_vec']] / 2 * scale_vect

        starts = dp.points[['x', 'y', 'z']].values - halfvect.values
        ends = dp.points[['x', 'y', 'z']].values + halfvect.values

        x_coords = [n for sublist in zip(
            starts[:, 0], ends[:, 0], [None] * starts.shape[0]) for n in sublist]
        y_coords = [n for sublist in zip(
            starts[:, 1], ends[:, 1], [None] * starts.shape[0]) for n in sublist]
        z_coords = [n for sublist in zip(
            starts[:, 2], ends[:, 2], [None] * starts.shape[0]) for n in sublist]

        try:
            c = 'rgb{}'.format(c)
        except BaseException:
            c = 'rgb(10,10,10)'

        trace_data.append(go.Scatter3d(x=x_coords,
                                       y=y_coords,
                                       z=z_coords,
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
                                      volumes=x,
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
    layout = dict(width=kwargs.get('width', 1200),
                  height=kwargs.get('height', 600),
                  autosize=kwargs.get('fig_autosize', False),
                  title=kwargs.get('pl_title', None),
                  scene=dict(xaxis=dict(gridcolor='rgb(255, 255, 255)',
                                        zerolinecolor='rgb(255, 255, 255)',
                                        showbackground=False,
                                        showline=False,
                                        showgrid=False,
                                        backgroundcolor='rgb(240, 240, 240)'
                                        ),
                             yaxis=dict(gridcolor='rgb(255, 255, 255)',
                                        zerolinecolor='rgb(255, 255, 255)',
                                        showbackground=False,
                                        showline=False,
                                        showgrid=False,
                                        backgroundcolor='rgb(240, 240, 240)'
                                        ),
                             zaxis=dict(gridcolor='rgb(255, 255, 255)',
                                        zerolinecolor='rgb(255, 255, 255)',
                                        showbackground=False,
                                        showline=False,
                                        showgrid=False,
                                        backgroundcolor='rgb(240, 240, 240)'
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
