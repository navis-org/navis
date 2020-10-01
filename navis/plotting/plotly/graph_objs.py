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

import uuid

import numpy as np
import pandas as pd

import plotly.graph_objs as go

from ..colors import *
from ..plot_utils import *
from ... import core, utils, config, morpho, conversion

logger = config.logger

__all__ = ['neuron2plotly', 'scatter2plotly', 'dotprops2plotly',
           'volume2plotly', 'layout2plotly']

# Generate sphere for somas
fib_points = fibonacci_sphere(samples=30)


def neuron2plotly(x, **kwargs):
    """Convert neurons to plotly objects."""
    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)
    elif not isinstance(x, core.NeuronList):
        raise TypeError('Unable to process data of type "{}"'.format(type(x)))

    colors = kwargs.pop('color',
                        kwargs.pop('c',
                                   kwargs.pop('colors', None)))

    colormap, _, _ = prepare_colormap(colors,
                                      neurons=x,
                                      alpha=kwargs.get('alpha', None),
                                      use_neuron_color=kwargs.get('use_neuron_color', False),
                                      color_range=255)

    syn_lay = {
        0: {'name': 'Presynapses',
            'color': (255, 0, 0)},
        1: {'name': 'Postsynapses',
            'color': (0, 0, 255)},
        2: {'name': 'Gap junctions',
            'color': (0, 255, 0)},
        'display': 'lines',  # 'circles'
        'size': 2  # for circles only
    }
    syn_lay['pre'] = syn_lay[0]
    syn_lay['post'] = syn_lay[1]
    syn_lay['gap_junction'] = syn_lay['gapjunction'] = syn_lay[2]
    syn_lay.update(kwargs.get('synapse_layout', {}))

    trace_data = []
    for i, neuron in enumerate(x):
        name = str(getattr(neuron, 'name', neuron.id))
        color = colormap[i]

        try:
            # Try converting this neuron's ID
            neuron_id = str(neuron.id)
        except BaseException:
            # If that doesn't work generate a new ID
            neuron_id = str(str(uuid.uuid1()))
        legend_group = kwargs.get('legend_group', neuron_id)

        if not kwargs.get('connectors_only', False):
            if kwargs.get('radius', False):
                neuron = conversion.tree2meshneuron(neuron)

            if isinstance(neuron, core.TreeNeuron):
                trace_data += skeleton2plotly(neuron,
                                              neuron_id=neuron_id,
                                              color=color, **kwargs)
            elif isinstance(neuron, core.MeshNeuron):
                trace_data += mesh2plotly(neuron,
                                          neuron_id=neuron_id,
                                          color=color, **kwargs)
            else:
                raise TypeError(f'Unable to plot neurons of type "{type(neuron)}"')

        # Add connectors
        if kwargs.get('connectors', False) or \
           kwargs.get('connectors_only', False):
            for j in neuron.connectors.type.unique():
                if kwargs.get('cn_mesh_colors', False):
                    c = color
                else:
                    c = syn_lay.get(j, {'color': (10, 10, 10)})['color']

                this_cn = neuron.connectors[neuron.connectors.type == j]

                if syn_lay['display'] == 'circles' or isinstance(neuron, core.MeshNeuron):
                    trace_data.append(go.Scatter3d(
                        x=this_cn.x.values,
                        y=this_cn.y.values,
                        z=this_cn.z.values,
                        mode='markers',
                        marker=dict(color=f'rgb{c}', size=syn_lay.get('size', 2)),
                        name=f'{syn_lay.get(j, {"name": "connector"})["name"]} of {name}',
                        showlegend=False,
                        legendgroup=legend_group,
                        hoverinfo='none'
                    ))
                elif syn_lay['display'] == 'lines':
                    # Find associated treenodes
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
                        name=f'{syn_lay.get(j, {"name": "connector"})["name"]} of {name}',
                        showlegend=False,
                        legendgroup=legend_group,
                        hoverinfo='none'
                    ))
                else:
                    raise ValueError(f'Unknown display type for connectors "{syn_lay["display"]}"')

    return trace_data


def mesh2plotly(neuron, neuron_id, color, **kwargs):
    """Convert MeshNeuron to plotly object."""
    name = str(getattr(neuron, 'name', neuron.id))
    legend_group = kwargs.get('legend_group', neuron_id)

    # Skip empty neurons
    if neuron.n_vertices == 0:
        return []

    try:
        if len(color) == 3:
            c = 'rgb{}'.format(color)
        elif len(color) == 4:
            c = 'rgba{}'.format(color)
    except BaseException:
        c = 'rgb(10,10,10)'

    trace_data = [go.Mesh3d(x=neuron.vertices[:, 0],
                            y=neuron.vertices[:, 1],
                            z=neuron.vertices[:, 2],
                            i=neuron.faces[:, 0],
                            j=neuron.faces[:, 1],
                            k=neuron.faces[:, 2],
                            color=c,
                            name=name,
                            legendgroup=legend_group,
                            showlegend=True,
                            hoverinfo='none')]

    return trace_data


def skeleton2plotly(neuron, neuron_id, color, **kwargs):
    """Convert skeleton (i.e. TreeNeuron) to plotly line plot."""
    coords = segments_to_coords(neuron, neuron.segments)
    name = str(getattr(neuron, 'name', neuron.id))
    linewidth = kwargs.get('linewidth', kwargs.get('lw', 2))
    legend_group = kwargs.get('legend_group', neuron_id)

    # We have to add (None, None, None) to the end of each slab to
    # make that line discontinuous there
    coords = np.vstack([np.append(t, [[None] * 3], axis=0) for t in coords])

    if kwargs.get('by_strahler', False):
        s_index = morpho.strahler_index(neuron, return_dict=True)
        max_strahler = max(s_index.values())
        c = []
        for k, s in enumerate(coords):
            this_c = f'rgba({color[0]},{color[1]},{color[2]},{s_index[s[0]] / max_strahler})'
            # Slabs are separated by a <None> coordinate -> this is
            # why we need one more color entry
            c += [this_c] * (len(s) + 1)
    else:
        try:
            c = 'rgb{}'.format(color)
        except BaseException:
            c = 'rgb(10,10,10)'

    trace_data = [go.Scatter3d(x=coords[:, 0],
                               y=coords[:, 1],
                               z=coords[:, 2],
                               mode='lines',
                               line=dict(color=c,
                                         width=linewidth),
                               name=name,
                               legendgroup=legend_group,
                               showlegend=True,
                               hoverinfo='none'
                               )]

    # Add soma(s):
    soma = utils.make_iterable(neuron.soma)
    if kwargs.get('soma', True) and any(soma):
        # If soma detection is messed up we might end up producing
        # dozens of soma which will freeze the kernel
        if len(soma) >= 5:
            logger.warning(f'{neuron.id}: {len(soma)} somas found - ignoring.')
        else:
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
                    legendgroup=legend_group,
                    alphahull=.5,
                    showlegend=False,
                    color=c,
                    hoverinfo='name'))

    return trace_data


def scatter2plotly(x, **kwargs):
    """Convert DataFrame with x,y,z columns to plotly scatter plot."""
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
                                      alpha=kwargs.get('alpha', None),
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
    """Convert Volumes to plotly objects."""
    colors = kwargs.get('color',
                        kwargs.get('c',
                                   kwargs.get('colors', None)))

    _, _, colormap = prepare_colormap(colors,
                                      volumes=x,
                                      alpha=kwargs.get('alpha', None),
                                      use_neuron_color=kwargs.get('use_neuron_color', False),
                                      color_range=255)

    trace_data = []
    for i, v in enumerate(x):
        # Skip empty data
        if isinstance(v.vertices, np.ndarray):
            if not v.vertices.any():
                continue
        elif not v.vertices:
            continue

        name = getattr(v, 'name', None)

        c = colormap[i]
        if len(c) == 3:
            c = (c[0], c[1], c[2], .5)

        rgba_str = f'rgba({c[0]:.0f},{c[1]:.0f},{c[2]:.0f},{c[3]:.1f})'
        trace_data.append(go.Mesh3d(x=v.vertices[:, 0],
                                    y=v.vertices[:, 1],
                                    z=v.vertices[:, 2],
                                    i=v.faces[:, 0],
                                    j=v.faces[:, 1],
                                    k=v.faces[:, 2],
                                    color=rgba_str,
                                    name=name,
                                    hoverinfo='none'))

    return trace_data


def layout2plotly(**kwargs):
    """Generate layout for plotly figures."""
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
