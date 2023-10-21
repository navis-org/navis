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

import uuid
import warnings

import numpy as np
import pandas as pd
import trimesh as tm

import plotly.graph_objs as go

from scipy import ndimage

from ..colors import vertex_colors, prepare_colormap, eval_color
from ..plot_utils import segments_to_coords, fibonacci_sphere
from ... import core, utils, config, conversion

logger = config.get_logger(__name__)

__all__ = ['neuron2plotly', 'scatter2plotly', 'dotprops2plotly',
           'volume2plotly', 'layout2plotly']

# Generate sphere for somas
BASE_SPHERE = tm.primitives.Sphere(radius=1, center=(0, 0, 0), subdivisions=2)


def neuron2plotly(x, colormap, **kwargs):
    """Convert neurons to plotly objects."""
    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)
    elif not isinstance(x, core.NeuronList):
        raise TypeError('Unable to process data of type "{}"'.format(type(x)))

    palette = kwargs.get('palette', None)
    color_by = kwargs.get('color_by', None)
    shade_by = kwargs.get('shade_by', None)
    lg = kwargs.pop('legend_group', None)

    if not isinstance(color_by, type(None)):
        if not palette:
            raise ValueError('Must provide `palette` (e.g. "viridis") argument '
                             'if using `color_by`')

        colormap = vertex_colors(x,
                                 by=color_by,
                                 alpha=kwargs.get('alpha', 1),
                                 use_alpha=False,
                                 palette=palette,
                                 vmin=kwargs.get('vmin', None),
                                 vmax=kwargs.get('vmax', None),
                                 na=kwargs.get('na', 'raise'),
                                 color_range=255)

    if not isinstance(shade_by, type(None)):
        alphamap = vertex_colors(x,
                                 by=shade_by,
                                 use_alpha=True,
                                 palette='viridis',  # palette is irrelevant here
                                 vmin=kwargs.get('smin', None),
                                 vmax=kwargs.get('smax', None),
                                 na=kwargs.get('na', 'raise'),
                                 color_range=255)

        new_colormap = []
        for c, a in zip(colormap, alphamap):
            if not (isinstance(c, np.ndarray) and c.ndim == 2):
                c = np.tile(c, (a.shape[0],  1))

            if c.dtype not in (np.float16, np.float32, np.float64):
                c = c.astype(np.float16)

            if c.shape[1] == 4:
                c[:, 3] = a[:, 3]
            else:
                c = np.insert(c, 3, a[:, 3], axis=1)

            new_colormap.append(c)
        colormap = new_colormap

    cn_lay = config.default_connector_colors.copy()
    cn_lay.update(kwargs.get('synapse_layout', {}))

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

        showlegend = True
        label = neuron.label
        if isinstance(lg, dict) and neuron.id in lg:
            # Check if this the first entry for this legendgroup
            label = legendgroup = lg[neuron.id]
            for d in trace_data:
                # If it is not the first entry, hide it
                if getattr(d, 'legendgroup', None) == legendgroup:
                    showlegend = False
                    break
        elif isinstance(lg, str):
            legendgroup = lg
        else:
            legendgroup = neuron_id

        if kwargs.get('radius', False):
            # Convert and carry connectors with us
            if isinstance(neuron, core.TreeNeuron):
                _neuron = conversion.tree2meshneuron(neuron)
                _neuron.connectors = neuron.connectors
                neuron = _neuron

        if not kwargs.get('connectors_only', False):
            if isinstance(neuron, core.TreeNeuron):
                trace_data += skeleton2plotly(neuron,
                                              label=label,
                                              legendgroup=legendgroup,
                                              showlegend=showlegend,
                                              color=color, **kwargs)
            elif isinstance(neuron, core.MeshNeuron):
                trace_data += mesh2plotly(neuron,
                                          label=label,
                                          legendgroup=legendgroup,
                                          showlegend=showlegend,
                                          color=color, **kwargs)
            elif isinstance(neuron, core.VoxelNeuron):
                trace_data += voxel2plotly(neuron,
                                           label=label,
                                           legendgroup=legendgroup,
                                           showlegend=showlegend,
                                           color=color, **kwargs)
            elif isinstance(neuron, core.Dotprops):
                trace_data += dotprops2plotly(neuron,
                                              label=label,
                                              legendgroup=legendgroup,
                                              showlegend=showlegend,
                                              color=color, **kwargs)
            else:
                raise TypeError(f'Unable to plot neurons of type "{type(neuron)}"')

        # Add connectors
        if (kwargs.get('connectors', False)
                or kwargs.get('connectors_only', False)) and neuron.has_connectors:
            cn_colors = kwargs.get('cn_colors', None)
            for j in neuron.connectors.type.unique():
                if isinstance(cn_colors, dict):
                    c = cn_colors.get(j, cn_lay.get(j, {'color': (10, 10, 10)})['color'])
                elif cn_colors == 'neuron':
                    c = color
                elif cn_colors:
                    c = cn_colors
                else:
                    c = cn_lay.get(j, {'color': (10, 10, 10)})['color']

                c = eval_color(c, color_range=255)

                this_cn = neuron.connectors[neuron.connectors.type == j]

                if cn_lay['display'] == 'circles' or isinstance(neuron, core.MeshNeuron):
                    trace_data.append(go.Scatter3d(
                        x=this_cn.x.values,
                        y=this_cn.y.values,
                        z=this_cn.z.values,
                        mode='markers',
                        marker=dict(color=f'rgb{c}', size=cn_lay.get('size', 2)),
                        name=f'{cn_lay.get(j, {"name": "connector"})["name"]} of {name}',
                        showlegend=False,
                        legendgroup=legendgroup,
                        hoverinfo='none'
                    ))
                elif cn_lay['display'] == 'lines':
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
                        name=f'{cn_lay.get(j, {"name": "connector"})["name"]} of {name}',
                        showlegend=False,
                        legendgroup=legendgroup,
                        hoverinfo='none'
                    ))
                else:
                    raise ValueError(f'Unknown display type for connectors "{cn_lay["display"]}"')

    return trace_data


def mesh2plotly(neuron, legendgroup, showlegend, label, color, **kwargs):
    """Convert MeshNeuron to plotly object."""
    # Skip empty neurons
    if neuron.n_vertices == 0:
        return []

    if isinstance(color, np.ndarray) and color.ndim == 2:
        if len(color) == len(neuron.vertices):
            # For some reason single colors are 0-255 but face/vertex colors
            # have to be 0-1
            color_kwargs = dict(vertexcolor=color / [255, 255, 255, 1])
        elif len(color) == len(neuron.faces):
            color_kwargs = dict(facecolor=color / [255, 255, 255, 1])
        else:
            color_kwargs = dict(color=color)
    else:
        try:
            if len(color) == 3:
                c = 'rgb{}'.format(color)
            elif len(color) == 4:
                c = 'rgba{}'.format(color)
        except BaseException:
            c = 'rgb(10,10,10)'
        color_kwargs = dict(color=c)

    if kwargs.get('hover_name', False):
        hoverinfo = 'text'
        hovertext = neuron.label
    else:
        hoverinfo = 'none'
        hovertext = ' '

    trace_data = [go.Mesh3d(x=neuron.vertices[:, 0],
                            y=neuron.vertices[:, 1],
                            z=neuron.vertices[:, 2],
                            i=neuron.faces[:, 0],
                            j=neuron.faces[:, 1],
                            k=neuron.faces[:, 2],
                            name=label,
                            legendgroup=legendgroup,
                            showlegend=showlegend,
                            hovertext=hovertext,
                            hoverinfo=hoverinfo,
                            **color_kwargs
                            )]

    return trace_data


def voxel2plotly(neuron, legendgroup, showlegend, label, color,
                 as_scatter=True, **kwargs):
    """Convert VoxelNeuron to plotly object.

    Turns out that plotly is horrendous for plotting voxel data (Volumes):
    anything more than a few thousand voxels (e.g. 40x40x40) and the html
    encoding and loading the plot takes ages. Unfortunately, the same happens
    with Isosurfaces.

    I'm adding an implementation here but until plotly gets MUCH better at this,
    there is really no point. For now, we will fallback to plotting the
    voxels as scatter plots using the top 10k voxels sorted by brightness.

    """
    # Skip empty neurons
    if min(neuron.shape) == 0:
        return []

    try:
        if len(color) == 3:
            c = 'rgb{}'.format(color)
        elif len(color) == 4:
            c = 'rgba{}'.format(color)
    except BaseException:
        c = 'rgb(10,10,10)'

    if kwargs.get('hover_name', False):
        hoverinfo = 'text'
        hovertext = neuron.label
    else:
        hoverinfo = 'none'
        hovertext = ' '

    if not as_scatter:
        # Downsample heavily
        ds = ndimage.zoom(neuron.grid, .2, order=1)

        # Generate X, Y, Z, coordinates for values in grid
        X, Y, Z = np.meshgrid(range(ds.shape[0]),
                              range(ds.shape[1]),
                              range(ds.shape[2]),
                              indexing='ij')

        # Flatten and scale coordinates
        X = X.flatten() * neuron.units_xyz[0] + neuron.offset[0]
        Y = Y.flatten() * neuron.units_xyz[1] + neuron.offset[1]
        Z = Z.flatten() * neuron.units_xyz[2] + neuron.offset[2]

        # Flatten and normalize values
        values = ds.flatten() / ds.max()

        trace_data = [go.Isosurface(x=X, y=Y, z=Z,
                                    value=values,
                                    isomin=0.001,
                                    isomax=1,
                                    opacity=0.1,
                                    surface_count=21,
                                    )]
    else:
        voxels, values = neuron.voxels, neuron.values

        # Sort by brightness
        srt = np.argsort(values)

        # Take the top 100k voxels
        values = values[srt[-100000:]]
        voxels = voxels[srt[-100000:]]

        # Scale and offset voxels
        voxels = voxels * neuron.units_xyz.magnitude + neuron.offset

        with warnings.catch_warnings():
            trace_data = [go.Scatter3d(x=voxels[:, 0],
                                       y=voxels[:, 1],
                                       z=voxels[:, 2],
                                       mode='markers',
                                       marker=dict(color=values,
                                                   size=4,
                                                   colorscale='viridis',
                                                   opacity=.1),
                                       name=label,
                                       legendgroup=legendgroup,
                                       showlegend=showlegend,
                                       hovertext=hovertext,
                                       hoverinfo=hoverinfo)]

    return trace_data


def skeleton2plotly(neuron, legendgroup, showlegend, label, color, **kwargs):
    """Convert skeleton (i.e. TreeNeuron) to plotly line plot."""
    if not hasattr(neuron, 'nodes') or neuron.nodes.empty:
        logger.warning(f'Skipping TreeNeuron w/o nodes: {neuron.label}')
        return []
    elif neuron.nodes.shape[0] == 1:
        logger.warning(f'Skipping single-node TreeNeuron: {neuron.label}')
        return []

    coords = segments_to_coords(neuron, neuron.segments)
    linewidth = kwargs.get('linewidth', kwargs.get('lw', 2))

    # We have to add (None, None, None) to the end of each segment to
    # make that line discontinuous
    coords = np.vstack([np.append(t, [[None] * 3], axis=0) for t in coords])

    if isinstance(color, np.ndarray) and color.ndim == 2:
        # Change colors to rgb/a strings
        if color.shape[1] == 4:
            c = [f'rgba({c[0]},{c[1]},{c[2]},{c[3]:.3f})' for c in color]
        else:
            c = [f'rgb({c[0]},{c[1]},{c[2]})' for c in color]

        # Next we have to make colors match the segments in `coords`
        c = np.asarray(c)
        ix = dict(zip(neuron.nodes.node_id.values, np.arange(neuron.n_nodes)))
        c = [col for s in neuron.segments for col in np.append(c[[ix[n] for n in s]], 'rgb(0,0,0)')]

    else:
        c = f'rgb({color[0]},{color[1]},{color[2]})'

    if kwargs.get('hover_id', False):
        hoverinfo = 'text'
        hovertext = [str(i) for seg in neuron.segments for i in seg + [None]]
    elif kwargs.get('hover_name', False):
        hoverinfo = 'text'
        hovertext = neuron.label
    else:
        hoverinfo = 'none'
        hovertext = ' '

    trace_data = [go.Scatter3d(x=coords[:, 0],
                               y=coords[:, 1],
                               z=coords[:, 2],
                               mode='lines',
                               line=dict(color=c,
                                         width=linewidth),
                               name=label,
                               legendgroup=legendgroup,
                               showlegend=showlegend,
                               hoverinfo=hoverinfo,
                               hovertext=hovertext
                               )]

    # Add soma(s):
    soma = utils.make_iterable(neuron.soma)
    if kwargs.get('soma', True):
        # If soma detection is messed up we might end up producing
        # hundrets of soma which will freeze the session
        if len(soma) >= 10:
            logger.warning(f'Neuron {neuron.id} appears to have {len(soma)} '
                           'somas. That does not look right - will ignore '
                           'them for plotting.')
        else:
            for s in soma:
                # Skip `None` somas
                if isinstance(s, type(None)):
                    continue

                # If we have colors for every vertex, we need to find the
                # color that corresponds to this root (or it's parent to be
                # precise)
                if isinstance(c, list):
                    s_ix = np.where(neuron.nodes.node_id == s)[0][0]
                    soma_color = c[s_ix]
                else:
                    soma_color = c

                n = neuron.nodes.set_index('node_id').loc[s]
                r = getattr(n, neuron.soma_radius) if isinstance(neuron.soma_radius, str) else neuron.soma_radius

                trace_data += [go.Mesh3d(x=BASE_SPHERE.vertices[:, 0] * r + n.x,
                                         y=BASE_SPHERE.vertices[:, 1] * r + n.y,
                                         z=BASE_SPHERE.vertices[:, 2] * r + n.z,
                                         i=BASE_SPHERE.faces[:, 0],
                                         j=BASE_SPHERE.faces[:, 1],
                                         k=BASE_SPHERE.faces[:, 2],
                                         legendgroup=legendgroup,
                                         showlegend=False,
                                         hoverinfo='name',
                                         color=soma_color,
                                         )]

    return trace_data


def scatter2plotly(x, **kwargs):
    """Convert DataFrame with x,y,z columns to plotly scatter plot."""
    c = eval_color(kwargs.get('color', kwargs.get('c', (100, 100, 100))),
                   color_range=255)
    s = kwargs.get('size', kwargs.get('s', 2))
    name = kwargs.get('name', None)

    trace_data = []
    for scatter in x:
        if isinstance(scatter, pd.DataFrame):
            if not all([c in scatter.columns for c in ['x', 'y', 'z']]):
                raise ValueError('DataFrame must have x, y and z columns')
            scatter = scatter[['x', 'y', 'z']].values

        if not isinstance(scatter, np.ndarray):
            scatter = np.array(scatter)

        trace_data.append(go.Scatter3d(x=scatter[:, 0],
                                       y=scatter[:, 1],
                                       z=scatter[:, 2],
                                       mode=kwargs.get('mode', 'markers'),
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


def dotprops2plotly(x, legendgroup, showlegend, label, color, **kwargs):
    """Convert Dotprops to plotly graph object."""
    scale_vec = kwargs.get('dps_scale_vec', 'auto')
    tn = x.to_skeleton(scale_vec=scale_vec)

    return skeleton2plotly(tn, legendgroup, showlegend, label, color, **kwargs)


def volume2plotly(x, colormap, **kwargs):
    """Convert Volumes to plotly objects."""
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
                                    showlegend=kwargs.get('volume_legend', False),
                                    hoverinfo='none'))

    return trace_data


def layout2plotly(**kwargs):
    """Generate layout for plotly figures."""
    layout = dict(width=kwargs.get('width', None),   # these override autosize
                  height=kwargs.get('height', 600),  # these override autosize
                  autosize=kwargs.get('fig_autosize', True),
                  title=kwargs.get('pl_title', None),
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,0,0,0)',
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

    return layout
