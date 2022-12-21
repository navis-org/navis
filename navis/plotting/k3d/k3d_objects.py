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

import k3d
import uuid
import warnings

import numpy as np
import pandas as pd
import trimesh as tm

from ..colors import vertex_colors, eval_color, color_to_int
from ..plot_utils import segments_to_coords, fibonacci_sphere
from ... import core, utils, config, conversion

logger = config.get_logger(__name__)

__all__ = ['neuron2k3d', 'scatter2k3d', 'dotprops2k3d', 'voxel2k3d',
           'volume2k3d']

# Generate sphere for somas
fib_points = fibonacci_sphere(samples=30)


def neuron2k3d(x, colormap, **kwargs):
    """Convert neurons to k3d objects."""
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
        logger.warning('`shade_by` does not work with the k3d backend')

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
                trace_data += skeleton2k3d(neuron,
                                           label=label,
                                           legendgroup=legendgroup,
                                           showlegend=showlegend,
                                           color=color, **kwargs)
            elif isinstance(neuron, core.MeshNeuron):
                trace_data += mesh2k3d(neuron,
                                       label=label,
                                       legendgroup=legendgroup,
                                       showlegend=showlegend,
                                       color=color, **kwargs)
            elif isinstance(neuron, core.Dotprops):
                trace_data += dotprops2k3d(neuron,
                                           label=label,
                                           legendgroup=legendgroup,
                                           showlegend=showlegend,
                                           color=color, **kwargs)
            elif isinstance(neuron, core.VoxelNeuron):
                trace_data += voxel2k3d(neuron,
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

                c = color_to_int(eval_color(c, color_range=255))

                this_cn = neuron.connectors[neuron.connectors.type == j]
                cn_label = f'{cn_lay.get(j, {"name": "connector"})["name"]} of {name}'

                if cn_lay['display'] == 'circles' or not isinstance(neuron, core.TreeNeuron):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        trace_data.append(k3d.points(
                            positions=this_cn[['x', 'y', 'z']].values,
                            name=cn_label,
                            shader='flat',
                            point_size=1000,
                            color=c
                        ))
                elif cn_lay['display'] == 'lines':
                    # Find associated treenodes
                    co1 = this_cn[['x', 'y', 'z']].values
                    co2 = neuron.nodes.set_index('node_id').loc[this_cn.node_id.values,
                                                                ['x', 'y', 'z']].values

                    coords = np.array([co for seg in zip(co1, co1, co2, co2, [[np.nan] * 3] * len(co1)) for co in seg])

                    lw = kwargs.get('linewidth', kwargs.get('lw', 1))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        trace_data.append(k3d.line(coords,
                                                   color=c,
                                                   name=cn_label,
                                                   width=lw,
                                                   shader='thick'))
                else:
                    raise ValueError(f'Unknown display type for connectors "{cn_lay["display"]}"')

    return trace_data


def mesh2k3d(neuron, legendgroup, showlegend, label, color, **kwargs):
    """Convert MeshNeuron to k3d object."""
    # Skip empty neurons
    if neuron.n_vertices == 0:
        return []

    opacity = 1
    if isinstance(color, np.ndarray) and color.ndim == 2:
        if len(color) == len(neuron.vertices):
            color = [color_to_int(c) for c in color]
            color_kwargs = dict(colors=color)
        else:
            raise ValueError('Colors must match number of vertices for K3D meshes.')
    else:
        c = color_to_int(color)
        color_kwargs = dict(color=c)

        if len(color) == 4:
            opacity = color[3]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace_data = [k3d.mesh(vertices=neuron.vertices.astype('float32'),
                               indices=neuron.faces.astype('uint32'),
                               name=label,
                               flat_shading=False,
                               opacity=opacity,
                               **color_kwargs)]

    return trace_data


def voxel2k3d(neuron, legendgroup, showlegend, label, color, **kwargs):
    """Convert VoxelNeuron to k3d object."""
    # Skip empty neurons
    if min(neuron.shape) == 0:
        return []

    img = neuron.grid
    if img.dtype not in (np.float32, np.float64):
        img = img.astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace_data = [k3d.volume(img.T,
                                 bounds=neuron.bbox.flatten(),
                                 interpolation=False,
                                 )]

    return trace_data


def skeleton2k3d(neuron, legendgroup, showlegend, label, color, **kwargs):
    """Convert skeleton (i.e. TreeNeuron) to plotly line plot."""
    if neuron.nodes.empty:
        logger.warning(f'Skipping TreeNeuron w/o nodes: {neuron.label}')
        return []
    elif neuron.nodes.shape[0] == 1:
        logger.warning(f'Skipping single-node skeleton: {neuron.label}')
        return []

    coords = segments_to_coords(neuron, neuron.segments)
    linewidth = kwargs.get('linewidth', kwargs.get('lw', 1))

    # We have to add (None, None, None) to the end of each segment to
    # make that line discontinuous. For reasons I don't quite understand
    # we have to also duplicate the first and the last coordinate in each segment
    # (possibly a bug)
    coords = np.concatenate([co for seg in coords for co in [seg[:1], seg, seg[-1:], [[None] * 3]]], axis=0)

    color_kwargs = {}
    if isinstance(color, np.ndarray) and color.ndim == 2:
        # Change colors to rgb integers
        c = [color_to_int(c) for c in color]

        # Next we have to make colors match the segments in `coords`
        c = np.asarray(c)
        ix = dict(zip(neuron.nodes.node_id.values, np.arange(neuron.n_nodes)))
        # Construct sequence node IDs just like we did in `coords`
        # (note that we insert a "-1" for breaks between segments)
        seg_ids = [co for seg in neuron.segments for co in [seg[:1], seg, seg[-1:], [-1]]]
        seg_ids = np.concatenate(seg_ids, axis=0)
        # Translate to node indices
        seg_ix = [ix.get(n, 0) for n in seg_ids]

        # Now map this to vertex colors
        seg_colors = [c[i] for i in seg_ix]

        color_kwargs['colors'] = seg_colors
    else:
        color_kwargs['color'] = c = color_to_int(color)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace_data = [k3d.line(coords,
                               width=linewidth,
                               shader='thick',
                               name=label,
                               **color_kwargs)]

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
                if isinstance(c, (list, np.ndarray)):
                    s_ix = np.where(neuron.nodes.node_id == s)[0][0]
                    soma_color = int(c[s_ix])
                else:
                    soma_color = int(c)

                n = neuron.nodes.set_index('node_id').loc[s]
                r = getattr(n, neuron.soma_radius) if isinstance(neuron.soma_radius, str) else neuron.soma_radius

                sp = tm.primitives.Sphere(radius=r, subdivisions=2)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    trace_data.append(k3d.mesh(vertices=sp.vertices + n[['x', 'y', 'z']].values.astype('float32'),
                                               indices=sp.faces.astype('uint32'),
                                               color=soma_color,
                                               flat_shading=False,
                                               name=f"soma of {label}"
                                               ))

    return trace_data


def scatter2k3d(x, **kwargs):
    """Convert DataFrame with x,y,z columns to plotly scatter plot."""
    c = eval_color(kwargs.get('color', kwargs.get('c', (100, 100, 100))),
                   color_range=255)
    c = color_to_int(c)
    s = kwargs.get('size', kwargs.get('s', 1))
    name = kwargs.get('name', None)

    trace_data = []
    for scatter in x:
        if isinstance(scatter, pd.DataFrame):
            if not all([c in scatter.columns for c in ['x', 'y', 'z']]):
                raise ValueError('DataFrame must have x, y and z columns')
            scatter = [['x', 'y', 'z']].values

        if not isinstance(scatter, np.ndarray):
            scatter = np.array(scatter)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace_data.append(k3d.points(positions=scatter,
                                         name=name,
                                         color=c,
                                         point_size=s,
                                         shader='dot'))
    return trace_data


def dotprops2k3d(x, legendgroup, showlegend, label, color, **kwargs):
    """Convert Dotprops to plotly graph object."""
    scale_vec = kwargs.get('dps_scale_vec', 'auto')
    tn = x.to_skeleton(scale_vec=scale_vec)

    return skeleton2k3d(tn, legendgroup, showlegend, label, color, **kwargs)


def volume2k3d(x, colormap, **kwargs):
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
        if len(c) == 4:
            opacity = c[3]
        else:
            opacity = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace_data = [k3d.mesh(vertices=v.vertices.astype('float32'),
                                   indices=v.faces.astype('uint32'),
                                   name=name,
                                   color=color_to_int(c[:3]),
                                   flat_shading=False,
                                   opacity=opacity)]

    return trace_data
