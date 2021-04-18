#    This script is part of navis (http://www.github.com/schlegelp/navis).
#    Copyright (C) 2017 Philipp Schlegel
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
#
#    You should have received a copy of the GNU General Public License
#    along

""" Module contains functions to plot neurons in 2D and 3D.
"""
import uuid
import warnings

import pandas as pd
import numpy as np

import matplotlib.colors as mcl

from ... import core, config, utils, morpho, conversion
from ..colors import *
from ..plot_utils import segments_to_coords, make_tube

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import vispy
    from vispy import scene
    from vispy.geometry import create_sphere

__all__ = ['volume2vispy', 'neuron2vispy', 'dotprop2vispy',
           'points2vispy', 'combine_visuals']

logger = config.logger


def volume2vispy(x, **kwargs):
    """Convert Volume(s) to vispy visuals."""
    # Must not use make_iterable here as this will turn into list of keys!
    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    # List to fill with vispy visuals
    visuals = []
    for i, v in enumerate(x):
        if not isinstance(v, core.Volume):
            raise TypeError(f'Expected navis.Volume, got "{type(v)}"')

        object_id = uuid.uuid4()

        if 'color' in kwargs or 'c' in kwargs:
            color = kwargs.get('color', kwargs.get('c', (.95, .95, .95, .1)))
        else:
            color = getattr(v, 'color', (.95, .95, .95, .1))

        # Colors might be list, need to pick the correct color for this volume
        if isinstance(color, list):
            if all([isinstance(c, (tuple, list, np.ndarray)) for c in color]):
                color = color[i]

        if isinstance(color, str):
            color = mcl.to_rgb(color)

        color = np.array(color, dtype=float)

        # Add alpha
        if len(color) < 4:
            color = np.append(color, [.1])

        if max(color) > 1:
            color[:3] = color[:3] / 255

        s = scene.visuals.Mesh(vertices=v.vertices,
                               faces=v.faces, color=color,
                               shading=kwargs.get('shading', 'smooth'))

        # Set some aesthetic parameters
        s.shininess = 0
        # Possible presets are "additive", "translucent", "opaque"
        s.set_gl_state('additive' if color[3] < 1 else 'opaque',
                       cull_face=True,
                       depth_test=False if color[3] < 1 else True)

        # Make sure volumes are always drawn after neurons
        s.order = kwargs.get('order', 10)

        # Add custom attributes
        s.unfreeze()
        s._object_type = 'volume'
        s._volume_name = getattr(v, 'name', None)
        s._object = v
        s._object_id = object_id
        s.freeze()

        visuals.append(s)

    return visuals


def neuron2vispy(x, **kwargs):
    """Convert a Neuron/List to vispy visuals.

    Parameters
    ----------
    x :               TreeNeuron | MeshNeuron | Dotprops | NeuronList
                      Neuron(s) to plot.
    color :           list | tuple | array | str
                      Color to use for plotting.
    colormap :        tuple | dict | array
                      Color to use for plotting. Dictionaries should be mapped
                      by ID. Overrides ``color``.
    connectors :      bool, optional
                      If True, plot connectors.
    connectors_only : bool, optional
                      If True, only connectors are plotted.
    by_strahler :     bool, optional
                      If True, shade neurites by strahler order.
    by_confidence :   bool, optional
                      If True, shade neurites by confidence.
    linewidth :       int, optional
                      Set linewidth. Might not work depending on your backend.
    cn_mesh_colors :  bool, optional
                      If True, connectors will have same color as the neuron.
    synapse_layout :  dict, optional
                      Sets synapse layout. For example::

                        {
                            0: {
                                'name': 'Presynapses',
                                'color': (255, 0, 0)
                                },
                            1: {
                                'name': 'Postsynapses',
                                'color': (0, 0, 255)
                                },
                            2: {
                                'name': 'Gap junctions',
                                'color': (0, 255, 0)
                                },
                            'display': 'lines'  # can also be 'circles'
                        }

    Returns
    -------
    list
                    Contains vispy visuals for each neuron.

    """
    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)
    elif isinstance(x, core.NeuronList):
        pass
    else:
        raise TypeError(f'Unable to process data of type "{type(x)}"')

    colors = kwargs.get('color',
                        kwargs.get('c',
                                   kwargs.get('colors', None)))
    palette = kwargs.get('palette', None)
    color_by = kwargs.get('color_by', None)
    shade_by = kwargs.get('shade_by', None)

    if not isinstance(color_by, type(None)):
        if not palette:
            raise ValueError('Must provide `palette` (e.g. "viridis") argument '
                             'if using `color_by`')

        colormap = vertex_colors(x,
                                 by=color_by,
                                 alpha=False,
                                 palette=palette,
                                 vmin=kwargs.get('vmin', None),
                                 vmax=kwargs.get('vmax', None),
                                 na=kwargs.get('na', 'raise'),
                                 color_range=1)
    else:
        colormap, _ = prepare_colormap(colors,
                                       neurons=x,
                                       palette=palette,
                                       alpha=kwargs.get('alpha', None),
                                       color_range=1)

    if not isinstance(shade_by, type(None)):
        alphamap = vertex_colors(x,
                                 by=shade_by,
                                 alpha=True,
                                 palette='viridis',  # palette is irrelevant here
                                 vmin=kwargs.get('smin', None),
                                 vmax=kwargs.get('smax', None),
                                 na=kwargs.get('na', 'raise'),
                                 color_range=1)

        new_colormap = []
        for c, a in zip(colormap, alphamap):
            if not (isinstance(c, np.ndarray) and c.ndim == 2):
                c = np.tile(c, (a.shape[0],  1))

            if c.shape[1] == 4:
                c[:, 3] = a[:, 3]
            else:
                c = np.insert(c, 3, a[:, 3], axis=1)

            new_colormap.append(c)
        colormap = new_colormap

    # List to fill with vispy visuals
    visuals = []
    for i, neuron in enumerate(x):
        # Generate random ID -> we need this in case we have duplicate IDs
        object_id = uuid.uuid4()

        if kwargs.get('radius', False):
            neuron = conversion.tree2meshneuron(neuron)

        neuron_color = colormap[i]
        if not kwargs.get('connectors_only', False):
            if isinstance(neuron, core.TreeNeuron):
                visuals += skeleton2vispy(neuron,
                                          neuron_color,
                                          object_id,
                                          **kwargs)
            elif isinstance(neuron, core.MeshNeuron):
                visuals += mesh2vispy(neuron,
                                      neuron_color,
                                      object_id,
                                      **kwargs)
            elif isinstance(neuron, core.Dotprops):
                visuals += dotprop2vispy(neuron,
                                         neuron_color,
                                         object_id,
                                         **kwargs)
            else:
                logger.warning(f"Don't know how to plot neuron of type '{type(neuron)}'")

        if (kwargs.get('connectors', False)
            or kwargs.get('connectors_only', False)) and neuron.has_connectors:
            visuals += connectors2vispy(neuron,
                                        neuron_color,
                                        object_id,
                                        **kwargs)

    return visuals


def connectors2vispy(neuron, neuron_color, object_id, **kwargs):
    """Convert connectors to vispy visuals."""
    cn_lay = config.default_connector_colors
    cn_lay.update(kwargs.get('synapse_layout', {}))

    visuals = []
    cn_colors = kwargs.get('cn_colors', None)
    for j in neuron.connectors.type.unique():
        if isinstance(cn_colors, dict):
            color = cn_colors.get(j, cn_lay.get(j, {}).get('color', (.1, .1, .1)))
        elif cn_colors == 'neuron':
            color = neuron_color
        elif cn_colors:
            color = cn_colors
        else:
            color = cn_lay.get(j, {}).get('color', (.1, .1, .1))

        color = eval_color(color, color_range=1)

        this_cn = neuron.connectors[neuron.connectors.type == j]

        pos = this_cn[['x', 'y', 'z']].apply(pd.to_numeric).values

        mode = cn_lay['display']
        if mode == 'circles' or isinstance(neuron, core.MeshNeuron):
            con = scene.visuals.Markers()

            con.set_data(pos=np.array(pos),
                         face_color=color,
                         edge_color=color,
                         size=cn_lay.get('size', 1))

        elif mode == 'lines':
            tn_coords = neuron.nodes.set_index('node_id').loc[this_cn.node_id.values][['x', 'y', 'z']].apply(pd.to_numeric).values

            segments = [item for sublist in zip(pos, tn_coords) for item in sublist]

            con = scene.visuals.Line(pos=np.array(segments),
                                     color=color,
                                     # Can only be used with method 'agg'
                                     width=kwargs.get('linewidth', 1),
                                     connect='segments',
                                     antialias=False,
                                     method='gl')
            # method can also be 'agg' -> has to use connect='strip'
        else:
            raise ValueError(f'Unknown connector display mode "{mode}"')

        # Add custom attributes
        con.unfreeze()
        con._object_type = 'neuron'
        con._neuron_part = 'connectors'
        con._id = neuron.id
        con._name = str(getattr(neuron, 'name', neuron.id))
        con._object_id = object_id
        con._object = neuron
        con.freeze()

        visuals.append(con)
    return visuals


def mesh2vispy(neuron, neuron_color, object_id, **kwargs):
    """Convert mesh (i.e. MeshNeuron) to vispy visuals."""
    m = scene.visuals.Mesh(vertices=neuron.vertices,
                           faces=neuron.faces,
                           color=neuron_color,
                           shading=kwargs.get('shading', 'smooth'))

    # Set some aesthetic parameters
    m.shininess = 0
    # Possible presets are "additive", "translucent", "opaque"
    if len(neuron_color) == 4 and neuron_color[3] < 1:
        m.set_gl_state('additive',
                       cull_face=True,
                       depth_test=False)

    # Add custom attributes
    m.unfreeze()
    m._object_type = 'neuron'
    m._neuron_part = 'neurites'
    m._id = neuron.id
    m._name = str(getattr(neuron, 'name', neuron.id))
    m._object_id = object_id
    m._object = neuron
    m.freeze()
    return [m]


def skeleton2vispy(neuron, neuron_color, object_id, **kwargs):
    """Convert skeleton (i.e. TreeNeuron) into vispy visuals."""
    visuals = []
    if not kwargs.get('connectors_only', False) and not neuron.nodes.empty:
        # Make sure we have one color for each node
        neuron_color = np.asarray(neuron_color)
        if neuron_color.ndim == 1:
            neuron_color = np.tile(neuron_color, (neuron.nodes.shape[0],  1))

        # Get nodes
        non_roots = neuron.nodes[neuron.nodes.parent_id >= 0]
        connect = np.zeros((non_roots.shape[0], 2), dtype=int)
        node_ix = pd.Series(np.arange(neuron.nodes.shape[0]),
                            index=neuron.nodes.node_id.values)
        connect[:, 0] = node_ix.loc[non_roots.node_id].values
        connect[:, 1] = node_ix.loc[non_roots.parent_id].values

        # Create line plot from segments.
        t = scene.visuals.Line(pos=neuron.nodes[['x', 'y', 'z']].values,
                               color=neuron_color,
                               # Can only be used with method 'agg'
                               width=kwargs.get('linewidth', 1),
                               connect=connect,
                               antialias=True,
                               method='gl')
        # method can also be 'agg' -> has to use connect='strip'
        # Make visual discoverable
        t.interactive = True

        # Add custom attributes
        t.unfreeze()
        t._object_type = 'neuron'
        t._neuron_part = 'neurites'
        t._id = neuron.id
        t._name = str(getattr(neuron, 'name', neuron.id))
        t._object = neuron
        t._object_id = object_id
        t.freeze()

        visuals.append(t)

        # Extract and plot soma
        soma = utils.make_iterable(neuron.soma)
        if kwargs.get('soma', True) and any(soma):
            # If soma detection is messed up we might end up producing
            # hundrets of soma which will freeze the session
            if len(soma) >= 10:
                logger.warning(f'Neuron {neuron.id} appears to have {len(soma)}'
                               ' somas. That does not look right - will ignore '
                               'them for plotting.')
            else:
                for s in soma:
                    # If we have colors for every vertex, we need to find the
                    # color that corresponds to this root (or it's parent to be
                    # precise)
                    if isinstance(neuron_color, np.ndarray) and neuron_color.ndim > 1:
                        s_ix = np.where(neuron.nodes.node_id == s)[0][0]
                        soma_color = neuron_color[s_ix]
                    else:
                        soma_color = neuron_color

                    n = neuron.nodes.set_index('node_id').loc[s]
                    r = getattr(n, neuron.soma_radius) if isinstance(neuron.soma_radius, str) else neuron.soma_radius
                    sp = create_sphere(7, 7, radius=r)
                    verts = sp.get_vertices() + n[['x', 'y', 'z']].values
                    s = scene.visuals.Mesh(vertices=verts,
                                           shading='smooth',
                                           faces=sp.get_faces(),
                                           color=soma_color)
                    s.ambient_light_color = vispy.color.Color('white')

                    # Make visual discoverable
                    s.interactive = True

                    # Add custom attributes
                    s.unfreeze()
                    s._object_type = 'neuron'
                    s._neuron_part = 'soma'
                    s._id = neuron.id
                    s._name = str(getattr(neuron, 'name', neuron.id))
                    s._object = neuron
                    s._object_id = object_id
                    s.freeze()

                    visuals.append(s)

    return visuals


def dotprop2vispy(x, neuron_color, object_id, **kwargs):
    """Convert dotprops(s) to vispy visuals.

    Parameters
    ----------
    x :             navis.Dotprops | pd.DataFrame
                    Dotprop(s) to plot.

    Returns
    -------
    list
                    Contains vispy visuals for each dotprop.

    """
    # Generate TreeNeuron
    scale_vec = kwargs.get('dps_scale_vec', 1)
    tn = x.to_skeleton(scale_vec=scale_vec)
    return skeleton2vispy(tn, neuron_color, object_id, **kwargs)


def points2vispy(x, **kwargs):
    """Convert points to vispy visuals.

    Parameters
    ----------
    x :             list of arrays
                    Points to plot.
    color :         tuple | array
                    Color to use for plotting.
    size :          int, optional
                    Marker size.

    Returns
    -------
    list
                    Contains vispy visuals for points.

    """
    colors = kwargs.get('color',
                        kwargs.get('c',
                                   kwargs.get('colors',
                                              eval_color(config.default_color, 1))))

    visuals = []
    for p in x:
        object_id = uuid.uuid4()
        if not isinstance(p, np.ndarray):
            p = np.array(p)

        con = scene.visuals.Markers()
        con.set_data(pos=p,
                     face_color=colors,
                     edge_color=colors,
                     size=kwargs.get('size', 2))

        # Add custom attributes
        con.unfreeze()
        con._object_type = 'points'
        con._object_id = object_id
        con.freeze()

        visuals.append(con)

    return visuals


def combine_visuals(visuals, name=None):
    """Attempt to combine multiple visuals of similar type into one.

    Parameters
    ----------
    visuals :   List
                List of visuals.
    name :      str, optional
                Legend name for the combined visual.

    Returns
    -------
    list
                List of visuals some of which where combined.

    """
    if any([not isinstance(v, scene.visuals.VisualNode) for v in visuals]):
        raise TypeError('Visuals must all be instances of VisualNode')

    # Combining visuals (i.e. adding pos AND changing colors) fails if
    # they are already on a canvas
    if any([v.parent for v in visuals]):
        raise ValueError('Visuals must not have parents when combined.')

    # Sort into types
    types = set([type(v) for v in visuals])

    by_type = {ty: [v for v in visuals if type(v) == ty] for ty in types}

    combined = []

    # Now go over types and combine when possible
    for ty in types:
        # Skip if nothing to combine
        if len(by_type[ty]) <= 1:
            combined += by_type[ty]
            continue

        if ty == scene.visuals.Line:
            # Collate data
            pos = np.concatenate([vis._pos for vis in by_type[ty]])

            # We need to produce one color/vertex
            colors = np.concatenate([np.repeat([vis.color],
                                               vis.pos.shape[0],
                                               axis=0) for vis in by_type[ty]])

            t = scene.visuals.Line(pos=pos,
                                   color=colors,
                                   # Can only be used with method 'agg'
                                   connect='segments',
                                   antialias=True,
                                   method='gl')
            # method can also be 'agg' -> has to use connect='strip'
            # Make visual discoverable
            t.interactive = True

            # Add custom attributes
            t.unfreeze()
            t._object_type = 'neuron'
            t._neuron_part = 'neurites'
            t._id = 'NA'
            t._object_id = uuid.uuid4()
            t._name = name if name else 'NeuronCollection'
            t.freeze()

            combined.append(t)
        elif ty == scene.visuals.Mesh:
            vertices = []
            faces = []
            color = []
            for vis in by_type[ty]:
                verts_offset = sum([v.shape[0] for v in vertices])
                faces.append(vis.mesh_data.get_faces() + verts_offset)
                vertices.append(vis.mesh_data.get_vertices())

                vc = vis.mesh_data.get_vertex_colors()
                if not isinstance(vc, type(None)):
                    color.append(vc)
                else:
                    color.append(np.tile(vis.color.rgba, len(vertices[-1])).reshape(-1, 4))

            faces = np.vstack(faces)
            vertices = np.vstack(vertices)
            color = np.concatenate(color)

            if np.unique(color, axis=0).shape[0] == 1:
                base_color = color[0]
            else:
                base_color = (1, 1, 1, 1)

            t = scene.visuals.Mesh(vertices,
                                   faces=faces,
                                   color=base_color,
                                   vertex_colors=color,
                                   shading=by_type[ty][0].shading,
                                   mode=by_type[ty][0].mode)

            # Add custom attributes
            t.unfreeze()
            t._object_type = 'neuron'
            t._neuron_part = 'neurites'
            t._id = 'NA'
            t._name = name if name else 'MeshNeuronCollection'
            t._object_id = uuid.uuid4()
            t.freeze()

            combined.append(t)
        else:
            combined += by_type[ty]

    return combined
