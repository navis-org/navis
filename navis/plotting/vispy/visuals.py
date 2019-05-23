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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import vispy
    from vispy import scene
    from vispy.geometry import create_sphere

from ... import core, config, utils
from ..colors import *
from ..external.tube import Tube

__all__ = ['volume2vispy', 'neuron2vispy', 'dotprop2vispy',
           'points2vispy']

logger = config.logger


def volume2vispy(x, **kwargs):
    """ Converts Volume(s) to vispy visuals."""

    # Must not use _make_iterable here as this will turn into list of keys!
    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    # List to fill with vispy visuals
    visuals = []
    for i, v in enumerate(x):
        if not isinstance(v, core.Volume):
            raise TypeError('Expected navis.Volume, got "{}"'.format(type(v)))

        object_id = uuid.uuid4()

        if 'color' in kwargs or 'c' in kwargs:
            color = kwargs.get('color', kwargs.get('c', (.95, .95, .95, .1)))
        else:
            color = getattr(v, 'color', (.95, .95, .95, .1))

        # Colors might be list, need to pick the correct color for this volume
        if isinstance(color, list):
            if all([isinstance(c, (tuple, list, np.ndarray)) for c in color]):
                color = color[i]

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
        s.set_gl_state('additive', cull_face=True, depth_test=False)

        # Make sure volumes are always drawn after neurons
        s.order = kwargs.get('order', 10)

        # Add custom attributes
        s.unfreeze()
        s._object_type = 'volume'
        s._volume_name = getattr(v, 'name', None)
        s._object_id = object_id
        s.freeze()

        visuals.append(s)

    return visuals


def neuron2vispy(x, **kwargs):
    """ Converts a TreeNeuron/List to vispy visuals.

    Parameters
    ----------
    x :               TreeNeuron | NeuronList
                      Neuron(s) to plot.
    color :           list | tuple | array | str
                      Color to use for plotting.
    colormap :        tuple | dict | array
                      Color to use for plotting. Dictionaries should be mapped
                      by skeleton ID. Overrides ``color``.
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
                      Sets synapse layout. Default settings::
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
                            'display': 'lines'  # 'circles'
                        }

    Returns
    -------
    list
                    Contains vispy visuals for each neuron.
    """

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
                                    color_range=1)

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

    # List to fill with vispy visuals
    visuals = []

    for i, neuron in enumerate(x):
        # Generate random ID -> we need this in case we have duplicate
        # skeleton IDs
        object_id = uuid.uuid4()

        neuron_color = colormap[i]

        # Convert color 0-1
        if max(neuron_color) > 1:
            neuron_color = np.array(neuron_color) / 255

        # Get root node indices (may be more than one if neuron has been
        # cut weirdly)
        root_ix = neuron.nodes[neuron.nodes.parent_id < 0].index.tolist()

        if not kwargs.get('connectors_only', False):
            nodes = neuron.nodes[neuron.nodes.parent_id >= 0]

            # Extract treenode_coordinates and their parent's coordinates
            tn_coords = nodes[['x', 'y', 'z']].apply(pd.to_numeric).values
            parent_coords = neuron.nodes.set_index('node_id').loc[nodes.parent_id.values][['x', 'y', 'z']].apply(pd.to_numeric).values

            # Turn coordinates into segments
            segments = [item for sublist in zip(
                tn_coords, parent_coords) for item in sublist]

            # Add alpha to color based on strahler
            if kwargs.get('by_strahler', False) \
                    or kwargs.get('by_confidence', False):
                if kwargs.get('by_strahler', False):
                    if 'strahler_index' not in neuron.nodes:
                        morpho.strahler_index(neuron)

                    # Generate list of alpha values
                    alpha = neuron.nodes['strahler_index'].values

                if kwargs.get('by_confidence', False):
                    if 'arbor_confidence' not in neuron.nodes:
                        morpho.arbor_confidence(neuron)

                    # Generate list of alpha values
                    alpha = neuron.nodes['arbor_confidence'].values

                # Pop root from coordinate lists
                alpha = np.delete(alpha, root_ix, axis=0)

                alpha = alpha / (max(alpha) + 1)
                # Duplicate values (start and end of each segment!)
                alpha = np.array([v for l in zip(alpha, alpha) for v in l])

                # Turn color into array
                # (need 2 colors per segment for beginning and end)
                neuron_color = np.array(
                    [neuron_color] * (tn_coords.shape[0] * 2), dtype=float)
                neuron_color = np.insert(neuron_color, 3, alpha, axis=1)

            if segments:
                if not kwargs.get('use_radius', False):
                    # Create line plot from segments.
                    t = scene.visuals.Line(pos=np.array(segments),
                                           color=list(neuron_color),
                                           # Can only be used with method 'agg'
                                           width=kwargs.get('linewidth', 1),
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
                    t._uuid = neuron.uuid
                    t._name = str(getattr(neuron, 'name', neuron.uuid))
                    t._object_id = object_id
                    t.freeze()

                    visuals.append(t)
                else:
                    coords = _segments_to_coords(neuron,
                                                 neuron.segments,
                                                 modifier=(1, 1, 1))
                    nodes = neuron.nodes.set_index('node_id')
                    for s, c in zip(neuron.segments, coords):
                        radii = nodes.loc[s, 'radius'].values.astype(float)
                        radii[radii <= 100] = 100
                        t = Tube(c.astype(float),
                                 radius=radii,
                                 color=neuron_color,
                                 tube_points=5,)

                        # Add custom attributes
                        t.unfreeze()
                        t._object_type = 'neuron'
                        t._neuron_part = 'neurites'
                        t._uuid = neuron.uuid
                        t._name = str(getattr(neuron, 'name', neuron.uuid))
                        t._object_id = object_id
                        t.freeze()

                        visuals.append(t)

            if kwargs.get('by_strahler', False) or \
               kwargs.get('by_confidence', False):
                # Convert array back to a single color without alpha
                neuron_color = neuron_color[0][:3]

            # Extract and plot soma
            soma = utils.make_iterable(neuron.soma)
            if any(soma):
                for s in soma:
                    node = neuron.nodes.set_index('node_id').loc[s]
                    radius = node.radius
                    sp = create_sphere(7, 7, radius=radius)
                    verts = sp.get_vertices() + node[['x', 'y', 'z']].values
                    s = scene.visuals.Mesh(vertices=verts,
                                           shading='smooth',
                                           faces=sp.get_faces(),
                                           color=neuron_color)
                    s.ambient_light_color = vispy.color.Color('white')

                    # Make visual discoverable
                    s.interactive = True

                    # Add custom attributes
                    s.unfreeze()
                    s._object_type = 'neuron'
                    s._neuron_part = 'soma'
                    s._uuid = neuron.uuid
                    s._name = str(getattr(neuron, 'name', neuron.uuid))
                    s._object_id = object_id
                    s.freeze()

                    visuals.append(s)

        if kwargs.get('connectors', False) or kwargs.get('connectors_only',
                                                         False):
            for j in neuron.connectors.relation.unique():
                if kwargs.get('cn_mesh_colors', False):
                    color = neuron_color
                else:
                    color = syn_lay.get(j, {'color': (.1, .1, .1)})['color']

                if max(color) > 1:
                    color = np.array(color) / 255

                this_cn = neuron.connectors[neuron.connectors.relation == j]

                if this_cn.empty:
                    continue

                pos = this_cn[['x', 'y', 'z']].apply(
                    pd.to_numeric).values

                if syn_lay['display'] == 'circles':
                    con = scene.visuals.Markers()

                    con.set_data(pos=np.array(pos),
                                 face_color=color, edge_color=color,
                                 size=syn_lay.get('size', 1))

                    visuals.append(con)

                elif syn_lay['display'] == 'lines':
                    tn_coords = neuron.nodes.set_index('node_id').ix[this_cn.node_id.values][['x', 'y', 'z']].apply(pd.to_numeric).values

                    segments = [item for sublist in zip(
                        pos, tn_coords) for item in sublist]

                    t = scene.visuals.Line(pos=np.array(segments),
                                           color=color,
                                           # Can only be used with method 'agg'
                                           width=kwargs.get('linewidth', 1),
                                           connect='segments',
                                           antialias=False,
                                           method='gl')
                    # method can also be 'agg' -> has to use connect='strip'

                    # Add custom attributes
                    t.unfreeze()
                    t._object_type = 'neuron'
                    t._neuron_part = 'connectors'
                    t._uuid = neuron.uuid
                    t._name = str(getattr(neuron, 'name', neuron.uuid))
                    t._object_id = object_id
                    t.freeze()

                    visuals.append(t)

    return visuals


def dotprop2vispy(x, **kwargs):
    """ Converts dotprops(s) to vispy visuals.

    Parameters
    ----------
    x :             core.Dotprops | pd.DataFrame
                    Dotprop(s) to plot.
    colormap :      tuple | dict | array
                    Color to use for plotting. Dictionaries should be mapped
                    to gene names.
    scale_vect :    int, optional
                    Vector to scale dotprops by.

    Returns
    -------
    list
                    Contains vispy visuals for each dotprop.
    """

    if not isinstance(x, (core.Dotprops, pd.DataFrame)):
        raise TypeError('Unable to process data of type "{}"'.format(type(x)))

    visuals = []

    # Parse colors for dotprops
    colors = kwargs.get('color',
                        kwargs.get('c',
                                   kwargs.get('colors', None)))
    _, colormap, _ = prepare_colormap(colors,
                                    None, x, use_neuron_color=False,
                                    color_range=1)

    scale_vect = kwargs.get('scale_vect', 1)

    for i, n in enumerate(x.itertuples()):
        # Generate random ID -> we need this in case we have duplicate skeleton IDs
        object_id = uuid.uuid4()

        color = colormap[i]

        # Prepare lines - this is based on nat:::plot3d.dotprops
        halfvect = n.points[
            ['x_vec', 'y_vec', 'z_vec']] / 2 * scale_vect

        starts = n.points[['x', 'y', 'z']
                          ].values - halfvect.values
        ends = n.points[['x', 'y', 'z']
                        ].values + halfvect.values

        segments = [item for sublist in zip(
            starts, ends) for item in sublist]

        t = scene.visuals.Line(pos=np.array(segments),
                               color=color,
                               width=2,
                               connect='segments',
                               antialias=False,
                               method='gl')  # method can also be 'agg'

        # Add custom attributes
        t.unfreeze()
        t._object_type = 'dotprop'
        t._neuron_part = 'neurites'
        t._name = n.gene_name
        t._object_id = object_id
        t.freeze()

        visuals.append(t)

        # Add soma
        sp = create_sphere(5, 5, radius=4)
        s = scene.visuals.Mesh(vertices=sp.get_vertices() +
                                        np.array([n.X, n.Y, n.Z]),
                               faces=sp.get_faces(),
                               color=color)

        # Add custom attributes
        s.unfreeze()
        s._object_type = 'dotprop'
        s._neuron_part = 'soma'
        s._name = n.gene_name
        s._object_id = object_id
        s.freeze()

        visuals.append(s)

    return visuals


def points2vispy(x, **kwargs):
    """ Converts points to vispy visuals.

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