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
from ... import core, config, utils, morpho
from ..colors import *
from ..plot_utils import segments_to_coords, make_tube

import uuid
import warnings

import pandas as pd
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import vispy
    from vispy import scene
    from vispy.geometry import create_sphere

__all__ = ['volume2vispy', 'neuron2vispy', 'dotprop2vispy',
           'points2vispy', 'combine_visuals']

logger = config.logger


def volume2vispy(x, **kwargs):
    """ Converts Volume(s) to vispy visuals."""
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
        s._object = v
        s._object_id = object_id
        s.freeze()

        visuals.append(s)

    return visuals


def neuron2vispy(x, **kwargs):
    """Convert a TreeNeuron/List to vispy visuals.

    Parameters
    ----------
    x :               TreeNeuron | MeshNeuron | NeuronList
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

    colormap, _, _ = prepare_colormap(colors,
                                      neurons=x,
                                      use_neuron_color=kwargs.get('use_neuron_color', False),
                                      alpha=kwargs.get('alpha', None),
                                      color_range=1)

    # List to fill with vispy visuals
    visuals = []
    for i, neuron in enumerate(x):
        # Generate random ID -> we need this in case we have duplicate IDs
        object_id = uuid.uuid4()

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
            else:
                logger.warning(f"Don't know how to plot neuron of type '{type(neuron)}'")

        if (kwargs.get('connectors', False) or kwargs.get('connectors_only', False)) and neuron.has_connectors:
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
    for j in neuron.connectors.type.unique():
        if kwargs.get('cn_mesh_colors', False):
            color = neuron_color
        else:
            color = cn_lay.get(j, {}).get('color', (.1, .1, .1))

        if max(color) > 1:
            color = np.array(color) / 255

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
    #m.shininess = 0
    # Possible presets are "additive", "translucent", "opaque"
    #m.set_gl_state('additive', cull_face=True, depth_test=False)

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
        # Get root node indices (may be more than one if neuron has been
        # cut weirdly)
        root_ix = neuron.nodes[neuron.nodes.parent_id < 0].index.tolist()

        # Get nodes
        nodes = neuron.nodes[neuron.nodes.parent_id >= 0]

        # Extract treenode_coordinates and their parent's coordinates
        tn_coords = nodes[['x', 'y', 'z']].apply(pd.to_numeric).values
        parent_coords = neuron.nodes.set_index('node_id').loc[nodes.parent_id.values][['x', 'y', 'z']].apply(pd.to_numeric).values

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

        if not kwargs.get('radius', False):
            # Turn coordinates into segments
            segments = [item for sublist in zip(tn_coords, parent_coords) for item in sublist]
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
            t._id = neuron.id
            t._name = str(getattr(neuron, 'name', neuron.id))
            t._object = neuron
            t._object_id = object_id
            t.freeze()

            visuals.append(t)
        else:
            # Generate coordinates
            coords = segments_to_coords(neuron,
                                        neuron.segments,
                                        modifier=(1, 1, 1))
            # For each point of each segment get the radius
            nodes = neuron.nodes.set_index('node_id')
            radii = [nodes.loc[s, 'radius'].values.astype(float) for s in neuron.segments]

            # Generate faces and vertices for the tube
            verts, faces = make_tube(segments=coords,
                                     radii=radii,
                                     use_normals=kwargs.get('use_normals', True),
                                     tube_points=kwargs.get('tube_points', 3))

            vertex_colors = np.resize(neuron_color,
                                      (verts.shape[0],
                                       kwargs.get('tube_points', 3)))

            t = scene.visuals.Mesh(vertices=verts,
                                   faces=faces,
                                   vertex_colors=vertex_colors,
                                   shading='smooth',
                                   mode='triangles')

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

        if kwargs.get('by_strahler', False) or \
           kwargs.get('by_confidence', False):
            # Convert array back to a single color without alpha
            neuron_color = neuron_color[0][:3]

        # Extract and plot soma
        soma = utils.make_iterable(neuron.soma)
        if kwargs.get('soma', True) and any(soma):
            # If soma detection is messed up we might end up producing
            # dozens of soma which will freeze the kernel
            if len(soma) >= 10:
                logger.warning(f'{neuron.id}: {len(soma)} somas found.')
            for s in soma:
                n = neuron.nodes.set_index('node_id').loc[s]
                r = getattr(n, neuron.soma_radius) if isinstance(neuron.soma_radius, str) else neuron.soma_radius
                sp = create_sphere(7, 7, radius=r)
                verts = sp.get_vertices() + n[['x', 'y', 'z']].values
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
                s._id = neuron.id
                s._name = str(getattr(neuron, 'name', neuron.id))
                s._object = neuron
                s._object_id = object_id
                s.freeze()

                visuals.append(s)

    return visuals


def dotprop2vispy(x, **kwargs):
    """Converts dotprops(s) to vispy visuals.

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
        raise TypeError(f'Unable to process data of type "{type(x)}"')

    visuals = []

    # Parse colors for dotprops
    colors = kwargs.get('color',
                        kwargs.get('c',
                                   kwargs.get('colors', None)))
    _, colormap, _ = prepare_colormap(colors,
                                      dotprops=x,
                                      use_neuron_color=False,
                                      alpha=kwargs.get('alpha', None),
                                      color_range=1)

    scale_vect = kwargs.get('scale_vect', 1)

    for i, n in enumerate(x.itertuples()):
        # Generate random ID -> we need this in case we have duplicate IDs
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
        t._name = getattr(n, 'gene_name', getattr(n, 'name', 'NoName'))
        t._object = n
        t._object_id = object_id
        t.freeze()

        visuals.append(t)

        # Add soma (if provided as X/Y/Z)
        if all([hasattr(n, v) for v in ['X', 'Y', 'Z']]):
            sp = create_sphere(5, 5, radius=4)
            s = scene.visuals.Mesh(vertices=sp.get_vertices()
                                            + np.array([n.X, n.Y, n.Z]),
                                   faces=sp.get_faces(),
                                   color=color)

            # Add custom attributes
            s.unfreeze()
            s._object_type = 'dotprop'
            s._neuron_part = 'soma'
            s._name = n.gene_name
            s._object = n
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


def combine_visuals(visuals):
    """Attempt to combine multiple visuals of similar type into one.

    Parameters
    ----------
    visuals :   List
                List of visuals

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
            t._name = 'NeuronCollection'
            t.freeze()

            combined.append(t)
        elif ty == scene.visuals.Mesh:
            vertices = []
            faces = []
            for vis in by_type[ty]:
                verts_offset = sum([v.shape[0] for v in vertices])
                faces.append(vis.mesh_data.get_faces() + verts_offset)
                vertices.append(vis.mesh_data.get_vertices())

            faces = np.vstack(faces)
            vertices = np.vstack(vertices)

            color = np.concatenate([vis.mesh_data.get_vertex_colors() for vis in by_type[ty]])

            t = scene.visuals.Mesh(vertices,
                                   faces=faces,
                                   vertex_colors=color,
                                   shading=by_type[ty][0].shading,
                                   mode=by_type[ty][0].mode)

            # Add custom attributes
            t.unfreeze()
            t._object_type = 'neuron'
            t._neuron_part = 'neurites'
            t._id = 'NA'
            t._name = 'MeshNeuronCollection'
            t._object_id = uuid.uuid4()
            t.freeze()

            combined.append(t)
        else:
            combined += by_type[ty]

    return combined
