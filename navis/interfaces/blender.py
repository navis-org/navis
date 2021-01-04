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

""" Interface with Blender. Unlike other moduls of navis, this module is
not automatically imported as it only works from within Blender.
"""

# Important bit of advice:
# Avoid operators ("bpy.ops.") as much as possible:
# They cause scene updates which will exponentially slow down processing

import colorsys
import json
import math
import os
import time
import uuid

import pandas as pd
import numpy as np
import trimesh as tm

from .. import core, utils, config
from ..plotting.colors import eval_color

logger = config.logger

try:
    import bpy
    import bmesh
    import mathutils
except ImportError:
    logger.error('Unable to load Blender API - this module only works from '
                 'within Blender!')
except BaseException:
    raise


class Handler:
    """Class that interfaces with scene in Blender.

    Parameters
    ----------
    scaling :   float, optional
                   scaling factor between navis and Blender coordinates.

    Notes
    -----

        (1) The handler adds neurons and keeps track of them in the scene.
        (2) If you request a list of objects via its attributes (e.g. ``Handler.neurons``)
            or via :func:`~navis.interfaces.blender.Handler.select`, a :class:`~navis.interfaces.blender.ObjectList`
            is returned. This class lets you change basic parameters of your selected
            neurons.

    Attributes
    ----------
    neurons :       returns list containing all neurons
    connectors :    returns list containing all connectors
    soma :          returns list containing all somata
    selected :      returns list containing selected objects
    presynapses :   returns list containing all presynapses
    postsynapses :  returns list containing all postsynapses
    gapjunctions :  returns list containing all gap junctions
    abutting :      returns list containing all abutting connectors
    all :           returns list containing all objects

    Examples
    --------
    >>> # This example assumes you have alread imported and set up navis
    >>> # b3d module has to be imported explicitly
    >>> from navis import b3d
    >>> # Get some neurons (you have already set up a remote instance?)
    >>> nl = navis.example_neurons()
    >>> # Initialize handler
    >>> h = b3d.Handler()
    >>> # Add neurons
    >>> h.add(nl)
    >>> # Assign colors to all neurons
    >>> h.colorize()
    >>> # Select all somas and change color to black
    >>> h.soma.color(0, 0, 0)
    >>> # Clear scene
    >>> h.clear()
    >>> # Add only soma
    >>> h.add(nl, neurites=False, connectors=False)

    """
    cn_dict = {
        0: dict(name='presynapses',
                color=(1, 0, 0)),
        1: dict(name='postsynapses',
                color=(0, 0, 1)),
        2: dict(name='gapjunction',
                color=(0, 1, 0)),
        3: dict(name='abutting',
                color=(1, 0, 1))

    }  # : defines default colours/names for different connector types

    # Some synonyms
    cn_dict['pre'] = cn_dict[0]
    cn_dict['post'] = cn_dict[1]
    cn_dict['gap'] = cn_dict['gapjunction'] = cn_dict[2]
    cn_dict['abutting'] = cn_dict[3]

    # Some other parameters
    cn_dict['display'] = 'lines'  # "lines" or "spheres", overriden if MeshNeuron
    cn_dict['size'] = 0.01  # sets size of spheres only

    defaults = dict(bevel_depth=0.007,
                    bevel_resolution=5,
                    resolution_u=10)

    def __init__(self,
                 scaling=1 / 10000,
                 axes_order=[0, 1, 2],
                 ax_translate=[1, 1, 1]):
        self.scaling = scaling
        self.cn_dict = Handler.cn_dict
        self.axes_order = axes_order
        self.ax_translate = ax_translate

    def _selection_helper(self, type):
        return [ob.name for ob in bpy.data.objects if 'type' in ob and ob['type'] == type]

    def _cn_selection_helper(self, cn_type):
        return [ob.name for ob in bpy.data.objects if 'type' in ob and ob['type'] == 'CONNECTORS' and ob['cn_type'] == cn_type]

    def __getattr__(self, key):
        if key == 'neurons' or key == 'neuron' or key == 'neurites':
            return ObjectList(self._selection_helper('NEURON'))
        elif key == 'connectors' or key == 'connector':
            return ObjectList(self._selection_helper('CONNECTORS'))
        elif key == 'soma' or key == 'somas':
            return ObjectList(self._selection_helper('SOMA'))
        elif key == 'selected':
            return ObjectList([ob.name for ob in bpy.context.selected_objects if 'navis_object' in ob])
        elif key == 'presynapses':
            return ObjectList(self._cn_selection_helper(0))
        elif key == 'postsynapses':
            return ObjectList(self._cn_selection_helper(1))
        elif key == 'gapjunctions':
            return ObjectList(self._cn_selection_helper(2))
        elif key == 'abutting':
            return ObjectList(self._cn_selection_helper(3))
        elif key == 'all':
            return self.neurons + self.connectors + self.soma
        else:
            raise AttributeError('Unknown attribute ' + key)

    def add(self, x, neurites=True, soma=True, connectors=True, redraw=False,
            use_radii=False, skip_existing=False, downsample=False,
            collection=None, **kwargs):
        """Add neuron(s) to scene.

        Parameters
        ----------
        x :             TreeNeuron | MeshNeuron | NeuronList | core.Volume
                        Objects to import into Blender.
        neurites :      bool, optional
                        Plot neurites. TreeNeurons only.
        soma :          bool, optional
                        Plot somas. TreeNeurons only.
        connectors :    bool, optional
                        Plot connectors. Uses a defaults dictionary to set
                        color/type. See Examples on how to change.
        redraw :        bool, optional
                        If True, will redraw window after each neuron. This
                        will slow down loading!
        use_radii :     bool, optional
                        If True, will use treenode radii. TreeNeurons only.
        skip_existing : bool, optional
                        If True, will skip neurons that are already loaded.
        downsample :    False | int, optional
                        If integer < 1, will downsample neurites upon import.
                        Preserves branch point/roots. TreeNeurons only.
        collection :    str, optional
                        Only for Blender >2.8: add object(s) to given collection.
                        If collection does not exist, will be created.

        Examples
        --------
        >>> h = navis.interfaces.blender.Handler()
        >>> n = navis.example_neurons(1)
        >>> h.add(n, connectors=True)

        Change connector settings
        >>> h.cn_dict['display'] = 'sphere'
        >>> h.cn_dict[0]['color'] = (1, 1, 0)

        """
        start = time.time()

        if skip_existing:
            exists = [ob.get('id', None) for ob in bpy.data.objects]

        if isinstance(x, (core.BaseNeuron, core.NeuronList)):
            if redraw:
                print('Set "redraw=False" to vastly speed up import!')
            if isinstance(x, core.BaseNeuron):
                x = [x]
            wm = bpy.context.window_manager
            wm.progress_begin(0, len(x))
            for i, n in enumerate(x):
                # Skip existing if applicable
                if skip_existing and n.id in exists:
                    continue
                self._create_neuron(n,
                                    neurites=neurites,
                                    soma=soma,
                                    connectors=connectors,
                                    collection=collection,
                                    downsample=downsample,
                                    use_radii=use_radii)
                if redraw:
                    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                wm.progress_update(i)
            wm.progress_end()
        elif isinstance(x, core.Volume):
            self._create_volume(x, collection=collection)
        elif isinstance(x, np.ndarray):
            self._create_scatter(x, collection=collection, **kwargs)
        elif isinstance(x, core.Dotprops):
            self._create_dotprops(x, collection=collection, **kwargs)
        else:
            raise AttributeError(f'Unable add data type of type {type(x)}')

        print(f'Import done in {time.time()-start:.2f}s')

        return

    def clear(self):
        """Clear all neurons """
        self.all.delete()

    def _create_scatter2(self, x, collection=None, **kwargs):
        """Create scatter by reusing mesh data.

        This generate an individual objects for each data point. This is slower!

        """
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError('Array must be of shape N,3')

        # Get & scale coordinates and invert y
        coords = x.astype(float)[:, self.axes_order]
        coords *= float(self.scaling)
        coords *= self.ax_translate

        verts, faces = calc_sphere(kwargs.get('size', 0.02),
                                   kwargs.get('sp_res', 7),
                                   kwargs.get('sp_res', 7))

        mesh = bpy.data.meshes.new(kwargs.get('name', 'scatter'))
        mesh.from_pydata(verts, [], faces)
        mesh.polygons.foreach_set('use_smooth', [True] * len(mesh.polygons))

        objects = []
        for i, co in enumerate(coords):
            obj = bpy.data.objects.new(kwargs.get('name', 'scatter') + str(i),
                                       mesh)
            obj.location = co
            obj.show_name = False
            objects.append(obj)

        # Link to scene and add to group
        group_name = kwargs.get('name', 'scatter')
        if group_name != 'scatter' and group_name in bpy.data.groups:
            group = bpy.data.groups[group_name]
        else:
            group = bpy.data.groups.new(group_name)

        if not collection:
            col = bpy.context.scene.collection
        elif collection in bpy.data.collections:
            col = bpy.data.collections[collection]
        else:
            col = bpy.data.collections.new(collection)
            bpy.context.scene.collection.children.link(col)

        for obj in objects:
            col.objects.link(obj)
            group.objects.link(obj)

        return

    def _create_scatter(self, x, collection=None, **kwargs):
        """Create scatter."""
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError('Array must be of shape N,3')

        # Get & scale coordinates and invert y
        coords = x.astype(float)[:, self.axes_order]
        coords *= float(self.scaling)
        coords *= self.ax_translate

        # Generate a base sphere
        base_sphere = tm.creation.uv_sphere(radius=kwargs.get('size', 0.02),
                                            count=[kwargs.get('sp_res', 7),
                                                   kwargs.get('sp_res', 7)])
        base_verts, base_faces = base_sphere.vertices, base_sphere.faces

        # Repeat sphere vertices
        sp_verts = np.tile(base_verts.T, coords.shape[0]).T
        # Add coords offsets to each sphere
        offsets = np.repeat(coords, base_verts.shape[0], axis=0)
        sp_verts += offsets

        # Repeat sphere faces and offset vertex indices
        sp_faces = np.tile(base_faces.T, coords.shape[0]).T
        face_offsets = np.repeat(np.arange(coords.shape[0]),
                                 base_faces.shape[0], axis=0)
        face_offsets *= base_verts.shape[0]
        sp_faces += face_offsets.reshape((face_offsets.size, 1))

        # Generate mesh
        mesh = bpy.data.meshes.new(kwargs.get('name', 'scatter'))
        mesh.from_pydata(sp_verts, [], sp_faces.tolist())
        mesh.polygons.foreach_set('use_smooth', [True] * len(mesh.polygons))
        obj = bpy.data.objects.new(kwargs.get('name', 'scatter'), mesh)

        if not collection:
            col = bpy.context.scene.collection
        elif collection in bpy.data.collections:
            col = bpy.data.collections[collection]
        else:
            col = bpy.data.collections.new(collection)
            bpy.context.scene.collection.children.link(col)

        col.objects.link(obj)

        obj.location = (0, 0, 0)
        obj.show_name = False

        return obj

    def _create_neuron(self, x, neurites=True, soma=True, connectors=True,
                       use_radii=False, downsample=False, collection=None):
        """Create neuron object."""
        mat_name = (f'M#{x.id}')[:59]

        mat = bpy.data.materials.get(mat_name,
                                     bpy.data.materials.new(mat_name))

        if isinstance(x, core.TreeNeuron):
            if neurites:
                self._create_skeleton(x, mat,
                                      use_radii=use_radii,
                                      downsample=downsample,
                                      collection=collection)
            if soma and not isinstance(x.soma, type(None)):
                self._create_soma(x, mat, collection=collection)
        elif isinstance(x, core.MeshNeuron):
            self._create_mesh(x, mat, collection=collection)
        else:
            raise TypeError(f'Expected Mesh/TreeNeuron, got "{type(x)}"')

        if connectors and x.has_connectors:
            self._create_connectors(x, collection=collection)

        return

    def _create_mesh(self, x, mat, collection=None):
        """Create mesh from MeshNeuron."""
        name = getattr(x, 'name', '')

        # Make copy of vertices as we are potentially modifying them
        verts = x.vertices.copy()

        # Convert to Blender space
        verts *= self.scaling
        verts = verts[:, self.axes_order]
        verts *= self.ax_translate

        me = bpy.data.meshes.new(f'{name} mesh')
        ob = bpy.data.objects.new(f"#{x.id} - {name}", me)
        ob.location = (0, 0, 0)
        ob.show_name = True
        ob['type'] = 'NEURON'
        ob['navis_object'] = True
        ob['id'] = str(x.id)

        blender_verts = verts.tolist()
        me.from_pydata(list(blender_verts), [], list(x.faces))
        me.update()

        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))

        ob.active_material = mat

        if not collection:
            col = bpy.context.scene.collection
        elif collection in bpy.data.collections:
            col = bpy.data.collections[collection]
        else:
            col = bpy.data.collections.new(collection)
            bpy.context.scene.collection.children.link(col)

        col.objects.link(ob)

    def _create_skeleton(self, x, mat, use_radii=False, downsample=False,
                         collection=None):
        """Create neuron branches."""
        name = getattr(x, 'name', '')

        cu = bpy.data.curves.new(f"{name} mesh", 'CURVE')
        ob = bpy.data.objects.new(f"#{x.id} - {name}", cu)
        ob.location = (0, 0, 0)
        ob.show_name = True
        ob['type'] = 'NEURON'
        ob['navis_object'] = True
        ob['id'] = str(x.id)
        cu.dimensions = '3D'
        cu.fill_mode = 'FULL'
        cu.bevel_resolution = self.defaults.get('bevel_resolution', 5)
        cu.resolution_u = self.defaults.get('resolution_u', 10)

        if use_radii:
            cu.bevel_depth = 1
        else:
            cu.bevel_depth = self.defaults.get('bevel_depth', 0.007)

        # DO NOT touch this: lookup via dict is >10X faster!
        tn_coords = {r.node_id: (r.x * self.scaling,
                                 r.y * self.scaling,
                                 r.z * self.scaling) for r in x.nodes.itertuples()}
        if use_radii:
            tn_radii = {r.node_id: r.radius * self.scaling for r in x.nodes.itertuples()}

        for s in x.segments:
            if isinstance(downsample, int) and downsample > 1:
                mask = np.zeros(len(s), dtype=bool)
                mask[downsample::downsample] = True

                keep = np.isin(s, x.nodes[x.nodes['type'] != 'slab'].node_id.values)

                s = np.array(s)[mask | keep]

            sp = cu.splines.new('POLY')

            coords = np.array([tn_coords[tn] for tn in s])
            coords = coords[:, self.axes_order]
            coords *= self.ax_translate

            # Add points
            sp.points.add(len(coords) - 1)

            # Add this weird fourth coordinate
            coords = np.c_[coords, [0] * coords.shape[0]]

            # Set point coordinates
            sp.points.foreach_set('co', coords.ravel())
            sp.points.foreach_set('weight', s)

            if use_radii:
                r = [tn_radii[tn] for tn in s]
                sp.points.foreach_set('radius', r)

        ob.active_material = mat

        if not collection:
            col = bpy.context.scene.collection
        elif collection in bpy.data.collections:
            col = bpy.data.collections[collection]
        else:
            col = bpy.data.collections.new(collection)
            bpy.context.scene.collection.children.link(col)

        col.objects.link(ob)

        return

    def _create_dotprops(self, x, scale_vect=1, collection=None):
        """Create neuron branches."""
        # Generate uuid
        object_id = str(uuid.uuid4())

        mat_name = (f'M#{object_id}')[:59]
        mat = bpy.data.materials.get(mat_name,
                                     bpy.data.materials.new(mat_name))

        cu = bpy.data.curves.new(f"{getattr(x, 'dotprop', '')} mesh", 'CURVE')
        ob = bpy.data.objects.new(f"#{object_id} - {getattr(x, 'neuron_name', '')}",
                                  cu)
        ob.location = (0, 0, 0)
        ob.show_name = True
        ob['type'] = 'DOTPROP'
        ob['navis_object'] = True
        ob['id'] = object_id
        cu.dimensions = '3D'
        cu.fill_mode = 'FULL'
        cu.bevel_resolution = 5
        cu.bevel_depth = 0.007

        # Prepare lines - this is based on nat:::plot3d.dotprops
        halfvect = (np.vstack(x.vector) / 2 * scale_vect)
        starts = (np.vstack(x.point) - halfvect)
        ends = (np.vstack(x.point) + halfvect)

        halfvect *= self.scaling
        starts *= self.scaling
        ends *= self.scaling

        halfvect = halfvect[:, self.axes_order] * self.ax_translate
        starts = starts[:, self.axes_order] * self.ax_translate
        ends = ends[:, self.axes_order] * self.ax_translate

        segments = list(zip(starts, ends))

        for s in segments:
            sp = cu.splines.new('POLY')

            # Add points
            sp.points.add(1)

            # Add this weird fourth coordinate
            coords = np.c_[s, [0, 0]]

            # Set point coordinates
            sp.points.foreach_set('co', coords.ravel())

        ob.active_material = mat

        if not collection:
            col = bpy.context.scene.collection
        elif collection in bpy.data.collections:
            col = bpy.data.collections[collection]
        else:
            col = bpy.data.collections.new(collection)
            bpy.context.scene.collection.children.link(col)

        col.objects.link(ob)

        return

    def _create_soma(self, x, mat, collection=None):
        """Create soma."""
        if not collection:
            col = bpy.context.scene.collection
        elif collection in bpy.data.collections:
            col = bpy.data.collections[collection]
        else:
            col = bpy.data.collections.new(collection)
            bpy.context.scene.collection.children.link(col)

        for s in utils.make_iterable(x.soma):
            s = x.nodes[x.nodes.node_id == s]
            loc = s[['x', 'y', 'z']].values
            loc = loc[:, self.axes_order]
            loc *= self.scaling
            loc *= self.ax_translate

            rad = s.radius * self.scaling

            mesh = bpy.data.meshes.new(f'Soma of #{x.id} - mesh')
            soma_ob = bpy.data.objects.new(f'Soma of #{x.id}', mesh)

            soma_ob.location = loc[0]

            # Construct the bmesh cube and assign it to the blender mesh.
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, diameter=rad)
            bm.to_mesh(mesh)
            bm.free()

            mesh.polygons.foreach_set('use_smooth', [True] * len(mesh.polygons))

            soma_ob.name = f'Soma of #{x.id}'
            soma_ob['type'] = 'SOMA'
            soma_ob['navis_object'] = True
            soma_ob['id'] = str(x.id)

            soma_ob.active_material = mat

            # Add the object into the scene.
            col.objects.link(soma_ob)

        return

    def _create_connectors(self, x, collection=None):
        """Create connectors."""
        if not x.has_connectors:
            return

        if not collection:
            col = bpy.context.scene.collection
        elif collection in bpy.data.collections:
            col = bpy.data.collections[collection]
        else:
            col = bpy.data.collections.new(collection)
            bpy.context.scene.collection.children.link(col)

        for t in x.connectors['type'].unique():
            con = x.connectors[x.connectors.type == t]

            # See if we have pre-defined names/colors for this
            settings = self.cn_dict.get(t, {'name': t, 'color': (0, 0, 0)})

            if con.empty:
                continue

            # Get & scale coordinates and invert y
            cn_coords = con[['x', 'y', 'z']].values.astype(float)

            ob_name = f'{settings["name"]} of {x.id}'

            # Only plot as lines if this is a TreeNeuron
            if self.cn_dict.get('display', 'lines') == 'lines' and isinstance(x, core.TreeNeuron):
                cn_coords = cn_coords[:, self.axes_order]
                cn_coords *= float(self.scaling)
                cn_coords *= self.ax_translate

                tn_coords = x.nodes.set_index('node_id').loc[con.node_id.values,
                                                             ['x', 'y', 'z']].values.astype(float)
                tn_coords = tn_coords[:, self.axes_order]
                tn_coords *= float(self.scaling)
                tn_coords *= self.ax_translate

                # Add 4th coordinate for blender
                cn_coords = np.c_[cn_coords, [0] * con.shape[0]]
                tn_coords = np.c_[tn_coords, [0] * con.shape[0]]

                # Combine cn and tn coords in pairs
                # This will have to be transposed to get pairs of cn and tn
                # (see below)
                coords = np.dstack([cn_coords, tn_coords])
                cu = bpy.data.curves.new(ob_name + ' mesh', 'CURVE')
                ob = bpy.data.objects.new(ob_name, cu)
                cu.dimensions = '3D'
                cu.fill_mode = 'FULL'
                cu.bevel_resolution = 0
                cu.bevel_depth = 0.007
                cu.resolution_u = 0

                for cn in coords:
                    sp = cu.splines.new('POLY')

                    # Add a second point
                    sp.points.add(1)

                    # Move points
                    sp.points.foreach_set('co', cn.T.ravel())
                col.objects.link(ob)
            else:
                ob = self._create_scatter(cn_coords,
                                          collection=collection,
                                          size=self.cn_dict.get('size', 0.01))
                ob.name = ob_name

            ob['type'] = 'CONNECTORS'
            ob['navis_object'] = True
            ob['cn_type'] = t
            ob['id'] = str(x.id)
            ob.location = (0, 0, 0)
            ob.show_name = False

            mat_name = f'{settings["name"]} of #{str(x.id)}'
            mat = bpy.data.materials.get(mat_name,
                                         bpy.data.materials.new(mat_name))
            mat.diffuse_color = eval_color(settings['color'],
                                           color_range=1,
                                           force_alpha=True)
            ob.active_material = mat

        return

    def _create_volume(self, volume, collection=None):
        """Create mesh from volume.

        Parameters
        ----------
        volume :    core.Volume | dict
                    Must contain 'faces', 'vertices'

        """
        mesh_name = str(getattr(volume, 'name', 'mesh'))

        verts = volume.vertices.copy()

        # Convert to Blender space
        verts *= self.scaling
        verts = verts[:, self.axes_order]
        verts *= self.ax_translate

        blender_verts = verts.tolist()

        me = bpy.data.meshes.new(mesh_name + '_mesh')
        ob = bpy.data.objects.new(mesh_name, me)

        scn = bpy.context.scene
        scn.collection.objects.link(ob)

        me.from_pydata(list(blender_verts), [], list(volume.faces))
        me.update()

        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))

    def select(self, x, *args):
        """Select given neurons.

        Parameters
        ----------
        x :     list of neuron IDs | Neuron/List | pd Dataframe

        Returns
        -------
        :class:`navis.b3d.ObjectList` :  containing requested neurons

        Examples
        --------
        >>> selection = Handler.select([123456, 7890])
        >>> # Get only connectors
        >>> cn = selection.connectors
        >>> # Hide everything else
        >>> cn.hide_others()
        >>> # Change color of presynapses
        >>> selection.presynapses.color(0, 1, 0)

        """
        ids = utils.eval_id(x)

        if not ids:
            logger.error('No ids found.')

        names = []

        for ob in bpy.data.objects:
            ob.select_set(False)
            if 'id' in ob:
                if ob['id'] in ids:
                    ob.select_set(True)
                    names.append(ob.name)
        return ObjectList(names, handler=self)

    def color(self, r, g, b):
        """Assign color to all neurons.

        Parameters
        ----------
        r :     float
                Red value, range 0-1
        g :     float
                Green value, range 0-1
        b :     float
                Blue value, range 0-1

        Notes
        -----
        This will only change color of neurons, if you want to change
        color of e.g. connectors, use:

        >>> handler.connectors.color(r, g, b)

        """
        self.neurons.color(r, g, b)

    def colorize(self):
        """Randomly colorize ALL neurons.

        Notes
        -----
        This will only change color of neurons, if you want to change
        color of e.g. connectors, use:

        >>> handler.connectors.colorize()

        """
        self.neurons.colorize()

    def emit(self, v):
        """Change emit value."""
        self.neurons.emit(v)

    def use_transparency(self, v):
        """Change transparency (True/False)."""
        self.neurons.use_transparency(v)

    def alpha(self, v):
        """Change alpha (0-1)."""
        self.neurons.alpha(v)

    def bevel(self, r):
        """Change bevel of ALL neurons.

        Parameters
        ----------
        r :         float
                    New bevel radius

        Notes
        -----
        This will only change bevel of neurons, if you want to change
        bevel of e.g. connectors, use:

        >>> handler.connectors.bevel(.02)

        """
        self.neurons.bevel_depth(r)

    def hide(self):
        """Hide all neuron-related objects."""
        self.all.hide()

    def unhide(self):
        """Unide all neuron-related objects."""
        self.all.unhide()


class ObjectList:
    """Collection of Blender objects.

    Notes
    -----
    1.  ObjectLists should normally be constructed via the handler
        (see :class:`navis.b3d.Handler`)!
    2.  List works with object NAMES to prevent Blender from crashing when
        trying to access neurons that do not exist anymore. This also means
        that changing names manually will compromise a object list.
    3.  Accessing a neuron list's attributes (see below) return another
        ``ObjectList`` class which you can use to manipulate the new
        subselection.

    Attributes
    ----------
    neurons :       returns list containing just neurons
    connectors :    returns list containing all connectors
    soma :          returns list containing all somata
    presynapses :   returns list containing all presynapses
    postsynapses :  returns list containing all postsynapses
    gapjunctions :  returns list containing all gap junctions
    abutting :      returns list containing all abutting connectors
    id :            returns list of IDs

    Examples
    --------
    >>> # b3d module has to be import explicitly
    >>> from navis import b3d
    >>> nl = navis.example_neurons()
    >>> handler = b3d.Handler()
    >>> handler.add(nl)
    >>> # Select only neurons on the right
    >>> right = handler.select('annotation:uPN right')
    >>> # This can be nested to change e.g. color of all right presynases
    >>> handler.select('annotation:uPN right').presynapses.color(0, 1, 0)

    """
    def __init__(self, object_names, handler=None):
        if not isinstance(object_names, list):
            object_names = [object_names]

        self.object_names = object_names
        self.handler = handler

    def __getattr__(self, key):
        if key in ['neurons', 'neuron', 'neurites']:
            return ObjectList([n for n in self.object_names if n in bpy.data.objects and bpy.data.objects[n]['type'] == 'NEURON'])
        elif key in ['connectors', 'connector']:
            return ObjectList([n for n in self.object_names if n in bpy.data.objects and bpy.data.objects[n]['type'] == 'CONNECTORS'])
        elif key in ['soma', 'somas']:
            return ObjectList([n for n in self.object_names if n in bpy.data.objects and bpy.data.objects[n]['type'] == 'SOMA'])
        elif key == 'presynapses':
            return ObjectList([n for n in self.object_names if n in bpy.data.objects and bpy.data.objects[n]['type'] == 'CONNECTORS' and bpy.data.objects[n]['cn_type'] == 0])
        elif key == 'postsynapses':
            return ObjectList([n for n in self.object_names if n in bpy.data.objects and bpy.data.objects[n]['type'] == 'CONNECTORS' and bpy.data.objects[n]['cn_type'] == 1])
        elif key == 'gapjunctions':
            return ObjectList([n for n in self.object_names if n in bpy.data.objects and bpy.data.objects[n]['type'] == 'CONNECTORS' and bpy.data.objects[n]['cn_type'] == 2])
        elif key == 'abutting':
            return ObjectList([n for n in self.object_names if n in bpy.data.objects and bpy.data.objects[n]['type'] == 'CONNECTORS' and bpy.data.objects[n]['cn_type'] == 3])
        elif key.lower() in ['id', 'ids']:
            return [bpy.data.objects[n]['id'] for n in self.object_names if n in bpy.data.objects]
        else:
            raise AttributeError('Unknown attribute ' + key)

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return ObjectList(self.object_names[key], handler=self.handler)
        else:
            raise Exception('Unable to index non-integers.')

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        self._repr = pd.DataFrame([[n, n in bpy.data.objects] for n in self.object_names],
                                  columns=['name', 'still_exists']
                                  )
        return str(self._repr)

    def __len__(self):
        return len(self.object_names)

    def __add__(self, to_add):
        if not isinstance(to_add, ObjectList):
            raise AttributeError('Can only merge other object lists')
        print(to_add.object_names)
        return ObjectList(list(set(self.object_names + to_add.object_names)),
                          handler=self.handler)

    def select(self, unselect_others=True):
        """Select objects in 3D viewer

        Parameters
        ----------
        unselect_others :   bool, optional
                            If False, will not unselect other objects.

        """
        for ob in bpy.data.objects:
            if ob.name in self.object_names:
                ob.select_set(True)
            elif unselect_others:
                ob.select_set(False)

    def color(self, r, g, b, a=1):
        """Assign color to all objects in the list.

        Parameters
        ----------
        r :     float
                Red value, range 0-1
        g :     float
                Green value, range 0-1
        b :     float
                Blue value, range 0-1
        a :     float
                Alpha value, range 0-1

        """
        for ob in bpy.data.objects:
            if ob.name in self.object_names:
                ob.active_material.diffuse_color = eval_color((r, g, b, a),
                                                              color_range=1,
                                                              force_alpha=True)

    def colorize(self):
        """Assign colors across the color spectrum."""
        for i, n in enumerate(self.object_names):
            c = colorsys.hsv_to_rgb(1 / (len(self) + 1) * i, 1, 1)
            if n in bpy.data.objects:
                try:
                    bpy.data.objects[n].active_material.diffuse_color = eval_color(c,
                                                                                   color_range=1,
                                                                                   force_alpha=True)
                except BaseException:
                    logger.warning(f'Error changing color of object "{n}"')

    def emit(self, e):
        """Change emit value."""
        for ob in bpy.data.objects:
            if ob.name in self.object_names:
                ob.active_material.emit = e

    def use_transparency(self, t):
        """Change transparency (True/False)."""
        for ob in bpy.data.objects:
            if ob.name in self.object_names:
                ob.active_material.use_transparency = t

    def alpha(self, a):
        """Change alpha (0-1)."""
        for ob in bpy.data.objects:
            if ob.name in self.object_names:
                ob.active_material.alpha = a

    def bevel(self, r):
        """Change bevel radius of objects.

        Parameters
        ----------
        r :         float
                    New bevel radius.

        """
        for n in self.object_names:
            if n in bpy.data.objects:
                if bpy.data.objects[n].type == 'CURVE':
                    bpy.data.objects[n].data.bevel_depth = r

    def hide(self):
        """Hide objects."""
        for i, n in enumerate(self.object_names):
            if n in bpy.data.objects:
                bpy.data.objects[n].hide = True

    def unhide(self):
        """Unhide objects."""
        for i, n in enumerate(self.object_names):
            if n in bpy.data.objects:
                bpy.data.objects[n].hide = False

    def hide_others(self):
        """Hide everything BUT these objects."""
        for ob in bpy.data.objects:
            if ob.name in self.object_names:
                ob.hide = False
            else:
                ob.hide = True

    def delete(self):
        """Delete neurons in the selection."""
        self.select(unselect_others=True)
        bpy.ops.object.delete()

    def to_json(self, fname='selection.json'):
        """Save neuron selection as json file which can be loaded
        in navis selection table.

        Parameters
        ----------
        fname :     str, optional
                    Filename to save selection to.

        """
        neuron_objects = [
            n for n in bpy.data.objects if n.name in self.object_names and n['type'] == 'NEURON']

        data = [dict(id=int(n['id']),
                     color="#{:02x}{:02x}{:02x}".format(int(255 * n.active_material.diffuse_color[0]),
                                                        int(255 *
                                                            n.active_material.diffuse_color[1]),
                                                        int(255 * n.active_material.diffuse_color[2])),
                     opacity=1
                     ) for n in neuron_objects]

        with open(fname, 'w') as outfile:
            json.dump(data, outfile)

        logger.info(f'Selection saved as {fname} in {os.getcwd()}')
        print(f'Selection saved as {fname} in {os.getcwd()}')


def calc_sphere(radius, nrPolar, nrAzimuthal):
    """Calculate vertices and faces for a sphere."""
    dPolar = math.pi / (nrPolar - 1)
    dAzimuthal = 2.0 * math.pi / (nrAzimuthal)

    # 1/2: vertices
    verts = []
    currV = mathutils.Vector((0.0, 0.0, radius))        # top vertex
    verts.append(currV)
    for iPolar in range(1, nrPolar - 1):                # regular vertices
        currPolar = dPolar * float(iPolar)

        currCosP = math.cos(currPolar)
        currSinP = math.sin(currPolar)

        for iAzimuthal in range(nrAzimuthal):
            currAzimuthal = dAzimuthal * float(iAzimuthal)

            currCosA = math.cos(currAzimuthal)
            currSinA = math.sin(currAzimuthal)

            currV = mathutils.Vector((currSinP * currCosA,
                                      currSinP * currSinA,
                                      currCosP)) * radius
            verts.append(currV)
    currV = mathutils.Vector((0.0, 0.0, - radius))        # bottom vertex
    verts.append(currV)

    # 2/2: faces
    faces = []
    for iAzimuthal in range(nrAzimuthal):                # top faces
        iNextAzimuthal = iAzimuthal + 1
        if iNextAzimuthal >= nrAzimuthal:
            iNextAzimuthal -= nrAzimuthal
        faces.append([0, iAzimuthal + 1, iNextAzimuthal + 1])

    for iPolar in range(nrPolar - 3):                    # regular faces
        iAzimuthalStart = iPolar * nrAzimuthal + 1

        for iAzimuthal in range(nrAzimuthal):
            iNextAzimuthal = iAzimuthal + 1
            if iNextAzimuthal >= nrAzimuthal:
                iNextAzimuthal -= nrAzimuthal
            faces.append([iAzimuthalStart + iAzimuthal,
                          iAzimuthalStart + iAzimuthal + nrAzimuthal,
                          iAzimuthalStart + iNextAzimuthal + nrAzimuthal,
                          iAzimuthalStart + iNextAzimuthal])

    iLast = len(verts) - 1
    iAzimuthalStart = iLast - nrAzimuthal
    for iAzimuthal in range(nrAzimuthal):                # bottom faces
        iNextAzimuthal = iAzimuthal + 1
        if iNextAzimuthal >= nrAzimuthal:
            iNextAzimuthal -= nrAzimuthal
        faces.append([iAzimuthalStart + iAzimuthal,
                      iLast,
                      iAzimuthalStart + iNextAzimuthal])

    return np.vstack(verts), faces
