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

import colorsys
import platform
import png
import uuid

import matplotlib.colors as mcl
import numpy as np
import seaborn as sns
import trimesh as tm

from collections import OrderedDict

from ... import utils, config
from ..colors import *
from ..vispy.vputils import *


try:
    import pygfx as gfx
    from wgpu.gui.auto import WgpuCanvas, run
except ImportError:
    gfx = None


__all__ = ['Viewer']

logger = config.get_logger(__name__)

# TODO
# - generalize viewer class to reduce redundancy with vispy viewer
# - add styles for viewer (lights, background, etc.) - e.g. .set_style(dark)
#   - e.g. material.metalness = 2 looks good for background meshes 
#   - metalness = 1 with roughness = 0 makes for funky looking neurons
#   - m.material.side = "FRONT" makes volumes look better   
# - make Viewer reactive (see reactive_rendering.py) to save 
#   resources when not actively using the viewer - might help in Jupyter?
# - add specialised methods for adding neurons, volumes, etc. to the viewer


class Viewer:
    """PyGFX 3D viewer.

    Parameters
    ----------
    **kwargs
              Keyword arguments passed to ``gfx.Display``.

    """

    def __init__(self, reactive=False, max_fps=30, **kwargs):
        try:
            ip = get_ipython()            
        except NameError:
            ip = None 

        if ip and not ip.active_eventloop:
            # ip.enable_gui('qt6')
            raise ValueError('IPython event loop not running. Please use e.g. "%gui qt6" to hook into the event loop.')

        if not gfx:            
            raise ImportError('`navis.Viewer` requires the `pygfx` package to '
                              'be installed:\n  pip3 install pygfx')
        # Update some defaults as necessary
        defaults = dict(size=None,
                        title='navis Viewer',
                        max_fps=max_fps)
        defaults.update(kwargs)

        # If we're running in headless mode (primarily for tests on CI) we will
        # simply not initialize the gfx objects. Not ideal but it turns
        # out to be very annoying to correctly setup on Github Actions.
        if getattr(config, 'headless', False):
            return
        
        self.canvas = WgpuCanvas(**defaults)
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas,
                                                   show_fps=False)

        # Set up a default scene
        self.scene = gfx.Scene()
        self.scene.add(gfx.AmbientLight())
        self.scene.add(gfx.DirectionalLight())

        # Modify the light
        light = self.scene.children[-1]
        light.local.z = -10000  # move light forward
        light.local.euler_x = 2.5 # rotate light

        # Set up a default background
        self._background = gfx.BackgroundMaterial((0, 0, 0))
        self.scene.add(gfx.Background(None, self._background))  

        # Add camera
        self.camera = gfx.OrthographicCamera()
        #self.camera.show_object(scene, scale=1.4)

        # Add controller
        self.controller = gfx.TrackballController(self.camera, register_events=self.renderer)

        # Stats 
        self.stats = gfx.Stats(self.renderer)
        self._show_fps = False

        """
        from PyQt5.QtWidgets import QPushButton

        # Create canvas
        button = QPushButton('PyQt5 button', self.canvas.native)
        button.move(10, 10)
        self.canvas.show()
        """

        # Register events 
        def _keydown(event):
            """Handle key presses."""
            if event.key == '1':
                self.set_view('XY')
            elif event.key == '2':
                self.set_view('XZ')
            elif event.key == '3':
                self.set_view('YZ')
            elif event.key == 'f':
                self._toggle_fps()

        self.renderer.add_event_handler(_keydown, "key_down")
        
        self._show_bounds = False

        self.show()

    def _animate(self):
        """Animate the scene."""
        if self._show_fps:
            with self.stats:        
                self.renderer.render(self.scene, self.camera, flush=False)            
            self.stats.render()
        else:
            self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw()

    def show(self):
        """Show viewer."""
        # This is for e.g. headless testing
        if getattr(config, 'headless', False):
            logger.info("Viewer widget not shown - navis running in headless mode. ")
            return
    
        self.canvas.show()                
        self.canvas.request_draw(self._animate)        

    @property
    def visible(self):
        """List IDs of currently visible neurons."""
        neurons = self.neurons  # grab this only once to speed things up
        return [s for s in neurons if neurons[s][0].visible]

    @property
    def invisible(self):
        """List IDs of currently visible neurons."""
        neurons = self.neurons  # grab this only once to speed things up
        return [s for s in neurons if not neurons[s][0].visible]

    @property
    def pinned(self):
        """List IDs of currently pinned neurons."""
        neurons = self.neurons  # grab this only once to speed things up
        return [s for s in neurons if getattr(neurons[s][0], '_pinned', False)]

    @property
    def selected(self):
        """Return IDs of or set selected neurons."""
        return self.__selected

    @selected.setter
    def selected(self, val):
        n = np.asarray(val).astype('object')

        neurons = self.neurons  # grab once to speed things up
        logger.debug(f'{len(n)} neurons selected ({len(self.selected)} previously)')
        # First un-highlight neurons no more selected
        for s in [s for s in self.__selected if s not in set(n)]:
            for v in neurons[s]:
                if isinstance(v, scene.visuals.Mesh):
                    v.color = v._stored_color
                else:
                    v.set_data(color=v._stored_color)

        # Highlight new additions
        for s in n:
            if s not in self.__selected:
                for v in neurons[s]:
                    # Keep track of old colour
                    v.unfreeze()
                    v._stored_color = v.color
                    v.freeze()
                    if isinstance(v, scene.visuals.Mesh):
                        v.color = self.highlight_color
                    else:
                        v.set_data(color=self.highlight_color)

        self.__selected = n

        # Update legend
        if self.show_legend:
            self.update_legend()

        # Update data text
        # Currently only the development version of vispy supports escape
        # character (e.g. \n)
        t = '| '.join([f'{neurons[s][0]._name} - #{s}' for s in self.__selected])
        self._data_text.text = t

    @property
    def visuals(self):
        """List of all visuals on this canvas."""
        return [c for c in self.scene.children if hasattr(c, '_object_type')]

    @property
    def bounds(self):
        """Bounds of all currently visuals (visible and invisible)."""
        bounds = []
        for vis in self.visuals:
            # Skip the bounding box itself
            if getattr(vis, '_object_type', '') == 'boundingbox':
                continue

            try:
                bounds.append(vis._bounds)
            except BaseException:
                pass

        if not bounds:
            return None

        bounds = np.dstack(bounds)

        mn = bounds[:, 0, :].min(axis=1)
        mx = bounds[:, 1, :].max(axis=1)

        return np.vstack((mn, mx)).T

    @property
    def _object_ids(self):
        """All object IDs on this canvas in order of addition."""
        obj_ids = []
        for v in self.visuals:
            if hasattr(v, '_object_id'):
                obj_ids.append(v._object_id)                
        return sorted(set(obj_ids), key=lambda x: obj_ids.index(x))

    @property
    def objects(self):
        """Ordered dictionary {uuid->[visuals]} of all objects in order of addition."""
        objects = OrderedDict()
        for ob in self._object_ids:
            objects[ob] = [v for v in self.visuals if getattr(v, '_object_id', None) == ob]

        return objects

    @property
    def neurons(self):
        """Return visible and invisible neuron visuals currently on the canvas.

        Returns
        -------
        OrderedDict
                    ``{id: [neurites, soma]}``

        """
        # Collect neuron objects (neurites + somata)
        visuals = self.visuals  # Get this only once to speed things up
        neuron_obj = [c for c in visuals if 'neuron' in getattr(c,
                                                                '_object_type',
                                                                '')]

        # Collect IDs
        neuron_ids = set([ob._id for ob in neuron_obj])

        # Collect somata and neurites by ID
        coll = OrderedDict()
        for ob in neuron_ids:
            coll[ob] = [v for v in visuals if getattr(v, '_id') == ob]
        return coll

    @property
    def _neuron_obj(self):
        """Return neurons by their object id."""
        # Collect neuron objects
        neuron_obj = [c for c in self.visuals if 'neuron' in getattr(
            c, '_object_type', '')]

        # Collect skeleton IDs
        obj_ids = set([ob._object_id for ob in neuron_obj])

        # Map visuals to unique skids
        return {s: [ob for ob in neuron_obj if ob._object_id == s] for s in obj_ids}

    def clear(self):
        """Clear canvas."""
        # Skip if running in headless mode
        if getattr(config, 'headless', False):
            return

        # Remove everything but the lights and backgrounds 
        self.scene.remove(*self.visuals)

    def remove(self, to_remove):
        """Remove given neurons/visuals from canvas."""
        to_remove = utils.make_iterable(to_remove)

        neurons = self.neurons  # grab this only once to speed things up
        for vis in to_remove:
            if vis in self.scene.children:
                self.scene.children.remove(vis)
            else:
                uuids = utils.eval_id(to_remove)
                for u in uuids:
                    for v in neurons.get(u, []):
                        self.scene.children.remove(v)

        if self.show_bounds:
            self.update_bounds()

    def pop(self, N=1):
        """Remove the most recently added N visuals."""
        for vis in list(self.objects.values())[-N:]:
            self.remove(vis)

    @property
    def show_bounds(self):
        """Set to ``True`` to show bounding box."""
        return self._show_bounds

    def toggle_bounds(self):
        """Toggle bounding box."""
        self.show_bounds = not self.show_bounds

    @show_bounds.setter
    def show_bounds(self, v):
        if not isinstance(v, bool):
            raise TypeError(f'Need bool, got {type(v)}')

        self._show_bounds = v

        if self.show_bounds:
            self.update_bounds()
        else:
            self.remove_bounds()

    def remove_bounds(self):
        """Remove bounding box visual."""
        self._show_bounds = False
        for v in self.visuals:
            if getattr(v, '_object_type', '') == 'boundingbox':
                self.remove(v)

    def update_bounds(self, color='w', width=1):
        """Update bounding box visual."""
        # Remove any existing visual
        self.remove_bounds()

        self._show_bounds = True

        # Skip if no visual on canvas
        bounds = self.scene.get_bounding_box()
        if isinstance(bounds, type(None)):
            return

        # Create box visual        
        box = gfx.BoxHelper()
        box.set_transform_by_aabb(bounds)

        # Add custom attributes
        box._object_type = 'boundingbox'
        box._object_id = uuid.uuid4()

        self.scene.add(box)

    def center_camera(self):
        """Center camera on visuals."""
        self.camera.show_object(self.scene, scale=1.1, view_dir=(0., 0., 1.), up=(0., -1., 0.))

    def add(self, x, center=True, clear=False, **kwargs):
        """Add objects to canvas.

        Parameters
        ----------
        x :         Neuron/List | Dotprops | Volumes | Points | vispy Visuals
                    Object(s) to add to the canvas.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.
        clear :     bool, optional
                    If True, clear canvas before adding new objects.
        **kwargs
                    Keyword arguments passed when generating visuals. See
                    :func:`~navis.plot3d` for options.

        Returns
        -------
        None

        """
        from .objects import neuron2gfx, volume2gfx, points2gfx

        (neurons, volumes, points, visuals) = utils.parse_objects(x)

        if len(set(kwargs) & set(['c', 'color', 'colors'])) > 1:
            raise ValueError('Must not provide colors via multiple arguments')

        if neurons:
            visuals += neuron2gfx(neurons, **kwargs)
        if volumes:
            visuals += volume2gfx(volumes, **kwargs)
        if points:
            visuals += points2gfx(points, **kwargs.get('scatter_kws', {}))

        if not visuals:
            raise ValueError('No visuals created.')

        if clear:
            self.clear()

        # If we're runningg in headless mode (primarily for tests on CI) we will
        # simply not add the objects. Not ideal but it turns out to be very
        # annoying to correctly setup on Github Actions.
        if getattr(config, 'headless', False):
            return

        for v in visuals:
            # Give visuals an _object_id if they don't already have one
            if not hasattr(v, '_object_id'):                
                v._object_id = uuid.uuid4()                
            self.scene.add(v)

        if center:
            self.center_camera()

    def close(self):
        """Close viewer."""
        # Skip if this is headless mode
        if getattr(config, 'headless', False):
            return

        # Clear first to free all visuals
        self.clear()
        if self == getattr(config, 'primary_viewer', None):
            del config.primary_viewer

        # Close if not already closed
        if not self.canvas.is_closed():
            self.canvas.close() 

    def hide_neurons(self, n):
        """Hide given neuron(s)."""
        ids = utils.eval_id(n)

        neurons = self.neurons   # grab once to speed things up
        for s in ids:
            for v in neurons[s]:
                if getattr(v, '_pinned', False):
                    continue
                if v.visible:
                    v.visible = False

    def hide_selected(self):
        """Hide currently selected neuron(s)."""
        self.hide_neurons(self.selected)

    def unhide_neurons(self, n=None, check_alpha=False):
        """Unhide given neuron(s).

        Use ``n`` to unhide specific neurons.

        """
        neurons = self.neurons  # grab once to speed things up
        if not isinstance(n, type(None)):
            ids = utils.eval_id(n)
        else:
            ids = list(neurons.keys())

        for s in ids:
            for v in neurons[s]:
                if getattr(v, '_pinned', False):
                    continue
                if not v.visible:
                    v.visible = True
            if check_alpha:
                # Make sure color has an alpha channel
                c = to_rgba(neurons[s][0].color)
                # Make sure alpha is 1
                if c.ndim == 1 and c[3] != 1:
                    c[3] = 1
                    self.set_colors({s: c})
                elif c.ndim == 2 and np.any(c[:, 3] != 1):
                    c[:, 3] = 1
                    self.set_colors({s: c})

        self.update_legend()

    def pin_neurons(self, n):
        """Pin given neuron(s).

        Changes to the color or visibility of pinned neurons are silently
        ignored. You can use this to keep specific neurons visible while
        cycling through the rest - useful for comparisons.

        """
        ids = utils.eval_id(n)

        neurons = self.neurons  # grab only once to speed things up

        for s in ids:
            for v in neurons[s]:
                v._pinned = True

    def unpin_neurons(self, n=None):
        """Unpin given neuron(s).

        Use ``n`` to unhide specific neurons.

        """
        neurons = self.neurons  # grab once to speed things up
        if not isinstance(n, type(None)):
            ids = utils.eval_id(n)
        else:
            ids = list(neurons.keys())

        for s in ids:
            for v in neurons[s]:
                v.unfreeze()
                v._pinned = False
                v.freeze()

    def toggle_neurons(self, n):
        """Toggle neuron(s) visibility."""
        n = utils.make_iterable(n)

        if False not in [isinstance(u, uuid.UUID) for u in n]:
            obj = self._neuron_obj
        else:
            n = utils.eval_id(n)
            obj = self.neurons

        for s in n:
            for v in obj[s]:
                v.visible = v.visible is False

        self.update_legend()

    def toggle_select(self, n):
        """Toggle selected of given neuron."""
        skids = utils.eval_id(n)

        neurons = self.neurons  # grab once to speed things up

        for s in skids:
            if self.selected != s:
                self.selected = s
                for v in neurons[s]:
                    self._selected_color = v.color
                    v.set_data(color=self.highlight_color)
            else:
                self.selected = None
                for v in neurons[s]:
                    v.set_data(color=self._selected_color)

        self.update_legend()

    def set_colors(self, c, include_connectors=False):
        """Set neuron color.

        Parameters
        ----------
        c :      tuple | dict
                 RGB color(s) to apply. Values must be 0-1. Accepted:
                   1. Tuple of single color. Applied to all visible neurons.
                   2. Dictionary mapping skeleton IDs to colors.

        """
        neurons = self.neurons  # grab once to speed things up
        if isinstance(c, (tuple, list, np.ndarray, str)):
            cmap = {s: c for s in neurons}
        elif isinstance(c, dict):
            cmap = c
        else:
            raise TypeError(f'Unable to use colors of type "{type(c)}"')

        for n in neurons:
            if n in cmap:
                for v in neurons[n]:
                    if getattr(v, '_pinned', False):
                        continue
                    if v._neuron_part == 'connectors' and not include_connectors:
                        continue
                    if not hasattr(v, 'material'):
                        continue
                    # Note: there is currently a bug where removing or adding an alpha
                    # channel from a color will break the rendering pipeline
                    if len(v.material.color) == 4:
                        new_c = mcl.to_rgba(cmap[n])
                    else:
                        new_c = mcl.to_rgb(cmap[n])
                    v.material.color = gfx.Color(new_c)


    def set_alpha(self, a, include_connectors=True):
        """Set neuron color alphas.

        Parameters
        ----------
        a :      tuple | dict
                 Alpha value(s) to apply. Values must be 0-1. Accepted:
                   1. Tuple of single alpha. Applied to all visible neurons.
                   2. Dictionary mapping skeleton IDs to alpha.

        """
        neurons = self.neurons  # grab once to speed things up
        if isinstance(a, (tuple, list, np.ndarray, str)):
            amap = {s: a for s in neurons}
        elif isinstance(a, dict):
            amap = a
        else:
            raise TypeError(f'Unable to use colors of type "{type(a)}"')

        for n in neurons:
            if n in amap:
                for v in neurons[n]:
                    if getattr(v, '_pinned', False):
                        continue
                    if v._neuron_part == 'connectors' and not include_connectors:
                        continue
                    
                    this_c = np.asarray(v.material.color.rgba)

                    # For arrays of colors
                    if this_c.ndim == 2:
                        # If no alpha channel yet, add one
                        if this_c.shape[1] == 3:
                            this_c = np.insert(this_c,
                                               3,
                                               np.ones(this_c.shape[0]),
                                               axis=1)

                        # If already the correct alpha value
                        if np.all(this_c[:, 3] == amap[n]):
                            continue
                        else:
                            this_c[:, 3] = amap[n]
                    else:
                        if len(this_c) == 4 and this_c[3] == amap[n]:
                            continue
                        else:
                            this_c = tuple([this_c[0], this_c[1], this_c[2], amap[n]])

                    if isinstance(v, scene.visuals.Mesh):
                        v.color = this_c
                    else:
                        v.set_data(color=this_c)

    def colorize(self, palette='hls', include_connectors=False):
        """Colorize neurons using a seaborn color palette."""
        neurons = self.neurons  # grab once to speed things up
        colors = sns.color_palette(palette, len(neurons))
        cmap = {s: colors[i] for i, s in enumerate(neurons)}

        self.set_colors(cmap, include_connectors=include_connectors)

    def set_bgcolor(self, c):
        """Set background color."""
        if getattr(config, 'headless', False):
            return
        self._background.set_colors(mcl.to_rgba(c))

    def _cycle_neurons(self, increment):
        """Cycle through neurons."""
        self._cycle_index += increment

        # If mode is 'hide' cycle over all neurons
        neurons = self.neurons  # grab once to speed things up
        if self._cycle_mode == 'hide':
            to_cycle = neurons
        # If mode is 'alpha' ignore all hidden neurons
        elif self._cycle_mode == 'alpha':
            # Make sure to keep the order
            to_cycle = OrderedDict()
            for s in self.visible:
                to_cycle[s] = neurons[s]
        else:
            raise ValueError(f'Unknown cycle mode "{self._cycle_mode}".')

        if self._cycle_index < 0:
            self._cycle_index = len(to_cycle) - 1
        elif self._cycle_index > len(to_cycle) - 1:
            self._cycle_index = 0

        to_hide = [n for i, n in enumerate(to_cycle) if i != self._cycle_index]
        to_show = [list(to_cycle.keys())[self._cycle_index]]

        # Depending on background color, we have to use different alphas
        v = self.canvas.bgcolor.hsv[2]
        out_alpha = .05 + .2 * v

        if self._cycle_mode == 'hide':
            self.hide_neurons(to_hide)
            self.unhide_neurons(to_show)
        elif self._cycle_mode == 'alpha':
            # Get current colors
            new_amap = {}
            for n in to_cycle:
                this_c = np.asarray(to_cycle[n][0].color)

                if this_c.ndim == 2:
                    if this_c.shape[1] == 4:
                        this_a = this_c[0, 3]
                    else:
                        this_a = 1
                else:
                    if this_c.shape[0] == 4:
                        this_a = this_c[3]
                    else:
                        this_a = 1

                # If neuron needs to be hidden, add to cmap
                if n in to_hide and this_a != out_alpha:
                    new_amap[n] = out_alpha
                elif n in to_show and this_a != 1:
                    new_amap[n] = 1
            self.set_alpha(new_amap)
        else:
            raise ValueError(f'Unknown cycle mode: "{self._cycle_mode}". Use '
                             '"hide" or "alpha"!')

        self.active_neuron = to_show

        # Generate names
        names = []
        for u in to_show:
            n = getattr(neurons[u][0], "name", "NA")
            if not isinstance(u, uuid.UUID):
                n += f' ({u})'
            names.append(n)

        self._data_text.text = f'{"|".join(names)}' \
                               f' [{self._cycle_index + 1}/{len(neurons)}]'

    def _toggle_fps(self):
        """Switch FPS measurement on and off."""
        self._show_fps = not self._show_fps

    def screenshot(self, filename='screenshot.png', pixel_scale=2,
                   alpha=True):
        """Save a screenshot of this viewer.

        Parameters
        ----------
        filename :      str, optional
                        Filename to save to.
        pixel_scale :   int, optional
                        Factor by which to scale canvas. Determines image
                        dimensions.
        alpha :         bool, optional
                        If True, will export transparent background.

        """
        im = self._screenshot(alpha=alpha)
        png.from_array(im.reshape(im.shape[0], im.shape[1] * im.shape[2]), mode='RGBA').save(filename)

    def _screenshot(self, alpha=True):
        """Return image array for screenshot."""
        if alpha:
            op = self._background.opacity 
            self._background.opacity = 0
        try:
            im = self.renderer.snapshot()
        except BaseException:
            raise
        finally: 
            if alpha:
                self._background.opacity = op           

        return im

    def set_view(self, view):
        """(Re-)set camera position.

        Parameters
        ----------
        view :      XY | XZ | YZ

        """
        if view == 'XY':
            self.camera.show_object(self.scene, view_dir=(0., 0., 1.), up=(0., -1., 0.))
            #state['rotation'] = np.array([0., 0., 0., 1.])
            #state['reference_up'] = np.array([0., 1., 0.])
        elif view == 'XZ':
            self.camera.show_object(self.scene, scale=1, view_dir=(0., 1., 0.), up=(0., 0., 1.))
        elif view == 'YZ':
            self.camera.show_object(self.scene, scale=1, view_dir=(-1., 0., 0.), up=(0., -1., 0.))
        else:
            raise TypeError(f'Unable to set view from {type(view)}')



def to_rgba(c, alpha=None):
    """Convert color or array of colors to RGBA.

    matplotlib.colors.to_rgba can't deal with vispy color arrays.
    """
    # Vispy color arrays (used on meshes) have an _rgba property
    if hasattr(c, '_rgba'):
        c = c._rgba

    # Make sure we deal with an array
    c = np.asarray(c)

    if c.ndim == 2:
        if c.shape[1] == 3:
            c = np.insert(c,
                          3,
                          np.ones(c.shape[0]),
                          axis=1)

        if not isinstance(alpha, type(None)):
            c[:, 3] = alpha
    elif c.ndim == 1:
        if c.shape[0] == 3:
            c = np.insert(c, 3, 1)

        if not isinstance(alpha, type(None)):
            c[3] = alpha
    else:
        raise ValueError(f'Got {c.ndim} dimensional array of colors.')

    return c
