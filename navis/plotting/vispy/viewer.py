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

import colorsys
import platform
import scipy.spatial
import png
import uuid

import matplotlib.colors as mcl
import numpy as np
import seaborn as sns
import trimesh as tm

from functools import wraps
from vispy import scene
from vispy.util.quaternion import Quaternion

from collections import OrderedDict

from ... import utils, config
from ..colors import *
from .vputils import *
from .visuals import *

__all__ = ['Viewer']

logger = config.logger


def block_all(function):
    """ Decorator to block all events on canvas and view while changes
    are being made. """
    @wraps(function)
    def wrapper(*args, **kwargs):
        viewer = args[0]
        viewer.canvas.events.block_all()
        viewer.view3d.events.block_all()
        try:
            # Execute function
            res = function(*args, **kwargs)
        except BaseException:
            raise
        finally:
            viewer.canvas.events.unblock_all()
            viewer.view3d.events.unblock_all()
        # Return result
        return res
    return wrapper


def block_canvas(function):
    """ Decorator to block all events on canvas while changes are being made.
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        viewer = args[0]
        viewer.canvas.events.block_all()
        try:
            # Execute function
            res = function(*args, **kwargs)
        except BaseException:
            raise
        finally:
            viewer.canvas.events.unblock_all()
        # Return result
        return res
    return wrapper


class Viewer:
    """Vispy 3D viewer.

    Parameters
    ----------
    picking :   bool, default = False
                If ``True``, allow selecting neurons by shift-clicking on
                neurons and placing a 3D cursor via control-click (for OSX:
                command-click).
    **kwargs
              Keyword arguments passed to ``vispy.scene.SceneCanvas``.

    Attributes
    ----------
    picking :       bool,
                    Set to ``True`` to allow picking via shift-clicking.
    selected :      np.array
                    List of currently selected neurons. Can also be used to
                    set the selection.
    show_legend :   bool
                    Set to ``True`` or press ``L`` to show legend. This may
                    impact performance.
    legend_font_size : int
                    Font size for legend.

    Examples
    --------
    This viewer is what :func:`navis.plot3d` uses when ``backend='vispy'``.
    Instead of :func:`navis.plot3d` we can interact with the viewer directly:

    >>> # Open a 3D viewer
    >>> import navis
    >>> v = navis.Viewer()
    >>> # Close the 3D viewer
    >>> v.close()

    You can change the background color from the start or on-the-go:

    >>> # Set background to green
    >>> v = navis.Viewer(bgcolor='green')
    >>> # Set background back to white
    >>> v.canvas.bgcolor = (1, 1, 1)
    >>> # Alternative to v.close():
    >>> navis.close3d()

    """

    def __init__(self, picking=False, **kwargs):
        # Update some defaults as necessary
        defaults = dict(keys=None,
                        show=True,
                        title='navis Viewer',
                        bgcolor='black')
        defaults.update(kwargs)

        if getattr(config, 'headless'):
            defaults['show'] = False

        # Set border rim -> this depends on how the framework (e.g. QT5)
        # renders the window
        self._rim_bot = 15
        self._rim_top = 20
        self._rim_left = 10
        self._rim_right = 10

        # Generate canvas
        self.canvas = scene.SceneCanvas(**defaults)

        """
        from PyQt5.QtWidgets import QPushButton

        # Create canvas
        button = QPushButton('PyQt5 button', self.canvas.native)
        button.move(10, 10)
        self.canvas.show()
        """

        # Add and setup 3d view
        self.view3d = self.canvas.central_widget.add_view()
        self.camera3d = scene.ArcballCamera()
        self.view3d.camera = self.camera3d

        # Add permanent overlays
        self.overlay = self._draw_overlay()

        self.canvas.unfreeze()
        self.canvas._overlay = self.overlay
        self.canvas._view3d = self.view3d
        self.canvas._wrapper = self
        self.canvas.freeze()

        # Add picking functionality
        if picking:
            self.picking = True
        else:
            self.picking = False

        # Set cursor_pos to None
        self.cursor_pos = None

        # Add keyboard shortcuts
        self.canvas.connect(on_key_press)

        # Add resize control to keep overlay in position
        self.canvas.connect(on_resize)

        # Legend settings
        self.__show_legend = False
        self.__selected = np.array([], dtype='object')
        self._cycle_index = -1
        self.__legend_font_size = 7

        # Color to use when selecting neurons
        self.highlight_color = (1, .9, .6)

        # Keep track of initial camera position
        self._camera_default = self.view3d.camera.get_state()

        # Cycle mode can be 'hide' or 'alpha'
        self._cycle_mode = 'alpha'

        # Cursors
        self._cursor = None
        self._picking_radius = 20

        # Other stuff
        self._show_bounds = False
        self._show_axes = False

    def _draw_overlay(self):
        overlay = scene.widgets.ViewBox(parent=self.canvas.scene)
        self.view3d.add_widget(overlay)

        """
        # Legend title
        t = scene.visuals.Text('Legend', pos=(10,10),
                                  anchor_x='left', name='permanent',
                                  parent=overlay,
                                  color=(0,0,0), font_size=9)
        """

        # Text color depends on background color
        v = self.canvas.bgcolor.hsv[2]
        text_color = colorsys.hsv_to_rgb(0, 0, 1 - v)

        # Keyboard shortcuts
        self._key_shortcuts = {'O': 'toggle overlay',
                               'L': 'toggle legend',
                               'P': 'toggle picking',
                               'Q/W': 'cycle neurons',
                               'U': 'unhide all',
                               'B': 'bounding box',
                               'F': 'show/hide FPS',
                               '1': 'XY',
                               '2': 'XZ',
                               '3': 'YZ'}

        shorts_text = 'SHORTCUTS: ' + ' | '.join([f"<{k}> {v}" for k, v in self._key_shortcuts.items()])
        self._shortcuts = scene.visuals.Text(shorts_text,
                                             pos=(self._rim_left,
                                                  overlay.size[1] - self._rim_bot),
                                             anchor_x='left',
                                             anchor_y='bottom',
                                             name='permanent',
                                             method='gpu',
                                             parent=overlay,
                                             color=text_color,
                                             font_size=6)

        # FPS (hidden at start)
        self._fps_text = scene.visuals.Text('FPS',
                                            pos=(overlay.size[0] / 2,
                                                 self._rim_top),
                                            anchor_x='center',
                                            anchor_y='top',
                                            name='permanent',
                                            method='gpu',
                                            parent=overlay,
                                            color=(0, 0, 0), font_size=6)
        self._fps_text.visible = False

        # Picking shortcuts (hidden at start)
        self._picking_shortcuts = {'LMB @legend': 'show/hide neuron',
                                   'SHIFT+LMB @neuron': 'select neuron',
                                   'D': 'deselect all',
                                   'H': 'hide selected',
                                   'C': 'url to cursor'}
        # Add platform-specific modifiers
        if platform.system() == 'darwin':
            self._picking_shortcuts['CMD+LMB'] = 'set cursor'
        else:
            self._picking_shortcuts['CTRL+LMB'] = 'set cursor'

        shorts_text = 'PICKING: ' + ' | '.join(['<{k}> {v}' for k, v in self._picking_shortcuts.items()])
        self._picking_text = scene.visuals.Text(shorts_text,
                                                pos=(self._rim_left,
                                                     overlay.size[1] - self._rim_bot - 10),
                                                anchor_x='left',
                                                anchor_y='bottom',
                                                name='permanent',
                                                method='gpu',
                                                parent=overlay,
                                                color=text_color,
                                                font_size=6)
        self._picking_text.visible = False

        # Text box in top right to display arbitrary data
        self._data_text = scene.visuals.Text('',
                                             pos=(overlay.size[0] - self._rim_right,
                                                  self._rim_top),
                                             anchor_x='right',
                                             anchor_y='top',
                                             name='permanent',
                                             method='gpu',
                                             parent=overlay,
                                             color=text_color,
                                             font_size=6)

        return overlay

    @property
    def show_legend(self):
        """Set to ``True`` to hide neuron legend."""
        return self.__show_legend

    @show_legend.setter
    def show_legend(self, v):
        if not isinstance(v, bool):
            raise TypeError(f'Need boolean, got "{type(v)}"')

        if v != self.show_legend:
            self.__show_legend = v
            # Make sure changes take effect
            self.update_legend()

    @property
    def legend_font_size(self):
        """Change legend's font size."""
        return self.__legend_font_size

    @legend_font_size.setter
    def legend_font_size(self, val):
        self.__legend_font_size = val
        if self.show_legend:
            self.update_legend()

    @property
    def picking(self):
        """Set to ``True`` to allow picking."""
        return self.__picking

    def toggle_picking(self):
        """Toggle picking and overlay text."""
        if self.picking:
            self.picking = False
            self._picking_text.visible = False
        else:
            self.picking = True
            self._picking_text.visible = True

    @picking.setter
    def picking(self, v):
        if not isinstance(v, bool):
            raise TypeError(f'Need bool, got {type(v)}')

        self.__picking = v

        if self.picking:
            self.canvas.connect(on_mouse_press)
        else:
            self.canvas.events.mouse_press.disconnect(on_mouse_press)

    def _render_fb(self, crop=None):
        """Render framebuffer."""
        if not crop:
            crop = (0, 0,
                    self.canvas.size[0] * self.canvas.pixel_scale,
                    self.canvas.size[1] * self.canvas.pixel_scale)

        # We have to temporarily deactivate the overlay and view3d
        # otherwise we won't be able to see what's on the 3D or might
        # see holes in the framebuffer
        self.view3d.interactive = False
        self.overlay.interactive = False
        p = self.canvas._render_picking(crop=crop)
        self.view3d.interactive = True
        self.overlay.interactive = True
        return p

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
        return [s for s in neurons if getattr(neurons[s][0], 'pinned', False)]

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
        """List of all 3D visuals on this canvas."""
        return [v for v in self.view3d.children[0].children if isinstance(v, scene.visuals.VisualNode)]

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
        obj_ids = [getattr(v, '_object_id') for v in self.visuals]
        return sorted(set(obj_ids), key=lambda x: obj_ids.index(x))

    @property
    def objects(self):
        """Ordered dictionary {uuid->[visuals]} of all objects in order of addition."""
        objects = OrderedDict()
        for ob in self._object_ids:
            objects[ob] = [v for v in self.visuals if getattr(v, '_object_id') == ob]

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

    def clear_legend(self):
        """Clear legend."""
        # Clear legend except for title
        for l in [l for l in self.overlay.children if isinstance(l, scene.visuals.Text) and l.name != 'permanent']:
            l.parent = None

    def clear(self):
        """Clear canvas."""
        for v in self.visuals:
            v.parent = None

        # `remove_bounds` set this to False but
        # here we want the current setting to persist
        show_bounds = self.show_bounds

        self.remove_bounds()
        self.clear_legend()

        self.show_bounds = show_bounds

    def remove(self, to_remove):
        """Remove given neurons/visuals from canvas."""
        to_remove = utils.make_iterable(to_remove)

        neurons = self.neurons  # grab this only once to speed things up
        for vis in to_remove:
            if isinstance(vis, scene.visuals.VisualNode):
                vis.parent = None
            else:
                uuids = utils.eval_id(to_remove)
                for u in uuids:
                    for v in neurons.get(u, []):
                        v.parent = None

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

    @block_canvas
    def update_bounds(self, color='w', width=1):
        """Update bounding box visual."""
        # Remove any existing visual
        self.remove_bounds()

        bounds = self.bounds
        self._show_bounds = True

        # Skip if no visual on canvas
        if isinstance(bounds, type(None)):
            return

        # Create box visual
        dims = bounds[:, 1] - bounds[:, 0]
        center = bounds.mean(axis=1)
        box = tm.primitives.Box(extents=dims).apply_scale(1.1)

        # Recenter vertices
        vertices = np.array(box.vertices) + center
        connect = np.array([[0, 1], [0, 2], [0, 4],
                            [1, 3], [1, 5],
                            [2, 3], [2, 6],
                            [3, 7],
                            [4, 5], [4, 6],
                            [5, 7],
                            [6, 7]])

        box = scene.visuals.Line(pos=vertices,
                                 color=mcl.to_rgb(color),
                                 # Can only be used with method 'agg'
                                 width=width,
                                 connect=connect,
                                 antialias=True,
                                 name='BoundingBox',
                                 method='gl')

        # Add custom attributes
        box.unfreeze()
        box._object_type = 'boundingbox'
        box._object_id = uuid.uuid4()
        box.freeze()

        self.view3d.add(box)

    @block_canvas
    def update_legend(self):
        """Update legend."""
        # Get existing labels
        labels = {l._object_id: l for l in self.overlay.children if getattr(l, '_object_id', None)}

        # If legend is not meant to be shown, make sure everything is hidden and return
        if not self.show_legend:
            for v in labels.values():
                if v.visible:
                    v.visible = False
            return
        else:
            for v in labels.values():
                if not v.visible:
                    v.visible = True

        # Labels to be removed
        neuron_obj = self._neuron_obj  # grab only once to speed things up
        to_remove = [s for s in labels if s not in neuron_obj]
        for s in to_remove:
            labels[s].parent = None

        # Generate new labels
        to_add = [s for s in neuron_obj if s not in labels]
        for s in to_add:
            # Fallback is name or in lieu of that the object's type
            lbl = getattr(neuron_obj[s][0], '_name',
                          str(type(neuron_obj[s][0])))
            # See if we find a "label" property
            if hasattr(neuron_obj[s][0], '_object'):
                if hasattr(neuron_obj[s][0]._object, 'label'):
                    lbl = neuron_obj[s][0]._object.label

            txt = scene.visuals.Text(lbl,
                                     anchor_x='left',
                                     anchor_y='top',
                                     parent=self.overlay,
                                     method='gpu',
                                     font_size=self.legend_font_size)
            txt.interactive = True
            txt.unfreeze()
            txt._object_id = s
            txt._id = neuron_obj[s][0]._id
            txt.freeze()

        # Position and color labels
        labels = {l._object_id: l for l in self.overlay.children if getattr(
            l, '_object_id', None)}
        for i, s in enumerate(sorted(neuron_obj)):
            if neuron_obj[s][0].visible:
                color = neuron_obj[s][0].color
            else:
                color = (.3, .3, .3)

            offset = 10 * (self.legend_font_size / 7)

            labels[s].pos = (10, offset * (i + 1))
            labels[s].color = color
            labels[s].font_size = self.legend_font_size

    def toggle_overlay(self):
        """Toggle legend on and off."""
        self.overlay.visible = self.overlay.visible is False

    def center_camera(self):
        """Center camera on visuals."""
        visuals = self.visuals  # Get this only once to speed things up
        if not visuals:
            return

        xbounds = np.array([v.bounds(0) for v in visuals]).flatten()
        ybounds = np.array([v.bounds(1) for v in visuals]).flatten()
        zbounds = np.array([v.bounds(2) for v in visuals]).flatten()

        self.camera3d.set_range((xbounds.min(), xbounds.max()),
                                (ybounds.min(), ybounds.max()),
                                (zbounds.min(), zbounds.max()))

    def add(self, x, center=True, clear=False, as_group=False, **kwargs):
        """Add objects to canvas.

        Parameters
        ----------
        x :         Neuron/List | Dotprops | Volumes | Points | vispy Visuals
                    Object(s) to add to the canvas.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.
        clear :     bool, optional
                    If True, clear canvas before adding new objects.
        as_group :  bool, optional
                    If True, will try combining similar objects into a single
                    visual. This reduces the number of shader programs and
                    can greatly increase tthe frame rate. Downside: objects can
                    no longer be individually manipulated.
        **kwargs
                    Keyword arguments passed when generating visuals. See
                    :func:`~navis.plot3d` for options.

        Returns
        -------
        None

        """
        (neurons, volumes, points, visuals) = utils.parse_objects(x)

        if len(set(kwargs) & set(['c', 'color', 'colors'])) > 1:
            raise ValueError('Must not provide colors via multiple arguments')

        if neurons:
            visuals += neuron2vispy(neurons, **kwargs)
        if volumes:
            visuals += volume2vispy(volumes, **kwargs)
        if points:
            visuals += points2vispy(points, **kwargs.get('scatter_kws', {}))

        if not visuals:
            raise ValueError('No visuals created.')

        if clear:
            self.clear()

        if as_group:
            visuals = combine_visuals(visuals, kwargs.get('name'))

        for v in visuals:
            self.view3d.add(v)

        if center:
            self.center_camera()

        if self.show_legend:
            self.update_legend()

        if self.show_bounds:
            self.update_bounds()

    def show(self):
        """Show viewer."""
        # This is for e.g. headless testing
        if not getattr(config, 'headless', False):
            self.canvas.show()
        else:
            logger.info("Viewer widget not shown - navis running in headless mode. ")

    def close(self):
        """Close viewer."""
        if self == getattr(config, 'primary_viewer', None):
            del config.primary_viewer
        self.canvas.close()

    def hide_neurons(self, n):
        """Hide given neuron(s)."""
        ids = utils.eval_id(n)

        neurons = self.neurons   # grab once to speed things up
        for s in ids:
            for v in neurons[s]:
                if getattr(v, 'pinned', False):
                    continue
                if v.visible:
                    v.visible = False

        self.update_legend()

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
                if getattr(v, 'pinned', False):
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
                v.unfreeze()
                v.pinned = True
                v.freeze()

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
                v.pinned = False
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

    @block_all
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
                    if getattr(v, 'pinned', False):
                        continue
                    if v._neuron_part == 'connectors' and not include_connectors:
                        continue
                    new_c = mcl.to_rgba(cmap[n])
                    if isinstance(v, scene.visuals.Mesh):
                        v.color = new_c
                    else:
                        v.set_data(color=mcl.to_rgba(cmap[n]))

        if self.show_legend:
            self.update_legend()

    @block_all
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
                    if getattr(v, 'pinned', False):
                        continue
                    if v._neuron_part == 'connectors' and not include_connectors:
                        continue
                    try:
                        this_c = v.color.rgba
                    except BaseException:
                        this_c = v.color

                    this_c = np.asarray(this_c)

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

        if self.show_legend:
            self.update_legend()

    def colorize(self, palette='hls', include_connectors=False):
        """Colorize neurons using a seaborn color palette."""
        neurons = self.neurons  # grab once to speed things up
        colors = sns.color_palette(palette, len(neurons))
        cmap = {s: colors[i] for i, s in enumerate(neurons)}

        self.set_colors(cmap, include_connectors=include_connectors)

    def set_bgcolor(self, c):
        """Set background color."""
        self.canvas.bgcolor = c

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

    def _draw_fps(self, fps):
        """Callback for ``canvas.measure_fps``."""
        self._fps_text.text = f'{fps:.2f} FPS'

    def _toggle_fps(self):
        """Switch FPS measurement on and off."""
        if not self._fps_text.visible:
            self.canvas.measure_fps(1, self._draw_fps)
            self._fps_text.visible = True
        else:
            self.canvas.measure_fps(1, None)
            self._fps_text.visible = False

    def _snap_cursor(self, pos, visual, open_browser=False):
        """Snap cursor to clostest vertex of visual."""
        if not getattr(self, '_cursor', None):
            self._cursor = scene.visuals.Arrow(pos=np.array([(0, 0, 0), (1000, 0, 0)]),
                                               color=(1, 0, 0, 1),
                                               arrow_color=(1, 0, 0, 1),
                                               arrow_size=10,
                                               arrows=np.array([[800, 0, 0, 1000, 0, 0]]))

        if not self._cursor.parent:
            self.add(self._cursor, center=False)

        # Get vertices for this visual
        if isinstance(visual, scene.visuals.Line):
            verts = visual.pos
        elif isinstance(visual, scene.visuals.Mesh):
            verts = visual.mesh_data.get_vertices()

        # Map vertices to canvas
        tr = visual.get_transform(map_to='canvas')
        co_on_canvas = tr.map(verts)[:, [0, 1]]

        # Find the closest vertex to this mouse click pos
        tree = scipy.spatial.cKDTree(co_on_canvas)
        dist, ix = tree.query(pos)

        # Map canvas pos back to world coordinates
        self.cursor_pos = np.array(verts[ix])
        self.cursor_active_skeleton = getattr(visual, '_id', None)

        # Generate arrow coords
        vec_to_center = np.array(self.camera3d.center) - self.cursor_pos
        norm_to_center = vec_to_center / np.sqrt(np.sum(vec_to_center**2))
        start = self.cursor_pos - (norm_to_center * 10000)
        arrows = np.array([np.append(self.cursor_pos - (norm_to_center * 200),
                                     self.cursor_pos - (norm_to_center * 100))])

        self._cursor.set_data(pos=np.array([start, self.cursor_pos]),
                              arrows=arrows)

        logger.debug(f'World coordinates: {self.cursor_pos}')

    def screenshot(self, filename='screenshot.png', pixel_scale=2,
                   alpha=True, hide_overlay=True):
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
        hide_overlay :  bool, optional
                        If True, will hide overlay for screenshot.

        """

        m = self._screenshot(pixel_scale=pixel_scale,
                             alpha=alpha,
                             hide_overlay=hide_overlay)

        im = png.from_array(m, mode='RGBA')
        im.save(filename)

    def _screenshot(self, pixel_scale=2, alpha=True, hide_overlay=True):
        """Return image array for screenshot."""
        if alpha:
            bgcolor = list(self.canvas.bgcolor.rgb) + [0]
        else:
            bgcolor = list(self.canvas.bgcolor.rgb)

        # region = (0, 0, self.canvas.size[0], self.canvas.size[1])
        size = tuple(np.array(self.canvas.size) * pixel_scale)

        if hide_overlay:
            prev_state = self.overlay.visible
            self.overlay.visible = False

        try:
            m = self.canvas.render(size=size, bgcolor=bgcolor)
        except BaseException:
            raise
        finally:
            if hide_overlay:
                self.overlay.visible = prev_state

        return m

    def visuals_at(self, pos):
        """List visuals at given canvas position."""
        # There appears to be some odd y offset - perhaps because of the
        # window's top bar? On OSX this is about 15px
        pos = (pos[0], pos[1] - 15)

        # Map mouse pos to framebuffer
        tr = self.canvas.transforms.get_transform(map_from='canvas',
                                                  map_to='framebuffer')
        pos = tr.map(pos)

        # Render framebuffer in picking mode
        p = self._render_fb(crop=(pos[0] - self._picking_radius / 2,
                                  pos[1] - self._picking_radius / 2,
                                  self._picking_radius,
                                  self._picking_radius))

        logger.debug('Picking framebuffer:')
        logger.debug(p)

        # List visuals in order from distance to center
        ids = []
        seen = set()
        center = (np.array(p.shape) / 2).astype(int)
        for i in range(self._picking_radius * self.canvas.pixel_scale):
            subr = p[center[0] - i: center[0] + i + 1,
                     center[1] - i: center[1] + i + 1]
            subr_ids = set(list(np.unique(subr)))
            ids.extend(list(subr_ids - seen))
            seen |= subr_ids
        visuals = [scene.visuals.VisualNode._visual_ids.get(x, None) for x in ids]

        return [v for v in visuals if v is not None]

    def set_view(self, view):
        """(Re-)set camera position.

        Parameters
        ----------
        view :      XY | XZ | YZ

        """
        if isinstance(view, Quaternion):
            q = view
        elif view == 'XY':
            q = Quaternion(w=0.707, x=0.707, y=0, z=0)
        elif view == 'XZ':
            q = Quaternion(w=1, x=0, y=0, z=0)
        elif view == 'YZ':
            q = Quaternion(w=.5, x=0.5, y=0.5, z=-.5)
        else:
            raise TypeError(f'Unable to set view from {type(view)}')

        self.camera3d._quaternion = q
        # This is necessary to force a redraw
        self.camera3d.set_range()


def on_mouse_press(event):
    """Manage picking on canvas."""
    canvas = event.source
    viewer = canvas._wrapper

    try:
        viewer.interactive = False
        canvas._overlay.interactive = False
        vis_at = viewer.visuals_at([event.pos[0] + 15,
                                    event.pos[1] + 15])
    finally:
        viewer.interactive = True
        canvas._overlay.interactive = True

    logger.debug(f'Mouse press at {event.pos}: {vis_at}')

    modifiers = [key.name for key in event.modifiers]
    if event.modifiers:
        logger.debug(f'Modifiers found: {modifiers}')

    # Iterate over visuals in this canvas at cursor position
    for v in vis_at:
        # Skip views
        if isinstance(v, scene.widgets.ViewBox):
            continue
        # If legend entry, toggle visibility
        elif isinstance(v, scene.visuals.Text):
            viewer.toggle_neurons(v._object_id)
            break
        # If control modifier, try snapping cursor
        if 'Control' in modifiers:
            viewer._snap_cursor(event.pos, v,
                                open_browser='Shift' in modifiers)
            break
        # If shift modifier, add to/remove from current selection
        elif (isinstance(v, scene.visuals.VisualNode)
              and getattr(v, '_id', None)
              and 'Shift' in modifiers):
            if v._id not in set(viewer.selected):
                viewer.selected = np.append(viewer.selected, v._id).astype('object')
            else:
                viewer.selected = viewer.selected[viewer.selected != v._id]
            break


def on_key_press(event):
    """Manage keyboard shortcuts for canvas."""
    canvas = event.source
    viewer = canvas._wrapper

    if event.text.lower() == 'o':
        viewer.toggle_overlay()
    elif event.text.lower() == 'l':
        viewer.show_legend = viewer.show_legend is False
    elif event.text.lower() == 'd':
        viewer.selected = []
    elif event.text.lower() == 'q':
        viewer._cycle_neurons(-1)
    elif event.text.lower() == 'w':
        viewer._cycle_neurons(1)
    elif event.text.lower() == 'h':
        viewer.hide_selected()
    elif event.text.lower() == 'u':
        viewer.unhide_neurons(check_alpha=True)
    elif event.text.lower() == 'f':
        viewer._toggle_fps()
    elif event.text.lower() == 'p':
        viewer.toggle_picking()
    elif event.text.lower() == 'b':
        viewer.toggle_bounds()
    elif event.text.lower() == '1':
        viewer.set_view('XY')
    elif event.text.lower() == '2':
        viewer.set_view('XZ')
    elif event.text.lower() == '3':
        viewer.set_view('YZ')


def on_resize(event):
    """Keep overlay in place upon resize."""
    viewer = event.source._wrapper
    viewer._shortcuts.pos = (10, event.size[1])
    viewer._picking_text.pos = (10, event.size[1] - 10)
    viewer._fps_text.pos = (event.size[0] - 10, 10)

    # Idea for fixing fontsize/linebreaks:
    # Render canvas to framebuffer via `_render_picking` and with region
    # outside the current canvas size: if a text ID shows up, we have to
    # resize


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
