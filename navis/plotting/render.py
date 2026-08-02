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

"""Rasterize a 3D scene into a matplotlib axes.

[`navis.plot3d`][] with `snapshot=True` renders the scene through octarine's
offscreen canvas and hands back a matplotlib `(fig, ax)` instead of an
interactive viewer. The image is placed in *data* coordinates, so whatever you
add afterwards - labels, arrows, scale bars, scatter overlays - can be
positioned in the neurons' own coordinate system.

That only works if the camera looks straight down one of the data axes (which
is what the `view` parameter gives you). For any other camera the image falls
back to world units in the view plane - distances are still to scale, they just
no longer correspond to a data axis.

"""

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .. import config
from .dd import _view_frame

logger = config.get_logger(__name__)

# Canvas size (in px) used when the caller does not ask for one. Only the
# longest side is honoured - the other is derived from the scene, see
# `fit_canvas`.
DEFAULT_SIZE = 800

# Same view octarine's own `center_camera` uses, so that a render looks like the
# interactive plot3d rather than like plot2d (whose default is ("x", "y")).
DEFAULT_VIEW = ("x", "-y")

# Extreme aspect ratios make for unreadable figures, so `fit_canvas` stays
# within this range.
MAX_ASPECT = 4

# The camera math leaves some float noise, so a view axis counts as aligned
# while the off-axis components stay below this fraction of the main one.
AXIS_TOL = 1e-4

AXES = ("x", "y", "z")


def parse_view(view):
    """Turn a view specification into `(view_dir, up)` for a pygfx camera.

    Takes the same `("x", "-y")` axis pairs as [`navis.plot2d`][] - the first
    entry is the axis pointing right, the second the one pointing up - plus a
    string shorthand for the same thing (`"xy"`, `"x-z"`, `"-yz"`).

    """
    if isinstance(view, str):
        # A partial match is no match - "xyz" must not come through as ("x", "y")
        match = re.fullmatch(r"([+-]?[xyz])([+-]?[xyz])", view.lower())
        view = match.groups() if match else (view,)

    view = tuple(str(v).lower() for v in view)
    axes = tuple(v.lstrip("+-") for v in view)
    if len(view) != 2 or not set(axes) <= set(AXES) or axes[0] == axes[1]:
        raise ValueError(
            f"`view` must be two different axes out of {AXES}, each optionally "
            f'signed - e.g. ("x", "-y") or "x-y". Got {view}.'
        )

    # `_view_frame` is shared with plot2d so that a given `view` produces the
    # same picture in both. `w` points out of the screen, and the camera looks
    # the other way.
    _, up, w = _view_frame(view)

    return tuple(-w), tuple(up)


def set_view(viewer, view, margin=0.05):
    """Point the viewer's camera at the scene.

    Parameters
    ----------
    viewer :    octarine.Viewer
    view :      str | tuple | dict
                Either something `parse_view` understands, or a camera state
                dict as returned by `viewer.get_view()`.
    margin :    float
                Padding around the scene, as a fraction of its size. Ignored
                for camera state dicts, which already say how much to show.

    """
    if isinstance(view, dict):
        viewer.camera.set_state(view)
        return

    view_dir, up = parse_view(view)

    # `match_aspect` fits the scene's bounding box rather than its bounding
    # sphere, which is what makes the framing tight (a sphere has to be big
    # enough for the box to rotate inside it)
    viewer.camera.show_object(
        viewer.scene, view_dir=view_dir, up=up, match_aspect=True
    )
    viewer.camera.zoom = 1 / (1 + margin)


def fit_canvas(viewer, size=DEFAULT_SIZE):
    """Resize the canvas so the scene fills it, longest side `size` px."""
    width, height = viewer.camera.width, viewer.camera.height
    if not width or not height:
        return

    aspect = np.clip(width / height, 1 / MAX_ASPECT, MAX_ASPECT)
    viewer.resize((round(size * min(aspect, 1)), round(size / max(aspect, 1))))


def image_extent(camera, shape, focus=None):
    """Work out which slab of the world the rendered image covers.

    Parameters
    ----------
    camera :    pygfx camera
                Must have been used for a render already - the projection only
                knows about the viewport's aspect ratio after a draw.
    shape :     tuple
                `(height, width)` of the image in pixels.
    focus :     (3, ) array, optional
                World position to measure at. Only matters for a perspective
                camera, where the extent grows with distance - there it should
                be the middle of the scene. Ignored for orthographic cameras,
                which are the same at any depth.

    Returns
    -------
    extent :    tuple
                `(left, right, bottom, top)` for `ax.imshow(..., origin="upper")`.
                Any of these can be inverted (`left > right`) - that is how a
                flipped view such as `("x", "-y")` is expressed.
    labels :    tuple
                Axis labels, or `(None, None)` if the extent is not in data
                coordinates.
    aspect :    float
                Value for `ax.set_aspect()` that keeps the image's pixels
                square.

    """
    # pygfx ships with pylinalg; import it lazily because pygfx (via octarine)
    # is a soft dependency and this module is imported with navis.
    import pylinalg as la

    # Which depth to measure at. NDC z = 0 sits close to the near plane, which
    # for a perspective camera is nowhere near the scene - so put the plane on
    # `focus` instead. An orthographic projection doesn't care either way.
    ndc_z = 0.0 if focus is None else la.vec_transform(focus, camera.camera_matrix)[2]

    def unproject(x, y):
        """NDC -> world, on the plane the camera focuses on."""
        return la.vec_unproject(
            np.array([x, y], dtype=float), camera.camera_matrix, depth=ndc_z
        )

    # The image centre plus the two vectors spanning half its width/height. In
    # NDC, x/y run -1 to 1 with y pointing up.
    center = unproject(0, 0)
    u = unproject(1, 0) - center  # screen right
    v = unproject(0, 1) - center  # screen up

    x_ix, y_ix = _dominant_axis(u), _dominant_axis(v)

    if x_ix is not None and y_ix is not None and not camera.fov:
        # Camera looks down a data axis: the image is a (possibly flipped) slice
        # of the data's own coordinate system.
        extent = (
            center[x_ix] - u[x_ix],
            center[x_ix] + u[x_ix],
            center[y_ix] - v[y_ix],  # bottom of the image is NDC y = -1
            center[y_ix] + v[y_ix],
        )
        labels = (AXES[x_ix], AXES[y_ix])
    else:
        # Oblique (or perspective) camera: keep world units so that distances
        # and scale bars still work, but centre them on the camera's target
        # because they no longer belong to any one data axis.
        logger.info(
            "Image is in world units in the view plane rather than in data "
            "coordinates - that needs `camera='ortho'` and an axis-aligned "
            "`view`. Distances are still to scale."
        )
        w, h = np.linalg.norm(u), np.linalg.norm(v)
        extent = (-w, w, -h, h)
        labels = (None, None)

    height, width = shape[:2]
    # `aspect` is how much longer one data unit is on the y axis than on the x
    # axis. Equal world-units-per-pixel in both directions -> 1.
    aspect = (abs(extent[1] - extent[0]) / width) / (
        abs(extent[3] - extent[2]) / height
    )

    return extent, labels, aspect


def _scene_center(viewer):
    """Middle of everything in the scene, or `None` if there is nothing in it.

    Note that `Viewer.bounds` looks like the obvious thing to use here but only
    covers visuals that carry a `_bounds` attribute - it comes back empty for
    e.g. a plain `add_neurons()` scene.

    """
    bbox = viewer.scene.get_world_bounding_box()
    if bbox is None:
        return None
    return np.asarray(bbox).mean(axis=0)


def _dominant_axis(vec):
    """Index of the axis `vec` runs along, or `None` if it is oblique."""
    mag = np.abs(vec)
    ix = int(np.argmax(mag))
    if not mag[ix] or (mag.sum() - mag[ix]) > AXIS_TOL * mag[ix]:
        return None
    return ix


def to_mpl(viewer, settings, view=None, fit=False):
    """Point the camera, render `viewer` and place the image on an axes.

    Parameters
    ----------
    viewer :    octarine.Viewer
                Expected to already hold everything that should be in the
                picture.
    settings :  OctarineSettings
    view :      str | tuple | dict, optional
                Passed to `set_view`. `None` renders the camera as it is.
    fit :       bool
                Whether to also resize the canvas to the scene's aspect ratio.
                Only meaningful together with `view`, which is what sets it.

    Returns
    -------
    fig, ax

    """
    if view is not None:
        set_view(viewer, view, margin=settings.margin)
        if fit:
            fit_canvas(viewer)

    if viewer._is_offscreen:
        # An offscreen canvas only draws when asked to, and `screenshot()` asks
        # through the callback that `show()` registers. No window is opened.
        viewer.show()

    if settings.bgcolor is not None:
        viewer.set_bgcolor(settings.bgcolor)

    # A transparent background composits onto whatever the figure uses. With an
    # explicit `bgcolor` we obviously want to keep it.
    img = viewer.screenshot(
        filename=None,
        alpha=settings.bgcolor is None,
        # `screenshot` restores the renderer's own ratio afterwards, so this
        # does not leak into a viewer we were handed
        pixel_ratio=settings.pixel_ratio,
    )

    # Note: this has to happen *after* the screenshot - the camera only learns
    # about the viewport's aspect ratio when it is drawn. The scene centre is
    # only needed for a perspective camera, where the extent grows with depth.
    camera = viewer.camera
    extent, labels, aspect = image_extent(
        camera, img.shape, focus=_scene_center(viewer) if camera.fov else None
    )

    if settings.ax is not None:
        if not isinstance(settings.ax, mpl.axes.Axes):
            raise TypeError(f'`ax` must be matplotlib Axes, got "{type(settings.ax)}"')
        ax = settings.ax
        fig = ax.get_figure()
    else:
        # `figaspect` matches the render's aspect ratio, so the image does not
        # end up letterboxed in its own figure, and clamps the extremes
        figsize = settings.figsize if settings.figsize else plt.figaspect(img)
        fig, ax = plt.subplots(figsize=figsize, dpi=settings.dpi)

    ax.imshow(img, extent=extent, origin="upper", aspect=aspect, zorder=0)

    if settings.hide_axes:
        ax.set_axis_off()
    elif labels[0] is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    return fig, ax
