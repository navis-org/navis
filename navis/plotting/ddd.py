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

"""Module contains functions to plot neurons in 3D."""

import os

import numpy as np

from typing import Union, List
from importlib.util import find_spec

from .. import utils, config, core
from . import render
from .colors import prepare_colormap, parse_color_by
from .settings import OctarineSettings, PlotlySettings, K3dSettings
from .viewer_utils import import_octarine

__all__ = ["plot3d"]

logger = config.get_logger(__name__)

# Check if backends are available without importing them
BACKENDS = tuple(b for b in ("octarine", "plotly", "k3d") if find_spec(b) is not None)
JUPYTER_BACKENDS = tuple(b for b in ("plotly", "octarine", "k3d") if b in BACKENDS)
NON_JUPYTER_BACKENDS = tuple(b for b in ("octarine", "plotly") if b in BACKENDS)
AUTO_BACKEND = None  # choose the backend only the first time


def plot3d(
    x: Union[
        core.NeuronObject,
        core.Volume,
        np.ndarray,
        List[Union[core.NeuronObject, np.ndarray, core.Volume]],
    ],
    **kwargs,
):
    """Generate interactive 3D plot.

    Uses either [octarine], [k3d] or [plotly] as backend.
    By default, the choice is automatic depending on what backends
    are installed and the context:

      - Terminal: octarine > plotly
      - Jupyter: plotly > octarine > k3d

    See the `backend` parameter on how to change this behavior.

    [octarine]: https://schlegelp.github.io/octarine/
    [k3d]: https://k3d-jupyter.org/
    [plotly]: http://plot.ly

    Parameters
    ----------
    x :               Neuron/List | Volume | numpy.array | list thereof
                      The object(s) to plot. Can be:
                        - navis neurons, neuronlists or volumes
                        - numpy.array (N,3) is plotted as scatter plot
                        - multiple objects can be passed as list (see examples)
                      See parameters below for ways to customize the plot.

    Object parameters
    -----------------
    color :           None | str | tuple | list | dict, default=None

                      Use single str (e.g. `'red'`) or `(r, g, b)` tuple
                      to give all neurons the same color. Use `list` of
                      colors to assign colors: `['red', (1, 0, 1), ...].
                      Use `dict` to map colors to neurons:
                      `{neuron.id: (r, g, b), ...}`.

    palette :         str | array | list of arrays, default=None

                      Name of a matplotlib or seaborn palette. If `color` is
                      not specified will pick colors from this palette.

    alpha :           float [0-1], optional

                      Alpha value for neurons. Overriden if alpha is provided
                      as color specified in `color` has an alpha channel.

    connectors :      bool | "presynapses" | "postsynapses" | str | list, default=False

                      Plot connectors. This can either be `True` (plot all
                      connectors), `"presynapses"` (only presynaptic connectors)
                      or `"postsynapses"` (only postsynaptic connectors). If
                      a string or a list is provided, it will be used to filter the
                      `type` column in the connectors table.

                      Use these parameters to adjust the way connectors are plotted:

                        - `cn_colors` (str | tuple | dict | "neuron" ) overrides
                          the default connector (e.g. synpase) colors:
                            - single color as str (e.g. `'red'`) or rgb tuple
                              (e.g. `(1, 0, 0)`)
                            - dict mapping the connectors tables `type` column to
                              a color (e.g. `{"pre": (1, 0, 0)}`)
                            - with "neuron", connectors will receive the same color
                              as their neuron
                        - `cn_layout` (dict): Layout of the connectors. See
                          `navis.config.default_connector_colors` for options.
                        - `cn_size` (float): Size of the connectors.
                        - `cn_alpha` (float): Transparency of the connectors.
                        - `cn_mesh_colors` (bool): Whether to color the connectors
                          by the neuron's color.

    connectors_only : bool, default=False

                      Plot only connectors (e.g. synapses) if available and
                      ignore the neurons.

    cn_color_by :     str | array, optional

                      Color connectors by a column of the connector table (or by
                      an array with one value per connector) rather than by their
                      `type`. Numerical data gets a colormap, categorical data one
                      color per level - and either way the scale is shared by all
                      neurons. Missing values are drawn grey.

                      Overrides `cn_colors`/`cn_mesh_colors`. `plotly` and `k3d`
                      draw markers rather than stalks when this is set, since a
                      line there carries a single color.

                      `plotly` and `k3d` only - the octarine backend does not
                      support it.

    cn_palette :      str | list | dict, optional

                      Palette for `cn_color_by`: the name of a colormap for
                      numerical data, and for categorical data a palette name, a
                      list of colors or a dict keyed by level. Falls back to
                      `palette`, then to `"viridis"`.

    color_by :        str | array | list of arrays, default = None

                      Color neurons by a property. Can be:

                        - a list/array of labels, one per each neuron
                        - a neuron property (str)
                        - a column name in the node table of `Skeletons`
                        - a list/array of values for each node

                      Numerical values will be normalized. You can control
                      the normalization by passing a `vmin` and/or `vmax`
                      parameter. Must specify a colormap via `palette`.

    shade_by :        str | array | list of arrays, default=None

                      Similar to `color_by` but will affect only the alpha
                      channel of the color. If `shade_by='strahler'` will
                      compute Strahler order if not already part of the node
                      table (Skeletons only). Numerical values will be
                      normalized. You can control the normalization by passing
                      a `smin` and/or `smax` parameter. Does not work with
                      `k3d` backend.

    radius :          bool | "auto", default=False

                      If "auto" will plot neurites of `Skeletons` with radius
                      if they have radii. If True, will try plotting neurites of
                      `Skeletons` with radius regardless. The radius can be
                      scaled by `linewidth`. Note that this will increase rendering
                      time.

    soma :            bool, default=True

                      Skeletons only: Whether to plot soma if it exists. Size
                      of the soma is determined by the neuron's `.soma_radius`
                      property which defaults to the "radius" column for
                      `Skeletons`.

    linewidth :       float, default=3 for plotly and 1 for all others

                      Skeletons only. Note that with `radius=True` this is a
                      multiplier on the radius rather than a line width, and
                      there it defaults to 1 in every backend - so plotly's
                      default of 3 applies to lines only.

    linestyle :       str, default='-'

                      Skeletons only. Follows the same rules as in matplotlib.

    scatter_kws :     dict, optional

                      Use to modify scatter plots. Accepted parameters are:
                        - `size` to adjust size of dots
                        - `color` to adjust color

    Figure parameters
    -----------------
    backend :         'auto' (default) | 'octarine' | 'plotly' | 'k3d'

                      Which backend to use for plotting. Note that there will
                      be minor differences in what feature/parameters are
                      supported depending on the backend:

                        - `auto` selects backend based on availability and
                          context (see above). You can override this by setting an
                          environment variable e.g. `NAVIS_PLOT3D_BACKEND="plotly"`
                          or `NAVIS_PLOT3D_JUPYTER_BACKEND="k3d"`.
                        - `octarine` uses WGPU to generate high performances
                          interactive 3D plots. Works both terminal and Jupyter.
                        - `plotly` generates 3D plots using WebGL. Works
                          "inline" in Jupyter notebooks but can also produce a
                          HTML file that can be opened in any browers.
                        - `k3d` generates 3D plots using k3d. Works only in
                          Jupyter notebooks!

    hide_axes :       bool, default=True

                      If True (default), hides the axes, ticks and tick labels
                      for a clean render. Set to False to show the coordinate
                      axes. Applies to the `plotly` backend and to
                      `snapshot=True` (see below) - the interactive octarine
                      viewer has no axes to hide.

    **Below parameters are for plotly backend only:**

    fig :             plotly.graph_objs.Figure

                      Pass to add graph objects to existing plotly figure. Will
                      not change layout.

    title :           str, default=None

                      For plotly only! Change plot title.

    width/height :    int, optional

                      Use to adjust figure size.

    fig_autosize :    bool, default=True

                      For plotly only! Autoscale figure size.
                      Attention: autoscale overrides width and height

    hover_name :      bool, default=False

                      If True, hovering over neurons will show their label.

    hover_id :        bool, default=False

                      If True, hovering over skeleton nodes will show their ID.

    legend :          bool, default=True

                      Whether or not to show the legend.

    legend_orientation : "v" (default) | "h"

                      Orientation of the legend. Can be 'h' (horizontal) or 'v'
                      (vertical).

    legend_group :    dict, default=None

                      A dictionary mapping neuron IDs to labels (strings).
                      Use this to group neurons under a common label in the
                      legend.

    inline :          bool, default=True

                      If True and you are in an Jupyter environment, will
                      render plotly/k3d plots inline. If False, will generate
                      and return either a plotly Figure or a k3d Plot object
                      without immediately showing it.

    lighting :        bool | str | dict, default=True

                      For plotly only! Controls surface shading of meshes,
                      somata and volumes:

                        - `True` uses navis' default lit look
                        - `False` (or `"plotly"`) reverts to plotly's near-flat
                          default shading
                        - a preset name: `"default"`/`"studio"`, `"matte"`,
                          `"glossy"` or `"rim"`
                        - a dict passed straight to plotly's `Mesh3d.lighting`
                          (keys: `ambient`, `diffuse`, `specular`, `roughness`,
                          `fresnel`)

                      Related: `lightposition` (dict with x/y/z; by default
                      derived from each mesh's bounding box) and `flatshading`
                      (bool, default=False; `True` gives a faceted low-poly
                      look).

    background :      str | dict, default=None

                      For plotly only! Scene color theme. One of `"light"`
                      (default), `"white"` or `"dark"`, any color string, or a
                      dict overriding individual theme colors (`paper`, `scene`,
                      `axis_bg`, `grid`, `tick`).

    projection :      "perspective" (default) | "orthographic"

                      For plotly only! Camera projection. Orthographic avoids
                      perspective distortion and is often preferred for figures.

    dragmode :        "orbit" (default) | "turntable"

                      For plotly only! How click-dragging rotates the scene.
                      "orbit" rotates freely (trackball-style); "turntable"
                      keeps the z-axis pointing up.

    **Below parameters are for the Octarine backend only:**

    clear :           bool, default = False

                      If True, will clear the viewer before adding the new
                      objects.

    center :          bool, default = True

                      If True, will center camera on the newly added objects.

    size :            (width, height) tuple, optional

                      Use to adjust figure/window size. With `snapshot=True`
                      this is the canvas size in logical pixels - see
                      `pixel_ratio` for the supersampling applied on top. Leave
                      it unset to have the canvas match the scene's proportions.

    show :            bool, default=True

                      Whether to immediately show the viewer.

    snapshot :        bool, default=False

                      If True, will render the scene offscreen and return a
                      matplotlib `(fig, ax)` instead of an interactive viewer.
                      The image is placed in *data* coordinates (see `view`),
                      so you can add labels, arrows or scatter overlays in the
                      neurons' own coordinate system afterwards.

    **Below parameters are for `snapshot=True` only:**

    view :            tuple | str | dict, default="auto"

                      Which way to point the camera. Takes the same `("x", "y")`
                      axis pairs as [`navis.plot2d`][] - first entry is the axis
                      pointing right, second the one pointing up - or the same
                      as a string (`"xy"`, `"x-z"`). Pass a camera state dict
                      (from `viewer.get_view()`) to reproduce a view you set up
                      interactively, or `None` to leave the camera alone.
                      `"auto"` uses `("x", "-y")`, i.e. the same view you get
                      from the interactive viewer.

                      Note that only axis-aligned views can be expressed in data
                      coordinates. For any other camera the axes are in world
                      units in the view plane instead - distances are still to
                      scale but no longer tied to a data axis.

    margin :          float, default=0.05

                      Padding around the scene, as a fraction of its size.

    pixel_ratio :     float, optional

                      Supersampling factor on top of `size`: the image comes
                      back `size * pixel_ratio` pixels across and is scaled back
                      down when drawn, which is what smooths the edges. Defaults
                      to whatever pygfx uses (2 for an offscreen canvas), so
                      pass `1` if you want `size` to mean actual image pixels.

    bgcolor :         str, optional

                      Background color. By default the render is transparent
                      and picks up whatever is underneath it.

    ax :              matplotlib.axes.Axes, optional

                      Axes to draw the render on. If not provided, will create
                      a new figure.

    figsize/dpi :     Use to adjust the matplotlib figure. By default `figsize`
                      matches the aspect ratio of the render.

    Returns
    -------
    If `backend='octarine'`

        From terminal: opens a 3D window and returns :class:`octarine.Viewer`.
        From Jupyter: :class:`octarine.Viewer` displayed in an ipywidget.

        With `snapshot=True`: a matplotlib `(fig, ax)` with the rendered image.

    If `backend='plotly'`

        Returns either `None` if you are in a Jupyter notebook (see also
        `inline` parameter) or a `plotly.graph_objects.Figure`
        (see examples).

    If `backend='k3d'`

        Returns either `None` and immediately displays the plot or a
        `k3d.plot` object that you can manipulate further (see `inline`
        parameter).

    See Also
    --------
    [`octarine.Viewer`](https://schlegelp.github.io/octarine/)
        Interactive 3D viewer.

    [`navis.get_viewer`][]
        Grab the viewer most recently used by `plot3d`.

    Examples
    --------
    >>> import navis

    In a Jupyter notebook using plotly as backend:

    >>> nl = navis.example_neurons()
    >>> # Backend is automatically chosen but we can set it explicitly
    >>> # Plot inline
    >>> nl.plot3d(backend='plotly')                             # doctest: +SKIP
    >>> # Plot as separate html in a new window
    >>> fig = nl.plot3d(backend='plotly', inline=False)
    >>> import plotly.offline
    >>> _ = plotly.offline.plot(fig)                            # doctest: +SKIP

    In a Jupyter notebook using k3d as backend:

    >>> nl = navis.example_neurons()
    >>> # Plot inline
    >>> nl.plot3d(backend='k3d')                                # doctest: +SKIP

    In a terminal using octarine as backend:

    >>> # Plot list of neurons
    >>> nl = navis.example_neurons()
    >>> v = navis.plot3d(nl, backend='octarine')                # doctest: +SKIP
    >>> # Clear canvas
    >>> navis.clear3d()

    Rendering to matplotlib instead of an interactive viewer - the image comes
    back in data coordinates, so you can annotate it as usual:

    >>> nl = navis.example_neurons(2)
    >>> fig, ax = navis.plot3d(nl, snapshot=True, view=('x', '-z'))  # doctest: +SKIP
    >>> soma = nl[0].soma_pos[0]                                   # doctest: +SKIP
    >>> _ = ax.annotate("soma", (soma[0], soma[2]))                # doctest: +SKIP

    Some more advanced examples:

    >>> # plot3d() can deal with combinations of objects
    >>> nl = navis.example_neurons()
    >>> vol = navis.example_volume('LH')
    >>> vol.color = (255, 0, 0, .5)
    >>> # This plots a neuronlists, a single neuron and a volume
    >>> v = navis.plot3d([nl[0:2], nl[3], vol])
    >>> # Clear viewer (works only with octarine)
    >>> v = navis.plot3d(nl, clear=True)

    See the [plotting intro](../../generated/gallery/1a_plotting_general/tutorial_plotting_00_intro)
    for even more examples.

    """
    # Select backend
    backend = kwargs.pop("backend", "auto").lower()
    allowed_backends = ("auto", "octarine", "plotly", "k3d")

    # Rendering to matplotlib is octarine's offscreen canvas doing the work, so
    # there is nothing to choose here
    snapshot = bool(kwargs.get("snapshot"))
    if snapshot:
        if backend not in ("auto", "octarine"):
            raise ValueError(
                f'`snapshot=True` requires the "octarine" backend, got "{backend}".'
            )
        backend = "octarine"

    if backend == "auto":
        global AUTO_BACKEND
        if AUTO_BACKEND is not None:
            backend = AUTO_BACKEND
        else:
            if utils.is_jupyter():
                if not len(JUPYTER_BACKENDS):
                    raise ModuleNotFoundError(
                        "No 3D plotting backends available for Jupyter "
                        "environment. Please install one of the following: "
                        "plotly, octarine, k3d."
                    )
                backend = os.environ.get(
                    "NAVIS_PLOT3D_JUPYTER_BACKEND", JUPYTER_BACKENDS[0]
                )
            else:
                if not len(NON_JUPYTER_BACKENDS):
                    raise ModuleNotFoundError(
                        "No 3D plotting backends available for REPL/script. Please "
                        "install one of the following: octarine, plotly."
                    )
                backend = os.environ.get(
                    "NAVIS_PLOT3D_BACKEND", NON_JUPYTER_BACKENDS[0]
                )

            # Set the backend for the next time
            AUTO_BACKEND = backend

            logger.info(f'Using "{backend}" backend for 3D plotting.')
    elif backend not in allowed_backends:
        raise ValueError(
            f'Unknown backend "{backend}". ' f'Permitted: {", ".join(allowed_backends)}.'
        )
    elif backend not in BACKENDS:
        raise ModuleNotFoundError(
            f'Backend "{backend}" not installed. Please install it via pip '
            "(see https://navis.readthedocs.io/en/latest/source/install.html#optional-dependencies "
            "for more information)."
        )

    if backend == "k3d":
        if not utils.is_jupyter():
            logger.warning("k3d backend only works in Jupyter environments")
        return plot3d_k3d(x, **kwargs)
    elif backend == "plotly":
        return plot3d_plotly(x, **kwargs)
    elif backend == "octarine":
        if snapshot:
            return plot3d_snapshot(x, **kwargs)
        return plot3d_octarine(x, **kwargs)
    else:
        raise ValueError(
            f'Unknown backend "{backend}". ' f'Permitted: {", ".join(allowed_backends)}.'
        )


def plot3d_octarine(x, **kwargs):
    """Plot3d() helper function to generate octarine 3D plots.

    This is just to improve readability. Its only purpose is to find the
    existing viewer or generate a new one.

    """
    # Lazy import because octarine is not a hard dependency
    oc = import_octarine()

    settings = OctarineSettings().update_settings(**kwargs)

    # Check if any existing viewer has already been closed
    if isinstance(getattr(config, "primary_viewer", None), oc.Viewer):
        try:
            getattr(config, "primary_viewer").canvas.__repr__()
        except RuntimeError:
            config.primary_viewer = None

    if settings.viewer in (None, "new"):
        # If it does not exists yet, initialize a canvas object and make global
        if (
            not isinstance(getattr(config, "primary_viewer", None), oc.Viewer)
            or settings.viewer == "new"
        ):
            viewer = config.primary_viewer = _new_viewer(
                oc,
                settings,
                size=settings.size,
                offscreen=settings.offscreen
                or os.environ.get("NAVIS_HEADLESS", False),
            )
        else:
            viewer = getattr(config, "primary_viewer", None)
    else:
        viewer = settings.pop("viewer", getattr(config, "primary_viewer", None))

    # Make sure viewer is visible
    if settings.show:
        viewer.show()

    _add_objects(viewer, x, settings)

    return viewer


def plot3d_snapshot(x, **kwargs):
    """Plot3d() helper function to render a scene into a matplotlib figure.

    Unlike `plot3d_octarine` this never touches the primary viewer: unless we
    are handed one to take a picture of, the scene goes into a throwaway
    offscreen viewer that is closed again on the way out.

    """
    # Lazy import because octarine is not a hard dependency
    oc = import_octarine()

    settings = OctarineSettings().update_settings(**kwargs)

    temporary = not isinstance(settings.viewer, oc.Viewer)
    viewer = (
        _new_viewer(
            oc,
            settings,
            size=settings.size or (render.DEFAULT_SIZE, render.DEFAULT_SIZE),
            offscreen=True,
        )
        if temporary
        else settings.viewer
    )

    if getattr(viewer, "canvas", None) is None:
        raise RuntimeError(
            "The viewer has no canvas to render from - octarine appears to be "
            "running in headless mode (`octarine.config.HEADLESS`)."
        )

    try:
        view = settings.view
        if view == "auto":
            # Frame the scene for a viewer we just made, but leave a camera that
            # was handed to us (or set up interactively) alone
            view = render.DEFAULT_VIEW if temporary else None

        if view is not None:
            # We are about to point the camera ourselves, so letting octarine
            # center on every object we add is just thrown-away work
            settings.center = False
        elif not temporary and not settings.was_set("center"):
            # ... and here centering would undo the camera we were asked to keep
            settings.center = False

        _add_objects(viewer, x, settings)

        return render.to_mpl(
            viewer,
            settings,
            view=view,
            # Without an explicit size we can pick a canvas that matches the
            # scene, which avoids rendering a lot of empty space
            fit=temporary and not settings.size,
        )
    finally:
        if temporary:
            viewer.close()


def _new_viewer(oc, settings, *, size, offscreen):
    """A fresh octarine viewer, configured from the window-ish settings."""
    v = oc.Viewer(
        size=size,
        camera=settings.camera,
        control=settings.control,
        show=False,
        offscreen=offscreen,
    )
    # Uniform frontal lighting is a good default for neuron meshes
    if hasattr(v, "headlight"):
        v.headlight = True
    return v


def _add_objects(viewer, x, settings):
    """Add everything in `x` to the viewer, as the settings ask for."""
    (neurons, volumes, points, _) = utils.parse_objects(x)

    # We need to pop clear/clear3d to prevent clearing again later
    if settings.clear:
        settings.clear = False  # clear only once
        viewer.clear()

    # Add object (the viewer currently takes care of producing the visuals)
    if neurons:
        if settings.was_set("cn_color_by"):
            # Silently dropping it would leave connectors coloured by `type` and
            # no hint as to why
            logger.warning(
                "`cn_color_by` is not supported by the octarine backend - "
                "connectors will be coloured by type. Use `backend='plotly'` "
                "or `navis.plot2d` for per-connector colours."
            )
        # `add_neurons` has no **kwargs, so it must only see the settings that
        # describe the neurons - not the ones describing the window or figure
        neuron_settings = {
            k: v
            for k, v in settings.to_dict().items()
            if k in settings._neuron_settings
        }
        viewer.add_neurons(neurons, center=settings.center, **neuron_settings)
    if volumes:
        for v in volumes:
            viewer.add_mesh(
                v,
                name=getattr(v, "name", None),
                color=getattr(v, "color", (0.95, 0.95, 0.95, 0.1)),
                alpha=getattr(v, "alpha", None),
                center=settings.center,
            )
    if points:
        for p in points:
            viewer.add_points(p, center=settings.center, **settings.scatter_kws)


def plot3d_plotly(x, **kwargs):
    """
    Plot3d() helper function to generate plotly 3D plots. This is just to
    improve readability and structure of the code.
    """
    # Lazy import because plotly is not a hard dependency
    try:
        import plotly.graph_objs as go
        from .plotly.graph_objs import (
            neuron2plotly,
            volume2plotly,
            scatter2plotly,
            layout2plotly,
        )
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "navis.plot3d() with the `plotly` backend requires the `plotly` library "
            "to be installed:\n  pip3 install plotly -U"
        )

    settings = PlotlySettings().update_settings(**kwargs)

    # Parse objects to plot
    (neurons, volumes, points, visual) = utils.parse_objects(x)

    # `color_by` can be either a neuron property (-> one color per neuron, dealt
    # with here) or a per-node/vertex property (-> passed on to the backend)
    color_neurons_by, settings.color_by = parse_color_by(
        settings.color_by, neurons, settings.palette
    )

    neuron_cmap, volumes_cmap = prepare_colormap(
        settings.color,
        neurons=neurons,
        volumes=volumes,
        palette=settings.palette,
        color_by=color_neurons_by,
        alpha=settings.alpha,
        color_range=255,
    )

    data = []
    if neurons:
        data += neuron2plotly(neurons, neuron_cmap, settings)
    if volumes:
        data += volume2plotly(volumes, volumes_cmap, settings)
    if points:
        data += scatter2plotly(points, **settings.scatter_kws)

    layout = layout2plotly(**settings.to_dict())

    # If not provided generate a figure dictionary
    fig = settings.fig if settings.fig else go.Figure(layout=layout)
    if not isinstance(fig, (dict, go.Figure)):
        raise TypeError(
            "`fig` must be plotly.graph_objects.Figure or dict, got " f"{type(fig)}"
        )

    # Add data
    for trace in data:
        fig.add_trace(trace)

    if settings.inline and utils.is_jupyter():
        fig.show()
    else:
        logger.info("Use the `.show()` method to plot the figure.")
        return fig


def plot3d_k3d(x, **kwargs):
    """
    Plot3d() helper function to generate k3d 3D plots. This is just to
    improve readability and structure of the code.
    """
    # Lazy import because k3d is not (yet) a hard dependency
    try:
        import k3d
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "navis.plot3d() with `k3d` backend requires the k3d library "
            "to be installed:\n  pip3 install k3d -U"
        )

    from .k3d.k3d_objects import neuron2k3d, volume2k3d, scatter2k3d

    settings = K3dSettings().update_settings(**kwargs)

    # Parse objects to plot
    (neurons, volumes, points, visual) = utils.parse_objects(x)

    # `color_by` can be either a neuron property (-> one color per neuron, dealt
    # with here) or a per-node/vertex property (-> passed on to the backend)
    color_neurons_by, settings.color_by = parse_color_by(
        settings.color_by, neurons, settings.palette
    )

    neuron_cmap, volumes_cmap = prepare_colormap(
        settings.color,
        neurons=neurons,
        volumes=volumes,
        palette=settings.palette,
        color_by=color_neurons_by,
        alpha=settings.alpha,
        color_range=255,
    )

    data = []
    if neurons:
        data += neuron2k3d(neurons, neuron_cmap, settings)
    if volumes:
        data += volume2k3d(volumes, volumes_cmap, settings)
    if points:
        data += scatter2k3d(points, **settings.scatter_kws)

    # If not provided generate a plot
    if not settings.plot:
        plot = k3d.plot(height=settings.height)
        plot.camera_rotate_speed = 5
        plot.camera_zoom_speed = 2
        plot.camera_pan_speed = 1
        plot.grid_visible = False
    else:
        plot = settings.plot

    # Add data
    for trace in data:
        plot += trace

    if settings.inline and utils.is_jupyter():
        plot.display()
    else:
        logger.info("Use the `.display()` method to show the plot.")
        return plot
