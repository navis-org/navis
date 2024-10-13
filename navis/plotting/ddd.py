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
import warnings

import numpy as np

from typing import Union, List
from importlib.util import find_spec

from .. import utils, config, core
from .colors import prepare_colormap
from .settings import OctarineSettings, PlotlySettings, K3dSettings, VispySettings

__all__ = ["plot3d"]

logger = config.get_logger(__name__)

# Check if backends are available without importing them
BACKENDS = tuple(
    b for b in ("octarine", "vispy", "plotly", "k3d") if find_spec(b) is not None
)
JUPYTER_BACKENDS = tuple(b for b in ("plotly", "octarine", "k3d") if b in BACKENDS)
NON_JUPYTER_BACKENDS = tuple(
    b for b in ("octarine", "vispy", "plotly") if b in BACKENDS
)
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

    Uses either [octarine], [vispy], [k3d] or [plotly] as backend.
    By default, the choice is automatic depending on what backends
    are installed and the context:

      - Terminal: octarine > vispy > plotly
      - Jupyter: plotly > octarine > k3d

    See the `backend` parameter on how to change this behavior.

    [octarine]: https://schlegelp.github.io/octarine/
    [vispy]: http://vispy.org
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

    connectors :      bool | "presynapses" | "postsynapses" | str | list, default=True

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

    color_by :        str | array | list of arrays, default = None

                      Color neurons by a property. Can be:

                        - a list/array of labels, one per each neuron
                        - a neuron property (str)
                        - a column name in the node table of `TreeNeurons`
                        - a list/array of values for each node

                      Numerical values will be normalized. You can control
                      the normalization by passing a `vmin` and/or `vmax`
                      parameter. Must specify a colormap via `palette`.

    shade_by :        str | array | list of arrays, default=None

                      Similar to `color_by` but will affect only the alpha
                      channel of the color. If `shade_by='strahler'` will
                      compute Strahler order if not already part of the node
                      table (TreeNeurons only). Numerical values will be
                      normalized. You can control the normalization by passing
                      a `smin` and/or `smax` parameter. Does not work with
                      `k3d` backend.

    radius :          bool | "auto", default=False

                      If "auto" will plot neurites of `TreeNeurons` with radius
                      if they have radii. If True, will try plotting neurites of
                      `TreeNeurons` with radius regardless. The radius can be
                      scaled by `linewidth`. Note that this will increase rendering
                      time.

    soma :            bool, default=True

                      TreeNeurons only: Whether to plot soma if it exists. Size
                      of the soma is determined by the neuron's `.soma_radius`
                      property which defaults to the "radius" column for
                      `TreeNeurons`.

    linewidth :       float, default=3 for plotly and 1 for all others

                      TreeNeurons only.

    linestyle :       str, default='-'

                      TreeNeurons only. Follows the same rules as in matplotlib.

    scatter_kws :     dict, optional

                      Use to modify scatter plots. Accepted parameters are:
                        - `size` to adjust size of dots
                        - `color` to adjust color

    Figure parameters
    -----------------
    backend :         'auto' (default) | 'octarine' | 'vispy' | 'plotly' | 'k3d'

                      Which backend to use for plotting. Note that there will
                      be minor differences in what feature/parameters are
                      supported depending on the backend:

                        - `auto` selects backend based on availability and
                          context (see above). You can override this by setting an
                          environment variable e.g. `NAVIS_PLOT3D_BACKEND="vispy"`
                          or `NAVIS_PLOT3D_JUPYTER_BACKEND="k3d"`.
                        - `octarine` uses WGPU to generate high performances
                          interactive 3D plots. Works both terminal and Jupyter.
                        - `vispy` similar to octarine but uses OpenGL: slower
                          but runs on older systems. Works only from terminals.
                        - `plotly` generates 3D plots using WebGL. Works
                          "inline" in Jupyter notebooks but can also produce a
                          HTML file that can be opened in any browers.
                        - `k3d` generates 3D plots using k3d. Works only in
                          Jupyter notebooks!

    **Below parameters are for plotly backend only:**

    fig :             plotly.graph_objs.Figure

                      Pass to add graph objects to existing plotly figure. Will
                      not change layout.

    title :           str, default=None

                      For plotly only! Change plot title.

    width/height :    int, optional

                      Use to adjust figure size.

    fig_autosize :    bool, default=False

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

    **Below parameters are for the Octarine/vispy backends only:**

    clear :           bool, default = False

                      If True, will clear the viewer before adding the new
                      objects.

    center :          bool, default = True

                      If True, will center camera on the newly added objects.

    combine :         bool, default = False

                      If True, will combine objects of the same type into a
                      single visual. This can greatly improve performance but
                      also means objects can't be selected individually
                      anymore. This is Vispy only.

    size :            (width, height) tuple, optional

                      Use to adjust figure/window size.

    show :            bool, default=True

                      Whether to immediately show the viewer.

    Returns
    -------
    If `backend='octarine'`

        From terminal: opens a 3D window and returns :class:`octarine.Viewer`.
        From Jupyter: :class:`octarine.Viewer` displayed in an ipywidget.

    If `backend='vispy'`

        Opens a 3D window and returns [`navis.Viewer`][].

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

    [`navis.Viewer`][]
        Interactive vispy 3D viewer.

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

    Some more advanced examples:

    >>> # plot3d() can deal with combinations of objects
    >>> nl = navis.example_neurons()
    >>> vol = navis.example_volume('LH')
    >>> vol.color = (255, 0, 0, .5)
    >>> # This plots a neuronlists, a single neuron and a volume
    >>> v = navis.plot3d([nl[0:2], nl[3], vol])
    >>> # Clear viewer (works only with octarine and vispy)
    >>> v = navis.plot3d(nl, clear=True)

    See the [plotting intro](../../generated/gallery/1_plotting/tutorial_plotting_00_intro)
    for even more examples.

    """
    # Select backend
    backend = kwargs.pop("backend", "auto")
    allowed_backends = ("auto", "octarine", "vispy", "plotly", "k3d")
    if backend.lower() == "auto":
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
                        "install one of the following: octarine, vispy, plotly."
                    )
                backend = os.environ.get(
                    "NAVIS_PLOT3D_BACKEND", NON_JUPYTER_BACKENDS[0]
                )

            # Set the backend for the next time
            AUTO_BACKEND = backend

            logger.info(f'Using "{backend}" backend for 3D plotting.')
    elif backend.lower() not in allowed_backends:
        raise ValueError(
            f'Unknown backend "{backend}". ' f'Permitted: {".".join(allowed_backends)}.'
        )
    elif backend.lower() not in BACKENDS:
        raise ModuleNotFoundError(
            f'Backend "{backend}" not installed. Please install it via pip '
            "(see https://navis.readthedocs.io/en/latest/source/install.html#optional-dependencies "
            "for more information)."
        )

    if backend == "vispy":
        return plot3d_vispy(x, **kwargs)
    elif backend == "k3d":
        if not utils.is_jupyter():
            logger.warning("k3d backend only works in Jupyter environments")
        return plot3d_k3d(x, **kwargs)
    elif backend == "plotly":
        return plot3d_plotly(x, **kwargs)
    elif backend == "octarine":
        return plot3d_octarine(x, **kwargs)
    else:
        raise ValueError(
            f'Unknown backend "{backend}". ' f'Permitted: {".".join(allowed_backends)}.'
        )


def plot3d_vispy(x, **kwargs):
    """Plot3d() helper function to generate vispy 3D plots.

    This is just to improve readability. Its only purpose is to find the
    existing viewer or generate a new one.

    """
    from .vispy.viewer import Viewer

    # If this likely the first invoke, warn the user that vispy is deprecated
    if not hasattr(config, "primary_viewer"):
        warnings.warn(
            (
                "The `vispy` backend is depcrecated and will be removed in a future version of navis. "
                "We recommend to use the `octarine` backend instead. If that is for some reason not possible, "
                "please let us know via the issue tracker at https://github.com/navis-org/navis/issues asap."
            ),
            DeprecationWarning,
            stacklevel=2,
        )

    settings = VispySettings().update_settings(**kwargs)

    # Parse objects to plot
    (neurons, volumes, points, visuals) = utils.parse_objects(x)

    if settings.viewer in (None, "new"):
        # If does not exists yet, initialise a canvas object and make global
        if (
            not isinstance(getattr(config, "primary_viewer", None), Viewer)
            or settings.viewer == "new"
        ):
            viewer = config.primary_viewer = Viewer(size=settings.size)
        else:
            viewer = getattr(config, "primary_viewer", None)
    else:
        viewer = settings.viewer

    # Make sure viewer is visible
    if settings.show:
        viewer.show()

    # We need to pop clear/clear3d to prevent clearing again later
    if settings.clear:
        settings.clear = False
        viewer.clear()

    # Add objects (the viewer currently takes care of producing the visuals)
    if neurons:
        viewer.add(neurons, **settings.to_dict())
    if volumes:
        viewer.add(volumes, **settings.to_dict())
    if points:
        viewer.add(points, scatter_kws=settings.scatter_kws)

    return viewer


def plot3d_octarine(x, **kwargs):
    """Plot3d() helper function to generate octarine 3D plots.

    This is just to improve readability. Its only purpose is to find the
    existing viewer or generate a new one.

    """
    # Lazy import because octarine is not a hard dependency
    try:
        import octarine as oc
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "navis.plot3d() with the `octarine` backend requires the `octarine3d` library "
            "to be installed:\n  pip3 install octarine3 octarine-navis-plugin -U"
        )

    if not hasattr(oc.Viewer, "add_neurons"):
        raise ModuleNotFoundError(
            "Looks like the navis plugin for octarine is not installed. "
            "Please install it via pip:\n  pip install octarine-navis-plugin"
        )

    settings = OctarineSettings().update_settings(**kwargs)

    # Parse objects to plot
    (neurons, volumes, points, visuals) = utils.parse_objects(x)

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
            viewer = config.primary_viewer = oc.Viewer(
                size=settings.size,
                camera=settings.camera,
                control=settings.control,
                show=False,
                offscreen=settings.offscreen or os.environ.get("NAVIS_HEADLESS", False),
            )
        else:
            viewer = getattr(config, "primary_viewer", None)
    else:
        viewer = settings.pop("viewer", getattr(config, "primary_viewer"))

    # Make sure viewer is visible
    if settings.show:
        viewer.show()

    # We need to pop clear/clear3d to prevent clearing again later
    if settings.clear:
        settings.clear = False  # clear only once
        viewer.clear()

    # Add object (the viewer currently takes care of producing the visuals)
    if neurons:
        # We need to pop viewer-specific settings to prevent errors in plotting functions
        neuron_settings = settings.to_dict()
        for key in settings._viewer_settings:
            neuron_settings.pop(key, None)
        viewer.add_neurons(neurons, center=settings.get("center", True), **neuron_settings)
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

    return viewer


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

    neuron_cmap, volumes_cmap = prepare_colormap(
        settings.color,
        neurons=neurons,
        volumes=volumes,
        palette=settings.palette,
        color_by=None,
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

    neuron_cmap, volumes_cmap = prepare_colormap(
        settings.color,
        neurons=neurons,
        volumes=volumes,
        palette=settings.palette,
        color_by=None,
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

    # Add data
    for trace in data:
        plot += trace

    if settings.inline and utils.is_jupyter():
        plot.display()
    else:
        logger.info("Use the `.display()` method to show the plot.")
        return plot
