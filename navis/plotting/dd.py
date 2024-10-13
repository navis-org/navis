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

"""Module contains functions to plot neurons in 2D/2.5D."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcl
from mpl_toolkits.mplot3d.art3d import (
    Line3DCollection,
    Poly3DCollection,
    Path3DCollection,
    Patch3DCollection,
)
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.cm import ScalarMappable

import numpy as np
import pandas as pd

import pint
import warnings

from typing import Union, List, Tuple
import copy

from .. import utils, config, core, conversion
from .colors import prepare_colormap, vertex_colors
from .plot_utils import segments_to_coords, tn_pairs_to_coords
from .settings import Matplotlib2dSettings

__all__ = ["plot2d"]

logger = config.get_logger(__name__)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])

# Default colormap for depth coloring
DEPTH_CMAP = mpl.cm.jet


def plot2d(
    x: Union[
        core.NeuronObject,
        core.Volume,
        np.ndarray,
        List[Union[core.NeuronObject, np.ndarray, core.Volume]],
    ],
    **kwargs,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Generate 2D plots of neurons and neuropils.

    The main advantage of this is that you can save plot as vector graphics.

    Note
    ----
    This function uses `matplotlib` which "fakes" 3D as it has only very limited
    control over layering objects in 3D. Therefore neurites are not necessarily
    plotted in the right Z order. This becomes especially troublesome when
    plotting a complex scene with lots of neurons criss-crossing. See the
    `method` parameter for details.

    Parameters
    ----------
    x :                 TreeNeuron | MeshNeuron | NeuronList | Volume | Dotprops | np.ndarray
                        Objects to plot:
                         - multiple objects can be passed as list (see examples)
                         - numpy array of shape (N, 3) is intepreted as points for
                           scatter plots

    Object parameters
    -----------------
    soma :              bool | dict, default=True

                        Plot soma if one exists. Size of the soma is determined
                        by the neuron's `.soma_radius` property which defaults
                        to the "radius" column for `TreeNeurons`. You can also
                        pass `soma` as a dictionary to customize the appearance
                        of the soma - for example `soma={"color": "red", "lw": 2, "ec": 1}`.

    radius :            bool | "auto", default=False

                        If "auto" will plot neurites of `TreeNeurons` with radius
                        if they have radii. If True, will try plotting neurites of
                        `TreeNeurons` with radius regardless. The radius can be
                        scaled by `linewidth`. Note that this will increase rendering
                        time.

    linewidth :         int | float, default=.5

                        Width of neurites. Also accepts alias `lw`.

    linestyle :         str, default='-'

                        Line style of neurites. Also accepts alias `ls`.

    color :             None | str | tuple | list | dict, default=None

                        Use single str (e.g. `'red'`) or `(r, g, b)` tuple
                        to give all neurons the same color. Use `list` of
                        colors to assign colors: `['red', (1, 0, 1), ...].
                        Use `dict` to map colors to neuron IDs:
                        `{id: (r, g, b), ...}`.

    palette :           str | array | list of arrays, default=None

                        Name of a matplotlib or seaborn palette. If `color` is
                        not specified will pick colors from this palette.

    color_by :          str | array | list of arrays, default = None

                        Color neurons by a property. Can be:
                          - a list/array of labels, one per each neuron
                          - a neuron property (str)
                          - a column name in the node table of `TreeNeurons`
                          - a list/array of values for each node
                        Numerical values will be normalized. You can control
                        the normalization by passing a `vmin` and/or `vmax` parameter.

    shade_by :          str | array | list of arrays, default=None

                        Similar to `color_by` but will affect only the alpha
                        channel of the color. If `shade_by='strahler'` will
                        compute Strahler order if not already part of the node
                        table (TreeNeurons only). Numerical values will be
                        normalized. You can control the normalization by passing
                        a `smin` and/or `smax` parameter.

    alpha :             float [0-1], default=1

                        Alpha value for neurons. Overriden if alpha is provided
                        as fourth value in `color` (rgb*a*). You can override
                        alpha value for connectors by using `cn_alpha`.

    mesh_shade :        bool, default=False

                        Only relevant for meshes (e.g. `MeshNeurons`) and
                        `TreeNeurons` with radius, and when method is 3d or
                        3d complex. Whether to shade the object which will give it
                        a 3D look.

    depth_coloring :    bool, default=False

                        If True, will use neuron color to encode depth (Z).
                        Overrides `color` argument. Does not work with
                        `method = '3d_complex'`.

    depth_scale :       bool, default=True

                        If True and `depth_coloring=True` will plot a scale.

    connectors :        bool | "presynapses" | "postsynapses" | str | list, default=True

                        Plot connectors. This can either be `True` (plot all
                        connectors), `"presynapses"` (only presynaptic connectors)
                        or `"postsynapses"` (only postsynaptic connectors). If
                        a string or a list is provided, it will be used to filter the
                        `type` column in the connectors table.

    connectors_only :   boolean, default=False

                        Plot only connectors, not the neuron.

    cn_size :           int | float, default = 1

                        Size of connectors.

    cn_layout :         dict, default={}

                        Defines default settings (color, style) for connectors.
                        See `navis.config.default_connector_colors` for the
                        default layout.

    cn_colors :         str | tuple | dict | "neuron"

                        Overrides the default connector (e.g. synpase) colors:
                            - single color as str (e.g. `'red'`) or rgb tuple
                            (e.g. `(1, 0, 0)`)
                            - dict mapping the connectors tables `type` column to
                            a color (e.g. `{"pre": (1, 0, 0)}`)
                            - with "neuron", connectors will receive the same color
                            as their neuron

    cn_mesh_colors :    bool, default=False

                        If True, will use the neuron's color for its connectors.

    scatter_kws :       dict, default={}

                        Parameters to be used when plotting points. Accepted
                        keywords are: `size` and `color`.

    volume_outlines :   bool | "both", default=False

                        If True will plot volume outline with no fill. Only
                        works with `method="2d"`. Requires the `shapely` package.

    dps_scale_vec :     float

                        Scale vector for dotprops.

    Figure parameters
    -----------------
    method :            '2d' | '3d' (default) | '3d_complex'

                        Method used to generate plot. Comes in three flavours:
                         1. `2d` uses normal matplotlib. Neurons are plotted on
                            top of one another in the order their are passed to
                            the function. Use the `view` parameter (below) to
                            set the view (default = xy).
                         2. `3d` uses matplotlib's 3D axis. Here, matplotlib
                            decide the depth order (zorder) of plotting. Can
                            change perspective either interacively or by code
                            (see examples).
                         3. `3d_complex` same as 3d but each neuron segment is
                            added individually. This allows for more complex
                            zorders to be rendered correctly. Slows down
                            rendering!

    view :              tuple, default = ("x", "y")

                        Sets view for `method='2d'`. Can be any combination of
                        "x", "y", "z" and their negations. For example, to plot
                        from the top, use `view=('x', '-y')`. For 3D `methods`,
                        this will set the initial view which can be changed by
                        adjusting `ax.azim`, `ax.elev` and `ax.roll` (see examples).

    non_view_axes3d :   "show" | "hide" (default) | "fade"

                        Only relevant for methods '3d' and '3d_complex': what to
                        do with the axis that are not in the view. If 'hide', will
                        hide them. If 'show', will show them. If 'fade', will
                        make them semi-transparent. This is relevant if you
                        intend if you intend to customize the view after plotting.

    autoscale :         bool, default=True

                        If True, will scale the axes to fit the data.

    scalebar :          int | float | str | pint.Quantity | dict, default=False

                        Adds a scale bar. Provide integer, float or str to set
                        size of scalebar. Int|float are assumed to be in same
                        units as data. You can specify units in as string:
                        e.g. "1 um". For methods '3d' and '3d_complex', this
                        will create an axis object.

                        You can customize the scalebar by passing a dictionary.
                        For example:

                        `{size: "1 micron", color: 'k', lw: 3, alpha: 0.9}`


    ax :                matplotlib.Axes, default=None

                        Pass an axis object if you want to plot on an existing
                        canvas. Must match `method` - i.e. 2D or 3D axis.

    figsize :           tuple, default=None

                        Size of figure. Ignored if `ax` is provided.

    rasterize :         bool, default=False

                        Neurons produce rather complex vector graphics which can
                        lead to large files when saving to SVG, PDF or PS. Use
                        this parameter to rasterize neurons and meshes/volumes
                        (but not axes or labels) to reduce file size.

    orthogonal :        bool, default=True

                        Whether to use orthogonal or perspective view for
                        methods '3d' and '3d_complex'.

    group_neurons :     bool, default=False

                        If True, neurons will be grouped. Works with SVG export
                        but not PDF. Does NOT work with `method='3d_complex'`.

    Returns
    -------
    fig :               matplotlib.Figure
    ax :                matplotlib.Axes

    Examples
    --------

    >>> import navis
    >>> import matplotlib.pyplot as plt

    Plot list of neurons as simple 2d:

    >>> nl = navis.example_neurons()
    >>> fig, ax = navis.plot2d(nl, method='2d', view=('x', '-z'))
    >>> plt.show() # doctest: +SKIP

    Add a volume:

    >>> vol = navis.example_volume('LH')
    >>> fig, ax = navis.plot2d([nl, vol], method='2d', view=('x', '-z'))
    >>> plt.show() # doctest: +SKIP

    Change neuron colors:

    >>> fig, ax = navis.plot2d(
    ...              nl,
    ...              method='2d',
    ...              view=('x', '-z'),
    ...              color=['r', 'g', 'b', 'm', 'c', 'y']
    ...          )
    >>> plt.show() # doctest: +SKIP

    Plot in "fake" 3D:

    >>> fig, ax = navis.plot2d(nl, method='3d', view=('x', '-z'))
    >>> plt.show() # doctest: +SKIP
    >>> # In an interactive window you can dragging the plot to rotate

    Plot in "fake" 3D and change perspective:

    >>> fig, ax = navis.plot2d(nl, method='3d', view=('x', '-z'))
    >>> # Change view
    >>> ax.elev = -20
    >>> ax.azim = 45
    >>> ax.roll = 180
    >>> plt.show() # doctest: +SKIP

    Plot using depth-coloring:

    >>> fig, ax = navis.plot2d(nl, method='3d', depth_coloring=True, view=('x', '-z'))
    >>> plt.show() # doctest: +SKIP

    See the [plotting intro](../../generated/gallery/1_plotting/tutorial_plotting_00_intro)
    for more examples.

    See Also
    --------
    [`navis.plot3d`][]
            Use this if you want interactive, perspectively correct renders
            and if you don't need vector graphics as outputs.
    [`navis.plot1d`][]
            A nifty way to visualise neurons in a single dimension.
    [`navis.plot_flat`][]
            Plot neurons as flat structures (e.g. dendrograms).

    """
    # This handles (1) checking for invalid arguments, (2) setting defaults and
    # (3) synonyms
    settings = Matplotlib2dSettings().update_settings(**kwargs)

    _METHOD_OPTIONS = ["2d", "3d", "3d_complex"]
    if settings.method not in _METHOD_OPTIONS:
        raise ValueError(
            f'Unknown method "{settings.method}". Please use either: '
            f'{",".join(_METHOD_OPTIONS)}'
        )

    # Parse objects
    (neurons, volumes, points, _) = utils.parse_objects(x)

    # Here we check whether `color_by` is a neuron property which we
    # want to translate into a single color per neuron, or a
    # per node/vertex property which we will parse late
    color_neurons_by = None
    if settings.color_by is not None and neurons:
        if not settings.palette:
            raise ValueError(
                'Must provide palette (via e.g. `palette="viridis"`) '
                "when using `color_by` argument."
            )

        # Check if this may be a neuron property
        if isinstance(settings.color_by, str):
            # Check if this could be a neuron property
            has_prop = hasattr(neurons[0], settings.color_by)

            # For TreeNeurons, we also check if it is a node property
            # If so, prioritize this.
            if isinstance(neurons[0], core.TreeNeuron):
                if settings.color_by in neurons[0].nodes.columns:
                    has_prop = False

            if has_prop:
                # If it is, use it to color neurons
                color_neurons_by = [
                    getattr(neuron, settings.color_by) for neuron in neurons
                ]
                settings.color_by = None
        elif isinstance(settings.color_by, (list, np.ndarray)):
            if len(settings.color_by) == len(neurons):
                color_neurons_by = settings.color_by
                settings.color_by = None

    # Generate the per-neuron colors
    (neuron_cmap, volumes_cmap) = prepare_colormap(
        settings.color,
        neurons=neurons,
        volumes=volumes,
        palette=settings.palette,
        color_by=color_neurons_by,
        alpha=settings.alpha,
        color_range=1,
    )

    if not isinstance(settings.color_by, type(None)):
        neuron_cmap = vertex_colors(
            neurons,
            by=settings.color_by,
            use_alpha=False,
            palette=settings.palette,
            norm_global=settings.norm_global,
            vmin=settings.vmin,
            vmax=settings.vmax,
            na="raise",
            color_range=1,
        )

    if not isinstance(settings.shade_by, type(None)):
        alphamap = vertex_colors(
            neurons,
            by=settings.shade_by,
            use_alpha=True,
            palette="viridis",  # palette is irrelevant here
            norm_global=settings.norm_global,
            vmin=settings.smin,
            vmax=settings.smax,
            na="raise",
            color_range=1,
        )

        new_colormap = []
        for c, a in zip(neuron_cmap, alphamap):
            if not (isinstance(c, np.ndarray) and c.ndim == 2):
                c = np.tile(c, (a.shape[0], 1))

            if c.shape[1] == 4:
                c[:, 3] = a[:, 3]
            else:
                c = np.insert(c, 3, a[:, 3], axis=1)

            new_colormap.append(c)
        neuron_cmap = new_colormap

    # Generate axes
    if not settings.ax:
        if settings.method == "2d":
            fig, ax = plt.subplots(figsize=settings.figsize)
        elif settings.method in ("3d", "3d_complex"):
            fig = plt.figure(
                figsize=settings.figsize if settings.figsize else plt.figaspect(1) * 1.5
            )
            ax = fig.add_subplot(111, projection="3d")
        # Hide axes
        # ax.set_axis_off()
    else:
        # Check if correct axis were provided
        if not isinstance(settings.ax, mpl.axes.Axes):
            raise TypeError('Ax must be of type "mpl.axes.Axes", ' f'not "{type(ax)}"')
        ax = settings.ax
        fig = ax.get_figure()
        if settings.method in ("3d", "3d_complex") and ax.name != "3d":
            raise TypeError("Axis must be 3d.")
        elif settings.method == "2d" and ax.name == "3d":
            raise TypeError("Axis must be 2d.")

    # Set axis projection
    if settings.method in ("3d", "3d_complex"):
        # This sets the view
        _set_view3d(ax, settings)

        # Some styling:
        # Make background transparent (nicer for dark themes)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        # For 3d axes, we also need to set the pane color to transparent
        if hasattr(ax, "zaxis"):
            ax.xaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))

            ax.yaxis.pane.fill = False
            ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))

            ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
            ax.zaxis.pane.fill = False

        if settings.orthogonal:
            ax.set_proj_type("ortho")
        else:
            ax.set_proj_type("persp", focal_length=1)  # smaller = more perspective
    else:
        ax.set_aspect("equal")
        _set_view2d(ax, settings)

    # Prepare some stuff for depth coloring
    if settings.depth_coloring and not neurons.empty:
        if settings.method == "3d_complex":
            raise Exception(
                f'Depth coloring unavailable for method "{settings.method}"'
            )
        elif settings.method == "2d":
            bbox = neurons.bbox
            # Add to kwargs
            xy = [v.replace("-", "").replace("+", "") for v in settings.view]
            depth_ix = [v[1] for v in [("x", 0), ("y", 1), ("z", 2)] if v[0] not in xy]

            # We use this to track the normaliser
            settings.norm = plt.Normalize(
                vmin=bbox[depth_ix, 0], vmax=bbox[depth_ix, 1]
            )

    # Plot volumes first
    if volumes:
        for i, v in enumerate(volumes):
            _ = _plot_volume(v, volumes_cmap[i], ax, settings)

    # Create lines from segments
    visuals = {}
    for i, neuron in enumerate(
        config.tqdm(
            neurons,
            desc="Plot neurons",
            leave=False,
            disable=config.pbar_hide | len(neurons) <= 10,
        )
    ):
        if not settings.connectors_only:
            if isinstance(neuron, core.TreeNeuron) and neuron.nodes.empty:
                logger.warning(f"Skipping TreeNeuron w/o nodes: {neuron.label}")
                continue
            if isinstance(neuron, core.TreeNeuron) and neuron.nodes.shape[0] == 1:
                logger.warning(f"Skipping single-node TreeNeuron: {neuron.label}")
                continue
            elif isinstance(neuron, core.MeshNeuron) and neuron.faces.size == 0:
                logger.warning(f"Skipping MeshNeuron w/o faces: {neuron.label}")
                continue
            elif isinstance(neuron, core.Dotprops) and neuron.points.size == 0:
                logger.warning(f"Skipping Dotprops w/o points: {neuron.label}")
                continue

            if isinstance(neuron, core.TreeNeuron) and settings.radius == "auto":
                # Number of nodes with radii
                n_radii = (
                    neuron.nodes.get("radius", pd.Series([])).fillna(0) > 0
                ).sum()
                # If less than 30% of nodes have a radius, we will fall back to lines
                if n_radii / neuron.nodes.shape[0] < 0.3:
                    settings.radius = False

            if isinstance(neuron, core.TreeNeuron) and settings.radius:
                # Warn once if more than 5% of nodes have missing radii
                if not getattr(fig, "_radius_warned", False):
                    if (
                        (neuron.nodes.radius.fillna(0).values <= 0).sum()
                        / neuron.n_nodes
                    ) > 0.05:
                        logger.warning(
                            "Some skeleton nodes have radius <= 0. This may lead to "
                            "rendering artifacts. Set `radius=False` to plot skeletons "
                            "as single-width lines instead."
                        )
                        fig._radius_warned = True

                _neuron = conversion.tree2meshneuron(
                    neuron,
                    warn_missing_radii=False,
                    radius_scale_factor=settings.get("linewidth", 1),
                )
                _neuron.connectors = neuron.connectors
                neuron = _neuron

                # See if we need to map colors to vertices
                if isinstance(neuron_cmap[i], np.ndarray) and neuron_cmap[i].ndim == 2:
                    neuron_cmap[i] = neuron_cmap[i][neuron.vertex_map]

            if isinstance(neuron, core.TreeNeuron):
                lc, sc = _plot_skeleton(neuron, neuron_cmap[i], ax, settings)
                # Keep track of visuals related to this neuron
                visuals[neuron] = {"skeleton": lc, "somata": sc}
            elif isinstance(neuron, core.MeshNeuron):
                m = _plot_mesh(neuron, neuron_cmap[i], ax, settings)
                visuals[neuron] = {"mesh": m}
            elif isinstance(neuron, core.Dotprops):
                dp = _plot_dotprops(neuron, neuron_cmap[i], ax, settings)
                visuals[neuron] = {"dotprop": dp}
            elif isinstance(neuron, core.VoxelNeuron):
                dp = _plot_voxels(
                    neuron,
                    neuron_cmap[i],
                    ax,
                    settings,
                    **settings.scatter_kws,
                )
                visuals[neuron] = {"dotprop": dp}
            else:
                raise TypeError(
                    f"Don't know how to plot neuron of type '{type(neuron)}' "
                )

        if (settings.connectors or settings.connectors_only) and neuron.has_connectors:
            _ = _plot_connectors(neuron, neuron_cmap[i], ax, settings)

    # Plot points
    for p in points:
        _ = _plot_scatter(p, ax, settings)

    # Note: autoscaling is a bitch for 3d. In particular when we use Collections, because
    # these are currently ignored by matplotlib's built-in autoscaling.
    if settings.autoscale:
        ax.autoscale(tight=False)  # tight=False avoids clipping the neurons

        if "3d" in settings.method:
            update_axes3d_bounds(ax)

        # This is apparently still required and has to happen AFTER updating axis bounds
        ax.set_aspect("equal", adjustable="box")

    # Add scalebar after the dust has settled
    if settings.scalebar not in (False, None):
        if not settings.orthogonal:
            raise ValueError("Scalebar only available if `orthogonal=True`.")

        _ = _add_scalebar(settings.scalebar, neurons, ax, settings)

    def set_depth():
        """Set depth information for neurons according to camera position."""
        # Get projected coordinates
        proj_co = proj_points(all_co, ax.get_proj())

        # Get min and max of z coordinates
        z_min, z_max = min(proj_co[:, 2]), max(proj_co[:, 2])

        # Generate a new normaliser
        norm = plt.Normalize(vmin=z_min, vmax=z_max)

        # Go over all neurons and update Z information
        for neuron in visuals:
            # Get this neurons colletion and coordinates
            if "skeleton" in visuals[neuron]:
                c = visuals[neuron]["skeleton"]
                this_co = c._segments3d[:, 0, :]
            elif "mesh" in visuals[neuron]:
                c = visuals[neuron]["mesh"]
                # Note that we only get every third position -> that's because
                # these vectors actually represent faces, i.e. each vertex
                this_co = c._vec.T[::3, [0, 1, 2]]
            else:
                raise ValueError(
                    f"Neither mesh nor skeleton found for neuron {neuron.id}"
                )

            # Get projected coordinates
            this_proj = proj_points(this_co, ax.get_proj())

            # Normalise z coordinates
            ns = norm(this_proj[:, 2]).data

            # Set array
            c.set_array(ns)

            # No need for normaliser - already happened
            c.set_norm(None)

            if isinstance(neuron, core.TreeNeuron) and not isinstance(
                getattr(neuron, "soma", None), type(None)
            ):
                # Get depth of soma(s)
                soma = utils.make_iterable(neuron.soma)
                soma_co = (
                    neuron.nodes.set_index("node_id").loc[soma][["x", "y", "z"]].values
                )
                soma_proj = proj_points(soma_co, ax.get_proj())
                soma_cs = norm(soma_proj[:, 2]).data

                # Set soma color
                for cs, s in zip(soma_cs, visuals[neuron]["somata"]):
                    s.set_color(cmap(cs))

    def Update(event):
        set_depth()

    if settings.depth_coloring:
        if settings.palette:
            cmap = plt.get_cmap(settings.palette)
        else:
            cmap = DEPTH_CMAP
        if settings.method == "2d" and settings.depth_scale:
            sm = ScalarMappable(norm=settings.norm, cmap=cmap)
            fig.colorbar(sm, ax=ax, fraction=0.075, shrink=0.5, label="Depth")
        elif settings.method == "3d":
            # Collect all coordinates
            all_co = []
            for n in visuals:
                if "skeleton" in visuals[n]:
                    all_co.append(visuals[n]["skeleton"]._segments3d[:, 0, :])
                if "mesh" in visuals[n]:
                    all_co.append(visuals[n]["mesh"]._vec.T[:, [0, 1, 2]])

            all_co = np.concatenate(all_co, axis=0)
            fig.canvas.mpl_connect("draw_event", Update)
            set_depth()

    return fig, ax


def _add_scalebar(scalebar, neurons, ax, settings):
    """Add scalebar."""
    defaults = {
        "color": "black",
        "lw": 3,
        "alpha": 0.9,
    }

    if isinstance(scalebar, dict):
        if "size" not in scalebar:
            raise ValueError("`scalebar` dictionary must contain 'size' key.")
        defaults.update(scalebar)
        scalebar = defaults["size"]

    if isinstance(scalebar, bool):
        scalebar = "1 um"

    if isinstance(scalebar, str):
        scalebar = config.ureg(scalebar)

    if isinstance(scalebar, pint.Quantity):
        # If we have neurons as points of reference convert
        if neurons:
            scalebar = scalebar.to(neurons[0].units).magnitude
        # If no reference, use assume it's the same units
        else:
            scalebar = scalebar.magnitude

    # Hard-coded 5% offset from figure boundaries
    ax_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 100 * 5

    if settings.method == "2d":
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        coords = np.array([[xlim[0], ylim[0]], [xlim[0] + scalebar, ylim[0]]])

        if not ax.xaxis.get_inverted():
            coords[:, 0] += ax_offset
        else:
            coords[:, 0] -= ax_offset

        if not ax.yaxis.get_inverted():
            coords[:, 1] += ax_offset
        else:
            coords[:, 1] -= ax_offset

        sbar = mlines.Line2D(
            coords[:, 0],
            coords[:, 1],
            lw=defaults["lw"],
            alpha=defaults["alpha"],
            color=defaults["color"],
            zorder=1000,
        )
        sbar.set_gid(f"{scalebar}_scalebar")

        ax.add_line(sbar)
    elif settings.method in ["3d", "3d_complex"]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        left = xlim[0] + ax_offset
        bottom = zlim[0] + ax_offset
        front = ylim[0] + ax_offset

        sbar = [
            np.array([[left, front, bottom], [left, front, bottom]]),
            np.array([[left, front, bottom], [left, front, bottom]]),
            np.array([[left, front, bottom], [left, front, bottom]]),
        ]
        sbar[0][1][0] += scalebar
        sbar[1][1][1] += scalebar
        sbar[2][1][2] += scalebar

        lc = Line3DCollection(
            sbar, color=defaults["color"], lw=defaults["lw"], alpha=defaults["alpha"]
        )
        lc.set_gid(f"{scalebar}_scalebar")

        ax.add_collection3d(lc, autolim=False)


def _plot_scatter(points, ax, settings):
    """Plot dotprops."""
    if settings.method == "2d":
        default_settings = dict(c="black", zorder=4, edgecolor="none", s=1)
        default_settings.update(settings.scatter_kws)
        default_settings = _fix_default_dict(default_settings)

        x, y = _parse_view2d(points, settings.view)

        ax.scatter(x, y, **default_settings)
    elif settings.method in ["3d", "3d_complex"]:
        default_settings = dict(c="black", s=1, depthshade=False, edgecolor="none")
        default_settings.update(settings.scatter_kws)
        default_settings = _fix_default_dict(default_settings)

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], **default_settings)


def _plot_voxels(vx, color, ax, settings, **scatter_kws):
    """Plot VoxelNeuron as scatter plot."""
    # Use only the top N voxels
    assert isinstance(vx, core.VoxelNeuron)
    n_pts = 1000000
    v = vx.values
    pts = vx.voxels
    srt = np.argsort(v)[::-1]

    pts = pts[srt][:n_pts]
    v = v[srt][:n_pts]

    # Scale points by units
    pts = pts * vx.units_xyz.magnitude + vx.offset

    # Calculate colors
    cmap = color_to_cmap(color)
    colors = cmap(v / v.max())

    if settings.method == "2d":
        x, y = _parse_view2d(pts, settings.view)
        ax.scatter(x, y, c=colors, s=scatter_kws.get("size", 20))
    elif settings.method in ["3d", "3d_complex"]:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=colors,
            marker=scatter_kws.get("marker", "o"),
            s=scatter_kws.get("size", 0.1),
        )


def color_to_cmap(color):
    """Convert single color to color palette."""
    color = mcl.to_rgb(color)

    colors = [[color[0], color[1], color[2], 0], [color[0], color[1], color[2], 1]]

    return mcl.LinearSegmentedColormap.from_list("Palette", colors, N=256)


def _plot_dotprops(dp, color, ax, settings):
    """Plot dotprops."""
    # Here, we will effectively cheat and turn the dotprops into a skeleton
    # which we can then pass to _plot_skeleton
    tn = dp.to_skeleton(scale_vec=settings.dps_scale_vec)

    return _plot_skeleton(tn, color, ax, settings)


def _plot_connectors(neuron, color, ax, settings):
    """Plot connectors."""
    cn_layout = copy.deepcopy(config.default_connector_colors)

    if settings.connectors == "pre":
        connectors = neuron.presynapses
    elif settings.connectors == "post":
        connectors = neuron.postsynapses
    elif isinstance(settings.connectors, str):
        connectors = neuron.connectors[neuron.connectors.type == settings.connectors]
    elif isinstance(settings.connectors, (list, np.ndarray, tuple)):
        connectors = neuron.connectors[neuron.connectors.type.isin(settings.connectors)]
    else:
        connectors = neuron.connectors

    if connectors.empty:
        return

    # Update with user settings
    if settings.cn_layout:
        cn_layout.update(settings.cn_layout)

    # Update with user color
    if settings.cn_mesh_colors or settings.cn_layout == "neuron":
        # change all of the colors to color
        for inner_dict in cn_layout.values():
            # Skip non-color settings
            if not isinstance(inner_dict, dict):
                continue
            inner_dict["color"] = color

    if settings.cn_colors:
        if isinstance(settings.cn_colors, dict):
            cn_layout.update(settings.cn_colors)
        else:
            for inner_dict in cn_layout.values():
                # Skip non-color settings
                if not isinstance(inner_dict, dict):
                    continue
                inner_dict["color"] = settings.cn_colors

    if settings.method == "2d":
        for c, this_cn in connectors.groupby("type"):
            x, y = _parse_view2d(this_cn[["x", "y", "z"]].values, settings.view)

            ax.scatter(
                x,
                y,
                color=cn_layout[c]["color"],
                edgecolor="none",
                s=settings.cn_size if settings.cn_size else cn_layout["size"],
                zorder=1000,
            )
            ax.get_children()[-1].set_gid(f"CN_{neuron.id}")
    elif settings.method in ["3d", "3d_complex"]:
        c = [cn_layout[i]["color"] for i in connectors.type.values]
        ax.scatter(
            connectors.x.values,
            connectors.y.values,
            connectors.z.values,
            color=c,
            s=settings.cn_size if settings.cn_size else cn_layout["size"],
            depthshade=cn_layout.get("depthshade", False),
            zorder=0,
            edgecolor="none",
        )
        ax.get_children()[-1].set_gid(f"CN_{neuron.id}")


def _plot_mesh(neuron, color, ax, settings):
    """Plot mesh (i.e. MeshNeuron)."""
    # Map vertex colors to faces (if need be)
    if isinstance(color, np.ndarray) and color.ndim == 2:
        if len(color) != len(neuron.faces) and len(color) == len(neuron.vertices):
            color = np.array([color[f].mean(axis=0)[:4].tolist() for f in neuron.faces])

    ts = None
    if settings.method == "2d":
        # Generate 2d representation
        xy = np.dstack(_parse_view2d(neuron.vertices, settings.view))[0]

        pc = PolyCollection(
            xy[neuron.faces],
            linewidth=0,
            rasterized=settings.rasterize,
            edgecolor="none",
            label=getattr(neuron, "name"),
            zorder=100,  # unless we set this, the mesh will always be behind the grid
        )

        if settings.depth_coloring:
            if settings.palette:
                cmap = plt.get_cmap(settings.palette)
            else:
                cmap = DEPTH_CMAP

            pc.set_cmap(cmap)
            pc.set_norm(settings.norm)
            pc.set_alpha(settings.alpha if isinstance(settings.alpha, float) else None)

            # Get face centers
            if hasattr(neuron, "trimesh"):
                centers = neuron.trimesh.triangles_center[
                    :, _get_depth_axis(settings.view)
                ]
            else:
                centers = neuron.triangles_center[:, _get_depth_axis(settings.view)]
            # Set face centers to color the scale
            pc.set_array(centers)
        else:
            pc.set_facecolor(color)

        ax.add_collection(pc)
    else:
        ts = ax.plot_trisurf(
            neuron.vertices[:, 0],
            neuron.vertices[:, 1],
            neuron.faces,
            neuron.vertices[:, 2],
            label=getattr(neuron, "name"),
            rasterized=settings.rasterize,
            shade=settings.mesh_shade,
        )

        if settings.depth_coloring:
            if settings.palette:
                cmap = plt.get_cmap(settings.palette)
            else:
                cmap = DEPTH_CMAP
            ts.set_cmap(cmap)
            ts.set_alpha(settings.alpha)
        else:
            ts.set_facecolor(color)

        if settings.group_neurons:
            ts.set_gid(neuron.id)
    return ts


def _get_depth_axis(view):
    """Return index of axis which is not used for x/y."""
    view = [v.replace("-", "").replace("+", "") for v in view]
    depth = [ax for ax in ["x", "y", "z"] if ax not in view][0]
    map = {"x": 0, "y": 1, "z": 2}
    return map[depth]


def _parse_view2d(co, view):
    """Parse view parameter and returns x/y parameter."""
    if not isinstance(co, np.ndarray):
        co = np.array(co)

    map = {"x": 0, "y": 1, "z": 2}

    x_ix = map[view[0].replace("-", "").replace("+", "")]
    y_ix = map[view[1].replace("-", "").replace("+", "")]

    if co.ndim == 2:
        x = co[:, x_ix]
        y = co[:, y_ix]

        # Do NOT remove the list() here - for some reason the multiplication
        # above causes issues in matplotlib
        return (list(x), list(y))
    elif co.ndim == 3:
        xy = co[:, :, [x_ix, y_ix]]
        return xy
    else:
        raise ValueError(f"Expect coordinates to have 2 or 3 dimensions, got {co.ndim}")


def _set_view2d(ax, settings):
    """Set the axes based on the view parameter."""
    if settings.view[0].startswith("-") and not ax.xaxis.get_inverted():
        ax.invert_xaxis()
    if settings.view[1].startswith("-") and not ax.yaxis.get_inverted():
        ax.invert_yaxis()

    ax.set_xlabel(settings.view[0].replace("-", ""))
    ax.set_ylabel(settings.view[1].replace("-", ""))

    ax.grid()


def _set_view3d(ax, settings):
    """Parse view parameter into azimuth, elevation and roll for the camera."""
    # `view` can be e.g. ("x", "y"), ("x", "-y") or ("x", "z")
    # We need to convert this into azimuth, elevation and roll
    # Azimuth is the angle of the view in the x-y plane
    # Elevation is the angle of the view from the x-y plane
    # Roll is the rotation of the view around the z-axis
    view = tuple(settings.view)

    views = {
        ("x", "y"): (90, -90, 0),
        ("-x", "-y"): (90, 90, 0),
        ("x", "-y"): (-90, 90, 180),
        ("-x", "y"): (-90, 90, 0),
        ("x", "z"): (0, -90, 0),
        ("-x", "z"): (0, 90, 0),
        ("x", "-z"): (0, 90, -180),
        ("-x", "-z"): (0, 90, 90),
        ("y", "z"): (0, 0, 0),
        ("y", "-z"): (180, 0, 0),
        ("-y", "-z"): (0, 0, 180),
        ("-y", "z"): (180, 0, 180),
        ("z", "y"): (0, 0, -90),
        ("-z", "-y"): (0, 0, 90),
        ("-z", "y"): (180, 180, -90),
        ("z", "-y"): (180, 180, 90),
        # TODO: add (z, x) and (y, x) views
    }
    if view not in views:
        raise ValueError(
            f"View {view} not possible without flipping data. Please choose from {views.keys()}"
        )

    # Set view
    ax.view_init(*views[view])

    # This both sets the aspect ratio as well as zooming in slightly
    # Note: we do not have to use ax.set_aspect("equal") again
    ax.set_box_aspect([1, 1, 1], zoom=1.2)

    # Set aspect ratio
    # ax.set_aspect('equal', adjustable='box')

    # Set labels in case somebody unhides the axis
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    non_view_axis = [
        v for v in ["x", "y", "z"] if v not in [v.replace("-", "") for v in view]
    ][0]
    if settings.non_view_axes3d == "hide":
        getattr(ax, f"set_{non_view_axis}label")("")
        getattr(ax, f"set_{non_view_axis}ticks")([])
    elif settings.non_view_axes3d == "fade":
        getattr(ax, f"set_{non_view_axis}label")(non_view_axis, alpha=0.5)
        getattr(ax, f"set_{non_view_axis}ticks")([])


def _plot_skeleton(neuron, color, ax, settings):
    """Plot skeleton."""

    if settings.method == "2d":
        if not settings.depth_coloring and not (
            isinstance(color, np.ndarray) and color.ndim == 2
        ):
            # Generate by-segment coordinates
            coords = segments_to_coords(neuron, modifier=(1, 1, 1))

            # We have to add (None, None, None) to the end of each
            # slab to make that line discontinuous there
            coords = np.vstack([np.append(t, [[None] * 3], axis=0) for t in coords])

            x, y = _parse_view2d(coords, settings.view)
            this_line = mlines.Line2D(
                x,
                y,
                lw=settings.linewidth,
                ls=settings.linestyle,
                color=color,
                rasterized=settings.rasterize,
                label=f'{getattr(neuron, "name", "NA")} - #{neuron.id}',
            )
            ax.add_line(this_line)
        else:
            if isinstance(settings.palette, str):
                cmap = plt.get_cmap(settings.palette)
            else:
                cmap = DEPTH_CMAP

            coords = tn_pairs_to_coords(neuron, modifier=(1, 1, 1))
            xy = _parse_view2d(coords, settings.view)
            lc = LineCollection(
                xy,
                cmap=cmap if settings.depth_coloring else None,
                norm=settings.norm if settings.depth_coloring else None,
                rasterized=settings.rasterize,
                joinstyle="round",
            )

            lc.set_linewidth(settings.linewidth)
            lc.set_linestyle(settings.linestyle)
            lc.set_label(f'{getattr(neuron, "name", "NA")} - #{neuron.id}')

            if settings.depth_coloring:
                lc.set_array(
                    neuron.nodes.loc[neuron.nodes.parent_id >= 0][
                        ["x", "y", "z"]
                    ].values[:, _get_depth_axis(settings.view)]
                )
            elif isinstance(color, np.ndarray) and color.ndim == 2:
                # If we have a color for each node, we need to drop the roots
                if color.shape[1] != coords.shape[0]:
                    lc.set_color(color[neuron.nodes.parent_id.values >= 0])
                else:
                    lc.set_color(color)

            ax.add_collection(lc)

        if settings.soma and np.any(neuron.soma):
            soma = utils.make_iterable(neuron.soma)
            # If soma detection is messed up we might end up producing
            # dozens of soma which will freeze the kernel
            if len(soma) >= 10:
                logger.warning(f"{neuron.id} - {len(soma)} somas found.")
            for s in soma:
                if isinstance(color, np.ndarray) and color.ndim > 1:
                    s_ix = np.where(neuron.nodes.node_id == s)[0][0]
                    soma_color = color[s_ix]
                else:
                    soma_color = color

                n = neuron.nodes.set_index("node_id").loc[s]
                r = (
                    getattr(n, neuron.soma_radius)
                    if isinstance(neuron.soma_radius, str)
                    else neuron.soma_radius
                )

                if settings.depth_coloring:
                    d = [n.x, n.y, n.z][_get_depth_axis(settings.view)]
                    soma_color = DEPTH_CMAP(settings.norm(d))

                soma_defaults = dict(
                    radius=r,
                    fill=True,
                    fc=soma_color,
                    rasterized=settings.rasterize,
                    zorder=4,
                    edgecolor="none",
                )
                if isinstance(settings.soma, dict):
                    soma_defaults.update(settings.soma)

                sx, sy = _parse_view2d(np.array([[n.x, n.y, n.z]]), settings.view)
                c = mpatches.Circle((sx[0], sy[0]), **soma_defaults)
                ax.add_patch(c)
        return None, None

    elif settings.method in ["3d", "3d_complex"]:
        # For simple scenes, add whole neurons at a time to speed up rendering
        if settings.method == "3d":
            if (
                isinstance(color, np.ndarray) and color.ndim == 2
            ) or settings.depth_coloring:
                coords = tn_pairs_to_coords(neuron, modifier=(1, 1, 1))
                # If we have a color for each node, we need to drop the roots
                if isinstance(color, np.ndarray) and color.shape[1] != coords.shape[0]:
                    line_color = color[neuron.nodes.parent_id.values >= 0]
                else:
                    line_color = color
            else:
                # Generate by-segment coordinates
                coords = segments_to_coords(neuron, modifier=(1, 1, 1))
                line_color = color

            if settings.palette:
                cmap = plt.get_cmap(settings.palette)
            else:
                cmap = DEPTH_CMAP

            lc = Line3DCollection(
                coords,
                color=line_color if not settings.depth_coloring else None,
                label=neuron.id,
                cmap=cmap if settings.depth_coloring else None,
                lw=settings.linewidth,
                joinstyle="round",
                rasterized=settings.rasterize,
                linestyle=settings.linestyle,
            )
            if settings.group_neurons:
                lc.set_gid(neuron.id)
            # Need to get this before adding data
            line3D_collection = lc
            ax.add_collection3d(lc, autolim=False)

        # For complex scenes, add each segment as a single collection
        # -> helps reducing Z-order errors
        elif settings.method == "3d_complex":
            # Generate by-segment coordinates
            coords = segments_to_coords(neuron, modifier=(1, 1, 1))
            for c in coords:
                lc = Line3DCollection(
                    [c],
                    color=color,
                    lw=settings.linewidth,
                    rasterized=settings.rasterize,
                    linestyle=settings.linestyle,
                )
                if settings.group_neurons:
                    lc.set_gid(neuron.id)
                ax.add_collection3d(lc, autolim=False)
            line3D_collection = None

        surf3D_collections = []
        if settings.soma and not isinstance(getattr(neuron, "soma", None), type(None)):
            soma = utils.make_iterable(neuron.soma)
            # If soma detection is messed up we might end up producing
            # dozens of soma which will freeze the kernel
            if len(soma) >= 5:
                logger.warning(
                    f"Neuron {neuron.id} appears to have {len(soma)}"
                    " somas. Skipping plotting its somas."
                )
            else:
                for s in soma:
                    if isinstance(color, np.ndarray) and color.ndim > 1:
                        s_ix = np.where(neuron.nodes.node_id == s)[0][0]
                        soma_color = color[s_ix]
                    else:
                        soma_color = color

                    n = neuron.nodes.set_index("node_id").loc[s]
                    r = (
                        getattr(n, neuron.soma_radius)
                        if isinstance(neuron.soma_radius, str)
                        else neuron.soma_radius
                    )

                    resolution = 20
                    u = np.linspace(0, 2 * np.pi, resolution)
                    v = np.linspace(0, np.pi, resolution)
                    x = r * np.outer(np.cos(u), np.sin(v)) + n.x
                    y = r * np.outer(np.sin(u), np.sin(v)) + n.y
                    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + n.z

                    soma_defaults = dict(
                        color=soma_color,
                        shade=settings.mesh_shade,
                        rasterized=settings.rasterize,
                    )
                    if isinstance(settings.soma, dict):
                        soma_defaults.update(settings.soma)

                    surf = ax.plot_surface(x, y, z, **soma_defaults)

                    if settings.group_neurons:
                        surf.set_gid(neuron.id)

                    surf3D_collections.append(surf)

        return line3D_collection, surf3D_collections


def _plot_volume(volume, color, ax, settings):
    """Plot volume."""
    name = getattr(volume, "name")

    if len(color) == 4:
        this_alpha = color[3]
    else:
        this_alpha = 1

    if settings.volume_outlines:
        fill, lw, fc, ec = False, 1, "none", color
    else:
        fill, lw, fc, ec = True, 0, color, "none"

    if settings.method == "2d":
        if settings.volume_outlines in (False, "both"):
            # Generate 2d representation
            xy = np.dstack(_parse_view2d(volume.verts, settings.view))[0]

            # Generate a patch for each face
            pc = PolyCollection(
                xy[volume.faces],
                linewidth=lw,
                facecolor=fc,
                rasterized=settings.rasterize,
                edgecolor=ec,
                alpha=this_alpha,
                zorder=0,
            )
            ax.add_collection(pc)

        if settings.volume_outlines in (True, "both"):
            verts = volume.to_2d(view=settings.view, alpha=0.001)
            vpatch = mpatches.Polygon(
                verts,
                closed=True,
                lw=lw,
                fill=fill,
                rasterized=settings.rasterize,
                fc=fc,
                ec=ec,
                zorder=0,
                alpha=1 if settings.volume_outlines == "both" else this_alpha,
            )
            ax.add_patch(vpatch)

    elif settings.method in ["3d", "3d_complex"]:
        if settings.volume_outlines:
            logger.warning("Volume outlines are not supported for 3d plotting. ")

        verts = np.vstack(volume.vertices)

        # Add alpha
        if len(color) == 3:
            color = (color[0], color[1], color[2], 0.1)

        ts = ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            volume.faces,
            verts[:, 2],
            label=name,
            rasterized=settings.rasterize,
            color=color,
        )
        ts.set_gid(name)


def _fix_default_dict(x):
    """Consolidate duplicate settings.

    E.g. scatter kwargs when 'c' and 'color' is provided.

    """
    # The first entry is the "survivor"
    duplicates = [["color", "c"], ["size", "s"], ["alpha", "a"]]

    for dupl in duplicates:
        if sum([v in x for v in dupl]) > 1:
            to_delete = [v for v in dupl if v in x][1:]
            _ = [x.pop(v) for v in to_delete]

    return x


def proj_points(points, M):
    """Project points using a projection matrix.

    This was previously done using the analagous function
    mpl_toolkits.mplot3d.proj3d.proj_points but that is deprecated.
    """
    xs, ys, zs = zip(*points)
    vec = np.array([xs, ys, zs, np.ones_like(xs)])

    vecw = np.dot(M, vec)
    w = vecw[3]
    # clip here..
    txs, tys, tzs = vecw[0] / w, vecw[1] / w, vecw[2] / w

    return np.column_stack((txs, tys, tzs))


def update_axes3d_bounds(ax):
    """Update axis bounds and remove default points (0,0,0) and (1,1,1)."""
    # Collect data points present in the figure
    points = []
    for c in ax.collections:
        if isinstance(c, Line3DCollection):
            for s in c._segments3d:
                points.append(s)
        elif isinstance(c, Poly3DCollection):
            points.append(c._vec[:3, :].T)
        elif isinstance(c, (Path3DCollection, Patch3DCollection)):
            points.append(np.array(c._offsets3d).T)

    if not len(points):
        return

    points = np.vstack(points)

    # If this is the first set of points, we need to overwrite the defaults
    # That should happen automatically but for some reason doesn't for 3d axes
    if not getattr(ax, "had_data", False):
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        new_xybounds = np.array([[mn[0], mn[1]], [mx[0], mx[1]]])
        new_zzbounds = np.array([[mn[2], mn[2]], [mx[2], mx[2]]])
        ax.xy_dataLim.set_points(new_xybounds)
        ax.zz_dataLim.set_points(new_zzbounds)
        ax.xy_viewLim.set_points(new_xybounds)
        ax.zz_viewLim.set_points(new_zzbounds)
        ax.had_data = True
    else:
        ax.auto_scale_xyz(
            points[:, 0].tolist(),
            points[:, 1].tolist(),
            points[:, 2].tolist(),
            had_data=True,
        )
