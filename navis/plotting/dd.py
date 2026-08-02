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
import matplotlib.path as mpath
import matplotlib.colors as mcl
from mpl_toolkits.mplot3d.art3d import (
    Line3DCollection,
    Poly3DCollection,
    Path3DCollection,
    Patch3DCollection,
)
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.cm import ScalarMappable

import numpy as np

import pint
import warnings

from typing import Union, List, Tuple

from .. import utils, config, core, conversion
from ._common import (
    apply_shade_by,
    connector_colors,
    prepare_cn_colors,
    resolve_cn_layout,
    resolve_connectors,
    resolve_somata,
    use_radius,
)
from .colors import prepare_colormap, vertex_colors, parse_color_by
from .plot_utils import (
    mesh_faces,
    segments_to_coords,
    skeleton_capsules,
    tn_pairs_to_coords,
)
from .settings import Matplotlib2dSettings

__all__ = ["plot2d"]

logger = config.get_logger(__name__)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])

# Default colormap for depth coloring
DEPTH_CMAP = mpl.cm.jet

#: Named bundles of arguments for `plot2d(..., style=...)`. A style only fills in
#: arguments the caller did not pass themselves, so every part of it stays
#: overridable - there is nothing in here you cannot also switch on by hand.
PLOT_STYLES = {
    "publication": dict(
        radius="auto", depth_sort=True, soma=True, mesh_shade=True
    ),
}

# Arguments that can be spelled more than one way, so that a style does not
# override a value the caller passed under an alias. Styles are applied before
# the settings object exists, which is the one place that otherwise resolves
# these - so take the table from there rather than writing it out again.
_STYLE_ALIASES = {syn[0]: tuple(syn) for syn in Matplotlib2dSettings()._synonyms}

# Default number of depth bins for `depth_sort=True`. More bins resolve overlaps
# more finely but cost two artists each, per neuron.
DEPTH_BINS = 10

#: `depth_sort="global"`: sort every element of a kind together rather than
#: bucketing each neuron separately. See `_GlobalSort` for what that costs.
DEPTH_SORT_GLOBAL = "global"

# Default halo width in points for `halo=True`.
HALO_WIDTH = 3.0

# Width range for `taper`, as a fraction of `linewidth`.
TAPER_RANGE = (0.35, 3.5)

#: Surface shading modes for `plot2d(..., mesh_shade=...)` with `method="2d"`.
#: `True` is an alias for "lambert".
MESH_SHADE_MODES = ("lambert", "cel", "rim", "ghost")

# Direction the key light comes from, in view space (x right, y up, z at the
# viewer): over the viewer's left shoulder and slightly above, which is where
# every anatomical illustration puts it.
MESH_LIGHT = (-0.4, 0.6, 0.7)

# How much light a face gets when it faces straight away from the key light.
MESH_AMBIENT = 0.25

# Number of tones in "cel".
CEL_BANDS = 3

# Default z-order for meshes: well clear of skeletons at 1-2, which is how they
# have always been stacked. `depth_sort` overrides it and puts them in the
# skeleton band so the two interleave.
MESH_ZORDER = 100


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
    x :                 Skeleton | Mesh | Dotprops | Voxels | NeuronList | Volume | np.ndarray
                        Objects to plot:
                         - multiple objects can be passed as list (see examples)
                         - numpy array of shape (N, 3) is intepreted as points for
                           scatter plots

    Object parameters
    -----------------
    Each of these notes which objects it applies to and, where it matters, which
    `methods` it works with. Unless an entry says otherwise, a parameter that does
    not apply is simply ignored - `taper` on a `Mesh`, say - so these are safe to
    set for a mixed scene. The few that raise instead say so, and setting a
    `method="2d"`-only parameter for a 3d method logs a warning.

    soma :              bool | dict, default=True

                        Plot soma if one exists. Size of the soma is determined
                        by the neuron's `.soma_radius` property which defaults
                        to the "radius" column for `Skeletons`. You can also
                        pass `soma` as a dictionary to customize the appearance
                        of the soma - for example `soma={"color": "red", "lw": 2, "ec": 1}`.

                        `Skeletons` only, since no other neuron type has a soma.
                        Works with all `methods`.

    radius :            bool | "auto" | "lw", default=False

                        If "auto" will plot neurites of `Skeletons` with radius
                        if they have radii. If True, will try plotting neurites of
                        `Skeletons` with radius regardless. The radius can be
                        scaled by `linewidth`. Note that this will increase rendering
                        time.

                        With `method="2d"` the neurites are outlined directly in the
                        view plane; the 3d methods mesh them with
                        [`navis.conversion.tree2meshneuron`][] instead.

                        Use `"lw"` to map the radius onto the line width instead of
                        drawing an outline. It looks near-identical, gets exact
                        round joins for free, and more than halves the size of
                        vector output - at the cost of being somewhat slower to
                        render, and of only being correct on axes it can measure:
                        line widths are in points, so the conversion from data
                        units is redone on every draw.

                        `Skeletons` only. Works with all `methods`, but `"lw"` is
                        a `method="2d"` feature - the 3d methods have no line width
                        in data units and fall back to meshing, i.e. to `True`.

    taper :             "strahler" | "subtree", default=None

                        Vary the width of neurites by a topological measure instead
                        of keeping it constant - useful for the many skeletons whose
                        `radius` column is a placeholder. "strahler" tapers by
                        Strahler index (chunky, shows the branch hierarchy),
                        "subtree" by the height of the subtree below each node
                        (smooth). Widths range from 0.35x to 3.5x `linewidth`.
                        Ignored when neurites are drawn with `radius`.

                        `method="2d"` only, and only meaningful for `Skeletons`:
                        `Dotprops` are drawn as skeletons but have no branch
                        hierarchy to taper by, so they come out at a single width.

    halo :              bool | float | dict, default=False

                        Draw each neuron with a background-coloured outline
                        underneath, so that crossings read as one neuron passing in
                        front of another. Pass a number for the width in points
                        (`True` uses 3), or a dict with "width" and/or "color" keys.
                        The colour defaults to the axes' background.

                        `method="2d"` only. Applies to `Skeletons`, `Dotprops` and
                        `Meshes`; `Volumes` never get one, since they are scenery
                        and belong behind everything.

    depth_sort :        bool | int | "global", default=False

                        Interleave neurons by depth instead of drawing them one
                        after the other, which is as close to real occlusion as
                        matplotlib gets. Two ways to do it:

                          - `True` or an integer: bucket everything into that many
                            bins along the axis pointing into the screen (`True`
                            uses 10) and give each bin its own z-order. Approximate,
                            but cheap and it interleaves the different neuron types
                            with each other. A negative number flips which end of
                            the depth axis counts as nearest.
                          - `"global"`: sort *exactly*, merging each neuron type
                            into one artist so that two neurons interleave element
                            by element rather than bin by bin.

                        Bins cost one artist per bin per neuron (two with `halo`),
                        so keep the count modest for large `NeuronLists`.
                        `"global"` costs the aggregate artists that make 2d
                        plotting fast: a flat `Mesh` is normally one filled outline
                        and a `Skeleton` a handful of polylines, and sorting across
                        neurons forces both down to their elements. For skeletons
                        that still comes out cheaper than 10 bins; for meshes it is
                        several times the cost of either. It also gives up the
                        fill-once `alpha` that a single outline buys a `Mesh` or a
                        `radius` ribbon. Which is to say: reach for it on a finished
                        figure, not while you are exploring.

                        With `"global"`, artists are stacked in the order their
                        neuron type first appears in `x`, `Dotprops` merge with the
                        `Skeletons` they are drawn as, per-neuron legend entries
                        become proxy handles, and `halo` is not available - a halo
                        has to sit *between* two neurons of a type, which a single
                        artist cannot express, so passing both warns and falls back
                        to bins.

                        `method="2d"` only - the 3d methods do their own z-ordering.
                        Applies to `Skeletons`, `Dotprops` and `Meshes`; `Voxels`
                        are a scatter and keep their own artist, and `Volumes` stay
                        behind all of them either way.

    style :             str, default=None

                        Name of a bundle of the settings above, see
                        `navis.plotting.dd.PLOT_STYLES` for what each one sets.
                        Currently only "publication" (radius, depth-sorted, soma,
                        shaded meshes). A style never overrides an argument you
                        passed yourself, so e.g. `style="publication",
                        radius=False` does what it says.

                        A style is just a set of defaults, so it inherits the
                        applicability of whatever it sets - "publication" has parts
                        that only apply to `Skeletons` and parts that only work with
                        `method="2d"`.

    linewidth :         int | float, default=1

                        Width of neurites, and the scaling factor for `radius`.
                        Also accepts alias `lw`.

                        `Skeletons` and `Dotprops` only, with all `methods`. The
                        contour drawn by `volume_outlines` has a fixed width of its
                        own and does not follow this.

    linestyle :         str, default='-'

                        Line style of neurites. Also accepts alias `ls`.

                        `Skeletons` and `Dotprops` only, and only where they are
                        actually drawn as lines - an outline (`radius` in 2d) or a
                        mesh (`radius` in 3d) has no line style, though
                        `radius="lw"` keeps it. Works with all `methods`.

    color :             None | str | tuple | list | dict, default=None

                        Use single str (e.g. `'red'`) or `(r, g, b)` tuple
                        to give all neurons the same color. Use `list` of
                        colors to assign colors: `['red', (1, 0, 1), ...].
                        Use `dict` to map colors to neuron IDs:
                        `{id: (r, g, b), ...}`.

                        Applies to every neuron type and to `Volumes`, with all
                        `methods`. Bare `(N, 3)` point arrays take their color from
                        `scatter_kws` instead.

    palette :           str | array | list of arrays, default=None

                        Name of a matplotlib or seaborn palette. If `color` is
                        not specified will pick colors from this palette.

                        Same scope as `color`.

    color_by :          str | array | list of arrays, default = None

                        Color neurons by a property. Can be:
                          - a list/array of labels, one per each neuron
                          - a neuron property (str)
                          - a column name in the node table of `Skeletons`
                          - a list/array of values for each node
                        Numerical values will be normalized. You can control
                        the normalization by passing a `vmin` and/or `vmax` parameter.

                        One value per neuron works for any neuron type. Colouring
                        *within* a neuron needs a node or vertex table, so it is
                        `Skeletons` and `Meshes` only - `Dotprops` and `Voxels`
                        raise. Works with all `methods`.

    shade_by :          str | array | list of arrays, default=None

                        Similar to `color_by` but will affect only the alpha
                        channel of the color. If `shade_by='strahler'` will
                        compute Strahler order if not already part of the node
                        table (Skeletons only). Numerical values will be
                        normalized. You can control the normalization by passing
                        a `smin` and/or `smax` parameter.

                        Always per node/vertex, so `Skeletons` and `Meshes` only.
                        Works with all `methods`.

    alpha :             float [0-1], default=None

                        Alpha value for neurons. `None` means "leave alone", so
                        neurons are opaque unless `color` carries its own alpha
                        (rgb*a*); setting `alpha` explicitly overrides that. You
                        can set the alpha value for connectors with `cn_alpha`.

                        Applies to every neuron type except `Voxels` (whose opacity
                        encodes their value), and to `Volumes`. Works with all
                        `methods`.

    mesh_shade :        bool | str | dict, default=False

                        Shade `Mesh` neurons and `Volumes` so they look like
                        surfaces rather than flat blobs. Shading multiplies into
                        whatever colour they already have, so it composes with
                        `color`, `color_by` and `depth_coloring`.

                        With `method="2d"` this takes a mode:

                          - `True` or `"lambert"`: diffuse shading
                          - `"cel"`: the same, posterised into three tones
                          - `"rim"`: diffuse plus a bright rim at grazing angles
                          - `"ghost"`: opacity from the grazing angle instead of
                            brightness, so a neuron inside a neuropil stays visible

                        Pass a dict to tune it, e.g.
                        `mesh_shade={"mode": "lambert", "ambient": .4}`; `light` is
                        a direction in view space (x right, y up, z at the viewer)
                        and `strength` scales the "rim" and "ghost" terms.

                        The 3d methods only understand `True`/`False`, and there it
                        covers everything that is real geometry: `Meshes`, somata and
                        `Skeletons` drawn with `radius`. `Volumes` are the exception -
                        matplotlib always shades those in 3d, whatever you pass.

                        Note that culling back faces and painting the rest from
                        back to front happens either way in 2d - `mesh_shade` only
                        controls the lighting on top of that. A shaded mesh does
                        need one polygon per face rather than a single filled
                        outline, though, so where a translucent mesh overlaps
                        itself it will double-darken instead of filling once.

    depth_coloring :    bool, default=False

                        If True, will use neuron color to encode depth (Z).
                        Overrides `color` argument. Does not work with
                        `method = '3d_complex'` (raises).

                        Applies to `Skeletons`, `Dotprops` and `Meshes`; `Volumes`
                        and `Voxels` keep their own colors.

    depth_scale :       bool, default=True

                        If True and `depth_coloring=True` will plot a scale.

                        `method="2d"` only - with `method="3d"` the depth colors are
                        recomputed as you rotate, so there is no fixed scale to draw.

    connectors :        bool | "presynapses" | "postsynapses" | str | list, default=False

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
                            `display` is either `"lines"` (the default: a stalk
                            from each connector back to its node, `Skeletons`
                            only) or `"circles"`.
                          - `cn_size` (float): Size of the connectors.
                          - `cn_alpha` (float): Transparency of the connectors.
                          - `cn_mesh_colors` (bool): Whether to color the connectors
                            by the neuron's color.

                        All connectors of a neuron go into a single artist, drawn
                        in a fixed, shuffled order. Painting one type after
                        another would let a rare type bury a common one wherever
                        markers overlap; interleaving them keeps the visible mix
                        a fair sample of the real one.

                        Applies to any neuron that carries a connector table,
                        whatever its type, and works with all `methods`.

    connectors_only :   boolean, default=False

                        Plot only connectors, not the neuron. Same scope as
                        `connectors`.

    cn_color_by :       str | array, optional

                        Color connectors by a column of the connector table (or
                        by an array with one value per connector) rather than by
                        their `type`. Numerical data gets a colormap, categorical
                        data one color per level - and either way the scale is
                        shared by all neurons, so the same value is the same color
                        throughout. Missing values are drawn grey.

                        Overrides `cn_colors`/`cn_mesh_colors`. The `plotly` and
                        `k3d` backends fall back to markers when this is set,
                        since a line there carries a single color; `plot2d` colors
                        stalks per connector like anything else.

                        Same scope as `connectors`; works with all `methods`.

    cn_palette :        str | list | dict, optional

                        Palette for `cn_color_by`: the name of a colormap for
                        numerical data, and for categorical data a palette name,
                        a list of colors or a dict keyed by level. Falls back to
                        `palette`, then to `"viridis"`.

    cn_legend :         bool, default=False

                        Add a legend entry per connector *type* - not per type
                        per neuron, which is why this is not simply a label on
                        the artists. With a numerical `cn_color_by` you get a
                        colorbar instead. Call `ax.legend()` afterwards as usual.

                        `method="2d"` and `"3d"` only.

    cn_zorder :         int, optional

                        Where connectors sit in the stack. Defaults to 1000,
                        i.e. above everything, so that a synapse is never hidden
                        by the neurite it sits on; pass a lower value to put them
                        behind the neurons instead.

                        `method="2d"` only - the 3d methods leave stacking to
                        `matplotlib`.

    scatter_kws :       dict, default={}

                        Parameters to be used when plotting points. Accepted
                        keywords are: `size` and `color`.

                        Applies to bare `(N, 3)` point arrays and to `Voxels`, which
                        are drawn as a scatter too - though `Voxels` only take `size`
                        from here and their color from `color`/`palette`. Works with
                        all `methods`.

    volume_outlines :   bool | "both", default=False

                        If True will plot volume outline with no fill. Only
                        works with `method="2d"`. Requires the `shapely` package.
                        The contour is always drawn opaque - a volume's alpha is
                        a fill alpha, and there is nothing to see through a line.

                        `Volumes` only; `Meshes` are always drawn as surfaces.

    dps_scale_vec :     float

                        Scale vector for dotprops. `Dotprops` only; works with all
                        `methods`.

    Figure parameters
    -----------------
    method :            '2d' (default) | '3d' | '3d_complex'

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

                        If True, will scale the axes to fit the data. Works with all
                        `methods`.

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

                        Applies to neurons and `Volumes` with all `methods`. The
                        scatter artists - `Voxels`, point arrays and connectors -
                        are left as vectors.

    orthogonal :        bool, default=True

                        Whether to use orthogonal or perspective view for
                        methods '3d' and '3d_complex'.

    group_neurons :     bool, default=False

                        If True, neurons will be grouped by tagging their artists
                        with the neuron ID. Works with SVG export but not PDF.

                        The 3d `methods` only - nothing is tagged with
                        `method="2d"`. With `method='3d_complex'` a `Skeleton` is
                        drawn segment by segment, so it comes out as hundreds of
                        groups sharing an ID rather than as one group.

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

    Tell overlapping neurons apart by giving each a halo and letting them
    interleave by depth:

    >>> fig, ax = navis.plot2d(nl, method='2d', halo=True, depth_sort=True)
    >>> plt.show() # doctest: +SKIP

    Draw neurites at their real radius, with a soma - or taper them by branch
    order if the radii are not trustworthy:

    >>> fig, ax = navis.plot2d(nl[0], method='2d', style='publication')
    >>> fig, ax = navis.plot2d(nl[0], method='2d', taper='strahler')
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

    See the [plotting intro](../../generated/gallery/1a_plotting_general/tutorial_plotting_00_intro)
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
    # (3) synonyms. Styles are resolved first so that they only ever fill in
    # arguments that are not already spoken for.
    settings = Matplotlib2dSettings().update_settings(**_apply_style(kwargs))

    _METHOD_OPTIONS = ["2d", "3d", "3d_complex"]
    if settings.method not in _METHOD_OPTIONS:
        raise ValueError(
            f'Unknown method "{settings.method}". Please use either: '
            f'{",".join(_METHOD_OPTIONS)}'
        )

    # Validate up front and for every method - a typo in the mode name should not
    # depend on which one you asked for, and the 3d path would silently read it
    # as a plain `True`
    _mesh_shade_spec(settings)

    # A halo has to sit *between* two neurons of the same kind, which one merged
    # artist has no z-order to express - so the halo wins and we fall back to the
    # bins, which are the same sort, only quantised.
    if _depth_sort_mode(settings) == DEPTH_SORT_GLOBAL and settings.halo:
        logger.warning(
            'depth_sort="global" merges each neuron type into a single artist, '
            "which leaves no room for a halo between two of them - falling back "
            f"to {DEPTH_BINS} bins. Pass a bin count to choose your own."
        )
        settings.depth_sort = True

    # These all work in the view plane, which the 3d methods do not have
    if settings.method != "2d":
        ignored = [p for p in ("halo", "depth_sort", "taper") if settings.get(p)]
        if settings.mesh_shade and not isinstance(settings.mesh_shade, bool):
            ignored.append("mesh_shade modes")
        if ignored:
            logger.warning(
                f"{', '.join(ignored)} only work with `method=\"2d\"` and will be "
                f'ignored for method "{settings.method}".'
            )

    # Parse objects
    (neurons, volumes, points, _) = utils.parse_objects(x)

    # Here we check whether `color_by` is a neuron property which we
    # want to translate into a single color per neuron, or a
    # per node/vertex property which we will parse later
    color_neurons_by, settings.color_by = parse_color_by(
        settings.color_by, neurons, settings.palette
    )

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

    neuron_cmap = apply_shade_by(neuron_cmap, neurons, settings, color_range=1)

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

    # Depth bins have to be shared by all neurons for them to interleave
    _prepare_depth_bins(neurons, settings)
    if _depth_sort_mode(settings) == DEPTH_SORT_GLOBAL and settings.method == "2d":
        settings._global_sort = _GlobalSort()

    # ... and so does the colour scale behind `cn_color_by`
    prepare_cn_colors(neurons, settings)
    # dict rather than list: one entry per connector *type*, however many neurons
    # carry that type, and insertion-ordered so the legend follows the data
    settings._cn_legend = {}

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
    # width of one neuron's z-order slot, for when there are no depth bins to
    # stack them by instead - see `_neuron_slot`
    settings._z_step = 1 / max(len(neurons), 1)
    for i, neuron in enumerate(
        config.tqdm(
            neurons,
            desc="Plot neurons",
            leave=False,
            disable=config.pbar_hide or (len(neurons) <= 10),
        )
    ):
        settings._z_index = i
        if not settings.connectors_only:
            if isinstance(neuron, core.Skeleton) and neuron.nodes.empty:
                logger.warning(f"Skipping Skeleton w/o nodes: {neuron.label}")
                continue
            if isinstance(neuron, core.Skeleton) and neuron.nodes.shape[0] == 1:
                logger.warning(f"Skipping single-node Skeleton: {neuron.label}")
                continue
            elif isinstance(neuron, core.Mesh) and neuron.faces.size == 0:
                logger.warning(f"Skipping Mesh w/o faces: {neuron.label}")
                continue
            elif isinstance(neuron, core.Dotprops) and neuron.points.size == 0:
                logger.warning(f"Skipping Dotprops w/o points: {neuron.label}")
                continue

            # asked once per neuron and handed down: `use_radius` rescans the
            # radius column and warns when it cannot be used, so calling it again
            # further in would repeat the warning
            radius = use_radius(neuron, settings)
            if radius:
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

                # In 2d we can outline the tube in the view plane directly, which
                # is both cheaper and gives us something we can shade, fade and
                # halo. The 3d methods need actual geometry, so they still mesh.
                if settings.method != "2d":
                    _neuron = conversion.tree2meshneuron(
                        neuron,
                        warn_missing_radii=False,
                        radius_scale_factor=settings.get("linewidth", 1),
                    )
                    _neuron.connectors = neuron.connectors
                    neuron = _neuron

                    # See if we need to map colors to vertices
                    if (
                        isinstance(neuron_cmap[i], np.ndarray)
                        and neuron_cmap[i].ndim == 2
                    ):
                        neuron_cmap[i] = neuron_cmap[i][neuron.vertex_map]

            if isinstance(neuron, core.Skeleton):
                lc, sc = _plot_skeleton(
                    neuron, neuron_cmap[i], ax, settings, radius=radius
                )
                # Keep track of visuals related to this neuron
                visuals[neuron] = {"skeleton": lc, "somata": sc}
            elif isinstance(neuron, core.Mesh):
                m = _plot_mesh(neuron, neuron_cmap[i], ax, settings)
                visuals[neuron] = {"mesh": m}
            elif isinstance(neuron, core.Dotprops):
                dp = _plot_dotprops(neuron, neuron_cmap[i], ax, settings)
                visuals[neuron] = {"dotprop": dp}
            elif isinstance(neuron, core.Voxels):
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

        _plot_connectors(neuron, neuron_cmap[i], ax, settings)

    # Nothing has been drawn yet under `depth_sort="global"` - this is where the
    # collected geometry becomes one artist per neuron type
    if _global_sort(settings) is not None:
        settings._global_sort.flush(ax, settings)

    if settings.cn_legend:
        _add_cn_legend(ax, settings)

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

            if isinstance(neuron, core.Skeleton) and not isinstance(
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
        # Note: without neurons there is nothing to normalize against and hence
        # also nothing to depth-code
        if settings.method == "2d" and settings.depth_scale:
            if not isinstance(settings.norm, type(None)):
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

            if all_co:
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
    """Plot Voxels as scatter plot."""
    # Use only the top N voxels
    assert isinstance(vx, core.Voxels)
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

    # the decision is normally made in `plot2d`, which only sees the dotprops
    return _plot_skeleton(tn, color, ax, settings, radius=use_radius(tn, settings))


def _add_cn_legend(ax, settings):
    """Explain the connector colours - as legend entries or as a colorbar.

    Entries are collected across neurons and keyed by label, so three neurons
    with pre- and postsynapses contribute two entries, not six. A continuous
    `cn_color_by` has no entries to collect and gets a colorbar instead, which is
    the only honest way to show a ramp.
    """
    kind, scale = getattr(settings, "_cn_scale", None) or (None, None)
    if kind == "numeric":
        palette = settings.get("cn_palette", None) or settings.get("palette", None)
        mappable = ScalarMappable(
            norm=mpl.colors.Normalize(*scale),
            cmap=plt.get_cmap(palette if isinstance(palette, str) else "viridis"),
        )
        label = settings.cn_color_by
        ax.get_figure().colorbar(
            mappable,
            ax=ax,
            fraction=0.03,
            pad=0.02,
            label=label if isinstance(label, str) else None,
        )
        return

    for label, rgba in settings._cn_legend.items():
        # empty data, so the entry costs nothing and does not pull on the limits
        ax.add_line(
            mlines.Line2D([], [], color=rgba, label=label, marker="o", ls="none")
        )


def _cn_shuffle(n_items, colors):
    """A fixed shuffle of `n_items`, or None if there is nothing to interleave.

    Drawing one type after another means the type painted last wins wherever
    markers overlap, so a rare type can bury a common one - on the example neuron
    232 presynapses hide 1933 postsynapses and the arbour reads as an output
    region. Interleaving the draw order makes the visible mix a fair sample of
    the real one.

    The permutation is seeded, so the same data always produces the same figure -
    a plot that reshuffled itself on every call would be worse than the problem.
    """
    if n_items < 2 or len(np.unique(colors, axis=0)) < 2:
        return None
    return np.random.default_rng(0).permutation(n_items)


def _cn_stalks(neuron, connectors):
    """`(n, 2, 3)` segments from each connector back to the node it sits on, and
    a mask saying which connectors got one.

    This is what `cn_layout={"display": "lines"}` draws instead of a marker. It
    needs somewhere to draw *to*, so it is skeletons only - and only those whose
    `node_id`s still resolve, which pruning can easily break.
    """
    if not isinstance(neuron, core.Skeleton) or "node_id" not in connectors:
        return None

    nodes = neuron.nodes.set_index("node_id")
    ends = nodes.reindex(connectors.node_id.values)[["x", "y", "z"]].values
    keep = np.isfinite(ends).all(axis=1)
    if not keep.any():
        return None

    starts = connectors[["x", "y", "z"]].values
    return np.stack([starts[keep], ends[keep]], axis=1), keep


def _plot_connectors(neuron, color, ax, settings):
    """Plot connectors, all of them in one artist.

    One artist rather than one per type is what lets the draw order be shuffled
    (see `_cn_shuffle`) and what `cn_color_by` needs, since that colours every
    connector individually and has no groups to speak of.
    """
    connectors = resolve_connectors(neuron, settings)

    if connectors.empty:
        return

    cn_layout = resolve_cn_layout(settings)
    # Resolved to RGBA per row: a partial `cn_colors` dict leaves some types with
    # their default rgb tuple and others with whatever the user passed, and
    # matplotlib cannot build an array out of a mixed str/tuple sequence.
    colors, legend = connector_colors(connectors, cn_layout, color, settings)

    if settings.cn_legend:
        settings._cn_legend.update(legend)

    size = settings.cn_size if settings.cn_size else cn_layout["size"]
    alpha = settings.get("cn_alpha", None)
    flat = settings.method == "2d"
    zorder = settings.cn_zorder
    if zorder is None:
        # 2d: above everything, so a synapse is never hidden by its own neurite.
        # 3d: `zorder` does very little there - matplotlib sorts by depth instead.
        zorder = 1000 if flat else 0

    stalks = (
        _cn_stalks(neuron, connectors) if cn_layout.get("display") == "lines" else None
    )
    geometry = connectors[["x", "y", "z"]].values
    if stalks is not None:
        geometry, keep = stalks
        colors = colors[keep]

    # Same treatment either way: a stalk can bury its neighbour exactly as a
    # marker can, and the shuffle has to run before the view is applied so that
    # 2d and 3d end up with the same order
    order = _cn_shuffle(len(geometry), colors)
    if order is not None:
        geometry, colors = geometry[order], colors[order]

    if stalks is not None:
        x_ix, y_ix, _ = _view_axes(settings.view)
        artist = (LineCollection if flat else Line3DCollection)(
            geometry[:, :, [x_ix, y_ix]] if flat else geometry,
            colors=colors,
            # `cn_size` is a marker *area* everywhere else, so take its root to
            # get something that reads as about the same weight in line widths
            linewidths=np.sqrt(size),
            alpha=alpha,
            rasterized=settings.rasterize,
            zorder=zorder,
        )
        (ax.add_collection if flat else ax.add_collection3d)(artist)
    elif flat:
        x_ix, y_ix, _ = _view_axes(settings.view)
        artist = ax.scatter(
            geometry[:, x_ix], geometry[:, y_ix],
            color=colors, edgecolor="none", s=size, alpha=alpha, zorder=zorder,
        )
    else:
        artist = ax.scatter(
            geometry[:, 0], geometry[:, 1], geometry[:, 2],
            color=colors,
            s=size,
            depthshade=cn_layout.get("depthshade", False),
            alpha=alpha,
            zorder=zorder,
            edgecolor="none",
        )

    artist.set_gid(f"CN_{neuron.id}")


def _plot_mesh(neuron, color, ax, settings):
    """Plot mesh (i.e. Mesh)."""
    # Map vertex colors to faces (if need be)
    if isinstance(color, np.ndarray) and color.ndim == 2:
        if len(color) != len(neuron.faces) and len(color) == len(neuron.vertices):
            color = color[neuron.faces].mean(axis=1)[:, :4]

    ts = None
    if settings.method == "2d":
        _plot_surface(
            neuron.vertices,
            neuron.faces,
            color,
            ax,
            settings,
            label=getattr(neuron, "name"),
            zorder=MESH_ZORDER,
            depth_color=settings.depth_coloring,
            key=type(neuron).__name__,
        )
    else:
        ts = ax.plot_trisurf(
            neuron.vertices[:, 0],
            neuron.vertices[:, 1],
            neuron.faces,
            neuron.vertices[:, 2],
            label=getattr(neuron, "name"),
            rasterized=settings.rasterize,
            shade=bool(settings.mesh_shade),
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


def _plot_surface(
    vertices,
    faces,
    color,
    ax,
    settings,
    label=None,
    zorder=MESH_ZORDER,
    depth_color=False,
    depth_bins=True,
    halo=True,
    key=None,
):
    """Draw a triangle mesh into a 2d axes: culled, depth-sorted, maybe shaded.

    Back faces are dropped and the rest painted furthest-first, always - it costs
    less than drawing every face in mesh order and it is the difference between a
    flat blob and a surface with a front and a back.

    A mesh in one colour is filled as a single path rather than as one polygon per
    face. Under the nonzero winding rule the union of the front faces is filled
    exactly once, which is both an order of magnitude fewer paths and the only way
    a translucent mesh reads as translucent rather than as its own triangulation.
    Per-face colours (shading, `color_by`, depth coloring) rule that out, so those
    fall back to a `PolyCollection` - with antialiasing off, since neighbouring
    triangles would otherwise show a hairline of background between them.

    Parameters
    ----------
    color :         Colour, or one RGB(A) per *face*.
    label :         Legend entry; only the first artist gets it.
    zorder :        Base z-order, used when the faces are not put into depth bins.
    depth_color :   Colour faces by depth instead of by `color`.
    depth_bins :    Let `depth_sort` split the faces across the shared bins. Also
                    gates `depth_sort="global"`, which volumes stay out of too.
    halo :          Honour the `halo` setting.
    key :           Bucket name under `depth_sort="global"`, i.e. the neuron type.

    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    if not faces.size:
        return

    x_ix, y_ix, d_ix = _view_axes(settings.view)
    shade = _mesh_shade_spec(settings)

    tri, normals, depth, ix = mesh_faces(
        vertices,
        faces,
        (x_ix, y_ix),
        d_ix,
        front=_view_front(settings.view),
        smooth=shade is not None,
    )
    if not len(tri):
        return

    cmap = array = facecolors = None
    if depth_color:
        cmap = plt.get_cmap(settings.palette) if settings.palette else DEPTH_CMAP
        array = depth
    elif isinstance(color, np.ndarray) and color.ndim == 2:
        facecolors = color[ix]

    if shade is not None:
        if array is not None:
            # resolve the depth ramp to colours first, so shading multiplies into
            # it rather than replacing it
            facecolors = cmap(_norm_for(settings, array))
            cmap = array = None
        facecolors = _shade_faces(
            normals, color if facecolors is None else facecolors, shade, settings.view
        )

    halo = _halo_spec(settings, ax) if halo else None
    uniform = array is None and facecolors is None

    merge = _global_sort(settings, depth_bins)
    if merge is not None:
        resolved = cmap(_norm_for(settings, array)) if array is not None else (
            color if facecolors is None else facecolors
        )
        merge.add((key, "faces"), tri, depth, resolved)
        merge.label(label, resolved, "faces")
        return

    for z_under, z_over, sub in _depth_groups(depth, settings, zorder, depth_bins):
        sub_tri = tri if sub is None else tri[sub]
        path = _compound_path(sub_tri) if uniform or halo is not None else None

        if halo is not None:
            halo_width, halo_color = halo
            _outline_under(
                ax, path, halo_color, halo_width, z_under, settings.rasterize
            )

        z = z_over if halo is not None else z_under
        if uniform:
            artist = PathCollection(
                [path],
                facecolors=color,
                edgecolors="none",
                linewidths=0,
                rasterized=settings.rasterize,
                zorder=z,
                label=label,
            )
        else:
            artist = PolyCollection(
                sub_tri,
                cmap=cmap,
                norm=settings.norm if cmap is not None else None,
                edgecolors="none",
                linewidth=0,
                antialiased=False,
                rasterized=settings.rasterize,
                zorder=z,
                label=label,
            )
            if array is not None:
                artist.set_array(array if sub is None else array[sub])
                artist.set_alpha(
                    settings.alpha if isinstance(settings.alpha, float) else None
                )
            else:
                artist.set_facecolor(facecolors if sub is None else facecolors[sub])

        ax.add_collection(artist)
        label = None  # one legend entry per neuron, not one per bin


def _get_depth_axis(view):
    """Return index of axis which is not used for x/y."""
    view = [v.replace("-", "").replace("+", "") for v in view]
    depth = [ax for ax in ["x", "y", "z"] if ax not in view][0]
    map = {"x": 0, "y": 1, "z": 2}
    return map[depth]


def _view_axes(view):
    """Return coordinate indices for (x, y, depth) of a 2d view."""
    map = {"x": 0, "y": 1, "z": 2}
    xy = [map[v.replace("-", "").replace("+", "")] for v in view]
    return xy[0], xy[1], _get_depth_axis(view)


def _view_frame(view):
    """`(u, v, w)`: screen right, screen up and out-of-the-screen, in data space.

    Coordinates are never flipped - a `"-z"` in the view inverts the axis instead -
    so which way is "up" or "towards the viewer" has to be worked out from the view
    rather than read off the data. All three come out as signed unit axes.
    """
    map = {"x": 0, "y": 1, "z": 2}
    u, v = np.zeros(3), np.zeros(3)
    for e, spec in ((u, view[0]), (v, view[1])):
        e[map[spec.replace("-", "").replace("+", "")]] = (
            -1 if spec.startswith("-") else 1
        )
    return u, v, np.cross(u, v)


def _view_front(view):
    """Sign along the depth axis that points at the viewer."""
    _, _, w = _view_frame(view)
    return int(np.sign(w[_get_depth_axis(view)])) or 1


def _mesh_shade_spec(settings):
    """`dict(mode, light, ambient, strength)` for `mesh_shade`, or None if flat."""
    shade = settings.mesh_shade
    if not shade:
        return None

    mode, light, ambient, strength = "lambert", MESH_LIGHT, MESH_AMBIENT, 1.0
    if isinstance(shade, dict):
        mode = shade.get("mode", mode)
        light = shade.get("light", light)
        ambient = shade.get("ambient", ambient)
        strength = shade.get("strength", strength)
    elif not isinstance(shade, bool):
        mode = shade

    mode = str(mode).lower()
    if mode not in MESH_SHADE_MODES:
        raise ValueError(
            f'Unknown mesh_shade mode "{mode}". Available modes: '
            f"{', '.join(MESH_SHADE_MODES)}"
        )

    return dict(mode=mode, light=light, ambient=float(ambient),
                strength=float(strength))


def _shade_ramp(color, shade):
    """Map a 0-1 shade onto a dark -> `color` -> light ramp.

    A shade of 0.5 is the colour that was asked for, so a mid-lit neuron still
    matches its legend entry. `color` may be one colour or one per face, which is
    what lets shading multiply into `color_by` rather than replace it.
    """
    rgba = mcl.to_rgba_array(color)
    base, alpha = rgba[:, :3], rgba[:, 3:]
    dark = base * 0.38
    light = base + (1 - base) * 0.5

    s = np.clip(shade, 0, 1)[:, None]
    out = np.where(
        s < 0.5,
        dark + (base - dark) * (s / 0.5),
        base + (light - base) * ((s - 0.5) / 0.5),
    )
    return np.clip(np.hstack([out, np.broadcast_to(alpha, (len(out), 1))]), 0, 1)


def _shade_faces(normals, color, spec, view):
    """One RGBA per face for a `mesh_shade` mode.

    `color` may be a single colour or one per face - shading multiplies into it
    either way, so `color_by`, depth coloring and palettes all survive it.
    """
    mode = spec["mode"]
    u, v, w = _view_frame(view)

    if mode == "ghost":
        # opaque where the surface turns away from the viewer and clear where it
        # faces them, which is what keeps a neuron inside a neuropil visible
        rgba = np.broadcast_to(mcl.to_rgba_array(color), (len(normals), 4)).copy()
        rgba[:, 3] *= np.clip(
            (1 - np.abs(normals @ w)) ** 2 * spec["strength"], 0.02, 1
        )
        return rgba

    # the light direction is given in view space, so put it back on the data axes
    light = np.asarray(spec["light"], dtype=float)
    light = light[0] * u + light[1] * v + light[2] * w
    light = light / (np.linalg.norm(light) or 1)

    ambient = spec["ambient"]
    shade = np.clip(normals @ light, 0, 1) * (1 - ambient) + ambient

    if mode == "cel":
        shade = np.floor(np.clip(shade, 0, 0.999) * CEL_BANDS) / (CEL_BANDS - 1)
    elif mode == "rim":
        shade = shade + 0.55 * spec["strength"] * (1 - np.abs(normals @ w)) ** 3

    return _shade_ramp(color, shade)


def _apply_style(kwargs):
    """Fill `kwargs` with the defaults of `kwargs["style"]`.

    Anything the caller passed themselves wins, including under an alias - so
    `style="publication", radius=False` really does turn the radius off.
    """
    name = kwargs.get("style")
    if name is None:
        return kwargs

    if name not in PLOT_STYLES:
        raise ValueError(
            f'Unknown style "{name}". Available styles: '
            f"{', '.join(sorted(PLOT_STYLES))}"
        )

    kwargs = dict(kwargs)
    for key, value in PLOT_STYLES[name].items():
        if not any(alias in kwargs for alias in _STYLE_ALIASES.get(key, (key,))):
            kwargs[key] = value
    return kwargs


def _background_color(ax):
    """Colour to draw halos in: whatever the neuron is sitting on."""
    for color in (ax.get_facecolor(), ax.get_figure().get_facecolor()):
        # a fully transparent patch tells us nothing about what will show through
        if mcl.to_rgba(color)[3] > 0:
            return color
    return "w"


def _halo_spec(settings, ax):
    """`(width, color)` for the halo stroke, or None if there is no halo.

    Note that the halo has to be drawn as its own artist *underneath* the neuron
    rather than as a `patheffects.withStroke`: path effects are applied per path,
    so within a single collection each segment's halo would erase the segments
    drawn before it and the neuron would come out dashed.
    """
    halo = settings.halo
    if not halo:
        return None

    width, color = HALO_WIDTH, None
    if isinstance(halo, dict):
        width = halo.get("width", halo.get("lw", HALO_WIDTH))
        color = halo.get("color", halo.get("c", None))
    elif not isinstance(halo, bool):
        width = float(halo)

    return width, _background_color(ax) if color is None else color


def _depth_sort_mode(settings):
    """`None`, `"bins"` or `"global"` for the `depth_sort` setting."""
    value = settings.depth_sort
    if isinstance(value, str):
        if value != DEPTH_SORT_GLOBAL:
            raise ValueError(
                f'Unknown depth_sort "{value}". Use False, True, an integer '
                f'number of bins, or "{DEPTH_SORT_GLOBAL}".'
            )
        return DEPTH_SORT_GLOBAL
    return "bins" if value else None


def _prepare_depth_bins(neurons, settings):
    """Work out global depth bin edges for `depth_sort`.

    They have to be global: the whole point is that neurons interleave, which
    only works if the same depth maps to the same z-order for all of them.
    """
    if _depth_sort_mode(settings) != "bins" or settings.method != "2d" or neurons.empty:
        return

    if isinstance(settings.depth_sort, bool):
        n_bins, flip = DEPTH_BINS, False
    else:
        n_bins, flip = int(abs(settings.depth_sort)), settings.depth_sort < 0

    if n_bins < 2:
        return

    # bins are indexed along the raw depth axis, which for half the views runs away
    # from the viewer rather than towards them - `_view_front` is what tells them
    # apart, and a negative `depth_sort` still flips whatever it works out
    reverse = (_view_front(settings.view) < 0) != flip

    lo, hi = neurons.bbox[_get_depth_axis(settings.view)]
    if not np.isfinite([lo, hi]).all() or lo == hi:
        return

    settings._depth_edges = np.linspace(lo, hi, n_bins + 1)
    settings._depth_reverse = reverse


def _neuron_slot(settings, base):
    """`(z, step)` for the neuron being drawn, inside `[base, base + 1)`.

    Without depth bins every neuron would otherwise land on `base` itself, and a
    halo sharing a z-order with the neuron in front of it is background colour
    under everything - i.e. invisible. Giving each neuron its own slot makes input
    order the stacking order, which is what one-artist-per-neuron should mean.
    """
    step = getattr(settings, "_z_step", 1.0)
    return base + getattr(settings, "_z_index", 0) * step, step


def _depth_groups(depth, settings, base=1, bins=True):
    """Yield `(z_under, z_over, indices)` for each depth bin, back to front.

    `z_under` is for the halo, `z_over` for the artist itself: the halo has to sit
    below its own neuron but above everything further away.

    Without `depth_sort` this yields a single group of everything, with `None` for
    the indices and the neuron's own slot for its z-order. `bins=False` - which is
    how volumes stay out of the stack - pins it to `base` instead.
    """
    edges = getattr(settings, "_depth_edges", None) if bins else None
    if edges is None:
        z, step = _neuron_slot(settings, base) if bins else (base, 1.0)
        yield z, z + step / 2, None
        return

    n_bins = len(edges) - 1
    step = 1 / n_bins
    which = np.clip(np.digitize(depth, edges[1:-1]), 0, n_bins - 1)
    order = range(n_bins - 1, -1, -1) if settings._depth_reverse else range(n_bins)
    for rank, b in enumerate(order):
        ix = np.flatnonzero(which == b)
        if len(ix):
            # keep z-orders inside [1, 2) so somata (4) and connectors stay on top
            z = 1 + rank * step
            yield z, z + step / 2, ix


class _GlobalSort:
    """Geometry collected across neurons for `depth_sort="global"`.

    Bins approximate a depth sort by quantising it; this is the exact version.
    Everything drawn the same way - all skeleton lines, all mesh faces - goes
    into one artist and is sorted together, so two neurons interleave face by
    face rather than bin by bin.

    What it costs is the *aggregate* artists that make 2d plotting fast. A flat
    mesh is normally one compound path and a skeleton a handful of polylines;
    sorting across neurons forces both down to their elements, which is why this
    is opt-in and `depth_sort=<int>` remains the cheap approximation.

    Buckets come out in the order their neuron type first appeared in the input.
    Colours are resolved to RGBA on the way in - a merged artist has no room for
    a colormap that only made sense for one neuron - and per-neuron legend
    entries survive as proxy handles, since one artist carries one label.
    """

    def __init__(self):
        self.buckets = {}
        self.proxies = []

    def add(self, key, paths, depth, colors, widths=None):
        """Stash one neuron's worth of elements under `key = (type, kind)`."""
        bucket = self.buckets.setdefault(
            key, dict(paths=[], depth=[], colors=[], widths=[])
        )
        depth = np.asarray(depth, dtype=float)
        # one colour for the whole neuron still has to arrive per element, since
        # it is about to be interleaved with everyone else's
        rgba = mcl.to_rgba_array(colors, None)
        if len(rgba) == 1 and len(depth) != 1:
            rgba = np.repeat(rgba, len(depth), axis=0)

        bucket["paths"].append(paths)
        bucket["depth"].append(depth)
        bucket["colors"].append(rgba)
        if widths is not None:
            bucket["widths"].append(np.broadcast_to(
                np.asarray(widths, dtype=float), depth.shape
            ).copy())

    def label(self, label, colors, kind):
        """Remember a legend entry to re-create once the buckets are merged.

        `colors` is whatever the neuron was drawn in; a per-element array is
        averaged, since a swatch can only show one colour.
        """
        if label is None:
            return
        rgba = mcl.to_rgba_array(colors, None)
        self.proxies.append((label, tuple(rgba.mean(axis=0)), kind))

    def flush(self, ax, settings):
        """Emit one artist per bucket, back to front, and the legend proxies."""
        common = dict(
            linestyle=settings.linestyle,
            rasterized=settings.rasterize,
            joinstyle="round",
            capstyle="round",
        )
        # depth is the raw coordinate, which for half the views runs away from the
        # viewer rather than towards them - same correction the bins make
        front = _view_front(settings.view)
        # keep z-orders inside [1, 2), as the bins do, so that somata (4) and
        # connectors stay on top of every kind
        step = 1 / max(len(self.buckets), 1)
        for rank, ((_, kind), bucket) in enumerate(self.buckets.items()):
            paths = _concat_paths(bucket["paths"])
            order = np.argsort(np.concatenate(bucket["depth"]) * front)
            colors = np.concatenate(bucket["colors"])[order]
            sub = paths[order] if isinstance(paths, np.ndarray) else [paths[i] for i in order]
            zorder = 1 + rank * step

            if kind in ("lines", "radius_lines"):
                widths = np.concatenate(bucket["widths"])[order]
                if kind == "radius_lines":
                    artist = RadiusLineCollection(
                        sub, radii=widths, colors=colors, zorder=zorder, **common
                    )
                else:
                    artist = LineCollection(
                        sub, linewidths=widths, colors=colors, zorder=zorder, **common
                    )
            else:
                artist = PolyCollection(
                    sub,
                    facecolors=colors,
                    edgecolors="none",
                    linewidth=0,
                    # neighbouring mesh faces would otherwise show a hairline of
                    # background between them; ribbon capsules overlap and don't
                    antialiased=kind != "faces",
                    rasterized=settings.rasterize,
                    zorder=zorder,
                )
            ax.add_collection(artist)

        for label, color, kind in self.proxies:
            # empty data, so these carry a legend entry without drawing anything
            # or pulling on the axis limits
            ax.add_line(
                mlines.Line2D(
                    [], [], color=color, label=label,
                    **(dict(lw=4) if kind in ("lines", "radius_lines")
                       else dict(marker="s", ls="none", ms=8)),
                )
            )


def _concat_paths(chunks):
    """Join per-neuron path chunks, keeping an array an array where possible."""
    if all(isinstance(c, np.ndarray) and c.ndim == 3 for c in chunks):
        if len({c.shape[1] for c in chunks}) == 1:
            return np.concatenate(chunks)
    return [p for chunk in chunks for p in chunk]


def _global_sort(settings, active=True):
    """The `_GlobalSort` accumulator, or None if this artist is not part of it."""
    return getattr(settings, "_global_sort", None) if active else None


def _norm_for(settings, values):
    """Normalise `values` against the shared depth normaliser, or their own range.

    A merged artist cannot carry a colormap, so depth coloring has to be resolved
    to RGBA up front - which means normalising here rather than in the artist.
    """
    norm = settings.norm or plt.Normalize(np.min(values), np.max(values))
    return norm(values)


def _taper_widths(neuron, settings):
    """Per-node line widths from a topological measure.

    Skeletons without a trustworthy `radius` column can still be tapered: both
    measures shrink monotonically towards the tips, so the result reads like a
    tree rather than a wireframe. Widths are relative to `linewidth`.
    """
    kind = settings.taper
    node_ids = neuron.nodes.node_id.values
    parent_ids = neuron.nodes.parent_id.values

    if kind == "strahler":
        # already computed? then respect whatever options it was computed with
        if "strahler_index" in neuron.nodes.columns:
            value = neuron.nodes.strahler_index.values.astype(float)
        else:
            value = utils.fastcore.strahler_index(node_ids, parent_ids).astype(float)
    elif kind == "subtree":
        # cable below each node, which grows smoothly rather than in steps
        value = utils.fastcore.subtree_height(node_ids, parent_ids).astype(float)
        # heavily skewed towards the root, so pull it back towards linear
        value = value**0.35
    else:
        raise ValueError(
            f'Unknown taper "{kind}". Use either "strahler" or "subtree".'
        )

    top = value.max()
    frac = value / top if top > 0 else np.ones(len(value))
    return settings.linewidth * (TAPER_RANGE[0] + np.diff(TAPER_RANGE)[0] * frac)


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


def _colors_are_categorical(colors, frac=0.25, floor=20):
    """Guess whether per-node colors are categorical.

    Grouping a neuron's edges into contiguous same-color lines (see
    `_collapse_colored_segments`) only pays off when many neighbouring nodes
    share a color, i.e. the data is categorical (few distinct colors). For
    (quasi-)continuous colors almost every node differs, so grouping would
    achieve nothing and be slower than the vectorised per-edge path. We use
    the number of distinct colors as a cheap proxy to decide.
    """
    n_unique = len(np.unique(colors, axis=0))
    return n_unique <= max(floor, int(len(colors) * frac))


def _collapse_colored_segments(neuron, node_colors):
    """Merge a neuron's edges into contiguous same-color polylines.

    Instead of drawing every child->parent edge as its own 2-point segment
    (which looks jagged and bloats vector exports), we walk each linear stretch
    ("slab") of the neuron and merge consecutive edges that share a color into a
    single multi-point line. Adjacent runs share their boundary vertex so the
    lines still join up without gaps.

    Parameters
    ----------
    neuron :        Skeleton
    node_colors :   (N, 4) array
                    One RGBA color for each node in `neuron.nodes` (row order).

    Returns
    -------
    lines :         list of (M, 3) arrays
                    Contiguous polylines in xyz coordinates.
    colors :        (n_lines, 4) array
                    One RGBA color per polyline.

    """
    coords, seg_cols = segments_to_coords(neuron, node_colors=node_colors)

    lines = []
    colors = []
    for seg, cols in zip(coords, seg_cols):
        if len(seg) < 2:
            continue
        cols = np.asarray(cols)
        # Color each edge by its first node, then find where the color changes
        # along this slab. Segments run tip->root, so the first node of an edge
        # is its child - matching the per-edge coloring used elsewhere.
        edge_cols = cols[:-1]
        changes = np.any(edge_cols[1:] != edge_cols[:-1], axis=1)
        breaks = np.flatnonzero(changes) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(edge_cols)]))
        for s, e in zip(starts, ends):
            # Run spans edges s..e-1, i.e. vertices s..e. Sharing vertex `e`
            # with the next run keeps consecutive lines connected.
            lines.append(seg[s : e + 1])
            colors.append(edge_cols[s])

    colors = np.array(colors) if colors else np.zeros((0, node_colors.shape[1]))
    return lines, colors


def _plot_skeleton_2d(neuron, color, ax, settings, radius=False):
    """Plot a skeleton's neurites for `method="2d"`.

    Three renderers, picked in this order:

    1. `radius` - one filled outline at the neuron's real radius
    2. anything that needs per-path z-order, width or colour - a LineCollection
    3. the plain single-colour case - one Line2D, which stays the fast path

    """
    label = f'{getattr(neuron, "name", "NA")} - #{neuron.id}'
    per_node_color = isinstance(color, np.ndarray) and color.ndim == 2

    if radius and settings.radius != "lw":
        _plot_ribbon(neuron, color, ax, settings, label)
        return

    widths = None
    radius_lw = False
    if settings.taper:
        widths = _taper_widths(neuron, settings)
    elif radius:
        # `radius="lw"`: same idea as the ribbon, but the width is converted from
        # data units at draw time. See `RadiusLineCollection`. Note `use_radius`
        # is False if there are no radii to work with, in which case we fall
        # through to plain lines.
        widths = (
            neuron.nodes.radius.fillna(0).values.astype(float) * settings.linewidth
        )
        radius_lw = True

    # Merging contiguous same-color edges into polylines only works if we have
    # exactly one color per node, and only pays off for categorical colors
    can_collapse = (
        per_node_color
        and color.shape[0] == neuron.nodes.shape[0]
        and _colors_are_categorical(color)
    )

    # Anything that varies along the neuron needs per-edge segments; anything that
    # only varies between neurons can keep whole slabs and their proper joins.
    per_edge = (
        settings.depth_coloring
        or widths is not None
        or (per_node_color and not can_collapse)
    )

    fancy = per_edge or per_node_color or settings.depth_sort or settings.halo
    if not fancy:
        # Nothing to vary within the neuron: one Line2D with NaNs in between is
        # by far the cheapest way to draw it.
        coords = segments_to_coords(neuron, modifier=(1, 1, 1))
        # We have to add (None, None, None) to the end of each
        # slab to make that line discontinuous there
        coords = np.vstack([np.append(t, [[None] * 3], axis=0) for t in coords])

        x, y = _parse_view2d(coords, settings.view)
        ax.add_line(
            mlines.Line2D(
                x,
                y,
                lw=settings.linewidth,
                ls=settings.linestyle,
                color=color,
                rasterized=settings.rasterize,
                label=label,
            )
        )
        return

    array = None
    cmap = None
    if per_edge:
        # kept as the (n_edges, 2, 3) array it comes as - `_add_lines` projects
        # it in one slice, where a list of one-edge arrays would be a Python loop
        paths = coords = tn_pairs_to_coords(neuron, modifier=(1, 1, 1))
        line_colors = None
        if settings.depth_coloring:
            cmap = (
                plt.get_cmap(settings.palette)
                if isinstance(settings.palette, str)
                else DEPTH_CMAP
            )
            # Colour each edge by its child node, as we always have
            array = coords[:, 0, _get_depth_axis(settings.view)]
        elif per_node_color:
            # If we have a color for each node, we need to drop the roots
            line_colors = (
                color[neuron.nodes.parent_id.values >= 0]
                if color.shape[0] != coords.shape[0]
                else color
            )
        if widths is not None:
            # per-node -> per-edge, again taking the child's value
            widths = widths[neuron.nodes.parent_id.values >= 0]
    elif per_node_color:
        # Categorical per-node colors (e.g. `color_by='strahler_index'`):
        # merge contiguous same-color edges into continuous polylines. This
        # gives proper line joins (instead of a pile of short segments) and
        # far fewer artists, which keeps vector exports small.
        paths, line_colors = _collapse_colored_segments(neuron, color)
    else:
        paths = segments_to_coords(neuron, modifier=(1, 1, 1))
        line_colors = None

    _add_lines(
        ax,
        paths,
        settings,
        label=label,
        color=color if line_colors is None else None,
        colors=line_colors,
        widths=widths,
        array=array,
        cmap=cmap,
        scale_to_radius=radius_lw,
        key=type(neuron).__name__,
    )


def _add_lines(
    ax,
    paths,
    settings,
    label=None,
    color=None,
    colors=None,
    widths=None,
    array=None,
    cmap=None,
    scale_to_radius=False,
    key=None,
):
    """Add one neuron's lines, honouring `halo` and `depth_sort`.

    `paths` are xyz because we need the depth axis to bin them, not just the two
    axes we are drawing. They come either as one `(n, 2, 3)` array of per-edge
    segments or as a ragged list of polylines - the former is worth keeping whole,
    since a Python loop over one-edge arrays costs more than the projection does.
    """
    x_ix, y_ix, d_ix = _view_axes(settings.view)
    if isinstance(paths, np.ndarray) and paths.ndim == 3:
        xy = paths[:, :, [x_ix, y_ix]]
        depth = paths[:, :, d_ix].mean(axis=1)
    else:
        paths = [np.asarray(p, dtype=float) for p in paths]
        xy = [p[:, [x_ix, y_ix]] for p in paths]
        depth = np.array([p[:, d_ix].mean() for p in paths])
    if widths is None:
        widths = np.full(len(xy), float(settings.linewidth))
    halo = _halo_spec(settings, ax)

    merge = _global_sort(settings)
    if merge is not None:
        kind = "radius_lines" if scale_to_radius else "lines"
        resolved = (
            cmap(_norm_for(settings, array)) if array is not None
            else colors if colors is not None
            else color
        )
        merge.add((key, kind), xy, depth, resolved, widths)
        merge.label(label, resolved, kind)
        return

    common = dict(
        linestyle=settings.linestyle,
        rasterized=settings.rasterize,
        joinstyle="round",
        # Round caps make the individual node-to-node segments blend
        # into contiguous-looking lines instead of leaving notches at
        # every joint (`joinstyle` has no effect on 2-point segments).
        capstyle="round",
    )

    def build(sub, sub_widths, margin=0.0, **kwargs):
        # With `radius="lw"` the width is a radius in data units, which only the
        # collection itself can turn into points - see `RadiusLineCollection`.
        if scale_to_radius:
            return RadiusLineCollection(
                sub, radii=sub_widths, margin=margin, **common, **kwargs
            )
        return LineCollection(sub, linewidths=sub_widths + margin, **common, **kwargs)

    for z_under, z_over, ix in _depth_groups(depth, settings):
        sub = xy if ix is None else (
            xy[ix] if isinstance(xy, np.ndarray) else [xy[i] for i in ix]
        )
        sub_widths = widths if ix is None else widths[ix]

        if halo is not None:
            halo_width, halo_color = halo
            ax.add_collection(
                build(sub, sub_widths, margin=halo_width, colors=halo_color,
                      zorder=z_under)
            )

        lc = build(
            sub,
            sub_widths,
            cmap=cmap,
            norm=settings.norm if cmap is not None else None,
            zorder=z_over if halo is not None else z_under,
        )
        if label is not None:
            lc.set_label(label)
            label = None  # only the first group carries the legend entry

        if array is not None:
            lc.set_array(array if ix is None else array[ix])
        elif colors is not None:
            lc.set_color(colors if ix is None else np.asarray(colors)[ix])
        elif color is not None:
            lc.set_color(color)

        ax.add_collection(lc)


class RadiusLineCollection(LineCollection):
    """Lines whose width tracks a radius given in *data* units.

    This is what `radius="lw"` draws: one stroked path per edge instead of a
    polygon per edge plus a disc per node, which makes for a much smaller vector
    file and gives us exact round joins and caps for free.

    The catch is that matplotlib line widths are in points, so the conversion
    depends on the current axes scale. Doing it in `draw` rather than off a
    `draw_event` is what makes it correct: `draw_event` fires *after* the artist
    has been rendered, so the first `savefig` would go out with whatever the
    scale happened to be before the axes were autoscaled.
    """

    def __init__(self, *args, radii=None, margin=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._radii = np.asarray(radii, dtype=float)
        # constant width added on top, in points - this is how the halo keeps a
        # fixed margin around a neurite whose own width changes with the zoom
        self._margin = float(margin)
        self._scale = None

    def draw(self, renderer):
        if self.axes is not None and self._radii.size:
            trans = self.axes.transData
            p0, p1 = trans.transform((0, 0)), trans.transform((1, 0))
            points_per_unit = abs(p1[0] - p0[0]) * 72.0 / self.figure.dpi
            # `set_linewidth` rebuilds the dash list for every segment, which at
            # 100k edges is not free - and most draws are not zooms
            if points_per_unit != self._scale:
                self._scale = points_per_unit
                self.set_linewidth(2 * self._radii * points_per_unit + self._margin)
        super().draw(renderer)


def _compound_path(polys):
    """Merge polygons into a single path, so their union is filled just once.

    A `PolyCollection` composites each polygon separately, which is invisible
    while they are opaque but turns a translucent neuron into a chain of visible
    discs and quads. One path with many subpaths is filled in a single operation
    under the nonzero winding rule, so overlaps stay flat - which only holds if
    every subpath is wound the same way (`skeleton_capsules` takes care of that;
    wound against each other they cancel and leave a hole).
    """
    if isinstance(polys, np.ndarray) and polys.ndim == 3:
        # fast path: every subpath the same length (mesh triangles), so none of
        # the bookkeeping below is needed. The general branch handles this too.
        n, k, _ = polys.shape
        verts = np.empty((n * (k + 1), 2), dtype=float)
        view = verts.reshape(n, k + 1, 2)
        view[:, :k] = polys
        view[:, k] = polys[:, 0]
        codes = np.full(len(verts), mpath.Path.LINETO, dtype=mpath.Path.code_type)
        codes[:: k + 1] = mpath.Path.MOVETO
        return mpath.Path(verts, codes)

    # Built by scattering rather than by concatenating per polygon: at ~9k
    # polygons for a mid-sized neuron, a `vstack` per polygon costs more than
    # everything else in the renderer put together.
    lengths = np.fromiter((len(p) for p in polys), dtype=int, count=len(polys))
    src = np.concatenate(polys)

    # each subpath is its own vertices plus a repeat of the first, which closes
    # it without needing a CLOSEPOLY vertex
    sizes = lengths + 1
    ends = np.cumsum(sizes)
    starts = ends - sizes

    verts = np.empty((int(ends[-1]), 2), dtype=float)
    is_repeat = np.zeros(len(verts), dtype=bool)
    is_repeat[ends - 1] = True
    verts[~is_repeat] = src
    verts[is_repeat] = src[np.cumsum(lengths) - lengths]

    codes = np.full(len(verts), mpath.Path.LINETO, dtype=mpath.Path.code_type)
    codes[starts] = mpath.Path.MOVETO

    return mpath.Path(verts, codes)


def _outline_under(ax, path, color, width, zorder, rasterized=False):
    """Stroke a compound path *behind* its own fill, to outline the union.

    Stroking a compound path strokes every subpath, so drawn on top of a mesh you
    would get a wireframe of all its triangles. Underneath, the only part of the
    stroke that survives is the half sticking out past the fill - which is exactly
    the outline of the union. Same trick as the skeleton halo, for the same reason.

    `width` is the stroke width, so the margin it leaves is half of it - matching
    what `halo` means everywhere else (`_add_lines` adds it to the line width, so
    that too shows half on each side).
    """
    ax.add_collection(
        PathCollection(
            [path],
            facecolors=color,
            edgecolors=color,
            linewidths=width,
            joinstyle="round",
            capstyle="round",
            rasterized=rasterized,
            zorder=zorder,
        )
    )


def _plot_ribbon(neuron, color, ax, settings, label):
    """Plot a skeleton's neurites at their real radius, as filled outlines."""
    x_ix, y_ix, d_ix = _view_axes(settings.view)
    per_node_color = isinstance(color, np.ndarray) and color.ndim == 2

    # depth coloring throws per-node colours away again, so don't carry them
    carry = per_node_color and not settings.depth_coloring
    polys, depth, poly_colors = skeleton_capsules(
        neuron,
        (x_ix, y_ix),
        d_ix,
        radius_scale=settings.linewidth,
        node_values=color if carry else None,
    )

    cmap = None
    array = None
    if settings.depth_coloring:
        cmap = (
            plt.get_cmap(settings.palette)
            if isinstance(settings.palette, str)
            else DEPTH_CMAP
        )
        array = depth
        poly_colors = None

    halo = _halo_spec(settings, ax)
    # One colour for the whole neuron means we can fill it as a single path,
    # which keeps a translucent neuron from showing every capsule it is made of
    uniform = array is None and poly_colors is None

    merge = _global_sort(settings)
    if merge is not None:
        resolved = cmap(_norm_for(settings, array)) if array is not None else (
            color if poly_colors is None else poly_colors
        )
        merge.add((type(neuron).__name__, "polys"), polys, depth, resolved)
        merge.label(label, resolved, "polys")
        return

    for z_under, z_over, ix in _depth_groups(depth, settings):
        sub = polys if ix is None else [polys[i] for i in ix]
        path = _compound_path(sub) if uniform or halo is not None else None

        if halo is not None:
            halo_width, halo_color = halo
            _outline_under(
                ax, path, halo_color, halo_width, z_under, settings.rasterize
            )

        zorder = z_over if halo is not None else z_under
        if uniform:
            artist = PathCollection(
                [path],
                facecolors=color,
                edgecolors="none",
                rasterized=settings.rasterize,
                zorder=zorder,
            )
        else:
            artist = PolyCollection(
                sub,
                cmap=cmap,
                norm=settings.norm if cmap is not None else None,
                edgecolors="none",
                rasterized=settings.rasterize,
                zorder=zorder,
            )
            if array is not None:
                artist.set_array(array if ix is None else array[ix])
            else:
                artist.set_facecolor(poly_colors if ix is None else poly_colors[ix])

        if label is not None:
            artist.set_label(label)
            label = None

        ax.add_collection(artist)


def _plot_skeleton(neuron, color, ax, settings, radius=False):
    """Plot skeleton."""

    if settings.method == "2d":
        _plot_skeleton_2d(neuron, color, ax, settings, radius=radius)

        for soma in resolve_somata(neuron, color, settings):
            soma_color = soma.color
            if settings.depth_coloring:
                d = soma.center[_get_depth_axis(settings.view)]
                soma_color = DEPTH_CMAP(settings.norm(d))

            soma_defaults = dict(
                radius=soma.radius,
                fill=True,
                fc=soma_color,
                rasterized=settings.rasterize,
                zorder=4,
                edgecolor="none",
            )
            if isinstance(settings.soma, dict):
                soma_defaults.update(settings.soma)

            sx, sy = _parse_view2d(soma.center.reshape(1, 3), settings.view)
            ax.add_patch(mpatches.Circle((sx[0], sy[0]), **soma_defaults))
        return None, None

    elif settings.method in ["3d", "3d_complex"]:
        # For simple scenes, add whole neurons at a time to speed up rendering
        if settings.method == "3d":
            if (
                not settings.depth_coloring
                and isinstance(color, np.ndarray)
                and color.ndim == 2
                and color.shape[0] == neuron.nodes.shape[0]
                and _colors_are_categorical(color)
            ):
                # Categorical per-node colors: merge contiguous same-color edges
                # into continuous polylines (see 2D path for rationale).
                coords, line_color = _collapse_colored_segments(neuron, color)
            elif (
                isinstance(color, np.ndarray) and color.ndim == 2
            ) or settings.depth_coloring:
                coords = tn_pairs_to_coords(neuron, modifier=(1, 1, 1))
                # If we have a color for each node, we need to drop the roots
                if isinstance(color, np.ndarray) and color.shape[0] != coords.shape[0]:
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
                # Round caps avoid notches where per-node/depth-coloured
                # segments meet (see 2D path for details).
                capstyle="round",
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
        for soma in resolve_somata(neuron, color, settings):
            resolution = 20
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            r = soma.radius
            x = r * np.outer(np.cos(u), np.sin(v)) + soma.center[0]
            y = r * np.outer(np.sin(u), np.sin(v)) + soma.center[1]
            z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + soma.center[2]

            soma_defaults = dict(
                color=soma.color,
                shade=bool(settings.mesh_shade),
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
            # A neuropil is nearly always translucent, which is precisely the case
            # a per-face collection gets wrong - hence the same culled, sorted,
            # single-path treatment neurons get. Volumes stay out of the depth bins
            # and never take a halo: they are scenery, and belong behind everything.
            _plot_surface(
                np.asarray(volume.vertices),
                np.asarray(volume.faces),
                mcl.to_rgba(color, this_alpha),
                ax,
                settings,
                label=name,
                zorder=0,
                depth_bins=False,
                halo=False,
            )

        if settings.volume_outlines in (True, "both"):
            verts = volume.to_2d(
                view=settings.view,
                alpha=settings.get("volume_outlines_alpha", 0.001),
            )
            vpatch = mpatches.Polygon(
                verts,
                closed=True,
                lw=lw,
                fill=fill,
                rasterized=settings.rasterize,
                fc=fc,
                ec=ec,
                zorder=0,
                # A volume's alpha is a *fill* alpha - it exists so you can see the
                # neuron through the neuropil. There is nothing to see through a
                # contour, and at the 10-20% volumes default to it would be all but
                # invisible, so the contour is always drawn opaque.
                alpha=1,
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
            if hasattr(c, "_vec"):
                points.append(c._vec[:3, :].T)
            elif hasattr(c, "_faces"):
                points.append(c._faces.reshape(-1, 3))
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


def style_ax3d(ax):
    """Customize 3d axes."""
    ax.set_axis_off()

    # Trigger one rendering cycle to get the correct bounds
    ax.figure.canvas.draw()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    xmin, xmax = sorted(ax.get_xlim())
    ymin, ymax = sorted(ax.get_ylim())
    zmin, zmax = sorted(ax.get_zlim())

    x_interval = (xmax - xmin) / 10
    y_interval = (ymax - ymin) / 10
    z_interval = (zmax - zmin) / 10

    # Round to the nearest order of magnitude
    x_interval = np.floor(x_interval / 10 ** np.floor(np.log10(x_interval))) * 10 ** np.floor(np.log10(x_interval))
    y_interval = np.floor(y_interval / 10 ** np.floor(np.log10(y_interval))) * 10 ** np.floor(np.log10(y_interval))
    z_interval = np.floor(z_interval / 10 ** np.floor(np.log10(z_interval))) * 10 ** np.floor(np.log10(z_interval))

    # Use the largest interval
    interval = max(x_interval, y_interval, z_interval)

    xmin = np.floor(xmin / interval) * interval
    xmax = np.ceil(xmax / interval) * interval
    ymin = np.floor(ymin / interval) * interval
    ymax = np.ceil(ymax / interval) * interval
    zmin = np.floor(zmin / interval) * interval

    for d in np.arange(0, 11):
        color, linewidth, zorder = "0.75", 0.5, -100
        if d in [0, 5, 10]:
            color, linewidth, zorder = "0.5", 0.75, -50

        dx = xmin + d * interval
        dy = ymin + d * interval
        dz = zmin + d * interval

        ax.plot([xmin, xmin], [dy, dy], [zmin, zmax], linewidth=linewidth, color=color, zorder=zorder)
        ax.plot([xmin, xmin], [ymin, ymax], [dz, dz], linewidth=linewidth, color=color, zorder=zorder)
        ax.plot([xmin, xmax], [ymin, ymin], [dz, dz], linewidth=linewidth, color=color, zorder=zorder)
        ax.plot([dx, dx], [ymin, ymax], [zmin, zmin], linewidth=linewidth, color=color, zorder=zorder)
        ax.plot([xmin, xmax], [dy, dy], [zmin, zmin], linewidth=linewidth, color=color, zorder=zorder)
        ax.plot([dx, dx], [ymin, ymin], [zmin, zmax], linewidth=linewidth, color=color, zorder=zorder)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
