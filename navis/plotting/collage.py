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
"""Arranging many neurons on a single page.

[`navis.plot_collage`][] is the one entry point, with two layouts:

- `layout="grid"` puts one neuron per cell of a regular grid.
- `layout="dense"` packs them as tightly as they will go, either by their bounding
  boxes or - with `occupancy=True` - by their actual arbors, so that one neuron may
  reach into another's empty space.

Either can be drawn with matplotlib or rendered offscreen with octarine
(`backend=`); both put the neurons in data coordinates, so a page looks the same
and annotates the same way whichever one drew it.

The packing itself is `navis-fastcore`; see [`navis_fastcore.pack_masks`][] for why
it is not the cross correlation you would otherwise write.
"""

import math
import numbers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import navis_fastcore as fastcore

from .. import config, core
from ..utils import mesh_unique_edges
from .dd import plot2d, _view_axes, _view_frame
from .ddd import plot3d

__all__ = ["plot_collage"]

logger = config.get_logger(__name__)

EPS = 1e-9

#: A4 in inches - the default page for both layouts.
A4 = (8.27, 11.69)


# -----------------------------------------------------------------------------
# Views and pages
# -----------------------------------------------------------------------------


def _view_plane(view):
    """Return `(ix, iy, iz, sx, sy)` for a navis 2d `view`.

    `ix`/`iy` index the coordinates that end up on the horizontal/vertical axis,
    `iz` the remaining (depth) one, and `sx`/`sy` are the signs of the view. navis
    never flips coordinates - a "-y" in the view inverts the axis instead - so the
    signs are what we need to turn a position on the page into a position in the
    neuron's own coordinate system.

    The indices and the signs both come from [`navis.plot2d`][]'s own helpers, so a
    given `view` lays a page out the same way it draws one; all this adds is the
    up-front check, since those raise `KeyError`/`IndexError` on a bad view.
    """
    try:
        names = [v.lstrip("+-") for v in view]
    except (AttributeError, TypeError):
        names = None
    if not names or len(names) != 2 or not set(names) <= {"x", "y", "z"}:
        raise ValueError(
            f'`view` must be two of "x", "y", "z" with an optional sign, got {view!r}'
        )
    if names[0] == names[1]:
        raise ValueError(f"`view` must name two different axes, got {view!r}")

    ix, iy, iz = _view_axes(view)
    u, v, _ = _view_frame(view)
    return ix, iy, iz, int(u[ix]), int(v[iy])


def _set_page(ax, page_size, view):
    """Make `ax` show exactly the page, oriented the way `view` asks for."""
    *_, sx, sy = _view_plane(view)
    ax.set_aspect("equal")
    ax.set_axis_off()
    # A negative sign puts the origin on the right/top - the same thing navis' own
    # axis inversion does.
    ax.set_xlim(0, sx * page_size[0])
    ax.set_ylim(0, sy * page_size[1])


def _make_axes(ax, page_size, dpi):
    """The figure and axes to draw into: the page, edge to edge."""
    if ax is None:
        fig = plt.figure(figsize=page_size, dpi=dpi)
        return fig, fig.add_axes((0, 0, 1, 1))
    return ax.get_figure(), ax


def _per_neuron_colors(color, n):
    """Return `color` as a list with one entry per neuron.

    Returns `None` if this is a single color (a name, an `(r, g, b[, a])` tuple, a
    dict keyed by neuron, a colormap, ...) which navis can apply to all neurons
    itself, no matter what order they end up in.
    """
    if not isinstance(color, (list, tuple, np.ndarray)):
        return None

    color = list(color)
    # Same rule navis uses: all numbers means this is a single (r, g, b[, a])
    if all(isinstance(c, numbers.Number) for c in color):
        return None

    if len(color) != n:
        raise ValueError(f"Expected a color for each of the {n} neurons, got {len(color)}")

    return color


def _view_sizes(nl, view):
    """Width and height of each neuron's bounding box in the view plane."""
    ix, iy, *_ = _view_plane(view)
    bbox = np.array([n.bbox for n in nl])
    return np.maximum(bbox[:, [ix, iy], 1] - bbox[:, [ix, iy], 0], EPS)


# -----------------------------------------------------------------------------
# Moving neurons about
# -----------------------------------------------------------------------------


def _coords(neuron):
    """A neuron's coordinates as an (N, 3) array.

    Always a copy, so the caller can scribble on it.
    """
    if isinstance(neuron, core.Dotprops):
        return np.array(neuron.points, dtype=float)
    return np.array(neuron.vertices, dtype=float)


def _set_coords(neuron, co):
    """Write coordinates back, whichever way this neuron happens to hold them."""
    co = np.asarray(co, dtype=float)
    if isinstance(neuron, core.Skeleton):
        neuron.nodes[["x", "y", "z"]] = co
    elif isinstance(neuron, core.Dotprops):
        neuron.points = co
    else:
        neuron.vertices = co


def _move(neuron, offset):
    """Shift a neuron by `offset` (x, y, z), in place."""
    # navis' own `+=` rather than writing the coordinates back ourselves: it already
    # knows where each kind of neuron keeps them, moves the connectors along with
    # them, and invalidates whatever it caches - all of which this would otherwise
    # have to duplicate and keep in step.
    neuron += np.asarray(offset, dtype=float)
    return neuron


def _rotate90(neuron, view):
    """Turn a neuron by 90 degrees inside the view plane, in place."""
    ix, iy, *_ = _view_plane(view)

    co = _coords(neuron)
    co[:, ix], co[:, iy] = -co[:, iy].copy(), co[:, ix].copy()
    _set_coords(neuron, co)

    if neuron.has_connectors:
        cn = neuron.connectors[["x", "y", "z"]].to_numpy(dtype=float)
        cn[:, ix], cn[:, iy] = -cn[:, iy].copy(), cn[:, ix].copy()
        neuron.connectors[["x", "y", "z"]] = cn

    return neuron


def _place(neuron, scale, centre, view, rotate=False):
    """Scale a neuron and move it so that it is centred on `centre`.

    `scale` goes straight to navis, i.e. it can be a single number or an iterable of
    `[x, y, z, radius]` scale factors (meshes take three - they have no radius).
    Multiplication returns a copy, so the original neuron is left alone; the offset
    then has to be applied by hand.
    """
    ix, iy, _, sx, sy = _view_plane(view)

    n = neuron * scale
    if rotate:
        _rotate90(n, view)

    # Where we want the neuron: on the two view axes that is the requested page
    # position (with the sign of the view applied), on the remaining one we simply
    # centre on 0 so that all neurons share a depth range.
    target = np.zeros(3)
    target[ix] = sx * centre[0]
    target[iy] = sy * centre[1]

    return _move(n, target - n.bbox.mean(axis=1))


# -----------------------------------------------------------------------------
# Rasterising
# -----------------------------------------------------------------------------


def _page_coords(neuron, view):
    """A neuron's coordinates in page space, i.e. x to the right and y up."""
    ix, iy, _, sx, sy = _view_plane(view)
    co = _coords(neuron)
    return np.column_stack([sx * co[:, ix], sy * co[:, iy]])


def _page_turn(view):
    """The quarter turn, in page space, that `_rotate90` makes in data space.

    `_rotate90` turns the raw coordinates counter-clockwise, but a view with exactly
    one negative axis mirrors the page - and a mirror turns a counter-clockwise turn
    into the clockwise one. Getting this wrong point-reflects the mask relative to
    the neuron it is standing in for, which packs a shape that is never drawn.
    """
    *_, sx, sy = _view_plane(view)
    return 1 if sx * sy > 0 else 3


def _segments(neuron):
    """Index pairs into a neuron's coordinates that have to be drawn as lines.

    For a skeleton that is every node to its parent, for a mesh its unique edges. A
    `Dotprops` has no lines at all - the points are the neuron.
    """
    if isinstance(neuron, core.Skeleton):
        nodes = neuron.nodes
        parent = pd.Index(nodes.node_id.values).get_indexer(nodes.parent_id.values)
        child = np.nonzero(parent >= 0)[0]
        return np.column_stack([child, parent[child]]).astype(np.uint32)

    if isinstance(neuron, core.Dotprops):
        return np.zeros((0, 2), dtype=np.uint32)

    # Unique, not the three edges of every face: an interior edge is shared by two
    # faces, so walking the faces would rasterise half of them twice - the same
    # pixels, for twice the work and twice the array. `extra_edges=False` because we
    # are drawing the surface, not the connectivity.
    return mesh_unique_edges(neuron, extra_edges=False).astype(np.uint32)


class _Shapes:
    """Everything the rasteriser needs, worked out once instead of per scale.

    The scale search rasterises the same neurons over and over, and neither their
    page coordinates nor their edges depend on the scale - only the pixels do. For
    a mesh those are the largest arrays in play, so deriving them once is most of
    what makes the search affordable. The quarter turns are applied by the rasteriser
    rather than kept here as a second set of coordinates, for the same reason.
    """

    def __init__(self, nl, view, turns):
        self.turns = turns
        self.coords = [_page_coords(n, view) for n in nl]
        self.edges = [_segments(n) for n in nl]
        # A mesh is solid: what it occupies is its silhouette, not the wireframe of
        # its faces. A skeleton is the wireframe.
        self.fill = [not isinstance(n, (core.Skeleton, core.Dotprops)) for n in nl]

    def masks(self, scale, pad_px):
        """Rasterise every neuron, one list of variants each."""
        per_turn = [
            fastcore.rasterize_segments(
                self.coords,
                self.edges,
                scale=scale,
                pad=pad_px,
                fill=self.fill,
                turn=turn,
            )
            for turn in self.turns
        ]
        return [list(variants) for variants in zip(*per_turn)]


# -----------------------------------------------------------------------------
# Masks
# -----------------------------------------------------------------------------


def _resample_mask(mask, shape):
    """Nearest neighbour resample a mask onto a grid of `shape`."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be a 2d array, got {mask.ndim} dimensions")
    if mask.shape == tuple(shape):
        return mask

    # Stretching a mask onto a page of a different shape is rarely what anyone meant -
    # most likely it was built for a different `page_size`
    if abs((mask.shape[0] / mask.shape[1]) / (shape[0] / shape[1]) - 1) > 0.02:
        logger.warning(
            f"Mask {mask.shape} does not have the same aspect ratio as the page "
            f"{tuple(shape)} and will be stretched to fit it."
        )

    rows = (np.arange(shape[0]) * mask.shape[0]) // shape[0]
    cols = (np.arange(shape[1]) * mask.shape[1]) // shape[1]
    return mask[rows[:, None], cols]


def _resolve_mask(mask, page_size, threshold=0.5, invert=False):
    """A `mask=` argument as a bool array covering the page.

    A boolean array is already one, and is used as it is (resampled to the packing
    grid later). Anything else is a picture of the shape the collage should take -
    a file to read or the pixels themselves - and gets scaled onto the page by
    `_mask_from_image`. The dtype is what tells the two apart, since a picture may
    perfectly well arrive as an array.
    """
    if mask is None:
        return None
    if isinstance(mask, np.ndarray) and mask.dtype == bool:
        return mask
    return _mask_from_image(mask, page_size, threshold=threshold, invert=invert)


def _mask_from_image(image, page_size=A4, res=100, threshold=0.5, invert=False):
    """Turn a picture into a page-shaped mask.

    The image is scaled uniformly - i.e. without distorting it - until it fits the
    page and is centred on it; whatever is left over around it is out of bounds.

    Parameters
    ----------
    image :         str | (N, M) | (N, M, 3) | (N, M, 4) array
                    A file to read (anything matplotlib can open) or the pixels
                    themselves, either greyscale or RGB(A). Dark pixels are taken to
                    be the shape, so a black drawing on white paper works as is.
    page_size :     (width, height), default A4 in inches
                    Same page the collage will use.
    res :           int, default 100
                    Pixels per page unit.
    threshold :     float, default 0.5
                    How dark a pixel has to be to count as part of the shape, from 0
                    (black) to 1 (white).
    invert :        bool, default False
                    Use the light parts of the image as the shape instead.

    Returns
    -------
    mask :          (height, width) bool array
                    True wherever neurons may go. Row 0 is the *bottom* of the page,
                    so `plt.imshow(mask, origin="lower")` shows it the way the
                    collage will use it.

    """
    from scipy import ndimage

    if isinstance(image, str) or hasattr(image, "__fspath__"):
        image = plt.imread(image)

    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError(f"Expected a 2d or 3d image, got {img.ndim}d")

    # 0-255 or 0-1 - go by the dtype, which is how matplotlib hands them over
    integral = img.dtype.kind in "ui"

    if img.ndim == 3:
        if img.shape[2] == 4:
            # Lay the picture on white paper so that transparent stays background
            rgba = img / 255 if integral else img.astype(float)
            img, integral = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:]), False
        img = img[..., :3] @ [0.299, 0.587, 0.114]  # to greyscale

    if img.dtype == bool:
        shape = img.astype(float)
    else:
        img = img.astype(float)
        shape = 1 - (img / 255 if integral or img.max() > 1 else img)  # dark = shape
    if invert:
        shape = 1 - shape

    # Pictures start at the top, our pages at the bottom
    shape = shape[::-1]

    page_px = (int(np.ceil(page_size[1] * res)), int(np.ceil(page_size[0] * res)))
    zoom = min(page_px[0] / shape.shape[0], page_px[1] / shape.shape[1])
    shape = ndimage.zoom(shape, zoom, order=1)

    # Centre whatever came out of that on the page
    mask = np.zeros(page_px, dtype=bool)
    h, w = min(shape.shape[0], page_px[0]), min(shape.shape[1], page_px[1])
    top, left = (page_px[0] - h) // 2, (page_px[1] - w) // 2
    mask[top : top + h, left : left + w] = shape[:h, :w] > 1 - threshold

    return mask


# -----------------------------------------------------------------------------
# Layouts
# -----------------------------------------------------------------------------


def _boxes(sizes, rotated, scale, padding=0.0):
    """Sizes of the packed rectangles, with rotation and padding applied."""
    return np.where(rotated[:, None], sizes[:, ::-1], sizes) * scale + padding


def _max_scale(sizes, page, area=None):
    """Largest scale worth trying: no packing can beat it.

    Neither the one at which the neurons' boxes would cover all of `area` (the page,
    unless a mask has taken some of it away), nor the one at which a single neuron
    fills the page.
    """
    if area is None:
        area = page.prod()
    return min(
        math.sqrt(area / sizes.prod(axis=1).sum()),
        (page / sizes.max(axis=0)).min(),
    )


def _bisect(attempt, scale, iterations, hi=None):
    """The largest scale `attempt` still manages, or `None` if none of them did.

    `attempt(scale)` returns a layout or `None`. Given an upper bound that is known
    to be unbeatable this is a plain bisection; without one, the scale is grown (or
    halved) until there is both one that works and one that does not, and only then
    is the gap between them narrowed.
    """
    lo, best = 0.0, None
    for _ in range(iterations):
        out = attempt(scale)
        if out is None:
            hi = scale
        else:
            lo, best = scale, out

        if hi is None:
            scale = lo * 1.6
        elif best is None:
            scale = hi / 2
        else:
            scale = (lo + hi) / 2
    return best


def _layout_boxes(nl, fill, view, page, padding, allow_rotation, iterations):
    """Lay neurons out by packing their bounding boxes.

    Returns `(scale, centres, rotated, fits)` where `centres` and `rotated` cover the
    neurons in `nl` followed by the ones from `fill` that made it in, and `fits` says
    which of `fill` those were.
    """
    sizes = _view_sizes(nl, view)

    def attempt(scale):
        packed = fastcore.pack_rectangles(
            sizes * scale + padding, page, allow_rotation=allow_rotation
        )
        return None if packed[0] is None else (scale, *packed)

    hi = _max_scale(sizes, page)
    best = _bisect(attempt, hi / 2, iterations, hi=hi)

    if best is None:
        raise ValueError("Unable to pack these neurons onto the page. Try a smaller `padding`.")

    scale, positions, rotated, free = best
    boxes = _boxes(sizes, rotated, scale, padding)
    fits = np.zeros(len(fill), dtype=bool)

    # The gaps that are left can now be filled with the second set of neurons. These
    # have to make do with the space and the scale we already have - the ones that
    # don't fit anywhere are dropped.
    if len(fill):
        fill_sizes = _view_sizes(fill, view)
        fill_pos, fill_rot, _ = fastcore.pack_rectangles(
            fill_sizes * scale + padding,
            page,
            allow_rotation=allow_rotation,
            optional=True,
            free=free,
        )
        fits = ~np.isnan(fill_pos).any(axis=1)

        positions = np.vstack([positions, fill_pos[fits]])
        rotated = np.concatenate([rotated, fill_rot[fits]])
        boxes = np.vstack([boxes, _boxes(fill_sizes[fits], fill_rot[fits], scale, padding)])

    # Bisection stops just short of a perfect fit, so there is always a bit of slack
    # left over - spread it evenly instead of leaving it all on one side
    positions = positions + (page - (positions + boxes).max(axis=0)) / 2

    return scale, positions + boxes / 2, rotated, fits


def _layout_occupancy(
    nl,
    fill,
    view,
    page,
    padding,
    allow_rotation,
    res,
    iterations,
    scale_lo=None,
    mask=None,
):
    """Lay neurons out by their actual arbors instead of their bounding boxes.

    Same idea as `_layout_boxes` and the same return value, but on rasterised
    neurons. `scale_lo` is a scale that is known to work (the one the boxes settled
    on) if there is one, `mask` a shape the neurons have to stay inside of. Returns
    `None` if no scale worked out.
    """
    page_px = tuple(np.ceil(page * res).astype(int)[::-1])
    pad_px = int(round(padding / 2 * res))

    sizes = _view_sizes(nl, view)
    fill_sizes = _view_sizes(fill, view) if len(fill) else np.zeros((0, 2))

    # Page coordinates and edges do not change with the scale, so they are derived
    # once and reused by every attempt below.
    turns = [0, _page_turn(view)] if allow_rotation else [0]
    main = _Shapes(nl, view, turns)
    extra = _Shapes(fill, view, turns) if len(fill) else None

    # Everything outside the mask is simply marked as taken before we start
    blocked = None if mask is None else ~_resample_mask(mask, page_px)

    # Filling bottom up would leave the top of a shape empty, so with a mask the
    # neurons are laid down from the middle of it outwards instead
    cost = None
    if blocked is not None:
        centre_px = np.array(np.nonzero(~blocked)).mean(axis=1)
        rows, cols = np.ogrid[0 : page_px[0], 0 : page_px[1]]
        cost = np.hypot(rows - centre_px[0], cols - centre_px[1])

    def attempt(scale):
        grid = None if blocked is None else blocked.copy()
        positions, variant, grid = fastcore.pack_masks(
            main.masks(scale * res, pad_px), page_px, grid=grid, cost=cost
        )
        if positions is None:
            return None

        fits = np.zeros(len(fill), dtype=bool)
        if extra is not None:
            fill_pos, fill_var, grid = fastcore.pack_masks(
                extra.masks(scale * res, pad_px),
                page_px,
                grid=grid,
                cost=cost,
                optional=True,
            )
            fits = ~np.isnan(fill_pos).any(axis=1)
            positions = np.vstack([positions, fill_pos[fits]])
            variant = np.concatenate([variant, fill_var[fits]])

        # `variant` indexes `turns`, and only the turned one swaps the box
        rotated = np.asarray(turns)[variant] != 0
        boxes = _boxes(np.vstack([sizes, fill_sizes[fits]]), rotated, scale)

        # The lower left corner of a neuron sits `pad_px` inside its own mask
        centres = (positions + pad_px) / res + boxes / 2

        # Nothing forces the packing to end up in the middle of the page, so move
        # whatever we used to the centre. With a mask we must not: the neurons are
        # already where the shape wanted them.
        if blocked is None:
            ink = np.nonzero(grid)
            lo_px = np.array([ink[1].min(), ink[0].min()])
            hi_px = np.array([ink[1].max(), ink[0].max()]) + 1
            centres = centres + (page - (hi_px - lo_px) / res) / 2 - lo_px / res

        return scale, centres, rotated, fits

    # Where to start looking: the scale the boxes managed if we have it, else the one
    # at which the neurons would cover all of the available area. Unlike the boxes,
    # that is a starting point rather than a bound - packing the arbors is expected to
    # beat it - so the search has to find its own upper bound.
    area = page.prod() * (1 if blocked is None else 1 - blocked.mean())
    scale = scale_lo or _max_scale(sizes, page, area)

    return _bisect(attempt, scale, iterations)


# -----------------------------------------------------------------------------
# Laying the neurons out
# -----------------------------------------------------------------------------


def _place_grid(
    nl,
    colors,
    view,
    page_size,
    cols=None,
    margin=0.05,
    uniform_scale=False,
    sort=False,
    drop_dangling=False,
):
    """One neuron per cell of a regular grid.

    Returns the placed copies and the colors that go with them, which is not simply
    the ones handed in: `sort` reorders the neurons and `drop_dangling` removes some.
    """
    sizes = _view_sizes(nl, view)
    if sort:
        order = np.argsort(-sizes.prod(axis=1))
        nl, sizes = nl[order], sizes[order]
        if colors is not None:
            colors = [colors[i] for i in order]

    page_w, page_h = page_size
    if cols is None:
        cols = max(1, round(math.sqrt(len(nl) * page_w / page_h)))
    cols = min(int(cols), len(nl))

    dangling = len(nl) % cols
    if drop_dangling and dangling:
        logger.info(
            f"Dropping {dangling} neuron(s) to fill the grid: "
            f"{len(nl) - dangling} of {len(nl)} plotted."
        )
        nl, sizes = nl[: len(nl) - dangling], sizes[: len(nl) - dangling]
        if colors is not None:
            colors = colors[: len(nl)]

    rows = math.ceil(len(nl) / cols)
    cell = np.array([page_w / cols, page_h / rows])
    fit = cell * (1 - margin)

    # Scale to fit the cell: one factor per neuron, or the smallest of them for all
    # of them if we are to keep the relative sizes
    scales = (fit / sizes).min(axis=1)
    if uniform_scale:
        scales[:] = scales.min()

    placed = []
    for i, (neuron, scale) in enumerate(zip(nl, scales)):
        row, col = divmod(i, cols)
        centre = ((col + 0.5) * cell[0], page_h - (row + 0.5) * cell[1])
        placed.append(_place(neuron, scale, centre, view))

    return core.NeuronList(placed), colors


def _place_dense(
    nl,
    colors,
    view,
    page_size,
    padding=0.02,
    allow_rotation=False,
    backfill=None,
    mask=None,
    occupancy=False,
    occupancy_res=100,
    occupancy_iterations=6,
    iterations=18,
):
    """Neurons packed as tightly onto the page as they will go.

    Returns the placed copies and the colors that go with them - the backfill
    neurons that found no room are dropped, and their colors with them.
    """
    fill = core.NeuronList(backfill) if backfill is not None else core.NeuronList([])
    n_main = len(nl)
    page = np.array(page_size, dtype=float)

    # Boxes can only be packed into a rectangle, so a mask goes straight to the
    # occupancy packing - there is no box layout to start it off with
    layout = None
    if mask is None:
        layout = _layout_boxes(nl, fill, view, page, padding, allow_rotation, iterations)

    if occupancy or mask is not None:
        # Try again on the arbors themselves, starting from the scale the boxes
        # managed. If that finds nothing better we simply keep the boxes.
        better = _layout_occupancy(
            nl,
            fill,
            view,
            page,
            padding,
            allow_rotation,
            occupancy_res,
            occupancy_iterations,
            scale_lo=None if layout is None else layout[0],
            mask=mask,
        )
        if better is not None:
            layout = better
        elif mask is None:
            logger.warning(
                "Occupancy packing found no layout better than the bounding "
                "boxes - falling back to those."
            )

    if layout is None:
        raise ValueError(
            "Unable to fit these neurons into the mask. Try a smaller `padding` "
            "or more `occupancy_iterations`."
        )

    scale, centres, rotated, fits = layout

    # Backfill neurons that did not fit are dropped, together with their colors
    if len(fill):
        nl = nl + fill[fits]
        if colors is not None:
            colors = colors[:n_main] + [c for c, ok in zip(colors[n_main:], fits) if ok]

    placed = [
        _place(neuron, scale, centre, view, rotate=rot)
        for neuron, centre, rot in zip(nl, centres, rotated)
    ]
    return core.NeuronList(placed), colors


#: The `layout=` choices. Not a dispatch table — the two take different arguments,
#: so `plot_collage` calls them by name and this is what it validates against.
LAYOUTS = ("grid", "dense")


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------


def _draw(placed, view, ax, backend, color, page_size, dpi, **kwargs):
    """Draw the placed neurons onto `ax`, with whichever renderer was asked for.

    Both backends put the neurons in *data* coordinates - `plot3d(snapshot=True)`
    renders offscreen and hands back the image placed at its own extent - so the
    page can be set the same way for either, and matplotlib overlays drawn on top
    line up with both.
    """
    if backend == "matplotlib":
        plot2d(placed, view=view, ax=ax, color=color, **kwargs)
        return

    # An orthographic camera is what makes the data-coordinate mapping exact rather
    # than only true at the focal plane; it is octarine's default, but the collage
    # depends on it, so it is not left to chance.
    kwargs.setdefault("camera", "ortho")
    # Frame the page rather than the ink: the pixels then mean the same thing from
    # one collage to the next, and a neuron near the edge is not blown up to fill
    # the canvas. `size` is in pixels, and `dpi` is what the caller already used to
    # say how fine the page should be.
    kwargs.setdefault(
        "size", tuple(int(round(s * dpi)) for s in page_size)
    )
    kwargs.setdefault("margin", 0)
    plot3d(
        placed,
        backend="octarine",
        snapshot=True,
        view=view,
        ax=ax,
        color=color,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# The public function
# -----------------------------------------------------------------------------


def plot_collage(
    x=None,
    layout="grid",
    view=("x", "-y"),
    page_size=A4,
    backend="matplotlib",
    color=None,
    placed=None,
    dpi=300,
    ax=None,
    # `layout="grid"` only
    cols=None,
    margin=0.05,
    uniform_scale=False,
    sort=False,
    drop_dangling=False,
    # `layout="dense"` only
    padding=0.02,
    allow_rotation=False,
    backfill=None,
    mask=None,
    occupancy=False,
    occupancy_res=100,
    occupancy_iterations=6,
    iterations=18,
    **kwargs,
):
    """Arrange many neurons on a single page and plot them.

    Two layouts:

    - `"grid"` puts one neuron per cell of a regular grid.
    - `"dense"` packs them as tightly as they will go, keeping their relative
      sizes. By default that is done on their bounding boxes, which is fast but
      wastes all the empty space inside a box; `occupancy=True` packs the arbors
      themselves, so a neuron may reach into another's empty space - even into the
      loop of another neuron - as long as no cable collides.

    Parameters
    ----------
    x :             NeuronList
                    Skeletons, meshes or a mix of the two. Can be `None` if
                    `placed` is given.
    layout :        "grid" | "dense", default "grid"
    view :          tuple, default `("x", "-y")`
                    View to plot; also the plane the neurons are laid out in.
    page_size :     (width, height), default A4 in inches
    backend :       "matplotlib" | "octarine", default "matplotlib"
                    What draws the neurons. The layout is decided before either of
                    them sees anything, so both give the same page; what differs is
                    what ends up in the figure.

                    `"matplotlib"` goes through [`navis.plot2d`][] and draws every
                    neuron as vector paths - scalable and editable afterwards, but
                    the file grows with the amount of cable on the page.
                    `"octarine"` renders the page offscreen with
                    [`navis.plot3d`][]`(snapshot=True)` and places the resulting
                    image on the axes, which fixes the size by `dpi` no matter how
                    dense the page gets (160 neurons: 8.9 MB of SVG against 1.0 MB)
                    and shades meshes the way the interactive viewer does. Neither
                    is reliably faster than the other - at a few hundred skeletons
                    they are within a factor of two.

                    Either way you get a matplotlib figure back with the neurons in
                    their own coordinates, so overlays land in the same place.
    color :         color | iterable of colors, optional
                    Either a single color for all neurons or one color per neuron,
                    in the order of `x`. Colors follow their neurons through
                    whatever the layout does with them - `sort`, `drop_dangling`
                    and dropped `backfill` neurons all take their color along. With
                    `backfill`, pass `len(x) + len(backfill)` of them.
    placed :        NeuronList, optional
                    Neurons that have already been placed, as returned by an
                    earlier call. If given, the layout is skipped and these are
                    plotted as they are - use this to re-plot the same page with
                    e.g. different colors, or with the other backend. Every layout
                    parameter is then ignored; `view` and `page_size` still apply
                    and should match the ones the neurons were placed with.
    dpi :           int, default 300
                    Figure resolution, and - for `backend="octarine"` - what sizes
                    the render, `page_size * dpi` pixels.
    ax :            matplotlib.axes.Axes, optional
                    Axes to draw on. A new page-sized figure is made if not given.

    Parameters (`layout="grid"` only)
    ---------------------------------
    cols :          int, optional
                    Number of columns. Defaults to whatever makes the cells as
                    square as possible.
    margin :        float, default 0.05
                    Fraction of each cell kept free around the neuron.
    uniform_scale : bool, default False
                    If True, all neurons are scaled by the same factor and hence
                    keep their relative sizes. If False, each neuron is scaled to
                    fill its cell - note that this scales the radii along with it,
                    so use `uniform_scale=True` if you plot with `radius` and want
                    comparable line widths.
    sort :          bool, default False
                    If True, sort neurons by size (largest first).
    drop_dangling : bool, default False
                    If True, drop neurons off the end until the remaining ones fill
                    the grid exactly, i.e. no partially filled last row. Combine
                    with `sort=True` to make sure the ones dropped are the smallest.

    Parameters (`layout="dense"` only)
    ----------------------------------
    padding :       float, default 0.02
                    Gap between neurons, in page units.
    allow_rotation : bool, default False
                    If True, neurons may be turned by 90 degrees to pack tighter.
    backfill :      NeuronList, optional
                    Neurons used to fill the gaps left after `x` has been packed.
                    They do not influence the scale and are drawn at the same scale
                    as `x`; those that do not fit into any of the remaining gaps are
                    silently dropped. They are tried largest first, so the order you
                    pass them in is irrelevant.
    mask :          2d bool array | str | image array, optional
                    A shape the neurons have to stay inside of. A boolean array is
                    taken to cover the whole page, `True` where neurons are allowed,
                    and is resampled to the packing grid - so its resolution does
                    not have to match, and row 0 is the bottom of the page. Anything
                    else is read as a *picture* of the shape (a file path, or the
                    pixels), scaled onto the page with dark counting as the shape.
                    Bounding boxes have no way of following an arbitrary outline, so
                    a mask always packs by occupancy, laying the neurons down from
                    the middle of the shape outwards.
    occupancy :     bool, default False
                    If True, pack the neurons by their actual arbors instead of
                    their bounding boxes. Packs considerably tighter but is slower.
                    The box packing runs first either way and is used as the
                    starting point, so this can only improve on it.
    occupancy_res : int, default 100
                    Resolution of the occupancy grid in pixels per page unit.
                    Higher is more precise but slower. `occupancy` only.
    occupancy_iterations : int, default 6
                    Bisection steps spent looking for a scale better than the one
                    the boxes managed. `occupancy` only.
    iterations :    int, default 18
                    Bisection steps used to find the scale of the box packing.

    **kwargs :      Passed to [`navis.plot2d`][] or, for `backend="octarine"`, to
                    [`navis.plot3d`][].

    Returns
    -------
    fig, ax, placed
                    `placed` are the scaled and moved copies of the neurons, in the
                    order they were plotted - the originals are left alone. Pass
                    them back in as `placed` to reproduce this exact page.

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(5)
    >>> fig, ax, placed = navis.plot_collage(nl, color="k")

    Pack them instead of gridding them, and use a second set to fill the gaps:

    >>> fig, ax, placed = navis.plot_collage(
    ...     nl[:3], layout="dense", backfill=nl[3:], occupancy=True, color="k"
    ... )

    Re-plot that exact page in a different colour, without laying it out again:

    >>> fig, ax, _ = navis.plot_collage(placed=placed, color="r")

    """
    if x is None and placed is None:
        raise ValueError("Need either `x` or `placed`.")
    if layout not in LAYOUTS:
        raise ValueError(f'`layout` must be one of {LAYOUTS}, got "{layout}"')
    if backend not in ("matplotlib", "octarine"):
        raise ValueError(
            f'`backend` must be "matplotlib" or "octarine", got "{backend}"'
        )

    # Neurons that are already placed are plotted as they are - `x` is then not used
    nl = core.NeuronList(x if placed is None else placed)

    fig, ax = _make_axes(ax, page_size, dpi)

    if not len(nl):
        _set_page(ax, page_size, view)
        return fig, ax, nl

    if placed is not None:
        colors = _per_neuron_colors(color, len(nl))
    else:
        # The backfill neurons are packed into whatever gaps are left at the end but
        # we need to know about them now: they have to come with their own colors
        n_extra = len(core.NeuronList(backfill)) if backfill is not None else 0
        colors = _per_neuron_colors(color, len(nl) + n_extra)

        if layout == "grid":
            nl, colors = _place_grid(
                nl,
                colors,
                view,
                page_size,
                cols=cols,
                margin=margin,
                uniform_scale=uniform_scale,
                sort=sort,
                drop_dangling=drop_dangling,
            )
        else:
            nl, colors = _place_dense(
                nl,
                colors,
                view,
                page_size,
                padding=padding,
                allow_rotation=allow_rotation,
                backfill=backfill,
                mask=_resolve_mask(mask, page_size),
                occupancy=occupancy,
                occupancy_res=occupancy_res,
                occupancy_iterations=occupancy_iterations,
                iterations=iterations,
            )

    _draw(
        nl,
        view,
        ax,
        backend,
        color if colors is None else colors,
        page_size,
        dpi,
        **kwargs,
    )
    _set_page(ax, page_size, view)

    return fig, ax, nl
