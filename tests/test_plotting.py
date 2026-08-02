import matplotlib

# Use a headless backend so tests don't try to open windows.
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import ast
import inspect
import re

import numpy as np
import pytest

import navis
from navis.plotting._common import (
    SOMA_COUNT_LIMIT,
    apply_shade_by,
    resolve_cn_color,
    resolve_connectors,
    resolve_somata,
)
from navis.plotting.colors import vertex_colors
from navis.plotting.dd import (
    MESH_SHADE_MODES,
    MESH_ZORDER,
    _collapse_colored_segments,
    _colors_are_categorical,
    _view_frame,
    _view_front,
)
from navis.plotting.render import image_extent, parse_view
from navis.plotting.plot_utils import mesh_faces
from navis.plotting.settings import (
    K3dSettings,
    Matplotlib2dSettings,
    OctarineSettings,
    PlotlySettings,
)


@pytest.fixture
def two_skeletons():
    return navis.example_neurons(2, kind="skeleton")


@pytest.fixture
def skeleton():
    n = navis.example_neurons(1, kind="skeleton")
    navis.strahler_index(n)
    return n


def _n_edges(n):
    """Number of parent->child edges (nodes minus roots)."""
    return int((n.nodes.parent_id.values >= 0).sum())


def _first_line_collection(ax, kind=LineCollection):
    for c in ax.collections:
        if isinstance(c, kind):
            return c
    return None


def _node_colors(n, by="strahler_index"):
    """Per-node RGBA (0-1) as consumed by `_plot_skeleton`."""
    cols = vertex_colors(n, by=by, palette="viridis", na="raise")[0]
    # `vertex_colors` returns RGB in 0-255 but alpha in 0-1 here.
    if cols[:, :3].max() > 1:
        cols = cols.copy()
        cols[:, :3] = cols[:, :3] / 255.0
    return cols


# --------------------------------------------------------------------------- #
#  Grouping helper                                                             #
# --------------------------------------------------------------------------- #


def test_collapse_reduces_segments_and_covers_every_edge(skeleton):
    """Categorical colors collapse into few polylines that still draw every
    edge exactly once (lossless)."""
    colors = _node_colors(skeleton)
    lines, line_colors = _collapse_colored_segments(skeleton, colors)

    n_edges = _n_edges(skeleton)
    # Far fewer polylines than edges ...
    assert len(lines) < n_edges / 2
    assert len(line_colors) == len(lines)
    # ... yet every edge is drawn exactly once.
    assert sum(len(l) - 1 for l in lines) == n_edges


def test_collapse_colors_match_per_node(skeleton):
    """Each collapsed edge keeps the exact color of its child node, so the
    result is visually identical to the per-edge path."""
    colors = _node_colors(skeleton)
    lines, line_colors = _collapse_colored_segments(skeleton, colors)

    # Map every drawn edge (rounded coords) to the color it was given.
    xyz = skeleton.nodes.set_index("node_id")[["x", "y", "z"]]
    drawn = {}
    for line, c in zip(lines, line_colors):
        for i in range(len(line) - 1):
            key = tuple(np.round(np.concatenate([line[i], line[i + 1]]), 3))
            drawn[key] = tuple(np.round(c, 6))

    # Ground truth: edge (child, parent) should carry the child's color.
    nodes = skeleton.nodes
    has_parent = nodes.parent_id.values >= 0
    child_ids = nodes.node_id.values[has_parent]
    parent_ids = nodes.parent_id.values[has_parent]
    child_colors = colors[has_parent]

    for child, parent, c in zip(child_ids, parent_ids, child_colors):
        p1 = xyz.loc[child].values
        p2 = xyz.loc[parent].values
        key = tuple(np.round(np.concatenate([p1, p2]), 3))
        key_rev = tuple(np.round(np.concatenate([p2, p1]), 3))
        got = drawn.get(key, drawn.get(key_rev))
        assert got is not None
        assert got == tuple(np.round(c, 6))


def test_colors_are_categorical_gate(skeleton):
    """The gate says 'group' for few distinct colors and 'fall back' when
    (nearly) every node has its own color."""
    categorical = _node_colors(skeleton)  # strahler -> a handful of colors
    assert _colors_are_categorical(categorical)

    # Genuinely continuous: a unique color per node.
    rng = np.random.default_rng(0)
    unique = rng.random((skeleton.n_nodes, 4))
    assert not _colors_are_categorical(unique)


# --------------------------------------------------------------------------- #
#  plot2d integration                                                          #
# --------------------------------------------------------------------------- #


def test_plot2d_color_by_groups_segments(skeleton):
    """`plot2d(..., color_by=<categorical>)` draws grouped, contiguous lines
    rather than one short segment per edge."""
    fig, ax = navis.plot2d(
        skeleton, color_by="strahler_index", palette="viridis", method="2d"
    )
    try:
        lc = _first_line_collection(ax)
        assert lc is not None
        segs = lc.get_segments()

        n_edges = _n_edges(skeleton)
        assert len(segs) < n_edges / 2  # grouped
        assert sum(len(s) - 1 for s in segs) == n_edges  # lossless coverage
    finally:
        plt.close(fig)


def test_plot2d_depth_coloring_stays_per_edge(skeleton):
    """Depth coloring is a continuous gradient and must keep the per-edge
    LineCollection (one 2-point segment per edge)."""
    fig, ax = navis.plot2d(skeleton, depth_coloring=True, method="2d")
    try:
        lc = _first_line_collection(ax)
        assert lc is not None
        segs = lc.get_segments()
        assert len(segs) == _n_edges(skeleton)
        assert all(len(s) == 2 for s in segs)
    finally:
        plt.close(fig)


def test_plot2d_color_by_3d_groups_segments(skeleton):
    """The `method='3d'` path groups categorical colors too."""
    fig, ax = navis.plot2d(
        skeleton, color_by="strahler_index", palette="viridis", method="3d"
    )
    try:
        lc = _first_line_collection(ax, kind=Line3DCollection)
        assert lc is not None
        # Line3DCollection populates get_segments() only after projection.
        fig.canvas.draw()
        assert len(lc.get_segments()) < _n_edges(skeleton) / 2
    finally:
        plt.close(fig)


def test_plot2d_plain_color_renders(skeleton):
    """A single-color skeleton still renders via the fast Line2D path."""
    fig, ax = navis.plot2d(skeleton, color="k", method="2d")
    try:
        # No LineCollection - the uniform-color path uses Line2D artists.
        assert _first_line_collection(ax) is None
        assert len(ax.lines) >= 1
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
#  Radius, halo, depth_sort, taper and styles                                  #
# --------------------------------------------------------------------------- #


def _poly_collections(ax):
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def _ribbon_artists(ax):
    """Ribbon artists, in whichever form `_plot_ribbon` chose.

    A uniformly coloured ribbon is a single compound path (so that a translucent
    neuron does not show every capsule it is built from); per-node colours need
    one polygon per capsule.
    """
    return [c for c in ax.collections if isinstance(c, (PolyCollection, PathCollection))]


def _n_subpaths(artist):
    paths = artist.get_paths()
    if len(paths) != 1:
        return len(paths)
    return int((paths[0].codes == mpath.Path.MOVETO).sum())


def _line_collections(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def test_plot2d_radius_outlines_in_2d(skeleton):
    """`radius=True` outlines the tube in the view plane: one polygon per edge
    plus one disc per node, and no mesh conversion."""
    fig, ax = navis.plot2d(skeleton, radius=True, method="2d")
    try:
        artists = _ribbon_artists(ax)
        assert len(artists) == 1
        assert _n_subpaths(artists[0]) == _n_edges(skeleton) + skeleton.n_nodes
    finally:
        plt.close(fig)


def test_plot2d_radius_3d_still_meshes(skeleton):
    """The 3d methods have no view plane to outline in, so they keep meshing."""
    fig, ax = navis.plot2d(skeleton, radius=True, method="3d")
    try:
        polys = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert polys
        # Poly3DCollection populates get_paths() only after projection
        fig.canvas.draw()
        # tree2meshneuron gives us triangles, i.e. far more paths than edges
        assert sum(len(p.get_paths()) for p in polys) > _n_edges(skeleton)
    finally:
        plt.close(fig)


def test_plot2d_radius_lw_maps_onto_linewidth(skeleton):
    """`radius="lw"` stays a LineCollection and varies the width per edge."""
    fig, ax = navis.plot2d(skeleton, radius="lw", method="2d")
    try:
        assert not _ribbon_artists(ax)
        lcs = _line_collections(ax)
        assert len(lcs) == 1
        assert len(lcs[0].get_paths()) == _n_edges(skeleton)
        fig.canvas.draw()
        assert len(np.unique(lcs[0].get_linewidth())) > 1
    finally:
        plt.close(fig)


def test_plot2d_radius_lw_correct_on_first_draw(skeleton):
    """The width has to be right *during* the first draw, not after it.

    Computing it off a `draw_event` looks like it works interactively but fires
    after the artist has been rendered, so the first `savefig` goes out with
    whatever the scale was before the axes were autoscaled - which is a solid
    block of colour.
    """
    fig, ax = navis.plot2d(skeleton, radius="lw", method="2d")
    try:
        lc = _line_collections(ax)[0]
        rendered = []
        real_draw = lc.draw

        def spy(renderer):
            real_draw(renderer)
            rendered.append(np.max(lc.get_linewidth()))

        lc.draw = spy
        fig.canvas.draw()

        assert rendered
        # a sane width for a neuron that fills the axes, not thousands of points
        assert rendered[0] < 50
    finally:
        plt.close(fig)


def test_plot2d_radius_lw_tracks_zoom(skeleton):
    """Zooming in has to make the neurites wider, since the radius is in data
    units but the line width is in points."""
    fig, ax = navis.plot2d(skeleton, radius="lw", method="2d")
    try:
        lc = _line_collections(ax)[0]
        fig.canvas.draw()
        wide = np.max(lc.get_linewidth())

        # both axes, since `aspect="equal", adjustable="box"` would otherwise
        # just reshape the box and leave the scale alone
        for lim, setter in ((ax.get_xlim(), ax.set_xlim), (ax.get_ylim(), ax.set_ylim)):
            setter(lim[0], lim[0] + (lim[1] - lim[0]) / 10)
        fig.canvas.draw()
        assert np.max(lc.get_linewidth()) > wide * 5
    finally:
        plt.close(fig)


def test_plot2d_radius_falls_back_without_radii(skeleton):
    """A skeleton with no radius column plots as lines instead of raising."""
    n = skeleton.copy()
    n.nodes.drop(columns=["radius"], inplace=True)
    fig, ax = navis.plot2d(n, radius=True, method="2d")
    try:
        assert not _ribbon_artists(ax)
        assert len(ax.lines) >= 1
    finally:
        plt.close(fig)


def test_plot2d_halo_is_a_separate_artist(skeleton):
    """The halo must be its own collection *underneath* the neuron.

    `patheffects.withStroke` applies per path, so as a path effect each segment's
    halo would erase the segments drawn before it - the neuron would come out
    dashed. This is the regression guard for that.
    """
    fig, ax = navis.plot2d(skeleton, color="k", halo=True, method="2d")
    try:
        lcs = _line_collections(ax)
        assert len(lcs) == 2
        halo, line = lcs
        assert not halo.get_path_effects()
        assert not line.get_path_effects()
        assert halo.get_zorder() < line.get_zorder()
        assert halo.get_linewidth()[0] > line.get_linewidth()[0]
        # halo defaults to the background it sits on
        assert np.allclose(halo.get_color()[0], mcolors.to_rgba(ax.get_facecolor()))
    finally:
        plt.close(fig)


def test_plot2d_halo_without_depth_sort_stacks_per_neuron(two_skeletons):
    """Every neuron needs its own z-order slot, halo included.

    With one shared z-order per role all the halos end up below all the neurons,
    which on a white background means the halo is white-on-white under everything
    - i.e. invisible. Each neuron's halo has to sit above the previous neuron.
    """
    fig, ax = navis.plot2d(two_skeletons, color="k", halo=True, method="2d")
    try:
        first_halo, first_line, second_halo, second_line = _line_collections(ax)
        zorders = [c.get_zorder() for c in (first_halo, first_line, second_halo, second_line)]
        assert zorders == sorted(zorders)
        assert len(set(zorders)) == 4
        # ... and still inside the skeleton band, so somata and connectors stay on top
        assert all(1 <= z < 2 for z in zorders)
    finally:
        plt.close(fig)


def test_plot2d_halo_accepts_dict(skeleton):
    fig, ax = navis.plot2d(
        skeleton, color="k", halo={"width": 7, "color": "red"}, method="2d"
    )
    try:
        halo, line = _line_collections(ax)
        assert np.allclose(halo.get_color()[0], mcolors.to_rgba("red"))
        assert halo.get_linewidth()[0] == pytest.approx(line.get_linewidth()[0] + 7)
    finally:
        plt.close(fig)


def test_plot2d_depth_sort_interleaves_neurons(two_skeletons):
    """Depth bins are global, so the same depth gives the same z-order for every
    neuron - which is what lets them interleave."""
    fig, ax = navis.plot2d(two_skeletons, depth_sort=6, method="2d")
    try:
        lcs = _line_collections(ax)
        assert len(lcs) > 2  # more than one artist per neuron

        zorders = sorted({c.get_zorder() for c in lcs})
        # every bin's z-order shows up for both neurons
        counts = {z: sum(c.get_zorder() == z for c in lcs) for z in zorders}
        assert set(counts.values()) == {2}
        # and they stay below somata (4) and connectors
        assert max(zorders) < 4
    finally:
        plt.close(fig)


def test_plot2d_depth_sort_negative_flips_order(skeleton):
    """A negative bin count swaps which end of the depth axis is nearest."""

    def front_bin_depth(sort):
        fig, ax = navis.plot2d(skeleton, depth_sort=sort, method="2d")
        try:
            top = max(_line_collections(ax), key=lambda c: c.get_zorder())
            return np.mean([np.mean(p.vertices) for p in top.get_paths()])
        finally:
            plt.close(fig)

    # the frontmost bin should sit at opposite ends of the depth range
    assert front_bin_depth(6) != front_bin_depth(-6)


def test_plot2d_taper_varies_width(skeleton):
    """Taper turns a constant width into a per-edge one inside the documented
    range."""
    lw = 2
    fig, ax = navis.plot2d(skeleton, taper="strahler", linewidth=lw, method="2d")
    try:
        widths = np.asarray(_line_collections(ax)[0].get_linewidth())
        assert len(np.unique(widths)) > 1
        lo, hi = navis.plotting.dd.TAPER_RANGE
        assert widths.min() >= lw * lo - 1e-9
        assert widths.max() <= lw * hi + 1e-9
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["strahler", "subtree"])
def test_plot2d_taper_kinds_snapshot(skeleton, kind):
    fig, ax = navis.plot2d(skeleton, taper=kind, method="2d")
    try:
        assert len(_line_collections(ax)[0].get_paths()) == _n_edges(skeleton)
    finally:
        plt.close(fig)


def test_plot2d_taper_rejects_unknown(skeleton):
    fig = None
    try:
        with pytest.raises(ValueError, match="Unknown taper"):
            fig, _ = navis.plot2d(skeleton, taper="nonsense", method="2d")
    finally:
        plt.close("all")


def test_plot2d_style_fills_defaults_only(skeleton):
    """A style sets what the caller did not, and nothing else."""
    fig, ax = navis.plot2d(skeleton, style="publication", method="2d")
    try:
        assert _ribbon_artists(ax)  # radius="auto" kicked in
    finally:
        plt.close(fig)

    # ... but an explicit argument wins
    fig, ax = navis.plot2d(skeleton, style="publication", radius=False, method="2d")
    try:
        assert not _ribbon_artists(ax)
    finally:
        plt.close(fig)


def test_plot2d_style_shades_meshes(mesh):
    """The style has to cover meshes too, not just skeletons."""

    def face_colors(ax):
        # the style depth-sorts, so the faces come back across several artists
        return np.concatenate([c.get_facecolor() for c in ax.collections])

    fig, ax = navis.plot2d(mesh, style="publication", method="2d", color="b")
    try:
        # shaded, so per face rather than the single-path fill
        assert len(np.unique(face_colors(ax), axis=0)) > 50
    finally:
        plt.close(fig)

    fig, ax = navis.plot2d(
        mesh, style="publication", mesh_shade=False, method="2d", color="b"
    )
    try:
        assert len(np.unique(face_colors(ax), axis=0)) == 1
    finally:
        plt.close(fig)


def test_plot2d_style_respects_aliases(skeleton):
    """Passing `lw` must count as having passed `linewidth`."""
    style = dict(navis.plotting.dd.PLOT_STYLES["publication"])
    try:
        navis.plotting.dd.PLOT_STYLES["publication"] = dict(style, linewidth=9)
        kwargs = navis.plotting.dd._apply_style(
            dict(style="publication", lw=2, method="2d")
        )
        assert "linewidth" not in kwargs
        assert kwargs["lw"] == 2
    finally:
        navis.plotting.dd.PLOT_STYLES["publication"] = style


def test_plot2d_style_rejects_unknown(skeleton):
    with pytest.raises(ValueError, match="Unknown style"):
        navis.plot2d(skeleton, style="nonsense", method="2d")


def _two_node_skeleton(radius=5):
    """A single fat edge: its two node discs overlap the edge quad heavily."""
    import pandas as pd

    nodes = pd.DataFrame(
        {
            "node_id": [0, 1],
            "parent_id": [-1, 0],
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "radius": [float(radius), float(radius)],
        }
    )
    return navis.Skeleton(nodes)


def test_plot2d_radius_fills_overlaps_once(skeleton):
    """A translucent ribbon must not show the capsules it is built from.

    The union is filled as one compound path under the nonzero winding rule,
    which only works if the node discs are wound the same way as the edge quads.
    Wound the other way they cancel and punch a *hole* where they overlap - so
    this checks both the double-blend and the hole.
    """
    n = _two_node_skeleton(radius=5)
    fig, ax = navis.plot2d(n, method="2d", radius=True, color="b", alpha=0.5, soma=False)
    try:
        ax.set_xlim(-8, 18)
        ax.set_ylim(-8, 8)
        ax.set_axis_off()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float)

        def at(x, y):
            return img[
                int(round(ax.transData.transform((x, y))[1])) * -1 + img.shape[0] - 1,
                int(round(ax.transData.transform((x, y))[0])),
            ]

        disc_only = at(-3.5, 0)  # inside the left disc, past the quad
        overlap = at(0, 0)  # node centre: disc and quad on top of each other
        quad_only = at(5, 0)  # mid-edge, quad only

        # all three are the same single layer of 50% blue - no hole, no build-up
        assert np.allclose(overlap, disc_only, atol=2)
        assert np.allclose(overlap, quad_only, atol=2)
        # ... and it really is translucent, not the flat background
        assert overlap[2] > overlap[0] + 20
    finally:
        plt.close(fig)


def test_plot2d_radius_with_per_node_colors(skeleton):
    """Per-node colors have to survive the trip onto the polygons."""
    fig, ax = navis.plot2d(
        skeleton, radius=True, color_by="strahler_index", palette="viridis", method="2d"
    )
    try:
        pc = _poly_collections(ax)[0]
        colors = pc.get_facecolor()
        assert len(colors) == len(pc.get_paths())
        assert len(np.unique(colors, axis=0)) > 1
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
#  Meshes                                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def mesh():
    return navis.example_neurons(1, kind="mesh")


def _mesh_artist(ax):
    """The one artist a single-coloured, unbinned mesh should produce."""
    artists = [c for c in ax.collections if isinstance(c, (PathCollection, PolyCollection))]
    assert len(artists) == 1
    return artists[0]


def _signed_area(tri):
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )


def test_mesh_faces_culls_and_sorts(mesh):
    """Back faces go, and what is left is painted furthest away first."""
    tri, normals, depth, ix = mesh_faces(
        mesh.vertices, mesh.faces, (0, 2), 1, front=_view_front(("x", "-z"))
    )
    n_faces = len(mesh.faces)
    # a closed-ish surface shows roughly half of itself
    assert 0.3 * n_faces < len(tri) < 0.7 * n_faces
    assert len(tri) == len(normals) == len(depth) == len(ix)
    # front is +y for this view, so depth has to run from far to near
    assert np.all(np.diff(depth) >= 0)


def test_mesh_faces_winding_is_consistent(mesh):
    """Every kept triangle must project the same way round.

    The union fill relies on it: under the nonzero winding rule two overlapping
    subpaths wound against each other cancel and leave a hole. Culling by the sign
    of the normal is what guarantees it, so there is nothing to orient by hand.
    """
    for view, xy_ix, d_ix in [(("x", "-z"), (0, 2), 1), (("z", "y"), (2, 1), 0)]:
        tri, _, _, _ = mesh_faces(
            mesh.vertices, mesh.faces, xy_ix, d_ix, front=_view_front(view)
        )
        area = _signed_area(tri)
        assert np.all(area >= 0) or np.all(area <= 0)


def test_view_front_matches_the_view(mesh):
    """Which way the viewer is has to come from the view, not from the data."""
    assert _view_front(("x", "y")) == 1
    assert _view_front(("x", "-y")) == -1
    assert _view_front(("x", "z")) == -1
    assert _view_front(("x", "-z")) == 1
    assert _view_front(("y", "x")) == -1


def test_plot2d_mesh_is_a_single_path(mesh):
    """One colour, no shading: the whole mesh is one filled path."""
    fig, ax = navis.plot2d(mesh, method="2d", view=("x", "-z"), color="b")
    try:
        artist = _mesh_artist(ax)
        assert isinstance(artist, PathCollection)
        assert len(artist.get_paths()) == 1
        # one subpath per front-facing triangle, i.e. fewer than the mesh has
        assert 0 < _n_subpaths(artist) < len(mesh.faces)
    finally:
        plt.close(fig)


def test_plot2d_mesh_fills_overlaps_once(mesh):
    """A translucent mesh must not show its own triangulation.

    Every face composited separately turns "40% opaque" into "however many
    triangles the ray crossed", which is what the single compound path fixes.
    """
    fig, ax = navis.plot2d(mesh, method="2d", view=("x", "-z"), color="b", alpha=0.4)
    try:
        ax.set_axis_off()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float)
        painted = img[img[..., 2] > img[..., 0] + 5]
        assert len(painted) > 500

        # One 40% layer of blue on white is 0.6 * 255 in the red channel; two
        # layers would be 0.36 * 255. Antialiasing along the outline only ever
        # blends *towards* white - the neurites are thin enough that most pixels
        # are partly covered - so the guard is one-sided by design.
        single = round(0.6 * 255)
        assert painted[:, 0].min() > single - 3
        # ... and fully covered pixels are the most common kind
        assert np.bincount(painted[:, 0].astype(int)).argmax() == single
    finally:
        plt.close(fig)


def test_plot2d_mesh_shade_varies_facecolor(mesh):
    """Shading is per face, so it cannot ride on the single-path fill."""
    fig, ax = navis.plot2d(mesh, method="2d", view=("x", "-z"), color="b")
    try:
        assert len(np.unique(_mesh_artist(ax).get_facecolor(), axis=0)) == 1
    finally:
        plt.close(fig)

    fig, ax = navis.plot2d(
        mesh, method="2d", view=("x", "-z"), color="b", mesh_shade=True
    )
    try:
        artist = _mesh_artist(ax)
        assert isinstance(artist, PolyCollection)
        colors = artist.get_facecolor()
        assert len(colors) == len(artist.get_paths())
        assert len(np.unique(colors, axis=0)) > 50
    finally:
        plt.close(fig)


@pytest.mark.parametrize("mode", MESH_SHADE_MODES)
def test_plot2d_mesh_shade_modes_snapshot(mesh, mode):
    fig, ax = navis.plot2d(mesh, method="2d", view=("x", "-z"), mesh_shade=mode)
    try:
        fig.canvas.draw()
        assert len(np.unique(_mesh_artist(ax).get_facecolor(), axis=0)) > 1
    finally:
        plt.close(fig)


def test_plot2d_mesh_shade_ghost_varies_alpha(mesh):
    """"ghost" works on opacity rather than brightness."""
    fig, ax = navis.plot2d(
        mesh, method="2d", view=("x", "-z"), color="b", mesh_shade="ghost"
    )
    try:
        colors = _mesh_artist(ax).get_facecolor()
        assert len(np.unique(colors[:, 3])) > 50
        assert len(np.unique(colors[:, :3], axis=0)) == 1
    finally:
        plt.close(fig)


def test_plot2d_mesh_shade_accepts_dict(mesh):
    fig, ax = navis.plot2d(
        mesh,
        method="2d",
        view=("x", "-z"),
        color="b",
        mesh_shade={"mode": "lambert", "ambient": 0.9},
    )
    try:
        bright = _mesh_artist(ax).get_facecolor()
    finally:
        plt.close(fig)

    fig, ax = navis.plot2d(
        mesh,
        method="2d",
        view=("x", "-z"),
        color="b",
        mesh_shade={"mode": "lambert", "ambient": 0.0},
    )
    try:
        dim = _mesh_artist(ax).get_facecolor()
    finally:
        plt.close(fig)

    # more ambient light means a smaller spread between lit and unlit faces
    assert np.ptp(bright[:, :3]) < np.ptp(dim[:, :3])


@pytest.mark.parametrize("method", ["2d", "3d"])
def test_plot2d_mesh_shade_rejects_unknown(mesh, method):
    """A typo in the mode must not depend on which method you asked for."""
    with pytest.raises(ValueError, match="mesh_shade"):
        navis.plot2d(mesh, method=method, mesh_shade="glossy")


def test_plot2d_halo_width_means_the_same_everywhere(mesh, skeleton):
    """`halo=N` is a stroke width for lines, ribbons and meshes alike.

    The mesh halo goes through `_outline_under` and the other two do not, so this
    is the only thing stopping them drifting apart again.
    """
    widths = {}
    for label, obj, kw in [
        ("line", skeleton, dict(radius=False)),
        ("ribbon", skeleton, dict(radius=True)),
        ("mesh", mesh, {}),
    ]:
        fig, ax = navis.plot2d(
            obj, method="2d", view=("x", "-z"), color="b", linewidth=1, halo=4, **kw
        )
        try:
            under = min(ax.collections, key=lambda c: c.get_zorder())
            widths[label] = np.asarray(under.get_linewidth())[0]
        finally:
            plt.close(fig)

    # a 1 pt line grows by the halo; a filled outline is stroked with it
    assert widths["line"] == 1 + 4
    assert widths["ribbon"] == 4
    assert widths["mesh"] == 4


def test_plot2d_mesh_shade_multiplies_into_color_by(mesh):
    """Shading has to modulate the colours you asked for, not replace them."""
    common = dict(
        method="2d",
        view=("x", "-z"),
        color_by=mesh.vertices[:, 0],
        palette="viridis",
    )

    fig, ax = navis.plot2d(mesh, **common)
    try:
        flat = _mesh_artist(ax).get_facecolor()[:, :3]
    finally:
        plt.close(fig)

    fig, ax = navis.plot2d(mesh, mesh_shade=True, **common)
    try:
        shaded = _mesh_artist(ax).get_facecolor()[:, :3]
    finally:
        plt.close(fig)

    assert flat.shape == shaded.shape
    # the ramp survives - green faces stay greener than purple ones - but every
    # face has been moved by the light
    assert np.corrcoef(flat[:, 1], shaded[:, 1])[0, 1] > 0.7
    assert not np.allclose(flat, shaded)


def test_plot2d_mesh_halo_is_a_separate_artist(mesh):
    """The halo is the union path stroked *underneath*, not a path effect."""
    fig, ax = navis.plot2d(mesh, method="2d", view=("x", "-z"), color="b", halo=4)
    try:
        artists = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(artists) == 2
        halo, neuron = sorted(artists, key=lambda c: c.get_zorder())
        # `halo` is the stroke width, so it shows half of itself on each side -
        # the same thing it means for lines and ribbons
        assert halo.get_linewidth()[0] == 4
        assert neuron.get_linewidth()[0] == 0
        assert not artists[0].get_path_effects()
    finally:
        plt.close(fig)


def test_plot2d_mesh_zorder_keeps_meshes_over_skeletons(mesh, skeleton):
    """Without `depth_sort` a mesh stays one artist, on top, as it always was."""
    fig, ax = navis.plot2d([mesh, skeleton], method="2d", view=("x", "-z"))
    try:
        meshes = [c for c in ax.collections if isinstance(c, PathCollection)]
        lines = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(meshes) == 1
        assert meshes[0].get_zorder() >= MESH_ZORDER
        assert all(m.get_zorder() > line.get_zorder() for m in meshes for line in lines)
    finally:
        plt.close(fig)


def test_plot2d_mesh_depth_sort_shares_the_skeleton_bins(mesh, skeleton):
    """With `depth_sort` a mesh joins the same stack skeletons are binned into."""
    fig, ax = navis.plot2d(
        [mesh, skeleton], method="2d", view=("x", "-z"), depth_sort=8
    )
    try:
        meshes = [c for c in ax.collections if isinstance(c, PathCollection)]
        lines = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert 1 < len(meshes) <= 8
        z_mesh = sorted(c.get_zorder() for c in meshes)
        z_line = sorted(c.get_zorder() for c in lines)
        assert all(1 <= z < 2 for z in z_mesh)
        # they interleave rather than one lot sitting entirely above the other
        assert min(z_mesh) < max(z_line) and min(z_line) < max(z_mesh)
    finally:
        plt.close(fig)


def test_plot2d_depth_sort_direction_follows_the_view(two_skeletons):
    """Bins run along the raw depth axis, which for some views points away.

    `("x", "z")` looks down -y, so the *largest* y is nearest the viewer and has
    to end up on top; `("x", "-z")` is the other way round.
    """
    def z_vs_depth(view):
        fig, ax = navis.plot2d(two_skeletons, method="2d", view=view, depth_sort=4)
        try:
            rows = []
            for c in ax.collections:
                segs = c.get_segments() if isinstance(c, LineCollection) else []
                if len(segs):
                    # segments are polylines of differing length, so stack first
                    rows.append((c.get_zorder(), np.vstack(segs)[:, 1].mean()))
            z, depth = np.array(rows).T
            return np.corrcoef(z, depth)[0, 1]
        finally:
            plt.close(fig)

    assert z_vs_depth(("x", "-z")) > 0.5
    assert z_vs_depth(("x", "z")) < -0.5


def test_plot2d_volume_is_one_path():
    """Volumes are translucent by default, so they need the same union fill."""
    vol = navis.example_volume("LH")
    fig, ax = navis.plot2d(vol, method="2d", view=("x", "-z"))
    try:
        artists = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(artists) == 1
        assert artists[0].get_zorder() == 0
        assert 0 < _n_subpaths(artists[0]) < len(vol.faces)
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
#  depth_sort="global"                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_boxes():
    """Two cubes that overlap in xy, one 20 units deeper than the other."""
    trimesh = pytest.importorskip("trimesh")

    def box(shift):
        b = trimesh.creation.box(extents=(10, 10, 10))
        b.apply_translation(shift)
        return b

    near, far = navis.Mesh(box((4, 4, 20))), navis.Mesh(box((0, 0, 0)))
    near.id, far.id = "near", "far"
    return near, far


def _painted(fig, ax):
    ax.set_axis_off()
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3]


def _area(img, channel):
    """Pixels that are saturated in `channel` and dark in the other two."""
    other = [c for c in range(3) if c != channel]
    return int(((img[:, :, channel] > 200) & (img[:, :, other].max(axis=-1) < 60)).sum())


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_plot2d_global_ignores_input_order(two_boxes, order):
    """The whole point: who is in front is decided by depth, not draw order."""
    boxes = [two_boxes[i] for i in order]
    colors = ["r" if b.id == "near" else "b" for b in boxes]

    fig, ax = navis.plot2d(
        boxes, method="2d", view=("x", "y"), color=colors,
        depth_sort="global", figsize=(3, 3),
    )
    try:
        img = _painted(fig, ax)
        # "near" sits at the larger z, which ("x", "y") has facing the viewer
        assert _area(img, 0) > _area(img, 2)
    finally:
        plt.close(fig)


def test_plot2d_global_follows_the_view(two_boxes):
    """Flipping the view has to flip which box occludes the other."""
    near, far = two_boxes

    def red_wins(view):
        fig, ax = navis.plot2d(
            [near, far], method="2d", view=view, color=["r", "b"],
            depth_sort="global", figsize=(3, 3),
        )
        try:
            img = _painted(fig, ax)
            return _area(img, 0) > _area(img, 2)
        finally:
            plt.close(fig)

    assert red_wins(("x", "y"))  # depth axis points at the viewer
    assert not red_wins(("x", "-y"))  # ... and now away from it


def test_plot2d_global_is_one_artist_per_type(two_skeletons, mesh):
    """One merged artist per neuron type, ordered by first appearance."""
    fig, ax = navis.plot2d(
        [two_skeletons, mesh], method="2d", view=("x", "-z"), depth_sort="global"
    )
    try:
        kinds = [(type(c).__name__, round(c.get_zorder(), 3)) for c in ax.collections]
        assert kinds == [("LineCollection", 1.0), ("PolyCollection", 1.5)]
    finally:
        plt.close(fig)

    # ... and passing the mesh first puts it underneath
    fig, ax = navis.plot2d(
        [mesh, two_skeletons], method="2d", view=("x", "-z"), depth_sort="global"
    )
    try:
        kinds = [type(c).__name__ for c in ax.collections]
        assert kinds == ["PolyCollection", "LineCollection"]
    finally:
        plt.close(fig)


def test_plot2d_global_keeps_the_legend(two_skeletons, mesh):
    """One artist can carry one label, so the per-neuron entries need proxies."""
    for kwargs in (dict(depth_sort="global"), dict()):
        fig, ax = navis.plot2d(
            [two_skeletons, mesh], method="2d", view=("x", "-z"), **kwargs
        )
        try:
            assert len(ax.get_legend_handles_labels()[1]) == 3
        finally:
            plt.close(fig)


def test_plot2d_global_leaves_volumes_alone(mesh):
    """Volumes are scenery: they stay one path at the bottom, never merged."""
    vol = navis.example_volume("LH")
    fig, ax = navis.plot2d(
        [mesh, vol], method="2d", view=("x", "-z"), depth_sort="global"
    )
    try:
        paths = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(paths) == 1 and paths[0].get_zorder() == 0
    finally:
        plt.close(fig)


def test_plot2d_global_falls_back_for_halo(two_skeletons, caplog):
    """A halo needs a z-order *between* two neurons, which one artist has not."""
    fig, ax = navis.plot2d(
        two_skeletons, method="2d", view=("x", "-z"), depth_sort="global", halo=True
    )
    try:
        assert len(ax.collections) > 2  # binned, not merged
        assert "falling back" in caplog.text
    finally:
        plt.close(fig)


def test_plot2d_global_rejects_unknown_strings(skeleton):
    with pytest.raises(ValueError, match="Unknown depth_sort"):
        navis.plot2d(skeleton, method="2d", depth_sort="nonsense")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(radius=True),
        dict(radius="lw"),
        dict(taper="strahler"),
        dict(depth_coloring=True),
        dict(color_by="strahler_index", palette="viridis"),
    ],
)
def test_plot2d_global_covers_every_skeleton_renderer(two_skeletons, kwargs):
    """Every path through `_plot_skeleton_2d` has to end up in the same bucket."""
    fig, ax = navis.plot2d(
        two_skeletons, method="2d", view=("x", "-z"), depth_sort="global", **kwargs
    )
    try:
        merged = [c for c in ax.collections]
        assert len(merged) == 1
        assert len(merged[0].get_paths()) > 100
    finally:
        plt.close(fig)


def test_plot2d_mesh_3d_still_uses_trisurf(mesh):
    """None of this touches the 3d methods - they have no view plane to project to."""
    fig, ax = navis.plot2d(mesh, method="3d", mesh_shade="cel")
    try:
        assert any(isinstance(c, Poly3DCollection) for c in ax.collections)
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
#  Regressions                                                                 #
# --------------------------------------------------------------------------- #


def test_plotly_connectors_snapshot(skeleton):
    """Connectors must render with plotly's default ("lines") layout.

    `scatter3d.Line` has no `opacity` property - transparency belongs on the
    trace.
    """
    pytest.importorskip("plotly.graph_objs")

    fig = navis.plot3d(
        skeleton, connectors=True, cn_alpha=0.4, backend="plotly", inline=False
    )
    cn = [t for t in fig.data if t.name and "synapse" in str(t.name).lower()]
    assert cn
    assert all(t.opacity == 0.4 for t in cn)


def test_plotly_connector_color_alpha(skeleton):
    """An alpha channel on an explicit connector color becomes trace opacity."""
    pytest.importorskip("plotly.graph_objs")

    fig = navis.plot3d(
        skeleton,
        connectors=True,
        cn_colors=(255, 0, 0, 0.25),
        backend="plotly",
        inline=False,
    )
    cn = [t for t in fig.data if t.name and "synapse" in str(t.name).lower()]
    assert cn
    # Color must be plain `rgb(...)`; plotly silently accepts a malformed
    # 4-component `rgb(...)` string but the browser won't render it.
    for t in cn:
        assert t.line.color == "rgb(255,0,0)"
        assert t.opacity == 0.25


@pytest.mark.parametrize(
    "cn_colors", ["neuron", "red", {"pre": "red", "post": "blue"}, {"pre": "red"}]
)
def test_plot2d_cn_colors(skeleton, cn_colors):
    """All documented `cn_colors` forms work in plot2d.

    `"neuron"` used to be tested against `cn_layout` (a dict) and a dict used to
    clobber the whole per-type layout entry.
    """
    fig, ax = navis.plot2d(skeleton, connectors=True, cn_colors=cn_colors)
    plt.close(fig)


def test_color_by_neuron_property_all_backends(two_skeletons):
    """`color_by=<neuron property>` must work on every backend, not just
    matplotlib."""
    nl = two_skeletons.copy()
    for i, n in enumerate(nl):
        n.grp = "a" if i % 2 else "b"

    fig, ax = navis.plot2d(nl, color_by="grp", palette="viridis")
    plt.close(fig)

    for backend in ("plotly", "k3d"):
        pytest.importorskip(backend)
        navis.plot3d(nl, color_by="grp", palette="viridis", backend=backend,
                     inline=False)


def test_color_by_node_property_still_per_vertex(skeleton):
    """A node property must keep taking the per-vertex path even when the
    neuron also exposes it as an attribute."""
    n = skeleton.copy()
    fig, ax = navis.plot2d(n, color_by="strahler_index", palette="viridis")
    try:
        lc = _first_line_collection(ax)
        # Per-vertex coloring produces a LineCollection with >1 distinct color
        assert lc is not None
        assert len(np.unique(lc.get_colors(), axis=0)) > 1
    finally:
        plt.close(fig)


def test_color_by_requires_palette(two_skeletons):
    """`color_by` without a palette raises consistently across backends."""
    nl = two_skeletons.copy()
    for i, n in enumerate(nl):
        n.grp = i

    with pytest.raises(ValueError):
        navis.plot2d(nl, color_by="grp")

    pytest.importorskip("plotly.graph_objs")
    with pytest.raises(ValueError):
        navis.plot3d(nl, color_by="grp", backend="plotly", inline=False)


def test_radius_auto_is_per_neuron(two_skeletons):
    """`radius="auto"` must be decided per neuron - one neuron without radii
    must not force the whole list onto lines."""
    a, b = two_skeletons[0].copy(), two_skeletons[1].copy()
    a.nodes["radius"] = 0  # no usable radii -> lines
    nl = navis.NeuronList([a, b])

    fig, ax = navis.plot2d(nl, radius="auto")
    try:
        # `b` should still have been rendered with radius
        assert len(_ribbon_artists(ax)) == 1
    finally:
        plt.close(fig)

    pytest.importorskip("plotly.graph_objs")
    fig = navis.plot3d(nl, radius="auto", backend="plotly", inline=False)
    assert sum(t.type == "mesh3d" for t in fig.data) >= 1


def test_plot_flat_normalize_distance(skeleton):
    """`normalize_distance` scaled a key that doesn't exist."""
    ax, pos = navis.plot_flat(skeleton, layout="subway", normalize_distance=True)
    plt.close("all")
    assert pos


def test_plot_flat_connectors_and_highlight(skeleton):
    """Connectors and highlighted connectors must work together - the connector
    angles used to shadow the per-node angle lookup."""
    cn_ids = skeleton.connectors.connector_id.values[:3].tolist()
    ax, pos = navis.plot_flat(
        skeleton, layout="subway", connectors=True, highlight_connectors=cn_ids
    )
    plt.close("all")
    assert pos


@pytest.mark.parametrize("method", ["2d", "3d"])
def test_depth_coloring_without_neurons(method):
    """Depth coloring must not blow up when there are no neurons to normalize
    against."""
    vol = navis.example_volume("LH")
    fig, ax = navis.plot2d(vol, depth_coloring=True, method=method)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Connectors                                                                  #
# --------------------------------------------------------------------------- #

CIRCLES = {"display": "circles"}


def _cn_artists(ax):
    """The connector artists, which carry a `CN_<id>` gid."""
    return [c for c in ax.collections if (c.get_gid() or "").startswith("CN_")]


def test_connectors_go_into_one_artist(skeleton):
    """One artist per neuron, not one per type - which is what lets them mix."""
    fig, ax = navis.plot2d(
        skeleton, method="2d", view=("x", "-z"), connectors=True, cn_layout=CIRCLES
    )
    try:
        (artist,) = _cn_artists(ax)
        assert len(artist.get_offsets()) == len(skeleton.connectors)
        assert len(np.unique(artist.get_facecolor(), axis=0)) == 2  # pre and post
    finally:
        plt.close(fig)


def test_connectors_are_interleaved_not_stacked(skeleton):
    """Types must not be painted one after another.

    Drawing per type lets a rare type bury a common one wherever markers overlap:
    on this neuron 232 presynapses sit on top of 1933 postsynapses and the
    antennal lobe reads as an output region. The guard is that colours alternate
    along the draw order rather than coming in one block per type.
    """
    fig, ax = navis.plot2d(
        skeleton, method="2d", view=("x", "-z"), connectors=True, cn_layout=CIRCLES
    )
    try:
        colors = _cn_artists(ax)[0].get_facecolor()
        runs = 1 + (np.abs(np.diff(colors[:, 0])) > 1e-9).sum()
        # exactly 2 runs == sorted by type; expect many more once interleaved
        assert runs > len(colors) / 10
    finally:
        plt.close(fig)


def test_connector_order_is_deterministic(skeleton):
    """A figure that reshuffled itself on every call would be worse than the bug."""
    def draw():
        fig, ax = navis.plot2d(
            skeleton, method="2d", view=("x", "-z"), connectors=True, cn_layout=CIRCLES
        )
        try:
            return _cn_artists(ax)[0].get_offsets().copy()
        finally:
            plt.close(fig)

    assert np.array_equal(draw(), draw())


def test_cn_legend_is_per_type_not_per_neuron(two_skeletons):
    fig, ax = navis.plot2d(
        two_skeletons, method="2d", view=("x", "-z"), connectors=True, cn_legend=True
    )
    try:
        labels = ax.get_legend_handles_labels()[1]
        assert labels[-2:] == ["Presynapses", "Postsynapses"]
        # ... i.e. two entries for two neurons, not four
        assert labels.count("Presynapses") == 1
    finally:
        plt.close(fig)


def test_cn_legend_off_by_default(skeleton):
    fig, ax = navis.plot2d(skeleton, method="2d", view=("x", "-z"), connectors=True)
    try:
        assert "Presynapses" not in ax.get_legend_handles_labels()[1]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("method,expected", [("2d", LineCollection), ("3d", Line3DCollection)])
def test_cn_display_lines_draws_stalks(skeleton, method, expected):
    """`display="lines"` is the default, and used to be honoured only by plotly/k3d."""
    fig, ax = navis.plot2d(skeleton, method=method, view=("x", "-z"), connectors=True)
    try:
        (artist,) = _cn_artists(ax)
        assert isinstance(artist, expected)
        # a Line3DCollection only fills `get_segments` once it has been projected
        fig.canvas.draw()
        assert len(artist.get_segments()) == len(skeleton.connectors)
        # each stalk runs from the connector to the node it belongs to
        assert all(len(seg) == 2 for seg in artist.get_segments())
    finally:
        plt.close(fig)


def test_cn_display_lines_falls_back_without_nodes(mesh):
    """A stalk needs a node to point at, which a mesh has not."""
    fig, ax = navis.plot2d(mesh, method="2d", view=("x", "-z"), connectors=True)
    try:
        assert isinstance(_cn_artists(ax)[0], PathCollection)
    finally:
        plt.close(fig)


def test_cn_color_by_categorical(skeleton):
    fig, ax = navis.plot2d(
        skeleton, method="2d", view=("x", "-z"), connectors=True, cn_layout=CIRCLES,
        cn_color_by="roi", cn_palette="tab10", cn_legend=True,
    )
    try:
        rois = skeleton.connectors.roi
        labels = ax.get_legend_handles_labels()[1]
        assert set(labels[1:]) == set(rois.dropna().unique())
        # NaN rois get their own neutral colour but no legend entry
        n_expected = rois.nunique() + int(rois.isnull().any())
        assert len(np.unique(_cn_artists(ax)[0].get_facecolor(), axis=0)) == n_expected
    finally:
        plt.close(fig)


def test_cn_color_by_numeric_gets_a_colorbar(skeleton):
    """A ramp has no finite set of entries, so a legend cannot explain it."""
    fig, ax = navis.plot2d(
        skeleton, method="2d", view=("x", "-z"), connectors=True, cn_layout=CIRCLES,
        cn_color_by="confidence", cn_palette="viridis", cn_legend=True,
    )
    try:
        assert len(fig.axes) == 2
        assert fig.axes[-1].get_ylabel() == "confidence"
        assert len(np.unique(_cn_artists(ax)[0].get_facecolor(), axis=0)) > 2
    finally:
        plt.close(fig)


def test_cn_color_by_shares_one_scale(two_skeletons):
    """Each neuron normalising against its own range would recolour the same value."""
    fig, ax = navis.plot2d(
        two_skeletons, method="2d", view=("x", "-z"), connectors=True,
        cn_layout=CIRCLES, cn_color_by="confidence",
    )
    try:
        artists = _cn_artists(ax)
        assert len(artists) == 2
        # the same permutation the plotting code uses, so values line up with colours
        picked = []
        for artist, neuron in zip(artists, two_skeletons):
            values = neuron.connectors.confidence.values
            order = np.random.default_rng(0).permutation(len(values))
            hit = np.flatnonzero(np.round(values[order], 3) == 0.9)[0]
            picked.append(artist.get_facecolor()[hit])
        assert np.allclose(picked[0], picked[1])
    finally:
        plt.close(fig)


def test_cn_color_by_rejects_nonsense(skeleton):
    with pytest.raises(ValueError, match="no such column"):
        navis.plot2d(skeleton, method="2d", connectors=True, cn_color_by="nope")

    with pytest.raises(ValueError, match="values for"):
        navis.plot2d(skeleton, method="2d", connectors=True, cn_color_by=np.zeros(5))


@pytest.mark.parametrize("outlines", [True, "both"])
def test_volume_outlines_are_drawn_opaque(outlines):
    """A volume's alpha is a *fill* alpha - a contour has nothing to see through.

    Volumes default to 10-20% opacity, which on a 1pt line is all but invisible.
    """
    pytest.importorskip("shapely")
    vol = navis.example_volume("LH")
    vol.color = (0.2, 0.3, 0.7, 0.15)

    fig, ax = navis.plot2d(vol, method="2d", view=("x", "-z"), volume_outlines=outlines)
    try:
        (patch,) = ax.patches
        assert mcolors.to_rgba(patch.get_edgecolor())[3] == pytest.approx(1)
    finally:
        plt.close(fig)


def test_backend_is_case_insensitive(skeleton):
    """A capitalised but valid backend used to pass validation and then fall
    through to 'unknown backend'."""
    pytest.importorskip("plotly.graph_objs")
    navis.plot3d(skeleton, backend="Plotly", inline=False)

    with pytest.raises(ValueError):
        navis.plot3d(skeleton, backend="bogus")


# --------------------------------------------------------------------------- #
#  Shared settings resolution (navis/plotting/_common.py)                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "which,expected",
    [
        (True, "all"),
        ("pre", "pre"),
        ("presynapses", "pre"),
        ("post", "post"),
        ("postsynapses", "post"),
        (["pre"], "pre"),
        (("pre",), "pre"),
        (np.array(["pre"]), "pre"),
        (np.array(["pre", "post"]), "all"),
    ],
)
def test_resolve_connectors(skeleton, which, expected):
    """Every documented form of `connectors` selects the right rows.

    The multi-element array used to raise "truth value is ambiguous" in all
    three backends despite `np.ndarray` being in their isinstance checks.
    """
    settings = Matplotlib2dSettings().update_settings(connectors=which)
    got = resolve_connectors(skeleton, settings)
    if expected == "all":
        assert len(got) == len(skeleton.connectors)
    else:
        assert set(got.type.unique()) == {expected}
        assert len(got) < len(skeleton.connectors)


@pytest.mark.parametrize("which", [False, None, [], np.array([])])
def test_resolve_connectors_empty_when_not_wanted(skeleton, which):
    """Falsy `connectors` yields nothing, so callers need no guard of their own."""
    settings = Matplotlib2dSettings().update_settings(connectors=which)
    got = resolve_connectors(skeleton, settings)
    assert got.empty
    assert list(got.groupby("type")) == []


def test_resolve_connectors_without_connector_table():
    """A neuron with no connectors is handled inside the helper."""
    mesh = navis.example_neurons(1, kind="mesh")
    mesh.connectors = None
    settings = Matplotlib2dSettings().update_settings(connectors=True)
    assert resolve_connectors(mesh, settings).empty


@pytest.mark.parametrize("which", [np.array(["pre", "post"]), np.array(["pre"])])
def test_connectors_as_array_renders(skeleton, which):
    """...and the guard in front of it must not call bool() on that array."""
    pytest.importorskip("plotly.graph_objs")

    fig, ax = navis.plot2d(skeleton, connectors=which)
    plt.close(fig)
    navis.plot3d(skeleton, connectors=which, backend="plotly", inline=False)


def test_cn_color_precedence():
    """`cn_mesh_colors`/"neuron" beat `cn_colors`, which beats the layout.

    k3d ignored `cn_mesh_colors` entirely and ordered the dict check first,
    so it disagreed with the other two backends.
    """
    layout = {"pre": {"color": (1, 2, 3)}, "post": {"color": (4, 5, 6)}}
    neuron_color = (9, 9, 9)

    def resolve(cn_type, **kwargs):
        s = PlotlySettings().update_settings(**kwargs)
        return resolve_cn_color(cn_type, layout, neuron_color, s)

    # Layout default when nothing is set
    assert resolve("pre") == (1, 2, 3)
    # ... including for a type the layout doesn't know about. Note the fallback
    # is in the 0-1 range the layout itself uses (see the next test).
    assert resolve("unknown") == (0.04, 0.04, 0.04)
    # A single color applies to every type
    assert resolve("pre", cn_colors="red") == "red"
    assert resolve("post", cn_colors="red") == "red"
    # A dict may cover only some types; the rest keep the layout default
    assert resolve("pre", cn_colors={"pre": "red"}) == "red"
    assert resolve("post", cn_colors={"pre": "red"}) == (4, 5, 6)
    # "neuron" and cn_mesh_colors win over both
    assert resolve("pre", cn_colors="neuron") == neuron_color
    assert resolve("pre", cn_mesh_colors=True) == neuron_color
    assert resolve("pre", cn_colors={"pre": "red"}, cn_mesh_colors=True) == neuron_color


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_unknown_connector_type_renders(skeleton, backend):
    """A connector `type` navis has no default color for must still render.

    `config.default_connector_colors` is in the 0-1 range, so the fallback has
    to be too - a 0-255 fallback rendered fine in plotly (which rescales) but
    made matplotlib raise "'color' kwarg must be a color or sequence of colors".
    """
    n = skeleton.copy()
    cn = n.connectors.copy()
    cn.loc[cn.index[:50], "type"] = "foo"
    n.connectors = cn

    if backend == "matplotlib":
        fig, _ = navis.plot2d(n, connectors=True, method="2d")
        plt.close(fig)
    else:
        pytest.importorskip("plotly.graph_objs")
        navis.plot3d(n, connectors=True, backend="plotly", inline=False)


def test_cn_color_accepts_rgb_array():
    """A multi-element rgb array must not trip the "is cn_colors set?" check."""
    s = PlotlySettings().update_settings(cn_colors=np.array([255, 0, 0]))
    got = resolve_cn_color("pre", {"pre": {"color": (1, 2, 3)}}, (9, 9, 9), s)
    assert np.array_equal(got, [255, 0, 0])


def test_partial_cn_colors_dict_3d(skeleton):
    """A partial dict leaves other types with their default rgb tuple; the
    resulting mixed str/tuple sequence used to break matplotlib's 3d scatter."""
    fig, ax = navis.plot2d(
        skeleton, connectors=True, cn_colors={"pre": "red"}, method="3d"
    )
    plt.close(fig)


def test_soma_count_limit(skeleton):
    """A runaway soma detection is skipped rather than rendered.

    matplotlib's 2d path used to warn and then plot them anyway, and its 3d
    path used a different threshold (5) than plotly/k3d (10).
    """
    n = skeleton.copy()
    # Pick nodes that have a usable radius so only the count can reject them
    usable = n.nodes.node_id.values[n.nodes.radius.fillna(0).values > 0]
    assert len(usable) >= SOMA_COUNT_LIMIT

    # Note `_soma`: the setter only takes a scalar, but the getter (and soma
    # detection) happily hand back several.
    n._soma = usable[: SOMA_COUNT_LIMIT - 1].tolist()
    assert len(list(resolve_somata(n, (1, 0, 0), Matplotlib2dSettings()))) == (
        SOMA_COUNT_LIMIT - 1
    )

    n._soma = usable[:SOMA_COUNT_LIMIT].tolist()
    assert list(resolve_somata(n, (1, 0, 0), Matplotlib2dSettings())) == []


def test_soma_skipped_without_radius(skeleton):
    """A soma with a NaN radius has no sphere to draw."""
    n = skeleton.copy()
    n._soma = [n.nodes.node_id.values[0]]
    n.nodes.loc[n.nodes.node_id == n.soma[0], "radius"] = np.nan

    assert list(resolve_somata(n, (1, 0, 0), Matplotlib2dSettings())) == []


def test_soma_color_from_per_node_colormap(skeleton):
    """With one color per node, the soma gets *its own* node's color.

    plotly indexed a segment-ordered list of color strings (built for the
    line trace, complete with per-segment sentinels) with a node-table index,
    so it picked an unrelated color - or the black sentinel.
    """
    n = skeleton.copy()
    n._soma = [n.nodes.node_id.values[10]]

    colors = np.zeros((n.n_nodes, 3))
    colors[:, 0] = np.arange(n.n_nodes)  # unique per node
    soma_ix = np.where(n.nodes.node_id.values == n.soma[0])[0][0]

    (spec,) = resolve_somata(n, colors, Matplotlib2dSettings())
    assert np.array_equal(spec.color, colors[soma_ix])


def test_plotly_soma_color_matches_its_node(skeleton):
    """End-to-end version of the above, through `plot3d`.

    plotly built its soma color from the *line trace's* color list, which is
    ordered by segment and padded with a black sentinel per segment - so with
    per-node colors the soma came out an unrelated color.
    """
    pytest.importorskip("plotly.graph_objs")

    n = skeleton.copy()
    n._soma = [n.nodes.node_id.values[10]]
    # A (near) unique color per node, so a wrong index can't coincide
    by = np.arange(n.n_nodes)

    fig = navis.plot3d(
        n, color_by=by, palette="viridis", backend="plotly", inline=False
    )
    somata = [t for t in fig.data if t.type == "mesh3d"]
    assert len(somata) == 1

    expected = vertex_colors(
        n, by=by, alpha=None, use_alpha=False, palette="viridis",
        na="raise", color_range=255,
    )[0][10]
    got = [float(v) for v in somata[0].color[len("rgb("):-1].split(",")]
    assert np.allclose(got, expected[:3])


def test_soma_single_color_passed_through(skeleton):
    """A single neuron color is used for every soma as-is."""
    n = skeleton.copy()
    n._soma = [n.nodes.node_id.values[10]]

    (spec,) = resolve_somata(n, (0.1, 0.2, 0.3), Matplotlib2dSettings())
    assert spec.color == (0.1, 0.2, 0.3)


@pytest.mark.parametrize("norm_global", [True, False])
def test_shade_by_norm_global(two_skeletons, norm_global):
    """`norm_global=False` used to raise for both `shade_by` and `color_by`.

    `vmin`/`vmax` stayed plain lists on that path, so `vmin == vmax` was a
    single bool rather than an elementwise comparison. plotly additionally
    never forwarded the setting at all.
    """
    pytest.importorskip("plotly.graph_objs")
    nl = two_skeletons

    fig, ax = navis.plot2d(
        nl, shade_by="strahler_index", palette="viridis", norm_global=norm_global
    )
    plt.close(fig)
    fig, ax = navis.plot2d(
        nl, color_by="strahler_index", palette="viridis", norm_global=norm_global
    )
    plt.close(fig)
    navis.plot3d(
        nl,
        shade_by="strahler_index",
        palette="viridis",
        norm_global=norm_global,
        backend="plotly",
        inline=False,
    )


def test_shade_by_turns_flat_color_per_node(two_skeletons):
    """`shade_by` expands a single flat color into one rgba per node."""
    nl = two_skeletons
    settings = Matplotlib2dSettings().update_settings(
        shade_by="strahler_index", palette="viridis"
    )
    flat = [(1.0, 0.0, 0.0)] * len(nl)

    shaded = apply_shade_by(flat, nl, settings, color_range=1)

    for c, n in zip(shaded, nl):
        assert c.shape == (n.n_nodes, 4)
        assert np.allclose(c[:, :3], [1, 0, 0])   # hue is untouched
        assert len(np.unique(c[:, 3])) > 1        # alpha varies with the property


def test_shade_by_noop_without_setting(two_skeletons):
    """No `shade_by` means the colormap comes back untouched."""
    nl = two_skeletons
    flat = [(1.0, 0.0, 0.0)] * len(nl)
    assert apply_shade_by(flat, nl, Matplotlib2dSettings(), color_range=1) is flat


# --------------------------------------------------------------------------- #
#  plot3d(snapshot=True)                                                       #
# --------------------------------------------------------------------------- #


def _ortho_camera(view, position=(10, 20, 30), width=100, height=200, size=(400, 300)):
    """A pygfx ortho camera pointed the way `view` says - no canvas needed."""
    gfx = pytest.importorskip("pygfx")

    cam = gfx.PerspectiveCamera(fov=0, width=width, height=height)
    # Otherwise the projection stretches to the viewport and the numbers below
    # stop being predictable
    cam.maintain_aspect = False
    cam.set_view_size(*size)

    view_dir, up = parse_view(view)
    cam.world.position = position
    cam.world.reference_up = up
    cam.world.forward = view_dir
    return cam


@pytest.mark.parametrize("view", [("x", "y"), ("x", "-y"), ("y", "z"), ("-z", "y")])
def test_parse_view_agrees_with_plot2d(view):
    """A `view` has to mean the same thing in `plot2d` and in a render."""
    u, v, w = _view_frame(view)
    view_dir, up = parse_view(view)

    assert np.allclose(up, v)         # screen up is the second axis ...
    assert np.allclose(view_dir, -w)  # ... and the camera looks *into* the screen


@pytest.mark.parametrize(
    "string,tuple_", [("xy", ("x", "y")), ("x-z", ("x", "-z")), ("-YZ", ("-y", "z"))]
)
def test_parse_view_string_shorthand(string, tuple_):
    assert parse_view(string) == parse_view(tuple_)


@pytest.mark.parametrize("view", ["xyz", "q", ("x", "x"), ("x", "q"), ("x",)])
def test_parse_view_rejects_nonsense(view):
    with pytest.raises(ValueError):
        parse_view(view)


# The camera sits at (10, 20, 30) and shows 100 x 200 world units around it, so
# each view maps the image onto a known rectangle of the data's own space. A
# negative axis comes out as an inverted extent (left > right), which is how
# matplotlib is told to flip rather than mirror the data.
@pytest.mark.parametrize(
    "view,labels,extent",
    [
        (("x", "-y"), ("x", "y"), (-40, 60, 120, -80)),
        (("x", "y"), ("x", "y"), (-40, 60, -80, 120)),
        (("-x", "y"), ("x", "y"), (60, -40, -80, 120)),
        (("z", "-y"), ("z", "y"), (-20, 80, 120, -80)),
    ],
)
def test_image_extent_is_in_data_coordinates(view, labels, extent):
    got_extent, got_labels, _ = image_extent(_ortho_camera(view), (300, 400))

    assert got_labels == labels
    assert got_extent == pytest.approx(extent)


def test_image_extent_aspect_keeps_pixels_square():
    """Equal world-units-per-pixel in both directions -> aspect 1."""
    cam = _ortho_camera(("x", "y"), width=100, height=100, size=(400, 400))
    _, _, aspect = image_extent(cam, (400, 400))
    assert aspect == pytest.approx(1)


def test_image_extent_oblique_falls_back_to_world_units():
    """A rotated camera has no data axes to report - but stays to scale."""
    gfx = pytest.importorskip("pygfx")
    cam = _ortho_camera(("x", "-y"))
    cam.world.rotation = gfx.utils.transform.la.quat_from_euler((0.3, 0.2, 0.1))

    extent, labels, _ = image_extent(cam, (300, 400))

    assert labels == (None, None)
    # Centred on the camera, and still 100 x 200 world units across
    assert extent[0] == pytest.approx(-extent[1])
    assert extent[1] - extent[0] == pytest.approx(100)
    assert extent[3] - extent[2] == pytest.approx(200)


def test_snapshot_requires_the_octarine_backend(skeleton):
    with pytest.raises(ValueError, match="octarine"):
        navis.plot3d(skeleton, snapshot=True, backend="plotly")


@pytest.fixture(scope="module")
def offscreen():
    """Skip the tests below if this box can't render offscreen at all.

    Deliberately narrow: it probes the capability *once*, up front, so that a
    failure inside `plot3d` itself surfaces as a failure rather than as another
    "no GPU here" skip.

    """
    oc = pytest.importorskip("octarine")
    try:
        viewer = oc.Viewer(offscreen=True, size=(16, 16), show=False)
        viewer.show()
        viewer.screenshot(filename=None)
        viewer.close()
    except Exception as e:  # no GPU/adapter, missing plugin, ...
        pytest.skip(f"offscreen rendering unavailable: {e}")


@pytest.fixture
def close_figures():
    """Snapshots hold a multi-MB image each - don't leave them in `Gcf`."""
    yield
    plt.close("all")


def test_snapshot_returns_axes_with_the_image(two_skeletons, offscreen, close_figures):
    _, ax = navis.plot3d(two_skeletons, snapshot=True)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape[2] == 4  # RGBA


def test_snapshot_places_the_image_in_data_coordinates(
    skeleton, offscreen, close_figures
):
    """The projected nodes have to land on the pixels that drew them."""
    _, ax = navis.plot3d(
        skeleton, snapshot=True, view=("x", "-z"), color="k", soma=False
    )

    img = ax.images[0]
    left, right, bottom, top = img.get_extent()
    arr = np.asarray(img.get_array())
    height, width = arr.shape[:2]

    nodes = skeleton.nodes[["x", "z"]].values[::25]
    cols = ((nodes[:, 0] - left) / (right - left) * width).astype(int)
    rows = ((nodes[:, 1] - top) / (bottom - top) * height).astype(int)

    assert cols.min() >= 0 and cols.max() < width
    assert rows.min() >= 0 and rows.max() < height

    # Every sampled node should have something opaque within a couple of pixels
    hits = [
        arr[max(r - 2, 0) : r + 3, max(c - 2, 0) : c + 3, 3].max() > 0
        for r, c in zip(rows, cols)
    ]
    assert np.mean(hits) > 0.95


@pytest.mark.parametrize("pixel_ratio", [1, 2])
def test_snapshot_pixel_ratio_scales_the_image_only(
    skeleton, pixel_ratio, offscreen, close_figures
):
    """Supersampling changes how many pixels there are, not what they cover."""
    _, ax = navis.plot3d(
        skeleton, snapshot=True, size=(200, 200), pixel_ratio=pixel_ratio
    )

    img = ax.images[0]
    assert np.asarray(img.get_array()).shape[:2] == (200 * pixel_ratio,) * 2
    # The extent is in world units, so it must not move with the resolution
    assert img.get_extent() == pytest.approx(
        navis.plot3d(skeleton, snapshot=True, size=(200, 200), pixel_ratio=1)[
            1
        ].images[0].get_extent()
    )


def test_snapshot_view_flips_the_axis(skeleton, offscreen, close_figures):
    _, ax = navis.plot3d(skeleton, snapshot=True, view=("-x", "-y"))
    assert ax.xaxis_inverted()
    assert ax.yaxis_inverted()


def test_snapshot_does_not_touch_the_primary_viewer(skeleton, offscreen, close_figures):
    """A render is a throwaway - it must not become `get_viewer()`."""
    navis.close3d()
    navis.plot3d(skeleton, snapshot=True)
    assert navis.get_viewer() is None


def test_snapshot_closes_its_viewer_when_plotting_fails(skeleton, offscreen):
    """The throwaway viewer must not outlive a failure part-way through."""
    oc = pytest.importorskip("octarine")
    before = len(oc.viewer.viewers)

    with pytest.raises(Exception):
        # `color_by` without a `palette` raises once the neurons are being added
        navis.plot3d(skeleton, snapshot=True, color_by="x")

    assert len(oc.viewer.viewers) == before


def test_snapshot_draws_onto_a_given_axes(skeleton, offscreen, close_figures):
    _, ax = plt.subplots()
    _, ax2 = navis.plot3d(skeleton, snapshot=True, ax=ax)
    assert ax2 is ax
    assert len(ax.images) == 1


# --------------------------------------------------------------------------- #
#  Docstrings vs. Settings defaults                                            #
# --------------------------------------------------------------------------- #
#
# `plot2d`/`plot3d` take **kwargs and hand them to a `Settings` dataclass, so the
# docstring is the only place a default is written down for the user - and the
# only place nothing checks. These tests are that check.
#
# Two things they deliberately do NOT check, so don't read a pass as more than
# it is:
#   - only doc -> code, never code -> doc. A `Settings` field with no numpydoc
#     entry at all is invisible here (there are ~14 in `Matplotlib2dSettings`;
#     several are documented as prose sub-bullets under `connectors`).
#   - extraction keys off the literal `default=X` form, so rewording an entry to
#     "defaults to X" silently drops that parameter from the check rather than
#     failing. Keep the `default=` spelling.

# Documented parameters that never reach a `Settings` object.
_NOT_SETTINGS = {
    "x",        # the positional argument
    "backend",  # popped by plot3d before the Settings are built
}

# `linewidth` is a deliberate sentinel in `PlotlySettings` (its two consumers
# want different defaults - see the comment on the field), so the field can't
# state the default the docstring correctly describes.
_SENTINEL = {("plot3d", "linewidth")}

# numpydoc sections that end the parameter list. Anything else underlined with
# `---` still holds parameters ("Object parameters", "Figure parameters", ...).
_STOP_SECTIONS = ("Returns", "Yields", "See Also", "Examples", "Notes", "References")

_PARAM_HDR = re.compile(r"^(\w+)\s*:\s*(\S.*?)\s*$")
_KEY_VALUE = re.compile(r"default\s*=\s*(.+?)\s*$", re.I)
_MARKER = re.compile(r"([^|]+?)\s*\(default\)")


def _parse_default(token):
    """Turn a documented default (as written) into the value it denotes."""
    token = token.strip().strip("`").rstrip(".").strip()
    try:
        return ast.literal_eval(token)
    except (ValueError, SyntaxError):
        # Not a literal - prose, e.g. "3 for plotly and 1 for all others"
        return token


def _documented_defaults(func):
    """`({param: default}, [duplicated params])` from a numpydoc docstring."""
    lines = (inspect.getdoc(func) or "").splitlines()
    defaults, seen, dupes = {}, set(), []
    in_params = False
    for i, line in enumerate(lines):
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if line.strip() and nxt and set(nxt) == {"-"}:
            in_params = line.strip() not in _STOP_SECTIONS
            continue
        if not in_params:
            continue
        m = _PARAM_HDR.match(line)
        if not m:
            continue
        name, rest = m.group(1), m.group(2)
        if name in seen:
            dupes.append(name)
        seen.add(name)
        kv = _KEY_VALUE.search(rest)
        if kv:
            defaults[name] = _parse_default(kv.group(1))
        elif marker := _MARKER.search(rest):
            defaults[name] = _parse_default(marker.group(1))
    return defaults, dupes


_DOC_SETTINGS = {
    "plot2d": [Matplotlib2dSettings],
    "plot3d": [OctarineSettings, PlotlySettings, K3dSettings],
}


@pytest.mark.parametrize("fname", sorted(_DOC_SETTINGS))
def test_documented_defaults_match_settings(fname):
    """Every documented default must be the default the code actually uses.

    These drifted badly before: `plot2d` advertised `linewidth=.5` (really 1),
    `alpha=1` (really None - which is not the same thing, it forces opacity),
    `connectors=True` (really False) and `method='3d'` (really '2d');
    `plot3d` advertised `connectors=True` and `fig_autosize=False`, both wrong.
    """
    func = getattr(navis, fname)
    documented, _ = _documented_defaults(func)
    # A freshly built Settings holds exactly the field defaults, and `to_dict`
    # already drops the private keys.
    by_class = {cls.__name__: cls().to_dict() for cls in _DOC_SETTINGS[fname]}

    problems = []
    for name, doc_default in documented.items():
        if name in _NOT_SETTINGS or (fname, name) in _SENTINEL:
            continue
        actual = {
            cname: fields[name] for cname, fields in by_class.items() if name in fields
        }
        if not actual:
            problems.append(f"{name}: documented but not a setting of {fname}")
            continue
        if len({repr(v) for v in actual.values()}) > 1:
            problems.append(f"{name}: backends disagree on the default: {actual}")
            continue
        real = next(iter(actual.values()))
        if doc_default != real:
            problems.append(
                f"{name}: docstring says {doc_default!r}, actual default is {real!r}"
            )

    assert not problems, f"{fname} docstring has drifted:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("fname", sorted(_DOC_SETTINGS))
def test_no_duplicate_documented_parameters(fname):
    """`plot2d` documented `connectors` twice, with the two copies disagreeing."""
    _, dupes = _documented_defaults(getattr(navis, fname))
    assert not dupes, f"{fname} documents these parameters twice: {sorted(set(dupes))}"
