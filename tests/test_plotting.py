import matplotlib

# Use a headless backend so tests don't try to open windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import numpy as np
import pytest

import navis
from navis.plotting.colors import vertex_colors
from navis.plotting.dd import _collapse_colored_segments, _colors_are_categorical


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
