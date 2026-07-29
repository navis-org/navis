import matplotlib

# Use a headless backend so tests don't try to open windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import numpy as np
import pytest

import navis
from navis.plotting.colors import vertex_colors
from navis.plotting.dd import _collapse_colored_segments, _colors_are_categorical


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
#  Regressions                                                                 #
# --------------------------------------------------------------------------- #


def test_plotly_connectors_render(skeleton):
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
    matplotlib/vispy."""
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
        # `b` should still have been rendered as a tube mesh
        assert sum(isinstance(c, PolyCollection) for c in ax.collections) == 1
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


def test_backend_is_case_insensitive(skeleton):
    """A capitalised but valid backend used to pass validation and then fall
    through to 'unknown backend'."""
    pytest.importorskip("plotly.graph_objs")
    navis.plot3d(skeleton, backend="Plotly", inline=False)

    with pytest.raises(ValueError):
        navis.plot3d(skeleton, backend="bogus")
