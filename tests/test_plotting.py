import matplotlib

# Use a headless backend so tests don't try to open windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
from navis.plotting.dd import _collapse_colored_segments, _colors_are_categorical
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
