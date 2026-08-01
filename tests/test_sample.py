"""Tests for `navis.ml.sample_cable` and `navis.ml.sample_surface`."""
import navis
import numpy as np
import pandas as pd

import pytest


# --------------------------------------------------------------------------- #
# Builders for controlled geometry
# --------------------------------------------------------------------------- #
def uneven_line():
    """A straight line along x, densely noded in [0, 0.5] then two long edges.

    radius == x (so linear interpolation must reproduce x exactly). Rooted at the
    origin; each node's parent is the previous one.
    """
    xs = np.concatenate([np.arange(0, 0.5001, 0.05), [1.5, 2.5]])  # total length 2.5
    coords = np.zeros((len(xs), 3))
    coords[:, 0] = xs
    nodes = pd.DataFrame(coords, columns=["x", "y", "z"])
    nodes["node_id"] = np.arange(len(xs))
    nodes["parent_id"] = [-1] + list(range(len(xs) - 1))
    nodes["radius"] = xs.astype(np.float32)                 # radius == x
    nodes["label"] = (xs >= 0.5).astype(np.int64)           # categorical-ish
    return navis.Skeleton(nodes, units="1 um")


def box_mesh():
    import trimesh
    return navis.Mesh(trimesh.creation.box(extents=[1, 1, 1]))


# =========================================================================== #
# Namespace: exposed via navis.ml, NOT at the top level
# =========================================================================== #
def test_samplers_are_ml_only():
    for f in ("sample_cable", "sample_surface"):
        assert callable(getattr(navis.ml, f)) and f in navis.ml.__all__
        assert not hasattr(navis, f)            # deliberately not top-level
        assert not hasattr(navis.sampling, f)   # nor lifted into navis.sampling


# =========================================================================== #
# sample_cable
# =========================================================================== #
def test_sample_cable_returns_dataframe_of_right_shape():
    n = uneven_line()
    df = navis.ml.sample_cable(n, n_points=500, interpolate="radius", random_state=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    assert list(df.columns) == ["x", "y", "z", "radius", "source_id"]


def test_sample_cable_is_arclength_uniform():
    """Samples must be uniform along the cable, not clumped at densely-noded parts."""
    n = uneven_line()  # first 0.5 of length holds 10 of 12 edges
    df = navis.ml.sample_cable(n, n_points=40000, random_state=0)
    frac_first_half = float((df.x < 0.5).mean())
    # arclength-uniform -> ~0.5/2.5 = 0.2; node-random would be ~0.85
    assert abs(frac_first_half - 0.2) < 0.03


def test_sample_cable_interpolates_floats():
    n = uneven_line()  # radius == x by construction
    df = navis.ml.sample_cable(n, n_points=2000, interpolate="radius", random_state=0)
    assert np.allclose(df.radius.values, df.x.values, atol=1e-4)


def test_sample_cable_carries_categorical_from_source():
    n = uneven_line()
    df = navis.ml.sample_cable(n, n_points=2000, interpolate="label", random_state=0)
    assert set(df.label.unique()).issubset({0, 1})
    # label is 1 exactly where the source node sits at x >= 0.5
    src_x = n.nodes.set_index("node_id").loc[df.source_id, "x"].values
    assert np.array_equal(df.label.values == 1, src_x >= 0.5)


def test_sample_cable_source_id_valid():
    n = uneven_line()
    df = navis.ml.sample_cable(n, n_points=1000, random_state=0)
    assert set(df.source_id.unique()).issubset(set(n.nodes.node_id))


def test_sample_cable_weights_bias_toward_thick_cable():
    n = navis.example_neurons(1, kind="skeleton")
    plain = navis.ml.sample_cable(n, n_points=5000, random_state=1)
    weighted = navis.ml.sample_cable(n, n_points=5000, weights="radius", random_state=1)
    r = n.nodes.set_index("node_id").radius
    assert r.loc[weighted.source_id].mean() > r.loc[plain.source_id].mean()


def test_sample_cable_interpolate_true_grabs_non_structural():
    n = uneven_line()
    df = navis.ml.sample_cable(n, n_points=50, interpolate=True, random_state=0)
    # structural columns excluded, feature columns present
    assert {"radius", "label"}.issubset(df.columns)
    assert not ({"node_id", "parent_id", "type"} & set(df.columns))


def test_sample_cable_density_and_spacing_set_count_from_length():
    """`density`/`spacing` derive the count from total cable length (here 2.5)."""
    n = uneven_line()
    d = navis.ml.sample_cable(n, density=100, random_state=0)
    assert len(d) == round(100 * 2.5)          # points per unit length -> 250
    s = navis.ml.sample_cable(n, spacing=0.01, random_state=0)
    assert len(s) == round(2.5 / 0.01)          # length / spacing -> 250
    # for a 1-D cable density and spacing are reciprocals
    assert len(d) == len(s)


def test_sample_cable_reproducible():
    n = uneven_line()
    a = navis.ml.sample_cable(n, n_points=1000, interpolate="radius", random_state=42)
    b = navis.ml.sample_cable(n, n_points=1000, interpolate="radius", random_state=42)
    assert a.equals(b)


def test_sample_cable_neuronlist_returns_list():
    nl = navis.example_neurons(2, kind="skeleton")
    out = navis.ml.sample_cable(nl, n_points=100, interpolate="radius", random_state=0)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(d, pd.DataFrame) and len(d) == 100 for d in out)


@pytest.mark.parametrize("call", [
    lambda: navis.ml.sample_cable(navis.example_neurons(1, kind="mesh"), n_points=100),
    lambda: navis.ml.sample_cable(uneven_line(), n_points=0),
    lambda: navis.ml.sample_cable(uneven_line(), n_points=100, interpolate="nope"),
    lambda: navis.ml.sample_cable(uneven_line(), n_points=100, weights="nope"),
    # n_points is keyword-only now: a positional count is a TypeError
    lambda: navis.ml.sample_cable(uneven_line(), 100),
    # mutually-exclusive knobs: none given / more than one given
    lambda: navis.ml.sample_cable(uneven_line()),
    lambda: navis.ml.sample_cable(uneven_line(), n_points=100, density=1),
    lambda: navis.ml.sample_cable(uneven_line(), density=0),
    lambda: navis.ml.sample_cable(uneven_line(), spacing=-1),
])
def test_sample_cable_bad_args_raise(call):
    with pytest.raises((TypeError, ValueError)):
        call()


# =========================================================================== #
# sample_skeleton (top-level: navis.sample_skeleton)
# =========================================================================== #
def test_sample_skeleton_n_points_fixed_count():
    n = navis.example_neurons(1, kind="skeleton")
    pts = navis.sample_skeleton(n, n_points=200)
    assert pts.shape == (200, 3)


def test_sample_skeleton_density_sets_count_from_length():
    n = uneven_line()  # total cable length 2.5
    pts = navis.sample_skeleton(n, density=100)
    assert pts.shape == (round(100 * 2.5), 3)   # points per unit length -> 250


def test_sample_skeleton_spacing_is_exact():
    """`spacing` places points exactly `spacing` apart along the (straight) cable."""
    n = uneven_line()  # straight line along x, length 2.5
    pts = navis.sample_skeleton(n, spacing=0.5)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.allclose(d, 0.5, atol=1e-6)       # exact equal spacing
    assert abs(len(pts) - 2.5 / 0.5) <= 2       # count is variable, ~ length / spacing


def test_sample_skeleton_neuronlist_returns_list():
    nl = navis.example_neurons(2, kind="skeleton")
    out = navis.sample_skeleton(nl, n_points=100)
    assert isinstance(out, list) and len(out) == 2
    assert all(p.shape == (100, 3) for p in out)


@pytest.mark.parametrize("call", [
    lambda: navis.sample_skeleton(navis.example_neurons(1, kind="mesh"), n_points=100),
    lambda: navis.sample_skeleton(uneven_line(), n_points=0),
    lambda: navis.sample_skeleton(uneven_line(), 100),        # positional -> TypeError
    lambda: navis.sample_skeleton(uneven_line()),             # none of the three
    lambda: navis.sample_skeleton(uneven_line(), n_points=10, spacing=1),  # more than one
    lambda: navis.sample_skeleton(uneven_line(), spacing=0),  # non-positive
])
def test_sample_skeleton_bad_args_raise(call):
    with pytest.raises((TypeError, ValueError)):
        call()


# =========================================================================== #
# sample_surface
# =========================================================================== #
@pytest.mark.parametrize("mode", ["even", "surface", "vertex"])
def test_sample_surface_shape_and_columns(mode):
    m = navis.example_neurons(1, kind="mesh")
    df = navis.ml.sample_surface(m, n_points=1500, mode=mode, random_state=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1500
    assert list(df.columns) == ["x", "y", "z", "source_id", "face"]


@pytest.mark.parametrize("mode", ["even", "surface", "vertex"])
def test_sample_surface_source_id_valid(mode):
    m = navis.example_neurons(1, kind="mesh")
    df = navis.ml.sample_surface(m, n_points=1500, mode=mode, random_state=0)
    assert df.source_id.min() >= 0
    assert df.source_id.max() < len(m.vertices)


def test_sample_surface_vertex_mode_returns_vertices():
    m = box_mesh()
    df = navis.ml.sample_surface(m, n_points=500, mode="vertex", random_state=0)
    assert np.allclose(df[["x", "y", "z"]].values, m.vertices[df.source_id.values])
    assert (df.face == -1).all()


def test_sample_surface_points_lie_on_surface():
    """Area-sampled points should sit within the mesh bounding box, near a vertex."""
    m = box_mesh()
    df = navis.ml.sample_surface(m, n_points=2000, mode="surface", random_state=0)
    pts = df[["x", "y", "z"]].values
    lo, hi = m.vertices.min(0), m.vertices.max(0)
    assert (pts >= lo - 1e-6).all() and (pts <= hi + 1e-6).all()
    # a box has side 1 -> every surface point is within 1 of its source corner
    d = np.linalg.norm(pts - m.vertices[df.source_id.values], axis=1)
    assert d.max() <= np.sqrt(2) + 1e-6


def test_sample_surface_transfers_vertex_attributes():
    m = navis.example_neurons(1, kind="mesh")
    lab = np.arange(len(m.vertices)) % 4
    df = navis.ml.sample_surface(m, n_points=2000, attributes={"label": lab}, random_state=0)
    assert "label" in df.columns
    assert np.array_equal(df.label.values, lab[df.source_id.values])


def test_sample_surface_even_delivers_exactly_n():
    """`sample_surface_even` under-delivers; the top-up must reach exactly n."""
    m = navis.example_neurons(1, kind="mesh")
    df = navis.ml.sample_surface(m, n_points=3000, mode="even", random_state=0)
    assert len(df) == 3000


def test_sample_surface_reproducible():
    m = navis.example_neurons(1, kind="mesh")
    a = navis.ml.sample_surface(m, n_points=1000, random_state=7)
    b = navis.ml.sample_surface(m, n_points=1000, random_state=7)
    assert a.equals(b)


def test_sample_surface_neuronlist_returns_list():
    nl = navis.example_neurons(2, kind="mesh")
    out = navis.ml.sample_surface(nl, n_points=500, random_state=0)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(d, pd.DataFrame) and len(d) == 500 for d in out)


def test_sample_surface_records_density_in_attrs():
    m = navis.example_neurons(1, kind="mesh")
    df = navis.ml.sample_surface(m, n_points=1000, random_state=0)
    assert set(("area", "density", "spacing")).issubset(df.attrs)
    # density = points / area; spacing = sqrt(area / points)
    assert np.isclose(df.attrs["density"], 1000 / df.attrs["area"])
    assert np.isclose(df.attrs["spacing"], np.sqrt(df.attrs["area"] / 1000))


@pytest.mark.parametrize("mode", ["even", "surface"])
def test_sample_surface_density_sets_count_from_area(mode):
    """`density` fixes points-per-area, so the count scales with surface area."""
    m = navis.example_neurons(1, kind="mesh")
    dens = 1e-5
    df = navis.ml.sample_surface(m, density=dens, mode=mode, random_state=0)
    expected = round(dens * df.attrs["area"])
    assert len(df) == expected
    # a mesh scaled up 2x has 4x the area -> ~4x the points at the same density
    big = m * 2
    df_big = navis.ml.sample_surface(big, density=dens, mode=mode, random_state=0)
    assert len(df_big) > len(df) * 3


def test_sample_surface_spacing_enforces_min_distance():
    """`spacing` thins via Poisson-disk rejection: no pair closer than `spacing`."""
    m = box_mesh()  # unit box, area 6
    s = 0.1
    df = navis.ml.sample_surface(m, spacing=s, random_state=0)
    assert len(df) > 0
    pts = df[["x", "y", "z"]].values
    # nearest-neighbour distance must respect the requested minimum spacing
    assert navis.ml.estimate_spacing(pts, aggregate="min") >= s * 0.95
    # spacing is not topped up, so it under-delivers vs the area/count budget
    assert len(df) <= np.ceil(df.attrs["area"] / (3 * s ** 2))


def test_sample_surface_density_and_spacing_are_scale_aware():
    """Same density -> same neighbourhood scale regardless of mesh size."""
    m = box_mesh()
    small = navis.ml.sample_surface(m, density=200.0, random_state=0)
    big = navis.ml.sample_surface(m * 3, density=200.0, random_state=0)
    # constant density -> nearly constant local spacing across the two meshes
    ss = navis.ml.estimate_spacing(small[["x", "y", "z"]].values)
    sb = navis.ml.estimate_spacing(big[["x", "y", "z"]].values)
    assert np.isclose(ss, sb, rtol=0.25)


@pytest.mark.parametrize("call", [
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="skeleton"), n_points=100),
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="mesh"), n_points=0),
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="mesh"), n_points=100, mode="bad"),
    lambda: navis.ml.sample_surface(
        navis.example_neurons(1, kind="mesh"), n_points=100, attributes={"l": np.zeros(3)}
    ),
    # n_points is keyword-only now: a positional count is a TypeError
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="mesh"), 100),
    # mutually-exclusive knobs: none given...
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="mesh")),
    # ...or more than one given
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="mesh"), n_points=100, density=1e-5),
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="mesh"), density=1e-5, spacing=1),
    # spacing only makes sense for the (Poisson-disk) "even" mode
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="mesh"), spacing=1, mode="surface"),
    # non-positive density/spacing
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="mesh"), density=0),
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="mesh"), spacing=-1),
])
def test_sample_surface_bad_args_raise(call):
    with pytest.raises((TypeError, ValueError)):
        call()


# =========================================================================== #
# estimate_spacing
# =========================================================================== #
def test_estimate_spacing_on_unit_grid():
    # points on an integer grid -> nearest-neighbour spacing is exactly 1
    g = np.stack(np.meshgrid(np.arange(8), np.arange(8), [0]), -1).reshape(-1, 3)
    assert np.isclose(navis.ml.estimate_spacing(g.astype(float)), 1.0)


def test_estimate_spacing_aggregates_and_validates():
    pts = np.random.RandomState(0).rand(200, 3)
    for agg in ("median", "mean", "min"):
        assert navis.ml.estimate_spacing(pts, aggregate=agg) > 0
    with pytest.raises(ValueError):
        navis.ml.estimate_spacing(pts[:1])          # need >= 2 points
    with pytest.raises(AssertionError):
        navis.ml.estimate_spacing(pts, aggregate="nope")
