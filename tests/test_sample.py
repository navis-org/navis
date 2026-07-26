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
    return navis.TreeNeuron(nodes, units="1 um")


def box_mesh():
    import trimesh
    return navis.MeshNeuron(trimesh.creation.box(extents=[1, 1, 1]))


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
    df = navis.ml.sample_cable(n, 500, interpolate="radius", random_state=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    assert list(df.columns) == ["x", "y", "z", "radius", "source_id"]


def test_sample_cable_is_arclength_uniform():
    """Samples must be uniform along the cable, not clumped at densely-noded parts."""
    n = uneven_line()  # first 0.5 of length holds 10 of 12 edges
    df = navis.ml.sample_cable(n, 40000, random_state=0)
    frac_first_half = float((df.x < 0.5).mean())
    # arclength-uniform -> ~0.5/2.5 = 0.2; node-random would be ~0.85
    assert abs(frac_first_half - 0.2) < 0.03


def test_sample_cable_interpolates_floats():
    n = uneven_line()  # radius == x by construction
    df = navis.ml.sample_cable(n, 2000, interpolate="radius", random_state=0)
    assert np.allclose(df.radius.values, df.x.values, atol=1e-4)


def test_sample_cable_carries_categorical_from_source():
    n = uneven_line()
    df = navis.ml.sample_cable(n, 2000, interpolate="label", random_state=0)
    assert set(df.label.unique()).issubset({0, 1})
    # label is 1 exactly where the source node sits at x >= 0.5
    src_x = n.nodes.set_index("node_id").loc[df.source_id, "x"].values
    assert np.array_equal(df.label.values == 1, src_x >= 0.5)


def test_sample_cable_source_id_valid():
    n = uneven_line()
    df = navis.ml.sample_cable(n, 1000, random_state=0)
    assert set(df.source_id.unique()).issubset(set(n.nodes.node_id))


def test_sample_cable_weights_bias_toward_thick_cable():
    n = navis.example_neurons(1, kind="skeleton")
    plain = navis.ml.sample_cable(n, 5000, random_state=1)
    weighted = navis.ml.sample_cable(n, 5000, weights="radius", random_state=1)
    r = n.nodes.set_index("node_id").radius
    assert r.loc[weighted.source_id].mean() > r.loc[plain.source_id].mean()


def test_sample_cable_interpolate_true_grabs_non_structural():
    n = uneven_line()
    df = navis.ml.sample_cable(n, 50, interpolate=True, random_state=0)
    # structural columns excluded, feature columns present
    assert {"radius", "label"}.issubset(df.columns)
    assert not ({"node_id", "parent_id", "type"} & set(df.columns))


def test_sample_cable_reproducible():
    n = uneven_line()
    a = navis.ml.sample_cable(n, 1000, interpolate="radius", random_state=42)
    b = navis.ml.sample_cable(n, 1000, interpolate="radius", random_state=42)
    assert a.equals(b)


def test_sample_cable_neuronlist_returns_list():
    nl = navis.example_neurons(2, kind="skeleton")
    out = navis.ml.sample_cable(nl, 100, interpolate="radius", random_state=0)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(d, pd.DataFrame) and len(d) == 100 for d in out)


@pytest.mark.parametrize("call", [
    lambda: navis.ml.sample_cable(navis.example_neurons(1, kind="mesh"), 100),
    lambda: navis.ml.sample_cable(uneven_line(), 0),
    lambda: navis.ml.sample_cable(uneven_line(), 100, interpolate="nope"),
    lambda: navis.ml.sample_cable(uneven_line(), 100, weights="nope"),
])
def test_sample_cable_bad_args_raise(call):
    with pytest.raises((TypeError, ValueError)):
        call()


# =========================================================================== #
# sample_surface
# =========================================================================== #
@pytest.mark.parametrize("mode", ["even", "surface", "vertex"])
def test_sample_surface_shape_and_columns(mode):
    m = navis.example_neurons(1, kind="mesh")
    df = navis.ml.sample_surface(m, 1500, mode=mode, random_state=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1500
    assert list(df.columns) == ["x", "y", "z", "source_id", "face"]


@pytest.mark.parametrize("mode", ["even", "surface", "vertex"])
def test_sample_surface_source_id_valid(mode):
    m = navis.example_neurons(1, kind="mesh")
    df = navis.ml.sample_surface(m, 1500, mode=mode, random_state=0)
    assert df.source_id.min() >= 0
    assert df.source_id.max() < len(m.vertices)


def test_sample_surface_vertex_mode_returns_vertices():
    m = box_mesh()
    df = navis.ml.sample_surface(m, 500, mode="vertex", random_state=0)
    assert np.allclose(df[["x", "y", "z"]].values, m.vertices[df.source_id.values])
    assert (df.face == -1).all()


def test_sample_surface_points_lie_on_surface():
    """Area-sampled points should sit within the mesh bounding box, near a vertex."""
    m = box_mesh()
    df = navis.ml.sample_surface(m, 2000, mode="surface", random_state=0)
    pts = df[["x", "y", "z"]].values
    lo, hi = m.vertices.min(0), m.vertices.max(0)
    assert (pts >= lo - 1e-6).all() and (pts <= hi + 1e-6).all()
    # a box has side 1 -> every surface point is within 1 of its source corner
    d = np.linalg.norm(pts - m.vertices[df.source_id.values], axis=1)
    assert d.max() <= np.sqrt(2) + 1e-6


def test_sample_surface_transfers_vertex_attributes():
    m = navis.example_neurons(1, kind="mesh")
    lab = np.arange(len(m.vertices)) % 4
    df = navis.ml.sample_surface(m, 2000, attributes={"label": lab}, random_state=0)
    assert "label" in df.columns
    assert np.array_equal(df.label.values, lab[df.source_id.values])


def test_sample_surface_even_delivers_exactly_n():
    """`sample_surface_even` under-delivers; the top-up must reach exactly n."""
    m = navis.example_neurons(1, kind="mesh")
    df = navis.ml.sample_surface(m, 3000, mode="even", random_state=0)
    assert len(df) == 3000


def test_sample_surface_reproducible():
    m = navis.example_neurons(1, kind="mesh")
    a = navis.ml.sample_surface(m, 1000, random_state=7)
    b = navis.ml.sample_surface(m, 1000, random_state=7)
    assert a.equals(b)


def test_sample_surface_neuronlist_returns_list():
    nl = navis.example_neurons(2, kind="mesh")
    out = navis.ml.sample_surface(nl, 500, random_state=0)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(d, pd.DataFrame) and len(d) == 500 for d in out)


@pytest.mark.parametrize("call", [
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="skeleton"), 100),
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="mesh"), 0),
    lambda: navis.ml.sample_surface(navis.example_neurons(1, kind="mesh"), 100, mode="bad"),
    lambda: navis.ml.sample_surface(
        navis.example_neurons(1, kind="mesh"), 100, attributes={"l": np.zeros(3)}
    ),
])
def test_sample_surface_bad_args_raise(call):
    with pytest.raises((TypeError, ValueError)):
        call()
