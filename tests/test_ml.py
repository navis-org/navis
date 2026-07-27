"""Tests for `navis.ml` - machine-learning input helpers."""
import navis
import numpy as np
import pandas as pd

import pytest


@pytest.fixture
def skeleton():
    return navis.example_neurons(1, kind="skeleton")


def _coords(n):
    if isinstance(n, navis.TreeNeuron):
        return n.nodes[["x", "y", "z"]].values
    if isinstance(n, navis.MeshNeuron):
        return n.vertices
    return n.points  # Dotprops


def _rms(co):
    return float(np.sqrt((co ** 2).sum(axis=1).mean()))


# --------------------------------------------------------------------------- #
# Namespace
# --------------------------------------------------------------------------- #
def test_lives_in_ml_namespace_only():
    assert callable(navis.ml.normalize_neuron)
    assert "normalize_neuron" in navis.ml.__all__
    # Deliberately NOT lifted to the top level.
    assert not hasattr(navis, "normalize_neuron")


# --------------------------------------------------------------------------- #
# Core canonicalization invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["skeleton", "mesh"])
def test_default_is_centered_oriented_unit_rms(kind):
    n = navis.example_neurons(1, kind=kind)
    norm = navis.ml.normalize_neuron(n)
    co = _coords(norm)
    assert np.allclose(co.mean(axis=0), 0, atol=1e-5)          # centered
    assert np.isclose(_rms(co), 1.0, atol=1e-6)                # unit RMS radius
    # principal axes aligned -> variance descends along x, y, z
    v = co.var(axis=0)
    assert v[0] >= v[1] >= v[2]


def test_input_is_not_modified(skeleton):
    before = skeleton.nodes[["x", "y", "z"]].values.copy()
    r_before = skeleton.nodes.radius.values.copy()
    navis.ml.normalize_neuron(skeleton)
    assert np.array_equal(skeleton.nodes[["x", "y", "z"]].values, before)
    assert np.array_equal(skeleton.nodes.radius.values, r_before)


def test_return_matrix_is_the_applied_transform(skeleton):
    norm, M = navis.ml.normalize_neuron(skeleton, return_matrix=True)
    assert M.shape == (4, 4)
    orig = skeleton.nodes[["x", "y", "z"]].values
    got = (M @ np.c_[orig, np.ones(len(orig))].T).T[:, :3]
    assert np.allclose(got, norm.nodes[["x", "y", "z"]].values, atol=1e-6)
    # ... and it inverts cleanly back to the original frame.
    back = (np.linalg.inv(M) @ np.c_[got, np.ones(len(got))].T).T[:, :3]
    assert np.allclose(back, orig, atol=1e-4)


def test_orientation_is_pose_invariant(skeleton):
    """Re-orienting the input must not change the canonical result."""
    base = navis.ml.normalize_neuron(skeleton).nodes[["x", "y", "z"]].values
    theta = 0.9
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1.0]])
    M = np.eye(4)
    M[:3, :3] = Rz
    M[:3, 3] = [123, -45, 67]
    moved = navis.xform(skeleton, navis.transforms.AffineTransform(M))
    again = navis.ml.normalize_neuron(moved).nodes[["x", "y", "z"]].values
    # Sort per-axis: identical shape, just row order may differ by node identity.
    assert np.allclose(np.sort(base, axis=0), np.sort(again, axis=0), atol=1e-3)


def test_handedness_preserved_not_mirrored(skeleton):
    """A mirrored neuron must not canonicalize to the same thing as the original."""
    base = navis.ml.normalize_neuron(skeleton).nodes[["x", "y", "z"]].values
    mirror = skeleton.copy()
    co = mirror.nodes[["x", "y", "z"]].values.copy()
    co[:, 0] *= -1
    mirror.nodes[["x", "y", "z"]] = co
    mir = navis.ml.normalize_neuron(mirror).nodes[["x", "y", "z"]].values
    assert not np.allclose(np.sort(base, axis=0), np.sort(mir, axis=0), atol=1e-3)


# --------------------------------------------------------------------------- #
# Knobs
# --------------------------------------------------------------------------- #
def test_center_modes(skeleton):
    co = skeleton.nodes[["x", "y", "z"]].values
    # explicit point
    shifted = navis.ml.normalize_neuron(
        skeleton, center=[10, 20, 30], rotate=None, scale=None
    ).nodes[["x", "y", "z"]].values
    assert np.allclose(shifted, co - [10, 20, 30], atol=1e-5)
    # soma
    soma = navis.ml.normalize_neuron(skeleton, center="soma", rotate=None, scale=None)
    sid = np.atleast_1d(skeleton.soma)
    soma_co = soma.nodes.loc[soma.nodes.node_id.isin(sid), ["x", "y", "z"]].values
    assert np.allclose(soma_co.mean(axis=0), 0, atol=1e-5)


def test_center_soma_without_soma_raises(skeleton):
    skeleton = skeleton.copy()
    skeleton.soma = None
    with pytest.raises(ValueError):
        navis.ml.normalize_neuron(skeleton, center="soma")


@pytest.mark.parametrize("scale,check", [
    ("rms", lambda co: np.isclose(_rms(co), 1.0, atol=1e-6)),
    ("max", lambda co: np.isclose(np.sqrt((co ** 2).sum(1)).max(), 1.0, atol=1e-6)),
    ("extent", lambda co: np.isclose((co.max(0) - co.min(0)).max(), 1.0, atol=1e-6)),
])
def test_scale_modes(skeleton, scale, check):
    co = navis.ml.normalize_neuron(skeleton, scale=scale).nodes[["x", "y", "z"]].values
    assert check(co)


def test_scale_none_leaves_radius_and_units(skeleton):
    r0 = skeleton.nodes.radius.values.copy()
    norm = navis.ml.normalize_neuron(skeleton, scale=None)
    assert np.allclose(norm.nodes.radius.values, r0)
    assert str(norm.units) == str(skeleton.units)


def test_radius_scales_by_exact_factor(skeleton):
    """Radii must scale by the *exact* coordinate factor (not xform's power-of-10)."""
    r0 = skeleton.nodes.radius.values.copy()
    norm, M = navis.ml.normalize_neuron(skeleton, return_matrix=True)
    s = float(abs(np.linalg.det(M[:3, :3])) ** (1 / 3))
    assert np.allclose(norm.nodes.radius.values, r0 * s, rtol=1e-9)


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
def test_dotprops_tangent_vectors_stay_unit():
    dp = navis.make_dotprops(navis.example_neurons(1, kind="skeleton"), k=5)
    nd = navis.ml.normalize_neuron(dp)
    assert np.allclose(nd.points.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(np.linalg.norm(nd.vect, axis=1), 1, atol=1e-4)


def test_mesh_faces_preserved():
    m = navis.example_neurons(1, kind="mesh")
    nm = navis.ml.normalize_neuron(m)
    assert nm.faces.shape == m.faces.shape
    assert np.array_equal(nm.faces, m.faces)


def test_neuronlist_normalized_independently():
    nl = navis.example_neurons(3, kind="skeleton")
    out = navis.ml.normalize_neuron(nl)
    assert isinstance(out, navis.NeuronList) and len(out) == 3
    for n in out:
        assert np.allclose(n.nodes[["x", "y", "z"]].values.mean(axis=0), 0, atol=1e-5)
    # with matrices
    out2, mats = navis.ml.normalize_neuron(nl, return_matrix=True)
    assert isinstance(out2, navis.NeuronList) and len(mats) == 3
    assert all(m.shape == (4, 4) for m in mats)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [
    {"rotate": "nope"},
    {"scale": "nope"},
    {"center": "nope"},
    {"center": [1, 2]},        # coordinate must have 3 values
])
def test_bad_arguments_raise(skeleton, kwargs):
    with pytest.raises(ValueError):
        navis.ml.normalize_neuron(skeleton, **kwargs)


def test_unsupported_type_raises():
    with pytest.raises(TypeError):
        navis.ml.normalize_neuron(navis.example_volume("LH"))


# =========================================================================== #
# Augmentation suite
# =========================================================================== #
@pytest.fixture
def mesh():
    return navis.example_neurons(1, kind="mesh")


@pytest.fixture
def dotprops(skeleton):
    return navis.make_dotprops(skeleton, k=5)


AUGMENTORS = [
    "jitter_neuron", "rotate_neuron", "translate_neuron", "scale_neuron",
    "warp_neuron", "drop_nodes", "augment_neuron",
]


def test_augmentors_in_ml_namespace_only():
    for f in AUGMENTORS:
        assert f in navis.ml.__all__ and callable(getattr(navis.ml, f))
        assert not hasattr(navis, f)   # not lifted to top level


# --------------------------------------------------------------------------- #
# jitter
# --------------------------------------------------------------------------- #
def test_jitter_moves_points_but_keeps_topology(skeleton):
    before = skeleton.nodes[["x", "y", "z"]].values.copy()
    j = navis.ml.jitter_neuron(skeleton, sigma=50, random_state=0)
    disp = np.linalg.norm(j.nodes[["x", "y", "z"]].values - before, axis=1)
    assert (disp > 0).all()
    assert np.array_equal(j.nodes.parent_id.values, skeleton.nodes.parent_id.values)
    assert np.allclose(j.nodes.radius.values, skeleton.nodes.radius.values)  # jitter != scale
    assert np.array_equal(skeleton.nodes[["x", "y", "z"]].values, before)    # input untouched


def test_jitter_per_axis_sigma(skeleton):
    before = skeleton.nodes[["x", "y", "z"]].values.copy()
    j = navis.ml.jitter_neuron(skeleton, sigma=[0, 0, 100], random_state=1)
    d = j.nodes[["x", "y", "z"]].values - before
    assert np.allclose(d[:, :2], 0, atol=1e-9)   # x, y untouched
    assert d[:, 2].std() > 0                      # z jittered


def test_jitter_reproducible(skeleton):
    a = navis.ml.jitter_neuron(skeleton, sigma=50, random_state=7)
    b = navis.ml.jitter_neuron(skeleton, sigma=50, random_state=7)
    assert np.array_equal(a.nodes[["x", "y", "z"]].values, b.nodes[["x", "y", "z"]].values)


def test_jitter_bad_sigma_raises(skeleton):
    with pytest.raises(ValueError):
        navis.ml.jitter_neuron(skeleton, sigma=-1)


def test_jitter_large_sigma_keeps_radius_and_units(skeleton):
    """A large sigma must not let xform's power-of-10 guess rescale radius/units."""
    r0 = skeleton.nodes.radius.values.copy()
    u0 = str(skeleton.units)
    # sigma comparable to the neuron's extent - enough to trip xform's guess
    j = navis.ml.jitter_neuron(skeleton, sigma=100000, random_state=0)
    assert np.allclose(j.nodes.radius.values, r0)   # jitter moves points, not sizes
    assert str(j.units) == u0


# --------------------------------------------------------------------------- #
# rotate
# --------------------------------------------------------------------------- #
def test_rotate_is_rigid_about_centroid(skeleton):
    from scipy.spatial.distance import pdist
    co0 = skeleton.nodes[["x", "y", "z"]].values.astype(float)
    r = navis.ml.rotate_neuron(skeleton, random_state=0)
    rc = r.nodes[["x", "y", "z"]].values.astype(float)
    # centroid preserved (rotation about centroid)
    assert np.allclose(rc.mean(0), co0.mean(0), rtol=1e-4)
    # pairwise distances preserved (rigid) - loose tol for float32 storage
    idx = np.random.default_rng(0).integers(0, len(co0), 60)
    assert np.allclose(pdist(co0[idx]), pdist(rc[idx]), rtol=1e-4)
    # radius & units untouched by a pure rotation
    assert np.allclose(r.nodes.radius.values, skeleton.nodes.radius.values)
    assert str(r.units) == str(skeleton.units)


def test_rotate_axis_constrained(skeleton):
    co0 = skeleton.nodes[["x", "y", "z"]].values.astype(float)
    r = navis.ml.rotate_neuron(skeleton, axis="z", max_angle=45, random_state=0)
    rc = r.nodes[["x", "y", "z"]].values.astype(float)
    # a rotation about z leaves the z coordinate unchanged
    assert np.allclose(rc[:, 2], co0[:, 2], rtol=1e-4, atol=1e-2)


def test_rotate_bad_axis_raises(skeleton):
    with pytest.raises(ValueError):
        navis.ml.rotate_neuron(skeleton, axis="w")


# --------------------------------------------------------------------------- #
# translate
# --------------------------------------------------------------------------- #
def test_translate_is_uniform_shift(skeleton):
    co0 = skeleton.nodes[["x", "y", "z"]].values.astype(float)
    t = navis.ml.translate_neuron(skeleton, magnitude=1000, random_state=0)
    disp = t.nodes[["x", "y", "z"]].values.astype(float) - co0
    # every node moved by the SAME offset (a pure translation; std across nodes is
    # zero bar float32 storage rounding) ...
    assert np.allclose(disp.std(axis=0), 0, atol=0.5)
    # ... a non-zero shift within the requested per-axis bound ...
    assert np.linalg.norm(disp.mean(axis=0)) > 0
    assert np.all(np.abs(disp.mean(axis=0)) <= 1000 + 0.5)
    # ... leaving topology, radius and units untouched.
    assert np.array_equal(t.nodes.parent_id.values, skeleton.nodes.parent_id.values)
    assert np.allclose(t.nodes.radius.values, skeleton.nodes.radius.values)
    assert str(t.units) == str(skeleton.units)
    assert np.array_equal(skeleton.nodes[["x", "y", "z"]].values, co0)   # input untouched


def test_translate_per_axis_magnitude(skeleton):
    co0 = skeleton.nodes[["x", "y", "z"]].values.astype(float)
    t = navis.ml.translate_neuron(skeleton, magnitude=[0, 0, 500], random_state=1)
    d = t.nodes[["x", "y", "z"]].values.astype(float) - co0
    assert np.allclose(d[:, :2], 0, atol=1e-2)   # x, y not moved
    assert abs(d[:, 2].mean()) > 0                # z shifted


def test_translate_reproducible(skeleton):
    a = navis.ml.translate_neuron(skeleton, magnitude=1000, random_state=7)
    b = navis.ml.translate_neuron(skeleton, magnitude=1000, random_state=7)
    assert np.array_equal(a.nodes[["x", "y", "z"]].values, b.nodes[["x", "y", "z"]].values)


def test_translate_bad_magnitude_raises(skeleton):
    with pytest.raises(ValueError):
        navis.ml.translate_neuron(skeleton, magnitude=-1)


# --------------------------------------------------------------------------- #
# scale
# --------------------------------------------------------------------------- #
def test_scale_scales_coords_and_radius_exactly(skeleton):
    co0 = skeleton.nodes[["x", "y", "z"]].values.astype(float)
    r0 = skeleton.nodes.radius.values.copy()
    c = co0.mean(0)
    s = navis.ml.scale_neuron(skeleton, scale_range=(2.0, 2.0), random_state=0)  # exactly 2x
    sc = s.nodes[["x", "y", "z"]].values.astype(float)
    assert np.allclose(sc - c, 2 * (co0 - c), rtol=1e-5)          # scaled about centroid
    assert np.allclose(s.nodes.radius.values, r0 * 2, rtol=1e-5)   # radius scaled exactly (not power of 10)


def test_scale_anisotropic_renormalizes_dotprops(dotprops):
    s = navis.ml.scale_neuron(dotprops, scale_range=(0.5, 2.0), anisotropic=True, random_state=3)
    assert np.allclose(np.linalg.norm(s.vect, axis=1), 1, atol=1e-4)


def test_scale_bad_range_raises(skeleton):
    with pytest.raises(ValueError):
        navis.ml.scale_neuron(skeleton, scale_range=(0, 1))


# --------------------------------------------------------------------------- #
# warp
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["skeleton", "mesh"])
def test_warp_deforms_and_keeps_topology(kind):
    n = navis.example_neurons(1, kind=kind)
    before = _coords(n).astype(float).copy()
    w = navis.ml.warp_neuron(n, sigma=0.5, magnitude=0.05, random_state=0)
    after = _coords(w).astype(float)
    diag = np.linalg.norm(before.max(0) - before.min(0))
    disp = np.linalg.norm(after - before, axis=1)
    assert disp.mean() > 0
    assert disp.max() < diag                        # a gentle warp, not a teleport
    if kind == "skeleton":
        assert np.array_equal(w.nodes.parent_id.values, n.nodes.parent_id.values)
    else:
        assert np.array_equal(w.faces, n.faces)     # mesh connectivity preserved


def test_warp_reproducible_and_grid_validated(skeleton):
    a = navis.ml.warp_neuron(skeleton, random_state=0).nodes[["x", "y", "z"]].values
    b = navis.ml.warp_neuron(skeleton, random_state=0).nodes[["x", "y", "z"]].values
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        navis.ml.warp_neuron(skeleton, grid=1)


def test_warp_strong_keeps_radius_and_units(skeleton):
    """A strong warp must not let xform's power-of-10 guess rescale radius/units."""
    r0 = skeleton.nodes.radius.values.copy()
    u0 = str(skeleton.units)
    w = navis.ml.warp_neuron(skeleton, magnitude=10.0, random_state=0)
    assert np.allclose(w.nodes.radius.values, r0)
    assert str(w.units) == u0


# --------------------------------------------------------------------------- #
# drop_nodes
# --------------------------------------------------------------------------- #
def test_drop_nodes_skeleton_stays_connected(skeleton):
    d = navis.ml.drop_nodes(skeleton, fraction=0.2, random_state=0)
    assert d.n_nodes < skeleton.n_nodes
    assert d.n_trees == skeleton.n_trees                       # still connected
    # branch points and root are never dropped -> branching structure preserved
    assert set(skeleton.branch_points.node_id).issubset(set(d.nodes.node_id))
    assert set(np.atleast_1d(skeleton.root)).issubset(set(d.nodes.node_id))


def test_drop_nodes_dotprops_drops_points(dotprops):
    d = navis.ml.drop_nodes(dotprops, fraction=0.3, random_state=0)
    assert len(d.points) < len(dotprops.points)
    assert d.vect.shape[0] == len(d.points)                    # vect kept in sync


def test_drop_nodes_mesh_raises(mesh):
    with pytest.raises(TypeError):
        navis.ml.drop_nodes(mesh, fraction=0.1)


def test_drop_nodes_zero_fraction_is_noop(skeleton):
    d = navis.ml.drop_nodes(skeleton, fraction=0.0)
    assert d.n_nodes == skeleton.n_nodes


def test_drop_nodes_bad_fraction_raises(skeleton):
    with pytest.raises(ValueError):
        navis.ml.drop_nodes(skeleton, fraction=1.0)


def test_drop_nodes_reattaches_connectors(skeleton):
    """Connectors on dropped nodes must be reattached, not left dangling."""
    assert skeleton.has_connectors
    d = navis.ml.drop_nodes(skeleton, fraction=0.5, random_state=0)
    assert len(d.connectors) == len(skeleton.connectors)     # none lost
    live = set(d.nodes.node_id.values)
    assert d.connectors.node_id.isin(live).all()             # none dangling


# --------------------------------------------------------------------------- #
# augment_neuron (orchestrator)
# --------------------------------------------------------------------------- #
def test_augment_pipeline_runs_and_is_reproducible(skeleton):
    kw = dict(drop=0.1, warp=0.03, rotate=True, scale=(0.8, 1.25), translate=200, jitter=20)
    a = navis.ml.augment_neuron(skeleton, random_state=0, **kw)
    b = navis.ml.augment_neuron(skeleton, random_state=0, **kw)
    assert isinstance(a, navis.TreeNeuron)
    assert a.n_nodes < skeleton.n_nodes                        # drop happened
    assert a.n_nodes == b.n_nodes
    assert np.array_equal(a.nodes[["x", "y", "z"]].values, b.nodes[["x", "y", "z"]].values)


def test_augment_none_skips_everything(skeleton):
    out = navis.ml.augment_neuron(skeleton)   # all steps None
    assert out.n_nodes == skeleton.n_nodes


def test_augment_dict_config(skeleton):
    out = navis.ml.augment_neuron(
        skeleton, rotate={"axis": "z", "max_angle": 10}, random_state=0
    )
    assert out.n_nodes == skeleton.n_nodes


def test_augment_false_skips_step_like_none(skeleton):
    """`False` must skip a step exactly like `None` - same output, no RNG draw."""
    # `rotate` is documented as accepting a bool; `False` should skip, not run a
    # zero-angle no-op that still consumes the RNG (which would shift later draws).
    none_ = navis.ml.augment_neuron(
        skeleton, rotate=None, jitter=10, random_state=0
    ).nodes[["x", "y", "z"]].values
    false_ = navis.ml.augment_neuron(
        skeleton, rotate=False, jitter=10, random_state=0
    ).nodes[["x", "y", "z"]].values
    assert np.array_equal(none_, false_)
    # `scale=False` must skip rather than crash on `lo, hi = False`.
    out = navis.ml.augment_neuron(skeleton, scale=False, random_state=0)
    assert out.n_nodes == skeleton.n_nodes


# --------------------------------------------------------------------------- #
# NeuronList
# --------------------------------------------------------------------------- #
def test_neuronlist_augments_independently():
    nl = navis.example_neurons(2, kind="skeleton")
    out = navis.ml.rotate_neuron(nl, random_state=0)
    assert isinstance(out, navis.NeuronList) and len(out) == 2
    # two different neurons -> two different rotations (independent draws)
    assert not np.allclose(
        out[0].nodes[["x", "y", "z"]].values.mean(0) - nl[0].nodes[["x", "y", "z"]].values.mean(0),
        out[1].nodes[["x", "y", "z"]].values.mean(0) - nl[1].nodes[["x", "y", "z"]].values.mean(0),
    )


# =========================================================================== #
# chunk_neuron
# =========================================================================== #
from navis.ml import chunk_neuron


@pytest.mark.parametrize("connected", [True, False])
def test_chunk_partition_is_disjoint_and_complete(skeleton, connected):
    parts = chunk_neuron(skeleton, size=50, mode="partition",
                         connected=connected, undersized="keep")
    flat = np.concatenate(parts)
    assert (flat >= 0).all() and flat.max() < skeleton.n_nodes   # valid indices
    uniq = np.unique(flat)
    assert len(flat) == len(uniq)                                # disjoint
    assert len(uniq) == skeleton.n_nodes                         # complete


def test_chunk_pad_gives_uniform_length_and_pad_token(skeleton):
    parts = chunk_neuron(skeleton, size=64, mode="partition", undersized="pad")
    assert {len(c) for c in parts} == {64}                       # stackable
    assert np.stack(parts).shape == (len(parts), 64)
    # padded slots carry the pad token, never a real (>=0) index
    assert any((c == -1).any() for c in parts)


@pytest.mark.parametrize("connected", [True, False])
def test_chunk_cover_covers_every_node(skeleton, connected):
    cov = chunk_neuron(skeleton, size=50, mode="cover",
                       connected=connected, undersized="keep")
    assert set(np.concatenate(cov)) == set(range(skeleton.n_nodes))


@pytest.mark.parametrize("mode", ["random", "spaced"])
def test_chunk_k_and_reproducibility(skeleton, mode):
    a = chunk_neuron(skeleton, size=40, mode=mode, k=12, random_state=0)
    b = chunk_neuron(skeleton, size=40, mode=mode, k=12, random_state=0)
    assert len(a) == 12
    assert all(np.array_equal(x, y) for x, y in zip(a, b))       # seeded -> identical


def test_chunk_spaced_fragments_are_connected(skeleton):
    """connected=True fragments must be connected subgraphs of the arbor."""
    import networkx as nx
    g = skeleton.graph.to_undirected()
    ids = skeleton.nodes.node_id.values
    for frag in chunk_neuron(skeleton, size=40, mode="spaced", connected=True,
                             k=15, undersized="discard"):
        assert nx.is_connected(g.subgraph(ids[frag]))


def test_chunk_euclidean_partition_packs_tight(skeleton):
    """connected=False partition wastes at most `size - 1` nodes."""
    tight = chunk_neuron(skeleton, size=50, mode="partition",
                         connected=False, undersized="discard")
    assert skeleton.n_nodes - sum(len(c) for c in tight) < 50


def test_chunk_mesh_indices_address_vertices():
    m = navis.example_neurons(1, kind="mesh")
    parts = chunk_neuron(m, size=100, mode="partition", undersized="keep")
    flat = np.concatenate(parts)
    assert flat.max() < len(m.vertices) and len(np.unique(flat)) == len(m.vertices)


def test_chunk_bad_mode_raises(skeleton):
    with pytest.raises(ValueError):
        chunk_neuron(skeleton, size=50, mode="nope")


# =========================================================================== #
# sample_patches (resample at a density, then tile into fixed-size patches)
# =========================================================================== #
from navis.ml import sample_patches


# density is points-per-area (mesh) vs points-per-length (skeleton) - different scales
@pytest.mark.parametrize("kind,density", [("mesh", 1e-5), ("skeleton", 1e-2)])
def test_sample_patches_returns_long_frame_with_chunk_id(kind, density):
    n = navis.example_neurons(1, kind=kind)
    df = sample_patches(n, n_points=64, density=density, mode="spaced", random_state=0)
    assert isinstance(df, pd.DataFrame)
    assert "chunk_id" in df.columns and {"x", "y", "z", "source_id"}.issubset(df.columns)
    # spaced patches are full-size (grow returns the n_points nearest points)
    assert df.groupby("chunk_id").size().unique().tolist() == [64]


def test_sample_patches_density_fixes_physical_patch_extent():
    """n_points + density pin each patch's radius; it's ~constant across patches."""
    m = navis.example_neurons(1, kind="mesh")
    df = sample_patches(m, n_points=64, density=1e-5, mode="spaced", random_state=0)
    radii = []
    for _, g in df.groupby("chunk_id"):
        c = g[["x", "y", "z"]].values
        radii.append(np.linalg.norm(c - c.mean(0), axis=1).max())
    radii = np.array(radii)
    # coefficient of variation is small -> the resampled cloud is uniform-density
    assert radii.std() / radii.mean() < 0.5


def test_sample_patches_spacing_and_attribute_passthrough():
    # `spacing` (mesh even mode) and per-vertex attribute transfer both flow through
    m = navis.example_neurons(1, kind="mesh")
    lab = np.arange(len(m.vertices)) % 3
    df = sample_patches(m, n_points=32, spacing=400, attributes={"label": lab},
                        mode="partition", random_state=0)
    assert "label" in df.columns
    assert set(np.unique(df.label.dropna())).issubset({0, 1, 2})


def test_sample_patches_cover_overlap_duplicates_rows():
    """Overlapping patches duplicate a point's row once per chunk it lands in."""
    m = navis.example_neurons(1, kind="mesh")
    df = sample_patches(m, n_points=64, density=1e-5, mode="cover", random_state=0)
    # cover reuses points across patches -> total rows exceed the unique cloud points
    assert len(df) > df.source_id.nunique()


def test_sample_patches_pad_gives_uniform_groups(skeleton):
    df = sample_patches(skeleton, n_points=64, density=1e-4, mode="partition",
                        undersized="pad", random_state=0)
    # every patch has exactly n_points rows; pad rows carry source_id == -1
    assert df.groupby("chunk_id").size().unique().tolist() == [64]
    assert (df.source_id == -1).any()
    # pad rows have NaN coordinates
    assert df.loc[df.source_id == -1, "x"].isna().all()


def test_sample_patches_reproducible(skeleton):
    a = sample_patches(skeleton, n_points=50, density=1e-4, random_state=7)
    b = sample_patches(skeleton, n_points=50, density=1e-4, random_state=7)
    assert a.equals(b)


@pytest.mark.parametrize("call", [
    # neither density nor spacing / both given
    lambda: sample_patches(navis.example_neurons(1, kind="mesh"), n_points=64),
    lambda: sample_patches(navis.example_neurons(1, kind="mesh"), n_points=64,
                           density=1e-5, spacing=400),
    # n_points positional (keyword-only) / non-positive
    lambda: sample_patches(navis.example_neurons(1, kind="mesh"), 64, density=1e-5),
    lambda: sample_patches(navis.example_neurons(1, kind="mesh"), n_points=0, density=1e-5),
    # bad mode / unsupported type
    lambda: sample_patches(navis.example_neurons(1, kind="mesh"), n_points=64,
                           density=1e-5, mode="nope"),
    lambda: sample_patches(navis.make_dotprops(navis.example_neurons(1, kind="skeleton"), k=5),
                           n_points=64, density=1e-5),
])
def test_sample_patches_bad_args_raise(call):
    with pytest.raises((TypeError, ValueError)):
        call()
