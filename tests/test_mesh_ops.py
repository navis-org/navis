"""Tests for the mesh operations that are not rebuilds - `fix_mesh` and
`smooth_mesh`.

For `fix_mesh` the thing worth pinning is the trimesh-version gate:
`Trimesh.remove_duplicate_faces`/`.remove_degenerate_faces` were replaced by
`.unique_faces()`/`.nondegenerate_faces()` in trimesh 3.23 and removed in 4.10,
and `fix_mesh` supports both.

For `smooth_mesh` it is the contract that lets it stay out of the rebuild
machinery: it moves vertices and replaces none of them, so the faces, the vertex
count and the vertex order all come back untouched and everything indexed by
vertex is still valid. `navis.simplify_mesh`, which does replace them, is
covered in `test_links.py` alongside the rest of the repair system.
"""

import warnings

import navis
import numpy as np
import pytest
import trimesh as tm

from navis.meshes.mesh_utils import TRIMESH_HAS_FACE_FILTERS, _version_tuple


@pytest.fixture
def messy_box():
    """A unit cube with one duplicate and one degenerate face bolted on."""
    box = tm.creation.box((1, 1, 1))
    verts, faces = np.asarray(box.vertices), np.asarray(box.faces)
    faces = np.vstack([faces, faces[0:1], [[0, 0, 1]]])
    return navis.Mesh((verts, faces), process=False)


@pytest.mark.parametrize(
    "version,expected",
    [
        ("3.8", (3, 8)),
        ("3.22.0", (3, 22, 0)),
        ("3.23.0", (3, 23, 0)),
        ("4.12.2", (4, 12, 2)),
        # Pre-releases and dev versions must compare as their release
        ("4.10.0rc1", (4, 10, 0)),
        ("4.10.0-rc.1", (4, 10, 0)),
        ("4.0.0.dev3", (4, 0, 0)),
    ],
)
def test_version_tuple(version, expected):
    assert _version_tuple(version) == expected


def test_version_gate_matches_installed_trimesh():
    """The gate must agree with what the installed trimesh actually provides.

    This is the test that catches it if trimesh moves the goalposts again.
    """
    has_new = hasattr(tm.Trimesh, "unique_faces") and hasattr(
        tm.Trimesh, "nondegenerate_faces"
    )
    assert TRIMESH_HAS_FACE_FILTERS == has_new

    # And whichever branch we take must be callable
    if not TRIMESH_HAS_FACE_FILTERS:
        assert hasattr(tm.Trimesh, "remove_duplicate_faces")
        assert hasattr(tm.Trimesh, "remove_degenerate_faces")


def test_fix_mesh(messy_box):
    """Duplicate and degenerate faces must go; the cube itself must survive."""
    assert len(messy_box.faces) == 14

    fixed = navis.fix_mesh(messy_box)

    assert len(fixed.faces) == 12
    assert fixed.n_vertices == 8
    assert fixed.trimesh.is_watertight
    assert float(fixed.trimesh.volume) == pytest.approx(1)


def test_fix_mesh_no_deprecation_warning(messy_box):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        navis.fix_mesh(messy_box)


def test_fix_mesh_copy_semantics(messy_box):
    fixed = navis.fix_mesh(messy_box)
    assert len(messy_box.faces) == 14  # untouched
    assert fixed is not messy_box

    out = navis.fix_mesh(messy_box, inplace=True)
    assert out is messy_box
    assert len(messy_box.faces) == 12


def test_fix_mesh_remove_fragments():
    """Small disconnected bits are dropped."""
    m = navis.example_neurons(1, kind="mesh")
    sizes = sorted(len(c) for c in navis.graph.graph_utils._connected_components(m))

    fixed = navis.fix_mesh(m, remove_fragments=5)

    # Everything with <= 5 vertices goes
    assert fixed.n_vertices == sum(s for s in sizes if s > 5)


def test_fix_mesh_fill_holes(messy_box):
    fixed = navis.fix_mesh(messy_box, fill_holes=True)
    assert fixed.trimesh.is_watertight


def test_fix_mesh_trimesh_in_trimesh_out(messy_box):
    """`fix_mesh` also takes a raw trimesh (e.g. from `Volume.validate`)."""
    raw = tm.Trimesh(messy_box.vertices, messy_box.faces, process=False)

    fixed = navis.fix_mesh(raw)

    assert isinstance(fixed, tm.Trimesh)
    assert len(fixed.faces) == 12


def test_validate(messy_box):
    """`Mesh.validate()` routes here and defaults to returning a copy."""
    assert len(messy_box.validate().faces) == 12
    assert len(messy_box.faces) == 14

    assert len(messy_box.validate(inplace=True).faces) == 12
    assert len(messy_box.faces) == 12


@pytest.mark.parametrize("process", [True, False])
def test_validate_on_construction(messy_box, process):
    """`Mesh(validate=True)` must actually fix the mesh.

    With `process=True` trimesh's own constructor does it; with `process=False`
    it is down to us.
    """
    verts, faces = messy_box.vertices, messy_box.faces

    built = navis.Mesh((verts, faces), process=process, validate=True)
    assert len(built.faces) == 12

    # ... and must not fix it when not asked to
    assert len(navis.Mesh((verts, faces), process=process).faces) == 14


@pytest.mark.skipif(
    not hasattr(tm.Trimesh, "remove_duplicate_faces"),
    reason="installed trimesh has no legacy face-removal methods",
)
def test_both_branches_agree(messy_box, monkeypatch):
    """Where trimesh offers both APIs, they must produce the same mesh."""
    new = navis.fix_mesh(messy_box)

    monkeypatch.setattr(navis.meshes.mesh_utils, "TRIMESH_HAS_FACE_FILTERS", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = navis.fix_mesh(messy_box)

    assert np.array_equal(new.vertices, old.vertices)
    assert np.array_equal(new.faces, old.faces)


# ---------------------------------------------------------------------------
# smooth_mesh
# ---------------------------------------------------------------------------


@pytest.fixture
def sheet():
    """A flat grid with every other vertex pushed out of the plane.

    Two things make this the right fixture. The clean surface is `z = 0`, so
    residual `z` *is* the noise and sliding within the plane - which the uniform
    umbrella does plenty of - cannot be mistaken for smoothing. And it is open,
    so it has a rim, which is what `preserve_border` is about.
    """
    n = 8
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    verts = np.stack([i.ravel(), j.ravel(), np.zeros(n * n)], axis=1).astype(float)
    verts[:, 2] = np.where(np.arange(n * n) % 2 == 0, 0.4, -0.4)

    faces = [
        f
        for a in range(n - 1)
        for b in range(n - 1)
        for f in (
            [a * n + b, (a + 1) * n + b, (a + 1) * n + b + 1],
            [a * n + b, (a + 1) * n + b + 1, a * n + b + 1],
        )
    ]
    mesh = navis.Mesh((verts, np.array(faces)), process=False)
    # Everything not in the interior of the grid, i.e. the rim
    mesh._rim = (i.ravel() % (n - 1) == 0) | (j.ravel() % (n - 1) == 0)
    return mesh


@pytest.fixture
def neuron():
    return navis.example_neurons(1, kind="mesh")


def test_smooth_mesh_takes_the_noise_out(sheet):
    """Interior only - the rim is pinned by default, see `preserve_border`.

    The ring just inside the rim is held back by it, so this is a good deal less
    than the full flattening the same filter manages on a closed mesh.
    """
    smoothed = navis.smooth_mesh(sheet, iterations=5)
    assert np.abs(smoothed.vertices[~sheet._rim, 2]).max() < 0.4 / 5


def test_smooth_mesh_replaces_no_vertices(neuron):
    """The whole contract: vertices move, nothing is replaced."""
    smoothed = navis.smooth_mesh(neuron)

    assert smoothed.n_vertices == neuron.n_vertices
    assert np.array_equal(smoothed.faces, neuron.faces)
    # And they did move - otherwise the above is trivially true
    assert not np.allclose(smoothed.vertices, neuron.vertices)


def test_smooth_mesh_keeps_everything_indexed_by_vertex(neuron):
    """Connectors, extra edges and attached data all survive untouched."""
    connectors = neuron.connectors.copy()
    connectors["vertex_id"] = neuron.snap(connectors[["x", "y", "z"]].values)[0]
    neuron.connectors = connectors
    neuron.extra_edges = [[0, neuron.n_vertices - 1]]
    neuron.attach(
        "depth",
        np.arange(neuron.n_vertices, dtype=float),
        axis="vertices",
        on_rebuild="carry",
    )

    smoothed = navis.smooth_mesh(neuron)

    assert np.array_equal(
        smoothed.connectors.vertex_id.values, connectors.vertex_id.values
    )
    assert np.array_equal(smoothed.extra_edges, neuron.extra_edges)
    assert np.array_equal(smoothed.depth, neuron.depth)


def test_smooth_mesh_lets_the_skeleton_go_stale(neuron):
    """A skeleton traced through the old coordinates no longer describes these."""
    skeleton = neuron.skeleton

    smoothed = navis.smooth_mesh(neuron)

    assert smoothed.skeleton is not skeleton


def test_smooth_mesh_copy_semantics(neuron):
    before = np.asarray(neuron.vertices).copy()

    assert navis.smooth_mesh(neuron) is not neuron
    assert np.array_equal(neuron.vertices, before)

    assert navis.smooth_mesh(neuron, inplace=True) is neuron
    assert not np.array_equal(neuron.vertices, before)


@pytest.mark.parametrize("method", ["taubin", "laplacian", "humphrey"])
def test_smooth_mesh_methods(neuron, method):
    """Every method smooths; only the plain Laplacian eats the volume."""
    smoothed = navis.smooth_mesh(neuron, method=method)
    kept = smoothed.volume / neuron.volume

    assert not np.allclose(smoothed.vertices, neuron.vertices)
    assert kept < 0.5 if method == "laplacian" else kept > 0.7


def test_smooth_mesh_kwargs_reach_fastcore(neuron):
    """`volume_correction` is the cheapest thing to see from out here."""
    shrunk = navis.smooth_mesh(neuron, method="laplacian")
    corrected = navis.smooth_mesh(neuron, method="laplacian", volume_correction=True)

    assert shrunk.volume / neuron.volume < 0.5
    assert corrected.volume / neuron.volume == pytest.approx(1, abs=0.01)


def test_smooth_mesh_preserves_the_border_by_default(sheet):
    """navis' own default, because its meshes are routinely cut-out fragments."""
    rim = sheet._rim

    assert np.array_equal(navis.smooth_mesh(sheet).vertices[rim], sheet.vertices[rim])
    assert not np.allclose(
        navis.smooth_mesh(sheet, preserve_border=False).vertices[rim],
        sheet.vertices[rim],
    )


def test_smooth_mesh_lock(sheet):
    """`lock` pins vertices that are not on the border."""
    lock = np.zeros(sheet.n_vertices, dtype=bool)
    lock[~sheet._rim] = True

    smoothed = navis.smooth_mesh(sheet, lock=lock)

    assert np.array_equal(smoothed.vertices, sheet.vertices)


def test_smooth_mesh_rejects_non_meshes():
    with pytest.raises(TypeError):
        navis.smooth_mesh(navis.example_neurons(1, kind="skeleton"))


def test_smooth_mesh_takes_volumes_and_trimeshes(neuron):
    volume = navis.example_volume("LH")
    assert navis.smooth_mesh(volume).vertices.shape == volume.vertices.shape

    raw = tm.Trimesh(neuron.vertices, neuron.faces, process=False)
    assert navis.smooth_mesh(raw).vertices.shape == raw.vertices.shape


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (navis.smooth_mesh, dict(backend="trimesh")),
        (navis.smooth_mesh, dict(L=0.5)),
        (navis.simplify_mesh, dict(F=0.5, backend="fqmr")),
    ],
)
def test_deprecated_arguments_blame_the_caller(neuron, fn, kwargs):
    """The `stacklevel`s in `_deprecated_backend` and friends, pinned.

    Python's default filters only surface a `DeprecationWarning` attributed to
    `__main__`, so one blamed on navis' own decorators is one nobody ever sees.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(neuron, **kwargs)

    (warned,) = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert warned.filename == __file__


def test_smooth_mesh_L_is_lamb(neuron):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        renamed = navis.smooth_mesh(neuron, L=0.3)

    assert np.array_equal(renamed.vertices, navis.smooth_mesh(neuron, lamb=0.3).vertices)

    with pytest.raises(TypeError, match="same argument"):
        navis.smooth_mesh(neuron, L=0.3, lamb=0.5)
