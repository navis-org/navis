"""Tests for `navis.fix_mesh` and the trimesh-version gate behind it.

`Trimesh.remove_duplicate_faces`/`.remove_degenerate_faces` were replaced by
`.unique_faces()`/`.nondegenerate_faces()` in trimesh 3.23 and removed in 4.10.
`fix_mesh` supports both; these tests make sure it keeps working whichever
trimesh is installed.
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
