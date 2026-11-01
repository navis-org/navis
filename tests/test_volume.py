"""Tests for mesh volume.

Trimesh measures a volume by summing, over the faces, the tetrahedron each one
makes with the *origin*. On a closed surface the anchor cancels exactly, so that
costs only precision. On an open one the cone to the anchor is real geometry and
grows with the distance to it, which for a mesh in dataset nanometres is enough
to swamp the answer: one missing triangle in five thousand is a percent-level
error and a barely-open neuron can come back negative.

`navis` therefore anchors on the mesh's own bounding box, on both classes that
can report a volume - `Mesh` (through `TrimeshPlus`) and `Volume`, which are
siblings rather than one inheriting from the other. What that buys is not an
inside for open surfaces, which nothing can give them, but an answer that does
not change when the mesh is moved.
"""

import numpy as np
import pytest
import trimesh

import navis
from navis.utils.subclasses import TrimeshPlus, anchored_volume


@pytest.fixture
def sphere():
    """A closed mesh at a neuron-ish scale, in nanometres."""
    return trimesh.creation.icosphere(subdivisions=3, radius=5000.0)


def _punch(mesh, n):
    """The same mesh with `n` triangles taken out of it."""
    return mesh.vertices, mesh.faces[n:]


@pytest.mark.parametrize("cls", [TrimeshPlus, navis.Volume])
@pytest.mark.parametrize("holes", [0, 1, 50])
def test_volume_does_not_care_where_the_mesh_sits(sphere, cls, holes):
    """The property the anchoring exists for, on both classes that have it."""
    verts, faces = _punch(sphere, holes)
    here = cls(verts, faces, process=False).volume
    for offset in (1e5, 4.6e5, 1e7):
        there = cls(verts + offset, faces, process=False).volume
        assert there == pytest.approx(here, rel=1e-9), (
            f"{cls.__name__} with {holes} holes moved by {offset}"
        )


@pytest.mark.parametrize("cls", [TrimeshPlus, navis.Volume])
def test_closed_volume_agrees_with_trimesh(sphere, cls):
    """Where trimesh is trustworthy - closed, near the origin - so are we."""
    got = cls(sphere.vertices, sphere.faces, process=False).volume
    assert got == pytest.approx(sphere.volume, rel=1e-12)


def test_trimesh_is_the_one_that_drifts(sphere):
    """Guards the premise: if trimesh ever anchors on the mesh, this is moot."""
    verts, faces = _punch(sphere, 1)
    near = trimesh.Trimesh(verts, faces, process=False).volume
    far = trimesh.Trimesh(verts + 4.6e5, faces, process=False).volume
    assert abs(far - near) > 0.01 * abs(near), (
        "trimesh no longer drifts on an open mesh - the override may be redundant"
    )


def test_neuron_volume_is_translation_invariant():
    """End to end, on the mesh a user actually has: open, and far from zero."""
    m = navis.example_neurons(1, kind="mesh")
    pruned = navis.prune_twigs(m, min_length="5 microns")  # opens the stumps
    assert not pruned.trimesh.is_watertight, "fixture should be open"

    here = float(pruned.volume)
    moved = pruned.copy()
    moved.vertices = pruned.vertices + 1e6
    assert float(moved.volume) == pytest.approx(here, rel=1e-9)


def test_empty_mesh_has_no_volume():
    assert anchored_volume(trimesh.Trimesh()) == 0.0
