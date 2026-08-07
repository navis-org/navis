"""Tests for closing the holes cut into a mesh.

Two entry points, with deliberately different scope:

- `navis.fill_holes` closes every opening the mesh has.
- `subset_neuron(..., cap_holes=True)` closes only the ones that call made,
  which means it has to leave pre-existing openings alone.

"""

import navis
import navis_fastcore as fastcore
import numpy as np
import pytest
import trimesh as tm


def boundary(mesh):
    """Directed half-edges with only one face on them."""
    return fastcore.boundary_halfedges(np.asarray(mesh.faces))


@pytest.fixture
def tube():
    """A watertight capsule, subdivided so it can be cut anywhere."""
    m = tm.creation.capsule(height=100, radius=10, count=[24, 24])
    for _ in range(3):
        m = m.subdivide()
    return navis.Mesh(m, units="1 nm")


@pytest.fixture
def cut_tube(tube):
    """The same capsule with its top lopped off - one clean, ragged opening."""
    return navis.subset_neuron(tube, np.flatnonzero(tube.vertices[:, 2] < 20))


def test_cut_leaves_a_hole(cut_tube):
    assert len(boundary(cut_tube)) > 0
    assert not cut_tube.trimesh.is_watertight


@pytest.mark.parametrize("close", ["fill_holes", "cap_holes"])
def test_closes_the_hole(tube, cut_tube, close):
    if close == "fill_holes":
        closed = navis.fill_holes(cut_tube)
    else:
        closed = navis.subset_neuron(
            tube, np.flatnonzero(tube.vertices[:, 2] < 20), cap_holes=True
        )

    assert len(boundary(closed)) == 0
    assert closed.trimesh.is_watertight
    # A cap wound the wrong way round would still close the hole, so check that
    # the new faces agree with their neighbours about which side is out.
    assert closed.trimesh.is_winding_consistent
    assert closed.volume > 0


def test_no_vertices_added(cut_tube):
    """Vertex indices have to keep meaning what they meant."""
    filled = navis.fill_holes(cut_tube)

    assert filled.n_vertices == cut_tube.n_vertices
    assert np.array_equal(filled.vertices, cut_tube.vertices)
    # existing faces are untouched, the caps are appended
    assert filled.n_faces > cut_tube.n_faces
    assert np.array_equal(filled.faces[: cut_tube.n_faces], cut_tube.faces)


def test_copy_semantics(cut_tube):
    n_faces = cut_tube.n_faces

    filled = navis.fill_holes(cut_tube, inplace=False)
    assert cut_tube.n_faces == n_faces
    assert filled.n_faces > n_faces

    assert navis.fill_holes(cut_tube, inplace=True) is cut_tube
    assert cut_tube.n_faces > n_faces


def test_method_matches_function(cut_tube):
    assert cut_tube.fill_holes().n_faces == navis.fill_holes(cut_tube).n_faces


def test_cap_holes_spares_pre_existing_openings(tube):
    """A subset must not seal an opening it did not make."""
    # Chop the bottom cap off: an opening the mesh now simply has.
    open_tube = navis.subset_neuron(tube, np.flatnonzero(tube.vertices[:, 2] > -50))
    pre_existing = len(boundary(open_tube))
    assert pre_existing > 0

    # Now cut the *top* off with capping on.
    capped = navis.subset_neuron(
        open_tube,
        np.flatnonzero(open_tube.vertices[:, 2] < 20),
        cap_holes=True,
    )

    # The new opening is closed; the one it inherited is still open.
    assert len(boundary(capped)) == pre_existing
    assert not capped.trimesh.is_watertight

    # `fill_holes`, in contrast, is asked to close everything.
    assert len(boundary(navis.fill_holes(capped))) == 0


def test_prune_twigs_does_not_cap_by_default():
    m = navis.example_neurons(1, kind="mesh")
    pruned = navis.prune_twigs(m, size="5 microns")

    assert len(boundary(pruned)) > len(boundary(m))
    assert len(boundary(navis.fill_holes(pruned))) < len(boundary(pruned))


def test_fill_holes_on_neuron_meshes():
    """The real thing: messy, already non-manifold, hundreds of openings."""
    m = navis.example_neurons(1, kind="mesh")
    pruned = navis.prune_twigs(m, size="5 microns")
    filled = navis.fill_holes(pruned)

    # Most of the raw boundary goes; what is left sits on the junctions the
    # mesh was already non-manifold at, which no amount of capping can close.
    assert len(boundary(filled)) < 0.1 * len(boundary(pruned))
    assert filled.n_vertices == pruned.n_vertices
    # Capping the stumps makes the surface enclose less, not more.
    assert filled.volume < m.volume


def test_cap_holes_matches_fill_holes_on_a_clean_cut(tube):
    """With nothing pre-existing to spare, the two should agree."""
    keep = np.flatnonzero(tube.vertices[:, 2] < 20)
    capped = navis.subset_neuron(tube, keep, cap_holes=True)
    filled = navis.fill_holes(navis.subset_neuron(tube, keep))

    assert capped.n_faces == filled.n_faces
    assert np.array_equal(np.sort(capped.faces, axis=None), np.sort(filled.faces, axis=None))


def test_nothing_to_do_is_harmless(tube):
    """A watertight mesh comes back unchanged rather than mangled."""
    assert len(boundary(tube)) == 0
    filled = navis.fill_holes(tube)
    assert filled.n_faces == tube.n_faces
    assert filled.n_vertices == tube.n_vertices


def test_empty_subset_with_capping(tube):
    empty = navis.subset_neuron(tube, [], cap_holes=True)
    assert empty.n_vertices == 0
    assert empty.n_faces == 0


def test_ring_tracing_covers_every_half_edge(cut_tube):
    """The whole point of tracing half-edges rather than using cycle bases."""
    halfedges = boundary(cut_tube)
    rings, offsets = fastcore.trace_loops(halfedges)
    assert offsets[-1] == len(rings) == len(halfedges)


def test_cap_holes_keeps_provenance_usable(tube):
    """Capping adds faces after the subset, so tracking must still line up."""
    keep = np.flatnonzero(tube.vertices[:, 2] < 20)
    sub = navis.subset_neuron(tube, keep, cap_holes=True, track=True)

    merged = navis.merge_subset(tube, sub)
    assert merged.n_vertices == tube.n_vertices
