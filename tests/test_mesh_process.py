"""Tests for `navis.utils.meshproc` - our stand-in for trimesh's `process=True`.

The contract is not "produces a sensible mesh" but "produces *exactly* what
trimesh produces": same vertices, same faces, same order. Anything less would
renumber vertices, and a vertex index is what connectors, extra edges and the
skeleton correspondence are all pinned to.

So nearly everything here is a differential test against trimesh itself. The
cases that matter are the ones where the two implementations could plausibly
disagree: duplicate coordinates, vertices no face names, non-finite values, and
coordinate ranges at either end of what a packing can hold.
"""

import navis
import numpy as np
import pytest
import trimesh as tm

from trimesh import grouping

from navis.utils import meshproc


def as_arrays(mesh):
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def reference(vertices, faces, validate=False):
    """What trimesh alone makes of this mesh."""
    return as_arrays(
        tm.Trimesh(np.asarray(vertices).copy(), np.asarray(faces).copy(),
                   process=True, validate=validate)
    )


def assert_same(got, want):
    got_v, got_f = got
    want_v, want_f = want
    assert got_v.shape == want_v.shape
    assert np.array_equal(got_v, want_v)
    assert got_f.shape == want_f.shape
    assert np.array_equal(got_f, want_f)


# ---------------------------------------------------------------------------
# Row grouping


def test_unique_rows_matches_trimesh():
    """The grouping itself, against the function it replaces."""
    rng = np.random.default_rng(0)
    rows = rng.integers(-20, 20, size=(500, 3))

    index, inverse = meshproc.unique_rows(rows)
    want_index, want_inverse = grouping.unique_rows(rows, keep_order=True)

    assert np.array_equal(index, want_index)
    assert np.array_equal(inverse, want_inverse)
    # ... and the pair actually reconstructs the input
    assert np.array_equal(rows[index][inverse], rows)


def test_unique_rows_orders_by_first_occurrence():
    """Groups are numbered as they first appear, not as they sort."""
    rows = np.array([[9, 9, 9], [0, 0, 0], [9, 9, 9], [5, 5, 5]])
    index, inverse = meshproc.unique_rows(rows)

    assert np.array_equal(index, [0, 1, 3])
    assert np.array_equal(inverse, [0, 1, 0, 2])


def _wide_values():
    """Values spanning more than a key could hold, but few of them distinct.

    What the packing has to survive is the *combination* - a column too wide to
    pack by value is fine as long as it takes few distinct values.
    """
    rows = np.random.default_rng(1).integers(-(2**40), 2**40, size=(2000, 3))
    rows[:500] = rows[500:1000]  # and a healthy dose of genuine duplicates
    return rows


def _more_columns_than_fit():
    """Enough columns that the packing has to re-code partway through.

    A textured mesh with normals groups on eight columns, not three, and with
    that many the product of the per-column value counts overruns an int64 long
    before the mesh is large. No mesh in the test suite is wide enough to reach
    the re-code, so this is the only thing holding that branch up.
    """
    rows = np.random.default_rng(2).integers(0, 10**6, size=(2000, 6))
    rows[:200] = rows[200:400]
    return rows


@pytest.mark.parametrize(
    "make_rows", [_wide_values, _more_columns_than_fit],
    ids=["wide values", "more columns than fit"],
)
def test_unique_rows_is_a_packing_not_a_hash(make_rows):
    """Distinct rows must never collide - a collision would weld two vertices."""
    rows = make_rows()
    index, inverse = meshproc.unique_rows(rows)

    assert len(index) == len(np.unique(rows, axis=0))
    assert np.array_equal(rows[index][inverse], rows)


def test_unique_rows_empty():
    index, inverse = meshproc.unique_rows(np.zeros((0, 3), dtype=np.int64))
    assert len(index) == 0
    assert len(inverse) == 0


# ---------------------------------------------------------------------------
# `process` against trimesh


def _mesh_cases():
    """The mesh shapes where the two implementations could plausibly disagree."""
    rng = np.random.default_rng(3)
    faces = rng.integers(0, 300, size=(150, 3)).astype(np.int64)
    empty = np.zeros((0, 3), dtype=np.int64)

    def case(id, vertices, faces=faces):
        return pytest.param(vertices, faces, id=id)

    yield case("integer lattice, many duplicates",
               rng.integers(0, 8, size=(300, 3)).astype(float))
    yield case("sub-lattice floats", rng.integers(0, 10, size=(300, 3)) / 4.0)
    yield case("no duplicates at all", rng.random((300, 3)) * 1e4)
    yield case("most vertices unreferenced",
               rng.integers(0, 5, size=(300, 3)).astype(float),
               rng.integers(0, 50, size=(40, 3)).astype(np.int64))
    yield case("negative coordinates",
               rng.integers(-1000, 1000, size=(300, 3)).astype(float))
    # Coordinates this small are the one case trimesh's own bit-packing fast
    # path actually fires on, so it is also the one case where we are not
    # racing the slow path.
    yield case("tiny coordinates",
               rng.integers(0, 6, size=(300, 3)).astype(float) * 1e-6)
    yield case("no faces", rng.integers(0, 6, size=(50, 3)).astype(float), empty)
    yield case("single vertex", np.zeros((1, 3)), empty)
    yield case("empty", np.zeros((0, 3)), empty)
    # Separations either side of `tol.merge`: 1e-9 apart must fuse, 2e-8 apart
    # must not.
    yield case(
        "separations around the merge tolerance",
        np.array([[0.0, 0, 0], [1e-9, 0, 0], [2e-8, 0, 0], [1.0, 1, 1], [1.0, 1, 1]]),
        np.array([[0, 1, 2], [2, 3, 4], [0, 3, 4]], dtype=np.int64))

    nonfinite = rng.integers(0, 6, size=(200, 3)).astype(float)
    nonfinite[3] = np.nan
    nonfinite[17, 1] = np.inf
    nonfinite[50, 2] = -np.inf
    yield case("NaN and inf", nonfinite,
               rng.integers(0, 200, size=(100, 3)).astype(np.int64))


@pytest.mark.parametrize(("vertices", "faces"), list(_mesh_cases()))
def test_process_matches_trimesh(vertices, faces):
    assert_same(meshproc.process(vertices, faces), reference(vertices, faces))


@pytest.mark.parametrize("seed", range(20))
def test_process_matches_trimesh_on_random_meshes(seed):
    """Fuzz, because the interesting cases are combinations.

    A group of the wrong size, a group split across the referenced mask, a face
    naming a vertex that gets dropped - enumerating those by hand is how you
    miss one.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 400))
    style = seed % 5
    if style == 0:
        vertices = rng.integers(-6, 6, size=(n, 3)).astype(float)
    elif style == 1:
        vertices = rng.integers(-(10**6), 10**6, size=(n, 3)).astype(float)
    elif style == 2:
        vertices = rng.random((n, 3)) * rng.choice([1e-6, 1.0, 1e6])
    elif style == 3:
        vertices = rng.integers(0, 20, size=(n, 3)) / 8.0
    else:
        vertices = rng.integers(0, 4, size=(n, 3)).astype(float)
        vertices[rng.random(n) < 0.05] = rng.choice([np.nan, np.inf, -np.inf])

    faces = rng.integers(
        0, max(1, int(n * rng.choice([1.0, 0.3]))),
        size=(int(rng.integers(0, 300)), 3),
    ).astype(np.int64)

    assert_same(meshproc.process(vertices, faces), reference(vertices, faces))


def test_process_matches_trimesh_on_a_real_mesh():
    m = navis.example_neurons(1, kind="mesh")
    assert_same(meshproc.process(m.vertices, m.faces),
                reference(m.vertices, m.faces))


def test_process_coerces_dtypes():
    """Trimesh's setters coerce; anything downstream of us may rely on that."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    verts, faces = meshproc.process(vertices, faces)
    assert verts.dtype == np.float64
    assert faces.dtype == np.int64


# ---------------------------------------------------------------------------
# `merge_vertices` against trimesh, on a live mesh object


def _welded_mesh():
    """A mesh whose duplicate vertices are genuinely there to be merged."""
    rng = np.random.default_rng(4)
    vertices = rng.integers(0, 8, size=(200, 3)).astype(float)
    faces = rng.integers(0, 200, size=(120, 3)).astype(np.int64)
    return vertices, faces


@pytest.mark.parametrize(
    ("normals", "kwargs"),
    [(False, {}), (True, {}), (True, {"merge_norm": True})],
    ids=["plain", "differing normals kept apart", "merge_norm"],
)
def test_merge_vertices_matches_trimesh(normals, kwargs):
    """Cached vertex normals join the grouping key, so they must reach ours too.

    If they were dropped, two vertices sharing a coordinate but not a normal
    would be welded and the shading would change - unless `merge_norm` says to
    weld them anyway.
    """
    vertices, faces = _welded_mesh()
    mine = tm.Trimesh(vertices.copy(), faces.copy(), process=False)
    theirs = tm.Trimesh(vertices.copy(), faces.copy(), process=False)
    if normals:
        mine.vertex_normals, theirs.vertex_normals  # noqa: B018  (fill the cache)

    meshproc.merge_vertices(mine, **kwargs)
    grouping.merge_vertices(theirs, **kwargs)

    assert_same(as_arrays(mine), as_arrays(theirs))


# ---------------------------------------------------------------------------
# The classes that go through it


def test_mesh_processing_matches_trimesh():
    vertices, faces = _welded_mesh()
    got = navis.Mesh((vertices.copy(), faces.copy()), process=True)
    assert_same((got.vertices, got.faces), reference(vertices, faces))


def test_mesh_validate_still_goes_through_trimesh():
    """`validate=True` is trimesh-driven, and has to stay bit-identical.

    It cannot be compared against `tm.Trimesh(validate=True)` directly, because
    `Mesh` runs `navis.fix_mesh` on top - so the comparison is against the same
    mesh built the way this used to be built, with trimesh rather than
    `TrimeshPlus` doing the processing.
    """
    vertices, faces = _welded_mesh()
    got = navis.Mesh((vertices.copy(), faces.copy()), process=True, validate=True)

    want = navis.Mesh(reference(vertices, faces, validate=True), process=False)
    want.validate(inplace=True)

    assert_same((got.vertices, got.faces), (want.vertices, want.faces))


def test_volume_processing_matches_trimesh():
    vertices, faces = _welded_mesh()
    got = navis.Volume(vertices.copy(), faces.copy())
    assert_same(as_arrays(got), reference(vertices, faces))


def test_trimeshplus_processing_matches_trimesh():
    vertices, faces = _welded_mesh()
    got = navis.utils.TrimeshPlus(vertices.copy(), faces.copy(), process=True)
    assert_same(as_arrays(got), reference(vertices, faces))


def test_mesh_without_processing_is_untouched():
    """`process=False` still has to mean what it says."""
    vertices, faces = _welded_mesh()
    got = navis.Mesh((vertices.copy(), faces.copy()), process=False)
    assert np.array_equal(got.vertices, vertices)
    assert np.array_equal(got.faces, faces)
