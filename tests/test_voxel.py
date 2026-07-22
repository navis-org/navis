"""Tests for `navis.VoxelNeuron`.

`VoxelNeuron` stores its data in one of two backings, picked at construction and
reported by `._base_data_type`:

  - "grid":   a dense 3D array; values live in the array itself
  - "voxels": a sparse (N, 3) array of integer coordinates, with the values held
              separately in `._values`

Most of the class' methods have to handle both. Historically only the grid path
was exercised (the sole voxel coverage in the test suite was NRRD round-tripping
in `test_io.py`), so the sparse path had drifted: `threshold()` filtered the
coordinates but not the values, `normalize()` scaled the coordinates *instead of*
the values, and a documented (N, 4) input silently dropped its value column.

The tests below therefore lean on two ideas:

  - build the same neuron in both backings and assert they agree, so the sparse
    path cannot drift from the grid path again
  - assert the geometric invariants (`flip` preserves the bounding box and is its
    own inverse) rather than just re-stating the implementation
"""

import copy
import pickle

import navis
import numpy as np
import pandas as pd

import pytest


# The three filled voxels shared by both backings, as (x, y, z, value).
VOXELS = np.array([[1, 1, 1], [2, 3, 4], [3, 4, 5]])
VALUES = np.array([1.0, 5.0, 10.0])
SHAPE = (4, 5, 6)

UNITS = "8 nm"
OFFSET = np.array([100.0, 200.0, 300.0])


@pytest.fixture
def grid_neuron():
    """The reference neuron, backed by a dense grid."""
    grid = np.zeros(SHAPE, dtype=np.float64)
    grid[VOXELS[:, 0], VOXELS[:, 1], VOXELS[:, 2]] = VALUES
    # `sparsify=False` is load-bearing: this neuron is sparse enough that the
    # default ("auto") would store it as voxels and these tests would silently
    # stop exercising the grid backing at all
    return navis.VoxelNeuron(
        grid, units=UNITS, offset=OFFSET.copy(), sparsify=False
    )


@pytest.fixture
def voxel_neuron():
    """The same neuron, backed by sparse coordinates + values."""
    n = navis.VoxelNeuron(VOXELS.copy(), units=UNITS, offset=OFFSET.copy())
    n.values = VALUES.copy()
    return n


@pytest.fixture(params=["grid", "voxels"])
def neuron(request, grid_neuron, voxel_neuron):
    """The same neuron in both backings, so tests run against each."""
    return grid_neuron if request.param == "grid" else voxel_neuron


def sorted_voxels(n):
    """Voxels + values as one array, in a canonical order.

    The two backings have no reason to enumerate voxels in the same order, so
    anything comparing across them has to sort first.
    """
    combined = np.column_stack((n.voxels.astype(float), n.values.astype(float)))
    return combined[np.lexsort(combined[:, :3].T[::-1])]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_backings(grid_neuron, voxel_neuron):
    assert grid_neuron._base_data_type == "grid"
    assert voxel_neuron._base_data_type == "voxels"


def test_construction_sparsify_auto():
    """By default a sparse-enough grid is stored as voxels, a dense one is not."""
    sparse_grid = np.zeros((40, 40, 40), dtype=np.float64)
    sparse_grid[0, 0, 0] = sparse_grid[1, 2, 3] = 1
    assert navis.VoxelNeuron(sparse_grid)._base_data_type == "voxels"

    # Above the break-even density the dense grid is the compact one. Note this
    # is dtype-dependent: bool breaks even at ~4% but float64 only at ~25%.
    dense_grid = np.ones((20, 20, 20), dtype=bool)
    assert navis.VoxelNeuron(dense_grid)._base_data_type == "grid"

    # ...and the decision can always be forced either way
    assert navis.VoxelNeuron(dense_grid, sparsify=True)._base_data_type == "voxels"
    assert navis.VoxelNeuron(sparse_grid, sparsify=False)._base_data_type == "grid"

    with pytest.raises(ValueError, match="sparsify"):
        navis.VoxelNeuron(sparse_grid, sparsify="yes")


def test_construction_sparsify_preserves_data():
    """Auto-sparsifying must not change the neuron it is handed."""
    # Trailing empty planes: the shape must survive, or `.grid` comes back
    # a different size than what was passed in
    grid = np.zeros((6, 6, 6), dtype=np.float64)
    grid[0, 0, 0] = 5
    grid[1, 2, 3] = 7

    n = navis.VoxelNeuron(grid)
    assert n._base_data_type == "voxels"
    assert n.shape == grid.shape
    assert np.array_equal(n.grid, grid)
    assert n.dtype == grid.dtype


def test_values_uses_nonzero_not_positive():
    """`.values` must line up with `.voxels`/`.nnz`, which use non-zero."""
    grid = np.zeros((3, 3, 3), dtype=np.float64)
    grid[0, 0, 0], grid[1, 1, 1], grid[2, 2, 2] = 5, -2, 3
    n = navis.VoxelNeuron(grid, sparsify=False)

    assert n.voxels.shape[0] == n.values.shape[0] == n.nnz == 3
    # The negative voxel is a real value, not an empty one
    assert n.min() == -2


def test_backings_are_equivalent(grid_neuron, voxel_neuron):
    """The two backings must describe the exact same neuron."""
    assert grid_neuron.shape == voxel_neuron.shape == SHAPE
    assert grid_neuron.nnz == voxel_neuron.nnz == len(VOXELS)
    assert np.array_equal(grid_neuron.grid, voxel_neuron.grid)
    assert np.allclose(sorted_voxels(grid_neuron), sorted_voxels(voxel_neuron))


def test_construction_from_xyz_values():
    """(N, 4) input keeps its value column.

    Regression: the 4th column used to be dropped and `.values` fell back to an
    array of ones.
    """
    data = np.column_stack((VOXELS, VALUES))
    n = navis.VoxelNeuron(data)

    assert np.array_equal(n.voxels, VOXELS)
    assert np.array_equal(n.values, VALUES)
    assert n.nnz == len(VOXELS)


def test_construction_casts_float_coordinates():
    """Float coordinates are cast to int so they stay usable as grid indices.

    This bites (N, 4) inputs in particular: a float value column forces the
    whole input array - coordinates included - to float.
    """
    data = np.column_stack((VOXELS, VALUES))  # float, because VALUES is float
    n = navis.VoxelNeuron(data)

    assert np.issubdtype(n.voxels.dtype, np.integer)
    assert n.grid.shape == SHAPE  # would raise if coordinates were float


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((5, 2)),  # too few columns
        np.zeros((5, 5)),  # too many columns
        np.zeros(5),  # 1D
        np.zeros((2, 2, 2, 2)),  # 4D
    ],
)
def test_construction_rejects_bad_shapes(data):
    with pytest.raises(navis.utils.ConstructionError):
        navis.VoxelNeuron(data)


def test_construction_rejects_non_array():
    with pytest.raises(navis.utils.ConstructionError):
        navis.VoxelNeuron([[1, 2, 3]])


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_dtype_reports_value_dtype(grid_neuron, voxel_neuron):
    """`.dtype` describes the values, not the voxel coordinates.

    Regression: the sparse backing used to report the dtype of the *coordinate*
    array, which also made `normalize()` scale to `iinfo(int64).max`.
    """
    assert grid_neuron.dtype == np.float64
    assert voxel_neuron.dtype == np.float64

    as_int = navis.VoxelNeuron(VOXELS.copy())
    as_int.values = VALUES.astype(np.uint8)
    assert as_int.dtype == np.uint8


def test_density(neuron):
    assert neuron.density == pytest.approx(len(VOXELS) / np.prod(SHAPE))


def test_volume_uses_all_three_axes():
    """Volume must use x, y *and* z voxel dimensions.

    Regression: the z dimension used to be squared and y dropped entirely, which
    only showed up for anisotropic neurons.
    """
    n = navis.VoxelNeuron(VOXELS.copy(), units=["1 nm", "2 nm", "4 nm"])
    # 3 voxels of 1 x 2 x 4 nm each
    assert n.volume.to("nm ** 3").magnitude == pytest.approx(3 * 8)


def test_min_max(neuron):
    assert neuron.min() == VALUES.min()
    assert neuron.max() == VALUES.max()


def test_bbox_spans_grid(neuron):
    """The bounding box starts at the offset and spans the grid."""
    bbox = neuron.bbox
    assert np.allclose(bbox[:, 0], OFFSET)
    assert np.allclose(bbox[:, 1], np.array(SHAPE) * 8 + OFFSET)


# ---------------------------------------------------------------------------
# Caching / staleness
# ---------------------------------------------------------------------------


def test_changing_values_invalidates_cached_grid(voxel_neuron):
    """Cached representations must not survive a change to `.values`.

    Regression: `_values` was missing from `CORE_DATA`, so the staleness hash
    ignored it and a cached `.grid` kept serving the old values.
    """
    voxel_neuron.grid  # populate the cache

    new = VALUES * 2
    voxel_neuron.values = new

    assert np.array_equal(
        voxel_neuron.grid[VOXELS[:, 0], VOXELS[:, 1], VOXELS[:, 2]], new
    )


def test_changing_voxels_invalidates_cached_shape(voxel_neuron):
    voxel_neuron.values = None  # so the new voxels don't fail validation
    assert voxel_neuron.shape == SHAPE

    voxel_neuron.voxels = np.array([[0, 0, 0], [9, 9, 9]])
    assert voxel_neuron.shape == (10, 10, 10)


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deep", [False, True])
def test_copy(neuron, deep):
    """Copies are independent of the original."""
    other = neuron.copy(deepcopy=deep)

    assert np.array_equal(other.grid, neuron.grid)
    assert other.units == neuron.units
    assert np.array_equal(other.offset, neuron.offset)

    other._data[:] = 0
    assert neuron.nnz == len(VOXELS)


def test_deepcopy(neuron):
    """Regression: `copy()` did not accept the `deepcopy` kwarg that
    `BaseNeuron.__deepcopy__` passes it, so `copy.deepcopy` raised a TypeError.
    """
    other = copy.deepcopy(neuron)
    assert np.array_equal(other.grid, neuron.grid)


# ---------------------------------------------------------------------------
# threshold / normalize
# ---------------------------------------------------------------------------


def test_threshold(neuron):
    """Regression: the sparse backing filtered the coordinates but left the
    values untouched, so the two ended up different lengths.
    """
    out = neuron.threshold(5)

    assert out.nnz == 2
    assert len(out.voxels) == len(out.values) == 2
    assert set(out.values) == {5.0, 10.0}
    assert neuron.nnz == len(VOXELS)  # original untouched


def test_threshold_inplace(neuron):
    assert neuron.threshold(5, inplace=True) is None
    assert neuron.nnz == 2


def test_normalize_scales_values_not_coordinates(neuron):
    """Regression: the sparse backing divided the voxel *coordinates* by the
    maximum value, corrupting the neuron's geometry.
    """
    before = neuron.voxels.copy()
    out = neuron.normalize()

    assert np.array_equal(out.voxels, before)
    assert out.max() == pytest.approx(1.0)
    assert np.allclose(np.sort(out.values), np.sort(VALUES / VALUES.max()))


def test_normalize_integer_scales_to_dtype_max():
    """Integer data is scaled to the full range of its dtype."""
    n = navis.VoxelNeuron(VOXELS.copy())
    n.values = np.array([1, 2, 4], dtype=np.uint8)

    assert n.normalize().max() == pytest.approx(np.iinfo(np.uint8).max)


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_flip_preserves_bounding_box(neuron, axis):
    """Flipping in place must not move the neuron.

    Regression: the offset used to be flipped as though it were in voxels
    (`shape - 1 - offset`), which both moved the neuron and mixed voxel indices
    into a quantity that is in units.
    """
    before = neuron.bbox
    out = neuron.flip(axis)

    assert np.allclose(out.bbox, before)
    assert np.allclose(out.offset, neuron.offset)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_flip_is_its_own_inverse(neuron, axis):
    out = neuron.flip(axis).flip(axis)
    assert np.array_equal(out.grid, neuron.grid)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_flip_mirrors_voxels(neuron, axis):
    ix = "xyz".index(axis)
    out = neuron.flip(axis)

    expected = np.sort(SHAPE[ix] - 1 - VOXELS[:, ix])
    assert np.array_equal(np.sort(out.voxels[:, ix]), expected)
    # the other two axes are untouched
    for other in set(range(3)) - {ix}:
        assert np.array_equal(
            np.sort(out.voxels[:, other]), np.sort(VOXELS[:, other])
        )


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_flip_mirrors_connectors(neuron, axis):
    """Connectors are in units, so they mirror across the grid's extent.

    Regression: they used to be mirrored with `shape - 1 - coordinate`, i.e. in
    voxels, which is a different (and for any non-trivial offset, very wrong)
    space.
    """
    ix = "xyz".index(axis)
    # Place a connector on each filled voxel, in units
    pos = VOXELS * 8 + OFFSET
    neuron.connectors = pd.DataFrame(pos, columns=["x", "y", "z"])

    out = neuron.flip(axis)

    # Mirroring the connectors must match mirroring the voxels they sit on
    expected = np.sort((SHAPE[ix] - 1 - VOXELS[:, ix]) * 8 + OFFSET[ix])
    assert np.allclose(np.sort(out.connectors[axis].values), expected)

    # ...and flipping twice must be a no-op
    assert np.allclose(
        out.flip(axis).connectors[["x", "y", "z"]].values, pos
    )


def test_flip_rejects_unknown_axis(neuron):
    with pytest.raises(AssertionError):
        neuron.flip("a")


# ---------------------------------------------------------------------------
# strip
# ---------------------------------------------------------------------------


def test_strip(neuron):
    """Stripping empty planes shrinks the grid but keeps the neuron in place."""
    before = neuron.bbox
    out = neuron.strip()

    # Leading empty planes are dropped: voxels start at (1, 1, 1)
    assert out.shape == tuple(np.array(SHAPE) - VOXELS.min(axis=0))
    assert out.nnz == neuron.nnz
    # The offset takes up the slack, so the filled voxels don't move
    assert np.allclose(out.offset, before[:, 0] + VOXELS.min(axis=0) * 8)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_multiplication_scales_units_and_offset(neuron):
    out = neuron * 2

    assert out.units.to("nm").magnitude == pytest.approx(16)
    assert np.allclose(out.offset, OFFSET * 2)
    # The voxels themselves are untouched - only the space they live in scales
    assert np.array_equal(out.voxels, neuron.voxels)


def test_division_scales_units_and_offset(neuron):
    out = neuron / 2

    assert out.units.to("nm").magnitude == pytest.approx(4)
    assert np.allclose(out.offset, OFFSET / 2)


@pytest.mark.parametrize("op,expected", [("add", 1), ("sub", -1)])
def test_addition_and_subtraction_move_offset(neuron, op, expected):
    out = neuron + 10 if op == "add" else neuron - 10

    assert np.allclose(out.offset, OFFSET + expected * 10)
    assert out.units == neuron.units  # units are unchanged by a translation


def physical_size(n):
    """How much space the neuron occupies, in nm."""
    return np.array(n.shape) * n.units_xyz.to("nm").magnitude


def test_convert_units_preserves_physical_size(neuron):
    """Converting units re-labels the neuron without resizing it.

    Regression: `convert_units` is implemented in `BaseNeuron` as `n *= conv`,
    which assumes the Tree/MeshNeuron convention where multiplication scales the
    coordinates and the units move inversely. A VoxelNeuron cannot scale its
    coordinates - they are grid indices - so its `__mul__` scales the voxel size
    instead, and converting used to shrink the neuron by the conversion factor
    (125x for 8 nm -> um) while missing the target unit entirely.
    """
    out = neuron.convert_units("um")

    assert np.allclose(physical_size(out), physical_size(neuron))
    assert np.array_equal(out.voxels, neuron.voxels)
    assert out.units_xyz.units == navis.config.ureg("um").units


def test_convert_units_moves_offset(neuron):
    """The offset is in the same base unit as `.units` and must convert too."""
    out = neuron.convert_units("um")

    # The neuron must not move in absolute space
    assert np.allclose(out.bbox * 1000, neuron.bbox)
    assert np.allclose(out.offset, OFFSET / 1000)


def test_convert_units_roundtrip(neuron):
    out = neuron.convert_units("um").convert_units("nm")

    assert np.allclose(out.bbox, neuron.bbox)
    assert np.allclose(
        out.units_xyz.to("nm").magnitude, neuron.units_xyz.to("nm").magnitude
    )


def test_convert_units_anisotropic():
    """Non-isometric units convert per axis."""
    n = navis.VoxelNeuron(VOXELS.copy(), units=["1 nm", "2 nm", "4 nm"])

    out = n.convert_units("um")

    assert np.allclose(physical_size(out), physical_size(n))
    assert np.allclose(out.units_xyz.magnitude, [0.001, 0.002, 0.004])


def test_convert_units_inplace(neuron):
    ret = neuron.convert_units("um", inplace=True)

    assert ret is neuron
    assert np.allclose(neuron.units_xyz.to("nm").magnitude, 8)


# ---------------------------------------------------------------------------
# Memory guards
# ---------------------------------------------------------------------------


@pytest.fixture
def far_apart():
    """Two voxels at opposite corners: trivial data, astronomical dense grid."""
    return navis.VoxelNeuron(np.array([[0, 0, 0], [100_000, 100_000, 100_000]]))


def test_grid_nbytes_does_not_allocate(far_apart):
    """`.grid_nbytes` reports the projected size without building the grid."""
    # ~1e15 voxels: this must be computed in Python ints, not int64
    assert far_apart.grid_nbytes == 100_001**3 * 8
    assert far_apart.memory_usage() < 1000  # the neuron itself is tiny


def test_grid_guard_raises(far_apart):
    """Materializing an oversized grid raises instead of being OOM-killed."""
    with pytest.raises(MemoryError, match="exceeds navis' limit"):
        far_apart.grid


def test_grid_guard_message_is_actionable(far_apart):
    with pytest.raises(MemoryError) as exc:
        far_apart.grid

    msg = str(exc.value)
    assert "100001x100001x100001" in msg  # what it tried to allocate
    assert "2 non-zero voxels" in msg  # how little data there actually is
    assert "max_grid_size" in msg  # how to override


def test_grid_guard_allows_normal_neurons(neuron):
    """The guard must not get in the way of ordinary neurons."""
    assert neuron.grid.shape == SHAPE
    assert neuron.grid_nbytes == np.prod(SHAPE) * neuron.dtype.itemsize


def test_grid_guard_can_be_disabled(far_apart, monkeypatch):
    monkeypatch.setattr(navis.config, "max_grid_size", None)
    # Still too big to actually allocate, but the check itself is now skipped
    navis.utils.check_grid_size(far_apart.shape, far_apart.dtype)


def test_grid_guard_not_applied_to_grid_backed(grid_neuron, monkeypatch):
    """Grid-backed neurons hand back existing data, so there is nothing to guard."""
    monkeypatch.setattr(navis.config, "max_grid_size", 1)
    assert grid_neuron.grid is grid_neuron._data


def test_voxelize_guard():
    """`voxelize` guards its own allocation - a too-fine pitch used to SIGKILL."""
    m = navis.example_neurons(1, kind="mesh")

    with pytest.raises(MemoryError, match="coarser `pitch`"):
        navis.voxelize(m, pitch=1)

    # ...but a sensible pitch still works
    assert navis.voxelize(m, pitch=100).nnz > 0


# ---------------------------------------------------------------------------
# Round-tripping through the wider API
# ---------------------------------------------------------------------------


def test_voxelize_roundtrip():
    """A voxelized mesh survives conversion back to a mesh and to dotprops."""
    m = navis.example_neurons(1, kind="mesh")
    vx = navis.voxelize(m, pitch="2 microns")

    assert isinstance(vx, navis.VoxelNeuron)
    assert vx.nnz > 0

    mesh = navis.conversion.voxels2mesh(vx)
    assert isinstance(mesh, navis.MeshNeuron)
    assert mesh.n_vertices > 0

    dp = navis.make_dotprops(vx, k=5)
    assert isinstance(dp, navis.Dotprops)
    assert dp.n_points == vx.nnz


# ---------------------------------------------------------------------------
# Explicit conversion between representations
# ---------------------------------------------------------------------------


def test_sparsify_densify_roundtrip():
    """A densify/sparsify round-trip preserves the data and the canvas."""
    # Trailing empty planes: the shape must not shrink to fit the coordinates
    grid = np.zeros((5, 5, 5))
    grid[0, 0, 0] = 5
    grid[1, 2, 3] = 7
    n = navis.VoxelNeuron(grid, units="8 nm", sparsify=False)

    sparse = n.sparsify()
    assert sparse._base_data_type == "voxels"
    assert n._base_data_type == "grid"  # original untouched

    dense = sparse.densify()
    assert dense._base_data_type == "grid"

    assert np.array_equal(n.grid, dense.grid)
    assert n.shape == sparse.shape == dense.shape
    assert n.nnz == sparse.nnz == dense.nnz


def test_sparsify_densify_from_voxels():
    """Same round-trip, starting from a voxel-backed neuron."""
    n = navis.VoxelNeuron(np.array([[0, 0, 0, 3.0], [1, 1, 1, 4.0]]))

    dense = n.densify()
    assert dense._base_data_type == "grid"
    # Grid-backed neurons hold their values in the grid - a leftover `_values`
    # would silently feed into the core hash
    assert not hasattr(dense, "_values")

    sparse = dense.sparsify()
    assert np.array_equal(np.sort(n.values), np.sort(sparse.values))
    assert np.array_equal(n.voxels, sparse.voxels)


def test_sparsify_negative_values():
    """Negative voxels must not desync coordinates from values."""
    grid = np.zeros((3, 3, 3))
    grid[0, 0, 0], grid[1, 1, 1], grid[2, 2, 2] = 5, -2, 3
    n = navis.VoxelNeuron(grid)

    sparse = n.sparsify()
    assert sparse.voxels.shape[0] == sparse.values.shape[0]
    # Conversion is lossless - the negative voxel survives
    assert np.array_equal(n.grid, sparse.densify().grid)


def test_sparsify_densify_inplace_and_noop():
    """`inplace=True` returns None; converting twice is a no-op."""
    grid = np.zeros((3, 3, 3))
    grid[0, 0, 0] = 1
    n = navis.VoxelNeuron(grid)

    assert n.sparsify(inplace=True) is None
    assert n._base_data_type == "voxels"

    # Already sparse: still a no-op, and still respects `inplace`
    assert n.sparsify(inplace=True) is None
    assert n._base_data_type == "voxels"
    assert n.sparsify() is not n


def test_sparsify_preserves_metadata():
    """Units, offset, name and connectors survive the conversion."""
    grid = np.zeros((3, 3, 3))
    grid[0, 0, 0] = 1
    n = navis.VoxelNeuron(grid, units="8 nm", name="test", offset=(1, 2, 3))

    sparse = n.sparsify()
    assert sparse.units == n.units
    assert np.array_equal(sparse.offset, n.offset)
    assert sparse.name == n.name
    assert sparse.dtype == n.dtype
    assert n == sparse


def test_supplied_grid_never_fails_to_rebuild():
    """A grid we were handed fits in memory, so `.grid` must never refuse it."""
    grid = np.zeros((40, 40, 40), dtype=np.float64)
    grid[0, 0, 0] = grid[1, 2, 3] = 1
    n = navis.VoxelNeuron(grid)
    assert n._base_data_type == "voxels"  # auto-sparsified, grid dropped

    navis.config.max_grid_size = 1  # anything at all is now "too big"
    try:
        assert np.array_equal(n.grid, grid)
        # the guarantee has to survive copying and pickling
        assert np.array_equal(n.copy().grid, grid)
        assert np.array_equal(pickle.loads(pickle.dumps(n)).grid, grid)

        # ...but a grid grown past what we were given is guarded again
        grown = navis.VoxelNeuron(grid.copy())
        grown.voxels = np.array([[0, 0, 0], [5000, 5000, 5000]])
        with pytest.raises(MemoryError):
            grown.grid

        # ...as is one that never came from a grid at all
        with pytest.raises(MemoryError):
            navis.VoxelNeuron(np.array([[0, 0, 0], [39, 39, 39]])).grid
    finally:
        navis.config.max_grid_size = 2 * 1024**3


def test_canvas_survives_temp_attr_clearing():
    """`_shape` is a TEMP_ATTR - the canvas must not be cleared along with it."""
    grid = np.zeros((6, 6, 6), dtype=np.float64)
    grid[0, 0, 0] = 5
    grid[1, 2, 3] = 7  # trailing empty planes: canvas > bounding box
    n = navis.VoxelNeuron(grid)
    assert n.shape == grid.shape

    # Each of these clears temp attributes; none may reshape the neuron
    n.offset = np.array([1, 2, 3])
    assert n.shape == grid.shape
    n.threshold(1, inplace=True)
    assert n.shape == grid.shape
    n.normalize(inplace=True)
    assert n.shape == grid.shape
    assert n.copy().shape == grid.shape
    assert pickle.loads(pickle.dumps(n)).shape == grid.shape


def test_flip_roundtrip_survives_clearing():
    """Flipping twice returns the original, even across a temp-attr clear."""
    grid = np.zeros((6, 6, 6), dtype=np.float64)
    grid[0, 0, 0] = 5
    grid[1, 2, 3] = 7
    n = navis.VoxelNeuron(grid)

    n.flip("x", inplace=True)
    n.offset = np.array([0, 0, 0])  # clears temp attrs mid-flight
    n.flip("x", inplace=True)

    assert n.shape == grid.shape
    assert np.array_equal(n.grid, grid)


def test_strip_clears_the_canvas():
    """`strip()` redefines the canvas, so a preserved one must not resurrect it."""
    grid = np.zeros((6, 6, 6), dtype=np.float64)
    grid[0, 0, 0] = 5
    grid[1, 2, 3] = 7
    stripped = navis.VoxelNeuron(grid).strip()

    assert stripped.shape == (2, 3, 4)
    stripped.offset = np.array([9, 9, 9])  # must not spring back to (6, 6, 6)
    assert stripped.shape == (2, 3, 4)


def test_canvas_never_clips_grown_data():
    """A canvas must not hide voxels that have grown past it."""
    grid = np.zeros((6, 6, 6), dtype=np.float64)
    grid[0, 0, 0] = 1
    n = navis.VoxelNeuron(grid)

    n.voxels = np.array([[0, 0, 0], [9, 9, 9]])
    assert n.shape == (10, 10, 10)


# ---------------------------------------------------------------------------
# Binary operations
# ---------------------------------------------------------------------------

@pytest.fixture
def cube():
    """A solid 5x5x5 block of voxels with distinct, all non-zero values.

    Deliberately offset from the origin by one voxel so that dilating it does
    not push coordinates negative - that would shift the whole frame (see
    `test_growing_past_the_origin_shifts_the_frame`) and move every voxel,
    which would get in the way of checking how values are carried.
    """
    coords = np.argwhere(np.ones((5, 5, 5))) + 1
    values = np.arange(1, len(coords) + 1, dtype=float)
    return navis.VoxelNeuron(np.column_stack([coords, values]), units=UNITS)


@pytest.mark.parametrize("op", ["erode", "opening", "thin"])
def test_shrinking_ops_are_subsets_and_keep_values(cube, op):
    """Ops that only remove voxels must carry the survivors' values exactly."""
    out = getattr(cube, op)()

    before = {tuple(v): x for v, x in zip(cube.voxels.tolist(), cube.values)}
    assert out.nnz <= cube.nnz
    for v, val in zip(out.voxels.tolist(), out.values):
        assert tuple(v) in before, "op invented a voxel"
        assert before[tuple(v)] == val, "value was not carried"


def test_dilate_grows_and_fills_new_voxels(cube):
    out = cube.dilate()

    assert out.nnz > cube.nnz
    # every original voxel survives with its own value
    after = {tuple(v): x for v, x in zip(out.voxels.tolist(), out.values)}
    for v, val in zip(cube.voxels.tolist(), cube.values):
        assert after[tuple(v)] == val
    # new voxels default to the neuron's maximum, so a binary neuron stays binary
    new = set(after) - {tuple(v) for v in cube.voxels.tolist()}
    assert {after[v] for v in new} == {cube.values.max()}


def test_dilate_explicit_fill(cube):
    out = cube.dilate(fill=-1)

    old = {tuple(v) for v in cube.voxels.tolist()}
    new_values = [
        val for v, val in zip(out.voxels.tolist(), out.values) if tuple(v) not in old
    ]
    assert set(new_values) == {-1}


def test_dilate_erode_are_inverses_on_a_solid_block(cube):
    """Dilating then eroding a convex block returns the original voxels."""
    out = cube.dilate().erode()

    assert sorted(map(tuple, out.voxels.tolist())) == sorted(
        map(tuple, cube.voxels.tolist())
    )


def test_connectivity_is_passed_through(cube):
    """26-connectivity grows more than 6-connectivity."""
    assert cube.dilate(connectivity=26).nnz > cube.dilate(connectivity=6).nnz


def test_set_algebra():
    a = navis.VoxelNeuron(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), units=UNITS)
    b = navis.VoxelNeuron(np.array([[2, 2, 2], [3, 3, 3]]), units=UNITS)

    def vox(n):
        return sorted(map(tuple, n.voxels.tolist()))

    assert vox(a.union(b)) == [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    assert vox(a.intersection(b)) == [(2, 2, 2)]
    assert vox(a.difference(b)) == [(0, 0, 0), (1, 1, 1)]
    assert vox(a.symmetric_difference(b)) == [(0, 0, 0), (1, 1, 1), (3, 3, 3)]


def test_set_algebra_accepts_raw_voxels():
    a = navis.VoxelNeuron(np.array([[0, 0, 0], [1, 1, 1]]), units=UNITS)

    out = a.difference(np.array([[1, 1, 1]]))
    assert out.voxels.tolist() == [[0, 0, 0]]


def test_union_prefers_own_values():
    """On overlap, this neuron's value wins; the other's voxels get `fill`."""
    a = navis.VoxelNeuron(np.array([[0, 0, 0, 5.0], [1, 1, 1, 6.0]]), units=UNITS)
    b = navis.VoxelNeuron(np.array([[1, 1, 1, 99.0], [2, 2, 2, 99.0]]), units=UNITS)

    out = a.union(b, fill=-1)
    got = {tuple(v): x for v, x in zip(out.voxels.tolist(), out.values)}

    assert got[(0, 0, 0)] == 5.0
    assert got[(1, 1, 1)] == 6.0  # overlap: `a` wins, not 99
    assert got[(2, 2, 2)] == -1  # contributed by `b`


def test_offsets_are_translated_onto_a_common_lattice():
    """Neurons on the same lattice but different origins are aligned, not rejected."""
    a = navis.VoxelNeuron(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), units=UNITS)
    # voxel 0 here sits at 16 nm, which is voxel 2 of `a`
    shifted = navis.VoxelNeuron(
        np.array([[0, 0, 0]]), units=UNITS, offset=np.array([16.0, 16.0, 16.0])
    )

    assert a.intersection(shifted).voxels.tolist() == [[2, 2, 2]]


@pytest.mark.parametrize(
    "other,match",
    [
        (dict(units="4 nm"), "different voxel sizes"),
        (dict(units=UNITS, offset=np.array([4.0, 0, 0])), "non-integer number"),
        (dict(), "incompatible units"),
    ],
)
def test_misaligned_neurons_are_rejected(other, match):
    a = navis.VoxelNeuron(np.array([[0, 0, 0]]), units=UNITS)
    b = navis.VoxelNeuron(np.array([[0, 0, 0]]), **other)

    with pytest.raises(ValueError, match=match):
        a.union(b)


def test_growing_past_the_origin_shifts_the_frame():
    """Dilating a voxel at index 0 must not produce negative coordinates."""
    n = navis.VoxelNeuron(
        np.array([[0, 0, 0]]), units=UNITS, offset=np.array([100.0, 100.0, 100.0])
    )

    out = n.dilate()

    assert (out.voxels >= 0).all()
    # The neuron must not have moved: the original voxel still sits at 100 nm
    assert np.allclose(out.offset + np.array([1, 1, 1]) * 8, [100, 100, 100])


def test_binary_ops_preserve_metadata(cube):
    cube.name = "test"
    out = cube.dilate()

    assert out.name == "test"
    assert out.id == cube.id
    assert out.units == cube.units


@pytest.mark.parametrize("op", ["dilate", "erode", "opening", "closing"])
def test_binary_ops_inplace(cube, op):
    expected = getattr(cube, op)()  # the non-inplace result to compare against

    assert getattr(cube, op)(inplace=True) is None
    assert np.array_equal(cube.voxels, expected.voxels)
    assert np.array_equal(cube.values, expected.values)


def test_binary_ops_work_on_grid_backed(grid_neuron):
    """Grid-backed neurons go through `.voxels` and come back sparse."""
    out = grid_neuron.dilate()

    assert out._base_data_type == "voxels"
    assert out.nnz > grid_neuron.nnz


def test_erode_to_empty_is_not_an_error():
    """A thin neuron can erode away entirely."""
    n = navis.VoxelNeuron(np.array([[1, 1, 1]]), units=UNITS)

    out = n.erode()
    assert out.nnz == 0
    assert len(out.voxels) == 0


# ---------------------------------------------------------------------------
# mesh() shorthand
# ---------------------------------------------------------------------------


def test_mesh_shorthand():
    """`.mesh()` is `navis.mesh()` applied to this neuron."""
    vx = navis.voxelize(navis.example_neurons(1, kind="mesh"), pitch="2 microns")

    m = vx.mesh()

    assert isinstance(m, navis.MeshNeuron)
    assert m.n_vertices > 0
    assert m.n_vertices == navis.mesh(vx).n_vertices


def test_mesh_shorthand_passes_kwargs():
    vx = navis.voxelize(navis.example_neurons(1, kind="mesh"), pitch="2 microns")

    assert vx.mesh(step_size=2).n_vertices != vx.mesh(step_size=1).n_vertices


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


@pytest.fixture
def solid():
    """A solid 4x4x4 block, one voxel off the origin."""
    return navis.VoxelNeuron(np.argwhere(np.ones((4, 4, 4))) + 1, units=UNITS)


def test_volume_is_not_double_scaled(solid, monkeypatch):
    """`volume` must not apply its units twice.

    Regression: the property builds a pint Quantity itself *and* was wrapped in
    `@add_units(power=3)`, so with `config.add_units` on it reported a volume in
    micrometer**6.
    """
    expected = solid.volume

    monkeypatch.setattr(navis.config, "add_units", True)

    assert solid.volume == expected
    assert str(solid.volume.units) == "nanometer ** 3"


def test_volume_counts_voxels(solid):
    # 64 voxels of 8x8x8 nm
    assert solid.volume.to("nm ** 3").magnitude == pytest.approx(64 * 8**3)


def test_surface_area(solid):
    """A solid 4x4x4 block has 6 faces of 4x4 voxels, each 8x8 nm."""
    assert solid.surface_area.to("nm ** 2").magnitude == pytest.approx(6 * 4 * 4 * 8**2)


def test_surface_area_scales_with_units():
    small = navis.VoxelNeuron(np.argwhere(np.ones((4, 4, 4))), units="1 nm")
    big = navis.VoxelNeuron(np.argwhere(np.ones((4, 4, 4))), units="2 nm")

    ratio = big.surface_area.to("nm ** 2") / small.surface_area.to("nm ** 2")
    assert ratio.magnitude == pytest.approx(4)  # area goes with the square


def test_centroid_includes_offset():
    """The centroid is in units and sits inside the bounding box."""
    n = navis.VoxelNeuron(
        np.argwhere(np.ones((4, 4, 4))), units=UNITS, offset=OFFSET.copy()
    )

    centroid = n.centroid
    manual = n.voxels.mean(axis=0) * 8 + OFFSET

    assert np.allclose(centroid, manual)
    assert ((centroid >= n.bbox[:, 0]) & (centroid <= n.bbox[:, 1])).all()


def test_distance_transform_is_row_aligned(solid):
    dist = solid.distance_transform()

    assert dist.shape == (solid.voxels.shape[0],)
    # Defaults to the neuron's units, so a surface voxel is one voxel (8 nm) out
    assert dist.min() == pytest.approx(8.0)
    # ...and asking for voxel counts gives 1
    assert solid.distance_transform(spacing=[1, 1, 1]).min() == pytest.approx(1.0)


def test_connected_components_method(solid):
    labels = solid.connected_components()

    assert labels.shape == (solid.voxels.shape[0],)
    assert len(np.unique(labels)) == 1  # solid block is one component


# ---------------------------------------------------------------------------
# _connected_components / drop_fluff on voxels
# ---------------------------------------------------------------------------


@pytest.fixture
def fluffy():
    """One big blob (64 voxels), one small (8) and one speck (1)."""
    coords = np.vstack(
        [
            np.argwhere(np.ones((4, 4, 4))),
            np.argwhere(np.ones((2, 2, 2))) + 20,
            np.array([[40, 40, 40]]),
        ]
    )
    values = np.arange(1, len(coords) + 1, dtype=float)
    return navis.VoxelNeuron(np.column_stack([coords, values]), units=UNITS)


def test_connected_components_on_voxels(fluffy):
    """Components come back as indices into `.voxels`, biggest-first agnostic."""
    cc = navis.graph.graph_utils._connected_components(fluffy)

    assert sorted(len(c) for c in cc) == [1, 8, 64]
    # every voxel is accounted for exactly once
    assert sorted(np.concatenate(cc).tolist()) == list(range(fluffy.voxels.shape[0]))


def test_connected_components_connectivity(fluffy):
    """`connectivity` is passed through to sparse-cubes."""
    # Two voxels touching only at a corner: one component at 26, two at 6
    diag = navis.VoxelNeuron(np.array([[0, 0, 0], [1, 1, 1]]), units=UNITS)

    cc = navis.graph.graph_utils._connected_components
    assert len(cc(diag, connectivity=26)) == 1
    assert len(cc(diag, connectivity=6)) == 2


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, 64),  # default: largest component only
        (dict(keep_size=5), 72),  # 64 + 8, speck dropped
        (dict(n_largest=2), 72),
        (dict(keep_size=5, n_largest=1), 64),
        (dict(keep_size=0.5), 64),  # fraction of 73 voxels
        (dict(n_largest=3), 73),  # everything
    ],
)
def test_drop_fluff_on_voxels(fluffy, kwargs, expected):
    assert navis.drop_fluff(fluffy, **kwargs).nnz == expected


def test_drop_fluff_carries_values(fluffy):
    out = navis.drop_fluff(fluffy, keep_size=5)

    before = {tuple(v): x for v, x in zip(fluffy.voxels.tolist(), fluffy.values)}
    for v, val in zip(out.voxels.tolist(), out.values):
        assert before[tuple(v)] == val


def test_drop_fluff_inplace_returns_the_neuron(fluffy):
    ret = navis.drop_fluff(fluffy, inplace=True)

    assert ret is fluffy
    assert fluffy.nnz == 64


def test_drop_fluff_preserves_frame(fluffy):
    """Dropping fluff must not move the surviving voxels."""
    out = navis.drop_fluff(fluffy)

    assert np.allclose(out.offset, fluffy.offset)
    assert out.units == fluffy.units


# ---------------------------------------------------------------------------
# thin_voxels backends
# ---------------------------------------------------------------------------


def test_thin_voxels_skimage_backend():
    pytest.importorskip("skimage")
    vx = navis.voxelize(navis.example_neurons(1, kind="mesh"), pitch="2 microns")

    out = navis.thin_voxels(vx, backend="skimage")
    assert 0 < out.nnz < vx.nnz


def test_thin_voxels_sparsecubes_backend():
    vx = navis.voxelize(navis.example_neurons(1, kind="mesh"), pitch="2 microns")

    out = navis.thin_voxels(vx, backend="sparsecubes")

    assert 0 < out.nnz < vx.nnz
    # the sparse backend never densifies, so the result stays voxel-backed
    assert out._base_data_type == "voxels"


def test_thin_voxels_auto_prefers_sparsecubes():
    vx = navis.voxelize(navis.example_neurons(1, kind="mesh"), pitch="2 microns")

    assert navis.thin_voxels(vx)._base_data_type == "voxels"


def test_thin_voxels_2d_falls_back_to_skimage():
    """sparse-cubes is 3D only, so 2D image data must go to scikit-image."""
    pytest.importorskip("skimage")
    img = np.zeros((20, 20), dtype=bool)
    img[5:15, 9:12] = True

    assert navis.thin_voxels(img).sum() > 0


def test_thin_voxels_sparsecubes_rejects_2d():
    """Asking for the sparse backend explicitly on 2D data is an error.

    Note this is gated on 0.4.0: on an older install the *version* complaint
    takes precedence, which is the more useful message of the two.
    """
    img = np.zeros((20, 20), dtype=bool)
    img[5:15, 9:12] = True

    with pytest.raises(ValueError, match="only handles 3D"):
        navis.thin_voxels(img, backend="sparsecubes")


def test_thin_voxels_dense_array_roundtrip():
    """Dense array in, dense array of the same shape out."""
    img = np.zeros((20, 20, 20), dtype=bool)
    img[5:15, 9:12, 9:12] = True

    out = navis.thin_voxels(img, backend="sparsecubes")

    assert out.shape == img.shape
    assert out.dtype == bool
    assert 0 < out.sum() < img.sum()


def test_thin_voxels_rejects_bad_backend():
    vx = navis.VoxelNeuron(VOXELS.copy())

    with pytest.raises(ValueError):
        navis.thin_voxels(vx, backend="nope")


# ---------------------------------------------------------------------------
# navis.mesh on raw voxel arrays
# ---------------------------------------------------------------------------


def test_mesh_accepts_voxel_array():
    """Regression: `navis.mesh` tested `x.ndims`, which numpy spells `x.ndim`,
    so passing the documented (N, 3) array raised AttributeError.
    """
    import trimesh as tm

    voxels = np.argwhere(np.ones((4, 4, 4)))

    m = navis.mesh(voxels)

    assert isinstance(m, tm.Trimesh)
    assert len(m.vertices) > 0


def test_mesh_rejects_other_arrays():
    with pytest.raises(TypeError):
        navis.mesh(np.zeros((4, 4, 4)))


# ---------------------------------------------------------------------------
# Set similarity
# ---------------------------------------------------------------------------


def test_iou_and_dice():
    a = navis.VoxelNeuron(np.array([[0, 0, 0], [1, 1, 1]]), units=UNITS)
    b = navis.VoxelNeuron(np.array([[1, 1, 1], [2, 2, 2]]), units=UNITS)

    # one shared voxel out of three total
    assert a.iou(b) == pytest.approx(1 / 3)
    # dice counts the intersection twice: 2*1 / (2 + 2)
    assert a.dice(b) == pytest.approx(0.5)


def test_iou_bounds():
    a = navis.VoxelNeuron(np.array([[0, 0, 0], [1, 1, 1]]), units=UNITS)

    assert a.iou(a) == pytest.approx(1.0)
    assert a.dice(a) == pytest.approx(1.0)
    assert a.iou(np.array([[9, 9, 9]])) == pytest.approx(0.0)


def test_iou_respects_frame_alignment():
    """Similarity uses the same lattice alignment as the set operations."""
    a = navis.VoxelNeuron(np.array([[0, 0, 0], [1, 1, 1]]), units=UNITS)
    # voxel 0 here is at 8 nm, i.e. voxel [1, 1, 1] of `a`
    shifted = navis.VoxelNeuron(
        np.array([[0, 0, 0]]), units=UNITS, offset=np.array([8.0, 8.0, 8.0])
    )

    assert a.iou(shifted) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="different voxel sizes"):
        a.iou(navis.VoxelNeuron(np.array([[0, 0, 0]]), units="4 nm"))


# ---------------------------------------------------------------------------
# Voxel adjacency / graph conversion
# ---------------------------------------------------------------------------


@pytest.fixture
def blob():
    """A solid 6x6x6 block, off the origin."""
    return navis.VoxelNeuron(np.argwhere(np.ones((6, 6, 6))) + 2, units=UNITS)


@pytest.mark.parametrize("connectivity,metric", [(6, "manhattan"), (26, "chebyshev")])
def test_voxels2edges_indexes_into_voxels(blob, connectivity, metric):
    """Edges must index into `.voxels` - `neuron2igraph` builds a graph with
    one vertex per voxel row, so consumers map vertices straight back.

    Regression guard for the switch to `sparsecubes.edges`, which returns
    indices into its own deduplicated+sorted node array rather than the input.
    """
    from navis.graph.converters import _voxels2edges

    edges = _voxels2edges(blob, connectivity=connectivity)
    delta = np.abs(blob.voxels[edges[:, 0]] - blob.voxels[edges[:, 1]])

    if metric == "manhattan":
        assert (delta.sum(axis=1) == 1).all()
    else:
        assert (delta.max(axis=1) == 1).all()

    # undirected, each edge once, no self-loops
    assert (edges[:, 0] != edges[:, 1]).all()
    assert len(np.unique(np.sort(edges, axis=1), axis=0)) == len(edges)


def test_voxels2edges_survives_unsorted_input():
    """The index mapping must not assume the input is sorted."""
    from navis.graph.converters import _voxels2edges

    coords = np.argwhere(np.ones((4, 4, 4)))
    rng = np.random.RandomState(0)
    shuffled = navis.VoxelNeuron(coords[rng.permutation(len(coords))], units=UNITS)

    edges = _voxels2edges(shuffled, connectivity=6)
    delta = np.abs(shuffled.voxels[edges[:, 0]] - shuffled.voxels[edges[:, 1]])

    assert (delta.sum(axis=1) == 1).all()


def test_voxels2edges_single_voxel():
    from navis.graph.converters import _voxels2edges

    edges = _voxels2edges(navis.VoxelNeuron(np.array([[1, 1, 1]])), connectivity=6)
    assert edges.shape == (0, 2)


def test_neuron2igraph_voxels(blob):
    g = navis.neuron2igraph(blob, connectivity=6)

    assert g.vcount() == blob.nnz
    # a solid block is fully connected
    assert len(g.components(mode="WEAK")) == 1


@pytest.mark.parametrize("connectivity", [6, 26])
def test_neuron2nx_voxels(blob, connectivity):
    """networkx nodes are keyed by coordinate tuple, not index."""
    g = navis.neuron2nx(blob, connectivity=connectivity)

    assert g.number_of_nodes() == blob.nnz
    assert tuple(blob.voxels[0]) in g.nodes
    # more connectivity means more edges
    assert g.number_of_edges() > 0


def test_neuron2nx_connectivity_matters(blob):
    n6 = navis.neuron2nx(blob, connectivity=6).number_of_edges()
    n26 = navis.neuron2nx(blob, connectivity=26).number_of_edges()

    assert n26 > n6


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def test_downsample_voxels(blob):
    before_extent = blob.extents.copy()

    out = navis.downsample_neuron(blob, 2)

    assert out.nnz < blob.nnz
    # voxels got bigger, so units grow to match
    assert np.allclose(out.units_xyz.magnitude, blob.units_xyz.magnitude * 2)
    # ...and the neuron occupies roughly the same space
    assert np.allclose(out.extents, before_extent, rtol=0.25)


def test_downsample_voxels_stays_sparse(blob):
    """Downsampling must not densify - that is the whole point."""
    assert navis.downsample_neuron(blob, 2)._base_data_type == "voxels"


def test_downsample_voxels_carries_values():
    """Values are aggregated (max by default), not dropped."""
    coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    values = np.array([1.0, 9.0, 2.0, 3.0])
    n = navis.VoxelNeuron(np.column_stack([coords, values]), units=UNITS)

    out = navis.downsample_neuron(n, 2)

    assert out.nnz == 2
    assert sorted(out.values.tolist()) == [3.0, 9.0]


def test_downsample_voxels_never_allocates_the_grid():
    """A neuron whose dense grid cannot fit must still be downsamplable.

    Regression: this went through `ndimage.zoom(x.grid, ...)`, so it tripped the
    dense-grid memory guard on exactly the sparse neurons worth downsampling.
    """
    far = navis.VoxelNeuron(np.array([[0, 0, 0], [100_000, 100_000, 100_000]]))
    assert far.grid_nbytes > navis.config.max_grid_size  # would refuse to densify

    out = navis.downsample_neuron(far, 10)

    assert out.voxels.tolist() == [[0, 0, 0], [10_000, 10_000, 10_000]]


def test_downsample_voxels_rounds_non_integer_factor(caplog):
    """Voxel pooling needs whole factors; rounding must be applied consistently.

    The factor scales both the pooling and the units, so using the raw float for
    one and the rounded int for the other would resize the neuron.
    """
    n = navis.VoxelNeuron(np.argwhere(np.ones((8, 8, 8))), units=UNITS)
    before = n.extents.copy()

    out = navis.downsample_neuron(n, 2.5)

    assert np.allclose(out.extents, before)
    assert np.allclose(out.units_xyz.magnitude, 16)  # 8 nm * 2, not * 2.5


# ---------------------------------------------------------------------------
# Skeletonization
# ---------------------------------------------------------------------------


@pytest.fixture
def voxelized_neuron():
    return navis.voxelize(navis.example_neurons(1, kind="mesh"), pitch="2 microns")


@pytest.mark.parametrize("method", ["wavefront", "teasar", "thin"])
def test_voxels2skeleton(voxelized_neuron, method):
    sk = navis.conversion.voxels2skeleton(voxelized_neuron, method=method)

    assert isinstance(sk, navis.TreeNeuron)
    assert sk.n_nodes > 0
    assert not sk.nodes[["x", "y", "z"]].isna().any().any()
    # radii should be meaningful, not all zero
    assert (sk.nodes.radius > 0).any()


@pytest.mark.parametrize("method", ["wavefront", "teasar", "thin"])
def test_skeleton_lands_in_the_right_place(voxelized_neuron, method):
    """The skeleton must sit inside the voxel neuron's bounding box.

    Voxel coordinates are grid indices, so the offset has to be added back
    after scaling - forgetting it puts the skeleton in the wrong place entirely.
    """
    sk = navis.conversion.voxels2skeleton(voxelized_neuron, method=method)

    assert (sk.bbox[:, 0] >= voxelized_neuron.bbox[:, 0] - 1e-6).all()
    assert (sk.bbox[:, 1] <= voxelized_neuron.bbox[:, 1] + 1e-6).all()


def test_skeleton_offset_is_applied():
    """A shifted neuron produces a correspondingly shifted skeleton."""
    coords = np.argwhere(np.ones((3, 3, 12)))
    a = navis.VoxelNeuron(coords, units=UNITS)
    b = navis.VoxelNeuron(coords, units=UNITS, offset=np.array([1000.0, 2000.0, 3000.0]))

    sk_a = navis.conversion.voxels2skeleton(a)
    sk_b = navis.conversion.voxels2skeleton(b)

    assert np.allclose(sk_b.bbox - sk_a.bbox, np.array([[1000.0], [2000.0], [3000.0]]))


def test_skeleton_carries_metadata(voxelized_neuron):
    sk = navis.conversion.voxels2skeleton(voxelized_neuron)

    assert sk.id == voxelized_neuron.id
    assert sk.name == voxelized_neuron.name
    # spacing is baked into the coordinates, so one unit is now one base unit
    assert sk.units_xyz.magnitude.tolist() == [1, 1, 1]
    assert sk.units.units == voxelized_neuron.units.units


def test_skeletonize_dispatches_to_voxels(voxelized_neuron):
    """`navis.skeletonize` used to reject VoxelNeurons outright."""
    sk = navis.skeletonize(voxelized_neuron)

    assert isinstance(sk, navis.TreeNeuron)
    assert sk.n_nodes > 0


def test_skeletonize_method_shorthand(voxelized_neuron):
    assert (
        voxelized_neuron.skeletonize().n_nodes
        == navis.skeletonize(voxelized_neuron).n_nodes
    )


def test_voxels2skeleton_accepts_raw_array(voxelized_neuron):
    sk = navis.conversion.voxels2skeleton(voxelized_neuron.voxels)

    assert isinstance(sk, navis.TreeNeuron)
    assert sk.n_nodes > 0


def test_voxels2skeleton_rejects_empty():
    with pytest.raises(ValueError, match="empty voxel data"):
        navis.conversion.voxels2skeleton(np.zeros((0, 3), dtype=int))


def test_voxels2skeleton_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        navis.conversion.voxels2skeleton(np.zeros((5, 2), dtype=int))


def test_voxels2skeleton_rejects_bad_method(voxelized_neuron):
    with pytest.raises(ValueError):
        navis.conversion.voxels2skeleton(voxelized_neuron, method="nope")


def test_skeletonize_neuronlist(voxelized_neuron):
    nl = navis.NeuronList([voxelized_neuron, voxelized_neuron.copy()])

    out = navis.skeletonize(nl)

    assert isinstance(out, navis.NeuronList)
    assert all(isinstance(n, navis.TreeNeuron) and n.n_nodes > 0 for n in out)


def test_skeletonize_defaults_to_wavefront(voxelized_neuron):
    """Wavefront is the default: fastest, and radii come for free.

    This mirrors `mesh2skeleton`, which also defaults to "wavefront".
    """
    default = navis.conversion.voxels2skeleton(voxelized_neuron)
    explicit = navis.conversion.voxels2skeleton(voxelized_neuron, method="wavefront")

    assert default.n_nodes == explicit.n_nodes
    assert np.allclose(default.nodes.radius, explicit.nodes.radius)


def test_wavefront_radii_are_not_voxel_quantized(voxelized_neuron):
    """Wavefront radii come from ring volumes, so they are sub-voxel.

    The thinning-based backends snap radii to the lattice, which shows up as a
    handful of distinct values; wavefront should give a genuine spread.
    """
    sk = navis.conversion.voxels2skeleton(voxelized_neuron, method="wavefront")

    assert (sk.nodes.radius > 0).all()
    # far more distinct radii than the lattice would allow
    assert sk.nodes.radius.nunique() > 20


def test_wavefront_kwargs_passthrough(voxelized_neuron):
    """`radius_agg` reaches sparse-cubes and changes the radii."""
    default = navis.conversion.voxels2skeleton(voxelized_neuron)
    mean_agg = navis.conversion.voxels2skeleton(
        voxelized_neuron, method="wavefront", radius_agg="mean"
    )

    assert not np.allclose(
        np.sort(default.nodes.radius), np.sort(mean_agg.nodes.radius)
    )


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def test_smooth_voxels_spreads_values(voxelized_neuron):
    """Smoothing blurs the neuron outwards, so it occupies more voxels."""
    out = navis.smooth_voxels(voxelized_neuron, sigma=2)

    assert out.nnz > voxelized_neuron.nnz
    assert np.issubdtype(out.dtype, np.floating)
    # original is untouched
    assert voxelized_neuron.nnz < out.nnz


def test_smooth_voxels_stays_sparse(voxelized_neuron):
    """The default backend must not densify."""
    assert navis.smooth_voxels(voxelized_neuron, sigma=1)._base_data_type == "voxels"


def test_smooth_voxels_matches_scipy_exactly():
    """The sparse filter is scipy's `mode="constant"`, not an approximation."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.RandomState(0)
    grid = np.zeros((30, 30, 30), dtype=np.float64)
    pts = rng.randint(5, 25, size=(40, 3))
    grid[pts[:, 0], pts[:, 1], pts[:, 2]] = rng.rand(40) + 0.5

    n = navis.VoxelNeuron(grid, sparsify=True)
    out = navis.smooth_voxels(n, sigma=2)

    dense = gaussian_filter(grid, sigma=2, mode="constant", cval=0)

    # Smoothing spreads past index 0, so the frame gets shifted and the offset
    # compensated. Map back through the offset to recover original grid indices
    # - this doubles as a check that the compensation is right.
    idx = out.voxels + np.round(out.offset).astype(int)

    # ...and it grows past the far edge too, so compare within the old canvas
    rebuilt = np.zeros_like(dense)
    inb = ((idx >= 0) & (idx < np.array(grid.shape))).all(axis=1)
    keep = idx[inb]
    rebuilt[keep[:, 0], keep[:, 1], keep[:, 2]] = out.values[inb]

    assert np.allclose(rebuilt, dense, atol=1e-6)


def test_smooth_voxels_epsilon_prunes(voxelized_neuron):
    """`epsilon` trades exactness for keeping the cloud small."""
    exact = navis.smooth_voxels(voxelized_neuron, sigma=2)
    pruned = navis.smooth_voxels(voxelized_neuron, sigma=2, epsilon=1e-4)

    assert pruned.nnz < exact.nnz


def test_smooth_voxels_truncate_bounds_growth(voxelized_neuron):
    wide = navis.smooth_voxels(voxelized_neuron, sigma=2, truncate=4.0)
    tight = navis.smooth_voxels(voxelized_neuron, sigma=2, truncate=2.0)

    assert tight.nnz < wide.nnz


def test_smooth_voxels_anisotropic_sigma(voxelized_neuron):
    """`sigma=0` on an axis disables smoothing along it."""
    out = navis.smooth_voxels(voxelized_neuron, sigma=[2, 0, 0])

    # only the x extent should have grown
    assert out.shape[1] == voxelized_neuron.shape[1]
    assert out.shape[2] == voxelized_neuron.shape[2]
    assert out.shape[0] > voxelized_neuron.shape[0]


def test_smooth_voxels_scipy_backend(voxelized_neuron):
    """The scipy backend keeps the canvas and hands back a dense grid."""
    out = navis.smooth_voxels(voxelized_neuron, sigma=2, backend="scipy")

    assert out._base_data_type == "grid"
    assert out.shape == voxelized_neuron.shape  # clipped, not grown


def test_smooth_voxels_inplace(voxelized_neuron):
    before = voxelized_neuron.nnz
    ret = navis.smooth_voxels(voxelized_neuron, sigma=1, inplace=True)

    assert ret is voxelized_neuron
    assert voxelized_neuron.nnz > before


def test_smooth_voxels_rejects_bad_backend(voxelized_neuron):
    with pytest.raises(ValueError):
        navis.smooth_voxels(voxelized_neuron, sigma=1, backend="nope")


def test_smooth_voxels_rejects_non_voxel():
    with pytest.raises(TypeError):
        navis.smooth_voxels(navis.example_neurons(1, kind="mesh"))


def _rasterize(n, shape):
    """Put a (possibly frame-shifted) neuron back onto a grid of `shape`."""
    out = np.zeros(shape, dtype=np.float64)
    idx = n.voxels + np.round(n.offset).astype(int)
    inb = ((idx >= 0) & (idx < np.array(shape))).all(axis=1)
    keep = idx[inb]
    out[keep[:, 0], keep[:, 1], keep[:, 2]] = n.values[inb]
    return out


@pytest.mark.parametrize("touches_edge", [False, True])
@pytest.mark.parametrize("sigma", [1.0, 3.0])
def test_smooth_backends_agree(touches_edge, sigma):
    """Both backends treat outside the neuron as empty, so they must agree.

    The scipy backend used to run with scipy's default `mode="reflect"`, which
    mirrors data at the canvas boundary and invents mass outside the imaged
    volume - that made the two disagree for neurons touching the edge.
    """
    shape = (30, 30, 30)
    rng = np.random.RandomState(0)
    grid = np.zeros(shape)
    lo, hi = (0, 30) if touches_edge else (8, 22)
    pts = rng.randint(lo, hi, size=(60, 3))
    grid[pts[:, 0], pts[:, 1], pts[:, 2]] = rng.rand(60) + 0.5

    n = navis.VoxelNeuron(grid, sparsify=True)

    sparse = _rasterize(navis.smooth_voxels(n, sigma=sigma), shape)
    dense = _rasterize(navis.smooth_voxels(n, sigma=sigma, backend="scipy"), shape)

    assert np.allclose(sparse, dense, atol=1e-6)


def test_smooth_conserves_mass():
    """Constant/zero boundaries mean smoothing redistributes, never creates."""
    grid = np.zeros((30, 30, 30))
    grid[10:20, 10:20, 10:20] = 1.0
    n = navis.VoxelNeuron(grid, sparsify=True)

    out = navis.smooth_voxels(n, sigma=2)

    assert out.values.sum() == pytest.approx(n.values.sum(), rel=1e-4)


def test_smooth_scipy_honours_truncate():
    """`truncate` reaches the scipy backend too, not just sparse-cubes."""
    grid = np.zeros((40, 40, 40))
    grid[20, 20, 20] = 1.0
    n = navis.VoxelNeuron(grid, sparsify=False)

    wide = navis.smooth_voxels(n, sigma=2, backend="scipy", truncate=4.0)
    tight = navis.smooth_voxels(n, sigma=2, backend="scipy", truncate=1.0)

    assert wide.nnz > tight.nnz
