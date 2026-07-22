"""Tests for `navis.cast_neuron`.

The function converts the *spatial* data of a neuron to a given dtype. What that
means is per-type, so each neuron class gets its own test. The recurring theme is
the distinction between data and indices: mesh faces, voxel coordinates and
node/parent IDs index into the converted data and must survive untouched.
Downcasting them is not merely useless but actively destructive - `uint8` voxel
coordinates wrap around for any grid wider than 256 voxels.
"""

import navis
import numpy as np
import pytest


@pytest.fixture
def skeleton():
    n = navis.example_neurons(1, kind="skeleton")
    return navis.cast_neuron(n, np.float64)


@pytest.fixture
def mesh():
    return navis.example_neurons(1, kind="mesh")


@pytest.fixture
def dotprops(skeleton):
    return navis.make_dotprops(skeleton, k=5)


def test_treeneuron(skeleton):
    conv = navis.cast_neuron(skeleton, np.float32)

    for c in ("x", "y", "z", "radius"):
        assert conv.nodes[c].dtype == np.float32

    # IDs index into the node table and must not be touched
    assert conv.nodes.node_id.dtype == skeleton.nodes.node_id.dtype
    assert conv.nodes.parent_id.dtype == skeleton.nodes.parent_id.dtype

    # Topology survives
    assert conv.n_nodes == skeleton.n_nodes
    assert np.isclose(conv.cable_length, skeleton.cable_length, rtol=1e-4)


def test_connectors(skeleton):
    assert skeleton.has_connectors
    conv = navis.cast_neuron(skeleton, np.float32)

    for c in ("x", "y", "z"):
        assert conv.connectors[c].dtype == np.float32


def test_meshneuron(mesh):
    conv = navis.cast_neuron(mesh, np.float32)

    assert conv.vertices.dtype == np.float32
    # Faces are indices into `vertices`
    assert conv.faces.dtype == mesh.faces.dtype

    assert np.isclose(conv.volume, mesh.volume, rtol=1e-4)


def test_dotprops(dotprops):
    conv = navis.cast_neuron(dotprops, np.float32)

    assert conv.points.dtype == np.float32
    assert conv.vect.dtype == np.float32
    assert conv.alpha.dtype == np.float32


def test_dotprops_integer_keeps_vectors(dotprops):
    """Tangents are unit vectors and alphas live in [0, 1] - casting either to an
    integer type would zero them out, so they are left as-is."""
    conv = navis.cast_neuron(dotprops, np.int32)

    assert conv.points.dtype == np.int32
    assert conv.vect.dtype == dotprops.vect.dtype
    assert conv.alpha.dtype == dotprops.alpha.dtype


def test_voxelneuron_grid():
    grid = (np.random.rand(20, 20, 20) > 0.8).astype(np.float64)
    n = navis.VoxelNeuron(grid)
    assert n._base_data_type == "grid"

    conv = navis.cast_neuron(n, np.float32)

    assert conv.dtype == np.float32
    assert conv.shape == n.shape
    assert conv.nnz == n.nnz


def test_voxelneuron_sparse(skeleton):
    n = navis.voxelize(skeleton, pitch="2 microns")
    assert n._base_data_type == "voxels"

    conv = navis.cast_neuron(n, np.uint8)

    assert conv.dtype == np.uint8
    # Voxel coordinates are indices - downcasting them to uint8 would wrap
    assert conv._data.dtype == n._data.dtype
    assert conv.nnz == n.nnz
    assert np.array_equal(conv.voxels, n.voxels)


def test_copy_vs_inplace(skeleton):
    conv = navis.cast_neuron(skeleton, np.float32)
    assert conv is not skeleton
    assert skeleton.nodes.x.dtype == np.float64

    out = navis.cast_neuron(skeleton, np.float32, inplace=True)
    assert out is skeleton
    assert skeleton.nodes.x.dtype == np.float32


def test_neuronlist(skeleton, mesh):
    nl = navis.NeuronList([skeleton, mesh])

    conv = navis.cast_neuron(nl, np.float32)
    assert conv[0].nodes.x.dtype == np.float32
    assert conv[1].vertices.dtype == np.float32
    # Originals untouched
    assert nl[0].nodes.x.dtype == np.float64
    assert nl[1].vertices.dtype == np.float64

    navis.cast_neuron(nl, np.float32, inplace=True)
    assert nl[0].nodes.x.dtype == np.float32
    assert nl[1].vertices.dtype == np.float32


@pytest.mark.parametrize("dtype", ["str", np.bool_, object])
def test_rejects_non_numeric(skeleton, dtype):
    with pytest.raises(ValueError):
        navis.cast_neuron(skeleton, dtype)


def test_rejects_non_neuron():
    with pytest.raises(TypeError):
        navis.cast_neuron("not a neuron", np.float32)
