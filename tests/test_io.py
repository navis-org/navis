import navis
import pytest
import tempfile
import numpy as np
import pandas as pd

from pathlib import Path


@pytest.mark.parametrize("filename", ['', '{neuron.id}.swc',
                                      'neurons.zip',
                                      '{neuron.id}.swc@neurons.zip'])
def test_swc_io(filename):
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / filename

        # Load example neurons
        n = navis.example_neurons(2, kind='skeleton')

        # Save to file / folder
        navis.write_swc(n, filepath)

        # Load again
        if str(filepath).endswith('.zip'):
            n2 = navis.read_swc(Path(tempdir) / 'neurons.zip')
        else:
            n2 = navis.read_swc(tempdir)

        # Assert that we loaded the same number of neurons
        assert len(n) == len(n2)


@pytest.fixture
def simple_neuron():
    """Neuron with 1 branch and no connectors.

    [3]
     |
     2 -- 4
     |
     1
    """
    nrn = navis.TreeNeuron(None)
    dtypes = {
        "node_id": np.uint64,
        "parent_id": np.int64,
        "x": float,
        "y": float,
        "z": float,
    }
    df = pd.DataFrame([
        [1, 2, 0, 2, 0],
        [2, 3, 0, 1, 0],
        [3, -1, 0, 0, 0],  # root
        [4, 2, 1, 1, 0]
    ], columns=list(dtypes)).astype(dtypes)
    nrn.nodes = df
    return nrn


def assert_parent_defined(df: pd.DataFrame):
    defined = set()
    for node, _structure, _x, _y, _z, _r, parent in df.itertuples(index=False):
        defined.add(node)
        if parent == -1:
            continue
        assert parent in defined, f"Child {node} has undefined parent {parent}"


def test_swc_io_order(simple_neuron):
    df = navis.io.swc_io.make_swc_table(simple_neuron, True)
    assert_parent_defined(df)


@pytest.mark.parametrize("filename", ['',
                                      'neurons.zip',
                                      '{neuron.id}@neurons.zip'])
def test_precomputed_skeleton_io(filename):
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / filename

        # Load example neurons
        n = navis.example_neurons(2, kind='skeleton')

        # Save to file / folder
        navis.write_precomputed(n, filepath, radius=True)

        # Load again
        if str(filepath).endswith('.zip'):
            n2 = navis.read_precomputed(Path(tempdir) / 'neurons.zip')
        else:
            n2 = navis.read_precomputed(tempdir)

        # Assert that we loaded the same number of neurons
        assert len(n) == len(n2)


@pytest.mark.parametrize("filename", ['',
                                      'neurons.zip',
                                      '{neuron.id}@neurons.zip'])
def test_precomputed_mesh_io(filename):
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / filename

        # Load example neurons
        n = navis.example_neurons(2, kind='mesh')

        # Save to file / folder
        navis.write_precomputed(n, filepath, write_manifest=True)

        # Load again
        if str(filepath).endswith('.zip'):
            n2 = navis.read_precomputed(Path(tempdir) / 'neurons.zip')
        else:
            n2 = navis.read_precomputed(tempdir)

        # Assert that we loaded the same number of neurons
        assert len(n) == len(n2)


def test_read_nrrd(voxel_nrrd_path):
    navis.read_nrrd(voxel_nrrd_path, output="voxels", errors="raise")


def test_roundtrip_nrrd(voxel_nrrd_path):
    vneuron = navis.read_nrrd(voxel_nrrd_path, output="voxels", errors="raise")
    outpath = voxel_nrrd_path.parent / "written.nrrd"
    navis.write_nrrd(vneuron, outpath)
    vneuron2 = navis.read_nrrd(outpath, output="voxels", errors="raise")
    assert np.allclose(vneuron._data, vneuron2._data)
    assert np.allclose(vneuron.units_xyz.magnitude, vneuron2.units_xyz.magnitude)
    assert vneuron.units_xyz.units == vneuron2.units_xyz.units
