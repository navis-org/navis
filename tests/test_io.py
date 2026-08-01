import navis
import pytest
import tempfile
import numpy as np

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


@pytest.mark.parametrize("filename", ['neurons.zip',
                                      '*.ply'])
def test_mesh_io(filename):
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        # Load example neurons
        n = navis.example_neurons(2, kind='mesh')

        # Save to neurons folder
        if str(filename).endswith('.zip'):
            # Into a zip file
            navis.write_mesh(n, tempdir / 'neurons.zip', filetype='ply')
        else:
            # As individual files
            navis.write_mesh(n, tempdir, filetype='ply')

        # Load again
        if str(filename).endswith('.zip'):
            n2 = navis.read_mesh(tempdir / 'neurons.zip')
        else:
            n2 = navis.read_mesh(tempdir / filename)

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


@pytest.mark.parametrize("suffix", [".rds", ".rda"])
def test_r_data_roundtrip(suffix):
    """Write neurons to R data files and read them back in."""
    nl = navis.example_neurons(2, kind="skeleton")

    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / f"neurons{suffix}"

        if suffix == ".rds":
            navis.write_rds(nl, filepath)
            nl2 = navis.read_rds(filepath)
        else:
            navis.write_rda(nl, filepath)
            nl2 = navis.read_rda(filepath)

        assert len(nl) == len(nl2)

        for n in nl:
            n2 = nl2.idx[str(n.id)]
            assert n.n_nodes == n2.n_nodes
            assert n.n_branches == n2.n_branches
            # nat stores diameters, so radii have to survive the halving
            assert np.allclose(n.nodes.radius.values, n2.nodes.radius.values)
            assert np.allclose(
                n.nodes[["x", "y", "z"]].values, n2.nodes[["x", "y", "z"]].values
            )
            assert np.isclose(n.cable_length, n2.cable_length, rtol=1e-4)


def test_r_data_types():
    """Check that all navis types survive a trip through an .rda file."""
    import pandas as pd

    nl = navis.example_neurons(2, kind="skeleton")
    mesh = navis.example_neurons(1, kind="mesh")
    dps = navis.make_dotprops(nl, k=5)
    vol = navis.example_volume("LH")
    vox = navis.Voxels(
        np.arange(24).reshape(2, 3, 4).astype("float32"),
        offset=(10, 20, 30),
        units="4 nm",
    )
    df = pd.DataFrame(
        {
            "int_na": pd.array([1, 2, None], dtype="Int32"),
            "float_na": [1.5, np.nan, 3.0],
            "str_na": pd.array(["a", None, "c"], dtype="string"),
            "big_int": [2**40, 1, 2],
        }
    )

    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "data.rda"
        navis.write_rda(
            {
                "neurons": nl,
                "mesh": mesh,
                "dps": dps,
                "vol": vol,
                "vox": vox,
                "table": df,
            },
            filepath,
        )

        data = navis.read_rda(filepath, neurons_only=False, combine=False)

    assert isinstance(data["vol"], navis.Volume)
    assert np.allclose(data["vol"].vertices, vol.vertices)
    assert np.array_equal(data["vol"].faces, vol.faces)

    assert len(data["neurons"]) == 2
    assert len(data["dps"]) == 2
    assert np.allclose(data["dps"].idx[str(nl[0].id)].points, dps[0].points)
    assert np.allclose(data["dps"].idx[str(nl[0].id)].vect, dps[0].vect)


@pytest.mark.parametrize(
    "keys",
    [
        ("dps",),
        ("dps", "vol"),
        ("neurons", "dps"),
        ("neurons", "dps", "vol"),
        ("vox", "dps"),
        ("mesh", "dps"),
    ],
)
def test_r_data_object_order(keys):
    """Objects must not depend on each other's R symbol definitions.

    Wrapped values (dotprops' `points`/`vect`, voxel grids) carry attributes
    that navis merges into the ones `rdata` generates. Getting that merge wrong
    writes a reference to a symbol that never lands in the file, which only
    shows up for *some* orderings - e.g. dotprops before any plain 2d array.
    """
    nl = navis.example_neurons(2, kind="skeleton")
    objects = {
        "neurons": nl,
        "dps": navis.make_dotprops(nl, k=5),
        "vol": navis.example_volume("LH"),
        "mesh": navis.example_neurons(1, kind="mesh"),
        "vox": navis.Voxels(
            np.zeros((2, 3, 4), dtype="float32"), offset=(0, 0, 0), units="1 nm"
        ),
    }

    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "data.rda"
        navis.write_rda({k: objects[k] for k in keys}, filepath)
        data = navis.read_rda(filepath, neurons_only=False, combine=False)

    assert set(data) == set(keys)


def test_r_data_write_edge_cases():
    """Corner cases that used to trip up the R writer."""
    nl = navis.example_neurons(2, kind="skeleton")

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        # Empty neuronlist
        navis.write_rds(navis.NeuronList([]), tempdir / "empty.rds")

        # Neuron without radii
        no_radius = nl[0].copy()
        no_radius.nodes.drop(columns=["radius"], inplace=True)
        navis.write_rds(no_radius, tempdir / "no_radius.rds")

        # Fragmented neuron (multiple trees)
        frag = nl[0].copy()
        frag.nodes.loc[100, "parent_id"] = -1
        frag._clear_temp_attr()
        navis.write_rds(frag, tempdir / "frag.rds")

        # Duplicate IDs must be made unique
        dupes = navis.NeuronList([nl[0], nl[0]])
        navis.write_rds(dupes, tempdir / "dupes.rds")
        assert len(navis.read_rds(tempdir / "dupes.rds")) == 2

        # Dotprops without tangent vectors nor k
        with pytest.raises(ValueError):
            navis.write_rds(
                navis.Dotprops(np.random.rand(10, 3), k=None), tempdir / "bad.rds"
            )

        # `name` and dict are mutually exclusive
        with pytest.raises(ValueError):
            navis.write_rda({"a": nl}, tempdir / "x.rda", name="b")


def _have_nat():
    """Check whether R and the `nat` package are available."""
    import shutil
    import subprocess

    if not shutil.which("Rscript"):
        return False
    try:
        return subprocess.run(
            ["Rscript", "-e", 'stopifnot(requireNamespace("nat", quietly=TRUE))'],
            capture_output=True,
            timeout=120,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _have_nat(), reason="requires R with the `nat` package")
def test_r_data_readable_by_nat():
    """The whole point of writing .rds/.rda: check that nat groks the result."""
    import subprocess

    nl = navis.example_neurons(2, kind="skeleton")
    dps = navis.make_dotprops(nl, k=5)
    vol = navis.example_volume("LH")

    # nat cares about cable length, so we compare against navis'
    expected = [f"{n.cable_length:.1f}" for n in nl]

    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "data.rda"
        navis.write_rda({"neurons": nl, "dps": dps, "vol": vol}, filepath)

        script = f"""
        suppressMessages(library(nat))
        load("{filepath}")
        stopifnot(inherits(neurons, "neuronlist"), length(neurons) == {len(nl)})
        stopifnot(inherits(dps, "neuronlist"), inherits(dps[[1]], "dotprops"))
        stopifnot(inherits(vol, "mesh3d"))
        # Exercise a few functions that rely on nat's internal representation
        invisible(resample(neurons[[1]], 1000))
        invisible(prune_strahler(neurons[[1]]))
        invisible(as.ngraph(neurons[[1]]))
        cat(sprintf("%.1f", summary(neurons)$cable.length), sep="\\n")
        """
        proc = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, timeout=300
        )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == expected, proc.stdout


@pytest.mark.skipif(not _have_nat(), reason="requires R with the `nat` package")
def test_r_data_nat_roundtrip():
    """navis -> nat -> navis on nat's own data, checked against nat itself."""
    import subprocess

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        original = tempdir / "original.rds"
        roundtrip = tempdir / "roundtrip.rds"

        subprocess.run(
            [
                "Rscript",
                "-e",
                f'suppressMessages(library(nat)); saveRDS(Cell07PNs, "{original}")',
            ],
            capture_output=True,
            check=True,
            timeout=300,
        )

        nl = navis.read_rds(original)
        assert len(nl) == 40
        # Some of nat's own neurons have a cyclic node table (the root points
        # back at its child) - reading has to repair that from the seglists
        assert all(len(n.root) == 1 for n in nl)

        navis.write_rds(nl, roundtrip)

        script = f"""
        suppressMessages(library(nat))
        orig <- Cell07PNs
        rt <- readRDS("{roundtrip}")
        stopifnot(identical(names(orig), names(rt)))
        stopifnot(identical(summary(orig)$nodes, summary(rt)$nodes))
        stopifnot(identical(summary(orig)$branchpoints, summary(rt)$branchpoints))
        stopifnot(identical(summary(orig)$endpoints, summary(rt)$endpoints))
        stopifnot(max(abs(xyzmatrix(orig) - xyzmatrix(rt))) == 0)
        # Segment order is arbitrary, so compare the seglists as sets
        segs <- function(n) sort(sapply(n$SegList, paste, collapse="-"))
        stopifnot(all(sapply(seq_along(orig), function(i)
            identical(segs(orig[[i]]), segs(rt[[i]])))))
        # Cable length has to match what nat itself derives from the seglists
        stopifnot(max(abs(sapply(orig, function(n) sum(seglengths(n)))
                          - sapply(rt, function(n) sum(seglengths(n))))) < 1e-4)
        cat("OK")
        """
        proc = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, timeout=300
        )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("OK")
