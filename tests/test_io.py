import navis
import pytest
import struct
import tempfile
import numpy as np

from pathlib import Path


def _can_write_r():
    """Check whether the installed `rdata` can write .rds/.rda files."""
    import rdata

    # Writing arrived in rdata 1.0, which requires Python >= 3.11. On 3.10 we
    # install the last version that runs there, which can only read.
    return hasattr(rdata, "unparser")


needs_r_writer = pytest.mark.skipif(
    not _can_write_r(),
    reason="writing .rds/.rda requires rdata >= 1.0, i.e. Python >= 3.11",
)


@pytest.mark.parametrize(
    "filename", ["", "{neuron.id}.swc", "neurons.zip", "{neuron.id}.swc@neurons.zip"]
)
def test_swc_io(filename):
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / filename

        # Load example neurons
        n = navis.example_neurons(2, kind="skeleton")

        # Save to file / folder
        navis.write_swc(n, filepath)

        # Load again
        if str(filepath).endswith(".zip"):
            n2 = navis.read_swc(Path(tempdir) / "neurons.zip")
        else:
            n2 = navis.read_swc(tempdir)

        # Assert that we loaded the same number of neurons
        assert len(n) == len(n2)


@pytest.mark.parametrize("filename", ["", "neurons.zip", "{neuron.id}@neurons.zip"])
def test_precomputed_skeleton_io(filename):
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / filename

        # Load example neurons
        n = navis.example_neurons(2, kind="skeleton")

        # Save to file / folder
        navis.write_precomputed(n, filepath, radius=True)

        # Load again
        if str(filepath).endswith(".zip"):
            n2 = navis.read_precomputed(Path(tempdir) / "neurons.zip")
        else:
            n2 = navis.read_precomputed(tempdir)

        # Assert that we loaded the same number of neurons
        assert len(n) == len(n2)


@pytest.mark.parametrize("flip", [False, True])
def test_precomputed_skeleton_edge_order(flip):
    """Precomputed edges are undirected: both column orders must read the same.

    A branch point repeats in whichever column holds the parents, and the reader
    used to lose every repeat - so a file written "backwards" fell apart into as
    many fragments as it had branches.

    """
    # A stick with a fork at node 2, written as (parent, child)
    edges = np.array([[0, 1], [1, 2], [2, 3], [2, 4]], dtype=np.uint32)
    if flip:
        edges = edges[:, ::-1]
    vertices = np.arange(15, dtype=np.float32).reshape(5, 3)

    buf = struct.pack("<II", len(vertices), len(edges))
    buf += vertices.tobytes() + edges.tobytes()

    sk = navis.read_precomputed(buf, datatype="skeleton")

    assert sk.n_trees == 1
    assert len(sk.root) == 1
    # Rooted at the free end of the stick, whichever way the file is written
    assert sk.root[0] == 0
    assert sk.n_branches == 1


@pytest.mark.parametrize("filename", ["", "neurons.zip", "{neuron.id}@neurons.zip"])
def test_precomputed_mesh_io(filename):
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / filename

        # Load example neurons
        n = navis.example_neurons(2, kind="mesh")

        # Save to file / folder
        navis.write_precomputed(n, filepath, write_manifest=True)

        # Load again
        if str(filepath).endswith(".zip"):
            n2 = navis.read_precomputed(Path(tempdir) / "neurons.zip")
        else:
            n2 = navis.read_precomputed(tempdir)

        # Assert that we loaded the same number of neurons
        assert len(n) == len(n2)


@pytest.mark.parametrize("filename", ["neurons.zip", "*.ply"])
def test_mesh_io(filename):
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        # Load example neurons
        n = navis.example_neurons(2, kind="mesh")

        # Save to neurons folder
        if str(filename).endswith(".zip"):
            # Into a zip file
            navis.write_mesh(n, tempdir / "neurons.zip", filetype="ply")
        else:
            # As individual files
            navis.write_mesh(n, tempdir, filetype="ply")

        # Load again
        if str(filename).endswith(".zip"):
            n2 = navis.read_mesh(tempdir / "neurons.zip")
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


@needs_r_writer
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


@needs_r_writer
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
@needs_r_writer
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


@needs_r_writer
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


def test_r_data_read_fixture():
    """Read a pre-written .rda file.

    Every other R test writes before it reads, so they all skip where `rdata`
    is too old to write (Python 3.10). This one keeps the reading side covered
    there - see `tests/fixtures/r_data/generate.py` for how the file is made.
    """
    filepath = Path(__file__).parent / "fixtures" / "r_data" / "objects.rda"

    data = navis.read_rda(filepath, neurons_only=False, combine=False)

    assert set(data) == {"neurons", "dps", "vol"}

    neurons = data["neurons"]
    assert isinstance(neurons, navis.NeuronList) and len(neurons) == 2
    for n in neurons:
        assert isinstance(n, navis.Skeleton)
        assert n.n_nodes > 0
        assert n.cable_length > 0
        assert n.n_connectors > 0

    dps = data["dps"]
    assert isinstance(dps, navis.NeuronList) and len(dps) == 2
    for dp in dps:
        assert isinstance(dp, navis.Dotprops)
        assert dp.points.shape[1] == 3
        assert dp.vect.shape == dp.points.shape

    vol = data["vol"]
    assert isinstance(vol, navis.Volume)
    assert vol.vertices.shape[1] == 3
    assert vol.faces.shape[1] == 3


@pytest.mark.skipif(
    _can_write_r(), reason="only meaningful where `rdata` is too old to write"
)
def test_r_data_write_needs_new_rdata():
    """Writing with an old `rdata` must fail with a message that helps."""
    import sys

    nl = navis.example_neurons(1, kind="skeleton")

    with tempfile.TemporaryDirectory() as tempdir:
        with pytest.raises(ImportError, match=r"requires `rdata` >= 1\.0\.0"):
            navis.write_rds(nl, Path(tempdir) / "neurons.rds")

        if sys.version_info < (3, 11):
            # Telling these users to upgrade `rdata` would send them in circles
            with pytest.raises(ImportError, match=r"requires Python >= 3\.11"):
                navis.write_rds(nl, Path(tempdir) / "neurons.rds")


def _have_nat():
    """Check whether R and the `nat` package are available."""
    import shutil
    import subprocess

    if not shutil.which("Rscript"):
        return False
    try:
        return (
            subprocess.run(
                ["Rscript", "-e", 'stopifnot(requireNamespace("nat", quietly=TRUE))'],
                capture_output=True,
                timeout=120,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@needs_r_writer
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


@needs_r_writer
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


def _sorted_connectors(n):
    cols = ["connector_id", "node_id", "type", "x", "y", "z"]
    return n.connectors[cols].sort_values(cols[:3]).reset_index(drop=True)


def test_parquet_roundtrip():
    """Skeletons - including their connectors - must survive a round-trip."""
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "skeletons.parquet"

        nl = navis.example_neurons(3, kind="skeleton")
        navis.write_parquet(nl, filepath)

        # Connectors go into a sidecar file
        assert (Path(tempdir) / "skeletons.connectors.parquet").is_file()

        nl2 = navis.read_parquet(filepath)
        assert len(nl2) == len(nl)

        for n2 in nl2:
            n = nl.idx[n2.id]
            assert n.n_nodes == n2.n_nodes
            assert np.array_equal(
                n.nodes.sort_values("node_id")[
                    ["node_id", "parent_id", "x", "y", "z"]
                ].values,
                n2.nodes.sort_values("node_id")[
                    ["node_id", "parent_id", "x", "y", "z"]
                ].values,
            )
            assert n.name == n2.name
            assert str(n.units) == str(n2.units)
            # This is the bit that used to silently go missing
            assert n2.connectors is not None
            assert n.n_connectors == n2.n_connectors
            assert _sorted_connectors(n).equals(_sorted_connectors(n2))


def test_parquet_single_neuron():
    """A file with a single neuron must read back as a single neuron."""
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "skeleton.parquet"

        n = navis.example_neurons(1, kind="skeleton")
        navis.write_parquet(n, filepath)

        n2 = navis.read_parquet(filepath)
        assert isinstance(n2, navis.Skeleton)
        assert n.n_nodes == n2.n_nodes
        assert n.n_connectors == n2.n_connectors

        # ... but subsetting always gives a NeuronList
        nl = navis.read_parquet(filepath, subset=[n2.id])
        assert isinstance(nl, navis.NeuronList) and len(nl) == 1


def test_parquet_no_connectors():
    """`write_connectors=False` must not leave a stale sidecar behind."""
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "skeletons.parquet"
        sidecar = Path(tempdir) / "skeletons.connectors.parquet"

        nl = navis.example_neurons(2, kind="skeleton")
        navis.write_parquet(nl, filepath)
        assert sidecar.is_file()

        navis.write_parquet(nl, filepath, write_connectors=False)
        assert not sidecar.is_file()
        assert all(n.connectors is None for n in navis.read_parquet(filepath))

        # Neurons without connectors don't produce a sidecar in the first place
        for n in nl:
            n.connectors = None
        navis.write_parquet(nl, filepath)
        assert not sidecar.is_file()


def test_parquet_dotprops_roundtrip():
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "dotprops.parquet"

        dp = navis.make_dotprops(navis.example_neurons(2, kind="skeleton"), k=5)
        navis.write_parquet(dp, filepath)

        dp2 = navis.read_parquet(filepath)
        assert len(dp2) == len(dp)
        for n2 in dp2:
            n = dp.idx[n2.id]
            assert np.allclose(n.points, n2.points)
            assert np.allclose(n.vect, n2.vect)
            assert n.k == n2.k


@pytest.mark.parametrize("kind", ["skeleton", "dotprops"])
def test_parquet_neurarrow_roundtrip(kind):
    """Neurons written to the neurarrow spec must read back in.

    Note that neurarrow has no place for navis' scale factor (e.g. "8
    nanometer"), so coordinates come back converted into the base unit.
    """
    pq = pytest.importorskip("pyarrow.parquet")

    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "neurons.parquet"

        nl = navis.example_neurons(2, kind="skeleton")
        if kind == "dotprops":
            nl = navis.make_dotprops(nl, k=5)
        navis.write_parquet(nl, filepath, format="neurarrow", context="test")

        schema = pq.read_schema(filepath)
        meta = {k.decode(): v.decode() for k, v in schema.metadata.items()}
        assert meta["context"] == "test"
        assert meta["unit"] == "nanometer"
        # Required fields must be non-nullable and correctly typed
        for field in ("sample_id", "fragment_id"):
            assert schema.field(field).type == "uint64"
            assert not schema.field(field).nullable
        for field in ("x", "y", "z"):
            assert schema.field(field).type == "double"
            assert not schema.field(field).nullable
        assert "neuron" not in schema.names and "node_id" not in schema.names

        table = pq.read_table(filepath).to_pandas()
        # Sample IDs must be unique across the whole file
        assert table.sample_id.nunique() == len(table)

        if kind == "skeleton":
            # Roots are encoded as a null parent - one per neuron
            assert schema.field("parent_id").nullable
            assert table.parent_id.isnull().sum() == len(nl)
        else:
            assert meta["neighborhood_size"] == "5"

        nl2 = navis.read_parquet(filepath)
        assert len(nl2) == len(nl)
        for n2 in nl2:
            n = nl.idx[n2.id]
            scale = n.units.to("nm").magnitude
            if kind == "skeleton":
                assert np.array_equal(n.nodes.node_id.values, n2.nodes.node_id.values)
                assert np.array_equal(
                    n.nodes.parent_id.values, n2.nodes.parent_id.values
                )
                assert np.allclose(
                    n.nodes[["x", "y", "z"]].values * scale,
                    n2.nodes[["x", "y", "z"]].values,
                )
                assert n.n_connectors == n2.n_connectors
            else:
                assert np.allclose(n.points * scale, n2.points)
                assert np.allclose(n.vect, n2.vect)
                assert n.k == n2.k


def test_parquet_neurarrow_rejects_mixed_units():
    """neurarrow tracks units per file, so they have to be homogeneous."""
    with tempfile.TemporaryDirectory() as tempdir:
        nl = navis.example_neurons(2, kind="skeleton")
        nl[0].units = "1 micron"

        with pytest.raises(ValueError, match="share the same units"):
            navis.write_parquet(
                nl, Path(tempdir) / "neurons.parquet", format="neurarrow"
            )


def test_parquet_unknown_format():
    with tempfile.TemporaryDirectory() as tempdir:
        nl = navis.example_neurons(1, kind="skeleton")
        with pytest.raises(ValueError, match="format"):
            navis.write_parquet(nl, Path(tempdir) / "n.parquet", format="swc")


def test_parquet_scan():
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "skeletons.parquet"

        nl = navis.example_neurons(3, kind="skeleton")
        navis.write_parquet(nl, filepath)

        scan = navis.scan_parquet(filepath)
        assert set(scan.id) == set(nl.id)
        assert "name" in scan.columns
        assert "units" in scan.columns

        # `limit` reads the first N neurons off the back of the scan
        assert len(navis.read_parquet(filepath, limit=2)) == 2


def test_parquet_legacy_files():
    """Files written before connectors/`label` were added must still read."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    old_columns = ("node_id", "x", "y", "z", "radius", "parent_id", "neuron")

    def old_write(x, filepath):
        """The pre-sidecar writer, verbatim."""
        nodes = x.nodes[x.nodes.columns[np.isin(x.nodes.columns, old_columns)]]
        table = pa.Table.from_pandas(nodes)
        metadata = {}
        for n in navis.NeuronList(x):
            metadata[f"{n.id}:id"] = str(n.id)
            for p in ("name", "units", "soma"):
                if getattr(n, p, None):
                    metadata[f"{n.id}:{p}"] = str(getattr(n, p, None))
        schema = pa.schema(
            [table.schema.field(i) for i in range(len(table.schema))],
            metadata=metadata,
        )
        pq.write_table(table.cast(schema), filepath)

    with tempfile.TemporaryDirectory() as tempdir:
        nl = navis.example_neurons(3, kind="skeleton")

        multi = Path(tempdir) / "old_multi.parquet"
        old_write(nl, multi)
        nl2 = navis.read_parquet(multi)
        assert len(nl2) == 3
        for n2 in nl2:
            n = nl.idx[n2.id]
            assert n.n_nodes == n2.n_nodes
            assert n.name == n2.name
            assert str(n.units) == str(n2.units)
            assert n2.connectors is None  # these files never had any

        assert set(navis.scan_parquet(multi).id) == set(nl.id)
        assert len(navis.read_parquet(multi, limit=2)) == 2

        # A single neuron used to be written without the `neuron` column
        single = Path(tempdir) / "old_single.parquet"
        old_write(nl[0], single)
        assert "neuron" not in pq.read_schema(single).names
        n2 = navis.read_parquet(single)
        assert isinstance(n2, navis.Skeleton)
        assert n2.n_nodes == nl[0].n_nodes


def test_parquet_read_foreign_neurarrow():
    """Read a neurarrow file written by some other tool."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    with tempfile.TemporaryDirectory() as tempdir:
        filepath = Path(tempdir) / "foreign.skeletons.parquet"

        # Two little trees. No navis meta data whatsoever, roots as null
        # parents, SWC structure IDs via the net.clbarnes.swc extension.
        parent = np.array([0, 1, 2, 2, 0, 10, 11], dtype=np.uint64)
        is_root = np.array([True, False, False, False, True, False, False])
        xyz = np.arange(21, dtype=np.float64).reshape(7, 3)

        schema = pa.schema(
            [
                pa.field("sample_id", pa.uint64(), False),
                pa.field("fragment_id", pa.uint64(), False),
                pa.field("x", pa.float64(), False),
                pa.field("y", pa.float64(), False),
                pa.field("z", pa.float64(), False),
                pa.field("parent_id", pa.uint64(), True),
                pa.field("net.clbarnes.swc:type_id", pa.int64(), False),
            ],
            metadata={
                "version": "0.2.1",
                "context": "some-uuid",
                "unit": "micrometer",
                "net.clbarnes.swc:version": "0.1",
            },
        )
        pq.write_table(
            pa.Table.from_arrays(
                [
                    pa.array(np.array([1, 2, 3, 4, 10, 11, 12], dtype=np.uint64)),
                    pa.array(np.array([7, 7, 7, 7, 9, 9, 9], dtype=np.uint64)),
                    pa.array(xyz[:, 0]),
                    pa.array(xyz[:, 1]),
                    pa.array(xyz[:, 2]),
                    pa.array(parent, mask=is_root, type=pa.uint64()),
                    pa.array(np.array([1, 3, 3, 3, 1, 3, 3], dtype=np.int64)),
                ],
                schema=schema,
            ),
            filepath,
        )

        nl = navis.read_parquet(filepath)
        assert len(nl) == 2
        assert set(nl.id) == {7, 9}

        n = nl.idx[7]
        assert np.array_equal(n.nodes.node_id.values, [1, 2, 3, 4])
        # Null parents become navis' -1 roots
        assert np.array_equal(n.nodes.parent_id.values, [-1, 1, 2, 2])
        assert n.n_trees == 1
        # The file-level unit applies to every neuron
        assert str(n.units) == "1 micrometer"
        assert np.array_equal(n.nodes.label.values, [1, 3, 3, 3])

        # Without per-neuron meta data, scanning falls back to the ID column
        assert set(navis.scan_parquet(filepath).id) == {7, 9}
        assert len(navis.read_parquet(filepath, subset=[9])) == 1
