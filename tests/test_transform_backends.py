"""Tests for the CMTK/elastix transform backends.

navis can run CMTK and elastix point transforms either by shelling out to the
external binaries (`streamxform`, `transformix`) or - if navis-fastcore is
installed - via its in-process Rust implementation.

Two kinds of test live here:

* Backend *plumbing* (selection, invertibility gating, copying, pickling). These
  build small synthetic registrations and need no binaries and no downloads, so
  they run anywhere navis-fastcore is installed.
* Backend *parity* - the two implementations must agree numerically. These need
  both the binaries and real registrations, so they skip unless you have them.
  Note the parity tests are the point of the exercise: run them locally.

NB: navis-fastcore only grew `ElastixTransform`/`CmtkRegistration` after 0.6.1,
so an older fastcore will skip every fastcore test here rather than fail.

"""

import pathlib
import pickle

import numpy as np
import pytest

import navis
from navis import utils
from navis.transforms import backends
from navis.transforms.cmtk import CMTKtransform, _cmtkbin
from navis.transforms.elastix import ElastixTransform, _elastixbin
from navis.transforms.moving_least_squares import MovingLeastSquaresTransform
from navis.transforms.thinplate import TPStransform

HAS_FASTCORE = backends.fastcore_transforms_available()

needs_fastcore = pytest.mark.skipif(
    not HAS_FASTCORE, reason="navis-fastcore has no CMTK/elastix transforms"
)
needs_cmtk = pytest.mark.skipif(not _cmtkbin, reason="CMTK binaries not found")
needs_elastix = pytest.mark.skipif(not _elastixbin, reason="elastix binaries not found")


@pytest.fixture(autouse=True)
def clean_backend_state():
    """Backend config and the parse cache are global - reset between tests."""
    yield
    navis.config.default_transform_backend = "auto"
    navis.config.elastix_invertible = False
    backends.clear_transform_cache()
    navis.transforms.registry.clear_caches()


@pytest.fixture
def no_fastcore(monkeypatch):
    """Force the binary path."""
    # Note: must patch the module attribute (not re-import the name) - that's
    # how the backend looks it up.
    monkeypatch.setattr(utils, "fastcore", None)
    backends.clear_transform_cache()


@pytest.fixture
def tiny_cmtk(tmp_path):
    """A minimal affine-only CMTK registration."""
    reg = tmp_path / "tiny.list"
    reg.mkdir()
    (reg / "registration").write_text(
        "! TYPEDSTREAM 1.1\n\n"
        "registration {\n"
        '\treference_study "ref.nrrd"\n'
        '\tfloating_study "flt.nrrd"\n'
        "\taffine_xform {\n"
        "\t\txlate 10 -5 2\n"
        "\t\trotate 0 0 5\n"
        "\t\tscale 1.1 0.9 1\n"
        "\t\tshear 0 0 0\n"
        "\t\tcenter 50 50 25\n"
        "\t}\n"
        "}\n"
    )
    return reg


@pytest.fixture
def tiny_elastix(tmp_path):
    """A minimal translation-only elastix transform."""
    fp = tmp_path / "TransformParameters.0.txt"
    fp.write_text(
        '(Transform "TranslationTransform")\n'
        "(NumberOfParameters 3)\n"
        "(TransformParameters 10.0 -5.0 2.0)\n"
        '(InitialTransformParametersFileName "NoInitialTransform")\n'
        '(HowToCombineTransforms "Compose")\n'
        "(FixedImageDimension 3)\n"
        "(MovingImageDimension 3)\n"
    )
    return fp


POINTS = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 50.0, 25.0]])


# --------------------------------------------------------------- backend selection


@needs_fastcore
def test_auto_prefers_fastcore(tiny_cmtk):
    navis.config.default_transform_backend = "auto"
    assert CMTKtransform(tiny_cmtk).backend == "fastcore"


def test_auto_falls_back_to_binary(tiny_cmtk, no_fastcore):
    navis.config.default_transform_backend = "auto"
    assert CMTKtransform(tiny_cmtk).backend == "binary"


def test_explicit_fastcore_raises_when_unavailable(tiny_cmtk, no_fastcore):
    navis.config.default_transform_backend = "fastcore"
    with pytest.raises(ValueError, match="not installed or too old"):
        CMTKtransform(tiny_cmtk).backend


def test_unknown_backend_raises(tiny_cmtk):
    with pytest.raises(ValueError, match="Unknown transform backend"):
        CMTKtransform(tiny_cmtk, backend="nonsense").backend


@needs_fastcore
def test_per_instance_overrides_config(tiny_cmtk):
    navis.config.default_transform_backend = "fastcore"
    assert CMTKtransform(tiny_cmtk, backend="binary").backend == "binary"


@needs_fastcore
def test_backend_resolved_lazily(tiny_cmtk):
    """Backend must follow a *later* config change.

    Template packages (flybrains et al.) build their transforms at import time,
    i.e. before a user can touch the config - so snapshotting the backend in
    __init__ would make `navis.config.default_transform_backend = ...` a no-op.
    """
    tr = CMTKtransform(tiny_cmtk)
    assert tr.backend == "fastcore"

    navis.config.default_transform_backend = "binary"
    assert tr.backend == "binary"


# ------------------------------------------------------------------ invertibility


@pytest.fixture
def add_chain_elastix(tmp_path):
    """A two-step elastix chain combined via "Add" - which cannot be inverted."""
    base = tmp_path / "base.txt"
    base.write_text(
        '(Transform "TranslationTransform")\n'
        "(NumberOfParameters 3)\n"
        "(TransformParameters 10.0 -5.0 2.0)\n"
        '(InitialTransformParametersFileName "NoInitialTransform")\n'
        '(HowToCombineTransforms "Compose")\n'
        "(FixedImageDimension 3)\n"
        "(MovingImageDimension 3)\n"
    )
    child = tmp_path / "child.txt"
    child.write_text(
        '(Transform "TranslationTransform")\n'
        "(NumberOfParameters 3)\n"
        "(TransformParameters 1.0 1.0 1.0)\n"
        f'(InitialTransformParametersFileName "{base.name}")\n'
        '(HowToCombineTransforms "Add")\n'
        "(FixedImageDimension 3)\n"
        "(MovingImageDimension 3)\n"
    )
    return child


@needs_fastcore
def test_elastix_can_invert_only_with_fastcore(tiny_elastix):
    assert ElastixTransform(tiny_elastix, backend="fastcore").can_invert is True
    assert ElastixTransform(tiny_elastix, backend="binary").can_invert is False


@needs_fastcore
def test_elastix_add_chain_is_not_invertible(add_chain_elastix):
    """`can_invert` is honest per *file*, not just per backend.

    An "Add" chain has no inverse. We must say so up front rather than let a
    ValueError surface from Rust half way through a bridging sequence.
    """
    tr = ElastixTransform(add_chain_elastix, backend="fastcore")
    assert tr.can_invert is False

    with pytest.raises(NotImplementedError, match="Add"):
        -tr

    tr._invert = True
    assert "Add" in tr.check_if_possible(on_error="ignore")


@needs_fastcore
def test_elastix_resolves_initial_transform_by_basename(tmp_path):
    """fastcore follows a chained transform even when the recorded path is stale.

    elastix routinely records an absolute path from the author's machine. Falling
    back to the basename alongside the naming file is exactly what `copy_files`
    achieves by copying everything into one directory - so it is not needed here.
    """
    base = tmp_path / "base_affine.txt"
    base.write_text(
        '(Transform "TranslationTransform")\n'
        "(NumberOfParameters 3)\n"
        "(TransformParameters 10.0 -5.0 2.0)\n"
        '(InitialTransformParametersFileName "NoInitialTransform")\n'
        '(HowToCombineTransforms "Compose")\n'
        "(FixedImageDimension 3)\n"
        "(MovingImageDimension 3)\n"
    )
    child = tmp_path / "child.txt"
    child.write_text(
        '(Transform "TranslationTransform")\n'
        "(NumberOfParameters 3)\n"
        "(TransformParameters 1.0 1.0 1.0)\n"
        '(InitialTransformParametersFileName "/gone/from/this/machine/base_affine.txt")\n'
        '(HowToCombineTransforms "Compose")\n'
        "(FixedImageDimension 3)\n"
        "(MovingImageDimension 3)\n"
    )

    # No `copy_files`, and the recorded path does not exist - must still chain.
    out = ElastixTransform(child, backend="fastcore").xform(np.zeros((1, 3)))
    assert np.allclose(out, [[11.0, -4.0, 3.0]])


@needs_fastcore
def test_elastix_graph_invertibility_is_opt_in(tiny_elastix):
    """`-tr` always works on fastcore, but the *graph* only inverts if asked.

    Every elastix registration navis knows about ships with a purpose-built
    reverse, so letting the graph invert adds no connectivity - it just drags
    unrelated routes through a cheaper-looking parallel edge.
    """
    tr = ElastixTransform(tiny_elastix, backend="fastcore")

    navis.config.elastix_invertible = False
    assert tr.can_invert is True
    assert tr.invertible is False  # ... but the graph won't route through it

    navis.config.elastix_invertible = True
    assert tr.invertible is True


def test_elastix_neg_raises_without_fastcore(tiny_elastix, no_fastcore):
    with pytest.raises(NotImplementedError, match="requires navis-fastcore"):
        -ElastixTransform(tiny_elastix)


@needs_fastcore
def test_elastix_neg_roundtrips(tiny_elastix):
    tr = ElastixTransform(tiny_elastix, backend="fastcore")
    xf = tr.xform(POINTS)
    assert np.allclose((-tr).xform(xf), POINTS, atol=1e-4)


@needs_fastcore
def test_elastix_not_equal_to_its_inverse(tiny_elastix):
    """Else the registry would dedup a forward and an inverse edge into one."""
    tr = ElastixTransform(tiny_elastix, backend="fastcore")
    assert tr != -tr
    assert tr == tr.copy()


@needs_fastcore
def test_double_negation_is_identity(tiny_elastix):
    tr = ElastixTransform(tiny_elastix, backend="fastcore")
    assert -(-tr) == tr


def test_is_invertible_helper(tiny_elastix, tiny_cmtk):
    from navis.transforms.base import is_invertible

    # CMTK is invertible either way; elastix only on fastcore
    assert is_invertible(CMTKtransform(tiny_cmtk)) is True
    assert (
        is_invertible(ElastixTransform(tiny_elastix, backend="binary")) is False
    )


@needs_fastcore
@pytest.mark.parametrize("opt_in", [False, True])
def test_registry_honours_elastix_flag(tiny_elastix, opt_in):
    """The registry's invertibility snapshot must follow the flag."""
    navis.config.default_transform_backend = "fastcore"
    navis.config.elastix_invertible = opt_in

    reg = navis.transforms.templates.TemplateRegistry(scan_paths=False)
    reg.register_transform(
        ElastixTransform(tiny_elastix), source="A", target="B", transform_type="bridging"
    )
    assert reg.transforms[0].invertible is opt_in


# ---------------------------------------------------------------------- plumbing


@needs_fastcore
def test_copy_preserves_backend(tiny_cmtk, tiny_elastix):
    assert CMTKtransform(tiny_cmtk, backend="binary").copy().backend == "binary"

    tr = -ElastixTransform(tiny_elastix, backend="fastcore")
    assert tr.copy()._invert is True
    assert tr.copy().backend == "fastcore"


@needs_fastcore
def test_append_refuses_to_merge_across_backends(tiny_cmtk):
    """Must raise NotImplementedError - that's what makes TransformSequence chain."""
    a = CMTKtransform(tiny_cmtk, backend="fastcore")
    b = CMTKtransform(tiny_cmtk, backend="binary")
    with pytest.raises(NotImplementedError):
        a.append(b)

    # ... and a sequence of the two must therefore keep them separate
    seq = navis.transforms.base.TransformSequence(a, b)
    assert len(seq.transforms) == 2


@needs_fastcore
def test_append_merges_within_a_backend(tiny_cmtk):
    a = CMTKtransform(tiny_cmtk, backend="fastcore")
    b = CMTKtransform(tiny_cmtk, backend="fastcore")
    seq = navis.transforms.base.TransformSequence(a, b)
    assert len(seq.transforms) == 1
    assert len(seq.transforms[0].regs) == 2


@needs_fastcore
def test_pickle_carries_no_rust_object(tiny_cmtk):
    """The parsed registration is cached module-side, never on the instance.

    If it ever ends up in __dict__ the transforms get expensive to ship to
    multiprocessing workers.
    """
    tr = CMTKtransform(tiny_cmtk, backend="fastcore")
    tr.xform(POINTS)  # populate the cache

    blob = pickle.dumps(tr)
    assert len(blob) < 2000
    assert pickle.loads(blob) == tr


@needs_fastcore
def test_cache_is_not_confused_by_direction(tiny_cmtk):
    """Cache keys are derived from live state, so a mutated transform can't go stale."""
    fwd = CMTKtransform(tiny_cmtk, directions="forward", backend="fastcore")
    inv = CMTKtransform(tiny_cmtk, directions="inverse", backend="fastcore")

    a = fwd.xform(POINTS)
    b = inv.xform(a)
    assert np.allclose(b, POINTS, atol=1e-4)
    assert not np.allclose(a, POINTS, atol=1e-4)


@needs_fastcore
def test_return_logs_is_binary_only(tiny_elastix):
    with pytest.raises(ValueError, match="binary"):
        ElastixTransform(tiny_elastix, backend="fastcore").xform(
            POINTS, return_logs=True
        )


@needs_fastcore
def test_missing_file_is_reported(tmp_path):
    tr = ElastixTransform(tmp_path / "nope.txt", backend="fastcore")
    assert "not found" in tr.check_if_possible(on_error="ignore")
    with pytest.raises(BaseException, match="not found"):
        tr.xform(POINTS)


@needs_fastcore
def test_fastcore_needs_no_binaries(tiny_cmtk, tiny_elastix, monkeypatch):
    """The whole point: point transforms without CMTK/elastix installed."""
    monkeypatch.setattr("navis.transforms.cmtk._cmtkbin", None)
    monkeypatch.setattr("navis.transforms.elastix._elastixbin", None)

    CMTKtransform(tiny_cmtk, backend="fastcore").xform(POINTS)
    ElastixTransform(tiny_elastix, backend="fastcore").xform(POINTS)


@needs_fastcore
def test_image_transforms_still_need_cmtk(tiny_cmtk, monkeypatch):
    """fastcore does points, not images - `xform_image` must still demand CMTK."""
    monkeypatch.setattr("navis.transforms.cmtk._cmtkbin", None)
    tr = CMTKtransform(tiny_cmtk, backend="fastcore")

    with pytest.raises(ValueError, match="Cannot find CMTK"):
        tr.xform_image(np.zeros((4, 4, 4)), target=(4, 4, 4, 1, 1, 1))
    with pytest.raises(ValueError, match="Cannot find CMTK"):
        tr.to_dfield((4, 4, 4, 1, 1, 1))


# ------------------------------------------------------------------------ parity
#
# These are the tests that actually matter: the Rust and binary implementations
# must agree. They need real registrations - skipped if you don't have them.

FLYBRAIN_DATA = pathlib.Path("~/flybrain-data").expanduser()
CMTK_REG = FLYBRAIN_DATA / "BridgingRegistrations" / "JFRC2_FCWB.list"

try:
    import flybrains

    ELASTIX_TP = (
        pathlib.Path(flybrains.__file__).parent
        / "data"
        / "FANC_JRCVNC2018F"
        / "TransformParameters.FixedFANC.txt"
    )
except ImportError:
    ELASTIX_TP = None

has_cmtk_reg = pytest.mark.skipif(
    not CMTK_REG.is_dir(), reason="no CMTK registration available"
)
has_elastix_tp = pytest.mark.skipif(
    ELASTIX_TP is None or not ELASTIX_TP.is_file(),
    reason="no elastix transform available",
)


def _brain_points(n=500, seed=0):
    return np.random.default_rng(seed).uniform([50, 50, 20], [300, 400, 150], (n, 3))


@needs_fastcore
@needs_cmtk
@has_cmtk_reg
@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize(
    "kwargs", [{}, {"affine_only": True}, {"affine_fallback": True}]
)
def test_cmtk_parity(direction, kwargs):
    pts = _brain_points()

    binary = CMTKtransform(CMTK_REG, directions=direction, backend="binary")
    fast = CMTKtransform(CMTK_REG, directions=direction, backend="fastcore")

    a, b = binary.xform(pts, **kwargs), fast.xform(pts, **kwargs)

    # Points that fail to transform must fail on *both* sides - CMTK reports
    # them as FAILED, fastcore as NaN.
    assert np.array_equal(np.isnan(a), np.isnan(b))

    m = ~np.isnan(a).any(axis=1)
    assert np.allclose(a[m], b[m], atol=1e-4)


CMTK_REG_B = FLYBRAIN_DATA / "BridgingRegistrations" / "FCWB_JFRC2.list"
has_cmtk_chain = pytest.mark.skipif(
    not (CMTK_REG.is_dir() and CMTK_REG_B.is_dir()),
    reason="no CMTK registrations available",
)


@needs_fastcore
@needs_cmtk
@has_cmtk_chain
@pytest.mark.parametrize(
    "directions",
    [
        ["forward", "inverse"],
        ["forward", "forward"],
        ["inverse", "inverse"],
    ],
)
def test_cmtk_chain_affine_fallback_matches_binary(directions):
    """The affine fallback on a *chain* must fall back over the whole chain.

    i.e. a point that fails anywhere is re-run affine-only through the *entire*
    chain from its original position, which is what `streamxform --affine-only`
    does. fastcore also offers a per-hop fallback ("hop"), which keeps the warps
    a point did clear - arguably nicer, but not what CMTK does, so navis must not
    silently get it. This pins that we ask for, and get, the faithful one.

    It is not a corner case: ~half of navis' bridging paths merge several CMTK
    registrations into a single transform.
    """
    pts = _brain_points(2000)
    regs = [CMTK_REG, CMTK_REG_B]

    a = CMTKtransform(regs, directions=directions, backend="binary").xform(
        pts, affine_fallback=True
    )
    b = CMTKtransform(regs, directions=directions, backend="fastcore").xform(
        pts, affine_fallback=True
    )

    assert np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a).any(axis=1)
    assert np.allclose(a[m], b[m], atol=1e-4)


@needs_fastcore
@has_cmtk_chain
def test_one_parse_serves_every_direction():
    """Direction is a traversal property, so all directions share a cache entry."""
    backends.clear_transform_cache()
    for directions in (
        ["forward", "forward"],
        ["forward", "inverse"],
        ["inverse", "inverse"],
    ):
        CMTKtransform(
            [CMTK_REG, CMTK_REG_B], directions=directions, backend="fastcore"
        ).xform(POINTS)

    assert backends.get_cmtk_reg.cache_info().misses == 1


@needs_fastcore
@has_cmtk_reg
@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_cmtk_affine_fallback_leaves_no_nans(direction):
    """`affine_fallback` must rescue every failed point - in *both* directions.

    fastcore used to attach the fallback affine only to the forward warp, so this
    silently did nothing on an inverted registration and navis quietly dropped
    points that the binary path recovered.
    """
    pts = _brain_points(2000)
    tr = CMTKtransform(CMTK_REG, directions=direction, backend="fastcore")

    assert np.isnan(tr.xform(pts)).any(), "expected some points to fall outside"
    assert not np.isnan(tr.xform(pts, affine_fallback=True)).any()


@needs_fastcore
@needs_elastix
@has_elastix_tp
def test_elastix_parity():
    pts = np.random.default_rng(0).uniform(
        [5000, 5000, 2000], [40000, 80000, 15000], (200, 3)
    )
    a = ElastixTransform(ELASTIX_TP, backend="binary").xform(pts)
    b = ElastixTransform(ELASTIX_TP, backend="fastcore").xform(pts)
    assert np.allclose(a, b, atol=1e-4)


@needs_fastcore
@needs_elastix
@has_elastix_tp
def test_elastix_parity_out_of_bounds():
    """transformix passes out-of-grid points through unchanged; so must we."""
    oob = np.array([[-5e5, -5e5, -5e5], [9e5, 9e5, 9e5]])
    a = ElastixTransform(ELASTIX_TP, backend="binary").xform(oob)
    b = ElastixTransform(ELASTIX_TP, backend="fastcore").xform(oob)
    assert np.allclose(a, b, atol=1e-4)
    assert np.allclose(b, oob, atol=1e-4)


# ------------------------------------------------------------------ bridging graph


@pytest.fixture
def flybrains_registry():
    fb = pytest.importorskip("flybrains")
    return fb


# The paths navis picks today. Making elastix invertible adds a numerically
# inverted edge alongside every purpose-built reverse registration - none of
# these may start routing through one.
BASELINE_PATHS = {
    ("JRC2018F", "BANCum"): ["JRC2018F", "BANCum"],
    ("BANCum", "JRC2018F"): ["BANCum", "JRC2018F"],
    ("FANC", "JRCVNC2018F"): [
        "FANC",
        "FANCum_fixed",
        "JRCVNC2018F_reflected",
        "JRCVNC2018F",
    ],
    ("JRCVNC2018F", "FANC"): [
        "JRCVNC2018F",
        "JRCVNC2018F_reflected",
        "FANCum_fixed",
        "FANC",
    ],
    ("JFRC2", "FCWB"): ["JFRC2", "FCWB"],
}


@needs_fastcore
@pytest.mark.parametrize("backend", ["binary", "fastcore"])
@pytest.mark.parametrize("pair,expected", list(BASELINE_PATHS.items()))
def test_bridging_paths_unchanged(flybrains_registry, backend, pair, expected):
    navis.config.default_transform_backend = backend
    navis.transforms.registry.clear_caches()

    path, _ = navis.transforms.registry.find_bridging_path(*pair)
    assert list(path) == expected


def _all_routes():
    """Every bridging route in the registry, as the transforms it would use."""
    reg = navis.transforms.registry
    reg.clear_caches()

    def descr(t):
        if isinstance(t, ElastixTransform):
            return f"Elastix({t.file.name},inv={t._invert})"
        if isinstance(t, CMTKtransform):
            return f"CMTK({[str(r) for r in t.regs]},{list(t.directions)})"
        return type(t).__name__

    nodes = sorted(reg.bridging_graph(inverse_weight=1).nodes)
    routes = {}
    for source in nodes:
        for target in nodes:
            if source == target:
                continue
            try:
                _, transforms = reg.find_bridging_path(
                    source, target, inverse_weight=1
                )
            except Exception:
                continue
            routes[(source, target)] = [descr(t) for t in transforms]
    return routes


@needs_fastcore
def test_switching_backend_does_not_reroute(flybrains_registry):
    """The Rust backend must be a drop-in: same routes, same transforms.

    This is the regression test that matters. It walks *every* pair of templates
    in the registry (~3900 routes), not a hand-picked few - picking a different
    transform anywhere would silently change people's coordinates.
    """
    navis.transforms.set_transform_backend("binary")
    binary = _all_routes()

    navis.transforms.set_transform_backend("fastcore")
    fastcore = _all_routes()

    assert set(binary) == set(fastcore)
    differing = {k for k in binary if binary[k] != fastcore[k]}
    assert not differing, f"{len(differing)} routes changed, e.g. {list(differing)[:3]}"


@needs_fastcore
def test_elastix_inversion_changes_nothing(flybrains_registry):
    """Allowing the graph to invert elastix must not disturb any existing route.

    Every elastix registration ships with a purpose-built reverse, so the inverse
    edges should be pure dead weight: no new connectivity, and never chosen.

    This used to fail badly - inverse edges either got picked over the dedicated
    registration next to them, or (weighted the other way) looked cheap and dragged
    ~800 unrelated routes through them. Both were symptoms of `weight` being asked
    to decide which transform to use as well as what a hop costs; `_pick_edge` now
    decides the former on its own.
    """
    navis.transforms.set_transform_backend("fastcore", elastix_invertible=False)
    without = _all_routes()

    navis.transforms.set_transform_backend("fastcore", elastix_invertible=True)
    with_inv = _all_routes()

    assert set(with_inv) == set(without), "expected no new connectivity"

    changed = {k for k in without if without[k] != with_inv[k]}
    assert not changed, f"{len(changed)} routes re-routed, e.g. {list(changed)[:3]}"

    used = [k for k, v in with_inv.items() if any("inv=True" in t for t in v)]
    assert not used, f"{len(used)} routes used an inverted elastix over a dedicated one"


def test_forward_edge_beats_an_inverse_one_regardless_of_weight():
    """A forward registration must win even when the inverse edge outweighs it.

    This is the crux of the old contradiction: `nx.shortest_path` *minimises*
    weight, so to stop an inverse edge attracting unrelated routes you have to
    weight it *up* - but the old selection took the *highest*-weight edge, so
    weighting it up made the graph choose it over the dedicated registration.
    Preference no longer rides on weight, so both can now hold at once.
    """
    from navis.transforms.affine import AffineTransform
    from navis.transforms.templates import TemplateRegistry

    forward = AffineTransform(np.eye(4))            # the dedicated A -> B
    other = AffineTransform(np.diag([2, 2, 2, 1]))  # B -> A; its inverse also does A -> B

    reg = TemplateRegistry(scan_paths=False)
    reg.register_transform(
        forward, source="A", target="B", transform_type="bridging", weight=1
    )
    # Deliberately give the inverse edge a *heavier* weight than the forward one.
    reg.register_transform(
        other, source="B", target="A", transform_type="bridging", weight=1, weight_inv=10
    )

    _, transforms = reg.find_bridging_path("A", "B", inverse_weight=1)
    assert np.allclose(transforms[0].matrix, np.eye(4)), "picked the inverse edge"


def _two_forward_edges(w_good, w_poor):
    """Registry with two parallel A->B registrations of differing weight."""
    from navis.transforms.affine import AffineTransform
    from navis.transforms.templates import TemplateRegistry

    good = AffineTransform(np.eye(4))
    poor = AffineTransform(np.diag([9.0, 9.0, 9.0, 1.0]))

    reg = TemplateRegistry(scan_paths=False)
    reg.register_transform(
        poor, source="A", target="B", transform_type="bridging", weight=w_poor
    )
    reg.register_transform(
        good, source="A", target="B", transform_type="bridging", weight=w_good
    )
    return reg


def test_lower_weight_wins():
    """Lower weight = more likely to be used. Everywhere, including selection.

    `nx.shortest_path` costs the hop at min(weights), i.e. it plans the route
    assuming the *cheap* edge. Selection must therefore hand back that same edge -
    otherwise the route is chosen on an assumption the transform then violates.
    """
    reg = _two_forward_edges(w_good=0.5, w_poor=1.0)
    _, transforms = reg.find_bridging_path("A", "B")
    assert np.allclose(transforms[0].matrix, np.eye(4)), "picked the heavier edge"


def test_prefer_forward_can_be_turned_off():
    """`prefer_forward=False` takes the weights at face value."""
    from navis.transforms.affine import AffineTransform
    from navis.transforms.templates import TemplateRegistry

    forward = AffineTransform(np.eye(4))
    other = AffineTransform(np.diag([2.0, 2.0, 2.0, 1.0]))

    reg = TemplateRegistry(scan_paths=False)
    reg.register_transform(
        forward, source="A", target="B", transform_type="bridging", weight=1
    )
    # Its inverse also gets A -> B, and we make it *cheaper* than the forward.
    reg.register_transform(
        other, source="B", target="A", transform_type="bridging", weight=1, weight_inv=0.1
    )

    # On by default: the purpose-built registration wins despite being dearer.
    _, trs = reg.find_bridging_path("A", "B")
    assert np.allclose(trs[0].matrix, np.eye(4))

    # Off: weights are taken at face value, so the cheaper inverse wins.
    _, trs = reg.find_bridging_path("A", "B", prefer_forward=False)
    assert not np.allclose(trs[0].matrix, np.eye(4))


def test_inverse_weight_factor_and_its_override():
    """A transform says how dear it is to invert; `weight_inv` overrides it."""
    from navis.transforms.affine import AffineTransform
    from navis.transforms.templates import TemplateRegistry

    class DearToInvert(AffineTransform):
        inverse_weight_factor = 5

    reg = TemplateRegistry(scan_paths=False)
    reg.register_transform(
        DearToInvert(np.eye(4)), source="A", target="B", transform_type="bridging", weight=2
    )
    reg.register_transform(
        DearToInvert(np.eye(4)),
        source="C",
        target="D",
        transform_type="bridging",
        weight=2,
        weight_inv=0.5,  # explicit -> must ignore inverse_weight_factor
    )

    by_edge = {(t.source, t.target): t for t in reg.transforms}
    assert by_edge[("A", "B")].weight_inv == 10  # 2 * 5
    assert by_edge[("C", "D")].weight_inv == 0.5


def test_reciprocal_is_deprecated():
    from navis.transforms.templates import TemplateRegistry

    reg = TemplateRegistry(scan_paths=False)
    with pytest.warns(DeprecationWarning, match="inverse_weight"):
        reg.bridging_graph(reciprocal=0.5)


@needs_fastcore
@pytest.mark.parametrize("elastix_invertible", [False, True])
def test_bridging_graph_never_parses_a_transform(
    flybrains_registry, elastix_invertible
):
    """Building the graph asks every transform whether it is invertible.

    That question must never turn into a full parse: flybrains registers 50+
    registrations at 3-37 ms each, so a parsing probe would add ~1 s to
    `import flybrains`. With elastix inversion off we don't touch the disk at
    all; with it on we use fastcore's header-only probe.
    """
    import time

    navis.transforms.set_transform_backend(
        "fastcore", elastix_invertible=elastix_invertible
    )
    backends.clear_transform_cache()
    navis.transforms.registry.clear_caches()

    start = time.perf_counter()
    navis.transforms.registry.bridging_graph()
    elapsed = time.perf_counter() - start

    assert backends.get_elastix_transform.cache_info().misses == 0, "graph parsed a file"
    assert backends.get_cmtk_reg.cache_info().misses == 0, "graph parsed a registration"
    assert elapsed < 0.5

    # ... and with inversion off we shouldn't even probe
    if not elastix_invertible:
        assert backends.elastix_is_invertible.cache_info().misses == 0


@needs_fastcore
@needs_cmtk
@needs_elastix
@pytest.mark.parametrize(
    "source,target", [("JRCVNC2018F", "FANC"), ("FANC", "JRCVNC2018F")]
)
def test_xform_brain_parity(flybrains_registry, source, target):
    """End-to-end: the two backends must transform neurons the same way."""
    pts = _brain_points(200) if source != "FANC" else np.random.default_rng(0).uniform(
        [1e5, 1e5, 1e4], [5e5, 2e5, 1e5], (200, 3)
    )

    navis.transforms.set_transform_backend("binary")
    a = navis.xform_brain(pts, source=source, target=target)

    navis.transforms.set_transform_backend("fastcore")
    b = navis.xform_brain(pts, source=source, target=target)

    assert np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a).any(axis=1)
    # Tolerance is relative: FANC is in nm, so absolute errors scale with it.
    assert np.allclose(a[m], b[m], rtol=1e-5, atol=1e-2 * np.abs(a[m]).max())


# ------------------------------------------------------------ landmark transforms
#
# TPStransform and MovingLeastSquaresTransform are the other pair of transforms
# with a fastcore implementation. Their fallback is not a binary but a Python
# library (morphops / molesq), so both are always installed and the parity tests
# below need no external tooling - they run wherever fastcore does.

HAS_FASTCORE_LANDMARKS = backends.fastcore_landmarks_available()

needs_fastcore_landmarks = pytest.mark.skipif(
    not HAS_FASTCORE_LANDMARKS,
    reason="navis-fastcore has no TpsTransform/MlsTransform",
)


@pytest.fixture
def landmarks():
    """A source/target landmark pair plus points to transform."""
    rng = np.random.default_rng(0)
    source = rng.normal(size=(60, 3)) * 100
    target = source + rng.normal(size=(60, 3)) * 5
    points = rng.normal(size=(500, 3)) * 100
    return source, target, points


def _landmark_transforms(source, target, backend):
    """One of each landmark transform, on the given backend."""
    return [
        TPStransform(source, target, backend=backend),
        MovingLeastSquaresTransform(source, target, backend=backend),
        MovingLeastSquaresTransform(
            source, target, direction="inverse", backend=backend
        ),
    ]


@needs_fastcore_landmarks
def test_landmark_backend_selection(landmarks):
    """"auto" picks fastcore; the fallback is spelled "python", not "binary"."""
    source, target, _ = landmarks
    for tr in _landmark_transforms(source, target, None):
        assert tr.backend == "fastcore"
    for tr in _landmark_transforms(source, target, "python"):
        assert tr.backend == "python"
    # "binary" is the CMTK/elastix spelling of the same request and must be
    # accepted here too, so one global config value can steer both families.
    for tr in _landmark_transforms(source, target, "binary"):
        assert tr.backend == "python"


@needs_fastcore_landmarks
def test_landmark_backend_follows_config(landmarks):
    """The backend is resolved lazily, so config changes are picked up late."""
    source, target, _ = landmarks
    trs = _landmark_transforms(source, target, None)

    navis.config.default_transform_backend = "python"
    assert [t.backend for t in trs] == ["python"] * len(trs)

    navis.config.default_transform_backend = "fastcore"
    assert [t.backend for t in trs] == ["fastcore"] * len(trs)


def test_landmark_transforms_without_fastcore(landmarks, no_fastcore):
    """Without fastcore we fall back to morphops/molesq rather than raising."""
    source, target, points = landmarks
    for tr in _landmark_transforms(source, target, None):
        assert tr.backend == "python"
        assert tr.xform(points).shape == points.shape

    with pytest.raises(ValueError, match="not installed or too old"):
        TPStransform(source, target, backend="fastcore").backend


@needs_fastcore_landmarks
@pytest.mark.parametrize("kind", ["tps", "mls", "mls_inverse"])
def test_landmark_parity(landmarks, kind):
    """The two backends must agree numerically."""
    source, target, points = landmarks
    kwargs = {"direction": "inverse"} if kind == "mls_inverse" else {}
    cls = TPStransform if kind == "tps" else MovingLeastSquaresTransform

    fast = cls(source, target, backend="fastcore", **kwargs)
    slow = cls(source, target, backend="python", **kwargs)

    assert np.allclose(fast.xform(points), slow.xform(points), atol=1e-8)
    assert np.allclose(fast.matrix_affine, slow.matrix_affine, atol=1e-8)
    # Negation refits/flips in the opposite direction on both backends
    assert np.allclose((-fast).xform(points), (-slow).xform(points), atol=1e-8)


@needs_fastcore_landmarks
def test_tps_coefficients_parity(landmarks):
    """`W`/`A` come from whichever backend is active and must agree.

    The fastcore backend deliberately fits the spline with morphops too (only
    the *transform* runs in fastcore - its blocked-LU fit is slower than numpy's
    LAPACK solve), so the coefficients are not merely close but *identical*.
    """
    source, target, _ = landmarks
    fast = TPStransform(source, target, backend="fastcore")
    slow = TPStransform(source, target, backend="python")

    assert np.array_equal(fast.W, slow.W)
    assert np.array_equal(fast.A, slow.A)
    # A is (4, 3) with the translation in row 0 on both backends - the
    # `matrix_affine` property depends on that layout.
    assert fast.A.shape == slow.A.shape == (4, 3)


@needs_fastcore_landmarks
def test_tps_fastcore_fits_with_morphops_not_fastcore(landmarks, monkeypatch):
    """The fastcore backend must not call fastcore's own (slower) TPS fit.

    It should build its transform object via `TpsTransform.from_coefs` from the
    morphops coefficients. Guarding this keeps the hybrid from silently
    regressing to fastcore's blocked-LU fit if the wiring is refactored.
    """
    from navis import utils

    calls = {"fit": 0, "from_coefs": 0}
    real_cls = utils.fastcore.TpsTransform

    class SpyTps:
        def __init__(self, *a, **k):
            calls["fit"] += 1
            self._tr = real_cls(*a, **k)

        @classmethod
        def from_coefs(cls, *a, **k):
            calls["from_coefs"] += 1
            obj = cls.__new__(cls)
            obj._tr = real_cls.from_coefs(*a, **k)
            return obj

        def xform(self, *a, **k):
            return self._tr.xform(*a, **k)

    monkeypatch.setattr(utils.fastcore, "TpsTransform", SpyTps)

    source, target, points = landmarks
    TPStransform(source, target, backend="fastcore").xform(points)

    assert calls["from_coefs"] == 1, "expected a single from_coefs build"
    assert calls["fit"] == 0, "fastcore's own fit must not be used"


@needs_fastcore_landmarks
def test_landmark_transforms_accept_dataframes(landmarks):
    """A DataFrame with x/y/z columns works on either backend."""
    import pandas as pd

    source, target, points = landmarks
    df = pd.DataFrame(points, columns=["x", "y", "z"])
    for backend in ("fastcore", "python"):
        for tr in _landmark_transforms(source, target, backend):
            assert np.allclose(tr.xform(df), tr.xform(points), atol=1e-8)

    tr = TPStransform(source, target, backend="fastcore")
    with pytest.raises(ValueError, match="x/y/z"):
        tr.xform(pd.DataFrame(points, columns=["a", "b", "c"]))


@needs_fastcore_landmarks
def test_landmark_copy_and_pickle_preserve_backend(landmarks):
    """Copies keep the backend request, and both survive a pickle round-trip."""
    source, target, points = landmarks
    for tr in _landmark_transforms(source, target, "fastcore"):
        cp = tr.copy()
        assert cp.backend == "fastcore"
        assert np.allclose(cp.xform(points), tr.xform(points))

        rt = pickle.loads(pickle.dumps(tr))
        assert rt.backend == "fastcore"
        assert np.allclose(rt.xform(points), tr.xform(points))


@needs_fastcore_landmarks
def test_mls_direction_is_not_cached(landmarks):
    """Direction is per-call, so flipping it after a transform must be honoured.

    navis keeps the direction in a mutable `.reverse` that `__neg__` (and users)
    flip in place, while the cached fastcore object is built once - so this
    would silently return stale results if `xform` did not pass `reverse`
    through on every call.
    """
    source, target, points = landmarks
    tr = MovingLeastSquaresTransform(source, target, backend="fastcore")

    forward = tr.xform(points)          # populates the cached fastcore object
    inverse = (-tr).xform(points)
    assert not np.allclose(forward, inverse)

    # ... and flipping `.reverse` by hand must be picked up just the same
    tr.reverse = True
    assert np.allclose(tr.xform(points), inverse)
    tr.reverse = False
    assert np.allclose(tr.xform(points), forward)
