import navis
import numpy as np

import pytest

from navis.data.load_data import SOMA_POS


# Ground-truth soma positions are in 8 nm voxel units; a soma here is ~250 vox
# in radius, so 300 is a meaningful tolerance (roughly one soma radius).
SOMA_ATOL = 300


@pytest.fixture
def meshes():
    return navis.example_neurons(5, kind="mesh")


def test_find_soma_mesh_detection(meshes):
    """Detect the 4 example somas and reject the one without."""
    for n in meshes:
        gt = SOMA_POS[n.id]
        ell = navis.find_soma_mesh(n)
        if gt is None:
            # 722817260 has no soma and must be rejected
            assert ell is None
        else:
            assert ell is not None
            assert np.allclose(ell.center, gt, atol=SOMA_ATOL)


def test_find_soma_mesh_inplace(meshes):
    """`inplace=True` sets `.soma_pos` (or leaves it None) and returns the neuron."""
    for n in meshes:
        gt = SOMA_POS[n.id]
        ret = navis.find_soma_mesh(n, inplace=True)
        assert ret is n
        if gt is None:
            assert n.soma_pos is None
        else:
            assert n.soma_pos is not None
            assert n.soma_pos.shape == (3,)
            assert np.allclose(n.soma_pos, gt, atol=SOMA_ATOL)


def test_find_soma_mesh_deterministic(meshes):
    """Repeated calls must return byte-identical results (no hidden RNG)."""
    for n in meshes:
        a = navis.find_soma_mesh(n)
        b = navis.find_soma_mesh(n)
        if a is None:
            assert b is None
        else:
            assert np.array_equal(a.center, b.center)
            assert np.array_equal(a.radii, b.radii)
            assert np.array_equal(a.axes, b.axes)


def test_find_soma_mesh_neuronlist(meshes):
    """Over a NeuronList we get one result per neuron."""
    res = navis.find_soma_mesh(meshes)
    assert len(res) == len(meshes)
    for n, r in zip(meshes, res):
        if SOMA_POS[n.id] is None:
            assert r is None
        else:
            assert isinstance(r, navis.SomaEllipsoid)


def test_soma_ellipsoid_properties(meshes):
    """The fitted ellipsoid is well-formed and its geometry helpers work."""
    n = next(n for n in meshes if SOMA_POS[n.id] is not None)
    ell = navis.find_soma_mesh(n)

    # semi-axes sorted descending and positive
    assert np.all(np.diff(ell.radii) <= 0)
    assert np.all(ell.radii > 0)
    # principal axes are orthonormal
    assert np.allclose(ell.axes.T @ ell.axes, np.eye(3), atol=1e-6)
    assert ell.equiv_radius > 0 and ell.volume > 0

    # the center is inside the ellipsoid; a far-away point is outside
    assert ell.contains(ell.center)[0]
    assert ell.distance_to_surface(ell.center)[0] < 0
    far = ell.center + np.array([1e5, 0, 0])
    assert not ell.contains(far)[0]
    assert ell.distance_to_surface(far)[0] > 0


def test_find_soma_mesh_rejects_treeneuron():
    """A TreeNeuron should be rejected - use `find_soma` for skeletons."""
    sk = navis.example_neurons(1, kind="skeleton")
    with pytest.raises(TypeError):
        navis.find_soma_mesh(sk)


# --------------------------------------------------------------------------- #
#  Skeleton find_soma                                                          #
# --------------------------------------------------------------------------- #

# Ground-truth soma node for each example skeleton (None = no soma).
EXPECTED_SOMA_NODE = {
    1734350788: 4177,
    1734350908: 6,
    722817260: None,
    754534424: 4,
    754538881: 701,
}


@pytest.fixture
def skeletons():
    return navis.example_neurons(5, kind="skeleton")


def test_find_soma_single_node(skeletons):
    """find_soma returns a single node ID (scalar) or None - never an array."""
    for n in skeletons:
        s = navis.find_soma(n)
        # Must be a scalar/None so `.soma` is single-valued (no ambiguous-truth)
        assert not navis.utils.is_iterable(s)
        if EXPECTED_SOMA_NODE[n.id] is None:
            assert s is None
        else:
            assert int(s) == EXPECTED_SOMA_NODE[n.id]


def test_find_soma_position(skeletons):
    """The detected soma is at the ground-truth position (or None)."""
    for n in skeletons:
        gt = SOMA_POS[n.id]
        if gt is None:
            assert n.soma is None
            assert n.soma_pos is None
        else:
            assert n.soma_pos.shape == (1, 3)
            assert np.allclose(n.soma_pos[0], gt, atol=SOMA_ATOL)


def test_find_soma_radius_only_coherence(skeletons):
    """With labels off, radius-only detection still collapses to the one soma."""
    for n in skeletons:
        n = n.copy()
        n.soma_detection_label = None
        s = navis.find_soma(n)
        assert not navis.utils.is_iterable(s)
        if EXPECTED_SOMA_NODE[n.id] is not None:
            assert int(s) == EXPECTED_SOMA_NODE[n.id]


def test_find_soma_label_only(skeletons):
    """With radius off, the labelled soma is still found."""
    n = next(n for n in skeletons if n.id == 1734350788).copy()
    n.soma_detection_radius = None
    assert int(navis.find_soma(n)) == 4177


def test_find_soma_dist_factor_stable(skeletons):
    """dist_factor in {2, 3, 4} returns the same node on the examples."""
    for n in skeletons:
        if EXPECTED_SOMA_NODE[n.id] is None:
            continue
        results = {int(navis.find_soma(n, dist_factor=f)) for f in (2, 3, 4)}
        assert len(results) == 1


def test_find_soma_missing_radius(skeletons):
    """NaN and <= 0 are treated as missing; a labelled soma survives either."""
    base = next(n for n in skeletons if n.id == 1734350788)

    # All radii missing (NaN or -1) but labels present -> found via label rescue
    for sentinel in (np.nan, -1.0):
        n = base.copy()
        nodes = n.nodes.copy()
        nodes["radius"] = sentinel
        n.nodes = nodes
        assert int(navis.find_soma(n)) == 4177

    # Radius requested, all missing, and no label to fall back on -> None
    n = base.copy()
    n.soma_detection_label = None
    nodes = n.nodes.copy()
    nodes["radius"] = np.nan
    n.nodes = nodes
    assert navis.find_soma(n) is None
