"""Tests for `navis.align`: `align_rigid`'s `mirror_axis`, and that the transforms
each method hands back are the ones that actually moved the neurons."""
import navis
import numpy as np

import pytest

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _mirrored_copy(n, axis, angle=0.4, shift=(1e4, -2e3, 500)):
    """A copy of `n` mirrored on `axis`, then rotated and shifted.

    Only a transform containing a reflection can bring this back onto `n`, which is
    what makes it a test of `mirror_axis` rather than of the registration itself.
    """
    m = n.copy()
    co = navis.transforms.align._extract_coords(n).copy()
    co[:, axis] *= -1
    rot = Rotation.from_euler("z", angle).as_matrix()
    navis.transforms.align._set_coords(m, co @ rot.T + np.array(shift))
    if isinstance(m, navis.Mesh):
        m.faces = m.faces[:, ::-1]      # a proper mirror flips the winding too
    return m


@pytest.mark.parametrize("axis,spelled", [(0, "x"), (1, "y"), (2, 2)])
def test_mirror_axis_recovers_a_reflected_neuron(axis, spelled):
    n = navis.example_neurons(1, kind="skeleton")
    m = _mirrored_copy(n, axis)

    xf, regs = navis.align.align_rigid(m, target=n, mirror_axis=spelled,
                                       sample=0.2, progress=False)

    # The mirrored fit must have won, and the reflection must be folded into the
    # transform it returns - so that transform takes the *original* coordinates.
    assert np.linalg.det(regs[0].rotation) < 0
    co = navis.transforms.align._extract_coords(m)
    moved = navis.transforms.align._extract_coords(xf[0])
    assert np.allclose(regs[0].apply(co), moved, atol=1e-4)
    assert np.allclose((~regs[0]).apply(regs[0].apply(co)), co, atol=1e-3)

    # And it really did land on the target
    target_co = navis.transforms.align._extract_coords(n)
    extent = np.ptp(target_co, axis=0).max()
    assert np.abs(moved - target_co).max() < 1e-5 * extent

    # Without it, no rigid transform can get there
    _, plain = navis.align.align_rigid(m, target=n, sample=0.2, progress=False)
    assert plain[0].rms > 100 * regs[0].rms


def test_mirror_axis_keeps_the_better_fit():
    """Neurons that need no mirroring must come back exactly as without the flag."""
    n1, n2 = navis.example_neurons(2, kind="skeleton")

    xf0, regs0 = navis.align.align_rigid(n1, target=n2, sample=0.2, progress=False)
    xf1, regs1 = navis.align.align_rigid(n1, target=n2, mirror_axis="x",
                                         sample=0.2, progress=False)

    assert np.linalg.det(regs1[0].rotation) > 0
    assert regs1[0].rms == regs0[0].rms
    assert np.array_equal(xf1[0].nodes[["x", "y", "z"]].values,
                          xf0[0].nodes[["x", "y", "z"]].values)


def test_mirror_axis_over_a_neuronlist():
    """One mirrored neuron, one not, and the target itself - each judged separately."""
    n1, n2 = navis.example_neurons(2, kind="skeleton")
    nl = navis.NeuronList([_mirrored_copy(n1, 0), n2, n1])

    xf, regs = navis.align.align_rigid(nl, target=n1, mirror_axis="x",
                                       sample=0.2, progress=False)

    dets = [np.linalg.det(r.rotation) for r in regs]
    assert dets[0] < 0 and dets[1] > 0 and dets[2] > 0
    # A neuron that *is* the target is still left untouched
    assert regs[2].iterations == 0
    assert np.array_equal(xf[2].nodes[["x", "y", "z"]].values,
                          n1.nodes[["x", "y", "z"]].values)


def test_mirror_axis_flips_mesh_winding():
    """A reflection turns a mesh inside out; the winding must follow it back."""
    n = navis.example_neurons(1, kind="mesh")
    m = _mirrored_copy(n, 2)
    assert m.trimesh.volume > 0                  # the mirrored copy is well-formed

    xf, regs = navis.align.align_rigid(m, target=n, mirror_axis="z",
                                       sample=0.1, progress=False)

    assert np.linalg.det(regs[0].rotation) < 0
    assert xf[0].trimesh.volume > 0              # ... and so is the aligned one
    assert np.array_equal(xf[0].faces, m.faces[:, ::-1])


@pytest.mark.parametrize("bad", ["w", "xy", 3, -1, 1.5, True, np.nan])
def test_mirror_axis_rejects_nonsense(bad):
    n1, n2 = navis.example_neurons(2, kind="skeleton")
    with pytest.raises(ValueError, match="mirror_axis"):
        navis.align.align_rigid(n1, target=n2, mirror_axis=bad, sample=0.2,
                                progress=False)


def _sym_nn(a, b):
    """Symmetric mean nearest-neighbour distance, as a fraction of b's extent."""
    ca = navis.transforms.align._extract_coords(a)
    cb = navis.transforms.align._extract_coords(b)
    d = (cKDTree(cb).query(ca)[0].mean() + cKDTree(ca).query(cb)[0].mean()) / 2
    return d / np.ptp(cb, axis=0).max()


@pytest.mark.parametrize("func", [navis.align.align_rigid, navis.align.align_deform])
def test_fit_on_a_subsample_moves_every_point(func):
    """`sample=` fits on a fraction of the points; the transform still moves them all.

    Both registrations are functions of position, so the points that took no part in
    the fit go through the same transform as the rest - there is no landmark step.
    """
    n1, n2 = navis.example_neurons(2, kind="skeleton")

    xf, regs = func(n1, target=n2, sample=0.2, progress=False)

    co = navis.transforms.align._extract_coords(n1)
    moved = navis.transforms.align._extract_coords(xf[0])
    assert len(moved) == len(co)                      # not just the subsample
    assert np.allclose(regs[0].apply(co), moved, atol=1e-4)


def test_deform_beats_rigid_which_beats_doing_nothing():
    n1, n2 = navis.example_neurons(2, kind="skeleton")

    rigid = navis.align.align_rigid(n1, target=n2, sample=0.2, progress=False)[0][0]
    deform = navis.align.align_deform(n1, target=n2, sample=0.2, progress=False)[0][0]

    assert _sym_nn(deform, n2) < _sym_nn(rigid, n2) < _sym_nn(n1, n2)


def test_rigid_deform_returns_the_chain_that_moved_the_neuron():
    """The two transforms come back in the order they have to be applied."""
    n1, n2 = navis.example_neurons(2, kind="skeleton")

    align = navis.transforms.align._align_func("rigid+deform")
    xf, regs = align(n1, target=n2, sample=0.2, progress=False)

    rigid, deform = regs[0]
    co = navis.transforms.align._extract_coords(n1)
    moved = navis.transforms.align._extract_coords(xf[0])
    assert np.allclose(deform.apply(rigid.apply(co)), moved, atol=1e-4)
    # The warp was fitted on the points the rigid fit moved, not on the originals
    assert np.allclose(deform.source, rigid.apply(co[::5]), atol=1e-4)


@pytest.mark.parametrize("method", ["rigid", "deform", "rigid+deform", "pca"])
def test_pairwise_grid(method):
    nl = navis.example_neurons(3, kind="skeleton")

    aligned = navis.align.align_pairwise(nl, method=method, sample=0.1,
                                         progress=False)

    assert aligned.shape == (3, 3)
    if method != "pca":
        # A neuron aligned to itself is the identity, so it is handed straight back
        assert all(aligned[i, i] is nl[i] for i in range(3))
        assert _sym_nn(aligned[0, 1], nl[1]) < _sym_nn(nl[0], nl[1])


def test_pairwise_passes_options_to_the_right_fit():
    nl = navis.example_neurons(2, kind="skeleton")

    # `scale` is a rigid option, `beta` a deformable one
    navis.align.align_pairwise(nl, method="rigid+deform", scale=True, beta=0.5,
                               sample=0.1, progress=False)

    with pytest.raises(TypeError, match="beta"):
        navis.align.align_pairwise(nl, method="rigid", beta=0.5, sample=0.1,
                                   progress=False)


def test_scale_is_honoured():
    """`scale=False` must hold the scale at exactly 1 - `pycpd` used to fit one anyway."""
    n1, n2 = navis.example_neurons(2, kind="skeleton")

    _, off = navis.align.align_rigid(n1, target=n2, scale=False, sample=0.2,
                                     progress=False)
    _, on = navis.align.align_rigid(n1, target=n2, scale=True, sample=0.2,
                                    progress=False)

    assert off[0].scale == 1.0                   # and it is the default
    assert on[0].scale != 1.0
    # ... and `scale_bounds` keeps it inside the range it is given
    _, capped = navis.align.align_rigid(n1, target=n2, scale=True,
                                        scale_bounds=(1.0, 1.0001), sample=0.2,
                                        progress=False)
    assert 1.0 <= capped[0].scale <= 1.0001
