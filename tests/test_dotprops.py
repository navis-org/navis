"""Tests for dotprops construction (tangent vectors + alpha).

`make_dotprops` and `Dotprops.recalculate_tangents` compute these via
`navis_fastcore.dotprops` when it is available and fall back to
`scipy.spatial.cKDTree` + numpy SVDs otherwise. Both paths are exercised here.

Note the two backends are *not* expected to agree bit-for-bit: they disagree
exactly where the k-NN search hits a tied distance, which grid-quantised
coordinates produce readily. What must match is the point count, the degenerate
handling, and the tangents everywhere the neighbourhood is unambiguous.
"""

import numpy as np
import pytest

import navis
from navis import utils
from navis.core.core_utils import tangents_and_alpha, _degenerate_mask

HAS_FASTCORE_DOTPROPS = utils.fastcore is not None and hasattr(
    utils.fastcore, "dotprops"
)

needs_fastcore = pytest.mark.skipif(
    not HAS_FASTCORE_DOTPROPS, reason="navis-fastcore has no `dotprops`"
)


@pytest.fixture
def no_fastcore(monkeypatch):
    """Force the scipy/numpy path."""
    monkeypatch.setattr(utils, "fastcore", None)


@pytest.fixture
def duplicated_points():
    """A cloud whose duplicate block sits in the *middle* of the array.

    Position matters: the pre-fix implementation dropped the degenerate rows
    from `x` but then indexed it with neighbour indices offset by a flat
    `max_zero.sum()`, which is only correct when every duplicate sits at the
    front. Anywhere else the indices ran out of bounds, wrapped (numpy allows
    negative indices), and silently produced tangents for the wrong points.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 3)) * 20
    return np.vstack([base[:20], np.tile([100.0, 100, 100], (4, 1)), base[20:]])


# ------------------------------------------------------------------ primitives


@pytest.mark.parametrize("k", [3, 5, 20])
def test_tangents_match_a_hand_rolled_reference(k):
    """Both backends must match a spelled-out cKDTree + SVD reference."""
    from scipy.spatial import cKDTree

    # Random float64 points essentially never tie, so the two agree exactly.
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3)) * 100

    tree = cKDTree(pts)
    _, ix = tree.query(pts, k=k)
    nb = pts[ix.reshape(len(pts), k)]
    cpt = nb - nb.mean(axis=1)[:, None, :]
    _, s, vh = np.linalg.svd(cpt.transpose((0, 2, 1)) @ cpt)
    exp_vect, exp_alpha = vh[:, 0, :], (s[:, 0] - s[:, 1]) / s.sum(axis=1)

    vect, alpha = tangents_and_alpha(pts, k)

    assert np.allclose(alpha, exp_alpha, atol=1e-12)
    # Eigenvector sign is arbitrary - compare direction
    assert np.allclose(np.abs(np.einsum("ij,ij->i", vect, exp_vect)), 1, atol=1e-8)


def test_tangents_alpha_bounds():
    """Alpha is a ratio of eigenvalues and must stay in [0, 1]."""
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(300, 3))
    _, alpha = tangents_and_alpha(pts, 10)
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_tangents_collinear_and_planar():
    """The two extremes of alpha, on both backends."""
    line = np.zeros((10, 3))
    line[:, 0] = np.arange(10)
    vect, alpha = tangents_and_alpha(line, 5)
    assert np.allclose(alpha, 1)
    assert np.allclose(np.abs(vect[:, 0]), 1)

    # A square + its centre is perfectly isotropic in-plane -> alpha 0
    plane = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]])
    _, alpha = tangents_and_alpha(plane, 5)
    assert np.allclose(alpha[0], 0, atol=1e-12)


def test_tangents_are_unit_length():
    rng = np.random.default_rng(2)
    vect, _ = tangents_and_alpha(rng.normal(size=(200, 3)), 8)
    assert np.allclose(np.linalg.norm(vect, axis=1), 1)


def test_degenerate_neighbourhood_gives_zero_not_nan():
    """A fully coincident neighbourhood has no direction - alpha 0, never NaN."""
    pts = np.vstack([np.zeros((6, 3)), np.arange(60.0).reshape(20, 3)])
    vect, alpha = tangents_and_alpha(pts, 5)

    assert not np.isnan(alpha).any()
    assert np.allclose(alpha[:6], 0)
    assert np.isfinite(vect).all()
    assert np.allclose(np.linalg.norm(vect, axis=1), 1)


def test_degenerate_mask_needs_k_duplicates():
    """Fewer than `k` duplicates is not degenerate - the tangent is defined."""
    # 3 coincident points, k=5 -> the other 2 neighbours are elsewhere
    pts = np.vstack([np.zeros((3, 3)), np.arange(30.0).reshape(10, 3)])
    vect, alpha = tangents_and_alpha(pts, 5)
    assert not _degenerate_mask(pts, 5, alpha).any()

    # 5 coincident, k=5 -> degenerate
    pts = np.vstack([np.zeros((5, 3)), np.arange(30.0).reshape(10, 3)])
    vect, alpha = tangents_and_alpha(pts, 5)
    assert _degenerate_mask(pts, 5, alpha)[:5].all()


def test_degenerate_mask_ignores_symmetric_neighbourhoods():
    """`alpha == 0` alone is not degeneracy - a symmetric cloud also gives 0."""
    plane = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]])
    _, alpha = tangents_and_alpha(plane, 5)
    assert np.isclose(alpha[0], 0, atol=1e-12)
    # ... but there are no duplicates, so nothing may be dropped
    assert not _degenerate_mask(plane, 5, alpha).any()


# -------------------------------------------------------------- make_dotprops


def test_make_dotprops_drops_degenerate_points(duplicated_points):
    dp = navis.make_dotprops(duplicated_points, k=4, on_issue="fix")
    assert len(dp.points) == len(duplicated_points) - 4
    assert not np.isnan(dp.alpha).any()


def test_make_dotprops_raises_on_duplicates(duplicated_points):
    with pytest.raises(ValueError, match="zero distance"):
        navis.make_dotprops(duplicated_points, k=4, on_issue="raise")


def test_make_dotprops_duplicate_block_position_is_irrelevant():
    """Regression: duplicates away from the array start corrupted every tangent.

    The old code offset the neighbour indices by a flat `max_zero.sum()` after
    dropping rows, which only works if the duplicates are at the front. With the
    block in the middle the indices went negative, wrapped, and produced
    tangents computed from unrelated points - silently, since numpy permits
    negative indices.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 3)) * 20
    dupe = np.tile([100.0, 100, 100], (4, 1))

    front = navis.make_dotprops(np.vstack([dupe, base]), k=4, on_issue="fix")
    middle = navis.make_dotprops(
        np.vstack([base[:20], dupe, base[20:]]), k=4, on_issue="fix"
    )

    assert len(front.points) == len(middle.points) == len(base)

    # Same points (in a different order) must give the same tangents. Sort both
    # by coordinate to compare.
    def keyed(dp):
        order = np.lexsort(np.asarray(dp.points).T)
        return (np.asarray(dp.points)[order], np.asarray(dp.vect)[order],
                np.asarray(dp.alpha)[order])

    p1, v1, a1 = keyed(front)
    p2, v2, a2 = keyed(middle)
    assert np.allclose(p1, p2)
    assert np.allclose(a1, a2, atol=1e-6)
    assert np.allclose(np.abs(np.einsum("ij,ij->i", v1, v2)), 1, atol=1e-6)


def test_make_dotprops_k_clamped_to_point_count():
    pts = np.arange(15.0).reshape(5, 3)
    dp = navis.make_dotprops(pts, k=20, on_issue="fix")
    assert dp.k == 5
    assert len(dp.points) == 5


# ------------------------------------------------------- recalculate_tangents


def test_recalculate_tangents_keeps_every_point():
    """Unlike `make_dotprops` this may not change `.points`."""
    n = navis.example_neurons(1, kind="skeleton")
    dp = navis.make_dotprops(n, k=5)
    n_before = len(dp.points)

    out = dp.recalculate_tangents(10, inplace=False)

    assert len(out.points) == n_before
    assert out.k == 10
    assert len(out.vect) == n_before and len(out.alpha) == n_before
    assert not np.isnan(out.alpha).any()


def test_recalculate_tangents_matches_make_dotprops():
    """Recomputing with `k` must reproduce building with `k` from the start."""
    n = navis.example_neurons(1, kind="skeleton")
    a = navis.make_dotprops(n, k=5).recalculate_tangents(12, inplace=False)
    b = navis.make_dotprops(n, k=12)

    assert np.allclose(a.alpha, b.alpha, atol=1e-6)
    dots = np.abs(np.einsum("ij,ij->i", np.asarray(a.vect, float),
                            np.asarray(b.vect, float)))
    assert np.allclose(dots, 1, atol=1e-5)


def test_recalculate_tangents_no_nan_on_duplicates():
    """Degenerate points get alpha 0 here (they cannot be dropped)."""
    pts = np.vstack([np.zeros((5, 3)), np.arange(30.0).reshape(10, 3)])
    dp = navis.make_dotprops(pts[5:], k=3)
    dp.points = pts.astype(dp.points.dtype)

    out = dp.recalculate_tangents(3, inplace=False)

    assert len(out.alpha) == len(pts)
    assert not np.isnan(out.alpha).any()
    assert np.allclose(out.alpha[:5], 0)


# ------------------------------------------------------------ backend parity


@needs_fastcore
@pytest.mark.parametrize("k", [5, 20])
def test_backend_parity_on_a_real_neuron(k, monkeypatch):
    """Both backends must agree on everything but tie-broken neighbourhoods."""
    n = navis.example_neurons(1, kind="skeleton")

    fast = navis.make_dotprops(n, k=k)
    monkeypatch.setattr(utils, "fastcore", None)
    slow = navis.make_dotprops(n, k=k)

    assert len(fast.points) == len(slow.points)
    assert np.array_equal(np.asarray(fast.points), np.asarray(slow.points))

    dots = np.abs(np.einsum("ij,ij->i", np.asarray(fast.vect, float),
                            np.asarray(slow.vect, float)))
    # Ties are the only legitimate source of disagreement and are rare.
    differing = (dots < 1 - 1e-5).mean()
    assert differing < 0.05, f"{differing:.1%} of tangents differ - too many"


@needs_fastcore
def test_backend_parity_without_ties(monkeypatch):
    """With no tied distances the two backends must agree to floating point."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(2000, 3)) * 100

    v_fast, a_fast = tangents_and_alpha(pts, 10)
    monkeypatch.setattr(utils, "fastcore", None)
    v_slow, a_slow = tangents_and_alpha(pts, 10)

    assert np.allclose(a_fast, a_slow, atol=1e-12)
    assert np.allclose(np.abs(np.einsum("ij,ij->i", v_fast, v_slow)), 1, atol=1e-10)


@needs_fastcore
def test_backend_parity_on_degenerates(monkeypatch):
    """The alpha=0 / unit-vector convention must match across backends."""
    pts = np.vstack([np.zeros((6, 3)), np.arange(60.0).reshape(20, 3)])

    v_fast, a_fast = tangents_and_alpha(pts, 5)
    monkeypatch.setattr(utils, "fastcore", None)
    v_slow, a_slow = tangents_and_alpha(pts, 5)

    assert np.allclose(a_fast[:6], 0) and np.allclose(a_slow[:6], 0)
    assert not np.isnan(a_fast).any() and not np.isnan(a_slow).any()
    assert np.allclose(np.linalg.norm(v_fast, axis=1), 1)
    assert np.allclose(np.linalg.norm(v_slow, axis=1), 1)


def test_make_dotprops_without_fastcore(no_fastcore):
    """The scipy fallback must still produce usable dotprops."""
    n = navis.example_neurons(1, kind="skeleton")
    dp = navis.make_dotprops(n, k=5)
    assert len(dp.points) > 0
    assert not np.isnan(dp.alpha).any()
    assert np.allclose(np.linalg.norm(np.asarray(dp.vect, float), axis=1), 1)
