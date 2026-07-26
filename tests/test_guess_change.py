"""Tests for `navis.transforms.xfm_funcs._guess_change`.

`_guess_change` estimates the order-of-magnitude change in spatial units during
a transform (e.g. nm -> um) by comparing pairwise distances before/after. These
tests use synthetic data and don't require any template-brain registrations.
"""

import numpy as np
import pandas as pd
import pytest

from navis.transforms.xfm_funcs import _guess_change


def _random_points(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-100, 100, size=(n, 3))


def _rotation_matrix():
    """A fixed, arbitrary 3D rotation matrix (orthonormal)."""
    # Rotate 30 deg about x, 45 about y, 60 about z
    ax, ay, az = np.radians([30, 45, 60])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


@pytest.mark.parametrize("scale,expected_mag", [
    (1000.0, 3),
    (100.0, 2),
    (10.0, 1),
    (1.0, 0),
    (0.1, -1),
    (0.001, -3),
])
def test_pure_scaling(scale, expected_mag):
    """A pure uniform scaling must recover the exact scale and magnitude."""
    before = _random_points()
    after = before * scale

    change, magnitude = _guess_change(before, after, sample=1.0)

    assert magnitude == expected_mag
    assert change == pytest.approx(scale, rel=1e-6)


def test_invariant_to_rotation_and_translation():
    """Rotation + translation must not affect the estimate (only scale does)."""
    before = _random_points()
    scale = 10.0
    after = (before @ _rotation_matrix().T) * scale + np.array([500, -300, 42])

    change, magnitude = _guess_change(before, after, sample=1.0)

    assert magnitude == 1
    assert change == pytest.approx(scale, rel=1e-6)


def test_robust_to_outlier_points():
    """A minority of points that fly off (bad/warp transform) must not sway the
    median-based estimate, whereas the old mean-of-ratios would be skewed.

    Note a corrupted *point* taints every *pair* it participates in, so a
    fraction f of bad points corrupts ~(1 - (1 - f)**2) of pairs. We keep
    f = 0.2 (~36% of pairs) so the median stays safely on a clean pair.
    """
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(1)
    before = _random_points(n=400, seed=2)
    scale = 100.0
    after = before * scale

    # Corrupt 20% of the after-points by throwing them far away
    n_bad = int(0.2 * after.shape[0])
    bad_ix = rng.choice(after.shape[0], n_bad, replace=False)
    after[bad_ix] += rng.uniform(-1e5, 1e5, size=(n_bad, 3))

    change, magnitude = _guess_change(before, after, sample=1.0)

    # Median still recovers the true scale (clean pairs give exactly `scale`)
    assert magnitude == 2
    assert change == pytest.approx(scale, rel=1e-6)

    # Contrast: the old mean-of-ratios estimator is badly skewed by the outliers
    old_mean = np.nanmean(pdist(after) / pdist(before))
    assert old_mean > 2 * scale  # demonstrably wrong -> justifies the median


def test_robust_to_near_coincident_source_points():
    """Near-coincident source points (tiny dist_pre) used to blow up the ratio.
    They must be filtered / down-weighted so the estimate stays correct."""
    before = _random_points(n=300, seed=3)
    scale = 50.0

    # Duplicate a chunk of points so many source pairs have ~zero distance
    before = np.vstack([before, before[:100] + 1e-9])
    after = before * scale

    change, magnitude = _guess_change(before, after, sample=1.0)

    assert magnitude == 2  # round(log10(50)) == 2
    assert change == pytest.approx(scale, rel=1e-3)


def test_nan_points_are_ignored():
    """Points that failed to transform (NaN) must be dropped, not crash."""
    before = _random_points(seed=4)
    scale = 1000.0
    after = before * scale
    after[::10] = np.nan  # 10% failed to transform

    change, magnitude = _guess_change(before, after, sample=1.0)

    assert magnitude == 3
    assert change == pytest.approx(scale, rel=1e-6)


def test_all_points_collapse_returns_no_change():
    """If every point collapses onto one spot post-transform, don't raise on
    log10(0) -- fall back to 'no change'."""
    before = _random_points(seed=5)
    after = np.zeros_like(before)  # everything on top of each other

    change, magnitude = _guess_change(before, after, sample=1.0)

    assert change == 1.0
    assert magnitude == 0


def test_all_source_points_identical_returns_no_change():
    """If all source points are identical (dist_pre == 0 everywhere) there's
    nothing usable -- fall back to 'no change' rather than nan/crash."""
    before = np.ones((200, 3))
    after = _random_points(n=200, seed=6)

    change, magnitude = _guess_change(before, after, sample=1.0)

    assert change == 1.0
    assert magnitude == 0


def test_dataframe_input():
    """x/y/z DataFrames should be accepted just like arrays."""
    before = _random_points(seed=7)
    after = before * 100.0
    cols = ['x', 'y', 'z']
    df_before = pd.DataFrame(before, columns=cols)
    df_after = pd.DataFrame(after, columns=cols)

    change, magnitude = _guess_change(df_before, df_after, sample=1.0)

    assert magnitude == 2
    assert change == pytest.approx(100.0, rel=1e-6)


@pytest.mark.parametrize("sample", [0.1, 50, 1000000])
def test_sample_param_variants(sample):
    """Fractional, absolute, and oversized sample values must all work and
    (with enough points) recover the right magnitude."""
    before = _random_points(n=2000, seed=8)
    after = before * 1000.0

    change, magnitude = _guess_change(before, after, sample=sample)

    assert magnitude == 3
    assert change == pytest.approx(1000.0, rel=1e-6)


def test_subsampling_matches_full_on_pure_scale():
    """For a pure scale, any subsample must give the identical exact answer."""
    before = _random_points(n=1000, seed=9)
    after = before * 100.0

    # sub-sample only 30 points
    change, magnitude = _guess_change(before, after, sample=30)

    assert magnitude == 2
    assert change == pytest.approx(100.0, rel=1e-9)
