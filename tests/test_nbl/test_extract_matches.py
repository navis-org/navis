"""Tests for `navis.nbl.extract_matches`.

The bulk of these run each criterion through *both* backends (navis-fastcore
and the numpy fallback) and assert they agree. The one thing they are allowed
to disagree on is the order of *tied* scores - which of two equally good
matches comes first is arbitrary - so comparisons here are score-based rather
than label-based wherever ties are possible.

`fastcore_off` forces the fallback by blanking `navis.utils.fastcore`, which is
what `_fastcore_matrix` checks. That is also how a user without fastcore
installed gets there, so it exercises the real path.
"""

import numpy as np
import pandas as pd
import pytest

from navis import utils
from navis.nbl import extract_matches

FASTCORE = utils.fastcore
HAS_FASTCORE = FASTCORE is not None and hasattr(FASTCORE, "top_matches")

needs_fastcore = pytest.mark.skipif(
    not HAS_FASTCORE, reason="navis-fastcore does not provide `top_matches`"
)


@pytest.fixture
def fastcore_off():
    """Force the numpy fallback."""
    utils.fastcore = None
    try:
        yield
    finally:
        utils.fastcore = FASTCORE


def frame(a):
    a = np.asarray(a)
    return pd.DataFrame(
        a,
        index=[f"q{i}" for i in range(a.shape[0])],
        columns=[f"t{i}" for i in range(a.shape[1])],
    )


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    return frame(rng.random((30, 20)).astype(np.float32))


@pytest.fixture
def distances():
    rng = np.random.default_rng(1)
    d = rng.random((20, 20)).astype(np.float32)
    np.fill_diagonal(d, 0)
    return frame(d)


def both_backends(func):
    """Run `func` with fastcore on, then off."""
    on = func()
    utils.fastcore = None
    try:
        off = func()
    finally:
        utils.fastcore = FASTCORE
    return on, off


# ---------------------------------------------------------------- N (top_matches)


@pytest.mark.parametrize("N", [1, 3, 20])
@pytest.mark.parametrize("dist", [True, False])
def test_n_shape_and_order(scores, distances, N, dist):
    s = distances if dist else scores
    m = extract_matches(s, N=N, distances=dist)

    assert len(m) == len(s)
    assert list(m.columns) == ["id"] + [
        c for i in range(N) for c in (f"match_{i + 1}", f"score_{i + 1}")
    ]
    assert (m.id.values == s.index.values).all()

    # Scores must be monotonic across ranks: descending for similarities,
    # ascending for distances
    ranked = np.stack([m[f"score_{i + 1}"].values for i in range(N)], axis=1)
    diff = np.diff(ranked, axis=1)
    assert (diff <= 1e-6).all() if not dist else (diff >= -1e-6).all()

    # Every reported score must actually be that query's score for that match
    for i in range(N):
        got = s.values[np.arange(len(s)), [s.columns.get_loc(c) for c in m[f"match_{i + 1}"]]]
        assert np.allclose(got, m[f"score_{i + 1}"].values)


@needs_fastcore
@pytest.mark.parametrize("N", [1, 5])
@pytest.mark.parametrize("dist", [True, False])
def test_n_backends_agree(scores, distances, N, dist):
    s = distances if dist else scores
    on, off = both_backends(lambda: extract_matches(s, N=N, distances=dist))

    assert (on.id.values == off.id.values).all()
    for i in range(N):
        # Scores must be identical; labels only where there is no tie
        a, b = on[f"score_{i + 1}"].values, off[f"score_{i + 1}"].values
        assert np.allclose(a, b, equal_nan=True)
        neq = on[f"match_{i + 1}"].values != off[f"match_{i + 1}"].values
        assert np.allclose(a[neq], b[neq]), "labels differ on a non-tied score"


@needs_fastcore
def test_n_ties_are_valid_picks():
    """With heavy ties the backends may pick differently - both must be right."""
    rng = np.random.default_rng(2)
    s = frame(np.round(rng.random((20, 10)) * 2).astype(np.float32))
    on, off = both_backends(lambda: extract_matches(s, N=3))

    for i in range(3):
        assert np.allclose(on[f"score_{i + 1}"].values, off[f"score_{i + 1}"].values)


def test_n_too_large(scores):
    with pytest.raises(ValueError, match="Cannot extract"):
        extract_matches(scores, N=scores.shape[1] + 1)


@pytest.mark.parametrize("dist", [True, False])
def test_n_axis(scores, dist):
    """axis=1 must be the same as extracting from the transposed frame."""
    by_col = extract_matches(scores, N=3, axis=1, distances=dist)
    by_row = extract_matches(scores.T, N=3, axis=0, distances=dist)
    pd.testing.assert_frame_equal(by_col, by_row)
    assert (by_col.id.values == scores.columns.values).all()


# -------------------------------------------------------------------- NaN handling


@pytest.mark.parametrize("dist", [True, False])
def test_n_skips_nan(dist):
    """A single unscored pair must not become that query's top match."""
    s = frame(np.array([[0.1, 0.9, np.nan], [np.nan, 0.2, 0.3]], dtype=np.float32))
    m = extract_matches(s, N=1, distances=dist)

    assert not np.isnan(m.score_1.values).any()
    expected = ["t0", "t1"] if dist else ["t1", "t2"]
    assert list(m.match_1) == expected


def test_n_too_few_valid_scores():
    """Fewer than N valid scores -> None/NaN, not a wrapped-around label."""
    s = frame(np.array([[0.5, np.nan, np.nan], [0.1, 0.2, 0.3]], dtype=np.float32))
    m = extract_matches(s, N=2, distances=False)

    assert m.match_1.tolist() == ["t0", "t2"]
    # N.B. `pd.isna`, not `is None`: pandas >= 3 infers a string column here and
    # renders the missing label as NaN, pandas 2 keeps object/None
    assert pd.isna(m.match_2.iloc[0])
    assert np.isnan(m.score_2.values[0])
    assert m.match_2.tolist()[1] == "t1"


@needs_fastcore
def test_nan_backends_agree():
    rng = np.random.default_rng(3)
    a = rng.random((10, 10)).astype(np.float32)
    a[2, :8] = np.nan
    a[5, :] = np.nan
    s = frame(a)
    on, off = both_backends(lambda: extract_matches(s, N=3, distances=False))
    for i in range(3):
        assert np.allclose(
            on[f"score_{i + 1}"].values.astype(float),
            off[f"score_{i + 1}"].values.astype(float),
            equal_nan=True,
        )


def test_perc_all_nan_row_is_quiet(recwarn):
    s = frame(np.array([[0.5, 0.4], [np.nan, np.nan]], dtype=np.float32))
    m = extract_matches(s, percentage=0.1, distances=False)
    assert m.matches.tolist()[1] == ""
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# ------------------------------------------------------------ threshold / percentage


@pytest.mark.parametrize("dist", [True, False])
def test_threshold(scores, distances, dist):
    s = distances if dist else scores
    thr = float(np.median(s.values))
    m = extract_matches(s, threshold=thr, distances=dist)

    assert m.index.names == ["query", "match"]
    n_expected = (s.values <= thr).sum() if dist else (s.values >= thr).sum()
    assert len(m) == n_expected
    assert (m.score <= thr).all() if dist else (m.score >= thr).all()


@needs_fastcore
@pytest.mark.parametrize("dist", [True, False])
def test_threshold_backends_agree(scores, distances, dist):
    s = distances if dist else scores
    thr = float(np.median(s.values))
    on, off = both_backends(lambda: extract_matches(s, threshold=thr, distances=dist))
    pd.testing.assert_frame_equal(on, off)


@pytest.mark.parametrize("dist", [True, False])
def test_percentage_best_first(distances, scores, dist):
    """Matches are listed best first - which for distances means *ascending*."""
    # N.B. the distance matrix is shifted off zero: with a zero diagonal the
    # band around each row's best score collapses to "exactly 0", every row
    # comes back with a single match and this test asserts nothing at all
    s = (distances + 1) if dist else scores
    m = extract_matches(s, percentage=0.2, distances=dist)

    assert list(m.columns) == ["matches", "scores"]
    assert max(len(r.split(",")) for r in m.matches) > 1, "test data has no ties to order"
    for row in m.scores:
        if not row:
            continue
        vals = np.array([float(v) for v in row.split(",")])
        assert (np.diff(vals) >= -1e-3).all() if dist else (np.diff(vals) <= 1e-3).all()


def test_percentage_distance_order_regression():
    """Distances used to be listed worst first - the exact inverse of intended."""
    s = frame(np.array([[1.0, 1.05, 1.1, 5.0]], dtype=np.float32))
    m = extract_matches(s, percentage=0.15, distances=True)

    assert m.matches.iloc[0] == "t0,t1,t2"
    assert m.scores.iloc[0] == "1.0,1.05,1.1"


@pytest.mark.parametrize("dist", [True, False])
def test_percentage_membership(scores, distances, dist):
    """Every match must be within `perc` of that query's best score."""
    s = distances if dist else scores
    perc = 0.1
    m = extract_matches(s, percentage=perc, distances=dist)

    for i, row in enumerate(m.matches):
        best = s.values[i].min() if dist else s.values[i].max()
        cutoff = best + abs(best * perc) if dist else best - abs(best * perc)
        for c in row.split(","):
            v = s.values[i, s.columns.get_loc(c)]
            assert v <= cutoff + 1e-6 if dist else v >= cutoff - 1e-6


@needs_fastcore
@pytest.mark.parametrize("dist", [True, False])
def test_percentage_backends_agree(scores, distances, dist):
    s = distances if dist else scores
    on, off = both_backends(lambda: extract_matches(s, percentage=0.1, distances=dist))
    for a, b in zip(on.matches, off.matches):
        assert set(a.split(",")) == set(b.split(","))


# --------------------------------------------------------------------- max_matches


@pytest.mark.parametrize("criterion", [{"threshold": 0.1}, {"percentage": 0.9}])
def test_max_matches_raises(scores, criterion):
    with pytest.raises(ValueError):
        extract_matches(scores, distances=False, max_matches=5, **criterion)


def test_max_matches_allows(scores):
    m = extract_matches(scores, threshold=0.99, distances=False, max_matches=10_000)
    assert len(m) <= 10_000


# ------------------------------------------------------------------------ fallbacks


@pytest.mark.usefixtures("fastcore_off")
def test_numpy_fallback_runs(scores):
    """Everything must still work without fastcore installed."""
    assert len(extract_matches(scores, N=3)) == len(scores)
    assert extract_matches(scores, threshold=0.9).index.names == ["query", "match"]
    assert len(extract_matches(scores, percentage=0.1)) == len(scores)


@needs_fastcore
def test_non_float_dtype_falls_back():
    """fastcore only takes float16/32/64 - an int matrix must not crash."""
    s = frame(np.arange(20).reshape(4, 5))
    m = extract_matches(s, N=2, distances=False)
    assert m.match_1.tolist() == ["t4", "t4", "t4", "t4"]


def test_no_criterion(scores):
    with pytest.raises(ValueError, match="Must provide"):
        extract_matches(scores)


def test_two_criteria(scores):
    with pytest.raises(ValueError, match="either"):
        extract_matches(scores, N=3, threshold=0.5)
