"""Tests for `navis.nblast_knn`.

Unlike the other NBLAST functions this one has no built-in implementation - it
exists only in navis-fastcore - so everything here skips without it.

The parity tests below compare against a full `nblast_allbyall` with a
tolerance rather than asserting exact equality.

They currently *would* pass exactly: `nblast_knn` and `nblast` agree
bit-for-bit since the tie-breaking fix in fastcore's `aann-graph` dependency.
Before that they disagreed by up to ~2e-5, because a query point exactly
equidistant from two target points has no unique nearest neighbour and the two
implementations resolved that ambiguity differently - and since the score
depends on the chosen point's tangent, the pair score differed. Exact ties are
common in real data because coordinates sit on a quantised grid.

The tolerance stays because a small fraction of pathological cases remain
ambiguous by nature, and because navis' own built-in backend still resolves
ties differently from fastcore (~5e-5 on the example neurons). Do not tighten
this to an exact comparison. What must hold exactly is the *identity* of the k
nearest neighbours.
"""

import numpy as np
import pandas as pd
import pytest

import navis
from navis.nbl.backends import get_backend

HAS_KNN = (
    get_backend("fastcore").available()
    and not get_backend("fastcore").unsupported("nblast_knn")
)

needs_knn = pytest.mark.skipif(
    not HAS_KNN, reason="navis-fastcore does not provide `nblast_knn`"
)

pytestmark = needs_knn


@pytest.fixture(scope="module")
def dps():
    """Example neurons as dotprops, in microns (what the FCWB matrix expects)."""
    nl = navis.example_neurons(n=5)
    return navis.make_dotprops(nl * (8 / 1000), k=5, progress=False)


@pytest.fixture(scope="module")
def exact(dps):
    """Ground truth: the full score matrix, symmetrised the way k-NN does it."""
    M = navis.nblast_allbyall(dps, backend="fastcore", progress=False).values
    M = (M + M.T) / 2
    # `nblast_knn` excludes each neuron from its own neighbour list
    np.fill_diagonal(M, -np.inf)
    return M


# ------------------------------------------------------------------- output shapes


def test_long_format(dps):
    out = navis.nblast_knn(dps, k=2, progress=False)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["query", "target", "score", "rank"]
    assert len(out) == len(dps) * 2
    assert set(out["query"]) == set(dps.id)
    assert sorted(out["rank"].unique()) == [1, 2]
    # No neuron is its own neighbour
    assert not (out["query"] == out["target"]).any()


def test_wide_format(dps):
    out = navis.nblast_knn(dps, k=2, format="wide", progress=False)
    assert list(out.columns) == ["id", "match_1", "score_1", "match_2", "score_2"]
    assert len(out) == len(dps)
    assert list(out["id"]) == list(dps.id)


def test_arrays_format(dps):
    idx, scores = navis.nblast_knn(dps, k=2, format="arrays", progress=False)
    assert idx.shape == scores.shape == (len(dps), 2)
    assert idx.dtype == np.int64
    # Indices address `dps` itself, and never the neuron's own row
    assert (idx >= 0).all() and (idx < len(dps)).all()
    assert (idx != np.arange(len(dps))[:, None]).all()


def test_formats_agree(dps):
    """All three formats must describe the same neighbours."""
    idx, scores = navis.nblast_knn(dps, k=3, format="arrays", progress=False)
    long = navis.nblast_knn(dps, k=3, format="long", progress=False)
    wide = navis.nblast_knn(dps, k=3, format="wide", progress=False)

    ids = np.asarray(dps.id)
    assert list(long["target"]) == list(ids[idx].ravel())
    assert np.allclose(long["score"], scores.ravel())
    for i in range(3):
        assert list(wide[f"match_{i + 1}"]) == list(ids[idx[:, i]])
        assert np.allclose(wide[f"score_{i + 1}"], scores[:, i])


# ---------------------------------------------------------------------- correctness


def test_neighbours_match_full_nblast(dps, exact):
    """The k nearest neighbours must be exactly those of the full matrix."""
    k = len(dps) - 1
    idx, scores = navis.nblast_knn(
        dps, k=k, format="arrays", n_candidates=1000, progress=False
    )
    expected = np.argsort(-exact, axis=1)[:, :k]
    assert np.array_equal(idx, expected)
    assert np.allclose(scores, np.take_along_axis(exact, expected, axis=1), atol=1e-4)


def test_scores_are_descending(dps):
    _, scores = navis.nblast_knn(dps, k=4, format="arrays", progress=False)
    assert (np.diff(scores, axis=1) <= 0).all()


def test_mean_symmetry(dps):
    """With scores='mean' a mutual pair must score the same in both directions."""
    long = navis.nblast_knn(
        dps, k=len(dps) - 1, n_candidates=1000, progress=False
    )
    lookup = {(r.query, r.target): r.score for r in long.itertuples()}
    for (q, t), score in lookup.items():
        assert np.isclose(score, lookup[(t, q)], atol=1e-6)


def test_explicit_target_keeps_self_matches(dps):
    """With an explicit `target` nothing is excluded - so self matches at 1.0."""
    idx, scores = navis.nblast_knn(
        dps, target=dps, k=1, format="arrays", n_candidates=1000, progress=False
    )
    assert np.array_equal(idx.ravel(), np.arange(len(dps)))
    assert np.allclose(scores.ravel(), 1)


def test_target_indexes_target_not_query(dps):
    """`idx` addresses `target` when one is given."""
    query, target = dps[:2], dps[2:]
    idx, scores = navis.nblast_knn(
        query, target=target, k=2, format="arrays", progress=False
    )
    assert idx.shape == (2, 2)
    assert (idx < len(target)).all()

    long = navis.nblast_knn(query, target=target, k=2, progress=False)
    assert set(long["query"]) <= set(query.id)
    assert set(long["target"]) <= set(target.id)


@pytest.mark.parametrize("precision,dtype", [(16, np.float16),
                                             (32, np.float32),
                                             (64, np.float64)])
def test_precision(dps, precision, dtype):
    _, scores = navis.nblast_knn(
        dps, k=2, format="arrays", precision=precision, progress=False
    )
    assert scores.dtype == dtype


# -------------------------------------------------------------------------- padding


def test_padding_when_k_exceeds_candidates(dps):
    """Only `len(dps) - 1` neighbours exist; the rest must be padded."""
    n_real = len(dps) - 1
    idx, scores = navis.nblast_knn(dps, k=10, format="arrays", progress=False)
    assert (idx[:, :n_real] >= 0).all()
    assert (idx[:, n_real:] == -1).all()
    assert np.isneginf(scores[:, n_real:]).all()

    # Long format simply omits the padding rather than inventing matches
    long = navis.nblast_knn(dps, k=10, progress=False)
    assert len(long) == len(dps) * n_real
    assert (long["target"] != -1).all()

    # Wide format pads with None/NaN - and must not wrap around to the *last*
    # neuron, which is what a raw `ids[-1]` lookup would silently do.
    wide = navis.nblast_knn(dps, k=10, format="wide", progress=False)
    assert wide["match_10"].isna().all()
    assert wide["score_10"].isna().all()
    assert wide["match_1"].notna().all()


# --------------------------------------------------------------------------- errors


def test_rejects_scores_both(dps):
    with pytest.raises(ValueError, match="scores"):
        navis.nblast_knn(dps, scores="both", progress=False)


def test_rejects_unknown_format(dps):
    with pytest.raises(ValueError, match="format"):
        navis.nblast_knn(dps, format="nope", progress=False)


@pytest.mark.parametrize("k", [0, -1])
def test_rejects_bad_k(dps, k):
    with pytest.raises(ValueError, match="positive integer"):
        navis.nblast_knn(dps, k=k, progress=False)


def test_builtin_backend_is_rejected(dps):
    """There is no built-in k-NN, and the error must say so plainly."""
    with pytest.raises(ValueError, match="does not implement"):
        navis.nblast_knn(dps, k=2, backend="builtin", progress=False)


def test_without_fastcore(dps, monkeypatch):
    """Without fastcore we must raise, not fall back to a backend that can't."""
    from navis import utils

    monkeypatch.setattr(utils, "fastcore", None)
    with pytest.raises(ValueError, match="No available NBLAST backend"):
        navis.nblast_knn(dps, k=2, progress=False)


def test_with_outdated_fastcore(dps, monkeypatch):
    """An older fastcore serves the other operations but must be told apart."""
    from navis import utils

    real = utils.fastcore

    class Outdated:
        def __getattr__(self, name):
            if name == "nblast_knn":
                raise AttributeError(name)
            return getattr(real, name)

    monkeypatch.setattr(utils, "fastcore", Outdated())
    with pytest.raises(ValueError, match="too old"):
        navis.nblast_knn(dps, k=2, progress=False)

    # ... and the operations it *does* have still work
    assert navis.nblast_allbyall(dps, backend="fastcore", progress=False).shape == (
        len(dps),
        len(dps),
    )
