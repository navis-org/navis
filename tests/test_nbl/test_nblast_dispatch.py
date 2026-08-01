"""How a built-in NBLAST is split into blocks, dispatched and reassembled.

Everything here pins the *builtin* backend explicitly. navis-fastcore computes
the whole matrix in a single Rust call and never touches this machinery, so
letting `backend="auto"` decide would silently stop testing it on any machine
that has fastcore installed.
"""

import navis
import numpy as np
import pytest

from pandas.testing import assert_frame_equal


@pytest.fixture(scope="module")
def dps():
    """Dotprops in microns with unique IDs. NBLAST only ever reads them."""
    nl = navis.example_neurons(n=5, kind="skeleton")
    return navis.NeuronList([navis.make_dotprops(n / 1000, k=5) for n in nl])


@pytest.fixture(scope="module")
def with_connectors():
    """Neurons in microns for SynBLAST (which needs connectors, not points)."""
    return navis.example_neurons(n=4, kind="skeleton") / 1000


# --------------------------------------------------------------------------- #
# scores='both'
# --------------------------------------------------------------------------- #
# `both` is the only mode whose result is not the same shape as query x target:
# `multi_query_target` stacks the forward and reverse score for each query under
# a (query, score) MultiIndex. The stitching used to assume one row per query,
# so any NBLAST that got split into more than one block died with an opaque
# "setting an array element with a sequence".
@pytest.mark.parametrize("n_cores", [1, 2, 4, 8])
def test_both_scores_survive_partitioning(dps, n_cores):
    query, target = dps[:3], dps[3:]
    expected = navis.nblast(query, target, backend="builtin", scores="both",
                            n_cores=1, progress=False)

    scores = navis.nblast(query, target, backend="builtin", scores="both",
                          n_cores=n_cores, progress=False)

    # Twice as tall as there are queries - one row per query per direction
    assert scores.shape == (2 * len(query), len(target))
    assert scores.index.names == ["query", "score"]
    assert scores.equals(expected)


def test_both_scores_really_are_the_two_directions(dps):
    """The reverse plane must be the transpose of the forward one.

    Partitioning splits queries and targets independently, so a block holds
    `q -> t` for its own slice while `t -> q` lives in a *different* block. Any
    mix-up in where a block's rows land shows up here and not in a shape check.
    """
    both = navis.nblast(dps, dps, backend="builtin", scores="both",
                        n_cores=4, progress=False)
    forward = both.xs("forward", level="score")
    reverse = both.xs("reverse", level="score")

    assert np.allclose(reverse.values, forward.values.T)
    # ... and the forward plane is just a plain forward NBLAST
    plain = navis.nblast(dps, dps, backend="builtin", n_cores=4, progress=False)
    assert np.allclose(forward.values, plain.values)


def test_smart_rejects_both_scores(dps):
    """`nblast_smart` scores individual pairs - there is no room for two."""
    with pytest.raises(ValueError, match="scores"):
        navis.nblast_smart(dps[:3], dps[3:], t=50, scores="both",
                           backend="builtin", progress=False)


def test_synblast_rejects_both_scores(with_connectors):
    """SynBLAST has no reverse-score path; it used to return forward twice."""
    with pytest.raises(ValueError, match="scores"):
        navis.synblast(with_connectors[:2], with_connectors[2:], scores="both",
                       backend="builtin", progress=False)


# --------------------------------------------------------------------------- #
# Partitioning must not change the answer
# --------------------------------------------------------------------------- #
# Every cell of the matrix is computed independently of how the matrix was cut
# up, so a partitioned run has to be *bit-identical* to a serial one - not
# merely close. Anything less means a block landed in the wrong place.
OPERATIONS = ["nblast", "nblast_mean", "nblast_min", "nblast_both",
              "allbyall", "smart_percentile", "smart_N", "smart_aba",
              "synblast", "synblast_by_type"]


def run_operation(op, dps, with_connectors, n_cores):
    """Run one NBLAST operation at a given core count."""
    query, target = dps[:3], dps[3:]
    syn_query, syn_target = with_connectors[:2], with_connectors[2:]
    kwargs = dict(backend="builtin", n_cores=n_cores, progress=False)

    if op.startswith("nblast"):
        _, _, scores = op.partition("_")
        return navis.nblast(query, target, scores=scores or "forward", **kwargs)
    if op == "allbyall":
        return navis.nblast_allbyall(dps, **kwargs)
    if op == "smart_percentile":
        return navis.nblast_smart(query, target, t=50, **kwargs)
    if op == "smart_N":
        return navis.nblast_smart(query, target, t=2, criterion="N", **kwargs)
    if op == "smart_aba":
        return navis.nblast_smart(dps, t=50, **kwargs)
    if op == "synblast":
        return navis.synblast(syn_query, syn_target, **kwargs)
    if op == "synblast_by_type":
        return navis.synblast(syn_query, syn_target, by_type=True, **kwargs)
    raise ValueError(f"Unknown operation '{op}'")


@pytest.mark.parametrize("op", OPERATIONS)
def test_partitioned_matches_serial(op, dps, with_connectors):
    serial = run_operation(op, dps, with_connectors, n_cores=1)
    partitioned = run_operation(op, dps, with_connectors, n_cores=4)

    assert_frame_equal(partitioned, serial)


def test_smart_returns_the_mask_it_used(dps):
    """`return_mask` must describe the same cells whether split or not."""
    scores, mask = navis.nblast_smart(dps[:3], dps[3:], t=2, criterion="N",
                                      return_mask=True, backend="builtin",
                                      n_cores=4, progress=False)

    assert mask.shape == scores.shape
    # criterion="N" keeps the top `t` targets for every query
    assert (mask.sum(axis=1) == 2).all()

