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


# --------------------------------------------------------------------------- #
# Where the blocks run
# --------------------------------------------------------------------------- #
class Recorder(navis.compute.ParallelBackend):
    """Runs blocks inline, but records - and can reorder - what it was given."""

    name = "recorder"
    priority = 999
    auto_select = False

    def __init__(self, *, reverse=False, workers=None):
        self.reverse = reverse
        self.workers = workers
        self.units = []

    def worker_count(self, hint):
        return hint if self.workers is None else self.workers

    def map(self, func, payloads, *, n_workers):
        payloads = list(payloads)
        self.units.append(len(payloads))
        results = [func(p) for p in payloads]
        # Completion order is explicitly not input order
        yield from (reversed(results) if self.reverse else results)


@pytest.mark.parametrize("parallel_backend",
                         ["serial", "threads", "processes", "joblib", "pathos"])
def test_scores_do_not_depend_on_where_blocks_run(dps, parallel_backend):
    """Every backend has to produce the numbers the reference run produced."""
    if parallel_backend not in navis.list_parallel_backends():
        pytest.skip(f"{parallel_backend} not installed")

    expected = navis.nblast(dps[:3], dps[3:], backend="builtin", n_cores=1,
                            progress=False)

    with navis.set_parallel_backend(parallel_backend):
        scores = navis.nblast(dps[:3], dps[3:], backend="builtin", n_cores=4,
                              progress=False)

    assert_frame_equal(scores, expected)


def test_blocks_land_correctly_when_results_arrive_out_of_order(dps):
    """Completion order is the only order any transport guarantees.

    Every block carries the slice of the matrix it belongs to, so a backend
    that hands results back backwards must still produce the same matrix.
    """
    expected = navis.nblast_allbyall(dps, backend="builtin", n_cores=1,
                                     progress=False)

    backwards = Recorder(reverse=True)
    with navis.set_parallel_backend(backwards):
        scores = navis.nblast_allbyall(dps, backend="builtin", n_cores=4,
                                       progress=False)

    assert sum(backwards.units) > 1      # it really was split up
    assert_frame_equal(scores, expected)


def test_grid_is_sized_by_the_backend_not_by_n_cores(dps):
    """`n_cores` describes this machine; a cluster is a different size.

    A block is a unit of work, so how finely to cut the matrix is a question
    about the *cluster*, not about how many cores the submitting laptop has.
    """
    small = Recorder(workers=2)
    with navis.set_parallel_backend(small):
        navis.nblast_allbyall(dps, backend="builtin", n_cores=2, progress=False)

    # Same `n_cores`, but this backend says it has far more workers
    big = Recorder(workers=32)
    with navis.set_parallel_backend(big):
        navis.nblast_allbyall(dps, backend="builtin", n_cores=2, progress=False)

    assert sum(big.units) > sum(small.units)


def test_one_core_never_leaves_this_process(dps):
    """`n_cores=1` means serial, whatever backend happens to be configured."""
    recorder = Recorder()
    with navis.set_parallel_backend(recorder):
        scores = navis.nblast(dps[:3], dps[3:], backend="builtin", n_cores=1,
                              progress=False)

    assert recorder.units == []
    assert scores.shape == (3, 2)


def test_serial_backend_does_not_split_the_matrix(dps):
    """A backend that runs nothing side by side should not be handed blocks.

    Partitioning costs a copy of each neuron per block it appears in, which
    buys nothing when the blocks then run one after another anyway.
    """
    from navis.nbl.backends.builtin import BuiltinBackend
    from navis.compute.backends import get_backend

    serial = get_backend("serial")
    assert BuiltinBackend()._partition(dps, dps, n_cores=8, progress=False,
                                       backend=serial) == (1, 1)

    with navis.set_parallel_backend("serial"):
        scores = navis.nblast_allbyall(dps, backend="builtin", n_cores=8,
                                       progress=False)
    expected = navis.nblast_allbyall(dps, backend="builtin", n_cores=1,
                                     progress=False)
    assert_frame_equal(scores, expected)


def test_a_failing_block_raises_the_original_exception(dps, monkeypatch):
    """A block dying must not surface as something about the transport."""
    from navis.nbl.backends import builtin

    def boom(blaster):
        raise RuntimeError("block went bang")

    monkeypatch.setattr(builtin, "_run_job", boom)

    with pytest.raises(RuntimeError, match="block went bang"):
        with navis.set_parallel_backend("threads"):
            navis.nblast(dps[:3], dps[3:], backend="builtin", n_cores=4,
                         progress=False)

