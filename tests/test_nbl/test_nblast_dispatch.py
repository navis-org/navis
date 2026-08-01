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

from tests.test_compute_backends import DummyBackend


@pytest.fixture(scope="module")
def dps():
    """Dotprops in microns with unique IDs. NBLAST only ever reads them."""
    return navis.make_dotprops(navis.example_neurons(n=5, kind="skeleton") / 1000,
                               k=5, progress=False)


@pytest.fixture(scope="module")
def with_connectors():
    """Neurons in microns for SynBLAST (which needs connectors, not points)."""
    return navis.example_neurons(n=4, kind="skeleton") / 1000


@pytest.fixture(scope="module")
def serial_scores(dps):
    """The reference every dispatched run has to reproduce exactly."""
    query, target = dps[:3], dps[3:]
    return {
        scores: navis.nblast(query, target, backend="builtin", scores=scores,
                             n_cores=1, progress=False)
        for scores in ("forward", "both")
    }


# --------------------------------------------------------------------------- #
# scores='both'
# --------------------------------------------------------------------------- #
# `both` is the only mode whose result is not the same shape as query x target:
# `multi_query_target` stacks the forward and reverse score for each query under
# a (query, score) MultiIndex. The stitching used to assume one row per query,
# so any NBLAST that got split into more than one block died with an opaque
# "setting an array element with a sequence".
# n_cores 1 / 2 / 4 give a 1x1, a 2x2 and a 3x2 grid on this matrix - i.e.
# unpartitioned, partly split and split as far as it goes.
@pytest.mark.parametrize("n_cores", [1, 2, 4])
def test_both_scores_survive_partitioning(dps, serial_scores, n_cores):
    query, target = dps[:3], dps[3:]

    scores = navis.nblast(query, target, backend="builtin", scores="both",
                          n_cores=n_cores, progress=False)

    # Twice as tall as there are queries - one row per query per direction
    assert scores.shape == (2 * len(query), len(target))
    assert scores.index.names == ["query", "score"]
    assert scores.equals(serial_scores["both"])


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
#: name -> how to run it, given (dotprops, neurons-with-connectors, **kwargs)
OPERATIONS = {
    "nblast": lambda d, s, **kw: navis.nblast(d[:3], d[3:], **kw),
    "nblast_mean": lambda d, s, **kw: navis.nblast(d[:3], d[3:], scores="mean", **kw),
    "nblast_min": lambda d, s, **kw: navis.nblast(d[:3], d[3:], scores="min", **kw),
    "nblast_both": lambda d, s, **kw: navis.nblast(d[:3], d[3:], scores="both", **kw),
    "allbyall": lambda d, s, **kw: navis.nblast_allbyall(d, **kw),
    "smart_percentile": lambda d, s, **kw: navis.nblast_smart(d[:3], d[3:], t=50, **kw),
    "smart_N": lambda d, s, **kw: navis.nblast_smart(d[:3], d[3:], t=2,
                                                     criterion="N", **kw),
    "smart_aba": lambda d, s, **kw: navis.nblast_smart(d, t=50, **kw),
    "synblast": lambda d, s, **kw: navis.synblast(s[:2], s[2:], **kw),
    "synblast_by_type": lambda d, s, **kw: navis.synblast(s[:2], s[2:],
                                                          by_type=True, **kw),
}


@pytest.mark.parametrize("run", OPERATIONS.values(), ids=OPERATIONS)
def test_partitioned_matches_serial(run, dps, with_connectors):
    kwargs = dict(backend="builtin", progress=False)
    serial = run(dps, with_connectors, n_cores=1, **kwargs)
    partitioned = run(dps, with_connectors, n_cores=4, **kwargs)

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
class Cluster(DummyBackend):
    """A `DummyBackend` that claims to have more workers than we do.

    `DummyBackend` (see `tests/test_compute_backends.py`) already runs blocks
    inline and can hand results back in reverse; all NBLAST adds is a worker
    count it did not get from `n_cores`.
    """

    def __init__(self, *, workers, **kwargs):
        super().__init__(**kwargs)
        self.workers = workers

    def worker_count(self, hint):
        return self.workers


def blocks_dispatched(backend):
    """How many blocks a `DummyBackend` was handed, across all its calls."""
    return sum(len(call['payloads']) for call in backend.calls)


@pytest.mark.parametrize("parallel_backend",
                         ["serial", "threads", "processes", "joblib", "pathos"])
def test_scores_do_not_depend_on_where_blocks_run(dps, serial_scores,
                                                  parallel_backend):
    """Every backend has to produce the numbers the reference run produced."""
    if parallel_backend not in navis.list_parallel_backends():
        pytest.skip(f"{parallel_backend} not installed")

    with navis.set_parallel_backend(parallel_backend):
        scores = navis.nblast(dps[:3], dps[3:], backend="builtin", n_cores=4,
                              progress=False)

    assert_frame_equal(scores, serial_scores["forward"])


def test_blocks_land_correctly_when_results_arrive_out_of_order(dps):
    """Completion order is the only order any transport guarantees.

    Every block carries the slice of the matrix it belongs to, so a backend
    that hands results back backwards must still produce the same matrix.
    """
    expected = navis.nblast_allbyall(dps, backend="builtin", n_cores=1,
                                     progress=False)

    backwards = DummyBackend(reverse=True)
    with navis.set_parallel_backend(backwards):
        scores = navis.nblast_allbyall(dps, backend="builtin", n_cores=4,
                                       progress=False)

    assert blocks_dispatched(backwards) > 1      # it really was split up
    assert_frame_equal(scores, expected)


def test_grid_is_sized_by_the_backend_not_by_n_cores(dps):
    """`n_cores` describes this machine; a cluster is a different size.

    A block is a unit of work, so how finely to cut the matrix is a question
    about the *cluster*, not about how many cores the submitting laptop has.
    """
    dispatched = {}
    for workers in (2, 32):
        backend = Cluster(workers=workers)
        with navis.set_parallel_backend(backend):
            navis.nblast_allbyall(dps, backend="builtin", n_cores=2,
                                  progress=False)
        dispatched[workers] = blocks_dispatched(backend)

    # Same `n_cores` throughout - only what the backend claims differs
    assert dispatched[32] > dispatched[2]


def test_one_core_never_leaves_this_process(dps):
    """`n_cores=1` means serial, whatever backend happens to be configured."""
    backend = DummyBackend()
    with navis.set_parallel_backend(backend):
        scores = navis.nblast(dps[:3], dps[3:], backend="builtin", n_cores=1,
                              progress=False)

    assert backend.calls == []
    assert scores.shape == (3, 2)


def test_a_backend_that_runs_nothing_in_parallel_reports_one_worker(dps):
    """A matrix split into blocks that then run one after another costs a copy
    of each neuron per block and buys nothing.

    `serial` says so itself - `worker_count` is 1 - rather than the partitioner
    carrying a special case for it, which would still leave a single-slot
    executor or any other non-concurrent backend splitting work up.
    """
    from navis.compute.backends import get_backend

    assert get_backend("serial").worker_count(8) == 1

    with navis.set_parallel_backend("serial"):
        scores = navis.nblast_allbyall(dps, backend="builtin", n_cores=8,
                                       progress=False)
    expected = navis.nblast_allbyall(dps, backend="builtin", n_cores=1,
                                     progress=False)
    assert_frame_equal(scores, expected)


def test_a_failing_block_raises_the_original_exception(dps, monkeypatch):
    """A block dying must not surface as something about the transport."""
    from navis.nbl.backends import builtin

    def boom(blaster, omp_limit=None):
        raise RuntimeError("block went bang")

    monkeypatch.setattr(builtin, "_run_job", boom)

    with pytest.raises(RuntimeError, match="block went bang"):
        with navis.set_parallel_backend("threads"):
            navis.nblast(dps[:3], dps[3:], backend="builtin", n_cores=4,
                         progress=False)

