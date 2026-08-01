"""Tests for the cluster backends: dask.distributed and submitit.

These run against real machinery - a local dask cluster and submitit's local
executors - because the interesting failures are the ones a mock cannot have:
whether a neuron survives cloudpickle, whether results come back in input
order, and whether a worker's exception reaches the caller intact. Both
dependencies are optional, so everything here skips cleanly without them.

The contract every cluster backend has to meet is parametrised over both, so a
third one cannot be added while quietly meeting only half of it. Tests below
that are specific to one backend say so in their name.
"""

import concurrent.futures as cf

import pytest

import navis
from navis.compute.backends import resolve_backend

try:
    import distributed
except ImportError:
    distributed = None

try:
    import submitit
except ImportError:
    submitit = None

# Marks rather than a module-level `importorskip` so that having only one of
# the two installed still runs that one's tests.
requires_dask = pytest.mark.skipif(distributed is None,
                                   reason='dask.distributed not installed')
requires_submitit = pytest.mark.skipif(submitit is None,
                                       reason='submitit not installed')


@pytest.fixture
def neurons():
    return navis.example_neurons(4)


@pytest.fixture(scope='module')
def dask_client():
    """A small local cluster of *processes* - i.e. genuinely isolated."""
    with distributed.LocalCluster(n_workers=2, threads_per_worker=1,
                                  processes=True, dashboard_address=None,
                                  silence_logs=50) as c:
        with distributed.Client(c) as client:
            yield client


@pytest.fixture
def local_executor(tmp_path):
    """Runs jobs as subprocesses - slower, but really does pickle and spawn."""
    ex = submitit.AutoExecutor(folder=str(tmp_path), cluster='local')
    ex.update_parameters(timeout_min=5)
    return ex


@pytest.fixture
def debug_executor(tmp_path):
    """Runs jobs inline - fast, and exercises everything but serialisation."""
    return submitit.AutoExecutor(folder=str(tmp_path), cluster='debug')


@pytest.fixture(params=['dask', 'submitit'])
def cluster(request):
    """A configured scheduler object, once per cluster backend."""
    if request.param == 'dask':
        if distributed is None:
            pytest.skip('dask.distributed not installed')
        return request.getfixturevalue('dask_client')

    if submitit is None:
        pytest.skip('submitit not installed')
    return request.getfixturevalue('local_executor')


# --------------------------------------------------------------------------- #
# The contract both backends have to meet
# --------------------------------------------------------------------------- #
def test_cluster_matches_serial(cluster, neurons):
    expected = navis.prune_twigs(neurons, 5000, inplace=False)

    with navis.set_parallel_backend(cluster):
        got = navis.prune_twigs(neurons, 5000, parallel=True, inplace=False,
                                n_cores=2)

    # Results come back in completion order and are reordered by the caller -
    # this is what checks that they were put back correctly.
    assert [n.id for n in got] == [n.id for n in expected]
    assert [n.n_nodes for n in got] == [n.n_nodes for n in expected]


def test_cluster_leaves_the_caller_alone(cluster, neurons):
    before = [n.n_nodes for n in neurons]
    with navis.set_parallel_backend(cluster):
        navis.prune_twigs(neurons, 5000, parallel=True, inplace=False,
                          n_cores=2)
    assert [n.n_nodes for n in neurons] == before


def test_cluster_ships_a_lambda(cluster, neurons):
    with navis.set_parallel_backend(cluster):
        res = neurons.apply(lambda n: n.n_nodes, parallel=True, n_cores=2)
    assert res == [n.n_nodes for n in neurons]


def test_cluster_propagates_the_original_exception(cluster, neurons):
    """Which exception you catch must not depend on where the work ran."""
    def boom(n):
        raise ValueError('kaboom')

    with navis.set_parallel_backend(cluster):
        with pytest.raises(ValueError, match='kaboom'):
            neurons.apply(boom, parallel=True, n_cores=2)


def test_cluster_omit_failures_keeps_the_survivors(cluster, neurons):
    def odd_boom(n):
        if n.id % 2:
            raise ValueError('kaboom')
        return n.id

    with navis.set_parallel_backend(cluster):
        got = neurons.apply(odd_boom, parallel=True, n_cores=2,
                            omit_failures=True)

    assert list(got) == [n.id for n in neurons if not n.id % 2]


def test_cluster_bundles_neurons_into_fewer_units(cluster):
    """One neuron per unit is a round trip - or a queued job - per neuron."""
    be = resolve_backend(cluster)
    cs = be.chunksize(10_000, n_workers=20)
    assert cs > 1
    assert -(-10_000 // cs) <= (be.worker_count(20) or 20) * be.chunks_per_worker


# --------------------------------------------------------------------------- #
# dask
# --------------------------------------------------------------------------- #
@requires_dask
def test_dask_adopts_client_executor_and_cluster(dask_client):
    """All three ways of naming the same cluster must land in the same place."""
    for obj in (dask_client, dask_client.get_executor(), dask_client.cluster):
        be = resolve_backend(obj)
        assert be.name == 'dask'
        assert be.pickles_by_value is True
        assert be.chunks_per_worker > 1

    # A process-backed cluster is isolated, so `inplace=True` is safe there
    assert resolve_backend(dask_client).isolated is True


@requires_dask
def test_unrecognised_dask_executor_still_bundles():
    """We know it is dask even if we can't get a client out of it."""
    class SomeOtherExecutor(cf.Executor):
        pass

    SomeOtherExecutor.__module__ = 'distributed.somewhere_new'

    be = resolve_backend(SomeOtherExecutor())
    assert be.pickles_by_value is True          # dask uses cloudpickle
    assert be.chunks_per_worker > 1
    # ... but anything the user states explicitly wins
    from navis.compute.backends import set_parallel_backend
    with set_parallel_backend(SomeOtherExecutor(), chunks_per_worker=1):
        assert resolve_backend(None).chunks_per_worker == 1


@requires_dask
def test_dask_sizes_chunks_against_the_cluster(dask_client):
    """`n_workers` describes this machine and says nothing about the cluster."""
    be = resolve_backend(dask_client)
    assert be.worker_count(1) == len(dask_client.scheduler_info()['workers'])

    cs = be.chunksize(10_000, n_workers=1)
    assert -(-10_000 // cs) <= be.worker_count(1) * be.chunks_per_worker
    # ... and an explicit request still wins
    assert be.chunksize(10_000, n_workers=1, requested=7) == 7


@requires_dask
def test_threaded_dask_cluster_is_not_isolated(neurons):
    """`processes=False` runs workers in *our* interpreter.

    Treating that as isolated would let navis turn a caller's `inplace=False`
    into an in-place operation and quietly corrupt their neurons.
    """
    with distributed.LocalCluster(n_workers=2, threads_per_worker=1,
                                  processes=False, dashboard_address=None,
                                  silence_logs=50) as c:
        with distributed.Client(c) as client:
            be = resolve_backend(client)
            assert be.isolated is False

            before = [n.n_nodes for n in neurons]
            with navis.set_parallel_backend(client):
                navis.prune_twigs(neurons, 5000, parallel=True, inplace=False,
                                  n_cores=2)
            assert [n.n_nodes for n in neurons] == before


# --------------------------------------------------------------------------- #
# submitit
# --------------------------------------------------------------------------- #
@requires_submitit
def test_submitit_adopts_its_executors(local_executor, debug_executor):
    be = resolve_backend(local_executor)
    assert be.name == 'submitit'
    assert be.pickles_by_value is True
    assert be.isolated is True

    # The debug executor runs the job in this interpreter
    assert resolve_backend(debug_executor).isolated is False


@requires_submitit
def test_submitit_polls_faster_for_local_executors(local_executor):
    """10s between checks is right for a queue, absurd for a subprocess."""
    from navis.compute.backends._submitit import (POLL_FREQUENCY,
                                                  LOCAL_POLL_FREQUENCY,
                                                  SubmititBackend)

    assert resolve_backend(local_executor).poll_frequency == LOCAL_POLL_FREQUENCY
    assert SubmititBackend().poll_frequency == POLL_FREQUENCY


@requires_submitit
def test_submitit_without_an_executor_fails_when_you_set_it():
    """It can never work, so there is nothing to wait for."""
    with pytest.raises(ValueError, match='needs an executor'):
        navis.set_parallel_backend('submitit')

    # And a directly-constructed one, which bypasses that check, still refuses
    # rather than getting as far as submitting
    from navis.compute.backends._submitit import SubmititBackend
    from navis.compute import dispatch
    with pytest.raises(ValueError, match='needs an executor'):
        dispatch.map_tasks([(len, ('ab',), {})], backend=SubmititBackend(),
                           n_workers=2, disable=True)


@requires_submitit
def test_submitit_carries_the_worker_traceback_home(local_executor, neurons):
    """submitit only stores a traceback string - see `marshals_exceptions`.

    The exception is rebuilt in the parent, so without the chained
    `RemoteTraceback` the traceback would stop at the dispatcher and say
    nothing about which line in the user's function failed.
    """
    def boom(n):
        raise ValueError('kaboom')

    with navis.set_parallel_backend(local_executor):
        with pytest.raises(ValueError, match='kaboom') as exc:
            neurons.apply(boom, parallel=True, n_cores=2)

    from navis.compute.dispatch import RemoteTraceback
    assert isinstance(exc.value.__cause__, RemoteTraceback)
    assert 'in boom' in str(exc.value.__cause__)
