"""Tests for the per-worker thread cap in `navis.compute.threads`.

navis spreads work over processes; navis-fastcore (and BLAS, and OpenMP) spread
it over threads. Nothing tells a worker that it is one of twenty, so without a
cap the two multiply - 20 workers x 224 threads over 224 cores, which measured
*slower* than not parallelising at all. These tests cover the three things that
has to get right: the arithmetic, that the cap actually arrives in the worker,
and that it never touches the calling process.
"""

import os

import pytest

import navis
from navis import config
from navis.compute import dispatch, threads
from navis.compute.backends import get_backend


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def probe_env(neuron):
    """Report the thread caps in force. Module level so it pickles."""
    return (os.environ.get('RAYON_NUM_THREADS'),
            os.environ.get('OMP_NUM_THREADS'))


class FakeBackend:
    """Just enough backend for `resolve_thread_cap` to interrogate."""

    def __init__(self, shares_machine=True, workers=None):
        self.shares_machine = shares_machine
        self.workers = workers

    def worker_count(self, hint):
        """A real backend may know better than the caller's hint."""
        return self.workers or hint


@pytest.fixture
def restore_threads():
    """Undo anything a test does to the module's idea of the current cap."""
    before = threads._APPLIED
    yield
    threads._APPLIED = before


# --------------------------------------------------------------------------- #
# cpu_count
# --------------------------------------------------------------------------- #
def test_cpu_count_prefers_process_cpu_count(monkeypatch):
    """The 3.13+ spelling wins: it honours affinity *and* PYTHON_CPU_COUNT."""
    monkeypatch.setattr(os, 'process_cpu_count', lambda: 3, raising=False)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(os, 'cpu_count', lambda: 64)

    assert dispatch.cpu_count() == 3


def test_cpu_count_falls_back_to_affinity(monkeypatch):
    """Below 3.13, ask for the affinity mask - the machine's count over-counts
    under `taskset` and SLURM's `--cpus-per-task`."""
    monkeypatch.delattr(os, 'process_cpu_count', raising=False)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(os, 'cpu_count', lambda: 64)

    assert dispatch.cpu_count() == 2


def test_cpu_count_falls_back_to_machine(monkeypatch):
    """No affinity API at all (macOS, Windows)."""
    monkeypatch.delattr(os, 'process_cpu_count', raising=False)
    monkeypatch.delattr(os, 'sched_getaffinity', raising=False)
    monkeypatch.setattr(os, 'cpu_count', lambda: 8)

    assert dispatch.cpu_count() == 8


def test_cpu_count_never_zero(monkeypatch):
    """`os.cpu_count()` returns None where the platform can't say, and every
    caller divides by this."""
    monkeypatch.delattr(os, 'process_cpu_count', raising=False)
    monkeypatch.delattr(os, 'sched_getaffinity', raising=False)
    monkeypatch.setattr(os, 'cpu_count', lambda: None)

    assert dispatch.cpu_count() == 1
    assert dispatch.default_n_workers() >= 1


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #
def test_auto_divides_the_machine(monkeypatch):
    monkeypatch.setattr(dispatch, 'cpu_count', lambda: 16)

    assert dispatch.resolve_thread_cap(FakeBackend(), 4) == 4
    assert dispatch.resolve_thread_cap(FakeBackend(), 16) == 1
    # More workers than cores: never below one thread apiece
    assert dispatch.resolve_thread_cap(FakeBackend(), 64) == 1


def test_auto_leaves_cluster_backends_alone(monkeypatch):
    """`cpu_count() // n_workers` describes the submitting machine and says
    nothing about the node a job lands on."""
    monkeypatch.setattr(dispatch, 'cpu_count', lambda: 16)

    assert dispatch.resolve_thread_cap(FakeBackend(shares_machine=False), 4) is None


def test_auto_divides_by_the_real_worker_count(monkeypatch):
    """`n_workers` is a hint; a backend that knows its own width overrides it.

    Sizing against the hint is how you end up handing 32 real workers eight
    cores' worth of threads each - the bug this whole mechanism exists to stop.
    """
    monkeypatch.setattr(dispatch, 'cpu_count', lambda: 16)

    assert dispatch.resolve_thread_cap(FakeBackend(workers=16), 4) == 1


def test_a_pool_we_started_ourselves_needs_no_backend(monkeypatch):
    """`worker_initializer`'s case: always local, so there is nothing to ask."""
    monkeypatch.setattr(dispatch, 'cpu_count', lambda: 16)

    assert dispatch.resolve_thread_cap(None, 4) == 4


def test_explicit_value_applies_everywhere(monkeypatch):
    """A number is something the caller knows and we cannot derive - honour it
    even where the arithmetic would not apply."""
    monkeypatch.setattr(config, 'inner_max_num_threads', 2)

    assert dispatch.resolve_thread_cap(FakeBackend(), 4) == 2
    assert dispatch.resolve_thread_cap(FakeBackend(shares_machine=False), 4) == 2


def test_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, 'inner_max_num_threads', None)
    assert dispatch.resolve_thread_cap(FakeBackend(), 4) is None

    # 0 is how `set_parallel_backend` spells "no cap", since None there means
    # "leave the setting alone"
    monkeypatch.setattr(config, 'inner_max_num_threads', 0)
    assert dispatch.resolve_thread_cap(FakeBackend(), 4) is None


def test_requested_overrides_the_config(monkeypatch):
    """How NBLAST pins one thread per worker whatever the machine looks like."""
    monkeypatch.setattr(config, 'inner_max_num_threads', 8)

    assert dispatch.resolve_thread_cap(FakeBackend(), 4, requested=1) == 1


def test_setter_scopes_the_change():
    before = config.inner_max_num_threads
    with navis.set_parallel_backend(inner_max_num_threads=1):
        assert config.inner_max_num_threads == 1
    assert config.inner_max_num_threads == before


# --------------------------------------------------------------------------- #
# Does it actually reach the worker?
# --------------------------------------------------------------------------- #
def test_worker_gets_the_cap():
    nl = navis.example_neurons(2)

    caps = nl.apply(probe_env, parallel=True, n_cores=2, backend='processes')

    expected = str(max(1, dispatch.cpu_count() // 2))
    assert all(rayon == expected for rayon, _ in caps)


def probe_fastcore(neuron):
    """Ask fastcore itself. Module level so it pickles."""
    import navis_fastcore as fastcore
    return fastcore.get_num_threads()


@pytest.mark.skipif(
    not hasattr(pytest.importorskip('navis_fastcore'), 'get_num_threads'),
    reason='navis-fastcore < 0.11 cannot report its thread count')
def test_worker_fastcore_pool_is_actually_sized():
    """The env var is only the mechanism - this is the thing we care about.

    Rayon reads `RAYON_NUM_THREADS` when it lazily builds its global pool, so
    this also pins down the ordering: the cap has to be applied before the
    worker's first fastcore call, not merely at some point during the chunk.
    """
    nl = navis.example_neurons(2)

    with navis.set_parallel_backend(inner_max_num_threads=1):
        counts = nl.apply(probe_fastcore, parallel=True, n_cores=2,
                          backend='processes')

    assert all(n == 1 for n in counts)


def test_worker_honours_an_explicit_cap():
    nl = navis.example_neurons(2)

    with navis.set_parallel_backend(inner_max_num_threads=1):
        caps = nl.apply(probe_env, parallel=True, n_cores=2, backend='processes')

    assert all(cap == ('1', '1') for cap in caps)


def test_changing_the_cap_is_not_ignored():
    """A pool is reused between calls, but a thread pool cannot be resized once
    built - so a worker started under one cap must not serve a call that wants
    another. The backend keeps the cap in its pool identity to make sure of it.
    """
    nl = navis.example_neurons(2)

    with navis.set_parallel_backend(inner_max_num_threads=1):
        first = nl.apply(probe_env, parallel=True, n_cores=2, backend='processes')
    with navis.set_parallel_backend(inner_max_num_threads=2):
        second = nl.apply(probe_env, parallel=True, n_cores=2, backend='processes')

    assert first[0] == ('1', '1')
    assert second[0] == ('2', '2')


@pytest.mark.parametrize('backend', ['serial', 'threads'])
def test_in_process_backends_are_left_alone(backend):
    """rayon's pool is per *process*, so N threads sharing one pool is not
    oversubscription - and capping here would hobble the caller's own session.

    Asserts that nothing *changed* rather than that nothing is set: these are
    ordinary environment variables that a user's shell, or another library
    entirely (`distributed` sets `OMP_NUM_THREADS` when a cluster starts), may
    have set long before navis was imported.
    """
    nl = navis.example_neurons(2)
    before = probe_env(None)

    caps = nl.apply(probe_env, parallel=True, n_cores=2, backend=backend)

    assert all(cap == before for cap in caps)


def test_a_backend_that_lies_about_isolation_cannot_touch_us():
    """`isolated` is a claim, and worker set-up cannot always be undone.

    A backend that says its workers have their own address space but runs the
    work inline would otherwise cap the *caller's* interpreter - for the rest
    of the session, with nothing to say why. Getting `isolated` wrong should
    cost a needless copy, not the machine.
    """
    from navis.compute.dispatch import WorkerContext

    ctx = WorkerContext.snapshot(threads=1)
    before = dict(os.environ)

    assert not ctx.is_foreign()
    ctx.apply()      # i.e. applied in the process that took the snapshot

    assert dict(os.environ) == before


def test_parent_environment_is_never_touched():
    """`OMP_NUM_THREADS` is part of the environment loky builds its workers
    from: changing it here would make joblib rebuild its pool on the way in and
    again on the way out.
    """
    nl = navis.example_neurons(2)
    before = dict(os.environ)

    nl.apply(probe_env, parallel=True, n_cores=2, backend='processes')

    assert dict(os.environ) == before


# --------------------------------------------------------------------------- #
# limit_native_threads
# --------------------------------------------------------------------------- #
def test_limit_native_threads_is_a_no_op_without_a_cap(restore_threads):
    before = dict(os.environ)
    threads.limit_native_threads(None)
    threads.limit_native_threads(0)
    assert dict(os.environ) == before


def test_fastcore_refusal_does_not_propagate(monkeypatch, restore_threads):
    """A pool built at another size cannot be resized; fastcore says so by
    raising. That is a performance problem, not a failure - and it happens in a
    worker whose stderr nobody reads.
    """
    import navis_fastcore as fastcore

    def refuse(n):
        raise RuntimeError('pool already built')

    monkeypatch.setattr(fastcore, 'set_num_threads', refuse, raising=False)
    monkeypatch.setattr(os, 'environ', dict(os.environ))

    threads._APPLIED = None
    threads.limit_native_threads(2)      # must not raise


def test_set_num_threads_rejects_nonsense():
    with pytest.raises(ValueError):
        navis.set_num_threads(0)
    with pytest.raises(ValueError):
        navis.set_num_threads(-1)


# --------------------------------------------------------------------------- #
# The pools navis starts outside the backends
# --------------------------------------------------------------------------- #
def test_worker_initializer_is_picklable():
    """It travels to the worker as a pool `initializer`, so it has to be."""
    import pickle

    init = dispatch.worker_initializer(4)

    assert pickle.loads(pickle.dumps(init)) is not None


def test_worker_initializer_sizes_from_the_worker_count(monkeypatch):
    monkeypatch.setattr(dispatch, 'cpu_count', lambda: 16)

    assert dispatch.worker_initializer(4).keywords['threads'] == 4
    assert dispatch.worker_initializer(16).keywords['threads'] == 1
