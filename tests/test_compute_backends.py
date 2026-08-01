"""Tests for the pluggable parallel backends in `navis.compute`.

Two halves: the registry (which backend gets picked, and why) and the dispatch
seam (chunking, ordering, failure handling). The seam is exercised through a
recording dummy backend so the interesting properties - notably that results
are reordered correctly when they come back out of order - can be tested
deterministically, which is impossible with a real pool.
"""

import concurrent.futures as cf
import functools
import subprocess
import sys

import pytest

import navis
from navis import config
from navis.compute import dispatch
from navis.compute.backends import (ParallelBackend, register_backend,
                                    get_backend, list_backends,
                                    available_backends, resolve_backend,
                                    set_parallel_backend, SerialBackend,
                                    ProcessBackend, WrappedExecutorBackend,
                                    auto_chunksize)
from navis.compute.backends.base import _BACKENDS


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def double(x):
    """Module level so it is picklable by reference."""
    return x * 2


def boom(x):
    if x == 2:
        raise ValueError('boom')
    return x


class DummyBackend(ParallelBackend):
    """Records what it was asked to do; runs everything inline."""

    name = 'dummy'
    priority = 999
    auto_select = False

    def __init__(self, *, isolated=True, reverse=False, marshals=True):
        self.isolated = isolated
        self.reverse = reverse
        self.marshals_exceptions = marshals
        self.calls = []

    def map(self, func, payloads, *, n_workers):
        self.calls.append({'payloads': list(payloads), 'n_workers': n_workers})
        results = [func(p) for p in payloads]
        # Completion order is explicitly not input order
        yield from (reversed(results) if self.reverse else results)


@pytest.fixture
def registry():
    """Snapshot and restore the registry + config around a test."""
    before = dict(_BACKENDS)
    backend_before = config.default_parallel_backend
    yield
    _BACKENDS.clear()
    _BACKENDS.update(before)
    config.default_parallel_backend = backend_before


@pytest.fixture
def tasks():
    return [(double, (i,), {}) for i in range(7)]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_builtin_backends_registered():
    for name in ('serial', 'threads', 'processes', 'joblib', 'pathos'):
        assert name in list_backends()
    # serial has no dependencies, so it is always usable
    assert get_backend('serial') in available_backends()


def test_register_and_get(registry):
    be = DummyBackend()
    register_backend(be)
    assert get_backend('dummy') is be
    assert 'dummy' in list_backends()


def test_register_rejects_non_backend(registry):
    with pytest.raises(TypeError):
        register_backend(object())


def test_unknown_backend_names_the_alternatives():
    with pytest.raises(ValueError, match='Unknown parallel backend'):
        get_backend('does_not_exist')
    try:
        get_backend('does_not_exist')
    except ValueError as e:
        assert 'serial' in str(e)


def test_auto_picks_highest_priority(registry):
    be = DummyBackend()
    be.auto_select = True
    register_backend(be)
    # priority 999 beats everything shipped
    assert resolve_backend('auto').name == 'dummy'


def test_auto_skips_explicit_only_backends():
    """`serial`/`threads` must never be chosen implicitly."""
    picked = resolve_backend('auto')
    assert picked.name not in ('serial', 'threads')


def test_auto_preference_order():
    """joblib > pathos > processes.

    joblib and pathos can both ship lambdas; joblib is preferred because it
    keeps its workers alive between calls where pathos rebuilds the pool every
    time. `processes` needs no dependency at all but cannot ship lambdas, so it
    is the fallback.
    """
    order = [b.name for b in
             sorted(available_backends(), key=lambda b: -b.priority)
             if b.auto_select]
    # Only assert over the optional ones that are actually installed here.
    # `processes` needs nothing, so this is never empty.
    expected = [n for n in ('joblib', 'pathos', 'processes') if n in order]
    assert order == expected

    # ... and that really is what resolution hands back
    assert resolve_backend('auto').name == expected[0]


def test_auto_respects_by_value_requirement(registry):
    """A backend that can't ship lambdas is skipped when one is needed."""
    # Leave only backends that serialise by reference
    _BACKENDS.clear()
    register_backend(SerialBackend())
    register_backend(ProcessBackend())

    # `processes` can't serialise by value, and it's the only auto-selectable
    # backend left -> resolution must fail rather than silently mis-run
    with pytest.raises(ValueError) as exc:
        resolve_backend('auto', by_value=True)
    assert 'importable by name' in str(exc.value)
    assert 'parallel=False' in str(exc.value)


def test_named_backend_rejects_unsupported_request():
    with pytest.raises(ValueError, match='importable by name'):
        resolve_backend('processes', by_value=True)


def test_named_unavailable_backend_errors_clearly(registry, monkeypatch):
    be = DummyBackend()
    monkeypatch.setattr(be, 'available', lambda: False)
    register_backend(be)
    with pytest.raises(ValueError, match='not available'):
        resolve_backend('dummy')


@pytest.mark.parametrize('kwargs', [
    {'parallel': False},
    {'n_tasks': 1},
    {'n_workers': 1},
])
def test_degrades_to_serial(kwargs):
    """Nothing to parallelise -> don't pay to start workers.

    Also load-bearing for correctness: the caller keys the `inplace`
    optimisation off `isolated`, and an inline run is not isolated.
    """
    be = resolve_backend('auto', **kwargs)
    assert be.name == 'serial'
    assert be.isolated is False


def test_resolve_accepts_instance_and_executor():
    be = DummyBackend()
    assert resolve_backend(be) is be

    with cf.ThreadPoolExecutor(2) as ex:
        wrapped = resolve_backend(ex)
        assert wrapped.pickles_by_value is True
        # A thread pool shares memory - must not be treated as isolated
        assert wrapped.isolated is False


def test_resolve_falls_back_to_config(registry):
    config.default_parallel_backend = 'threads'
    assert resolve_backend(None).name == 'threads'


def test_cluster_backends_are_never_auto_selected():
    """Spinning up a cluster is a decision, not a default."""
    for name in ('dask', 'submitit'):
        assert get_backend(name).auto_select is False
    assert resolve_backend('auto').name not in ('dask', 'submitit')


def test_unrecognised_object_says_what_was_expected():
    for call in (resolve_backend, set_parallel_backend):
        with pytest.raises(TypeError, match='Cannot run navis on a'):
            call(object())


def test_adopt_is_offered_every_object(registry):
    """A backend can claim something that is not a `cf.Executor`."""
    seen = []

    class Claimable:
        pass

    Claimable.__module__ = 'somelib.executors'

    class Claiming(DummyBackend):
        name = 'claiming'
        adopts = ('somelib',)

        def _adopt(self, obj, **overrides):
            seen.append(obj)
            return self

    register_backend(Claiming())
    obj = Claimable()
    assert resolve_backend(obj) is get_backend('claiming')
    assert seen and seen[-1] is obj


def test_adopt_never_looks_at_objects_from_other_modules(registry):
    """`_adopt` imports the library, so it must not run speculatively.

    Otherwise handing navis any object at all would try to import every
    registered backend's dependency.
    """
    class Claiming(DummyBackend):
        name = 'claiming'
        adopts = ('a_library_that_is_not_installed',)

        def _adopt(self, obj, **overrides):
            raise AssertionError('_adopt must not be reached')

    register_backend(Claiming())
    with pytest.raises(TypeError, match='Cannot run navis on a'):
        resolve_backend(object())


def test_capability_overrides_need_an_executor():
    """They describe an executor - silently ignoring them would be worse."""
    with pytest.raises(TypeError, match='describe an executor'):
        set_parallel_backend('threads', isolated=True)

    with cf.ThreadPoolExecutor(1) as ex:
        with pytest.raises(TypeError, match='Unknown capability'):
            WrappedExecutorBackend(ex, islolated=True)


# --------------------------------------------------------------------------- #
# set_parallel_backend
# --------------------------------------------------------------------------- #
def test_set_parallel_backend_validates(registry):
    with pytest.raises(ValueError):
        set_parallel_backend('not_a_backend')


def test_set_parallel_backend_global_and_scoped(registry):
    set_parallel_backend('threads')
    assert config.default_parallel_backend == 'threads'

    with set_parallel_backend('serial'):
        assert config.default_parallel_backend == 'serial'
    assert config.default_parallel_backend == 'threads'


def test_set_parallel_backend_with_executor(registry):
    with cf.ThreadPoolExecutor(2) as ex:
        with set_parallel_backend(ex, isolated=True):
            be = resolve_backend(None)
            assert be.isolated is True
            assert be.executor is ex


# --------------------------------------------------------------------------- #
# Dispatch seam
# --------------------------------------------------------------------------- #
def test_chunking_splits_as_requested(tasks):
    be = DummyBackend()
    dispatch.map_tasks(tasks, backend=be, n_workers=2, chunksize=3)

    payloads = be.calls[0]['payloads']
    assert [len(p.tasks) for p in payloads] == [3, 3, 1]


def test_default_chunksize_is_one_task(tasks):
    be = DummyBackend()
    dispatch.map_tasks(tasks, backend=be, n_workers=2)
    assert [len(p.tasks) for p in be.calls[0]['payloads']] == [1] * 7


# --------------------------------------------------------------------------- #
# Chunking policy
# --------------------------------------------------------------------------- #
def test_shipped_backends_send_one_task_per_unit():
    """Bundling is for remote transports - a local pool must not pay for it."""
    for name in ('serial', 'threads', 'processes', 'joblib', 'pathos'):
        assert get_backend(name).chunksize(10_000, 8) == 1


def test_requested_chunksize_always_wins():
    be = DummyBackend()
    be.chunks_per_worker = 8
    assert be.chunksize(10_000, 20, requested=3) == 3
    # ... but never a nonsensical one
    assert be.chunksize(10_000, 20, requested=0) == 1


def test_auto_chunksize_bounds_the_number_of_units():
    """The count bound is the one that normally binds.

    10,000 tasks over 20 workers at 8 units each is 160 units of 63 - the
    smallest size that keeps the unit count at or under the target.
    """
    assert auto_chunksize(10_000, 20, chunks_per_worker=8) == 63


def test_auto_chunksize_caps_payload_size():
    """With enough tasks the byte bound takes over from the count bound."""
    mb = 1024 ** 2
    kwargs = dict(chunks_per_worker=8, max_bytes=128 * mb)
    # ~309 KB per neuron -> at most 434 per unit
    big = auto_chunksize(10_000_000, 20, size_hint=lambda: 309_000, **kwargs)
    assert big == 434
    # Without the hint the count bound alone would have allowed far more
    assert auto_chunksize(10_000_000, 20, **kwargs) > big


def test_auto_chunksize_stays_in_range():
    # Never zero, never more tasks than exist
    assert auto_chunksize(1, 20, chunks_per_worker=8) == 1
    assert auto_chunksize(5, 20, chunks_per_worker=8) == 1
    assert auto_chunksize(3, 1, chunks_per_worker=1) == 3
    # `n_workers=None` reaches here from an executor that sets its own
    assert auto_chunksize(100, None, chunks_per_worker=8) == 13


def test_size_hint_is_not_called_when_it_cannot_matter():
    """Sampling the data costs real time on the neuron path."""
    calls = []

    def hint():
        calls.append(1)
        return 1.0

    # Count bound already says 1 task per unit - nothing left to shrink
    auto_chunksize(5, 20, chunks_per_worker=8, max_bytes=10, size_hint=hint)
    assert not calls


def test_broken_size_hint_falls_back_to_task_count():
    def hint():
        raise RuntimeError('no idea how big these are')

    assert auto_chunksize(10_000, 20, chunks_per_worker=8, max_bytes=1,
                          size_hint=hint) == 63


def test_chunking_policy_reaches_the_backend(tasks):
    be = DummyBackend()
    be.chunks_per_worker = 2
    dispatch.map_tasks(tasks, backend=be, n_workers=2)
    # 7 tasks, 2 workers, 2 units each -> 4 units -> 2 tasks per unit
    assert [len(p.tasks) for p in be.calls[0]['payloads']] == [2, 2, 2, 1]


def test_local_executor_is_not_bundled():
    with cf.ThreadPoolExecutor(2) as ex:
        assert WrappedExecutorBackend(ex).chunks_per_worker is None
        assert WrappedExecutorBackend(ex).chunksize(10_000, 8) == 1
        # Explicit beats inferred
        assert WrappedExecutorBackend(ex, chunks_per_worker=4).chunksize(
            10_000, 8) > 1


def test_worker_count_is_the_seam_for_remote_backends(registry):
    """A cluster backend corrects the worker count without restating policy."""
    class Cluster(DummyBackend):
        name = 'cluster'
        chunks_per_worker = 2

        def worker_count(self, hint):
            return 100

    be = Cluster()
    # 100 workers x 2 units, not the 2 workers the caller thinks it has
    assert be.chunksize(10_000, n_workers=2) == 50
    # ... but an explicit request is still the last word
    assert be.chunksize(10_000, n_workers=2, requested=7) == 7


def test_tiny_workloads_never_ask_the_cluster_how_big_it_is(registry):
    """Counting remote workers is a round trip; skip it when it can't matter."""
    asked = []

    class Cluster(DummyBackend):
        name = 'cluster'
        chunks_per_worker = 8

        def worker_count(self, hint):
            asked.append(1)
            return 100

    assert Cluster().chunksize(4, n_workers=2) == 1
    assert not asked


# --------------------------------------------------------------------------- #
# Dispatch seam (continued)
# --------------------------------------------------------------------------- #
def test_results_are_reordered(tasks):
    """The backend yields backwards; the caller must still get input order."""
    be = DummyBackend(reverse=True)
    res = dispatch.map_tasks(tasks, backend=be, n_workers=2, chunksize=2)
    assert res == [i * 2 for i in range(7)]


def test_n_workers_reaches_the_backend(tasks):
    be = DummyBackend()
    dispatch.map_tasks(tasks, backend=be, n_workers=5)
    assert be.calls[0]['n_workers'] == 5


def test_worker_context_only_when_isolated(tasks):
    isolated = DummyBackend(isolated=True)
    dispatch.map_tasks(tasks, backend=isolated, n_workers=2)
    assert isinstance(isolated.calls[0]['payloads'][0].context,
                      dispatch.WorkerContext)

    shared = DummyBackend(isolated=False)
    dispatch.map_tasks(tasks, backend=shared, n_workers=2)
    assert shared.calls[0]['payloads'][0].context is None


def test_worker_context_roundtrips_config():
    """A spawned worker re-imports navis; the context is what carries state."""
    before = config.pbar_hide
    try:
        config.pbar_hide = True
        ctx = dispatch.WorkerContext.snapshot()
        config.pbar_hide = False
        ctx.apply()
        assert config.pbar_hide is True
    finally:
        config.pbar_hide = before


def test_progress_total_counts_tasks_not_chunks(tasks, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, *args, **kwargs):
            seen['total'] = kwargs.get('total')
            seen['updates'] = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, n):
            seen['updates'].append(n)

    monkeypatch.setattr(config, 'tqdm', Recorder)
    dispatch.map_tasks(tasks, backend=DummyBackend(), n_workers=2, chunksize=3)

    assert seen['total'] == len(tasks)
    assert sum(seen['updates']) == len(tasks)


def test_empty_tasks_short_circuits():
    be = DummyBackend()
    assert dispatch.map_tasks([], backend=be, n_workers=2) == []
    assert not be.calls


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
def test_omit_failures_returns_failedrun():
    tasks = [(boom, (i,), {}) for i in range(4)]
    res = dispatch.map_tasks(tasks, backend=DummyBackend(), n_workers=2,
                             omit_failures=True)

    assert [type(r).__name__ for r in res] == ['int', 'int', 'FailedRun', 'int']
    failed = res[2]
    assert isinstance(failed.exception, ValueError)
    # The parent rebuilds these, so args/kwargs must survive
    assert failed.args == (2,)
    assert failed.func is boom


def test_failures_propagate_by_default():
    tasks = [(boom, (i,), {}) for i in range(4)]
    with pytest.raises(ValueError, match='boom'):
        dispatch.map_tasks(tasks, backend=DummyBackend(), n_workers=2)


def test_marshalling_backend_lets_the_worker_raise(tasks):
    """The normal case: the transport carries the exception home itself."""
    be = DummyBackend()
    dispatch.map_tasks(tasks, backend=be, n_workers=2)
    payload = be.calls[0]['payloads'][0]
    assert payload.catch is False
    assert payload.want_traceback is False


def test_unmarshalled_exceptions_still_arrive_as_themselves():
    """A transport that mangles exceptions must not change what callers catch.

    submitit records a failure as a traceback *string* and raises its own
    `FailedJobError` regardless of what went wrong. Without this, the exception
    you catch would depend on which backend happened to be configured.
    """
    be = DummyBackend(marshals=False)
    failing = [(boom, (i,), {}) for i in range(4)]

    with pytest.raises(ValueError, match='boom'):
        dispatch.map_tasks(failing, backend=be, n_workers=2)

    # The worker was asked to hand failures back rather than raise them, and
    # to pay for a traceback because the parent is going to re-raise
    payload = be.calls[0]['payloads'][0]
    assert (payload.catch, payload.want_traceback) == (True, True)


def test_remote_traceback_is_chained():
    """The worker's frames must survive - the exception arrives without them."""
    be = DummyBackend(marshals=False)
    with pytest.raises(ValueError) as exc:
        dispatch.map_tasks([(boom, (2,), {})], backend=be, n_workers=1)

    cause = exc.value.__cause__
    assert isinstance(cause, dispatch.RemoteTraceback)
    assert 'in boom' in str(cause)


def test_unmarshalled_failures_still_honour_omit_failures():
    be = DummyBackend(marshals=False)
    failing = [(boom, (i,), {}) for i in range(4)]
    res = dispatch.map_tasks(failing, backend=be, n_workers=2,
                             omit_failures=True)
    assert [type(r).__name__ for r in res] == ['int', 'int', 'FailedRun', 'int']
    assert isinstance(res[2].exception, ValueError)


@pytest.mark.parametrize('marshals', [True, False])
def test_omitted_failures_do_not_pay_for_tracebacks(marshals):
    """Nothing reads them - `FailedRun` only carries the exception.

    Formatting one costs ~0.1 ms and a couple of kB *per failed task*, shipped
    back from the worker, so a run with many failures would pay a lot for
    strings that are dropped on arrival.
    """
    be = DummyBackend(marshals=marshals)
    failing = [(boom, (2,), {})]
    res = dispatch.map_tasks(failing, backend=be, n_workers=1,
                             omit_failures=True)

    assert be.calls[0]['payloads'][0].want_traceback is False
    assert isinstance(res[0].exception, ValueError)


def test_user_typeerror_is_not_blamed_on_serialisation():
    """A TypeError from the user's function must not be reported as pickling."""
    def bad(x):
        raise TypeError('this is my own bug')

    tasks = [(bad, (1,), {})]
    with pytest.raises(TypeError, match='my own bug'):
        dispatch.map_tasks(tasks, backend=DummyBackend(), n_workers=1)


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def test_picklable_by_reference():
    assert dispatch.picklable_by_reference(double)
    assert not dispatch.picklable_by_reference(lambda x: x)

    def closure(x):
        return x
    assert not dispatch.picklable_by_reference(closure)

    # decorated navis functions must qualify - that's what makes plain-pickle
    # backends usable at all
    assert dispatch.picklable_by_reference(navis.prune_twigs)

    # bound methods are fine: they pickle as (object, name)
    n = navis.example_neurons(1)
    assert dispatch.picklable_by_reference(n.resample)


def test_picklable_by_reference_unwraps_partial():
    """A partial travels exactly as far as the function it wraps."""
    assert dispatch.picklable_by_reference(functools.partial(double, 1))
    assert not dispatch.picklable_by_reference(functools.partial(lambda x: x, 1))


def test_picklable_by_reference_honours_the_hook():
    """A callable object can answer for itself - `navis.Pipeline` does.

    Without the hook these would both be False: the name check rejects any
    instance, because `__qualname__` lives on the class, not on it.
    """
    class Yes:
        __picklable_by_reference__ = True

    class No:
        __picklable_by_reference__ = False

    assert dispatch.picklable_by_reference(Yes())
    assert not dispatch.picklable_by_reference(No())


# --------------------------------------------------------------------------- #
# Real backends
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('name', ['serial', 'threads', 'processes'])
def test_real_backends_agree(name, tasks):
    be = get_backend(name)
    res = dispatch.map_tasks(tasks, backend=be, n_workers=2, disable=True)
    assert res == [i * 2 for i in range(7)]


@pytest.mark.parametrize('name', ['pathos', 'joblib'])
def test_optional_backends_agree(name, tasks):
    be = get_backend(name)
    if not be.available():
        pytest.skip(f'{name} not installed')
    res = dispatch.map_tasks(tasks, backend=be, n_workers=2, disable=True)
    assert res == [i * 2 for i in range(7)]


#: Runs a task that builds a progress bar in the worker - which is any navis
#: function worth parallelising. Has to be a separate interpreter: the warning
#: we are looking for is printed by a resource tracker *the worker* spawned, so
#: there is nothing in this process to observe.
_PBAR_IN_WORKER = '''
import navis
from navis import config
from navis.compute import dispatch
from navis.compute.backends import get_backend

def bar(x):
    with config.tqdm(total=1, disable=True):
        return x

if __name__ == '__main__':
    dispatch.map_tasks([(bar, (i,), {}) for i in range(2)],
                       backend=get_backend('pathos'), n_workers=2, disable=True)
'''


def test_pathos_workers_do_not_leak_semaphores(tmp_path):
    """tqdm's cross-process lock is a semaphore no pathos worker cleans up.

    Its workers come from `multiprocess`, which carries its own resource
    tracker, so the stdlib one has never been started there and each worker
    spawns a fresh one - which then reports the lock as leaked on the way out.
    Four workers, four scary warnings, for a bar that was disabled anyway.
    """
    if not get_backend('pathos').available():
        pytest.skip('pathos not installed')

    script = tmp_path / 'pbar_in_worker.py'
    script.write_text(_PBAR_IN_WORKER)
    res = subprocess.run([sys.executable, str(script)], timeout=300,
                         capture_output=True, text=True)

    assert res.returncode == 0, res.stderr
    assert 'leaked semaphore' not in res.stderr


def test_lambda_on_plain_pickle_backend_explains_itself():
    """A lambda over `processes` must name the fix, not raise PicklingError."""
    tasks = [(lambda x: x, (1,), {})]
    with pytest.raises(RuntimeError) as exc:
        dispatch.map_tasks(tasks, backend=get_backend('processes'),
                           n_workers=2, disable=True)
    assert 'set_parallel_backend' in str(exc.value)
    assert 'parallel=False' in str(exc.value)


def _live_pool(name):
    """The executor a backend is currently holding open, if any."""
    if name == 'processes':
        from navis.compute.backends import local
        return local._POOL
    from joblib.externals.loky import reusable_executor
    return reusable_executor._executor


def test_shutdown_releases_the_pools_backends_keep_alive():
    """Keeping workers warm is only acceptable if you can get them back.

    Each worker is a full navis import (~300 MB), so `compute.shutdown()` -
    which also runs at interpreter exit - has to reach every backend that holds
    a pool open between calls. `joblib` used to be missed, i.e. the default one.
    """
    pools = []
    for name in ('processes', 'joblib'):
        be = get_backend(name)
        if not be.available():
            continue
        dispatch.map_tasks([(double, (1,), {})], backend=be, n_workers=2,
                           disable=True)
        pools.append((name, _live_pool(name)))

    assert pools, 'expected at least the stdlib process pool'
    for name, pool in pools:
        assert pool is not None, f"'{name}' did not keep a pool alive"

    navis.compute.shutdown()

    for _, pool in pools:
        with pytest.raises(RuntimeError):
            pool.submit(len, 'ab')

    # ... and a torn-down backend still works afterwards. One is enough: this
    # costs a fresh pool (~2s of navis imports per worker) to check that a
    # shut-down executor is replaced rather than reused.
    name = pools[0][0]
    assert dispatch.map_tasks([(double, (2,), {})], backend=get_backend(name),
                              n_workers=2, disable=True) == [4]
    navis.compute.shutdown()

