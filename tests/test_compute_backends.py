"""Tests for the pluggable parallel backends in `navis.compute`.

Two halves: the registry (which backend gets picked, and why) and the dispatch
seam (chunking, ordering, failure handling). The seam is exercised through a
recording dummy backend so the interesting properties - notably that results
are reordered correctly when they come back out of order - can be tested
deterministically, which is impossible with a real pool.
"""

import concurrent.futures as cf

import numpy as np
import pytest

import navis
from navis import config
from navis.compute import dispatch
from navis.compute.backends import (ParallelBackend, register_backend,
                                    get_backend, list_backends,
                                    available_backends, resolve_backend,
                                    set_parallel_backend, SerialBackend,
                                    ProcessBackend)
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

    def __init__(self, *, isolated=True, reverse=False):
        self.isolated = isolated
        self.reverse = reverse
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
    assert [len(p[3]) for p in payloads] == [3, 3, 1]


def test_default_chunksize_is_one_task(tasks):
    be = DummyBackend()
    dispatch.map_tasks(tasks, backend=be, n_workers=2)
    assert [len(p[3]) for p in be.calls[0]['payloads']] == [1] * 7


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
    assert isinstance(isolated.calls[0]['payloads'][0][1], dispatch.WorkerContext)

    shared = DummyBackend(isolated=False)
    dispatch.map_tasks(tasks, backend=shared, n_workers=2)
    assert shared.calls[0]['payloads'][0][1] is None


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


def test_lambda_on_plain_pickle_backend_explains_itself():
    """A lambda over `processes` must name the fix, not raise PicklingError."""
    tasks = [(lambda x: x, (1,), {})]
    with pytest.raises(RuntimeError) as exc:
        dispatch.map_tasks(tasks, backend=get_backend('processes'),
                           n_workers=2, disable=True)
    assert 'set_parallel_backend' in str(exc.value)
    assert 'parallel=False' in str(exc.value)
