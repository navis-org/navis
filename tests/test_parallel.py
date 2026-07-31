import navis

import numpy as np
import pandas as pd
import pytest


#: Backends that need no optional dependency and so always run here. `pathos`
#: and `joblib` are added via `importorskip` in the parity test below.
LOCAL_BACKENDS = ['serial', 'threads', 'processes']


def test_parallel():
    # Load example neurons
    nl = navis.example_neurons(kind='skeleton')

    # Test decorator
    pr = navis.prune_by_strahler(nl, 1, parallel=True, inplace=False)

    assert isinstance(pr, navis.NeuronList)
    assert len(pr) == len(nl)
    assert pr[0].n_nodes < nl[0].n_nodes

    # Test apply
    pr = nl.apply(navis.prune_by_strahler, to_prune=1, inplace=False,
                  parallel=True)
    assert isinstance(pr, navis.NeuronList)
    assert len(pr) == len(nl)
    assert all(pr.n_nodes < nl.n_nodes)


def test_parallel_inplace():
    # Load example neurons
    nl = navis.example_neurons(kind='skeleton')

    # Test decorator with inplace=True
    pr = nl.copy()
    pr2 = navis.prune_by_strahler(pr, 1, parallel=True, inplace=True)
    assert len(pr) == len(pr2) == len(nl)
    assert pr[0].n_nodes == pr2[0].n_nodes

    # Test apply with inplace=True -> this should not work
    #pr = nl.apply(navis.prune_by_strahler, to_prune=1, inplace=True,
    #              parallel=True)
    #assert len(pr) == len(nl)
    #assert all(pr.n_nodes < nl.n_nodes)


def test_apply():
    # Load example neurons
    nl = navis.example_neurons(kind='skeleton')

    # Test apply
    ids = nl.apply(lambda x: x.id, parallel=False)
    assert isinstance(ids, list)
    assert len(ids) == len(nl)
    assert all(np.array(ids) == nl.id)

    # Test apply
    pr = nl.apply(navis.prune_by_strahler, to_prune=1, inplace=False,
                  parallel=False)
    assert isinstance(pr, navis.NeuronList)
    assert len(pr) == len(nl)
    assert all(pr.n_nodes < nl.n_nodes)

    # Test apply with inplace=True
    pr = nl.apply(navis.prune_by_strahler, to_prune=1, inplace=True,
                  parallel=False)
    assert isinstance(pr, navis.NeuronList)
    assert len(pr) == len(nl)
    assert all(pr.n_nodes == nl.n_nodes)


# --------------------------------------------------------------------------- #
# Parity across backends
# --------------------------------------------------------------------------- #
@pytest.fixture
def nl():
    return navis.NeuronList([n.copy() for n in navis.example_neurons(4)])


@pytest.fixture
def backend(request):
    name = request.param
    if name in ('pathos', 'joblib'):
        pytest.importorskip(name)
    return name


@pytest.mark.parametrize('backend', LOCAL_BACKENDS + ['pathos', 'joblib'],
                         indirect=True)
def test_backend_parity(nl, backend):
    """Every backend must produce exactly the serial result."""
    ref = [n.n_nodes for n in navis.prune_twigs(nl, 5000, inplace=False)]

    got = navis.prune_twigs(nl.copy(), 5000, inplace=False, parallel=True,
                            backend=backend)
    assert isinstance(got, navis.NeuronList)
    assert [n.n_nodes for n in got] == ref


def test_bare_executor_backend(nl):
    """A user-supplied Executor - the path a cluster client takes.

    Note the executor itself must never end up inside the payload we ship to
    the workers: it holds a thread lock and is not picklable.
    """
    import concurrent.futures as cf

    ref = [n.n_nodes for n in navis.prune_twigs(nl, 5000, inplace=False)]

    with cf.ProcessPoolExecutor(2) as ex:
        with navis.set_parallel_backend(ex):
            got = navis.prune_twigs(nl.copy(), 5000, inplace=False, parallel=True)

    assert [n.n_nodes for n in got] == ref


@pytest.mark.parametrize('backend', LOCAL_BACKENDS, indirect=True)
def test_backend_parity_must_zip(nl, backend):
    """`must_zip` arguments must still be distributed per neuron."""
    source = [n.root[0] for n in nl]
    ref = [n.n_nodes for n in navis.prune_at_depth(nl, 50000, source=source,
                                                   inplace=False)]

    got = navis.prune_at_depth(nl.copy(), 50000, source=source, inplace=False,
                               parallel=True, backend=backend)
    assert [n.n_nodes for n in got] == ref


@pytest.mark.parametrize('backend', LOCAL_BACKENDS, indirect=True)
def test_backend_parity_dataframe(nl, backend):
    """`map_neuronlist_df` functions under parallel."""
    ref = navis.branch_angles(nl)
    got = navis.branch_angles(nl.copy(), parallel=True, backend=backend)

    assert isinstance(got, pd.DataFrame)
    assert list(got.neuron) == list(ref.neuron)
    assert np.allclose(got.branch_angle, ref.branch_angle, equal_nan=True)


@pytest.mark.parametrize('backend', LOCAL_BACKENDS, indirect=True)
def test_n_cores_and_chunksize(nl, backend):
    """Neither knob may change the answer."""
    ref = [n.n_nodes for n in navis.prune_twigs(nl, 5000, inplace=False)]
    for kwargs in ({'n_cores': 2}, {'chunksize': 2}, {'n_cores': 2, 'chunksize': 3}):
        got = navis.prune_twigs(nl.copy(), 5000, inplace=False, parallel=True,
                                backend=backend, **kwargs)
        assert [n.n_nodes for n in got] == ref, kwargs


# --------------------------------------------------------------------------- #
# Regressions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('backend', ['threads', 'serial'], indirect=True)
def test_inplace_false_is_honoured_on_shared_memory(nl, backend):
    """`parallel=True` must not mutate the caller on an in-process backend.

    Under a process pool the neurons are copied, so the decorator turns
    `inplace=False` into an in-place run to save a copy. Threads share memory
    with us, so doing that there would silently prune the caller's neurons.
    """
    before = [n.n_nodes for n in nl]

    navis.prune_twigs(nl, 5000, inplace=False, parallel=True, backend=backend)

    assert [n.n_nodes for n in nl] == before


def test_inplace_false_is_honoured_on_a_one_core_machine(nl):
    """The decorator and the processor must agree on the backend.

    The decorator resolves one to decide whether it may force `inplace=True`;
    the processor resolves again to run. If they see different worker counts
    they can pick differently - decorator says "isolated, so mutating is free",
    processor degrades to serial and mutates the caller's own neurons. One
    worker is exactly that case, and is what a 2-core CI runner defaults to.
    """
    before = [n.n_nodes for n in nl]

    with navis.set_parallel_backend(n_workers=1):
        navis.prune_twigs(nl, 5000, inplace=False, parallel=True)

    assert [n.n_nodes for n in nl] == before


def test_single_neuron_does_not_mutate(nl):
    """A 1-neuron list degrades to serial - which must not force inplace."""
    single = navis.NeuronList([nl[0]])
    before = single[0].n_nodes

    navis.prune_twigs(single, 5000, inplace=False, parallel=True)

    assert single[0].n_nodes == before


@pytest.mark.parametrize('backend', LOCAL_BACKENDS, indirect=True)
def test_omit_failures_under_parallel(backend):
    """Failures are dropped, counted and reported - not raised."""
    sk = navis.example_neurons(3, kind='skeleton')
    # A Dotprops has no nodes, so `prune_twigs` fails on it
    mixed = navis.NeuronList([sk[0], navis.make_dotprops(sk[1], k=5), sk[2]])

    res = navis.prune_twigs(mixed, 5000, inplace=False, parallel=True,
                            backend=backend, omit_failures=True)

    assert len(res) == 2
    assert list(res.id) == [sk[0].id, sk[2].id]


@pytest.mark.parametrize('backend', LOCAL_BACKENDS, indirect=True)
def test_failures_propagate_by_default(backend):
    sk = navis.example_neurons(2, kind='skeleton')
    mixed = navis.NeuronList([sk[0], navis.make_dotprops(sk[1], k=5)])

    with pytest.raises(BaseException):
        navis.prune_twigs(mixed, 5000, inplace=False, parallel=True,
                          backend=backend)


def test_logger_level_restored_after_failure():
    """An exception during dispatch must not leave the logger silenced."""
    import logging

    sk = navis.example_neurons(2, kind='skeleton')
    mixed = navis.NeuronList([sk[0], navis.make_dotprops(sk[1], k=5)])

    navis.set_loggers('INFO')
    before = logging.getLogger('navis').getEffectiveLevel()
    try:
        with pytest.raises(BaseException):
            navis.prune_twigs(mixed, 5000, inplace=False, parallel=True,
                              backend='serial')
        assert logging.getLogger('navis').getEffectiveLevel() == before
    finally:
        navis.set_loggers('WARNING')


@pytest.mark.parametrize('reserved', ['chunksize', 'progress'])
def test_apply_does_not_forward_reserved_kwargs(nl, reserved):
    """`chunksize`/`progress` are ours - they must not reach the function."""
    def strict(x):
        # Would raise TypeError if navis forwarded the reserved keyword
        return x.n_nodes

    res = nl.apply(strict, **{reserved: 2 if reserved == 'chunksize' else False})
    assert res == [n.n_nodes for n in nl]


def test_apply_lambda_needs_a_by_value_backend(nl):
    """Plain pickle can't ship a lambda - say so, and say what to do."""
    with pytest.raises(BaseException) as exc:
        nl.apply(lambda x: x.id, parallel=True, backend='processes')
    assert 'set_parallel_backend' in str(exc.value)


@pytest.mark.parametrize('backend', ['pathos', 'joblib'], indirect=True)
def test_apply_lambda_works_on_by_value_backends(nl, backend):
    res = nl.apply(lambda x: x.id, parallel=True, backend=backend)
    assert list(res) == list(nl.id)
