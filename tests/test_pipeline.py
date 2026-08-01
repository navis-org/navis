import pickle

import numpy as np
import pytest

import navis
from navis.compute.dispatch import picklable_by_reference

from .test_compute_backends import DummyBackend, registry  # noqa: F401


#: Backends that need no optional dependency and so always run here. `pathos`
#: and `joblib` are added via `importorskip` in the parity tests below.
LOCAL_BACKENDS = ['serial', 'threads', 'processes']

#: What the pipeline is meant to save you: three stages that would otherwise be
#: three round-trips through the workers.
STEPS = [navis.heal_skeleton,
         (navis.prune_twigs, (5000,)),
         (navis.resample_skeleton, {'resample_to': 1000})]


# --------------------------------------------------------------------------- #
# Helpers. All module level, so they are picklable by reference.
# --------------------------------------------------------------------------- #
def n_nodes(x):
    return x.n_nodes


def boom(x):
    raise ValueError('boom')


def identity(x):
    """Hands its input straight back - i.e. grants no ownership."""
    return x


def gives_none(x):
    return None


#: `record_inplace` appends to this. Only meaningful in-process (a worker gets
#: its own copy of the module), which is exactly where we test the elision.
INPLACE_LOG = []


def record_inplace(x, inplace=False):
    """Records the `inplace` it was handed, then acts like a navis function."""
    INPLACE_LOG.append((x.name, inplace))
    return x if inplace else x.copy()


def make_neurons(n):
    """A step that turns something that isn't a neuron into neurons."""
    return navis.example_neurons(n, kind='skeleton')


@pytest.fixture
def nl():
    return navis.NeuronList([n.copy() for n in navis.example_neurons(4)])


@pytest.fixture
def named_nl():
    """Neurons with distinguishable names, for the per-neuron `inplace` log."""
    nl = navis.NeuronList([n.copy() for n in navis.example_neurons(2)])
    for i, n in enumerate(nl):
        n.name = f'n{i}'
    INPLACE_LOG.clear()
    return nl


@pytest.fixture
def backend(request):
    name = request.param
    if name in ('pathos', 'joblib'):
        pytest.importorskip(name)
    return name


def inplace_per_neuron():
    """Group `INPLACE_LOG` by neuron - workers interleave."""
    out = {}
    for name, inplace in INPLACE_LOG:
        out.setdefault(name, []).append(inplace)
    return out


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_step_specs():
    """All the ways of spelling a step must agree."""
    from functools import partial

    variants = [
        navis.Pipeline(navis.heal_skeleton, (navis.prune_twigs, (5000,))),
        navis.Pipeline(navis.heal_skeleton, (navis.prune_twigs, {'size': 5000})),
        navis.Pipeline(navis.heal_skeleton, (navis.prune_twigs, (5000,), {})),
        navis.Pipeline(navis.heal_skeleton).add(navis.prune_twigs, 5000),
        navis.Pipeline(navis.heal_skeleton) | (navis.prune_twigs, (5000,)),
        navis.Pipeline().heal_skeleton().prune_twigs(5000),
        # Note the keyword: a partial's positional arguments come *before* the
        # neuron, which is not what you want here.
        navis.Pipeline(navis.heal_skeleton, partial(navis.prune_twigs, size=5000)),
    ]

    nl = navis.example_neurons(2)
    ref = [n.n_nodes for n in variants[0](nl)]
    for pipe in variants:
        assert len(pipe) == 2
        assert [n.n_nodes for n in pipe(nl)] == ref


def test_container_arguments_survive_both_spellings():
    """A list argument must not be turned into an array by the tuple form."""
    subset = [0, 1, 2]

    from_tuple = navis.Pipeline((navis.subset_neuron, (subset,)))
    from_add = navis.Pipeline().add(navis.subset_neuron, subset)

    assert from_tuple[0].args == from_add[0].args == (subset,)


def test_pipelines_are_immutable():
    pipe = navis.Pipeline(navis.heal_skeleton)
    longer = pipe.add(navis.prune_twigs, 5000)

    assert len(pipe) == 1
    assert len(longer) == 2
    assert longer is not pipe


def test_nested_pipelines_are_flattened():
    inner = navis.Pipeline(navis.heal_skeleton, navis.prune_twigs)

    assert len(navis.Pipeline(inner, navis.resample_skeleton)) == 3
    assert len(navis.Pipeline(navis.resample_skeleton).add(inner)) == 3
    assert len(inner | inner) == 4


def test_bad_arguments_are_caught_at_construction():
    """A typo must not survive until we are five worker processes deep."""
    with pytest.raises(TypeError, match='resample_skeleton'):
        navis.Pipeline((navis.resample_skeleton, {'not_a_param': 1}))

    with pytest.raises(TypeError, match='callable'):
        navis.Pipeline(42)


def test_varkwargs_signature_is_accepted():
    """`skeletonize(x, **kwargs)` can't be checked, but must still build.

    It is also why `takes_inplace` looks at the parameter kind: an `inplace` we
    injected would just be forwarded to whatever the function wraps.
    """
    pipe = navis.Pipeline(navis.skeletonize)
    assert len(pipe) == 1
    assert pipe[0].takes_inplace is False


def test_repr_len_and_iter():
    pipe = navis.Pipeline(*STEPS)
    assert repr(pipe) == ('Pipeline with 3 step(s): heal_skeleton | '
                          'prune_twigs | resample_skeleton')
    assert len(pipe) == len(list(pipe)) == 3
    assert pipe[0].name == 'heal_skeleton'
    assert len(pipe[:2]) == 2
    assert repr(navis.Pipeline()) == 'Pipeline with 0 step(s)'


def test_empty_pipeline_refuses_to_run():
    with pytest.raises(ValueError, match='no steps'):
        navis.Pipeline()(navis.example_neurons(1))


def test_unknown_name_points_at_add():
    with pytest.raises(AttributeError, match='prune_twigs'):
        navis.Pipeline().prune_twiggs(5)
    with pytest.raises(AttributeError, match=r'\.add\('):
        navis.Pipeline().no_such_function_anywhere()


# --------------------------------------------------------------------------- #
# Parity
# --------------------------------------------------------------------------- #
def test_pipeline_matches_sequential_calls(nl):
    ref = navis.resample_skeleton(
        navis.prune_twigs(navis.heal_skeleton(nl), 5000), 1000)

    got = navis.Pipeline(*STEPS)(nl)

    assert isinstance(got, navis.NeuronList)
    assert [n.n_nodes for n in got] == [n.n_nodes for n in ref]


@pytest.mark.parametrize('backend', LOCAL_BACKENDS + ['pathos', 'joblib'],
                         indirect=True)
def test_backend_parity(nl, backend):
    """Every backend must produce exactly the serial result."""
    pipe = navis.Pipeline(*STEPS)
    ref = [n.n_nodes for n in pipe(nl)]

    got = pipe(nl.copy(), parallel=True, backend=backend, n_cores=2)

    assert isinstance(got, navis.NeuronList)
    assert [n.n_nodes for n in got] == ref


@pytest.mark.parametrize('n_cores,chunksize', [(2, 1), (3, 2)])
def test_n_cores_and_chunksize_do_not_change_the_answer(nl, n_cores, chunksize):
    pipe = navis.Pipeline(*STEPS)
    ref = [n.n_nodes for n in pipe(nl)]

    got = pipe(nl.copy(), parallel=True, n_cores=n_cores, chunksize=chunksize)

    assert [n.n_nodes for n in got] == ref


def test_type_changing_steps(nl):
    pipe = navis.Pipeline(navis.make_dotprops).add(navis.subset_neuron,
                                                   subset=np.arange(10))
    res = pipe(nl)

    assert isinstance(res, navis.NeuronList)
    assert all(isinstance(n, navis.Dotprops) for n in res)


def test_one_to_many_tail_is_flattened(nl):
    """A step that returns several neurons per input just adds to the list."""
    pipe = navis.Pipeline(navis.heal_skeleton).add(navis.break_fragments)

    res = pipe(nl)

    assert isinstance(res, navis.NeuronList)
    assert len(res) >= len(nl)


def test_non_neuron_tail_returns_a_list(nl):
    res = navis.Pipeline(navis.heal_skeleton).add(n_nodes)(nl)

    assert isinstance(res, list)
    assert res == [n.n_nodes for n in navis.heal_skeleton(nl)]


def test_all_none_tail_returns_none(nl):
    assert navis.Pipeline(gives_none)(nl) is None


# --------------------------------------------------------------------------- #
# The point of the whole thing: one task per neuron
# --------------------------------------------------------------------------- #
def test_dispatches_once_per_neuron(nl, registry):  # noqa: F811
    """A 3-step pipeline is ONE dispatch of N tasks, not three of N.

    This is the entire reason `Pipeline` exists: each neuron is serialised out
    to a worker and back once for the whole chain instead of once per function.
    """
    be = DummyBackend(isolated=True)

    # n_cores must be > 1 or `resolve_backend` degrades to serial
    navis.Pipeline(*STEPS)(nl, parallel=True, n_cores=2, backend=be)

    assert len(be.calls) == 1
    payloads = be.calls[0]['payloads']
    assert len(payloads) == len(nl)
    assert [len(p.tasks) for p in payloads] == [1] * len(nl)

    func, args, kwargs = payloads[0].tasks[0]
    assert len(func) == 3            # the whole chain, fused into one callable
    assert len(args) == 1            # ... called with a single neuron


def test_chained_calls_dispatch_once_per_step(nl, registry):  # noqa: F811
    """The negative control for the test above."""
    be = DummyBackend(isolated=True)

    x = navis.heal_skeleton(nl, parallel=True, n_cores=2, backend=be)
    x = navis.prune_twigs(x, 5000, parallel=True, n_cores=2, backend=be)
    navis.resample_skeleton(x, 1000, parallel=True, n_cores=2, backend=be)

    assert len(be.calls) == 3


def test_once_step_splits_the_dispatch(nl, registry):  # noqa: F811
    """A whole-list step has to run here, so it breaks the chain in two."""
    be = DummyBackend(isolated=True)

    pipe = (navis.Pipeline(navis.heal_skeleton)
            .add_once(navis.prune_twigs, 5000)
            .add(navis.resample_skeleton, 1000))
    res = pipe(nl, parallel=True, n_cores=2, backend=be)

    assert len(be.calls) == 2
    assert len(res) == len(nl)


# --------------------------------------------------------------------------- #
# Input that isn't neurons
# --------------------------------------------------------------------------- #
def test_first_step_can_take_anything(registry):  # noqa: F811
    """`Pipeline(fetch).resample()` on a query object: fetch once, then fan out."""
    be = DummyBackend(isolated=True)

    pipe = navis.Pipeline(make_neurons).add(navis.resample_skeleton, 1000)
    res = pipe(3, parallel=True, n_cores=2, backend=be)

    assert isinstance(res, navis.NeuronList) and len(res) == 3
    # `make_neurons` ran here; only the per-neuron tail was handed out
    assert len(be.calls) == 1
    assert len(be.calls[0]['payloads']) == 3


def test_add_each_fans_out_over_a_plain_list(registry):  # noqa: F811
    be = DummyBackend(isolated=True)

    pipe = navis.Pipeline().add_each(make_neurons)
    res = pipe([1, 1, 1], parallel=True, n_cores=2, backend=be)

    assert len(res) == 3
    assert len(be.calls) == 1
    assert len(be.calls[0]['payloads']) == 3


def test_add_each_on_something_not_iterable():
    with pytest.raises(TypeError, match='not iterable'):
        navis.Pipeline().add_each(n_nodes)(navis.example_neurons(1))


def test_bare_list_of_neurons_is_wrapped(nl):
    res = navis.Pipeline(n_nodes)(list(nl))

    assert res == [n.n_nodes for n in nl]


def test_single_neuron_passes_through():
    n = navis.example_neurons(1)
    nodes_before = n.n_nodes

    res = navis.Pipeline(*STEPS)(n)

    assert isinstance(res, navis.Skeleton)
    assert n.n_nodes == nodes_before, 'the caller\'s neuron was modified'


def test_generator_input_is_materialised(nl):
    res = navis.Pipeline(n_nodes)(n for n in nl)

    assert res == [n.n_nodes for n in nl]


# --------------------------------------------------------------------------- #
# Ownership / the `inplace` elision
# --------------------------------------------------------------------------- #
def test_first_step_copies_on_shared_memory(named_nl):
    """On threads or inline, step 1 must NOT work in place.

    "In place" there means the caller's own neurons. Step 1 therefore takes the
    one copy that makes every step after it free - the point being that we go
    from N copies to exactly one, not that we skip them all.
    """
    navis.Pipeline(record_inplace, record_inplace, record_inplace)(named_nl)

    assert inplace_per_neuron() == {'n0': [False, True, True],
                                    'n1': [False, True, True]}


@pytest.mark.parametrize('backend', ['serial', 'threads'], indirect=True)
def test_shared_memory_backends_do_not_mutate_the_caller(nl, backend):
    before = [n.n_nodes for n in nl]

    navis.Pipeline((navis.prune_twigs, (5000,)),
                   (navis.prune_twigs, (5000,)))(nl, parallel=True,
                                                 n_cores=2, backend=backend)

    assert [n.n_nodes for n in nl] == before


def test_isolated_backends_elide_every_copy(named_nl, registry):  # noqa: F811
    """A worker holds a private copy, so even step 1 may work in place."""
    be = DummyBackend(isolated=True)

    navis.Pipeline(record_inplace, record_inplace, record_inplace)(
        named_nl, parallel=True, n_cores=2, backend=be)

    assert inplace_per_neuron() == {'n0': [True, True, True],
                                    'n1': [True, True, True]}


def test_a_step_that_returns_its_input_grants_nothing(named_nl):
    """`identity` hands back the caller's neuron, so nothing may be elided."""
    pipe = navis.Pipeline(identity, record_inplace, record_inplace)

    pipe(named_nl)

    assert inplace_per_neuron() == {'n0': [False, True], 'n1': [False, True]}


def test_explicit_inplace_is_honoured(named_nl):
    """We may not override what the caller asked for - that would mutate."""
    pipe = navis.Pipeline(record_inplace,
                          (record_inplace, {'inplace': False}),
                          record_inplace)

    pipe(named_nl)

    assert inplace_per_neuron() == {'n0': [False, False, True],
                                    'n1': [False, False, True]}


def test_elide_copies_can_be_switched_off(named_nl):
    pipe = navis.Pipeline(record_inplace, record_inplace, record_inplace,
                          elide_copies=False)

    pipe(named_nl, parallel=True, n_cores=2, backend=DummyBackend(isolated=True))

    assert inplace_per_neuron() == {'n0': [False] * 3, 'n1': [False] * 3}


def test_one_core_machine_does_not_mutate(nl):
    """`parallel=True` degrades to serial at one worker - it must still copy."""
    before = [n.n_nodes for n in nl]

    with navis.set_parallel_backend(n_workers=1):
        navis.Pipeline((navis.prune_twigs, (5000,)))(nl, parallel=True)

    assert [n.n_nodes for n in nl] == before


def test_a_run_does_not_bind_the_pipeline(nl):
    """A pipeline is re-usable: a run must not leave state behind on it."""
    pipe = navis.Pipeline((navis.prune_twigs, (5000,)))

    first = [n.n_nodes for n in pipe(nl, parallel=True, n_cores=2,
                                     backend='processes')]
    second = [n.n_nodes for n in pipe(nl)]

    assert first == second


def test_inplace_writes_back(nl):
    before = [n.n_nodes for n in nl]

    res = navis.Pipeline((navis.prune_twigs, (5000,)))(nl, inplace=True)

    assert res is nl
    assert [n.n_nodes for n in nl] != before


def test_inplace_needs_a_neuronlist():
    with pytest.raises(TypeError, match='inplace'):
        navis.Pipeline(n_nodes)(navis.example_neurons(1), inplace=True)


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def test_picklable_by_reference():
    assert picklable_by_reference(navis.Pipeline(*STEPS)) is True
    assert picklable_by_reference(navis.Pipeline(lambda x: x)) is False


def test_survives_a_round_trip(nl):
    pipe = navis.Pipeline(*STEPS)

    revived = pickle.loads(pickle.dumps(pipe))

    assert len(revived) == len(pipe)
    assert [n.n_nodes for n in revived(nl)] == [n.n_nodes for n in pipe(nl)]


def test_lambda_step_needs_a_by_value_backend(nl):
    """Same contract - and same error - as `NeuronList.apply`."""
    pipe = navis.Pipeline(lambda x: x.n_nodes)

    with pytest.raises(ValueError, match='set_parallel_backend'):
        pipe(nl, parallel=True, n_cores=2, backend='processes')


@pytest.mark.parametrize('backend', ['pathos', 'joblib'], indirect=True)
def test_lambda_step_works_by_value(nl, backend):
    pipe = navis.Pipeline(lambda x: x.n_nodes)

    assert pipe(nl, parallel=True, n_cores=2, backend=backend) == list(nl.n_nodes)


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('backend', LOCAL_BACKENDS, indirect=True)
def test_failure_names_the_step(nl, backend):
    """The step index and name have to survive the trip back from a worker."""
    pipe = navis.Pipeline(navis.heal_skeleton, boom)

    with pytest.raises(navis.PipelineStepError) as exc:
        pipe(nl, parallel=True, n_cores=2, backend=backend)

    assert exc.value.index == 1
    assert exc.value.step == 'boom'
    assert 'boom' in str(exc.value)


def test_failure_error_is_picklable():
    err = navis.PipelineStepError('msg', 1, 'boom', "ValueError('boom')")

    revived = pickle.loads(pickle.dumps(err))

    assert str(revived) == 'msg'
    assert (revived.index, revived.step) == (1, 'boom')


def test_omit_failures_reports_the_step(nl):
    """Failures come back as data, still naming the step that raised."""
    mixed = nl + navis.make_dotprops(nl[0], k=5)
    pipe = navis.Pipeline((navis.prune_twigs, (5000,)))

    res = pipe(mixed, omit_failures=True)

    assert len(res) == len(nl)


def test_a_step_returning_none_names_itself(nl):
    pipe = navis.Pipeline(gives_none, n_nodes)

    with pytest.raises(navis.PipelineStepError, match='returned None'):
        pipe(nl)


# --------------------------------------------------------------------------- #
# Reserved keyword arguments
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('kwarg', ['parallel', 'n_cores', 'chunksize',
                                   'backend', 'progress', 'omit_failures'])
def test_reserved_kwargs_do_not_reach_steps(nl, kwarg):
    """`n_nodes` takes nothing but a neuron; none of these may be forwarded."""
    values = {'parallel': False, 'n_cores': 2, 'chunksize': 1,
              'backend': 'serial', 'progress': False, 'omit_failures': False}

    res = navis.Pipeline(n_nodes)(nl, **{kwarg: values[kwarg]})

    assert res == [n.n_nodes for n in nl]


def test_a_step_may_have_a_parameter_called_progress(nl):
    """Step arguments and call arguments are separate namespaces."""
    pipe = navis.Pipeline((record_inplace, {'inplace': False}))

    pipe(nl, progress=False)

    assert all(ip is False for _, ip in INPLACE_LOG)


# --------------------------------------------------------------------------- #
# NeuronList integration
# --------------------------------------------------------------------------- #
def test_bound_pipeline_matches_the_free_one(nl):
    ref = navis.Pipeline(*STEPS)(nl)

    got = (nl.pipeline
             .heal_skeleton()
             .prune_twigs(5000)
             .resample_skeleton(1000)
             .run())

    assert [n.n_nodes for n in got] == [n.n_nodes for n in ref]


def test_bound_pipeline_builders(nl):
    bound = nl.pipeline.add(navis.heal_skeleton).add_once(len)

    assert len(bound) == 2
    assert bound.run() == len(nl)
    assert isinstance(bound.pipeline, navis.Pipeline)


def test_bound_pipeline_keeps_its_binding(nl):
    """Every builder must stay bound - including the ones it only inherits."""
    bound = nl.pipeline.heal_skeleton()

    for derived in (bound.add(navis.prune_twigs, 5000),
                    bound.add_once(navis.prune_twigs, 5000),
                    bound | (navis.prune_twigs, (5000,)),
                    bound[:1]):
        assert hasattr(derived, 'run'), f'{derived!r} lost its NeuronList'
        assert len(derived.run()) == len(nl)

    # ... and `.pipeline` hands back a plain, re-usable one
    free = bound.pipeline
    assert type(free) is navis.Pipeline
    assert len(free) == len(bound)


def test_bound_pipeline_mode_proxies(nl, registry):  # noqa: F811
    """`once` hands the whole list over, so nothing is dispatched per neuron."""
    be = DummyBackend(isolated=True)

    bound = nl.pipeline.once.prune_twigs(5000)
    res = bound.run(parallel=True, n_cores=2, backend=be)

    assert len(bound) == 1
    assert be.calls == []
    assert [n.n_nodes for n in res] == [n.n_nodes for n in navis.prune_twigs(nl, 5000)]


def test_apply_accepts_a_pipeline(nl):
    pipe = navis.Pipeline(*STEPS)

    got = nl.apply(pipe, parallel=True, n_cores=2)

    assert [n.n_nodes for n in got] == [n.n_nodes for n in pipe(nl)]


def test_apply_rejects_kwargs_for_a_pipeline(nl):
    with pytest.raises(TypeError, match='when its steps'):
        nl.apply(navis.Pipeline(n_nodes), foo=1)
