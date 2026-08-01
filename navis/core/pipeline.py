#    This script is part of navis (http://www.github.com/navis-org/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

"""Composable pipelines.

Chaining navis functions with `parallel=True` pays the cost of moving neurons
between processes once per *function*:

    sk  = navis.skeletonize(nl, parallel=True)          # neurons out and back
    res = navis.resample_skeleton(sk, 500, parallel=True)   # ... and again

A [`navis.Pipeline`][] fuses consecutive per-neuron steps into a single task, so
each neuron makes that trip once for the whole chain no matter how many steps it
has. On top of that it tracks *ownership*: once a step has handed us an object we
made ourselves, the following steps may modify it in place rather than each
taking their own defensive copy.

Two things about `@map_neuronlist` (`navis/utils/decorators.py`) are load-bearing
here, and neither is obvious from this file alone:

  - The fusion works because a decorated function handed a *single* neuron
    passes straight through to the undecorated one, so calling the steps inside
    a worker spawns no further dispatch. If that pass-through ever goes away
    (roadmap item B.2 collapses these layers), every step of every pipeline
    would build a `NeuronProcessor` per neuron.
  - `_takes_inplace` relies on `@wraps` setting `__wrapped__`, so
    `inspect.signature` reports the wrapped function's real parameters rather
    than the wrapper's `(*args, **kwargs)`.

"""

import difflib
import functools
import inspect

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

from .. import config, utils
from ..compute.backends import resolve_backend
from ..compute.dispatch import default_n_workers, picklable_by_reference

logger = config.get_logger(__name__)

__all__ = ['Pipeline', 'PipelineStepError']

#: How a step is applied to the value flowing through the pipeline.
#:   'auto' - map over the neurons if the value is a NeuronList, else call once
#:   'each' - always map over the elements of the value
#:   'once' - always call once, with the whole value
MODES = ('auto', 'each', 'once')

#: Stands in for the value flowing through the pipeline when we check a step's
#: arguments against its signature at construction time.
_PLACEHOLDER = object()


class PipelineStepError(Exception):
    """A step of a [`navis.Pipeline`][] failed.

    Attributes
    ----------
    index :     int
                Position of the offending step in the pipeline.
    step :      str
                Name of the offending function.
    original :  str
                `repr` of the underlying exception.

    """

    # Everything needed to rebuild this lives in `args`, because `__cause__`
    # does not survive being pickled back from a worker process.
    def __init__(self, message, index=None, step=None, original=None):
        super().__init__(message, index, step, original)
        self.index = index
        self.step = step
        self.original = original

    def __str__(self):
        # Without this, the multi-argument `args` would print as a tuple
        return self.args[0]


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Step:
    """One step of a pipeline. Frozen and module level so that it pickles."""

    func: Callable
    args: tuple
    kwargs: dict
    mode: str
    #: Whether `func` has a real `inplace` parameter, worked out once at
    #: construction time. We keep the answer rather than the `Signature` it came
    #: from: this travels to workers, and a bool is a lot cheaper.
    takes_inplace: bool
    name: str


def _step_name(func) -> str:
    """Human-readable name for a callable."""
    if isinstance(func, functools.partial):
        return f'partial({_step_name(func.func)})'
    return getattr(func, '__name__', None) or type(func).__name__


def _takes_inplace(func) -> bool:
    """Whether `func` has an `inplace` parameter we can set."""
    try:
        # Follows `__wrapped__`, so this sees through @map_neuronlist to the
        # real signature underneath.
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return False    # builtins and other callables without a signature

    param = sig.parameters.get('inplace')
    # The `kind` check keeps us honest about `f(x, **inplace)`, and documents
    # that a function swallowing everything in `**kwargs` does *not* count:
    # `inplace=True` would just be forwarded to whatever it wraps.
    return param is not None and param.kind is not inspect.Parameter.VAR_KEYWORD


def _make_step(func, args, kwargs, mode) -> _Step:
    """Validate and normalise a single step."""
    if isinstance(func, (tuple, list)):
        func, args, kwargs = _unpack_spec(func, args, kwargs)

    if not callable(func):
        raise TypeError('Pipeline steps must be callable, got '
                        f'"{type(func).__name__}"')
    if mode not in MODES:
        raise ValueError(f'Unknown step mode "{mode}", expected {MODES}')

    name = _step_name(func)

    # Check the arguments now rather than inside five worker processes. Note
    # this cannot catch anything for a function that takes `**kwargs` only.
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        pass
    else:
        try:
            sig.bind_partial(_PLACEHOLDER, *args, **kwargs)
        except TypeError as e:
            raise TypeError(f'Bad arguments for pipeline step "{name}": {e}') from e

    return _Step(func=func, args=tuple(args), kwargs=dict(kwargs), mode=mode,
                 takes_inplace=_takes_inplace(func), name=name)


def _unpack_spec(spec, args, kwargs):
    """Unpack a `(func, args)` / `(func, kwargs)` / `(func, args, kwargs)`."""
    if args or kwargs:
        raise TypeError('Pass arguments either inside the tuple or as '
                        'arguments to `add`, not both.')
    if len(spec) == 2:
        func, second = spec
        if isinstance(second, dict):
            return func, (), second
        return func, _as_args(second), {}
    elif len(spec) == 3:
        func, args, kwargs = spec
        return func, _as_args(args), kwargs
    raise ValueError('Expected a step of the form `(func, args)`, '
                     f'`(func, kwargs)` or `(func, args, kwargs)`, got {spec}')


def _as_args(args) -> tuple:
    """Normalise the positional arguments of a tuple-spelled step.

    Note this must *not* go through `utils.make_iterable`: that turns a single
    list argument into an array, so `(func, ([1, 2],))` and `.add(func, [1, 2])`
    would not be the same step.
    """
    return tuple(args) if isinstance(args, (tuple, list)) else (args,)


# --------------------------------------------------------------------------- #
# The fused chain
# --------------------------------------------------------------------------- #
class _ChainRunner:
    """A run of pipeline steps, fused into a single callable.

    This is what crosses the process boundary: one of these per neuron, instead
    of one task per neuron *per step*.
    """

    __slots__ = ('steps', 'offset', 'owns_input', 'elide')

    def __init__(self, steps, offset, owns_input, elide):
        self.steps = tuple(steps)
        #: Index of `steps[0]` in the pipeline it came from - purely so that
        #: error messages can name the right step.
        self.offset = offset
        #: Whether we may modify the value we are handed. See `Pipeline`.
        self.owns_input = owns_input
        self.elide = elide

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return f'_ChainRunner({" | ".join(s.name for s in self.steps)})'

    def __call__(self, value):
        # Workers only want the result; the parent uses `run` for the ownership
        # it has to carry into the next segment.
        return self.run(value)[0]

    def run(self, value) -> Tuple[Any, bool]:
        """Run the steps in sequence. Returns `(result, do_we_own_it)`."""
        owns = self.owns_input
        for i, step in enumerate(self.steps):
            index = self.offset + i

            if value is None:
                prev = self.steps[i - 1].name if i else '<input>'
                raise PipelineStepError(
                    f'Pipeline step {index} ({step.name}) has nothing to run '
                    f'on: {prev} returned None.',
                    index, step.name, 'received None')

            kwargs = step.kwargs
            # The heart of it: if this object is ours, the step may modify it
            # instead of copying. Never rewrite the stored kwargs in place -
            # this same step object is reused for every neuron.
            if (self.elide and owns and step.takes_inplace
                    and 'inplace' not in step.kwargs):
                kwargs = {**kwargs, 'inplace': True}

            try:
                out = step.func(value, *step.args, **kwargs)
            except Exception as e:
                # Deliberately narrower than `BaseException`: a KeyboardInterrupt
                # or SystemExit must travel untouched.
                raise PipelineStepError(
                    f'Pipeline step {index} ({step.name}) failed: '
                    f'{type(e).__name__}: {e}',
                    index, step.name, repr(e)) from e

            # Some functions return nothing when told to work in place
            if out is None and kwargs.get('inplace', False):
                out = value

            # Monotone on purpose. Under `inplace=True` a navis function returns
            # the *same* object, so a rule that re-derived ownership from the
            # identity check alone would revoke it on the very step that used
            # it - and elide only every other copy. The check can therefore only
            # ever grant ownership, never take it away.
            owns = owns or out is not value
            value = out

        return value, owns


def _shares_objects(before, after) -> bool:
    """Whether any object in `after` also appears in `before`."""
    seen = {id(o) for o in before}
    return any(id(o) in seen for o in after)


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    """A reusable, composable sequence of operations.

    Chaining navis functions over a `NeuronList` with `parallel=True` sends the
    neurons to a worker and back once per function. A pipeline fuses consecutive
    per-neuron steps into a single task, so they make that trip once for the
    whole chain - and lets each step modify the intermediate the previous one
    produced instead of taking its own copy.

    Pipelines are immutable: [`add`][navis.Pipeline.add] and friends return a
    new pipeline, so one can be built up, shared and re-used freely.

    Parameters
    ----------
    *steps :        callable | tuple | Pipeline
                    The steps to run, in order. Each can be:

                      - a callable, e.g. `navis.heal_skeleton`
                      - `(callable, args)`, `(callable, kwargs)` or
                        `(callable, args, kwargs)`
                      - a `functools.partial` - bind its arguments by keyword,
                        since positional ones would come *before* the neuron
                      - another `Pipeline`, whose steps are spliced in

                    Steps do not have to be navis functions - anything callable
                    will do (see [`add`][navis.Pipeline.add]).
    elide_copies :  bool
                    Whether steps may modify intermediates in place instead of
                    copying them. Turn this off if a step returns a new neuron
                    that shares data with its input.
    desc :          str, optional
                    Progress bar description. Defaults to the step names.

    See Also
    --------
    [`navis.NeuronList.pipeline`][]
                    Build and run a pipeline straight off a `NeuronList`.
    [`navis.NeuronList.apply`][]
                    Map a single function over a `NeuronList`.

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(3, kind='skeleton')

    Steps can be given up front...

    >>> pipe = navis.Pipeline(
    ...     navis.heal_skeleton,
    ...     (navis.prune_twigs, (5000,)),
    ...     (navis.resample_skeleton, {'resample_to': 1000}),
    ... )
    >>> pipe
    Pipeline with 3 step(s): heal_skeleton | prune_twigs | resample_skeleton

    ... or added one at a time. Any navis function can be named directly:

    >>> pipe = navis.Pipeline().heal_skeleton().prune_twigs(5000)
    >>> len(pipe)
    2

    Running it returns a new `NeuronList` and leaves the input alone:

    >>> res = pipe(nl)
    >>> len(res)
    3
    >>> all(a.n_nodes >= b.n_nodes for a, b in zip(nl, res))
    True

    With `parallel=True` each neuron is sent to a worker once for the whole
    chain, instead of once per step:

    >>> res = pipe(nl, parallel=True, n_cores=2)             # doctest: +SKIP

    The input does not have to be neurons - it is whatever the first step
    accepts. Here a query object is turned into neurons by the first step, and
    everything after that runs per neuron:

    >>> import navis.interfaces.neuprint as neu                      # doctest: +SKIP
    >>> pipe = navis.Pipeline(neu.fetch_skeletons).resample_skeleton(1000)  # doctest: +SKIP
    >>> res = pipe(neu.NeuronCriteria(type='.*LPN'), parallel=True)  # doctest: +SKIP

    """

    def __init__(self, *steps, elide_copies: bool = True,
                 desc: Optional[str] = None):
        self._elide_copies = bool(elide_copies)
        self._desc = desc

        parsed: Tuple[_Step, ...] = ()
        for step in steps:
            if isinstance(step, Pipeline):
                # Splice, like `TransformSequence.append` does for sequences
                parsed += step._steps
            else:
                parsed += (_make_step(step, (), {}, 'auto'),)
        self._steps = parsed

    # ----------------------------------------------------------------- build
    def _derive(self, steps) -> 'Pipeline':
        """A copy of this pipeline with different steps."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._steps = tuple(steps)
        return new

    def _add(self, func, args, kwargs, mode) -> 'Pipeline':
        if isinstance(func, Pipeline):
            if args or kwargs:
                raise TypeError('Cannot pass arguments when adding a Pipeline.')
            return self._derive(self._steps + func._steps)
        return self._derive(self._steps + (_make_step(func, args, kwargs, mode),))

    def _add_named(self, func, mode, *args, **kwargs) -> 'Pipeline':
        """Backs the fluent form, where the arguments arrive on a later call."""
        return self._add(func, args, kwargs, mode)

    def add(self, func: Callable, *args, **kwargs) -> 'Pipeline':
        """Add a step and return the new pipeline.

        The step is applied to each neuron individually if the value reaching it
        is a `NeuronList`, and called once with the whole value otherwise. Use
        [`add_each`][navis.Pipeline.add_each] or
        [`add_once`][navis.Pipeline.add_once] to force one or the other.

        Parameters
        ----------
        func :      callable
                    Any callable. It is handed the value flowing through the
                    pipeline as its first argument.
        *args
        **kwargs
                    Bound to `func` now and passed on every call.

        Returns
        -------
        Pipeline
                    A **new** pipeline - the original is not modified.

        Examples
        --------
        >>> import navis
        >>> pipe = navis.Pipeline(navis.heal_skeleton)
        >>> longer = pipe.add(navis.prune_twigs, 5000)
        >>> len(pipe), len(longer)
        (1, 2)

        Steps do not have to be navis functions:

        >>> def n_nodes(x):
        ...     return x.n_nodes
        >>> res = navis.Pipeline(n_nodes)(navis.example_neurons(2))
        >>> len(res)
        2

        """
        return self._add(func, args, kwargs, 'auto')

    def add_each(self, func: Callable, *args, **kwargs) -> 'Pipeline':
        """Add a step that is mapped over the elements of the value.

        Use this to fan out over something that is not a `NeuronList` - e.g. to
        fetch a list of IDs in parallel.

        Examples
        --------
        >>> import navis
        >>> pipe = navis.Pipeline().add_each(navis.example_neurons)
        >>> res = pipe([1, 1])
        >>> len(res)
        2

        """
        return self._add(func, args, kwargs, 'each')

    def add_once(self, func: Callable, *args, **kwargs) -> 'Pipeline':
        """Add a step that is called once, with the whole value.

        Use this for functions that work across a whole `NeuronList` rather than
        neuron by neuron - [`navis.xform_brain`][] pools all the coordinates and
        transforms them in one go, for example.

        Examples
        --------
        >>> import navis
        >>> pipe = navis.Pipeline().add_once(len)
        >>> pipe(navis.example_neurons(3))
        3

        """
        return self._add(func, args, kwargs, 'once')

    def __or__(self, other) -> 'Pipeline':
        """Append a step or another pipeline.

        Examples
        --------
        >>> import navis
        >>> pipe = navis.Pipeline(navis.heal_skeleton) | navis.prune_twigs
        >>> len(pipe)
        2

        """
        # `_add` already splices a Pipeline and unpacks a tuple spec
        if not callable(other) and not isinstance(other, (tuple, list)):
            return NotImplemented
        return self._add(other, (), {}, 'auto')

    # ------------------------------------------------------------- introspect
    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self):
        return iter(self._steps)

    def __getitem__(self, ix):
        if isinstance(ix, slice):
            return self._derive(self._steps[ix])
        return self._steps[ix]

    def __repr__(self) -> str:
        if not self._steps:
            return 'Pipeline with 0 step(s)'
        names = ' | '.join(s.name for s in self._steps)
        return f'Pipeline with {len(self)} step(s): {names}'

    @property
    def __name__(self) -> str:
        # A property, so it stays off the instance `__dict__`. `NeuronList.apply`
        # builds its progress description from this.
        return 'Pipeline'

    @property
    def __picklable_by_reference__(self) -> bool:
        """Whether plain `pickle` can ship this pipeline to a worker.

        A `Pipeline` is not itself importable by name, but it travels fine as
        long as each of its steps does.
        """
        return all(picklable_by_reference(s.func) for s in self._steps)

    # --------------------------------------------------------- fluent lookups
    @property
    def each(self) -> '_ModeProxy':
        """Add the next named step with [`add_each`][navis.Pipeline.add_each]."""
        return _ModeProxy(self, 'each')

    @property
    def once(self) -> '_ModeProxy':
        """Add the next named step with [`add_once`][navis.Pipeline.add_once]."""
        return _ModeProxy(self, 'once')

    def __getattr__(self, name):
        """Resolve `name` as a navis function and return a step builder.

        This is what makes `Pipeline().prune_twigs(5000)` work.
        """
        return _named_step_builder(self, name, 'auto')

    def __dir__(self):
        import navis
        return sorted(set(super().__dir__())
                      | {n for n in dir(navis) if not n.startswith('_')
                         and callable(getattr(navis, n, None))})

    # ------------------------------------------------------------------- run
    def __call__(self, x, *,
                 parallel: bool = False,
                 n_cores: Optional[int] = None,
                 chunksize: Optional[int] = None,
                 backend=None,
                 progress: bool = True,
                 omit_failures: bool = False,
                 inplace: bool = False):
        """Run the pipeline.

        Parameters
        ----------
        x :             NeuronList | Neuron | any
                        The input to the first step. Usually neurons, but it can
                        be anything the first step accepts - a query object, a
                        list of IDs, a filepath.
        parallel :      bool
                        Whether to distribute the per-neuron steps across
                        multiple cores. Steps that run on the whole value at
                        once (see [`add_once`][navis.Pipeline.add_once]) always
                        run in this process.
        n_cores :       int, optional
                        Number of cores to use. Defaults to half of them.
        chunksize :     int, optional
                        Neurons to hand a worker at a time. Defaults to letting
                        the backend decide.
        backend :       str | ParallelBackend | concurrent.futures.Executor, optional
                        Where to run. Defaults to
                        [`navis.set_parallel_backend`][]'s setting.
        progress :      bool
                        Whether to show a progress bar.
        omit_failures : bool
                        If True, drop neurons whose chain raised instead of
                        propagating the error.
        inplace :       bool
                        If True, write the results back into `x` and return it.
                        Requires `x` to be a `NeuronList`.

        Returns
        -------
        Whatever the last step produced: a `NeuronList` if the results are all
        neurons, `None` if they are all `None`, else a list.

        """
        from .neuronlist import NeuronList

        if not len(self._steps):
            raise ValueError('This pipeline has no steps. Add some with e.g. '
                             '`pipeline.add(navis.prune_twigs, 5000)`.')

        if inplace and not isinstance(x, NeuronList):
            raise TypeError('`inplace=True` requires a NeuronList, got '
                            f'"{type(x).__name__}".')

        # A generator can only be walked once, and we may need its length
        if isinstance(x, Iterator):
            x = list(x)

        n_cores = n_cores or default_n_workers()

        value = _as_collection(x)
        # `inplace` is the caller telling us their neurons are ours to modify
        owns = bool(inplace)
        fanned_out = False
        i = 0

        while i < len(self._steps):
            step = self._steps[i]

            if not _is_fanout(step, value):
                value, owns = _ChainRunner(
                    [step], offset=i, owns_input=owns,
                    elide=self._elide_copies).run(value)
                value = _as_collection(value)
                i += 1
                continue

            # Absorb everything up to the next whole-value step into one task
            j = i + 1
            while j < len(self._steps) and self._steps[j].mode != 'once':
                j += 1

            value, owns = self._fanout(
                value, self._steps[i:j], offset=i, owns=owns,
                parallel=parallel, n_cores=n_cores, chunksize=chunksize,
                backend=backend, progress=progress,
                omit_failures=omit_failures)
            fanned_out = True
            i = j

        if parallel and not fanned_out and isinstance(x, NeuronList) and len(x) > 1:
            logger.warning('`parallel=True` had no effect: none of this '
                           "pipeline's steps run per neuron.")

        if inplace:
            if not isinstance(value, NeuronList):
                raise TypeError('`inplace=True` needs the pipeline to produce '
                                f'neurons, got "{type(value).__name__}".')
            x.neurons = value.neurons
            return x

        return value

    def _fanout(self, value, steps, *, offset, owns, parallel, n_cores,
                chunksize, backend, progress, omit_failures):
        """Run `steps` over the elements of `value`, in parallel if asked."""
        from .core_utils import assemble_results, mean_task_size, run_tasks
        from .neuronlist import NeuronList

        be = resolve_backend(
            backend,
            parallel=parallel,
            n_tasks=len(value),
            n_workers=n_cores,
            # Only this segment is shipped, so a lambda in an `add_once` step -
            # which never leaves this process - must not constrain the choice.
            by_value=not all(picklable_by_reference(s.func) for s in steps),
        )

        # A worker in its own process is handed a copy, so it owns it no matter
        # what we own out here - the same trade the @map_neuronlist decorator
        # makes. On threads (or when this degraded to running inline) "in place"
        # would mean the caller's own neurons, so we must not.
        runner = _ChainRunner(steps, offset=offset,
                              owns_input=owns or be.isolated,
                              elide=self._elide_copies)

        is_nl = isinstance(value, NeuronList)
        res, failed = run_tasks(
            # One runner shared by every task: pickle stores it once per chunk
            [(runner, (el,), {}) for el in value],
            backend=be,
            n_workers=n_cores,
            chunksize=chunksize,
            omit_failures=omit_failures,
            desc=self._desc or _segment_desc(steps),
            progress=progress,
            size_hint=(lambda: mean_task_size(value)) if is_nl else None,
            # Lazy - see the same call in `NeuronProcessor._run`
            labels=(lambda: value.id) if is_nl else None,
        )

        # Anything that came back from another process is a fresh object; on a
        # shared-memory backend we have to look.
        owns = owns or be.isolated or not _shares_objects(value, res)

        return assemble_results(res, cls=type(value) if is_nl else None), owns


class _BoundPipeline(Pipeline):
    """A pipeline being built up against a particular NeuronList.

    Backs [`navis.NeuronList.pipeline`][]. It *is* a `Pipeline` that happens to
    remember which neurons it was started from, so every builder it inherits
    keeps the binding for free - `_derive` copies the instance `__dict__`, and
    `_nl` rides along with it. Nothing runs until
    [`run`][navis.core.pipeline._BoundPipeline.run].
    """

    def __init__(self, nl, pipeline: Pipeline):
        self.__dict__.update(pipeline.__dict__)
        self._nl = nl

    def __repr__(self):
        return f'{super().__repr__()}, bound to {len(self._nl)} neuron(s)'

    @property
    def pipeline(self) -> Pipeline:
        """The unbound, re-usable pipeline built so far."""
        free = Pipeline()
        free.__dict__.update({k: v for k, v in self.__dict__.items()
                              if k != '_nl'})
        return free

    def run(self, **kwargs):
        """Run the pipeline on the NeuronList it was built from.

        Takes the same keyword arguments as [`navis.Pipeline.__call__`][] -
        `parallel`, `n_cores`, `chunksize`, `backend`, `progress`,
        `omit_failures` and `inplace`.
        """
        return self(self._nl, **kwargs)


class _ModeProxy:
    """Backs `pipeline.each.<func>()` and `pipeline.once.<func>()`."""

    __slots__ = ('_pipeline', '_mode')

    def __init__(self, pipeline, mode):
        self._pipeline = pipeline
        self._mode = mode

    def __repr__(self):
        return f'<add the next step with mode "{self._mode}">'

    def __getattr__(self, name):
        return _named_step_builder(self._pipeline, name, self._mode)

    def __dir__(self):
        return dir(self._pipeline)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _named_step_builder(pipeline, name, mode):
    """Resolve `name` in the navis namespace and bind it as the next step.

    Shared by `Pipeline.__getattr__` and `_ModeProxy.__getattr__`, which is why
    the underscore guard lives here: `pickle` and `copy` look up
    `__reduce_ex__`, `__getstate__`, `__deepcopy__` and friends, and would
    otherwise be handed a builder. Worse, an instance being unpickled has no
    `_steps` yet, so a lookup for one would recurse forever.
    """
    if name.startswith('_'):
        raise AttributeError(name)
    return functools.partial(pipeline._add_named,
                             _resolve_navis(name, type(pipeline)), mode)


def _resolve_navis(name, owner=Pipeline):
    """Look `name` up in the navis namespace."""
    # Deferred: importing navis at module level would be circular
    import navis

    func = getattr(navis, name, None)
    if callable(func):
        return func

    msg = f"'{owner.__name__}' object has no attribute '{name}'"
    if func is not None:
        raise AttributeError(f'{msg}: `navis.{name}` is not callable.')

    msg += ' and there is no `navis.{}`.'.format(name)
    close = difflib.get_close_matches(name, dir(navis), n=3)
    if close:
        msg += f' Did you mean: {", ".join(close)}?'
    raise AttributeError(f'{msg} To use a function that is not part of navis, '
                         'add it explicitly: `.add(my_func, ...)`.')


def _as_collection(value):
    """Wrap a bare sequence of neurons in a NeuronList; pass anything else on."""
    from .base import BaseNeuron
    from .neuronlist import NeuronList

    if isinstance(value, (NeuronList, BaseNeuron)):
        return value
    if isinstance(value, (list, tuple, np.ndarray)) and len(value):
        if all(isinstance(v, (BaseNeuron, NeuronList)) for v in value):
            return NeuronList(value)
    return value


def _is_fanout(step, value) -> bool:
    """Whether `step` should be mapped over the elements of `value`."""
    from .neuronlist import NeuronList

    if step.mode == 'once':
        return False
    if step.mode == 'each':
        if not utils.is_iterable(value):
            raise TypeError(f'Pipeline step "{step.name}" was added with '
                            '`add_each` but the value it received is not '
                            f'iterable: "{type(value).__name__}".')
        return True
    return isinstance(value, NeuronList)


def _segment_desc(steps) -> str:
    """Progress bar description for a run of steps."""
    names = [s.name for s in steps]
    desc = ' | '.join(names)
    if len(desc) > 40:
        desc = f'{names[0]} +{len(names) - 1}'
    return desc
