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

"""Module for decorators.

Important: defer importing from other navis modules to avoid circular imports!
Even if this doesn't cause immediate issues it might well break pickling
functions (e.g. for multiprocessing).

"""

import re
import inspect
import warnings

import numpy as np
import pandas as pd

from functools import wraps
from textwrap import dedent, indent

from typing import Optional, Union, List, Iterable, Dict, Tuple, Any
from typing_extensions import Literal

from .iterables import is_iterable, make_iterable


def map_neuronlist(
    desc: str = "",
    can_zip: List[Union[str, int]] = [],
    must_zip: List[Union[str, int]] = [],
    allow_parallel: bool = False,
):
    """Decorate function to run on all neurons in the NeuronList.

    This also updates the docstring.

    Parameters
    ----------
    desc :           str
                     Descriptor to show in the progress bar if run over multiple
                     neurons.
    can_zip/
    must_zip :       list
                     Names of keyword arguments that need to be zipped together
                     with the neurons in the neuronlist. For example:

                       some_function(NeuronList([n1, n2, n3]), [p1, p2, p3])

                     Should be executed as:

                       some_function(n1, p1)
                       some_function(n2, p2)
                       some_function(n3, p3)

                     `can_zip` will be zipped only if the length matches the
                     length of the neuronlist. If a `can_zip` argument has only
                     one value it will be re-used for all neurons.

                     `must_zip` arguments have to have one value for each of the
                     neurons.

                     Single `None` values are always just passed through.

                     Note that for this to consistently work the parameters in
                     question have to be keyword-only (*).
    allow_parallel : bool
                     If True and the function is called with `parallel=True`,
                     will use multiple cores to process the neuronlist. Number
                     of cores a can be set using `n_cores` keyword argument.

    """

    # TODO:
    # - make can_zip/must_zip work with positional-only argumens to, i.e. let
    #   it work with integers instead of strings
    def decorator(function):
        # Computed once at decoration time rather than per call: workers
        # re-enter this wrapper for every neuron, and `inspect.signature` is
        # expensive enough to dominate the per-neuron dispatch cost.
        sig = inspect.signature(function)

        @wraps(function)
        def wrapper(*args, **kwargs):
            from .. import core, compute

            try:
                fnname = function.__name__
            except BaseException:
                fnname = str(function)

            parallel = kwargs.pop("parallel", False)
            if parallel and not allow_parallel:
                raise ValueError(
                    f"Function {fnname} does not support parallel processing."
                )

            # First, we need to extract the neuronlist
            if args:
                # If there are positional arguments, the first one is
                # the input neuron(s)
                nl = args[0]
                nl_key = "__args"
            else:
                # If not, we need to look for the name of the first argument
                # in the signature
                nl_key = list(sig.parameters.keys())[0]
                nl = kwargs.get(nl_key, None)

            # Complain if we did not get what we expected
            if isinstance(nl, type(None)):
                raise ValueError(
                    "Unable to identify the neurons for call"
                    f"{fnname}:\n {args}\n {kwargs}"
                )

            # If we have a neuronlist
            if isinstance(nl, core.NeuronList):
                # Pop the neurons from kwargs or args so we don't pass the
                # neurons twice
                if nl_key == "__args":
                    args = args[1:]
                else:
                    _ = kwargs.pop(nl_key)

                # Check "can zip" arguments
                for p in can_zip:
                    # Skip if not present or is None
                    if p not in kwargs or isinstance(kwargs[p], type(None)):
                        continue

                    if is_iterable(kwargs[p]):
                        # If iterable but length does not match: complain
                        le = len(kwargs[p])
                        if le != len(nl):
                            raise ValueError(
                                f"Got {le} values of `{p}` for {len(nl)} neurons."
                            )

                # Parse "must zip" arguments
                for p in must_zip:
                    # Skip if not present or is None
                    if p not in kwargs or isinstance(kwargs[p], type(None)):
                        continue

                    values = make_iterable(kwargs[p])
                    if len(values) != len(nl):
                        raise ValueError(
                            f"Got {len(values)} values of `{p}` for {len(nl)} neurons."
                        )

                if "inplace" in kwargs:
                    # First check keyword arguments
                    inplace = kwargs["inplace"]
                elif "inplace" in sig.parameters:
                    # Next check signatures default
                    inplace = sig.parameters["inplace"].default
                else:
                    # All things failing assume it's not inplace
                    inplace = False

                # Prepare processor. `n_cores` is defaulted here rather than
                # left to `NeuronProcessor`, because which backend we resolve
                # below depends on it: resolving against `None` and then running
                # against the default can pick two *different* backends, and the
                # `inplace` decision hangs off the one we resolve here.
                n_cores = kwargs.pop("n_cores", None) or compute.default_n_workers()
                chunksize = kwargs.pop("chunksize", None)

                # Resolve where this will run *before* deciding on `inplace`
                # below - the two are not independent.
                be = compute.resolve_backend(
                    kwargs.pop("backend", None),
                    parallel=parallel,
                    n_tasks=len(nl),
                    n_workers=n_cores,
                )

                # If neurons are copied into worker processes anyway, we may as
                # well let the function modify them in place and save a copy.
                # Only where they really are copied, though: on a thread pool -
                # or when the work degraded to running inline - "in place" means
                # the caller's own neurons, so honouring `inplace=False` there
                # is the difference between a copy and silent mutation.
                if parallel and be.isolated and "inplace" in sig.parameters:
                    kwargs["inplace"] = True

                # Keyword arguments are not zipped unless they were declared
                # zippable - otherwise any kwarg whose length happened to match
                # the number of neurons would be silently sliced up.
                excl = [k for k in kwargs
                        if k not in can_zip and k not in must_zip]
                excl += list(range(1, len(args) + 1))
                proc = core.NeuronProcessor(
                    nl,
                    wrapper,
                    parallel=parallel,
                    desc=desc,
                    warn_inplace=False,
                    progress=kwargs.pop("progress", True),
                    omit_failures=kwargs.pop("omit_failures", False),
                    chunksize=chunksize,
                    exclude_zip=excl,
                    n_cores=n_cores,
                    backend=be,
                )
                # Apply function
                res = proc(nl, *args, **kwargs)

                # When using parallel processing, the neurons will not actually
                # have been modified inplace - in that case we will simply
                # replace the neurons in `nl`
                if inplace:
                    nl.neurons = res.neurons
                else:
                    nl = res

                return nl
            else:
                # If single neuron just pass through
                return function(*args, **kwargs)

        # Update the docstring
        wrapper = map_neuronlist_update_docstring(wrapper, allow_parallel)

        return wrapper

    return decorator


def map_neuronlist_df(
    desc: str = "",
    id_col: str = "neuron",
    reset_index: bool = True,
    allow_parallel: bool = False,
):
    """Decorate function to run on all neurons in the NeuronList.

    This version of the decorator is meant for functions that return a
    DataFrame. This decorator will add a `neuron` column with the respective
    neuron's ID and will then concatenate the dataframes.

    Parameters
    ----------
    desc :           str
                     Descriptor to show in the progress bar if run over multiple
                     neurons.
    id_col :         str
                     Name of the ID column to be added to the results dataframe.
    reset_index :    bool
                     Whether to reset the index of the dataframe after
                     concatenating.
    allow_parallel : bool
                     If True and the function is called with `parallel=True`,
                     will use multiple cores to process the neuronlist. Number
                     of cores a can be set using `n_cores` keyword argument.

    """

    # TODO:
    # - make can_zip/must_zip work with positional-only argumens to, i.e. let
    #   it work with integers instead of strings
    def decorator(function):
        # Computed once at decoration time rather than per call: workers
        # re-enter this wrapper for every neuron, and `inspect.signature` is
        # expensive enough to dominate the per-neuron dispatch cost.
        sig = inspect.signature(function)

        @wraps(function)
        def wrapper(*args, **kwargs):
            # Lazy import to avoid issues with circular imports and pickling
            from .. import core, compute

            try:
                fnname = function.__name__
            except BaseException:
                fnname = str(function)

            parallel = kwargs.pop("parallel", False)
            if parallel and not allow_parallel:
                raise ValueError(
                    f"Function {fnname} does not allow parallel processing."
                )

            # First, we need to extract the neuronlist
            if args:
                # If there are positional arguments, the first one is
                # the input neuron(s)
                nl = args[0]
                nl_key = "__args"
            else:
                # If not, we need to look for the name of the first argument
                # in the signature
                nl_key = list(sig.parameters.keys())[0]
                nl = kwargs.get(nl_key, None)

            # Complain if we did not get what we expected
            if isinstance(nl, type(None)):
                raise ValueError(
                    "Unable to identify the neurons for call"
                    f"{fnname}:\n {args}\n {kwargs}"
                )

            # If we have a neuronlist
            if isinstance(nl, core.NeuronList):
                # Pop the neurons from kwargs or args so we don't pass the
                # neurons twice
                if nl_key == "__args":
                    args = args[1:]
                else:
                    _ = kwargs.pop(nl_key)

                # Prepare processor. Unlike `map_neuronlist` this decorator has
                # no `inplace` decision to make, so it has no reason to resolve
                # the backend itself - `NeuronProcessor` does that once.
                n_cores = kwargs.pop("n_cores", None)
                chunksize = kwargs.pop("chunksize", None)
                be = kwargs.pop("backend", None)
                excl = list(kwargs.keys()) + list(range(1, len(args) + 1))
                proc = core.NeuronProcessor(
                    nl,
                    wrapper,
                    parallel=parallel,
                    desc=desc,
                    warn_inplace=False,
                    progress=kwargs.pop("progress", True),
                    omit_failures=kwargs.pop("omit_failures", False),
                    chunksize=chunksize,
                    exclude_zip=excl,
                    n_cores=n_cores,
                    backend=be,
                )
                # Apply function. Note we use `_run` rather than calling the
                # processor: with `omit_failures=True` the failed runs are
                # dropped, so `res` is shorter than `nl` and zipping the two
                # directly would label each dataframe with the wrong neuron.
                out = proc._run(nl, *args, **kwargs)
                res = out.results

                for n, df in zip(out.neurons, res):
                    df.insert(0, column=id_col, value=n.id)

                if not res:
                    # Every run failed (`_run` has already warned about it) -
                    # there is nothing to concatenate
                    return pd.DataFrame()

                df = pd.concat(res, axis=0)

                if reset_index:
                    df = df.reset_index(drop=True)

            else:
                # If single neuron just pass through
                df = function(*args, **kwargs)
                # df.insert(0, column=id_col, value=nl.id)

            return df

        # Update the docstring
        wrapper = map_neuronlist_update_docstring(wrapper, allow_parallel)

        return wrapper

    return decorator


def map_neuronlist_update_docstring(func, allow_parallel):
    """Add additional parameters to docstring of function."""
    # Parse docstring
    lines = func.__doc__.split("\n")

    # Find a line with a parameter
    pline = [l for l in lines if " : " in l][0]
    # Get the leading whitespaces
    wspaces = " " * re.search("( *)", pline).end(1)
    # Get the offset for type and description
    offset = re.search("( *: *)", pline).end(1) - len(wspaces)

    # Find index of the last parameters (assuming there is a single empty
    # line between Returns and the last parameter)
    try:
        lastp = [
            i
            for i, line in enumerate(lines[:-1])
            if "Returns" in line and "----" in lines[i + 1]
        ][0] - 1
    except IndexError:
        warnings.warn(f'Could not find "Returns" in docstring for function {func}')
        return func

    msg = ""
    if allow_parallel:
        msg += dedent(f"""\
        parallel :{" " * (offset - 10)}bool
                  {" " * (offset - 10)}If True and input is NeuronList, distribute the
                  {" " * (offset - 10)}work across multiple processes. See
                  {" " * (offset - 10)}`navis.set_parallel_backend` for where it runs.
        n_cores : {" " * (offset - 10)}int, optional
                  {" " * (offset - 10)}Numbers of cores to use if `parallel=True`.
                  {" " * (offset - 10)}Defaults to half the available cores.
        chunksize :{" " * (offset - 11)}int, optional
                  {" " * (offset - 10)}Number of neurons to hand a worker at a time.
                  {" " * (offset - 10)}Defaults to letting the backend decide.
        backend : {" " * (offset - 10)}str | ParallelBackend, optional
                  {" " * (offset - 10)}Override where this call runs. Defaults to
                  {" " * (offset - 10)}`navis.config.default_parallel_backend`.
        """)

    msg += dedent(f"""\
    progress :{" " * (offset - 10)}bool
              {" " * (offset - 10)}Whether to show a progress bar. Overruled by
              {" " * (offset - 10)}`navis.set_pbars`.
    omit_failures :{" " * (offset - 15)}bool
                   {" " * (offset - 15)}If True will omit failures instead of raising
                   {" " * (offset - 15)}an exception. Ignored if input is single neuron.
    """)

    # Insert new docstring
    lines.insert(lastp, indent(msg, wspaces))

    # Update docstring
    func.__doc__ = "\n".join(lines)

    return func


def lock_neuron(function):
    """Lock neuron while function is executed.

    This makes sure that temporary attributes aren't re-calculated as changes
    are being made.

    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        # Lazy import to avoid issues with circular imports and pickling
        from .. import core

        # Lock if first argument is a neuron
        if isinstance(args[0], core.BaseNeuron):
            args[0]._lock = getattr(args[0], "_lock", 0) + 1
        try:
            # Execute function
            res = function(*args, **kwargs)
        except BaseException:
            raise
        finally:
            # Unlock neuron
            if isinstance(args[0], core.BaseNeuron):
                args[0]._lock -= 1
                if args[0]._lock == 0:
                    # The neuron has stopped moving, so anything that carried a
                    # link through this call can now be vouched for. This lives
                    # here rather than in the functions that select because
                    # every one of them would otherwise have to remember, and
                    # forgetting is silent (see `schema.refresh_links`). Costs a
                    # dict lookup per link when there is nothing pending.
                    core.schema.refresh_links(args[0])
        # Return result
        return res

    return wrapper


def rebuilds(axis: str):
    """Decorate a function that replaces an axis' elements rather than selecting.

    Rebuilding is a different thing from selecting and needs saying so.
    [`navis.subset_neuron`][] takes elements away, so everything else - a
    connector's node, a tag, anything attached - follows by construction.
    [`navis.resample_skeleton`][] does not remove any part of the neuron; it
    re-samples it, so the node a connector named is gone while the connector
    still sits on the arbour. Only the function knows where it went.

    So the function says. It returns `(neuron, schema.Rebuild)` instead of just
    the neuron, and this unwraps that: the `Rebuild` goes to
    `schema.apply_rebuild` and the caller gets the neuron, as before.

    Two things happen around the call:

    - the axis' attached data is taken out of the way first, because the
      function will assign the new elements through the public setter, which -
      knowing nothing of the rebuild - would drop it;
    - `_replacing` is stood down for the same reason, so it does not warn about
      something we are about to put back.

    Goes *inside* `map_neuronlist`, so it always sees a single neuron.

    Parameters
    ----------
    axis :  str
            Name of the axis being rebuilt, e.g. `"nodes"`.

    """

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            from .. import core

            neuron = args[0] if args else None
            if not isinstance(neuron, core.BaseNeuron):
                return _split_rebuild(function(*args, **kwargs))[0]

            state = core.schema.capture_rebuild(neuron, axis)
            if state is None:
                return _split_rebuild(function(*args, **kwargs))[0]

            for attr in state.aligned:
                neuron.detach(attr)

            with core.schema.replacing(neuron, axis):
                res, rebuild = _split_rebuild(function(*args, **kwargs))

            # `inplace=False` means the result is a copy taken *after* we moved
            # the data aside, so it is the result that gets it back - and the
            # input keeps what it always had.
            if res is not neuron:
                for attr, value in state.aligned.items():
                    neuron.attach(
                        attr,
                        value,
                        axis,
                        on_rebuild="carry" if attr in state.axis.carried else "drop",
                    )
            core.schema.apply_rebuild(res, state, rebuild)
            return res

        return wrapper

    return decorator


def _split_rebuild(result):
    """Split a `(neuron, Rebuild)` return, insisting on one.

    Always strict: a wrapped function that returns the bare neuron has skipped
    saying where its elements went, which is silent and is the whole thing
    `@rebuilds` exists to prevent.
    """
    from ..core import schema

    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], schema.Rebuild)
    ):
        return result
    raise TypeError(
        "A function wrapped in `@rebuilds` must return "
        "`(neuron, schema.Rebuild(...))` saying where its old elements went, "
        f"got {type(result).__name__}. Pass `Rebuild()` if nothing can be said "
        "- references into the axis are then repaired as a selection would "
        "repair them."
    )


def meshneuron_skeleton(
    method: Union[
        Literal["subset"],
        Literal["split"],
        Literal["node_properties"],
        Literal["node_to_vertex"],
        Literal["pass_through"],
    ],
    include_connectors: bool = False,
    copy_properties: list = [],
    disallowed_kwargs: dict = {},
    node_props: list = [],
    reroot_soma: bool = False,
    heal: bool = False,
    cap_holes: bool = False,
):
    """Decorate function such that Meshes are automatically skeletonized,
    the function is run on the skeleton and changes are propagated
    back to the meshe.

    Parameters
    ----------
    method :    str
                What to do with the results:
                  - 'subset': subset Mesh to what's left of the skeleton
                  - 'split': split Mesh following the skeleton's splits
                  - 'node_to_vertex': map the returned node ID to the vertex IDs
                  - 'node_properties' map node properties to vertices (requires
                    `node_props` parameter)
                  - 'pass_through' simply passes through the return value
    include_connectors : bool
                If True, will try to make sure that if the Mesh has
                connectors, they will be carried over to the skeleton.
    copy_properties : list
                Any additional properties that need to be copied from the
                skeleton to the mesh.
    disallowed_kwargs : dict
                Keyword arguments (name + value) that are not permitted when
                input is Mesh.
    node_props : list
                For method 'node_properties'. String must be column names in
                node table of skeleton.
    reroot_soma :  bool
                If True and neuron has a soma (.soma_pos), will reroot to
                that soma.
    heal :      bool
                Whether or not to heal the skeleton if the mesh is fragmented.
    cap_holes : bool
                For methods 'subset' and 'split': whether to triangulate the
                openings the cut leaves in the mesh. Off by default because it
                can double the cost of the subset; callers who want it can reach
                for [`navis.fill_holes`][] afterwards.

    """
    assert isinstance(copy_properties, list)
    assert isinstance(disallowed_kwargs, dict)
    assert isinstance(node_props, list)

    allowed_methods = (
        "subset",
        "node_to_vertex",
        "split",
        "node_properties",
        "pass_through",
    )
    if method not in allowed_methods:
        raise ValueError(f'Unknown method "{method}"')

    if method == "node_properties" and not node_props:
        raise ValueError('Must provide `node_props` for method "node_properties"')

    def decorator(function):
        # Computed once at decoration time rather than per call: workers
        # re-enter this wrapper for every neuron, and `inspect.signature` is
        # expensive enough to dominate the per-neuron dispatch cost.
        sig = inspect.signature(function)

        @wraps(function)
        def wrapper(*args, **kwargs):
            try:
                fnname = function.__name__
            except BaseException:
                fnname = str(function)

            # First, we need to extract the neuron from args and kwargs
            if args:
                # If there are positional arguments, the first one is assumed to
                # be the input neuron
                x = args[0]
                args = args[1:]
                x_key = "__args"
            else:
                # If not, we need to look for the name of the first argument
                # in the signature
                x_key = list(sig.parameters.keys())[0]
                x = kwargs.pop(x_key, None)

            # Complain if we did not get what we expected
            if isinstance(x, type(None)):
                raise ValueError(
                    "Unable to identify the neurons for call"
                    f"{fnname}:\n {args}\n {kwargs}"
                )

            # If input not a Mesh, just pass through
            # Note delayed import to avoid circular imports and IMPORTANTLY
            # funky interactions with pickle/dill
            from .. import core

            if not isinstance(x, core.Mesh):
                return function(x, *args, **kwargs)

            # Check for disallowed kwargs
            for k, v in disallowed_kwargs.items():
                if k in kwargs and kwargs[k] == v:
                    raise ValueError(
                        f"{k}={v} is not allowed when input is Mesh(s)."
                    )

            # See if this is meant to be done inplace
            if "inplace" in kwargs:
                # First check keyword arguments
                inplace = kwargs["inplace"]
            elif "inplace" in sig.parameters:
                # Next check signatures default
                inplace = sig.parameters["inplace"].default
            else:
                # All things failing assume it's not inplace
                inplace = False

            # Now skeletonize
            sk = x.skeleton

            # Delayed import to avoid circular imports
            # Note that this HAS to be in the inner function otherwise
            # we get a weird error when pickling for parallel processing
            from .. import morpho

            if heal:
                sk = morpho.heal_skeleton(sk, method="LEAFS")

            if reroot_soma and sk.has_soma:
                sk = sk.reroot(sk.soma)

            if include_connectors and x.has_connectors and not sk.has_connectors:
                sk._connectors = x.connectors.copy()
                sk._connectors["node_id"] = sk.snap(
                    sk.connectors[["x", "y", "z"]].values
                )[0]

            # Apply function
            res = function(sk, *args, **kwargs)

            if method == "subset":
                # See which vertices we need to keep
                keep = np.isin(sk.vertex_map, res.nodes.node_id.values)

                x = morpho.subset_neuron(
                    x, keep, inplace=inplace, cap_holes=cap_holes
                )

                for p in copy_properties:
                    setattr(x, p, getattr(sk, p, None))
            elif method == "split":
                meshes = []
                for n in res:
                    # See which vertices we need to keep
                    keep = np.isin(sk.vertex_map, n.nodes.node_id.values)

                    meshes.append(
                        morpho.subset_neuron(
                            x, keep, inplace=False, cap_holes=cap_holes
                        )
                    )

                    for p in copy_properties:
                        setattr(meshes[-1], p, getattr(n, p, None))
                x = core.NeuronList(meshes)
            elif method == "node_to_vertex":
                x = np.where(sk.vertex_map == res)[0]
            elif method == "node_properties":
                for p in node_props:
                    node_map = sk.nodes.set_index("node_id")[p].to_dict()
                    vertex_props = np.array([node_map[n] for n in sk.vertex_map])
                    setattr(x, p, vertex_props)
            elif method == "pass_through":
                return res

            return x

        return wrapper

    return decorator
