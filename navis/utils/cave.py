#    This script is part of navis (http://www.github.com/navis-org/navis).
#    Copyright (C) 2017 Philipp Schlegel
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

import functools

import numpy as np
import pandas as pd

from .. import config, core

# Set up logging
logger = config.get_logger(__name__)

# The skeleton-service methods that hand back skeletons. `get_skeleton` takes a
# single root ID and returns one skeleton; the other two take a list and return
# a `{root_id: skeleton}` dictionary.
_SINGLE = ("get_skeleton",)
_BULK = ("get_bulk_skeletons", "fetch_skeletons")


def patch_caveclient():
    """Monkey patch caveclient to return navis neurons.

    Patches the skeleton service, i.e. the `client.skeleton` methods that
    return skeletons: [`get_skeleton`, `get_bulk_skeletons`, `fetch_skeletons`].
    Each gains an `as_navis=True` keyword argument, plus a `*_navis` twin that
    always converts. See examples for details.

    Both of the service's `output_format`s are understood and, importantly,
    they are **not in the same units**: `"dict"` (the default) is in nanometres
    and `"swc"` is in micrometres. The resulting neurons have their `.units` set
    accordingly. The service's per-vertex compartment (`1` soma, `2` axon,
    `3` dendrite) lands in the node table's `label` column, where the rest of
    navis - [`navis.ivscc_features`][], `color_by="label"`, ... - expects
    to find it.

    Unlike [`navis.patch_cloudvolume`][] this can be run at any point: it
    patches the client *class*, so clients created earlier pick it up too.
    Patching twice is a no-op.

    Note that `get_bulk_skeletons` and `fetch_skeletons` return a limited number
    of skeletons per call (ten, at time of writing) and silently drop the rest -
    a truncated `NeuronList` looks a lot like a complete one, so navis warns
    when it gets back fewer skeletons than you asked for.

    See Also
    --------
    [`navis.patch_cloudvolume`][]
                Does the same for `cloud-volume`, which is what CAVE hands you
                meshes through (`client.info.segmentation_cloudvolume()`).

    Examples
    --------
    >>> import navis
    >>> import caveclient                                       # doctest: +SKIP
    >>> # Monkey patch caveclient
    >>> navis.patch_caveclient()                                # doctest: +SKIP
    >>> client = caveclient.CAVEclient('minnie65_public')       # doctest: +SKIP
    >>> root_id = 864691135975633475                            # doctest: +SKIP
    >>> # Fetch as navis neuron using the newly added method or ...
    >>> n = client.skeleton.get_skeleton_navis(root_id)         # doctest: +SKIP
    >>> # ... alternatively use the `as_navis` keyword argument
    >>> n = client.skeleton.get_skeleton(root_id, as_navis=True)  # doctest: +SKIP
    >>> type(n)                                                 # doctest: +SKIP
    <class 'navis.core.skeleton.Skeleton'>
    >>> # Several at a time come back as a NeuronList
    >>> nl = client.skeleton.get_bulk_skeletons_navis([root_id])  # doctest: +SKIP

    """
    try:
        from caveclient.skeletonservice import SkeletonClient
    except ModuleNotFoundError:
        logger.info("caveclient appears to not be installed?")
        return
    except ImportError:
        # Older caveclients did not have a skeleton service at all
        logger.info("This version of caveclient has no skeleton service.")
        return

    patched = []
    for name in _SINGLE + _BULK:
        func = getattr(SkeletonClient, name, None)
        # Be lenient about methods this caveclient version does not have
        if func is None:
            continue
        # Wrapping a wrapper would pop `as_navis` twice and convert twice
        if getattr(func, "_navis_patched", False):
            continue

        single = name in _SINGLE
        setattr(SkeletonClient, f"{name}_navis",
                return_navis(func, single=single, only_on_kwarg=False))
        setattr(SkeletonClient, name,
                return_navis(func, single=single, only_on_kwarg=True))
        patched.append(name)

    if patched:
        logger.info(f"caveclient successfully patched: {', '.join(patched)}")
    else:
        logger.info("caveclient already patched.")


def return_navis(func, single=False, only_on_kwarg=False):
    """Wrap a caveclient skeleton-service method.

    Parameters
    ----------
    func :          callable
                    Function/method to wrap.
    single :        bool
                    Whether `func` returns a single skeleton (as opposed to a
                    `{root_id: skeleton}` dictionary). Determines whether the
                    wrapper returns a `Skeleton` or a `NeuronList`.
    only_on_kwarg : bool
                    If True, will look for an `as_navis=True` (default=False)
                    keyword argument to determine if results should be converted
                    to navis neurons.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ret_navis = kwargs.pop("as_navis", False)
        res = func(*args, **kwargs)

        if only_on_kwarg and not ret_navis:
            return res

        if single:
            # `root_id` is the first argument after `self`
            root_id = kwargs.get("root_id", args[1] if len(args) > 1 else None)
            n = _to_navis(res, id=_as_id(root_id))
            if n is None:
                logger.warning(f"Unable to convert {type(res)} to a navis neuron.")
                return res
            return n

        if not isinstance(res, dict):
            logger.warning(f"Unable to convert {type(res)} to navis neurons.")
            return res

        neurons = []
        for k, v in res.items():
            n = _to_navis(v, id=_as_id(k))
            if n is None:
                logger.warning(
                    f"Skipped {k}: unable to convert {type(v)} to a navis neuron."
                )
            else:
                neurons.append(n)

        # The bulk endpoints cap the number of skeletons they return (ten, at
        # time of writing) and quietly drop the rest. That is easy to spot in a
        # dictionary and easy to miss in a NeuronList, so say something.
        n_asked = _n_requested(args, kwargs)
        if n_asked is not None and len(neurons) < n_asked:
            logger.warning(
                f"Got {len(neurons)} skeletons for {n_asked} root IDs. The bulk "
                "endpoints return a limited number per call - request fewer at "
                "a time, or use `generate_bulk_skeletons_async` for large sets."
            )

        return core.NeuronList(neurons)

    wrapper._navis_patched = True

    return wrapper


def _as_id(value):
    """Root IDs come back as strings from the bulk endpoints."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _n_requested(args, kwargs):
    """How many root IDs the caller asked for; `None` if we can't tell."""
    root_ids = kwargs.get("root_ids", args[1] if len(args) > 1 else None)
    try:
        return len(root_ids)
    except TypeError:
        return None


def _to_navis(data, id=None):
    """Convert a single skeleton-service payload into a navis Skeleton.

    Returns `None` if we don't recognize `data`.

    """
    # `output_format="swc"`: SWC columns, in micrometres
    if isinstance(data, pd.DataFrame):
        if not {"id", "parent", "x", "y", "z"}.issubset(data.columns):
            return None
        nodes = data.rename(
            columns={"id": "node_id", "parent": "parent_id", "type": "label"}
        )
        # The service hands these over as floats
        for col in ("node_id", "parent_id", "label"):
            if col in nodes.columns:
                nodes[col] = nodes[col].astype(int)
        return core.Skeleton(nodes, id=id, units="um")

    # `output_format="dict"`: meshparty-style edges + vertices, in nanometres
    if isinstance(data, dict) and "vertices" in data and "edges" in data:
        # Deferred: `navis.graph` imports `navis.utils`, so this can't be
        # a module-level import
        from .. import graph

        n = graph.edges2neuron(
            np.asarray(data["edges"]),
            vertices=np.asarray(data["vertices"]),
            id=id,
            units="nm",
        )
        # `edges2neuron` keeps the vertex indices as node IDs but may reorder
        # the node table, so index the per-vertex arrays by node ID
        order = n.nodes.node_id.values
        for col, key in (("radius", "radius"), ("label", "compartment")):
            if key in data:
                n.nodes[col] = np.asarray(data[key])[order]
        if data.get("root", None) is not None:
            n = graph.reroot_skeleton(n, int(data["root"]), inplace=False)
        return n

    return None
