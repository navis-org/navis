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

"""Settings resolution shared by the plotting backends.

Everything in here turns a user-facing setting into the value a renderer
actually needs - which connectors to draw, what color they get, where the
somata are. None of it touches a backend API, which is what lets matplotlib,
plotly and k3d all go through the same code instead of keeping three copies
that drift apart.

Nothing here is public: these are called by `dd.py`, `plotly/graph_objs.py`
and `k3d/k3d_objects.py`.

"""

import copy

from collections import namedtuple

import numpy as np
import pandas as pd

from .. import config, core, utils
from .colors import vertex_colors

logger = config.get_logger(__name__)

# At this many somata we assume soma detection went wrong and skip them:
# rendering hundreds of spheres will freeze the session, and they're not real.
SOMA_COUNT_LIMIT = 10

#: One soma to render.
SomaSpec = namedtuple("SomaSpec", ["center", "radius", "color"])


def use_radius(neuron, settings) -> bool:
    """Whether this neuron should be rendered with radius.

    Note that we must not cache this decision on the settings: with
    `radius="auto"` it has to be made for each neuron individually.

    """
    if not isinstance(neuron, core.TreeNeuron) or not settings.radius:
        return False

    if settings.radius == "auto":
        # Number of nodes with radii
        n_radii = (neuron.nodes.get("radius", pd.Series([])).fillna(0) > 0).sum()
        # If less than 30% of nodes have a radius, we fall back to lines
        return (n_radii / neuron.nodes.shape[0]) >= 0.3

    return True


def resolve_connectors(neuron, settings):
    """The connectors of `neuron` that `settings` asks to draw.

    `connectors` is either something truthy for all of them,
    `"pre"`/`"presynapses"` or `"post"`/`"postsynapses"` for the two shorthands,
    any other string for an exact match against the `type` column, or an
    iterable of types.

    Parameters
    ----------
    neuron :    TreeNeuron | MeshNeuron | Dotprops
    settings :  BasePlottingSettings

    Returns
    -------
    pandas.DataFrame
                Empty if nothing should be drawn, so callers can go straight to
                `.groupby("type")` without a guard of their own.

    """
    which = settings.connectors

    # Careful: `bool(which)` raises "truth value is ambiguous" for an array of
    # connector types, so test emptiness rather than truthiness.
    wanted = settings.connectors_only or (
        which.size > 0 if isinstance(which, np.ndarray) else bool(which)
    )
    if not wanted or not neuron.has_connectors:
        return pd.DataFrame(columns=["type", "x", "y", "z", "node_id"])

    # Strings must be checked before iterables: comparing a multi-element array
    # against the shorthands raises "truth value is ambiguous" too.
    if isinstance(which, str):
        if which in ("pre", "presynapses"):
            return neuron.presynapses
        if which in ("post", "postsynapses"):
            return neuron.postsynapses
        return neuron.connectors[neuron.connectors.type == which]

    if utils.is_iterable(which):
        return neuron.connectors[neuron.connectors.type.isin(list(which))]

    return neuron.connectors


def resolve_cn_layout(settings):
    """Connector layout (colors, sizes, display mode) with `cn_layout` merged in.

    Deep-copied so that a caller's `cn_layout` can't leak into the global
    defaults through the nested per-type dicts.

    """
    layout = copy.deepcopy(config.default_connector_colors)
    if settings.cn_layout:
        layout.update(settings.cn_layout)
    return layout


def resolve_cn_color(cn_type, layout, neuron_color, settings):
    """Color for a single connector type.

    Precedence, highest first:

    1. `cn_mesh_colors=True` or `cn_colors="neuron"` - the neuron's own color
    2. `cn_colors` - either a dict keyed by connector type (which may cover
       only some of them) or a single color for all types
    3. the layout's default for this type

    Parameters
    ----------
    cn_type :       str
                    Value from the connector table's `type` column.
    layout :        dict
                    As returned by `resolve_cn_layout`.
    neuron_color :  color
                    The color this connector's neuron was given.
    settings :      BasePlottingSettings

    """
    cn_colors = settings.cn_colors

    if settings.cn_mesh_colors or (
        isinstance(cn_colors, str) and cn_colors == "neuron"
    ):
        return neuron_color

    # Note `np.size` rather than plain truthiness: a multi-element rgb array
    # would raise "truth value is ambiguous".
    if isinstance(cn_colors, dict):
        if cn_type in cn_colors:
            return cn_colors[cn_type]
    elif cn_colors is not None and np.size(cn_colors):
        return cn_colors

    # Fallback for a connector type navis has no default for. Note this is in
    # the same 0-1 range as `config.default_connector_colors`, which matplotlib
    # requires; plotly and k3d scale it up via `eval_color(color_range=255)`.
    return layout.get(cn_type, {"color": (0.04, 0.04, 0.04)})["color"]


def resolve_somata(neuron, color, settings):
    """Yield a [`SomaSpec`][] for each soma that should be rendered.

    Between them the backends managed to get every part of this wrong at least
    once, hence doing it in one place: a runaway soma detection (which used to
    freeze the session), `None` entries, a radius column that is missing or
    NaN, and picking the right entry out of a per-node colormap.

    Parameters
    ----------
    neuron :    TreeNeuron
    color :     color | np.ndarray
                The neuron's color as it came out of the colormap. A 2d array
                is taken to be one color per node and indexed accordingly;
                anything else is used for every soma as-is.
    settings :  BasePlottingSettings

    Yields
    ------
    SomaSpec

    """
    # `.soma` is an uncached property that re-runs soma detection on every read,
    # so grab it once - this used to cost ~1ms per neuron.
    soma = getattr(neuron, "soma", None)
    if not settings.soma or isinstance(soma, type(None)):
        return

    somata = utils.make_iterable(soma)

    if len(somata) >= SOMA_COUNT_LIMIT:
        logger.warning(
            f"Neuron {neuron.id} appears to have {len(somata)} somas. "
            "That does not look right - will ignore them for plotting."
        )
        return

    # One color per node: we need to find the entry for the soma node itself
    per_node = isinstance(color, np.ndarray) and color.ndim == 2

    nodes = neuron.nodes.set_index("node_id")
    for s in somata:
        if isinstance(s, type(None)):
            continue

        if per_node:
            soma_color = color[np.where(neuron.nodes.node_id == s)[0][0]]
        else:
            soma_color = color

        n = nodes.loc[s]
        r = (
            getattr(n, neuron.soma_radius)
            if isinstance(neuron.soma_radius, str)
            else neuron.soma_radius
        )

        # The radius column may be missing altogether or hold NaNs - either way
        # there is no sphere to draw.
        if pd.isnull(r):
            logger.warning(
                f"Skipping soma {s} of neuron {neuron.id} "
                "because it appears to have no radius."
            )
            continue

        yield SomaSpec(
            center=np.array([n.x, n.y, n.z], dtype=float),
            radius=r,
            color=soma_color,
        )


def apply_shade_by(colormap, neurons, settings, color_range):
    """Merge `shade_by` into the alpha channel of an existing colormap.

    `shade_by` maps a per-node/vertex property onto transparency, so neurons
    that had a single flat color come out of here with one color per node.

    Parameters
    ----------
    colormap :      list
                    One entry per neuron, as returned by `prepare_colormap`.
    neurons :       NeuronList
    settings :      BasePlottingSettings
    color_range :   1 | 255
                    Range the backend expects its colors in.

    Returns
    -------
    list
                    `colormap` unchanged if `shade_by` is not set.

    """
    if isinstance(settings.shade_by, type(None)):
        return colormap

    alphamap = vertex_colors(
        neurons,
        by=settings.shade_by,
        use_alpha=True,
        palette="viridis",  # irrelevant - we only keep the alpha channel
        norm_global=settings.norm_global,
        vmin=settings.smin,
        vmax=settings.smax,
        na="raise",
        color_range=color_range,
    )

    new_colormap = []
    for c, a in zip(colormap, alphamap):
        if not (isinstance(c, np.ndarray) and c.ndim == 2):
            c = np.tile(c, (a.shape[0], 1))

        # With `color_range=255` the colors are integers, which can't hold alpha
        if c.dtype not in (np.float16, np.float32, np.float64):
            c = c.astype(np.float16)

        if c.shape[1] == 4:
            c[:, 3] = a[:, 3]
        else:
            c = np.insert(c, 3, a[:, 3], axis=1)

        new_colormap.append(c)

    return new_colormap
