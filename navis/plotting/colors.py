#    This script is part of navis (http://www.github.com/schlegelp/navis).
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

""" Module contains functions to manage colours.
"""

import colorsys
import numbers

import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Union, List, Tuple, Optional, Dict, Any, Sequence, overload
from typing_extensions import Literal

from .. import core, config, utils, morpho

__all__ = ['generate_colors', 'prepare_connector_cmap', 'prepare_colormap',
           'eval_color', 'hex_to_rgb', 'vary_colors', 'vertex_colors']

logger = config.logger

# Some definitions for mypy
RGB_color = Tuple[float, float, float]
RGBA_color = Tuple[float, float, float, float]
Str_color = str
ColorList = Sequence[Union[RGB_color, RGBA_color, Str_color]]
AnyColor = Union[RGB_color, RGBA_color, Str_color, ColorList]


def generate_colors(N: int,
                    palette: str = 'hls',
                    color_range: Union[Literal[1],
                                       Literal[255]] = 1
                    ) -> List[Tuple[float, float, float]]:
    """Divide colorspace into N evenly distributed colors.

    Returns
    -------
    colormap :  list
                [(r, g, b), (r, g, b), ...]

    """
    if N == 1:
        return [eval_color(config.default_color, color_range)]
    elif N == 0:
        return []

    if not isinstance(palette, str):
        palette = 'hls'

    colormap = sns.color_palette(palette, N)

    if color_range == 255:
        colormap = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colormap]

    return colormap


def map_colors(colors: Optional[Union[str,
                                      Tuple[float, float, float],
                                      Dict[Any, str],
                                      Dict[Any, Tuple[float, float, float]],
                                      List[Union[str,
                                                 Tuple[float, float, float]]
                                           ]
                                      ]
                                ],
               objects: Sequence[Any],
               color_range: Union[Literal[1], Literal[255]] = 255
               ) -> List[Tuple[float, float, float]]:
    """Map color(s) onto list of objects.

    Parameters
    ----------
    colors :        None | str | tuple | list-like | dict | None
                    Color(s) to map onto ``objects``. Can be::

                      str: e.g. "blue", "k" or "y"
                      tuple: (0, 0, 1), (0, 0, 0) or (0, 1, 1)
                      list-like of the above: [(0, 0, 1), 'r', 'k', ...]
                      dict mapping objects to colors: {object1: 'r',
                                                       object2: (1, 1, 1)}

                    If list-like or dict do not cover all ``objects``, will
                    fall back to ``navis.config.default_color``. If ``None``,
                    will generate evenly spread out colors.

    objects :       list-like
                    Object(s) to map color onto.
    color_range :   int, optional

    Returns
    -------
    list of tuples
                    Will match length of ``objects``.

    """
    if not utils.is_iterable(objects):
        objects = [objects]

    # If no colors, generate random colors
    if isinstance(colors, type(None)):
        if len(objects) == 1:
            return [eval_color(config.default_color, color_range)]
        return generate_colors(len(objects),
                               color_range=color_range)

    # Bring colors in the right space
    colors = eval_color(colors, color_range=color_range)

    # Match them to objects
    if isinstance(colors, dict):
        # If dict, try mapping to objects
        if set(objects) - set(colors.keys()):
            logger.warning('Objects w/o colors - falling back to default.')
        return [colors.get(o, config.default_color) for o in objects]
    elif isinstance(colors, tuple):
        # If single color map to each object
        return [colors] * len(objects)
    elif isinstance(colors, list):
        # If list of correct length, map onto objets
        if len(colors) != len(objects):
            logger.warning('N colours does not match N objects.')
        miss = len(objects) - len(colors) if len(objects) > len(colors) else 0
        return colors[: len(objects)] + [config.default_color] * miss
    else:
        raise TypeError(f'Unable to interpret colors of type "{type(colors)}"')


def prepare_connector_cmap(x) -> Dict[str, Tuple[float, float, float]]:
    """Look for "label" or "type" column in connector tables and generates
    a color for every unique type. See ``navis.set_default_connector_colors``.

    Returns
    -------
    dict
            Maps type to color. Will be empty if no types.

    """
    if isinstance(x, (core.NeuronList, core.TreeNeuron)):
        connectors = x.get('connectors', None)

        if not isinstance(connectors, pd.DataFrame):
            unique: List[str] = []
        elif 'type' in connectors:
            unique = connectors.type.unique()
        elif 'label' in connectors:
            unique = connectors.label.unique()
        elif 'relation' in connectors:
            unique = connectors.relation.unique()
        else:
            unique = []
    else:
        unique = list(set(x))

    colors = config.default_connector_colors
    if isinstance(colors, (list, np.ndarray)):
        if len(unique) > len(colors):
            raise ValueError('Must define more default connector colors. See'
                             'navis.set_default_connector_colors')

        return {t: config.default_connector_colors[i] for i, t in enumerate(unique)}
    elif isinstance(colors, dict):
        miss = [l for l in unique if l not in colors]
        if miss:
            raise ValueError(f'Connector labels/types {",".join(miss)} are not'
                             ' defined in default connector colors. '
                             'See navis.set_default_connector_colors')
        return colors
    else:
        raise TypeError('config.default_color must be dict or iterable, '
                        f'not {type(config.default_color)}')


def vertex_colors(neurons, by, palette, alpha=False, vmin=None, vmax=None,
                  na='raise', norm_global=True, color_range=255):
    """Generate a color and/or alpha values for each node/face/point of a neuron.

    Parameters
    ----------
    neurons :   NeuronList | Neuron | pandas.DataFrame
                Neurons to generate colors for.
    by :        str | iterable | list of iterables
                Must provide a vector for each node/face of a neuron or map to
                a column in node table. Data can be numerical or categorical.
    palette :   str | list of colors | dict
                Name of a matplotlib or seaborn color palette, list of colors
                or (for caterogical) data a dict mapping colors to values. If
                data is numerical must be a matplotlib palette.
    alpha :     bool
                If True will also map the alpha channel. Applies only if data
                is numerical.
    vmin|vmax : float, optional
                Min/Max values for normalizing numerical data.
    na :        "raise" | color
                Determine what to do if ``by`` is missing for a given neuron or
                a node:
                 - "raise" will raise ValueError
                 - color (str, rgb tuple) will be used to fill missing values
    norm_global : bool
                If True and no vmin/vmax is provided, will normalize across
                all ``neurons``. If False, will normalize neurons individually.

    Returns
    -------
    List of (N, 4) arrays
                One list per neuron. Each array contains a color for each of the
                N faces/nodes.

    """
    if not isinstance(neurons, core.NeuronList):
        neurons = core.NeuronList(neurons)

    if not isinstance(palette, (str, dict)) and not utils.is_iterable(palette):
        raise TypeError('Expected palette as name (str), list of '
                        f'colors or dictionary, got "{type(palette)}"')

    # If by points to column collect values
    if isinstance(by, str):
        # Make sure we are dealing only with TreeNeurons
        if not all([isinstance(n, core.TreeNeuron) for n in neurons]):
            raise TypeError('Can only generate colors from a column if all '
                            'neurons are TreeNeurons.')

        # For convenience we will compute this if required
        if by == 'strahler_index':
            for n in neurons:
                if 'strahler_index' not in n.nodes:
                    _ = morpho.strahler_index(n)

        values = []
        for n in neurons:
            # If column exists add to values
            if by in n.nodes.columns:
                values.append(n.nodes[by].values)
            elif na == 'raise':
                raise ValueError(f'Column {by} does not exists in neuron {n.id}')
            # If column does not exists, add a bunch of NaNs - we will worry
            # about it later
            else:
                values.append(np.repeat(np.nan, n.nodes.shape[0]))
    # If by already contains the actual values
    else:
        # Make sure values are list of lists (in case we started with a single
        # neuron)
        if len(neurons) == 1 and len(by) != len(neurons):
            values = [by]
        else:
            values = by

    # At this point we expect to have values for each neuron
    if len(values) != len(neurons):
        raise ValueError(f'Got {len(values)} values for {len(neurons)} neurons.')

    # We also expect to have a value for every single node/face
    for n, v in zip(neurons, values):
        if isinstance(n, core.TreeNeuron):
            if len(v) != n.n_nodes:
                raise ValueError(f'Got {len(v)} for {neurons.n_nodes} nodes '
                                 f'for neuron {n.id}')
        elif isinstance(n, core.MeshNeuron):
            if len(v) != n.n_faces:
                raise ValueError(f'Got {len(v)} for {neurons.n_faces} faces '
                                 f'for neuron {n.id}')
        else:
            raise TypeError(f'Unable to map colors for neurons of type {type(n)}')

    # Now check for NaNs
    has_nan = False
    for v in values:
        if any(pd.isnull(v)):
            has_nan = True
            break

    if has_nan:
        if na == 'raise':
            raise ValueError('Values contain NaNs.')
        else:
            # Make sure na is a valid color
            try:
                na = mcl.to_rgba(na, alpha=1)
            except ValueError:
                raise ValueError('`na` must be either "raise" or a valid color '
                                 f'to replace NA values. Unable to convert {na}'
                                 ' to a color.')

    # First check if data is numerical or categorical
    is_num = [utils.is_numeric(a, bool_numeric=False) for a in values]
    # If numerical
    if all(is_num):
        # Get min/max values
        if not vmin:
            vmin = [np.nanmin(v) for v in values]

            if norm_global:
                vmin = np.repeat(np.min(vmin), len(values))
        else:
            vmin = np.repeat(vmin, len(values))

        if not vmax:
            vmax = [np.nanmax(v) for v in values]

            if norm_global:
                vmax = np.repeat(np.max(vmax), len(values))
        else:
            vmax = np.repeat(vmax, len(values))

        # Normalize values
        values = [(np.asarray(v) - mn) / (mx - mn) for v, mn, mx in zip(values, vmin, vmax)]

        # Get the colormap
        if not isinstance(palette, str):
            raise TypeError('Expected name of matplotlib colormap for numerical'
                            f' data, got {type(palette)}')
        cmap = plt.get_cmap(palette)
        colors = []
        for v in values:
            c = np.zeros((len(v), 4))
            if any(pd.isnull(v)):
                c[pd.isnull(v), :] = na
            c[~pd.isnull(v), :] = cmap(v[~pd.isnull(v)], alpha=1)

            if color_range == 255:
                c[:, :3] = (c[:, :3] * 255).astype(int)

            # Add alpha - note that we slightly clip the value to prevent
            # any color from being entirely invisible
            if alpha:
                c[:, 3] = np.clip(v + 0.05, a_max=1, a_min=0)

            colors.append(c)
    # We don't want to deal with mixed data
    elif any(is_num):
        raise ValueError('Data appears to be mixed numeric and non-numeric.')
    else:
        # Find unique values
        unique_v = np.unique([np.unique(v) for v in values])

        if isinstance(palette, str):
            palette = sns.color_palette(palette, len(unique_v))

        if not isinstance(palette, dict):
            if len(palette) != len(unique_v):
                raise ValueError(f'Got {len(palette)} colors for '
                                 f'{len(unique_v)} unique values.')
            palette = dict(zip(unique_v, palette))

        # Check if dict palette contains all possible values
        miss = [v for v in unique_v if v not in palette]
        if any(miss):
            raise ValueError('Value(s) missing from palette dictionary: '
                             ', '.join(miss))

        # Make sure colors are what we need
        palette = {v: mcl.to_rgba(c, alpha=1) for v, c in palette.items()}

        # Alpha values doesn't exactly make sense for categorical data but
        # who am I to judge? We will simply use the alphanumerical order.
        alpha_map = {v: (i + 1)/(len(palette) + 1) for i, v in enumerate(palette.keys())}

        colors = []
        for v in values:
            c = [palette.get(x, na) for x in v]
            c = np.array(c)

            if color_range == 255:
                c[:, :3] = (c[:, :3] * 255).astype(int)

            if alpha:
                c[:, 3] = [alpha_map.get(x, 0) for x in v]

            colors.append(c)

    return colors


def prepare_colormap(colors, neurons=None, volumes=None, alpha=None,
                     palette=None, color_range=255):
    """Map color(s) to neuron/dotprop colorlists."""
    # Prepare dummies in case either no neuron data, no dotprops or no volumes
    if isinstance(neurons, type(None)):
        neurons = core.NeuronList([])
    elif not isinstance(neurons, core.NeuronList):
        neurons = core.NeuronList((neurons))

    if isinstance(volumes, type(None)):
        volumes = np.array([])

    if not isinstance(volumes, np.ndarray):
        volumes = np.array(volumes)

    # Only neurons REQUIRE a color
    # Volumes are second class citiziens here :(
    colors_required = neurons.shape[0]

    if not colors_required and not len(volumes):
        # If no neurons to plot, just return None
        # This happens when there is only a scatter plot
        return [None], [None]

    # If no colors, generate random colors
    if isinstance(colors, type(None)):
        colors = []
        colors += generate_colors(colors_required,
                                  palette=palette,
                                  color_range=color_range)
        colors += [getattr(v, 'color', (1, 1, 1)) for v in volumes]

    # We need to parse once here to convert named colours to rgb
    colors = eval_color(colors, color_range=color_range)

    # If dictionary, map colors to neuron IDs
    neuron_cmap = []
    volumes_cmap = []
    dc = config.default_color
    if isinstance(colors, dict):
        # Try finding color first by neuron, then uuid and finally by name
        neuron_cmap = []
        for n in neurons:
            this_c = dc
            for k in [n, n.id, n.name]:
                if k in colors:
                    this_c = colors[k]
                    break
            neuron_cmap.append(this_c)

        # Try finding color first by volume, then uuid and finally by name
        # If no color found, fall back to color property
        volumes_cmap = []
        for v in volumes:
            this_c = getattr(v, 'color', (.95, .95, .95, .1))
            for k in [v, v.id, getattr(v, 'name', None)]:
                if k and k in colors:
                    this_c = colors[k]
                    break
            volumes_cmap.append(this_c)
    elif isinstance(colors, mcl.Colormap):
        # Generate colors for neurons and dotprops
        neuron_cmap = [colors(i / len(neurons)) for i in range(len(neurons))]

        # Colormaps are not applied to volumes
        volumes_cmap = [getattr(v, 'color', (.95, .95, .95, .1)) for v in volumes]
    # If list of colors
    elif isinstance(colors, (list, tuple, np.ndarray)):
        # If color is a single color, convert to list
        if all([isinstance(elem, numbers.Number) for elem in colors]):
            # Generate at least one color
            colors = [colors] * max(colors_required, 1)

        if len(colors) < colors_required:
            raise ValueError(f'Need colors for {colors_required} neurons, '
                             f'got {len(colors)}')
        elif len(colors) > colors_required:
            logger.debug(f'More colors than required: got {len(colors)}, '
                         f'needed {colors_required}')

        if len(neurons):
            neuron_cmap = [colors.pop(0) for i in range(neurons.shape[0])]

        if len(volumes):
            # Volume have their own color property as fallback
            volumes_cmap = []
            for v in volumes:
                if colors:
                    volumes_cmap.append(colors.pop(0))
                else:
                    volumes_cmap.append(getattr(v, 'color', (.8, .8, .8, .2)))
    else:
        raise TypeError(f'Unable to parse colors of type "{type(colors)}"')

    # If alpha is given, we will override all values
    if not isinstance(alpha, type(None)):
        neuron_cmap = [add_alpha(c, alpha) for c in neuron_cmap]

        # Only apply to volumes if there aren't any neurons
        if not neuron_cmap:
            volumes_cmap = [add_alpha(c, alpha) for c in volumes_cmap]

    # Make sure colour range checks out
    neuron_cmap = [eval_color(c, color_range=color_range)
                   for c in neuron_cmap]
    volumes_cmap = [eval_color(c, color_range=color_range)
                    for c in volumes_cmap]

    logger.debug('Neuron colormap: ' + str(neuron_cmap))
    logger.debug('Volumes colormap: ' + str(volumes_cmap))

    return neuron_cmap, volumes_cmap


def add_alpha(c, alpha):
    """Add/adjust alpha for color."""
    return (c[0], c[1], c[2], alpha)


def eval_color(x, color_range=255, force_alpha=False):
    """Evaluate colors and return tuples."""
    if color_range not in [1, 255]:
        raise ValueError('"color_range" must be 1 or 255')

    if isinstance(x, str):
        # Check if named color
        if mcl.is_color_like(x):
            c = mcl.to_rgb(x)
        # Assume it's a matplotlib color map
        else:
            try:
                c = plt.get_cmap(x)
            except ValueError:
                raise ValueError(f'Unable to interpret color "{x}"')
            except BaseException:
                raise
    elif isinstance(x, dict):
        return {k: eval_color(v, color_range=color_range) for k, v in x.items()}
    elif isinstance(x, (list, tuple, np.ndarray)):
        # If is this is not a list of RGB values:
        if any([not isinstance(elem, numbers.Number) for elem in x]):
            return [eval_color(c, color_range=color_range) for c in x]
        # If this is a single RGB color:
        c = x
    elif isinstance(x, type(None)):
        return None
    else:
        raise TypeError(f'Unable to interpret color of type "{type(x)}"')

    if not isinstance(c, mcl.Colormap):
        # Check if we need to convert
        if not any([v > 1 for v in c[:3]]) and color_range == 255:
            c = np.array(c, dtype=float)
            c[:3] = (c[:3] * 255).astype(int)
        elif any([v > 1 for v in c[:3]]) and color_range == 1:
            c = np.array(c, dtype=float)
            c[:3] = c[:3] / 255

        c = tuple(c)

    if force_alpha and len(c) == 3:
        c = (c[0], c[1], c[2], 1)

    return c


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    """Convert hex to rgb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))  # type: ignore


def vary_colors(color: AnyColor,
                by_max: float = .1) -> np.ndarray:
    """Add small variance to color."""
    if isinstance(color, str):
        color = mcl.to_rgb(color)

    if not isinstance(color, np.ndarray):
        color = np.array(color)

    if color.ndim == 1:
        color = color.reshape(1, color.shape[0])

    variance = (np.random.randint(0, 100, color.shape) / 100) * by_max
    variance = variance - by_max / 2

    # We need to make sure color is array of floats
    color = color.astype(float)
    color[:, :3] = color[:, :3] + variance[:, :3]

    return np.clip(color, 0, 1)
