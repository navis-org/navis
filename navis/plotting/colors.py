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

from typing import Union, List, Tuple, Optional, Dict, Any, Sequence, overload
from typing_extensions import Literal

from .. import core, config, utils

__all__ = ['generate_colors', 'prepare_connector_cmap', 'prepare_colormap',
           'eval_color', 'hex_to_rgb', 'vary_colors']

logger = config.logger

# Some definitions for mypy
RGB_color = Tuple[float, float, float]
RGBA_color = Tuple[float, float, float, float]
Str_color = str
ColorList = Sequence[Union[RGB_color, RGBA_color, Str_color]]
AnyColor = Union[RGB_color, RGBA_color, Str_color, ColorList]

def generate_colors(N: int,
                    color_space: Union[Literal['RGB'],
                                       Literal['Grayscale']] = 'RGB',
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

    # Make count_color an even number
    if N % 2 != 0:
        color_count = N + 1
    else:
        color_count = N

    colormap = []
    interval = 2 / color_count
    runs = int(color_count / 2)

    # Create first half with low brightness; second half with high brightness
    # and slightly shifted hue
    if color_space == 'RGB':
        for i in range(runs):
            # High brightness
            h = interval * i
            s = 1
            v = 1
            hsv = colorsys.hsv_to_rgb(h, s, v)
            colormap.append(tuple(v * color_range for v in hsv))

            # Lower brightness, but shift hue by half an interval
            h = interval * (i + 0.5)
            s = 1
            v = 0.5
            hsv = colorsys.hsv_to_rgb(h, s, v)
            colormap.append(tuple(v * color_range for v in hsv))
    elif color_space == 'Grayscale':
        h = 0
        s = 0
        for i in range(color_count):
            v = 1 / color_count * i
            hsv = colorsys.hsv_to_rgb(h, s, v)
            colormap.append(tuple(v * color_range for v in hsv))

    logger.debug(f'{color_count} random colors created: {colormap}')

    # Make sure we return exactly N colors
    return colormap[:N]


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
                               color_space='RGB',
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


def prepare_colormap(colors, neurons=None, dotprops=None, volumes=None,
                     alpha=None, use_neuron_color=False, color_range=255):
    """Map color(s) to neuron/dotprop colorlists."""
    # Prepare dummies in case either no neuron data, no dotprops or no volumes
    if isinstance(neurons, type(None)):
        neurons = core.NeuronList([])
    elif not isinstance(neurons, core.NeuronList):
        neurons = core.NeuronList((neurons))

    if isinstance(dotprops, type(None)):
        dotprops = core.Dotprops()
        dotprops['gene_name'] = []

    if isinstance(volumes, type(None)):
        volumes = np.array([])

    if not isinstance(volumes, np.ndarray):
        volumes = np.array(volumes)

    # Only dotprops and neurons REQUIRE a color
    # Volumes are second class citiziens here
    colors_required = neurons.shape[0] + dotprops.shape[0]

    if not colors_required and len(volumes) == 0:
        # If no neurons to plot, just return None
        # This happens when there is only a scatter plot
        return [None], [None], [None]

    # If no colors, generate random colors
    if isinstance(colors, type(None)):
        colors = []
        colors += generate_colors(colors_required,
                                  color_space='RGB',
                                  color_range=color_range)
        colors += [getattr(v, 'color', (1, 1, 1)) for v in volumes]

    # We need to parse once here to convert named colours to rgb
    colors = eval_color(colors, color_range=color_range)

    # If dictionary, map skids to dotprops gene names and neuron skeleton IDs
    dotprop_cmap = []
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

        dotprop_cmap = [colors.get(s, dc) for s in dotprops.gene_name.values]

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
        neuron_cmap = [colors(i/len(neurons)) for i in range(len(neurons))]
        dotprop_cmap = [colors(i/dotprops.shape[0]) for i in range(dotprops.shape[0])]

        # Colormaps are not applied to volumes
        volumes_cmap = [getattr(v, 'color', (.95, .95, .95, .1)) for v in volumes]
    # If list of colors
    elif isinstance(colors, (list, tuple, np.ndarray)):
        # If color is a single color, convert to list
        if all([isinstance(elem, numbers.Number) for elem in colors]):
            # Generate at least one color
            colors = [colors] * max(colors_required, 1)
        elif len(colors) < colors_required:
            raise ValueError(f'Need colors for {colors_required} neurons/'
                             f'dotprops, got {len(colors)}')
        elif len(colors) > colors_required:
            logger.debug(f'More colors than required: got {len(colors)}, '
                         f'needed {colors_required}')

        if neurons.shape[0]:
            neuron_cmap = [colors.pop(0) for i in range(neurons.shape[0])]
        if dotprops.shape[0]:
            dotprop_cmap = [colors.pop(0) for i in range(dotprops.shape[0])]
        if volumes.shape[0]:
            # Volume have their own color property as fallback
            volumes_cmap = []
            for v in volumes:
                if colors:
                    volumes_cmap.append(colors.pop(0))
                else:
                    volumes_cmap.append(getattr(v, 'color', (.8, .8, .8, .2)))
    else:
        raise TypeError(f'Unable to parse colors of type "{type(colors)}"')

    # Override neuron cmap if we are supposed to use neuron colors
    if use_neuron_color:
        neuron_cmap = [getattr(n, 'color', config.default_color)
                       for i, n in enumerate(neurons)]

    # If alpha is given, we will override all values
    if not isinstance(alpha, type(None)):
        neuron_cmap = [add_alpha(c, alpha) for c in neuron_cmap]
        dotprop_cmap = [add_alpha(c, alpha) for c in dotprop_cmap]
        volumes_cmap = [add_alpha(c, alpha) for c in volumes_cmap]

    # Make sure colour range checks out
    neuron_cmap = [eval_color(c, color_range=color_range)
                   for c in neuron_cmap]
    dotprop_cmap = [eval_color(c, color_range=color_range)
                    for c in dotprop_cmap]
    volumes_cmap = [eval_color(c, color_range=color_range)
                    for c in volumes_cmap]

    logger.debug('Neuron colormap: ' + str(neuron_cmap))
    logger.debug('Dotprops colormap: ' + str(dotprop_cmap))
    logger.debug('Volumes colormap: ' + str(volumes_cmap))

    return neuron_cmap, dotprop_cmap, volumes_cmap


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
