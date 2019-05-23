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
import numpy as np
import pandas as pd

from .. import core, config, utils

__all__ = ['generate_colors', 'prepare_connector_cmap', 'prepare_colormap',
           'eval_color']

logger = config.logger


def generate_colors(N, color_space='RGB', color_range=1):
    """ Divides colorspace into N evenly distributed colors.

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


def map_colors(colors, objects, color_range=255):
    """ Maps color(s) onto list of objects.

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


def prepare_connector_cmap(x):
    """ Looks for "label" or "type" column in connector tables and generates
    a color for every unique type. See ``navis.set_default_connector_colors``.

    Returns
    -------
    dict
            Maps type to color. Will be empty if no types.
    """

    if isinstance(x, (core.NeuronList, core.TreeNeuron)):
        connectors = x.get('connectors', None)

        if not isinstance(connectors, pd.DataFrame):
            unique = []
        elif 'type' in connectors:
            unique = connectors.type.unique()
        elif 'label' in connectors:
            unique = connectors.label.unique()
        elif 'relation' in connectors:
            unique = connectors.label.unique()
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


def prepare_colormap(colors, skdata=None, dotprops=None, volumes=None,
                     use_neuron_color=False, color_range=255):
    """ Maps color(s) to neuron/dotprop colorlists.
    """

    # Prepare dummies in case either no skdata or no dotprops
    if isinstance(skdata, type(None)):
        skdata = core.NeuronList([])

    if isinstance(dotprops, type(None)):
        dotprops = core.Dotprops()
        dotprops['gene_name'] = []

    if isinstance(volumes, type(None)):
        volumes = np.array([])
    elif not isinstance(volumes, np.ndarray):
        volumes = np.array(volumes)

    colors_required = skdata.shape[0] + dotprops.shape[0] + volumes.shape[0]

    # If no colors, generate random colors
    if isinstance(colors, type(None)):
        if colors_required > 0:
            colors = []
            colors += generate_colors(colors_required - volumes.shape[0],
                                      color_space='RGB',
                                      color_range=color_range)
            colors += eval_color([getattr(v, 'color', (1, 1, 1)) for v in volumes],
                                 color_range=color_range)
        else:
            # If no neurons to plot, just return None
            # This happens when there is only a scatter plot
            return [None], [None], [None]
    else:
        colors = eval_color(colors, color_range=color_range)

    # In order to cater for duplicate skeleton IDs in skdata (e.g. from
    # splitting into fragments), we will not map skids to colors but instead
    # keep colors as a list. That way users can pass a simple list of colors.

    # If dictionary, map skids to dotprops gene names and neuron skeleton IDs
    dotprop_cmap = []
    neuron_cmap = []
    volumes_cmap = []
    if isinstance(colors, dict):
        # We will try to get the skid first as str, then as int
        neuron_cmap = [colors.get(s,
                                  colors.get(int(s),
                                             eval_color(config.default_color,
                                                        color_range=color_range)))
                       for s in skdata.uuid]
        dotprop_cmap = [colors.get(s,
                                   eval_color(config.default_color,
                                              color_range=color_range))
                        for s in dotprops.gene_name.values]
        volumes_cmap = [colors.get(s.getattr('name', None),
                                   eval_color((1, 1, 1, .5),
                                              color_range=color_range))
                        for s in volumes]
    # If list of colors
    elif isinstance(colors, (list, tuple, np.ndarray)):
        # If color is a single color, convert to list
        if all([isinstance(elem, numbers.Number) for elem in colors]):
            colors = [colors] * colors_required
        elif len(colors) < colors_required:
            raise ValueError(f'Need colors for {colors_required} neurons/'
                             f'dotprops, got {len(colors)}')
        elif len(colors) > colors_required:
            logger.debug(f'More colors than required: got {len(colors)}, '
                         f'needed {colors_required}')

        if skdata.shape[0]:
            neuron_cmap = [colors[i] for i in range(skdata.shape[0])]
        if dotprops.shape[0]:
            dotprop_cmap = [colors[i + skdata.shape[0]] for i in range(dotprops.shape[0])]
        if volumes.shape[0]:
            volumes_cmap = [colors[i + skdata.shape[0] + dotprops.shape[0]] for i in range(volumes.shape[0])]
    else:
        raise TypeError(f'Got colors of type "{type(colors)}"')

    # Override neuron cmap if we are supposed to use neuron colors
    if use_neuron_color:
        neuron_cmap = [n.getattr('color',
                                 eval_color(config.default_color,
                                            color_range))
                       for i, n in enumerate(skdata)]

    return neuron_cmap, dotprop_cmap, volumes_cmap


def eval_color(x, color_range=255):
    """ Helper to evaluate colors. Always returns tuples.
    """

    if color_range not in [1, 255]:
        raise ValueError('"color_range" must be 1 or 255')

    if isinstance(x, str):
        c = mcl.to_rgb(x)
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

    # Check if we need to convert
    if not any([v > 1 for v in c]) and color_range == 255:
        c = [int(v * 255) for v in c]
    elif any([v > 1 for v in c]) and color_range == 1:
        c = [v / 255 for v in c]

    return tuple(c)
