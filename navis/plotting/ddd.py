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

""" Module contains functions to plot neurons in 2D and 3D.
"""

import warnings

import plotly.offline
import plotly.graph_objs as go
import numpy as np

from typing import Union, List, Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import vispy

from .. import utils, config, core
from .vispy.viewer import Viewer
from .colors import prepare_colormap
from .plotly.graph_objs import *

if not config.headless:
    try:
        # Try setting vispy backend to PyQt5
        vispy.use(app='PyQt5')
    except BaseException:
        pass

__all__ = ['plot3d']

logger = config.logger

#if utils.is_jupyter():
#    plotly.offline.init_notebook_mode(connected=True)


def plot3d(x: Union[core.NeuronObject,
                    core.Volume,
                    np.ndarray,
                    List[Union[core.NeuronObject, np.ndarray, core.Volume]]
                    ],
           **kwargs) -> Optional[Union[Viewer, dict]]:
    """Generate 3D plot.

    Uses either `vispy <http://vispy.org>`_ or `plotly <http://plot.ly>`_.
    By default, the choise is automatic and depends on context::

      terminal: vispy used
      Jupyter: plotly used

    See ``backend`` parameter on how to overwrite this default behavior.

    Parameters
    ----------
    x :               Neuron/List | Volume | numpy.array
                        - ``numpy.array (N,3)`` is plotted as scatter plot
                        - multiple objects can be passed as list (see examples)
    backend :         'auto' | 'vispy' | 'plotly', default='auto'
                        - ``auto`` selects backend based on context: ``vispy``
                          for terminal and ``plotly`` for Jupyter environments.
                        - ``vispy`` uses OpenGL to generate high-performance
                          3D plots. Works in terminals.
                        - ``plotly`` generates 3D plots using WebGL. Works in
                          Jupyter notebooks. For Jupyter lab, you need to
                          have the Plotly labextension installed.
    connectors :      bool, default=False
                      Plot connectors (e.g. synapses) if available.
    color :           None | str | tuple | list | dict, default=None
                      Use single str (e.g. ``'red'``) or ``(r, g, b)`` tuple
                      to give all neurons the same color. Use ``list`` of
                      colors to assign colors: ``['red', (1, 0, 1), ...].
                      Use ``dict`` to map colors to neurons:
                      ``{uuid: (r, g, b), ...}``.
    cn_colors :       str | tuple | dict | "neuron"
                      Overrides the default colors:

                      - single color as str (e.g. ``'red'``) or rgb tuple (e.g.
                        ``(1, 0, 0)``)
                      - dict mapping the connectors tables ``type`` column to a
                        color (e.g. `{0: (1, 0, 0)}`)
                      - with "neuron", connectors will receive the same color as
                        their neuron
    palette :         str | array | list of arrays, default=None
                      Name of a matplotlib or seaborn palette. If ``color`` is
                      not specified will pick colors from this palette.
    color_by :        str | array | list of arrays, default = None
                      Can be the name of a column in the node table of
                      ``TreeNeurons`` or an array of (numerical or categorical)
                      values for each node. Numerical values will be normalized.
                      You can control the normalization by passing a ``vmin``
                      and/or ``vmax`` parameter.
    shade_by :        str | array | list of arrays, default=None
                      Similar to ``color_by`` but will affect only the alpha
                      channel of the color. If ``shade_by='strahler'`` will
                      compute Strahler order if not already part of the node
                      table (TreeNeurons only). Numerical values will be
                      normalized. You can control the normalization by passing
                      a ``smin`` and/or ``smax`` parameter.
    radius :          bool, default=False
                      If True, will plot TreeNeurons as 3D tubes using the
                      ``radius`` column in their node tables.
    width/height :    int, default=600
                      Use to define figure/window size.
    scatter_kws :     dict, optional
                      Use to modify scatter plots. Accepted parameters are:

                      - ``size`` to adjust size of dots
                      - ``color`` to adjust color
    soma :            bool, default=True
                      Whether to plot soma if it exists (TreeNeurons only). Size
                      of the soma is determined by the neuron's ``.soma_radius``
                      property which defaults to the "radius" column for
                      ``TreeNeurons``.

                      ``Below parameters are for plotly backend only:``
    fig :             plotly.graph_objs.Figure
                      Pass to add graph objects to existing plotly figure. Will
                      not change layout.
    title :           str, default=None
                      For plotly only! Change plot title.
    fig_autosize :    bool, default=False
                      For plotly only! Autoscale figure size.
                      Attention: autoscale overrides width and height
    plotly_inline :   bool, default=True
                      If True and you are in an Jupyter environment, will
                      render plotly plots inline. Else, will generate a
                      plotly figure dictionary that can be used to generate
                      a html with an embedded 3D plot.
    hover_name :      bool, default=False
                      If True, hovering over neurons will show their label.
    hover_id :        bool, default=False
                      If True, hovering over skeleton nodes will show their ID.
    legend_group :    dict, default=None
                      A dictionary mapping neuron IDs to labels (strings).
                      Use this to group neurons under a common label in the
                      legend.

                      ``Below parameters are for vispy backend only:``
    clear :           bool, default = False
                      If True, will clear the viewer before adding the new
                      objects.
    center :          bool, default = True
                      If True, will center camera on the newly added objects.

    Returns
    -------
    If ``backend='vispy'``

        Opens a 3D window and returns :class:`navis.Viewer`.

    If ``backend='plotly'``

        Returns either ``None`` if you are in a Jupyter notebook (see
        ``plotly_inline`` parameter) or a ``fig`` dictionary to generate
        plotly 3D figure (see examples).

    See Also
    --------
    :class:`navis.Viewer`
        Interactive vispy 3D viewer. Makes it easy to add/remove/select
        objects.


    Examples
    --------
    >>> import navis

    In a Jupyter notebook using plotly as backend.

    >>> import plotly.offline
    >>> plotly.offline.init_notebook_mode()                     # doctest: +SKIP
    >>> nl = navis.example_neurons()
    >>> # Backend is automatically chosen but we can set it explicitly
    >>> # Plot inline
    >>> nl.plot3d(backend='plotly')                             # doctest: +SKIP
    >>> # Plot as separate html in a new window
    >>> fig = nl.plot3d(backend='plotly', plotly_inline=False)
    >>> _ = plotly.offline.plot(fig)                            # doctest: +SKIP

    In a terminal using vispy as backend.

    >>> # Plot list of neurons
    >>> nl = navis.example_neurons()
    >>> v = navis.plot3d(nl, backend='vispy')
    >>> # Clear canvas
    >>> navis.clear3d()

    Some more advanced examples (using vispy here but also works with plotly).

    >>> # plot3d() can deal with combinations of objects
    >>> nl = navis.example_neurons()
    >>> vol = navis.example_volume('LH')
    >>> vol.color = (255, 0, 0, .5)
    >>> # This plots a neuronlists, a single neuron and a volume
    >>> v = navis.plot3d([nl[0:2], nl[3], vol])
    >>> # Pass kwargs
    >>> v = navis.plot3d(nl, clear3d=True)

    """
    """
    TO-DO:
    - allow generic "plot_{}" arguments:
        - e.g. plot_nodes (default=True) or plot_connectors
        - either autodetect how to plot stuff (x, y, z required)
        - accept "{}_layout" parameter:
            {'size': either int/float or str (str=column)
             'as': 'scatter' | 'lineplot',
             'color': either color | str (column name)
             'colormap': if color is column, map to

              }
    """
    # Backend
    backend = kwargs.pop('backend', 'auto')
    allowed_backends = ['auto', 'vispy', 'plotly']
    if backend.lower() == 'auto':
        if utils.is_jupyter():
            backend = 'plotly'
        else:
            backend = 'vispy'
    elif backend.lower() not in allowed_backends:
        raise ValueError(f'Unknown backend "{backend}". '
                         f'Permitted: {".".join(allowed_backends)}.')

    if backend == 'vispy':
        return plot3d_vispy(x, **kwargs)
    else:
        return plot3d_plotly(x, **kwargs)


def plot3d_vispy(x, **kwargs):
    """Plot3d() helper function to generate vispy 3D plots.

    This is just to improve readability. Its only purpose is to find the
    existing viewer or generate a new one.

    """
    # Parse objects to plot
    (neurons, volumes, points, visuals) = utils.parse_objects(x)

    # Check for allowed static parameters
    ALLOWED = {'color', 'c', 'colors', 'by_strahler', 'by_confidence',
               'cn_colors', 'linewidth', 'scatter_kws', 'synapse_layout',
               'dps_scale_vec', 'title', 'width', 'height', 'alpha',
               'auto_limits', 'autolimits', 'viewer', 'radius', 'center',
               'clear', 'clear3d', 'connectors', 'connectors_only', 'soma',
               'palette', 'color_by', 'shade_by', 'vmin', 'vmax', 'smin',
               'smax'}

    # Check if any of these parameters are dynamic (i.e. attached data tables)
    notallowed = set(kwargs.keys()) - ALLOWED

    if any(notallowed):
        raise ValueError(f'Argument(s) "{", ".join(notallowed)}" not allowed '
                         'for plot3d using vispy. Allowed keyword '
                         f'arguments: {", ".join(ALLOWED)}')

    scatter_kws = kwargs.pop('scatter_kws', {})

    if 'viewer' not in kwargs:
        # If does not exists yet, initialise a canvas object and make global
        if not getattr(config, 'primary_viewer', None):
            viewer = config.primary_viewer = Viewer()
        else:
            viewer = getattr(config, 'primary_viewer', None)
    else:
        viewer = kwargs.pop('viewer', getattr(config, 'primary_viewer'))

    # Make sure viewer is visible
    viewer.show()

    # We need to pop clear/clear3d to prevent clearing again later
    if kwargs.pop('clear3d', False) or kwargs.pop('clear', False):
        viewer.clear()

    # Do not pass this on parameter on
    center = kwargs.pop('center', True)

    # Add object (the viewer currently takes care of producing the visuals)
    if neurons:
        viewer.add(neurons, center=center, **kwargs)
    if volumes:
        viewer.add(volumes, center=center, **kwargs)
    if points:
        viewer.add(points, center=center, scatter_kws=scatter_kws)

    return viewer


def plot3d_plotly(x, **kwargs):
    """
    Plot3d() helper function to generate plotly 3D plots. This is just to
    improve readability and structure of the code.
    """

    # Check for allowed static parameters
    ALLOWED = {'color', 'c', 'colors', 'by_strahler', 'by_confidence',
               'cn_colors', 'linewidth', 'lw', 'scatter_kws',
               'synapse_layout', 'legend_group',
               'dps_scale_vec', 'title', 'width', 'height', 'fig_autosize',
               'plotly_inline', 'alpha', 'radius', 'fig', 'soma',
               'connectors', 'connectors_only', 'palette', 'color_by',
               'shade_by', 'vmin', 'vmax', 'smin', 'smax', 'hover_id',
               'hover_name'}

    # Check if any of these parameters are dynamic (i.e. attached data tables)
    notallowed = set(kwargs.keys()) - ALLOWED

    if any(notallowed):
        raise ValueError(f'Argument(s) "{", ".join(notallowed)}" not allowed '
                         'for plot3d using plotly. Allowed keyword '
                         f'arguments: {", ".join(ALLOWED)}')

    # Parse objects to plot
    (neurons, volumes, points, visual) = utils.parse_objects(x)

    # Pop colors so we don't have duplicate parameters when we go into the
    # individual ``...2plotly` functions
    colors = kwargs.pop('color',
                        kwargs.pop('c',
                                   kwargs.pop('colors', None)))

    palette = kwargs.get('palette', None)

    neuron_cmap, volumes_cmap = prepare_colormap(colors,
                                                 neurons=neurons,
                                                 volumes=volumes,
                                                 palette=palette,
                                                 alpha=kwargs.get('alpha', None),
                                                 color_range=255)

    data = []
    if neurons:
        data += neuron2plotly(neurons, neuron_cmap, **kwargs)
    if volumes:
        data += volume2plotly(volumes, volumes_cmap, **kwargs)
    if points:
        scatter_kws = kwargs.pop('scatter_kws', {})
        data += scatter2plotly(points, **scatter_kws)

    layout = layout2plotly(**kwargs)

    # If not provided generate a figure dictionary
    fig = kwargs.get('fig')
    if not fig:
        fig = dict(layout=layout)

    if not isinstance(fig, (dict, go.Figure)):
        raise TypeError('`fig` must be plotly.graph_objects.Figure or dict, got '
                        f'{type(fig)}')

    # Add data
    fig['data'] = fig.get('data', []) + data

    if kwargs.get('plotly_inline', True) and utils.is_jupyter():
        plotly.offline.iplot(fig)
        return
    else:
        logger.info('Use plotly.offline.plot(fig, filename="3d_plot.html")'
                    ' to plot. Optimized for Google Chrome.')
        return fig
