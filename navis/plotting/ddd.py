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

"""Module contains functions to plot neurons in 3D."""
import os
import warnings

import plotly.graph_objs as go
import numpy as np

from typing import Union, List, Optional

from .. import utils, config, core
from .vispy.viewer import Viewer
from .colors import prepare_colormap
from .plotly.graph_objs import (neuron2plotly, volume2plotly, scatter2plotly,
                                layout2plotly)

__all__ = ['plot3d']

logger = config.get_logger(__name__)
_first_warning = True


def plot3d(x: Union[core.NeuronObject,
                    core.Volume,
                    np.ndarray,
                    List[Union[core.NeuronObject, np.ndarray, core.Volume]]
                    ],
           **kwargs) -> Optional[Union[Viewer, dict]]:
    """Generate 3D plot.

    Uses either `vispy <http://vispy.org>`_,  `k3d <https://k3d-jupyter.org/>`_
    or `plotly <http://plot.ly>`_. By default, the choice is automatic and
    depends on context::

      terminal: vispy used
      Jupyter: plotly used

    See ``backend`` parameter on how to change this behavior.

    Parameters
    ----------
    x :               Neuron/List | Volume | numpy.array
                        - ``numpy.array (N,3)`` is plotted as scatter plot
                        - multiple objects can be passed as list (see examples)
    backend :         'auto' | 'vispy' | 'plotly' | 'k3d', default='auto'
                      Which backend to use for plotting. Note that there will
                      be minor differences in what feature/parameters are
                      supported depending on the backend:
                        - ``auto`` selects backend based on context: ``vispy``
                          for terminal (if available) and ``plotly`` for Jupyter
                          environments. You can override this by setting an
                          environment variable `NAVIS_JUPYTER_PLOT3D_BACKEND="k3d"`.
                        - ``vispy`` uses OpenGL to generate high-performance
                          3D plots. Works in terminals.
                        - ``plotly`` generates 3D plots using WebGL. Works
                          "inline" in Jupyter notebooks but can also produce a
                          HTML file that can be opened in any browers.
                        - ``k3d`` generates 3D plots using k3d. Works only in
                          Jupyter notebooks!
    connectors :      bool, default=False
                      Plot connectors (e.g. synapses) if available.
    color :           None | str | tuple | list | dict, default=None
                      Use single str (e.g. ``'red'``) or ``(r, g, b)`` tuple
                      to give all neurons the same color. Use ``list`` of
                      colors to assign colors: ``['red', (1, 0, 1), ...].
                      Use ``dict`` to map colors to neurons:
                      ``{neuron.id: (r, g, b), ...}``.
    cn_colors :       str | tuple | dict | "neuron"
                      Overrides the default connector (e.g. synpase) colors:
                        - single color as str (e.g. ``'red'``) or rgb tuple
                          (e.g. ``(1, 0, 0)``)
                        - dict mapping the connectors tables ``type`` column to
                          a color (e.g. `{"pre": (1, 0, 0)}`)
                        - with "neuron", connectors will receive the same color
                          as their neuron
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
                      a ``smin`` and/or ``smax`` parameter. Does not work with
                      `k3d` backend.
    alpha :           float [0-1], optional
                      Alpha value for neurons. Overriden if alpha is provided
                      as fourth value in ``color`` (rgb*a*).
    clusters :        list, optional
                      A list assigning a cluster to each neuron (e.g.
                      ``[0, 0, 0, 1, 1]``). Overrides ``color`` and uses
                      ``palette`` to generate colors according to clusters.
    radius :          bool, default=False
                      If True, will plot TreeNeurons as 3D tubes using the
                      ``radius`` column in their node tables.
    width/height :    int, optional
                      Use to adjust figure/window size.
    scatter_kws :     dict, optional
                      Use to modify scatter plots. Accepted parameters are:
                        - ``size`` to adjust size of dots
                        - ``color`` to adjust color
    soma :            bool, default=True
                      Whether to plot soma if it exists (TreeNeurons only). Size
                      of the soma is determined by the neuron's ``.soma_radius``
                      property which defaults to the "radius" column for
                      ``TreeNeurons``.
    inline :          bool, default=True
                      If True and you are in an Jupyter environment, will
                      render plotly/k3d plots inline. If False, will generate
                      and return either a plotly Figure or a k3d Plot object
                      without immediately showing it.

                      ``Below parameters are for plotly backend only:``
    fig :             plotly.graph_objs.Figure
                      Pass to add graph objects to existing plotly figure. Will
                      not change layout.
    title :           str, default=None
                      For plotly only! Change plot title.
    fig_autosize :    bool, default=False
                      For plotly only! Autoscale figure size.
                      Attention: autoscale overrides width and height
    hover_name :      bool, default=False
                      If True, hovering over neurons will show their label.
    hover_id :        bool, default=False
                      If True, hovering over skeleton nodes will show their ID.
    legend_group :    dict, default=None
                      A dictionary mapping neuron IDs to labels (strings).
                      Use this to group neurons under a common label in the
                      legend.

                      ``Below parameters are for the vispy backend only:``
    clear :           bool, default = False
                      If True, will clear the viewer before adding the new
                      objects.
    center :          bool, default = True
                      If True, will center camera on the newly added objects.
    combine :         bool, default = False
                      If True, will combine objects of the same type into a
                      single visual. This can greatly improve performance but
                      also means objects can't be selected individually
                      anymore.

    Returns
    -------
    If ``backend='vispy'``

        Opens a 3D window and returns :class:`navis.Viewer`.

    If ``backend='plotly'``

        Returns either ``None`` if you are in a Jupyter notebook (see also
        ``inline`` parameter) or a ``plotly.graph_objects.Figure``
        (see examples).

    If ``backend='k3d'``

        Returns either ``None`` and immediately displays the plot or a
        ``k3d.plot`` object that you can manipulate further (see ``inline``
        parameter).

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
    >>> nl = navis.example_neurons()
    >>> # Backend is automatically chosen but we can set it explicitly
    >>> # Plot inline
    >>> nl.plot3d(backend='plotly')                             # doctest: +SKIP
    >>> # Plot as separate html in a new window
    >>> fig = nl.plot3d(backend='plotly', inline=False)
    >>> _ = plotly.offline.plot(fig)                            # doctest: +SKIP

    In a Jupyter notebook using k3d as backend.

    >>> nl = navis.example_neurons()
    >>> # Plot inline
    >>> nl.plot3d(backend='k3d')                                # doctest: +SKIP

    In a terminal using vispy as backend.

    >>> # Plot list of neurons
    >>> nl = navis.example_neurons()
    >>> v = navis.plot3d(nl, backend='vispy')
    >>> # Clear canvas
    >>> navis.clear3d()

    Some more advanced examples:

    >>> # plot3d() can deal with combinations of objects
    >>> nl = navis.example_neurons()
    >>> vol = navis.example_volume('LH')
    >>> vol.color = (255, 0, 0, .5)
    >>> # This plots a neuronlists, a single neuron and a volume
    >>> v = navis.plot3d([nl[0:2], nl[3], vol])
    >>> # Clear viewer (works only with vispy)
    >>> v = navis.plot3d(nl, clear3d=True)

    See the :ref:`plotting tutorial <plot_intro>` for even more examples.

    """
    # Backend
    backend = kwargs.pop('backend', 'auto')
    allowed_backends = ('auto', 'vispy', 'plotly', 'k3d')
    if backend.lower() == 'auto':
        if utils.is_jupyter():
            backend = os.environ.get('NAVIS_JUPYTER_PLOT3D_BACKEND', 'plotly')
        else:
            try:
                import vispy
                backend = 'vispy'
            except ImportError:
                # This is a warning (instead of logging) so that it only comes
                # up ones
                global _first_warning
                if _first_warning:  # warn only the first time
                    _first_warning = False
                    warnings.warn('The default backend for 3D plotting outside of '
                                  'Jupyter environments is `vispy` but it looks '
                                  'like vispy is not installed. Falling '
                                  'back to the plotly backend! If you would like '
                                  'to use vispy instead:\n\n  pip3 install vispy\n',
                                  category=UserWarning, stacklevel=2)
                backend = os.environ.get('NAVIS_JUPYTER_PLOT3D_BACKEND', 'plotly')
    elif backend.lower() not in allowed_backends:
        raise ValueError(f'Unknown backend "{backend}". '
                         f'Permitted: {".".join(allowed_backends)}.')

    if backend == 'vispy':
        return plot3d_vispy(x, **kwargs)
    elif backend == 'k3d':
        if not utils.is_jupyter():
            logger.warning('k3d backend only works in Jupyter environments')
        return plot3d_k3d(x, **kwargs)
    elif backend == 'plotly':
        return plot3d_plotly(x, **kwargs)
    else:
        raise ValueError(f'Unknown backend "{backend}". '
                         f'Permitted: {".".join(allowed_backends)}.')


def plot3d_vispy(x, **kwargs):
    """Plot3d() helper function to generate vispy 3D plots.

    This is just to improve readability. Its only purpose is to find the
    existing viewer or generate a new one.

    """
    try:
        import vispy
    except ImportError:
        raise ImportError('`navis.plot3d` requires the `vispy` package. Either '
                          'set e.g. `backend="plotly"` or install vispy:\n'
                          '  pip3 install vispy')

    # Parse objects to plot
    (neurons, volumes, points, visuals) = utils.parse_objects(x)

    # Check for allowed static parameters
    ALLOWED = {'color', 'c', 'colors', 'clusters',
               'cn_colors', 'linewidth', 'scatter_kws', 'synapse_layout',
               'dps_scale_vec', 'title', 'width', 'height', 'alpha',
               'auto_limits', 'autolimits', 'viewer', 'radius', 'center',
               'clear', 'clear3d', 'connectors', 'connectors_only', 'soma',
               'palette', 'color_by', 'shade_by', 'vmin', 'vmax', 'smin',
               'smax', 'shininess', 'volume_legend', 'combine'}

    # Check if any of these parameters are dynamic (i.e. attached data tables)
    notallowed = set(kwargs.keys()) - ALLOWED

    if any(notallowed):
        raise ValueError(f'Argument(s) "{", ".join(notallowed)}" not allowed '
                         'for plot3d using the vispy backend. Allowed keyword '
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
    combine = kwargs.pop('combine', False)

    # Add object (the viewer currently takes care of producing the visuals)
    if neurons:
        viewer.add(neurons, center=center, combine=combine, **kwargs)
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
    ALLOWED = {'color', 'c', 'colors', 'cn_colors',
               'linewidth', 'lw', 'legend_group',
               'scatter_kws', 'synapse_layout', 'clusters',
               'dps_scale_vec', 'title', 'width', 'height', 'fig_autosize',
               'inline', 'alpha', 'radius', 'fig', 'soma',
               'connectors', 'connectors_only', 'palette', 'color_by',
               'shade_by', 'vmin', 'vmax', 'smin', 'smax', 'hover_id',
               'hover_name', 'volume_legend'}

    # Check if any of these parameters are dynamic (i.e. attached data tables)
    notallowed = set(kwargs.keys()) - ALLOWED

    if any(notallowed):
        raise ValueError(f'Argument(s) "{", ".join(notallowed)}" not allowed '
                         'for plot3d using the plotly backend. Allowed keyword '
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
                                                 clusters=kwargs.get('clusters', None),
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
        fig = go.Figure(layout=layout)

    if not isinstance(fig, (dict, go.Figure)):
        raise TypeError('`fig` must be plotly.graph_objects.Figure or dict, got '
                        f'{type(fig)}')

    # Add data
    for trace in data:
        fig.add_trace(trace)

    if kwargs.get('inline', True) and utils.is_jupyter():
        fig.show()
        return
    else:
        logger.info('Use the `.show()` method to plot the figure.')
        return fig


def plot3d_k3d(x, **kwargs):
    """
    Plot3d() helper function to generate k3d 3D plots. This is just to
    improve readability and structure of the code.
    """
    # Lazy import because k3d is not yet a hard dependency
    try:
        import k3d
    except ImportError:
        raise ImportError('plot3d with `k3d` backend requires the k3d library '
                          'to be installed:\n  pip3 install k3d -U')

    from .k3d.k3d_objects import neuron2k3d, volume2k3d, scatter2k3d

    # Check for allowed static parameters
    ALLOWED = {'color', 'c', 'colors',
               'cn_colors', 'linewidth', 'lw', 'scatter_kws',
               'synapse_layout', 'clusters',
               'dps_scale_vec', 'height',
               'inline', 'alpha', 'radius', 'plot', 'soma',
               'connectors', 'connectors_only', 'palette', 'color_by',
               'vmin', 'vmax', 'smin', 'smax'}

    # Check if any of these parameters are dynamic (i.e. attached data tables)
    notallowed = set(kwargs.keys()) - ALLOWED

    if any(notallowed):
        raise ValueError(f'Argument(s) "{", ".join(notallowed)}" not allowed '
                         'for plot3d using the k3d backend. Allowed keyword '
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
                                                 clusters=kwargs.get('clusters', None),
                                                 alpha=kwargs.get('alpha', None),
                                                 color_range=255)

    data = []
    if neurons:
        data += neuron2k3d(neurons, neuron_cmap, **kwargs)
    if volumes:
        data += volume2k3d(volumes, volumes_cmap, **kwargs)
    if points:
        scatter_kws = kwargs.pop('scatter_kws', {})
        data += scatter2k3d(points, **scatter_kws)

    # If not provided generate a plot
    plot = kwargs.get('plot', None)
    if not plot:
        plot = k3d.plot(height=kwargs.get('height', 600))
        plot.camera_rotate_speed = 5
        plot.camera_zoom_speed = 2
        plot.camera_pan_speed = 1
        plot.grid_visible = False

    # Add data
    for trace in data:
        plot += trace

    if kwargs.get('inline', True) and utils.is_jupyter():
        plot.display()
    else:
        logger.info('Use the `.display()` method to show the plot.')
        return plot
