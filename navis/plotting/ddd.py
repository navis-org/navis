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
import numpy as np

from typing import Union, List, Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import vispy

from .. import utils, config, core
from .vispy.viewer import Viewer
from .plotly.graph_objs import *

try:
    # Try setting vispy backend to PyQt5
    vispy.use(app='PyQt5')
except BaseException:
    pass

__all__ = ['plot3d']

logger = config.logger

if utils.is_jupyter():
    plotly.offline.init_notebook_mode(connected=True)


def plot3d(x: Union[core.NeuronObject,
                    core.Volume,
                    np.ndarray,
                    List[Union[core.NeuronObject, np.ndarray, core.Volume]]
                    ],
           **kwargs) -> Optional[Union[Viewer, dict]]:
    """ Generate 3D plot.

    Uses either `vispy <http://vispy.org>`_ or `plotly <http://plot.ly>`_.
    By default choice depends on context::

      terminal: vispy
      Jupyter: plotly

    See ``backend`` parameter on how to change default behavior.

    Parameters
    ----------

    x :               TreeNeuron/List| navis.Dotprops | navis.Volume | numpy.array
                        - ``numpy.array (N,3)`` is plotted as scatter plot
                        - multiple objects can be passed as list (see examples)
    backend :         'auto' | 'vispy' | 'plotly', default='auto'
                        - ``auto`` selects backend based on context: ``vispy``
                          for terminal and ``plotly`` for Jupyter environments.
                        - ``vispy`` uses OpenGL to generate high-performance
                          3D plots. Works in terminals.
                        - ``plotly`` generates 3D plots using WebGL. Works in
                          Jupyter notebooks. For Jupyter lab, you need to
                          have the Plotly lab extension installed.
    connectors :      bool, default=False
                      Plot connectors (e.g. synapses) if available.
    by_strahler :     bool, default=False
                      Will shade neuron(s) by strahler index.
    by_confidence :   bool, default=False
                      Will shade neuron(s) by arbor confidence.
    cn_mesh_colors :  bool, default=False
                      Plot connectors using mesh colors.
    clear3d :         bool, default=False
                      If True, canvas is cleared before plotting (only for
                      vispy).
    color :           None | str | tuple | list | dict, default=None (random)
                      Use single str (e.g. ``'red'``) or ``(r, g, b)`` tuple
                      to give all neurons the same color. Use ``list`` of
                      colors to assign colors: ``['red', (1, 0, 1), ...].
                      Use ``dict`` to map colors to neurons:
                      ``{uuid: (r, g, b), ...}``. RGB must be 0-255.
    use_neuron_color : bool, default=False
                      If True, will try using the ``.color`` attribute of
                      each neuron.
    width/height :    int, default=600
                      Use to define figure/window size.
    scatter_kws :     dict, optional
                      Use to modify scatter plots. Accepted parameters are
                        - ``size`` to adjust size of dots
                        - ``color`` to adjust color

    Plotly only

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

    Returns
    --------
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
    >>> plotly.offline.init_notebook_mode()
    >>> nl = navis.example_neurons()
    >>> # Backend is automatically chosen but we can set it explicitly
    >>> # Plot inline
    >>> nl.plot3d(backend='plotly')
    >>> # Plot as separate html in a new window
    >>> fig = nl.plot3d(backend='plotly', plotly_inline=False)
    >>> plotly.offline.plot(fig)

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
    >>> navis.plot3d([nl[0:2], nl[3], vol])
    >>> # Pass kwargs
    >>> navis.plot3d(nl1, clear3d=True, by_strahler)

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
    """ Plot3d() helper function to generate vispy 3D plots. This is just to
    improve readability. It's only purpose is to find the existing viewer
    or generate a new one.
    """

    # Parse objects to plot
    skdata, dotprops, volumes, points, visual = utils.parse_objects(x)

    # Check for allowed static parameters
    ALLOWED = {'color', 'c', 'colors', 'by_strahler', 'by_confidence',
               'cn_mesh_colors', 'linewidth', 'scatter_kws', 'synapse_layout',
               'dps_scale_vec', 'title', 'width', 'height',
               'auto_limits', 'autolimits', 'viewer', 'radius',
               'clear', 'clear3d', 'connectors', 'connectors_only'}

    # Check if any of these parameters are dynamic (i.e. attached data tables)
    notallowed = set(kwargs.keys()) - ALLOWED

    if any(notallowed):
        raise ValueError(f'Arguments "{",".join(notallowed)}" are not allowed '
                         'for plot3d using vispy. Allowed keyword '
                         f'arguments: {",".join(ALLOWED)}')

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

    if skdata:
        viewer.add(skdata, **kwargs)
    if not dotprops.empty:
        viewer.add(dotprops, **kwargs)
    if volumes:
        viewer.add(volumes, **kwargs)
    if points:
        viewer.add(points, scatter_kws=scatter_kws)

    return viewer


def plot3d_plotly(x, **kwargs):
    """
    Plot3d() helper function to generate plotly 3D plots. This is just to
    improve readability and structure of the code.
    """

    # Check for allowed static parameters
    ALLOWED = {'color', 'c', 'colors', 'by_strahler', 'by_confidence',
               'cn_mesh_colors', 'linewidth', 'scatter_kws', 'synapse_layout',
               'dps_scale_vec', 'title', 'width', 'height', 'fig_autosize',
               'plotly_inline',
               'connectors', 'connectors_only'}

    # Check if any of these parameters are dynamic (i.e. attached data tables)
    notallowed = set(kwargs.keys()) - ALLOWED

    if any(notallowed):
        raise ValueError(f'Arguments "{",".join(notallowed)}" are not allowed '
                         'for plot3d using vispy. Allowed keyword '
                         f'arguments: {",".join(ALLOWED)}')

    # Parse objects to plot
    skdata, dotprops, volumes, points, visual = utils.parse_objects(x)

    trace_data = []

    scatter_kws = kwargs.pop('scatter_kws', {})

    if skdata:
        trace_data += neuron2plotly(skdata, **kwargs)
    if not dotprops.empty:
        trace_data += dotprops2plotly(dotprops, **kwargs)
    if volumes:
        trace_data += volume2plotly(volumes, **kwargs)
    if points:
        trace_data += scatter2plotly(points, scatter_kws=scatter_kws)

    layout = layout2plotly(**kwargs)

    fig = dict(data=trace_data, layout=layout)

    if kwargs.get('plotly_inline', True) and utils.is_jupyter():
        plotly.offline.iplot(fig)
        return
    else:
        logger.info('Use plotly.offline.plot(fig, filename="3d_plot.html")'
                    ' to plot. Optimized for Google Chrome.')
        return fig
