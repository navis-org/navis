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

import colorsys
from functools import wraps
import platform
import uuid
import warnings

import matplotlib.colors as mcl
import numpy as np
import pandas as pd
import png
import scipy.spatial
import seaborn as sns

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import vispy as vp

from ... import utils, config

__all__ = ['Viewer']

logger = config.logger


class Browser:
    """ Vispy browser.

    Parameters
    ----------
    n_rows/n_cols : int, optional
                    Number of rows and columns in the browser.

    """

    def __init__(self, n_rows=2, n_cols=3, link=False, **kwargs):
        # Update some defaults as necessary
        defaults = dict(keys=None,
                        show=True,
                        bgcolor='white')
        defaults.update(kwargs)

        # Generate canvas
        self.canvas = vp.scene.SceneCanvas(**defaults)

        # Add grid
        self.grid = self.canvas.central_widget.add_grid()

        # Add 3d widgets
        self.viewers = []
        for i in range(n_rows):
            for k in range(n_cols):
                v = self.grid.add_view(row=i, col=k)
                v.camera = 'turntable'
                v.border_color = (.5, .5, .5, 1)
                self.viewers.append(v)

        if link:
            for v in self.viewers[1:]:
                self.viewers[0].camera.link(v.camera)

    def add_to_all(self, x, center=True, clear=False, **kwargs):
        """ Add visuals to all viewers. """
        for v in range(len(self.viewers)):
            self.add(x, viewer=v, center=center, clear=clear, **kwargs)

    def add_and_divide(self, x, center=True, clear=False, **kwargs):
        """ Divide up visuals onto available viewers. """
        data = utils._parse_objects(x)

        v = 0
        for d in data:
            if isinstance(d, pd.DataFrame):
                d = d.itertuples()
            for to_add in d:
                self.add(to_add, viewer=v, center=center,
                         clear=clear, **kwargs)
                v += 1

                if v == len(self.viewers):
                    v = 0

    def add(self, x, viewer=1, center=True, clear=False, **kwargs):
        """ Add objects to canvas.

        Parameters
        ----------
        x :         skeleton IDs | Neuron/List | Dotprops | Volumes | Points | vispy visuals
                    Object(s) to add to the canvas.
        viewer :    int | slice, optional
                    Index of the viewer to add object to.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.
        clear :     bool, optional
                    If True, clear canvas before adding new objects.
        **kwargs
                    Keyword arguments passed when generating visuals. See
                    :func:`~navis.plot3d` for options.

        Returns
        -------
        None
        """

        (skids, skdata, dotprops, volumes,
         points, visuals) = utils._parse_objects(x)

        colors = kwargs.get('color',
                            kwargs.get('c',
                                       kwargs.get('colors', None)))
        # Parse colors for neurons and dotprops
        neuron_cmap, skdata_cmap = plotting._prepare_colormap(colors,
                                                              skdata, dotprops,
                                                              use_neuron_color=kwargs.get('use_neuron_color', False))
        kwargs['color'] = neuron_cmap + skdata_cmap

        if skids:
            visuals += plotting._neuron2vispy(fetch.get_neurons(skids),
                                              **kwargs)
        if skdata:
            visuals += plotting._neuron2vispy(skdata, **kwargs)
        if not dotprops.empty:
            visuals += plotting._dp2vispy(dotprops, **kwargs)
        if volumes:
            visuals += plotting._volume2vispy(volumes, **kwargs)
        if points:
            visuals += plotting._points2vispy(points,
                                              **kwargs.get('scatter_kws', {}))

        if not visuals:
            raise ValueError('No visuals generated.')

        if clear:
            self.clear()

        if isinstance(viewer, int):
            to_add = self.viewers[viewer:viewer + 1]
        elif isinstance(viewer, slice):
            to_add = self.viewers[viewer]
        else:
            raise TypeError('Unable to find viewer at {}'.format(type(viewer)))

        for view in to_add:
            for v in visuals:
                view.add(v)

            if center:
                self.center_camera(view)

    def show(self):
        """ Show viewer. """
        self.canvas.show()

    def close(self):
        """ Close viewer. """
        if self == globals().get('viewer', None):
            globals().pop('viewer')
        self.canvas.close()

    def get_visuals(self, viewer):
        """ Returns list of all 3D visuals on given viewer. """
        if isinstance(viewer, int):
            viewer = self.viewers[viewer]

        return [v for v in viewer.children[0].children if isinstance(v, vp.scene.visuals.VisualNode)]

    def center_camera(self, viewer):
        """ Center camera on visuals. """
        if isinstance(viewer, int):
            viewer = self.viewers[viewer]

        viewer.camera.set_range()
