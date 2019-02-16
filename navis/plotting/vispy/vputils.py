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

import png
import warnings

from ... import config

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from vispy.gloo.util import _screenshot


def get_viewer():
    """ Returns active 3D viewer.

    Returns
    -------
    :class:`~navis.Viewer`

    Examples
    --------
    >>> from vispy import scene
    >>> # Get and plot neuron in 3d
    >>> n = navis.example_neurons(1)
    >>> n.plot3d(color = 'red')
    >>> # Plot connector IDs
    >>> cn_ids = n.connectors.connector_id.values.astype(str)
    >>> cn_co = n.connectors[['x', 'y', 'z']].values
    >>> viewer = navis.get_viewer()
    >>> text = scene.visuals.Text(text=cn_ids,
    ...                           pos=cn_co * scale_factor)
    >>> viewer.add(text)

    """
    return getattr(config, 'primary_viewer', None)


def clear3d():
    """ Clear viewer 3D canvas.
    """
    viewer = get_viewer()

    if viewer:
        viewer.clear()


def close3d():
    """ Close existing vispy 3D canvas (wipes memory).
    """
    try:
        viewer = get_viewer()
        viewer.close()
        globals().pop('viewer')
        del viewer
    except BaseException:
        pass


def screenshot(file='screenshot.png', alpha=True):
    """ Saves a screenshot of active vispy 3D canvas.

    Parameters
    ----------
    file :      str, optional
                Filename
    alpha :     bool, optional
                If True, alpha channel will be saved

    See Also
    --------
    :func:`navis.Viewer.screenshot`
                Take screenshot of specific canvas.
    """
    if alpha:
        mode = 'RGBA'
    else:
        mode = 'RGB'

    im = png.from_array(_screenshot(alpha=alpha), mode=mode)
    im.save(file)

    return
