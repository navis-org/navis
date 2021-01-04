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

__all__ = ['get_viewer', 'clear3d', 'close3d', 'screenshot']


def get_viewer():
    """Grab active 3D viewer.

    Returns
    -------
    :class:`~navis.Viewer`

    Examples
    --------
    >>> import navis
    >>> from vispy import scene
    >>> # Get and plot neuron in 3d
    >>> n = navis.example_neurons(1)
    >>> _ = n.plot3d(color='red')
    >>> # Grab active viewer and add custom text
    >>> viewer = navis.get_viewer()
    >>> text = scene.visuals.Text(text='TEST',
    ...                           pos=(0, 0, 0))
    >>> viewer.add(text)
    >>> # Close viewer
    >>> viewer.close()

    """
    return getattr(config, 'primary_viewer', None)


def clear3d():
    """Clear viewer 3D canvas."""
    viewer = get_viewer()

    if viewer:
        viewer.clear()


def close3d():
    """Close existing vispy 3D canvas (wipes memory)."""
    try:
        viewer = get_viewer()
        viewer.close()
        globals().pop('viewer')
        del viewer
    except BaseException:
        pass


def screenshot(file='screenshot.png', alpha=True):
    """Save a screenshot of active vispy 3D canvas.

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
