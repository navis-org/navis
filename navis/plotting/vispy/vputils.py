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

import numpy as np

from ... import config

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from vispy.gloo.util import _screenshot
    import vispy.scene.visuals as vpvisuals


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


def _combine_visuals(visuals):
    """ Attempts to combine multiple visuals of similar type into one.
    """

    if any([not isinstance(v, vpvisuals.VisualNode) for v in visuals]):
        raise TypeError('Visuals must all be instances of VisualNode')

    # Sort into types
    types = set([type(v) for v in visuals])

    by_type = {ty: [v for v in visuals if type(v) == ty] for ty in types}

    combined = []

    # Now go over types and combine when possible
    for ty in types:
        # Skip if nothing to combine
        if len(by_type[ty]) <= 1:
            combined += by_type[ty]
            continue

        if ty == vpvisuals.Line:
            # Collate data
            pos_comb = np.concatenate([vis._pos for vis in by_type[ty]])

            color = np.concatenate([np.repeat([vis.color],
                                              vis.pos.shape[0],
                                              axis=0) for vis in by_type[ty]])

            if color.shape[1] == 3:
                color.append()





