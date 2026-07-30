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

"""Helpers for working with the octarine 3D viewer.

[`navis.plot3d`][] with the `octarine` backend keeps a reference to the viewer
it last used in `config.primary_viewer`. `get_viewer`/`clear3d`/`pop3d`/`close3d`
are the shorthand for poking at that viewer without having to hold on to it
yourself.

Also home to `import_octarine`, which is the one place that knows how to ask for
octarine and what to say when it (or its navis plugin) isn't installed.

"""

from .. import config

__all__ = ["get_viewer", "clear3d", "close3d", "pop3d"]


def import_octarine():
    """Import octarine, with actionable errors if it's not (properly) installed.

    Octarine is a soft dependency and the navis-specific parts of it live in a
    separate plugin, so there are two ways for this to go wrong.

    """
    try:
        import octarine as oc
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The `octarine` backend requires the `octarine3d` library to be "
            "installed:\n  pip3 install octarine3d octarine-navis-plugin -U"
        )

    if not hasattr(oc.Viewer, "add_neurons"):
        raise ModuleNotFoundError(
            "Looks like the navis plugin for octarine is not installed. "
            "Please install it via pip:\n  pip install octarine-navis-plugin"
        )

    return oc


def get_viewer():
    """Grab active 3D viewer.

    Returns
    -------
    [`octarine.Viewer`](https://schlegelp.github.io/octarine/) or `None`
        The viewer most recently used by [`navis.plot3d`][], or `None` if
        there isn't one.

    Examples
    --------
    >>> import navis
    >>> # Plot a neuron in 3d
    >>> n = navis.example_neurons(1)
    >>> _ = n.plot3d(color='red', backend='octarine')       # doctest: +SKIP
    >>> # Grab the active viewer and take a screenshot
    >>> viewer = navis.get_viewer()                         # doctest: +SKIP
    >>> viewer.screenshot('neuron.png')                     # doctest: +SKIP
    >>> # Close viewer
    >>> navis.close3d()

    """
    return getattr(config, "primary_viewer", None)


def clear3d():
    """Clear viewer 3D canvas."""
    viewer = get_viewer()

    if viewer:
        viewer.clear()


def close3d():
    """Close and forget existing 3D viewer."""
    try:
        viewer = get_viewer()
        if viewer is None:
            return
        viewer.close()
        # `None` rather than `delattr` so there is one representation of "no
        # viewer" - `plot3d_octarine` uses the same when it finds a dead one.
        config.primary_viewer = None
        del viewer
    except BaseException as e:
        config.logger.warning("Error closing 3D viewer: {}".format(e))


def pop3d():
    """Remove the last item added to the 3D canvas."""
    viewer = get_viewer()
    if viewer:
        viewer.pop()
    else:
        config.logger.warning("No active 3D viewer to pop from.")
