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

import skeletor as sk

from ... import core, config
from .objects import neuron2gfx

logger = config.get_logger(__name__)

def register_neuron2gfx():
    """Register the neuron2gfx converter with octarine."""
    try:
        import octarine as oc
    except ImportError:
        logger.error("octarine not found. Please install octarine to use the neuron2gfx converter.")
        return

    # Register the neuron2gfx converter
    oc.register_converter(core.BaseNeuron, neuron2gfx)
    oc.register_converter(core.NeuronList, neuron2gfx)
    oc.register_converter(sk.Skeleton, skeletor2gfx)

    oc.Viewer.add_neuron = add_neuron

    logger.info("neuron2gfx converter registered with octarine.")


def skeletor2gfx(s, **kwargs):
    """Convert a skeletor skeleton to a neuron2gfx object."""
    s = core.TreeNeuron(s, soma=None, id=0)
    return neuron2gfx(s, **kwargs)


def add_neuron(
        self,
        x,
        color=None,
        alpha=1,
        connectors=False,
        cn_colors=None,
        color_by=None,
        shade_by=None,
        palette=None,
        vmin=None,
        vmax=None,
        linewidth=1,
        synapse_layout=None,
        radius=False,
        center=True,
        clear=False
        ):
    """Add a neuron to the viewer.

    Parameters
    ----------
    x :         Neuron | NeuronList
                The neuron(s) to add to the viewer.
    color :     single color | list thereof, optional
                Color(s) for the neurons.
    connectors : bool, optional
                Whether to plot connectors.
    cn_colors : dict, optional
                A dictionary mapping connectors to colors.
    radius :    float, optional
                The radius of the skeleton.

    """
    vis = neuron2gfx(x, color=color, alpha=alpha, connectors=connectors, cn_colors=cn_colors,
                     color_by=color_by, shade_by=shade_by, palette=palette, vmin=vmin, vmax=vmax,
                     linewidth=linewidth, synapse_layout=synapse_layout, radius=radius)

    if clear:
        self.clear()

    for v in vis:
        self.scene.add(v)

    if center:
        self.center_camera()

