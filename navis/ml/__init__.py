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

"""Functions for preparing neurons as inputs to machine-learning models."""

from .chunk import chunk_neuron, sample_patches
from .normalize import normalize_neuron
from .augment import (
    jitter_neuron,
    rotate_neuron,
    translate_neuron,
    scale_neuron,
    warp_neuron,
    drop_nodes,
    augment_neuron,
)
# `sample_points_uniform`, `estimate_spacing`, `sample_cable` and `sample_surface`
# are defined in `navis.sampling` but exposed here (and NOT at the top level) so the
# ML-facing helpers share one namespace (`navis.ml`). `sample_cable`/`sample_surface`
# are the skeleton/mesh -> point-cloud adapters for feeding neurons to models;
# `estimate_spacing` measures the density of a resulting cloud.
from ..sampling.utils import sample_points_uniform, estimate_spacing
from ..sampling.points import sample_cable, sample_surface

__all__ = [
    "chunk_neuron",
    "sample_patches",
    "normalize_neuron",
    "jitter_neuron",
    "rotate_neuron",
    "translate_neuron",
    "scale_neuron",
    "warp_neuron",
    "drop_nodes",
    "augment_neuron",
    "sample_points_uniform",
    "estimate_spacing",
    "sample_cable",
    "sample_surface",
]
