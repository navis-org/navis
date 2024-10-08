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

import numpy as np

from scipy.ndimage import gaussian_filter
from typing import Union

from .. import core, utils

NeuronObject = Union["core.NeuronList", "core.TreeNeuron"]

__all__ = ["smooth_voxels", "thin_voxels"]


@utils.map_neuronlist(desc="Smoothing", allow_parallel=True)
def smooth_voxels(
    x: NeuronObject, sigma: int = 1, inplace: bool = False
) -> NeuronObject:
    """Smooth voxel(s) using a Gaussian filter.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
                    Neuron(s) to be processed.
    sigma :         int | (3, ) ints, optional
                    Standard deviation for Gaussian kernel. The standard
                    deviations of the Gaussian filter are given for each axis
                    as a sequence, or as a single number, in which case it is
                    equal for all axes.
    inplace :       bool, optional
                    If False, will use and return copy of original neuron(s).

    Returns
    -------
    VoxelNeuron/List
                    Smoothed neuron(s).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind='mesh')
    >>> vx = navis.voxelize(n, pitch='1 micron')
    >>> smoothed = navis.smooth_voxels(vx, sigma=2)

    See Also
    --------
    [`navis.smooth_mesh`][]
                    For smoothing MeshNeurons and other mesh-likes.
    [`navis.smooth_skeleton`][]
                    For smoothing TreeNeurons.

    """
    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, core.VoxelNeuron):
        raise TypeError(f"Can only process VoxelNeurons, not {type(x)}")

    if not inplace:
        x = x.copy()

    # Apply gaussian
    x._data = gaussian_filter(x.grid.astype(np.float32), sigma=sigma)
    x._clear_temp_attr()

    return x


@utils.map_neuronlist(desc="Thinning", allow_parallel=True)
def thin_voxels(x, inplace=False):
    """Skeletonize image data to single voxel width.

    This is a simple thin wrapper around scikit-learn's `skeletonize`.

    Parameters
    ----------
    x :         VoxelNeuron | numpy array
                The image to thin.
    inplace :   bool
                For VoxelNeurons only: Whether to manipulate the neuron
                in place.

    Returns
    -------
    thin
                Thinned VoxelNeuron or numpy array.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind='mesh')
    >>> vx = navis.voxelize(n, pitch='1 micron')
    >>> thinned = navis.thin_voxels(vx)

    """
    try:
        from skimage.morphology import skeletonize
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "`thin_image` requires the scikit-image packge:\n"
            "  pip install scikit-image"
        )

    if isinstance(x, core.VoxelNeuron):
        if not inplace:
            x = x.copy()

        x.grid = skeletonize(x.grid)
    elif isinstance(x, np.ndarray):
        x = skeletonize(x)
    else:
        raise TypeError(f"Unable to thin data of type {type(x)}")

    return x
