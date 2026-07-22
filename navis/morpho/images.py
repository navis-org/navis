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
import sparsecubes

from scipy.ndimage import gaussian_filter
from typing import Union

from .. import core, utils

NeuronObject = Union["core.NeuronList", "core.TreeNeuron"]

__all__ = ["smooth_voxels", "thin_voxels"]


@utils.map_neuronlist(desc="Smoothing", allow_parallel=True)
def smooth_voxels(
    x: NeuronObject,
    sigma: int = 1,
    backend: str = "sparsecubes",
    epsilon: float = 0,
    truncate: float = 4.0,
    inplace: bool = False,
) -> NeuronObject:
    """Smooth voxel(s) using a Gaussian filter.

    Parameters
    ----------
    x :             VoxelNeuron | NeuronList
                    Neuron(s) to be processed.
    sigma :         int | (3, ) ints, optional
                    Standard deviation for Gaussian kernel, **in voxels** (not
                    in the neuron's units). Given for each axis as a sequence,
                    or as a single number in which case it is equal for all
                    axes. `0` along an axis disables smoothing on it.
    backend :       "sparsecubes" | "scipy"
                    Which implementation to use. Both treat everything outside
                    the neuron as empty (scipy's `mode="constant", cval=0`) and
                    agree to floating-point round-off wherever they overlap:
                      - "sparsecubes" (default) smooths the sparse voxels
                        directly and never allocates the dense grid. Values
                        that spread beyond the neuron's canvas are kept, so the
                        neuron grows.
                      - "scipy" uses `scipy.ndimage.gaussian_filter` on the
                        dense grid, which keeps the canvas fixed and therefore
                        clips whatever spreads beyond it. Can be faster for
                        small, densely occupied neurons.
    epsilon :       float, optional
                    Only for the "sparsecubes" backend: drop output voxels at
                    or below this magnitude. Smoothing grows the voxel set by
                    the kernel radius, and the Gaussian's tail contributes very
                    little while occupying most of that growth - so a small
                    `epsilon` keeps the neuron sparse at a bounded cost in
                    exactness. The default (`0`) is exact.
    truncate :      float, optional
                    Truncate the kernel at this many standard deviations.
                    Lowering it is the cheapest way to bound the cost, which
                    grows as `(2 * truncate * sigma + 1) ** 3`.
    inplace :       bool, optional
                    If False, will use and return copy of original neuron(s).

    Returns
    -------
    VoxelNeuron/List
                    Smoothed neuron(s). Note that smoothing spreads values
                    outwards, so the neuron will occupy more voxels than
                    before.

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

    utils.eval_param(
        backend, name="backend", allowed_values=("sparsecubes", "scipy")
    )

    if not inplace:
        x = x.copy()

    if backend == "scipy":
        # `mode="constant", cval=0` to match the sparse backend: outside the
        # neuron is empty. Scipy's default ("reflect") would mirror the data at
        # the canvas boundary, inventing mass outside the imaged volume.
        # Note this keeps the grid's shape, i.e. anything that spreads past the
        # canvas is still clipped.
        x._data = gaussian_filter(
            x.grid.astype(np.float32),
            sigma=sigma,
            truncate=truncate,
            mode="constant",
            cval=0,
        )
        # Values now live in the grid itself; a stale `_values` would be ignored
        # by `.values` but still feed the core hash (see `CORE_DATA`)
        if hasattr(x, "_values"):
            delattr(x, "_values")
        x._clear_temp_attr()
        return x

    voxels, values = sparsecubes.filters.smooth(
        x.voxels,
        values=x.values.astype(np.float32),
        sigma=sigma,
        truncate=truncate,
        epsilon=epsilon,
    )

    # sparse-cubes computes in float64. Smoothing multiplies the voxel count
    # (by ~75x at sigma=2), so keep the float32 the dense path has always
    # produced rather than doubling the footprint for precision that image
    # data does not carry anyway.
    values = values.astype(np.float32)

    # Smoothing spreads outwards, so the neuron grows past its old canvas.
    # `_replace_voxels` widens the shape (and shifts the frame if the spread
    # pushed coordinates negative).
    x._replace_voxels(voxels, values, inplace=True)

    return x


@utils.map_neuronlist(desc="Thinning", allow_parallel=True)
def thin_voxels(x, backend="auto", inplace=False, **kwargs):
    """Skeletonize image data to single voxel width.

    Parameters
    ----------
    x :         VoxelNeuron | numpy array
                The image to thin. Arrays are interpreted as dense image data
                (not as voxel coordinates).
    backend :   "auto" | "sparsecubes" | "skimage"
                Which implementation to use:
                  - "sparsecubes" works straight off the sparse voxels and
                    never allocates the dense grid, which for a sparse neuron
                    is both much faster and dramatically lighter on memory. It
                    also preserves the values of surviving voxels. Requires
                    sparse-cubes >= 0.4.0.
                  - "skimage" uses `skimage.morphology.skeletonize` on the
                    dense grid. Requires scikit-image, and returns a *binary*
                    neuron (voxel values are discarded).
                  - "auto" (default) picks "sparsecubes" where it is available
                    and applicable, and falls back to "skimage" otherwise.
                    Note that 2D arrays always go to "skimage": sparse-cubes
                    is 3D only.
    inplace :   bool
                For VoxelNeurons only: Whether to manipulate the neuron
                in place.
    **kwargs
                For the "sparsecubes" backend: passed through to
                `sparsecubes.binary.thin`, e.g. `preserve_endpoints`.

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
    utils.eval_param(
        backend, name="backend", allowed_values=("auto", "sparsecubes", "skimage")
    )

    if not isinstance(x, (core.VoxelNeuron, np.ndarray)):
        raise TypeError(f"Unable to thin data of type {type(x)}")

    # sparse-cubes is 3D only, so 2D image data can only go to scikit-image
    is_2d = isinstance(x, np.ndarray) and x.ndim == 2

    if backend == "auto":
        backend = "skimage" if is_2d else "sparsecubes"
    elif backend == "sparsecubes" and is_2d:
        raise ValueError(
            "The `sparsecubes` backend only handles 3D data, got a 2D "
            "array. Use `backend='skimage'` for 2D images."
        )

    if backend == "sparsecubes":
        return _thin_sparsecubes(x, inplace=inplace, **kwargs)

    return _thin_skimage(x, inplace=inplace, **kwargs)


def _thin_sparsecubes(x, inplace=False, **kwargs):
    """Thin via sparse-cubes, straight off the sparse voxels."""
    if isinstance(x, core.VoxelNeuron):
        # The VoxelNeuron method already carries the surviving voxels' values
        if inplace:
            x.thin(inplace=True, **kwargs)
            return x
        return x.thin(**kwargs)

    # Dense image data: in and out, so that the two backends agree on the
    # contract even though sparse-cubes itself never sees the grid
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image data, got {x.ndim}D array")

    thinned = sparsecubes.binary.thin(np.argwhere(x), **kwargs)

    out = np.zeros(x.shape, dtype=bool)
    if len(thinned):
        out[thinned[:, 0], thinned[:, 1], thinned[:, 2]] = True
    return out


def _thin_skimage(x, inplace=False, **kwargs):
    """Thin via scikit-image, which needs the dense grid."""
    try:
        from skimage.morphology import skeletonize
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "`thin_voxels` with the scikit-image backend requires the "
            "scikit-image package:\n"
            "  pip install scikit-image"
        )

    if isinstance(x, core.VoxelNeuron):
        if not inplace:
            x = x.copy()

        x.grid = skeletonize(x.grid, **kwargs)
    else:
        x = skeletonize(x, **kwargs)

    return x
