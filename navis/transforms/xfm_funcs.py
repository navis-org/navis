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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.

import math
import numbers

import numpy as np
import pandas as pd
import trimesh as tm

from functools import lru_cache
from scipy import ndimage
from scipy.spatial.distance import pdist
from typing import Union, Optional

from .. import utils, core, config
from .base import BaseTransform, TransformSequence, TransOptimizer
from .affine import AffineTransform

logger = config.get_logger(__name__)


def xform(x: Union['core.NeuronObject', 'pd.DataFrame', 'np.ndarray'],
          transform: Union[BaseTransform, TransformSequence],
          affine_fallback: bool = True,
          caching: bool = True) -> Union['core.NeuronObject',
                                         'pd.DataFrame',
                                         'np.ndarray']:
    """Apply transform(s) to data.

    Notes
    -----
    For Neurons only: whether there is a change in units during transformation
    (e.g. nm -> um) is inferred by comparing distances between x/y/z coordinates
    before and after transform. This guesstimate is then used to convert
    ``.units`` and node/soma radii. This works reasonably well with base 10
    increments (e.g. nm -> um) but is off with odd changes in units.

    Parameters
    ----------
    x :                 Neuron/List | Volume/Trimesh | numpy.ndarray | pandas.DataFrame
                        Data to transform. Dataframe must contain ``['x', 'y', 'z']``
                        columns. Numpy array must be shape ``(N, 3)``.
    transform :         Transform/Sequence or list thereof
                        Either a single transform or a transform sequence.
    affine_fallback :   bool
                        In same cases the non-rigid transformation of points
                        can fail - for example if points are outside the
                        deformation field. If that happens, they will be
                        returned as ``NaN``. Unless ``affine_fallback`` is
                        ``True``, in which case we will apply only the rigid
                        affine  part of the transformation to at least get close
                        to the correct coordinates.
    caching :           bool
                        If True, will (pre-)cache data for transforms whenever
                        possible. Depending on the data and the type of
                        transforms this can tremendously speed things up at the
                        cost of increased memory usage:
                          - ``False`` = no upfront cost, lower memory footprint
                          - ``True`` = higher upfront cost, most definitely faster
                        Only applies if input is NeuronList and if transforms
                        include H5 transform.

    Returns
    -------
    same type as ``x``
                        Copy of input with transformed coordinates.

    Examples
    --------
    >>> import navis
    >>> # Example neurons are in 8nm voxel space
    >>> nl = navis.example_neurons()
    >>> # Make a simple Affine transform to go from voxel to nanometers
    >>> import numpy as np
    >>> M = np.diag([8, 8, 8, 8])
    >>> tr = navis.transforms.AffineTransform(M)
    >>> # Apply the transform
    >>> xf = navis.xform(nl, tr)

    See Also
    --------
    :func:`navis.xform_brain`
                    Higher level function that finds and applies a sequence of
                    transforms to go from one template brain to another.

    """
    # We need to work with TransformSequence
    if isinstance(transform, (list, np.ndarray)):
        transform = TransformSequence(*transform)
    elif isinstance(transform, BaseTransform):
        transform = TransformSequence(transform)
    elif not isinstance(transform, TransformSequence):
        raise TypeError(f'Expected Transform or TransformSequence, got "{type(transform)}"')

    if isinstance(x, core.NeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            xf = []
            # Get the transformation sequence
            with TransOptimizer(transform, bbox=x.bbox, caching=caching):
                try:
                    for i, n in enumerate(config.tqdm(x, desc='Xforming',
                                                      disable=config.pbar_hide,
                                                      leave=config.pbar_leave)):
                        xf.append(xform(n,
                                        transform=transform,
                                        caching=caching,
                                        affine_fallback=affine_fallback))

                        # If not caching we will clear the map cache after
                        # each neuron to free memory
                        if not caching:
                            _get_coordinates_map.cache_clear()
                except BaseException:
                    raise
                finally:
                    # Make sure we clear the coordinate map cache when done
                    _get_coordinates_map.cache_clear()

            return x.__class__(xf)

    if isinstance(x, core.BaseNeuron):
        # VoxelNeurons are a special case and have hence their own function
        if isinstance(x, core.VoxelNeuron):
            return _xform_image(x, transform=transform)

        xf = x.copy()
        # We will collate spatial data to reduce overhead from calling
        # R's xform_brain
        if isinstance(xf, core.TreeNeuron):
            xyz = xf.nodes[['x', 'y', 'z']].values
        elif isinstance(xf, core.MeshNeuron):
            xyz = xf.vertices
        elif isinstance(xf, core.Dotprops):
            xyz = xf.points
            # If this dotprops has a `k`, we only need to transform points and
            # can regenerate the rest. If not, we need to make helper points
            # to carry over vectors
            if isinstance(xf.k, type(None)) or xf.k <= 0:
                # To avoid problems with these helpers we need to make sure
                # they aren't too close to their cognate points (otherwise we'll
                # get NaNs later). We can fix this by scaling the vector by the
                # sampling resolution which should also help make things less
                # noisy.
                hp = xf.points + xf.vect * xf.sampling_resolution
                xyz = np.append(xyz, hp, axis=0)
        else:
            raise TypeError(f"Don't know how to transform neuron of type '{type(xf)}'")

        # Add connectors if they exist
        if xf.has_connectors:
            xyz = np.vstack([xyz, xf.connectors[['x', 'y', 'z']].values])

        # Do the xform of all spatial data
        xyz_xf = xform(xyz,
                       transform=transform,
                       affine_fallback=affine_fallback)

        # Guess change in spatial units
        if xyz.shape[0] > 1:
            change, magnitude = _guess_change(xyz, xyz_xf, sample=1000)
        else:
            change, magnitude = 1, 0
            logger.warning(f'Unable to assess change of units for neuron {x.id}: '
                           'must have at least two nodes/points.')

        # Round change -> this rounds to the first non-zero digit
        # change = np.around(change, decimals=-magnitude)

        # Map xformed coordinates back
        if isinstance(xf, core.TreeNeuron):
            xf.nodes[['x', 'y', 'z']] = xyz_xf[:xf.n_nodes]
            # Fix radius based on our best estimate
            if 'radius' in xf.nodes.columns:
                xf.nodes['radius'] *= 10**magnitude
        elif isinstance(xf, core.Dotprops):
            xf.points = xyz_xf[:xf.points.shape[0]]

            # If this dotprops has a `k`, set tangent vectors and alpha to
            # None so they will be regenerated
            if not isinstance(xf.k, type(None)) and xf.k > 0:
                xf._vect = xf._alpha = None
            else:
                # Re-generate vectors
                hp = xyz_xf[xf.points.shape[0]: xf.points.shape[0] * 2]
                vect = xf.points - hp
                vect = vect / np.linalg.norm(vect, axis=1).reshape(-1, 1)
                xf._vect = vect
        elif isinstance(xf, core.MeshNeuron):
            xf.vertices = xyz_xf[:xf.vertices.shape[0]]

        if xf.has_connectors:
            xf.connectors[['x', 'y', 'z']] = xyz_xf[-xf.connectors.shape[0]:]

        # Make an educated guess as to whether the units have changed
        if hasattr(xf, 'units') and magnitude != 0:
            if isinstance(xf.units, (config.ureg.Unit, config.ureg.Quantity)):
                xf.units = (xf.units / 10**magnitude).to_compact()

        # Fix soma radius if applicable
        if hasattr(xf, 'soma_radius') and isinstance(xf.soma_radius, numbers.Number):
            xf.soma_radius *= 10**magnitude

        return xf
    elif isinstance(x, pd.DataFrame):
        if any([c not in x.columns for c in ['x', 'y', 'z']]):
            raise ValueError('DataFrame must have x, y and z columns.')
        x = x.copy()
        x[['x', 'y', 'z']] = xform(x[['x', 'y', 'z']].values,
                                   transform=transform,
                                   affine_fallback=affine_fallback)
        return x
    elif isinstance(x, tm.Trimesh):
        x = x.copy()
        x.vertices = xform(x.vertices,
                           transform=transform,
                           affine_fallback=affine_fallback)
        return x
    else:
        try:
            # At this point we expect numpy arrays
            x = np.asarray(x)
        except BaseException:
            raise TypeError(f'Unable to transform data of type "{type(x)}"')

        if not x.ndim == 2 or x.shape[1] != 3:
            raise ValueError('Array must be of shape (N, 3).')

    # Apply transform and return xformed points
    return transform.xform(x, affine_fallback=affine_fallback)


def _xform_image(x: 'core.VoxelNeuron',
                 transform: Union[BaseTransform, TransformSequence]
                 ) -> 'core.VoxelNeuron':
    """Apply transform(s) to image (voxel) data.

    Parameters
    ----------
    x :                 VoxelNeuron
                        Data to transform.
    transform :         Transform/Sequence or list thereof
                        Either a single transform or a transform sequence.

    Returns
    -------
    VoxelNeuron
                        Copy of neuron with transformed coordinates.

    """
    if not isinstance(x, core.VoxelNeuron):
        raise TypeError(f'Unable to transform image of type "{type(x)}"')

    # Get a target->source mapping
    # This is in a separate function because we are caching it
    # This cache is cleared by `xform` depending on whether caching is active
    # Note the conversion to tuples to make parameters hashable
    (ix_array_source,
     ix_array_target,
     target_voxel_size,
     target_offset) = _get_coordinates_map(transform,
                                           tuple(x.bbox.flatten()),
                                           x.shape,
                                           tuple(x.units_xyz.magnitude))

    # Use target->source index mapping to interpolate the image
    # order=1 means linear interpolation (much faster)
    mapped = ndimage.map_coordinates(x.grid, ix_array_source.T, order=1)
    grid_xf = np.zeros(x.shape)
    grid_xf[ix_array_target[:, 0],
            ix_array_target[:, 1],
            ix_array_target[:, 2]] = mapped

    # Generate the transformed neuron
    xf = x.copy()
    # Grid should be the same data type as original
    # also: vispy doesn't like float64
    xf.grid = grid_xf.astype(xf.grid.dtype)
    # New offset based on bounding box
    xf.offset = target_offset
    # Set voxel size
    xf.units = target_voxel_size

    return xf


# Note that the output of this function can be fairly large in memory
# (order ~4Gb for VFB image with shape (1210, 566, 174))
@lru_cache(maxsize=2)
def _get_coordinates_map(transform, bbox, shape, spacing):
    """Get target->source coordinate map for scipy's map_coordinates.

    Parameters
    ----------
    transform :     Transform/Sequence
                    The source->target transform.
    bbox :          (6, ) tuple
                    Bounding box of the image (in model space).
    shape :         (3, ) tuple
                    Shape of the image.
    spacing :       (3, ) tuple
                    The voxel dimensions.
    """
    # Parameters must be hashable for cache - hence no arrays
    # Here, we convert bbox back to arrays
    bbox = np.array(bbox).reshape(3, 2)

    # We could just use the two points of the source's bounding box to calculate
    # the target's bounding box. However, this can yield incorrect results.
    # It's better to sample a couple more points on the surface of the source's
    # bounding box
    b = tm.primitives.Box(extents=bbox[:, 1] - bbox[:, 0]).to_mesh()
    b.vertices += bbox.mean(axis=1)
    b = b.subdivide().vertices  # Subdivide to get more points on the surface

    # Temporarily ignore warnings
    current_level = int(logger.level)
    try:
        logger.setLevel('ERROR')
        # Transform points individually to avoid caching the entire volume
        b_xf = np.vstack([transform.xform(p.reshape(-1, 3), affine_fallback=True) for p in b])
        bbox_xf = np.vstack([np.min(b_xf, axis=0), np.max(b_xf, axis=0)]).T
    except BaseException:
        raise
    finally:
        logger.setLevel(current_level)

    # Next: generate a voxel grid in the target space of the same shape as
    # our input grid
    target_voxel_size = np.abs((bbox_xf[:, 1] - bbox_xf[:, 0]) / shape)

    # Generate a grid of xyz coordinates
    XX, YY, ZZ = np.meshgrid(range(shape[0]),
                             range(shape[1]),
                             range(shape[2]),
                             indexing='ij')
    # Coordinate grid has, for each voxel, in the (M, N, K) voxel grid an xyz
    # index coordinate -> hence shape is (3, M, N, K)
    ix_grid = np.array([XX, YY, ZZ])

    # Potential idea:
    # - generate a downsampled grid and upsample by interpolation later
    #   -> tried that but the interpolation during upsampling is just as
    #      expensive as transforming all points from the get-go
    # - or process in bocks which allows to skip blocks that are entirely empty

    # Convert grid into (N * N * K, 3) voxel array
    ix_array_target = ix_grid.T.reshape(-1, 3)

    # Convert indices to actual coordinates
    coo_array_target = (ix_array_target * target_voxel_size) + bbox_xf[:, 0]

    # Transform these coordinates from target back to source space
    # This step is the VERY slow one since we are (potentially) xforming
    # millions of coordinates
    try:
        logger.setLevel('ERROR')
        coo_array_source = (-transform).xform(coo_array_target,
                                              affine_fallback=True)
    except BaseException:
        raise
    finally:
        logger.setLevel(current_level)

    # Convert coordinates back into voxels
    ix_array_source = (coo_array_source - bbox[:, 0]) / spacing

    # New offset
    target_offset = bbox_xf[:, 0]

    return ix_array_source, ix_array_target, target_voxel_size, target_offset


def _guess_change(xyz_before: np.ndarray,
                  xyz_after: np.ndarray,
                  sample: float = .1) -> tuple:
    """Guess change in units during xforming."""
    if isinstance(xyz_before, pd.DataFrame):
        xyz_before = xyz_before[['x', 'y', 'z']].values
    if isinstance(xyz_after, pd.DataFrame):
        xyz_after = xyz_after[['x', 'y', 'z']].values

    # Select the same random sample of points in both spaces
    if sample <= 1:
        sample = int(xyz_before.shape[0] * sample)

    # Make sure we don't sample more than we have
    sample = min(xyz_before.shape[0], sample)

    rnd_ix = np.random.choice(xyz_before.shape[0], sample, replace=False)
    sample_bef = xyz_before[rnd_ix, :]
    sample_aft = xyz_after[rnd_ix, :]

    # Get pairwise distance between those points
    dist_pre = pdist(sample_bef)
    dist_post = pdist(sample_aft)

    # Calculate how the distance between nodes changed and get the average
    # Note we are ignoring nans - happens e.g. when points did not transform.
    with np.errstate(divide='ignore', invalid='ignore'):
        change = dist_post / dist_pre
    # Drop infinite values in rare cases where nodes end up on top of another
    mean_change = np.nanmean(change[change < np.inf])

    # Find the order of magnitude
    magnitude = round(math.log10(mean_change))

    return mean_change, magnitude


def mirror(points: np.ndarray, mirror_axis_size: float,
           mirror_axis: str = 'x',
           warp: Optional['BaseTransform'] = None) -> np.ndarray:
    """Mirror 3D coordinates about given axis.

    This is a lower level version of `navis.mirror_brain` that:
     1. Flips object along midpoint of axis using a affine transformation.
     2. (Optional) Applies a warp transform that corrects asymmetries.

    Parameters
    ----------
    points :            (N, 3) numpy array
                        3D coordinates to mirror.
    mirror_axis_size :  int | float
                        A single number specifying the size of the mirror axis.
                        This is used to find the midpoint to mirror about.
    mirror_axis :       'x' | 'y' | 'z', optional
                        Axis to mirror. Defaults to `x`.
    warp :              Transform, optional
                        If provided, will apply this warp transform after the
                        affine flipping. Typically this will be a mirror
                        registration to compensate for left/right asymmetries.

    Returns
    -------
    points_mirrored
                        Mirrored coordinates.

    See Also
    --------
    :func:`navis.mirror_brain`
                    Higher level function that uses meta data from registered
                    template brains to transform data for you.

    """
    utils.eval_param(mirror_axis, name='mirror_axis',
                     allowed_values=('x', 'y', 'z'), on_error='raise')

    # At this point we expect numpy arrays
    points = np.asarray(points)
    if not points.ndim == 2 or points.shape[1] != 3:
        raise ValueError('Array must be of shape (N, 3).')

    # Translate mirror axis to index
    mirror_ix = {'x': 0, 'y': 1, 'z': 2}[mirror_axis]

    # Construct homogeneous affine mirroring transform
    mirrormat = np.eye(4, 4)
    mirrormat[mirror_ix, 3] = mirror_axis_size
    mirrormat[mirror_ix, mirror_ix] = -1

    # Turn into affine transform
    flip_transform = AffineTransform(mirrormat)

    # Flip about mirror axis
    points_mirrored = flip_transform.xform(points)

    if isinstance(warp, (BaseTransform, TransformSequence)):
        points_mirrored = warp.xform(points_mirrored)

    # Note that we are enforcing the same data type as the input data here.
    # This is unlike in `xform` or `xform_brain` where data might genuinely
    # end up in a space that requires higher precision (e.g. going from
    # nm to microns).
    return points_mirrored.astype(points.dtype)


def _surface_voxels(bounds, spacing):
    """Get surface voxels for given bounding box."""
    assert bounds.shape == (3, 2)

    # Create planes
    xy_plane = np.mgrid[bounds[0][0]:bounds[0][1]:spacing[0],
                        bounds[1][0]:bounds[1][1]:spacing[1]].reshape(2, -1).T
    xz_plane = np.mgrid[bounds[0][0]:bounds[0][1]:spacing[0],
                        bounds[2][0]:bounds[2][1]:spacing[2]].reshape(2, -1).T
    yz_plane = np.mgrid[bounds[1][0]:bounds[1][1]:spacing[1],
                        bounds[2][0]:bounds[2][1]:spacing[2]].reshape(2, -1).T

    top = np.zeros((len(xy_plane), 3))
    top[:, :2] = xy_plane
    top[:, 2] = bounds[2][1]

    bot = np.zeros((len(xy_plane), 3))
    bot[:, :2] = xy_plane
    bot[:, 2] = bounds[2][0]

    front = np.zeros((len(xz_plane), 3))
    front[:, [0, 2]] = xz_plane
    front[:, 1] = bounds[1][0]

    back = np.zeros((len(xz_plane), 3))
    back[:, [0, 2]] = xz_plane
    back[:, 1] = bounds[1][1]

    left = np.zeros((len(yz_plane), 3))
    left[:, [1, 2]] = yz_plane
    left[:, 0] = bounds[0][0]

    right = np.zeros((len(yz_plane), 3))
    right[:, [1, 2]] = yz_plane
    right[:, 0] = bounds[0][1]

    return np.vstack((top, bot, front, back, left, right))
