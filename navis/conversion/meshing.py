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

import warnings

import numpy as np
import trimesh as tm
from typing import Union, Optional
from typing_extensions import Literal

from scipy.ndimage import binary_erosion, binary_fill_holes

from .. import core, config, utils

try:
    from fastremap import unique
except ModuleNotFoundError:
    from numpy import unique

import sparsecubes

logger = config.get_logger(__name__)


def voxels2mesh(vox: Union['core.VoxelNeuron', np.ndarray],
                spacing: Union[Literal['auto'], np.ndarray] = 'auto',
                step_size: int = 1,
                smooth: bool = True,
                chunk_size: Optional[int] = None,
                pad_chunks: Optional[bool] = None,
                merge_fragments: Optional[bool] = None,
                progress: Optional[bool] = None) -> Union[tm.Trimesh, 'core.MeshNeuron']:
    """Generate mesh from voxels.

    Uses [`sparsecubes.mesh`][], which works directly off the sparse voxels
    instead of densifying them to a 3D grid the way marching cubes does.

    Parameters
    ----------
    vox :           VoxelNeuron | (N, 3) np.array
                    Object to voxelize. Can be a VoxelNeuron or an (N, 3) array
                    of x, y, z voxel coordinates.
    spacing :       np.array
                    (3, ) array with x, y, z voxel size. If `auto` and input is
                    a `VoxelNeuron` we will use the neuron's `.units` property,
                    else spacing will be `(1, 1, 1)`.
    step_size :     int, optional
                    Step size for the meshing algorithm.
                    Higher values = faster but coarser.
    smooth :        bool
                    How vertices are placed:
                      - True (default) uses SurfaceNets: one vertex per surface
                        cell, placed at the centroid of the surface crossings
                        around it. Smooths the staircase you would otherwise get
                        on diagonal surfaces. Vertices are floats.
                      - False gives blocky, axis-aligned quads with corners on
                        the integer voxel grid. Vertices keep the input's
                        integer dtype.
    chunk_size :    int, optional
                    Deprecated and ignored - meshing no longer densifies the
                    voxels, so there is nothing to chunk.
    pad_chunks :    bool, optional
                    Deprecated and ignored, see `chunk_size`.
    merge_fragments :  bool, optional
                    Deprecated and ignored, see `chunk_size`.
    progress :      bool, optional
                    Deprecated and ignored: meshing is now a single pass with
                    no chunk loop to report on.

    Returns
    -------
    mesh :          trimesh.Trimesh | MeshNeuron
                    Returns a trimesh or MeshNeuron depending on the input.
                    Data tables (e.g. `connectors`) are not carried over from
                    input neuron.

    """
    utils.eval_param(vox, 'vox', allowed_types=(core.VoxelNeuron, np.ndarray))

    # `sparsecubes` works straight off the sparse voxels, so the machinery that
    # existed only to keep marching cubes' dense grid in check is moot
    for name, value in (('chunk_size', chunk_size),
                        ('pad_chunks', pad_chunks),
                        ('merge_fragments', merge_fragments),
                        ('progress', progress)):
        if value is not None:
            warnings.warn(
                f'`{name}` is deprecated and ignored: meshing no longer '
                'densifies the voxels, so there is nothing to chunk.',
                DeprecationWarning,
                stacklevel=2
            )

    # Note the `isinstance` guard: `spacing` is usually an array by the time a
    # caller passes it explicitly, and comparing that to a string is not a
    # scalar bool
    if isinstance(spacing, str) and spacing == 'auto':
        if not isinstance(vox, core.VoxelNeuron):
            spacing = [1, 1, 1]
        else:
            spacing = vox.units_xyz.magnitude

    if isinstance(vox, core.VoxelNeuron):
        voxels = vox.voxels
    else:
        voxels = vox

    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise ValueError(f'Voxels must be shape (N, 3), got {voxels.shape}')

    mesh = sparsecubes.mesh(voxels,
                            spacing=spacing,
                            step_size=step_size,
                            smooth=smooth)

    if isinstance(vox, core.VoxelNeuron):
        # `smooth=False` keeps the voxels' integer dtype, which an in-place add
        # of a float offset would refuse - so reassign rather than `+=`
        mesh.vertices = mesh.vertices + vox.offset
        mesh = core.MeshNeuron(mesh, units=f'1 {vox.units.units}', id=vox.id)

    return mesh


def remove_surface_voxels(voxels, **kwargs):
    """Removes surface voxels."""
    # Use bounding boxes to keep matrix small
    bb_min = voxels.min(axis=0)
    #bb_max = voxels.max(axis=0)
    #dim = bb_max - bb_min

    # Voxel offset
    voxel_off = voxels - bb_min

    # Generate empty array
    mat = _voxels_to_matrix(voxel_off)

    # Erode
    mat_erode = binary_erosion(mat, **kwargs)

    # Turn back into voxels
    voxels_erode = _matrix_to_voxels(mat_erode) + bb_min

    return voxels_erode


def get_surface_voxels(voxels):
    """Return surface voxels."""
    # Use bounding boxes to keep matrix small
    bb_min = voxels.min(axis=0)
    #bb_max = voxels.max(axis=0)
    #dim = bb_max - bb_min

    # Voxel offset
    voxel_off = voxels - bb_min

    # Generate empty array
    mat = _voxels_to_matrix(voxel_off)

    # Erode
    mat_erode = binary_erosion(mat)

    # Substract
    mat_surface = np.bitwise_and(mat, np.invert(mat_erode))

    # Turn back into voxels
    voxels_surface = _matrix_to_voxels(mat_surface) + bb_min

    return voxels_surface


def parse_obj(obj):
    """Parse .obj string and return vertices and faces."""
    lines = obj.split('\n')
    verts = []
    faces = []
    for l in lines:
        if l.startswith('v '):
            verts.append([float(v) for v in l[2:].split(' ')])
        elif l.startswith('f '):
            f = [v.split('//')[0] for v in l[2:].split(' ')]
            faces.append([int(v) for v in f])

    # `.obj` faces start with vertex indices of 1 -> set to 0
    return np.array(verts), np.array(faces) - 1


def _voxels_to_matrix(voxels, fill=False, pad=1, dtype=bool):
    """Generate matrix from voxels/blocks.

    Parameters
    ----------
    voxels :    numpy array
                Either voxels [[x, y, z], ..] or blocks [[z, y, x1, x2], ...]
    fill :      bool, optional
                If True, will use binary fill to fill holes in matrix.

    Returns
    -------
    numpy array
                3D matrix with x, y, z axis (blocks will be converted)

    """
    if not isinstance(voxels, np.ndarray):
        voxels = np.array(voxels)

    offset = voxels.min(axis=0)
    voxels = voxels - offset

    # Populate matrix
    if voxels.shape[1] == 4:
        mat = np.zeros((voxels.max(axis=0) + 1)[[-1, 1, 0]], dtype=dtype)
        for col in voxels:
            mat[col[2]:col[3] + 1, col[1], col[0]] = 1
    elif voxels.shape[1] == 3:
        mat = np.zeros((voxels.max(axis=0) + 1), dtype=dtype)
        mat[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1
    else:
        raise ValueError('Unexpected voxel shape')

    # Fill holes
    if fill:
        mat = binary_fill_holes(mat)

    if pad:
        mat = np.pad(mat, pad_width=pad, mode='constant', constant_values=0)

    return mat, offset


def _matrix_to_voxels(matrix):
    """Turn matrix into voxels.

    Assumes that voxels have values True or 1.
    """
    # Turn matrix into voxels
    return np.vstack(np.where(matrix)).T


def _apply_mask(mat, mask):
    """Substracts (logical_xor) mask from mat.

    Assumes that both matrices are (a) of the same voxel size and (b) have the
    same origin.
    """
    # Get maximum dimension between mat and mask
    max_dim = np.max(np.array([mat.shape, mask.shape]), axis=0)

    # Bring mask in shape of mat
    mask_pad = np.vstack([np.array([0, 0, 0]), max_dim - mask.shape]).T
    mask = np.pad(mask, mask_pad, mode='constant', constant_values=0)
    mask = mask[:mat.shape[0], :mat.shape[1], :mat.shape[2]]

    # Substract mask
    return np.bitwise_and(mat, np.invert(mask))


def _mask_voxels(voxels, mask_voxels):
    """Mask voxels with other voxels.

    Assumes that both voxels have the same size and origin.
    """
    # Generate matrices of the voxels
    mat = _voxels_to_matrix(voxels)
    mask = _voxels_to_matrix(mask_voxels)

    # Apply mask
    masked = _apply_mask(mat, mask)

    # Turn matrix back into voxels
    return _matrix_to_voxels(masked)


def _blocks_to_voxels(blocks):
    return _matrix_to_voxels(_voxels_to_matrix(blocks))
