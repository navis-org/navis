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
import trimesh as tm
from typing import Union, Optional
from typing_extensions import Literal

from scipy.ndimage import binary_erosion, binary_fill_holes

from .. import core, config, utils

try:
    from fastremap import unique
except ImportError:
    from numpy import unique

try:
    import skimage
    from skimage import measure
except ImportError:
    skimage = None

logger = config.get_logger(__name__)


def voxels2mesh(vox: Union['core.VoxelNeuron', np.ndarray],
                spacing: Union[Literal['auto'], np.ndarray] = 'auto',
                step_size: int = 1,
                chunk_size: Optional[int] = 'auto',
                pad_chunks: bool = True,
                merge_fragments: bool = True,
                progress: bool = True) -> Union[tm.Trimesh, 'core.MeshNeuron']:
    """Generate mesh from voxels using marching cubes.

    Parameters
    ----------
    voxels :        VoxelNeuron | (N, 3) np.array
                    Object to voxelize. Can be a VoxelNeuron or an (N, 3) array
                    of x, y, z voxel coordinates.
    spacing :       np.array
                    (3, ) array with x, y, z voxel size. If `auto` and input is
                    a `VoxelNeuron` we will use the neuron's `.units` property,
                    else spacing will be `(1, 1, 1)`.
    step_size :     int, optional
                    Step size for marching cube algorithm.
                    Higher values = faster but coarser.
    chunk_size :    "auto" | int, optional
                    Whether to process voxels in chunks to keep memory footprint
                    low:
                      - "auto" will set chunk size automatically based on size
                        of input
                      - use ``int`` to set chunk size - smaller chunk mean lower
                        memory consumption but longer run time - 200 (i.e.
                        chunks of 200x200x200 voxels) appears to be a good value
                      - set to ``0`` to force processing in one go

    For `chunk_size != 0`:

    pad_chunks :    bool
                    If True, will pad each chunk. This helps making meshes
                    watertight but may introduce internal faces when merging
                    mesh fragments.
    merge_fragments :  bool
                    If True, will attempt to merge fragments at the chunk
                    boundaries.
    progress :      bool
                    Whether to show a progress bar.

    Returns
    -------
    mesh :          trimesh.Trimesh | MeshNeuron
                    Returns a trimesh or MeshNeuron depending on the input.
                    Data tables (e.g. `connectors`) are not carried over from
                    input neuron.

    """
    if not skimage:
        raise ImportError('Meshing requires `skimage`:\n '
                          'pip3 install scikit-image')

    utils.eval_param(vox, 'vox', allowed_types=(core.VoxelNeuron, np.ndarray))

    if spacing == 'auto':
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

    if chunk_size == 'auto':
        if len(voxels) > 1e6:
            chunk_size = 200
        else:
            chunk_size = 0

    if not chunk_size:
        mesh = _mesh_from_voxels_single(voxels=voxels,
                                        spacing=spacing,
                                        step_size=step_size)
    else:
        mesh = _mesh_from_voxels_chunked(voxels=voxels,
                                         spacing=spacing,
                                         step_size=step_size,
                                         chunk_size=chunk_size,
                                         pad_chunks=pad_chunks,
                                         merge_fragments=merge_fragments,
                                         progress=progress)

    if isinstance(vox, core.VoxelNeuron):
        mesh.vertices += vox.offset
        mesh = core.MeshNeuron(mesh, units=f'1 {vox.units.units}', id=vox.id)

    return mesh


def _mesh_from_voxels_single(voxels, spacing=(1, 1, 1), step_size=1):
    """Generate mesh from voxels using marching cubes in one go.

    Better meshes but potentially slower and much less memory efficient.

    Parameters
    ----------
    voxels :        np.array
                    Voxel coordindates. Array of either (N, 3) or (N, 4).
                    (N, 3) will be interpreted as single x, y, z voxel coordinates.
                    (N, 4) will be interpreted as blocks of z, y, x_start, x_end
                    coordindates.
    spacing :       np.array
                    (3, ) array with x, y, z voxel size.
    step_size :     int, optional
                    Step size for marching cube algorithm.
                    Higher values = faster but coarser.

    Returns
    -------
    trimesh.Trimesh

    """
    # Turn voxels into matrix
    # Add border to matrix - otherwise marching cube generates holes
    mat, offset = _voxels_to_matrix(voxels, pad=1)

    # Use marching cubes to create surface model
    # (newer versions of skimage have a "marching cubes" function and
    # the marching_cubes_lewiner is deprecreated)
    marching_cubes = getattr(measure, 'marching_cubes',
                             getattr(measure, 'marching_cubes_lewiner', None))
    verts, faces, normals, values = marching_cubes(mat,
                                                   level=.5,
                                                   step_size=step_size,
                                                   allow_degenerate=False,
                                                   gradient_direction='ascent',
                                                   spacing=spacing)

    # Compensate for earlier padding offset
    verts -= np.array(spacing) * 1

    # Add offset
    verts += offset * spacing

    return tm.Trimesh(verts, faces)


def _mesh_from_voxels_chunked(voxels,
                              spacing=(1, 1, 1),
                              step_size=1,
                              chunk_size=200,
                              pad_chunks=True,
                              merge_fragments=True,
                              progress=True):
    """Generate mesh from voxels in chunks using marching cubes.

    Potentially faster and much more memory efficient but might introduce
    internal faces and/or holes.

    Parameters
    ----------
    voxels :        np.array
                    Voxel coordindates. Array of (N, 3) XYZ indices.
    spacing :       np.array
                    (3, ) array with x, y, z voxel size.
    step_size :     int, optional
                    Step size for marching cube algorithm.
                    Higher values = faster but coarser.
    chunk_size :    int
                    Size of the cubes in voxels in which to process the data.
                    The bigger the chunks, the more memory is used but there is
                    less chance of errors in the mesh.
    pad_chunks :    bool
                    If True, will pad each chunk. This helps making meshes
                    watertight but may introduce internal faces when merging
                    mesh fragments.
    merge_fragments :  bool
                    If True, will attempt to merge fragments at the chunk
                    boundaries.

    Returns
    -------
    trimesh.Trimesh

    """
    # Use marching cubes to create surface model
    # (newer versions of skimage have a "marching cubes" function and
    # the marching_cubes_lewiner is deprecreated)
    marching_cubes = getattr(measure, 'marching_cubes',
                             getattr(measure, 'marching_cubes_lewiner', None))

    # Strip the voxels
    offset = voxels.min(axis=0)
    voxels = voxels - offset

    # Map voxels to chunks
    chunks = (voxels / chunk_size).astype(int)

    # Find the largest index
    # max_ix = chunks.max()
    # base = math.ceil(np.sqrt(max_ix))
    base = 16  # 2**16=65,536 max index - this should be sufficient for chunks

    # Now we encode the indices (x, y, z) chunk indices as packed integer:
    # Each (xyz) chunk is encoded as single integer which speeds things up a lot
    # For example with base = 16, chunk (1, 2, 3) becomes:
    # N = 2 ** 16 = 65,536
    # (1 * N ** 2) + (2 * N) + 3 = 4,295,098,371
    # This obviously only works as long as we can still squeeze the chunks into
    # 64bit integers but that should work out even at scale 0. If this ever
    # becomes an issue, we could start using strings instead. For now, base 16
    # should be enough.
    chunks_packed = pack_array(chunks, base=base)

    # Find unique chunks
    chunks_unique = unique(chunks_packed)

    # For each chunk also find voxels that are directly adjacent (in plus direction)
    # This makes it so that each voxel can belong to multiple chunks
    # What we want to get is an (4, N) array where for each voxel we have its
    # "original" chunk index and its chunks when offset by -1 along each axis.
    # original chunk -> [[1, 1, 1, 0, 0, 0, 0],
    # offset x by -1 ->  [1, 1, 1, 1, 0, 1, 0],
    # offset y by -1 ->  [1, 1, 1, 0, 1, 0, 0]
    # offset z by -1 ->  [1, 1, 1, 0, 0, 0, 1]]
    # The simple numbers (0, 1) in this example will obvs be our packed integers
    # Later on we can then ask: "find me all voxels in chunk 0, 1, etc."
    voxel2chunk = np.full((4, len(voxels)), fill_value=-1, dtype=int)
    voxel2chunk[-1, :] = chunks_packed

    # Find offset chunks along each axis
    for k in range(3):
        # Offset chunks and pack
        chunks_offset = voxels.copy()
        chunks_offset[:, k] -= 1
        chunks_offset = (chunks_offset / chunk_size).astype(int)
        chunks_offset = pack_array(chunks_offset, base=base)

        voxel2chunk[k] = chunks_offset

    # Unpack the unique chunks
    chunks_unique_unpacked = unpack_array(chunks_unique, base=base)

    # Generate the fragments
    fragments = []
    end_chunks = chunks_unique_unpacked.max(axis=0)
    pad = np.array([[1, 1], [1, 1], [1, 1]])
    for i, (ch, ix) in config.tqdm(enumerate(zip(chunks_unique, chunks_unique_unpacked)),
                                   total=len(chunks_unique),
                                   disable=not progress,
                                   leave=False,
                                   desc='Meshing'):
        # Pad the matrices only for the first and last chunks along each axis
        if not pad_chunks:
            pad = np.array([[0, 0], [0, 0], [0, 0]])
            for k in range(3):
                if ix[k] == 0:
                    pad[k][0] = 1
                if ix[k] == end_chunks[k]:
                    pad[k][1] = 1

        # Get voxels in this chunk
        this_vx = voxels[np.any(voxel2chunk == ch, axis=0)]

        # If only a single voxel, skip.
        if this_vx.shape[0] <= 1:
            continue

        # Turn voxels into matrix with given padding
        mat, chunk_offset = _voxels_to_matrix(this_vx, pad=pad.tolist())

        # Marching cubes needs at least a (2, 2, 2) matrix
        # We could in theory make this work by adding padding
        # but we probably don't loose much if we skip them.
        # There is a chance that we might introduce holes in the mesh
        # though
        if any([s < 2 for s in mat.shape]):
            continue

        # Run the actual marching cubes
        v, f, _, _ = marching_cubes(mat,
                                    level=.5,
                                    step_size=step_size,
                                    allow_degenerate=False,
                                    gradient_direction='ascent',
                                    spacing=(1, 1, 1))

        # Remove the padding (if any)
        v -= pad[:, 0]
        # Add chunk offset
        v += chunk_offset

        fragments.append((v, f))

    # Combine into a single mesh
    all_verts = []
    all_faces = []
    verts_offset = 0
    for frag in fragments:
        all_verts.append(frag[0])
        all_faces.append(frag[1] + verts_offset)
        verts_offset += len(frag[0])

    all_verts = (np.concatenate(all_verts, axis=0) + offset) * spacing
    all_faces = np.concatenate(all_faces, axis=0)

    # Make trimesh
    m = tm.Trimesh(all_verts, all_faces)

    # Deduplicate chunk boundaries
    # This is not necessarily the cleanest solution but it should do the
    # trick most of the time
    if merge_fragments:
        m.merge_vertices(digits_vertex=1)

    return m


def pack_array(arr, base=8):
    """Pack 2-d array along second axis."""
    N = 2 ** base
    packed = np.zeros(arr.shape, dtype='uint64')
    packed[:, 0] = arr[:, 0] * N ** 2
    packed[:, 1] = arr[:, 1] * N
    packed[:, 2] = arr[:, 2]
    return packed.sum(axis=1)


def unpack_array(arr, base=8):
    """Unpack 2-d array."""
    N = 2 ** base - 1
    unpacked = np.zeros((arr.shape[0], 3), dtype='uint64')
    unpacked[:, 0] = (arr >> (base * 2)) & N
    unpacked[:, 1] = (arr >> base) & N
    unpacked[:, 2] = arr & N
    return unpacked


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
