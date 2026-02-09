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

import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def h5reg_trilinear_sample(volume, z, y, x, cval):
    """Trilinear interpolation of a 3D volume at continuous coordinates.

    Parameters
    ----------
    volume : ndarray
        3D array to sample from.
    z, y, x : float
        Continuous coordinates at which to sample.
    cval : float
        Value to return for out-of-bounds coordinates.

    Returns
    -------
    float
        Interpolated value at (z, y, x).

    """
    z0 = int(np.floor(z))
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))

    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1

    if (
        z0 < 0
        or y0 < 0
        or x0 < 0
        or z1 >= volume.shape[2]
        or y1 >= volume.shape[1]
        or x1 >= volume.shape[0]
    ):
        return cval

    dz = z - z0
    dy = y - y0
    dx = x - x0

    c000 = volume[x0, y0, z0]
    c001 = volume[x0, y0, z1]
    c010 = volume[x0, y1, z0]
    c011 = volume[x0, y1, z1]
    c100 = volume[x1, y0, z0]
    c101 = volume[x1, y0, z1]
    c110 = volume[x1, y1, z0]
    c111 = volume[x1, y1, z1]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy

    return c0 * (1 - dz) + c1 * dz


@nb.njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def h5reg_trilinear_sample_vec(field, z, y, x, cval):
    """Trilinear interpolation of a 3D vector field at continuous coordinates.

    Parameters
    ----------
    field : ndarray
        4D array (z, y, x, 3) containing 3-component vectors at each voxel.
    z, y, x : float
        Continuous coordinates at which to sample.
    cval : float
        Value to return for out-of-bounds coordinates.

    Returns
    -------
    tuple of float
        Interpolated 3-component vector (dx, dy, dz) at (z, y, x).

    """
    z0 = int(np.floor(z))
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))

    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1

    if (
        z0 < 0
        or y0 < 0
        or x0 < 0
        or z1 >= field.shape[0]
        or y1 >= field.shape[1]
        or x1 >= field.shape[2]
    ):
        return cval, cval, cval

    dz = z - z0
    dy = y - y0
    dx = x - x0

    out0 = 0.0
    out1 = 0.0
    out2 = 0.0

    for c in range(3):
        c000 = field[z0, y0, x0, c]
        c001 = field[z0, y0, x1, c]
        c010 = field[z0, y1, x0, c]
        c011 = field[z0, y1, x1, c]
        c100 = field[z1, y0, x0, c]
        c101 = field[z1, y0, x1, c]
        c110 = field[z1, y1, x0, c]
        c111 = field[z1, y1, x1, c]

        c00 = c000 * (1 - dx) + c001 * dx
        c01 = c010 * (1 - dx) + c011 * dx
        c10 = c100 * (1 - dx) + c101 * dx
        c11 = c110 * (1 - dx) + c111 * dx

        c0 = c00 * (1 - dy) + c01 * dy
        c1 = c10 * (1 - dy) + c11 * dy

        val = c0 * (1 - dz) + c1 * dz
        if c == 0:
            out0 = val
        elif c == 1:
            out1 = val
        else:
            out2 = val

    return out0, out1, out2


@nb.njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def h5reg_warp_image_linear_constant(
    image,
    field,
    spacing,
    qmult,
    affine,
    apply_affine,
    out,
    out_res,
    image_res,
    cval,
):
    """Warp a 3D image using an H5 deformation field with numba acceleration.

    Uses backward mapping: for each output voxel, applies the deformation field
    to find the corresponding source location, then interpolates the source image.
    Supports optional affine pre-transformation and constant-mode boundary handling.

    Parameters
    ----------
    image : ndarray
        Source 3D image to warp.
    field : ndarray
        Deformation field (z, y, x, 3) with offsets at each voxel.
    spacing : ndarray
        Voxel spacing (z, y, x) of the deformation field.
    qmult : float
        Quantization multiplier to scale deformation field values.
    affine : ndarray
        4x4 affine transformation matrix.
    apply_affine : int
        Whether to apply the affine before (1), after (2) the
        transform or not at all (0).
    out : ndarray
        Preallocated output array to store the warped image.
    out_res : ndarray
        Output voxel resolution (x, y, z).
    image_res : ndarray
        Source image voxel resolution (x, y, z).
    cval : float
        Constant value for out-of-bounds pixels.

    Returns
    -------
    out : ndarray
        Warped output image.

    """
    # Determine maximum valid coordinates for sampling to avoid out-of-bounds access
    max_z = field.shape[0] - 1.000001
    max_y = field.shape[1] - 1.000001
    max_x = field.shape[2] - 1.000001

    # Iterate over all voxels in the output image
    for z in nb.prange(out.shape[2]):
        for y in range(out.shape[1]):
            for x in range(out.shape[0]):
                px = x * out_res[0]
                py = y * out_res[1]
                pz = z * out_res[2]

                # Apply affine transformation if specified and
                # should come after the lookup (inverse direction)
                if apply_affine == 1:
                    tx = (
                        affine[0, 0] * px
                        + affine[0, 1] * py
                        + affine[0, 2] * pz
                        + affine[0, 3]
                    )
                    ty = (
                        affine[1, 0] * px
                        + affine[1, 1] * py
                        + affine[1, 2] * pz
                        + affine[1, 3]
                    )
                    tz = (
                        affine[2, 0] * px
                        + affine[2, 1] * py
                        + affine[2, 2] * pz
                        + affine[2, 3]
                    )
                    px = tx
                    py = ty
                    pz = tz

                # Convert physical coordinates to voxel coordinates in the deformation field
                vx = px / spacing[2]
                vy = py / spacing[1]
                vz = pz / spacing[0]

                # Clamp coordinates to valid range to avoid out-of-bounds access during interpolation
                if vx < 0:
                    vx = 0.0
                elif vx > max_x:
                    vx = max_x
                if vy < 0:
                    vy = 0.0
                elif vy > max_y:
                    vy = max_y
                if vz < 0:
                    vz = 0.0
                elif vz > max_z:
                    vz = max_z

                # Sample the deformation field at the voxel coordinates to get the offsets
                dx, dy, dz = h5reg_trilinear_sample_vec(field, vz, vy, vx, 0.0)

                px = px + dx * qmult
                py = py + dy * qmult
                pz = pz + dz * qmult

                # Apply affine transformation if specified and
                # should come after the lookup (forward direction)
                if apply_affine == 2:
                    tx = (
                        affine[0, 0] * px
                        + affine[0, 1] * py
                        + affine[0, 2] * pz
                        + affine[0, 3]
                    )
                    ty = (
                        affine[1, 0] * px
                        + affine[1, 1] * py
                        + affine[1, 2] * pz
                        + affine[1, 3]
                    )
                    tz = (
                        affine[2, 0] * px
                        + affine[2, 1] * py
                        + affine[2, 2] * pz
                        + affine[2, 3]
                    )
                    px = tx
                    py = ty
                    pz = tz

                # Convert physical coordinates to source image voxel coordinates
                ix = px / image_res[0]
                iy = py / image_res[1]
                iz = pz / image_res[2]

                out[x, y, z] = h5reg_trilinear_sample(image, iz, iy, ix, cval)

    return out
