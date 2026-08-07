#    This script is part of navis (http://www.github.com/navis-org/navis).
#    Copyright (C) 2017 Philipp Schlegel
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
#
#    You should have received a copy of the GNU General Public License
#    along

"""Module contains functions to plot neurons in 2D and 3D."""

from .. import config, core, utils

import math
import random

import numpy as np

from collections.abc import Iterable
from typing import Tuple, Optional, List


__all__ = [
    "tn_pairs_to_coords",
    "segments_to_coords",
    "skeleton_capsules",
    "mesh_faces",
    "fibonacci_sphere",
    "make_tube",
]

logger = config.get_logger(__name__)


def tn_pairs_to_coords(
    x: core.Skeleton, modifier: Optional[Tuple[float, float, float]] = (1, 1, 1)
) -> np.ndarray:
    """Return pairs of child->parent node coordinates.

    Parameters
    ----------
    x :         Skeleton
                Must contain the nodes.
    modifier :  ints, optional
                Use to modify/invert x/y/z axes.

    Returns
    -------
    coords :    np.array
                `[[[x1, y1, z1], [x2, y2, z2]], [[x3, y3, y4], [x4, y4, z4]]]`

    """
    if not isinstance(modifier, np.ndarray):
        modifier = np.array(modifier)

    nodes = x.nodes[x.nodes.parent_id >= 0]

    tn_co = nodes.loc[:, ["x", "y", "z"]].values
    parent_co = (
        x.nodes.set_index("node_id").loc[nodes.parent_id.values, ["x", "y", "z"]].values
    )

    coords = np.append(tn_co, parent_co, axis=1)

    if any(modifier != 1):
        coords *= modifier

    return coords.reshape((coords.shape[0], 2, 3))


def segments_to_coords(
    x: core.Skeleton,
    modifier: Optional[Tuple[float, float, float]] = (1, 1, 1),
    node_colors: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Turn lists of node IDs into coordinates.

    Runs on navis-fastcore.

    Parameters
    ----------
    x :             Skeleton
                    Must contain the nodes
    node_colors :   numpy.ndarray, optional
                    A color for each node in `x.nodes`. If provided, will
                    also return a list of colors sorted to match coordinates.
    modifier :      ints, optional
                    Use e.g. to modify/invert x/y/z axes.

    Returns
    -------
    coords :        list of tuples
                    [(x, y, z), (x, y, z), ... ]
    colors :        list of colors
                    If `node_colors` provided will return a copy of it sorted
                    to match `coords`.

    """
    colors = None

    if node_colors is None:
        coords = utils.fastcore.segment_coords(
            x.nodes.node_id.values,
            x.nodes.parent_id.values,
            x.nodes[["x", "y", "z"]].values,
        )
    else:
        coords, colors = utils.fastcore.segment_coords(
            x.nodes.node_id.values,
            x.nodes.parent_id.values,
            x.nodes[["x", "y", "z"]].values,
            node_colors=node_colors,
        )

    modifier = np.asarray(modifier)
    if (modifier != 1).any():
        for seg in coords:
            np.multiply(seg, modifier, out=seg)

    if colors is not None:
        return coords, colors

    return coords


def skeleton_capsules(
    x: core.Skeleton,
    xy_ix: Tuple[int, int],
    depth_ix: int,
    radius_scale: float = 1.0,
    node_values: Optional[np.ndarray] = None,
    disc_res: int = 12,
) -> Tuple[List[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Outline a skeleton's neurites at their real radius, in 2D.

    Each edge becomes a rectangle perpendicular to it, with the child's radius at
    one end and the parent's at the other, and each node becomes a disc. Their
    union is the projection of a tapered tube: the discs are what round off the
    tips and stop branch points from showing a notch.

    The obvious alternative - offsetting each linear segment's polyline by its own
    radius - needs a fifth of the polygons but blows up wherever the polyline turns
    faster than its radius, and renders tips as spikes. Since the polygons are
    filled in a single opaque collection, overlap between neighbours is free.

    Note that this is a 2D construction: it relies on the projection, so it is only
    valid for `method="2d"`. 3D methods still go through
    [`navis.conversion.tree2meshneuron`][].

    Parameters
    ----------
    x :             Skeleton
                    Must have a `radius` column.
    xy_ix :         (int, int)
                    Indices of the two coordinate columns that make up the view.
    depth_ix :      int
                    Index of the remaining, into-the-screen axis.
    radius_scale :  float
                    Multiplier for the radii.
    node_values :   (N,) or (N, k) array, optional
                    Per-node values (e.g. colors) to carry over to the polygons.
                    Edges take the value of their child node.
    disc_res :      int
                    Number of vertices per node disc.

    Returns
    -------
    polygons :      list of (M, 2) arrays
                    Node discs first, then edge rectangles - so that with per-node
                    colors the edges paint over the discs and only the tips and
                    the notches at branch points show a disc's own color.
    depth :         (n_polygons,) array
                    Mean depth of each polygon, for painting back to front.
    values :        (n_polygons, k) array
                    Only if `node_values` was provided.

    """
    nodes = x.nodes
    radii = nodes.radius.fillna(0).values.astype(float) * radius_scale

    # fastcore sorts arbitrary per-node data onto the segments for us, so stack the
    # radii and whatever the caller wants to carry into one array and split later
    per_node = radii[:, None]
    if node_values is not None:
        node_values = np.asarray(node_values, dtype=float)
        if node_values.ndim == 1:
            node_values = node_values[:, None]
        per_node = np.hstack([per_node, node_values])

    coords, values = segments_to_coords(x, node_colors=per_node)

    xy_ix = list(xy_ix)
    quads, quad_depth, quad_values = [], [], []
    for co, va in zip(coords, values):
        co = np.asarray(co, dtype=float)
        va = np.asarray(va, dtype=float)
        if len(co) < 2:
            continue

        xy = co[:, xy_ix]
        r = va[:, 0]

        d = np.diff(xy, axis=0)
        length = np.hypot(d[:, 0], d[:, 1])
        # zero-length edges (duplicate nodes) would divide by zero; their quad is
        # degenerate either way and the node discs still cover the spot
        length[length == 0] = 1.0
        normal = np.column_stack([-d[:, 1], d[:, 0]]) / length[:, None]

        a, b = xy[:-1], xy[1:]
        ra, rb = r[:-1, None], r[1:, None]
        quads.append(
            np.stack(
                [a + normal * ra, b + normal * rb, b - normal * rb, a - normal * ra],
                axis=1,
            )
        )
        quad_depth.append((co[:-1, depth_ix] + co[1:, depth_ix]) / 2)
        if node_values is not None:
            # segments run tip->root, so the first node of an edge is its child -
            # matching the per-edge coloring used elsewhere
            quad_values.append(va[:-1, 1:])

    if quads:
        quads = np.concatenate(quads)
        quad_depth = np.concatenate(quad_depth)
    else:
        quads = np.zeros((0, 4, 2))
        quad_depth = np.zeros(0)

    co = nodes[["x", "y", "z"]].values
    # Clockwise, to match the winding of the quads above. Callers that fill the
    # polygons as one compound path rely on this: under the nonzero winding rule
    # two overlapping subpaths wound in *opposite* directions cancel out and
    # leave a hole where they overlap.
    angles = np.linspace(0, -2 * np.pi, disc_res, endpoint=False)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    discs = co[:, xy_ix][:, None, :] + unit[None, :, :] * radii[:, None, None]

    polygons = list(discs) + list(quads)
    depth = np.concatenate([co[:, depth_ix], quad_depth])

    if node_values is None:
        return polygons, depth, None

    quad_values = (
        np.concatenate(quad_values)
        if quad_values
        else np.zeros((0, per_node.shape[1] - 1))
    )
    return polygons, depth, np.concatenate([node_values, quad_values])


def mesh_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    xy_ix: Tuple[int, int],
    depth_ix: int,
    front: int = 1,
    smooth: bool = False,
    normals: bool = True,
    order: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Project a mesh's triangles into the view plane, back to front.

    Faces pointing away from the viewer are dropped and the rest are sorted along
    the view axis, which between them are what turn a flat blob into something with
    an inside and an outside: for a closed mesh the front faces are exactly what you
    would see, and painting them far-to-near gives correct occlusion. Culling also
    roughly halves the number of polygons the renderer ever sees.

    The projection is the same one the rest of `plot2d` uses - coordinates are never
    flipped, since a `"-x"` style view is handled by inverting the axis. `front` is
    therefore needed to say which end of the depth axis the viewer is on.

    Parameters
    ----------
    vertices :  (N, 3) array
    faces :     (M, 3) array
    xy_ix :     (int, int)
                Indices of the two coordinate columns that make up the view.
    depth_ix :  int
                Index of the remaining, into-the-screen axis.
    front :     1 | -1
                Sign along `depth_ix` that points at the viewer.
    smooth :    bool
                Shade with area-weighted vertex normals averaged back over each
                face rather than with the face's own normal. Culling always uses
                the geometric normal, so the silhouette is unaffected - this only
                changes how a coarse mesh takes the light.
    normals :   bool
                Return unit normals. Implied by `smooth`. Off is for callers that
                are not shading: the normals cost a cross product per surviving
                face and nothing else needs them.
    order :     bool
                Return the triangles sorted furthest-first, and with them their
                depths. Off is for callers that fill the whole mesh as one path
                in one colour - the nonzero winding rule is blind to the order
                its subpaths arrive in, so for those the sort cannot change a
                pixel, and it is the most expensive step left here.

    Returns
    -------
    tri :       (K, 3, 2) array
                Projected front-facing triangles, furthest first (unless `order`
                is off, in which case they keep their order in `faces`).
    normals :   (K, 3) array or None
                Unit normals to shade them with. None if `normals` is off.
    depth :     (K,) array or None
                Mean depth of each triangle along `depth_ix` (not sign-corrected,
                so it can be handed straight to depth coloring). None if `order`
                is off.
    ix :        (K,) array
                Index of each triangle in the original `faces`.

    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    # Project up front and work off the result. It is what the caller gets back,
    # and - see `_cull_backfaces` - it is also all the cull needs, so the depth
    # column is never widened and the (M, 3, 3) block every corner would
    # otherwise be gathered into is never built. That block is the single
    # largest allocation the old version made, and reading it a column at a time
    # strided over every triangle cost more than the arithmetic it fed.
    xy = np.ascontiguousarray(vertices[:, list(xy_ix)], dtype=np.float64)
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]

    raw = None
    if smooth:
        # Vertex normals take a contribution from every face, back-facing ones
        # included, so smooth shading needs the full cross whatever happens.
        # Compute it once off contiguous corners and let the cull read its sign
        # off the same array. The cross product's length is twice the triangle's
        # area, which is exactly the weighting we want when accumulating vertex
        # normals - so normalise it late.
        v3 = np.asarray(vertices, dtype=np.float64)
        p = v3[f0]
        raw = np.cross(v3[f1] - p, v3[f2] - p)
        del p
        keep = np.flatnonzero(raw[:, depth_ix] * front > 0)
    else:
        keep = _cull_backfaces(xy, faces, front * _cull_sign(xy_ix, depth_ix))

    if order:
        vd = vertices[:, depth_ix]
        depth = (vd[f0[keep]].astype(np.float64) + vd[f1[keep]] + vd[f2[keep]]) / 3
        o = np.argsort(depth * front)
        ix, depth = keep[o], depth[o]
    else:
        ix, depth = keep, None

    kept = faces[ix]
    if smooth:
        # bincount rather than `np.add.at`, which is unbuffered and ~3x slower
        flat = faces.ravel()
        vn = np.stack(
            [
                np.bincount(flat, np.repeat(raw[:, k], 3), len(vertices))
                for k in range(3)
            ],
            axis=1,
        )
        del raw
        out = _normalize(_normalize(vn)[kept].mean(axis=1))
    elif normals:
        # only the survivors need a normal, which is about half the mesh
        v3 = np.asarray(vertices, dtype=np.float64)
        p = v3[kept[:, 0]]
        out = _normalize(np.cross(v3[kept[:, 1]] - p, v3[kept[:, 2]] - p))
    else:
        out = None

    return xy[kept], out, depth, ix


def _cull_sign(xy_ix: Tuple[int, int], depth_ix: int) -> int:
    """+1 if `xy_ix` is already the cyclic pair for `depth_ix`, else -1.

    `cross(e1, e2)[depth_ix]` is a 2x2 determinant of the two *other* columns,
    taken in cyclic order. `xy_ix` is that same pair of columns but in whichever
    order the view asked for, so swapping them flips the sign of the determinant
    and with it which side of the surface the cull keeps.
    """
    return 1 if tuple(xy_ix) == ((depth_ix + 1) % 3, (depth_ix + 2) % 3) else -1


def _cull_backfaces(xy: np.ndarray, faces: np.ndarray, sign: int) -> np.ndarray:
    """Indices of the faces that wind `sign`-wards once projected onto `xy`.

    Telling front from back only needs the sign of the face normal's depth
    component, and that component is a determinant of the two *projected* edges -
    so the cull runs entirely off `xy` and never looks at the depth column or
    builds the other two components of the cross product.

    Done in blocks, which caps the intermediates at a few arrays the length of a
    block rather than of the mesh. That is a bound on what an arbitrarily large
    mesh costs here, and it is also faster, since the intermediates stay in cache.
    """
    parts = []
    for s in range(0, len(faces), _CULL_BLOCK):
        g0 = faces[s : s + _CULL_BLOCK, 0]
        g1 = faces[s : s + _CULL_BLOCK, 1]
        g2 = faces[s : s + _CULL_BLOCK, 2]
        x0, y0 = xy[g0, 0], xy[g0, 1]
        nd = (xy[g1, 0] - x0) * (xy[g2, 1] - y0)
        nd -= (xy[g1, 1] - y0) * (xy[g2, 0] - x0)
        block = np.flatnonzero(nd * sign > 0)
        block += s
        parts.append(block)

    if not parts:
        return np.empty(0, dtype=np.intp)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


#: Faces per block in `_cull_backfaces`. Large enough that the per-block overhead
#: is noise, small enough that the intermediates are not the peak allocation.
_CULL_BLOCK = 1 << 22


def _normalize(v: np.ndarray) -> np.ndarray:
    """Scale vectors to unit length, leaving zero-length ones alone."""
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(length == 0, 1, length)


def fibonacci_sphere(samples: int = 1, randomize: bool = True) -> list:
    """Generate points on a sphere."""
    rnd = 1.0
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2.0 / samples
    increment = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return np.array(points)


def make_tube(segments, radii=1.0, tube_points=8, use_normals=True):
    """Generate tube mesh (vertices + faces) from lines.

    This code was modified from the vispy library.

    Parameters
    ----------
    segments :      list
                    List of lists of x/y/z coordinates.
    radii :         float | list of floats
                    Either a single radius used for all nodes or list of lists of
                    floats with the same shape as `segments`.
    tube_points :   int
                    Number of points making up the circle of the cross-section
                    of the tube.
    use_normals :   bool
                    If True will rotate tube along it's curvature.

    Returns
    -------
    vertices :      np.ndarray
    faces :         np.ndarray

    """
    vertices = np.empty((0, 3), dtype=np.float64)
    indices = np.empty((0, 3), dtype=np.uint32)

    if not isinstance(radii, Iterable):
        radii = [[radii] * len(points) for points in segments]

    for points, radius in zip(segments, radii):
        # Need to make sure points are floats
        points = np.array(points).astype(float)

        # Skip single points
        if len(points) < 2:
            continue

        if use_normals:
            _, normals, binormals = _frenet_frames(points)
        else:
            _ = normals = binormals = np.ones((len(points), 3))

        n_segments = len(points) - 1

        if not isinstance(radius, Iterable):
            radius = [radius] * len(points)

        radius = np.array(radius)

        # Vertices for each point on the circle
        verts = np.repeat(points, tube_points, axis=0)

        v = np.arange(tube_points, dtype=np.float64) / tube_points * 2 * np.pi

        all_cx = (
            radius
            * -1.0
            * np.tile(np.cos(v), points.shape[0]).reshape(
                (tube_points, points.shape[0]), order="F"
            )
        ).T
        cx_norm = (all_cx[:, :, np.newaxis] * normals[:, np.newaxis, :]).reshape(
            verts.shape
        )

        all_cy = (
            radius
            * np.tile(np.sin(v), points.shape[0]).reshape(
                (tube_points, points.shape[0]), order="F"
            )
        ).T
        cy_norm = (all_cy[:, :, np.newaxis] * binormals[:, np.newaxis, :]).reshape(
            verts.shape
        )

        verts = verts + cx_norm + cy_norm

        # Generate indices for the first segment
        ix = np.arange(0, tube_points)

        # Repeat indices n_segments-times
        ix = np.tile(ix, n_segments)

        # Offset indices by number segments and tube points
        offsets = np.repeat((np.arange(0, n_segments)) * tube_points, tube_points)
        ix += offsets

        # Turn indices into faces
        ix_a = ix
        ix_b = ix + tube_points

        ix_c = ix_b.reshape((n_segments, tube_points))
        ix_c = np.append(ix_c[:, 1:], ix_c[:, [0]], axis=1)
        ix_c = ix_c.ravel()

        ix_d = ix_a.reshape((n_segments, tube_points))
        ix_d = np.append(ix_d[:, 1:], ix_d[:, [0]], axis=1)
        ix_d = ix_d.ravel()

        faces1 = np.concatenate((ix_a, ix_b, ix_d), axis=0).reshape(
            (n_segments * tube_points, 3), order="F"
        )
        faces2 = np.concatenate((ix_b, ix_c, ix_d), axis=0).reshape(
            (n_segments * tube_points, 3), order="F"
        )

        faces = np.append(faces1, faces2, axis=0)

        # Offset faces against already existing vertices
        faces += vertices.shape[0]

        # Add vertices and faces to total collection
        vertices = np.append(vertices, verts, axis=0)
        indices = np.append(indices, faces, axis=0)

    return vertices, indices


def _frenet_frames(points):
    """Calculate and return the tangents, normals and binormals for the tube.

    This code was modified from the vispy library.

    """
    tangents = np.zeros((len(points), 3))
    normals = np.zeros((len(points), 3))

    epsilon = 0.0001

    # Compute tangent vectors for each segment
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)

    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]

    mags = np.sqrt(np.sum(tangents * tangents, axis=1))

    # Make sure we don't have any zeros in `mags` that would mess with the
    # subdivision ->  we will set those to the smallest possible value
    mags[mags == 0] = np.finfo(mags.dtype).resolution

    tangents /= mags[:, np.newaxis]

    # Get initial normal and binormal
    t = np.abs(tangents[0])

    smallest = np.argmin(t)
    normal = np.zeros(3)
    normal[smallest] = 1.0

    vec = np.cross(tangents[0], normal)
    normals[0] = np.cross(tangents[0], vec)

    all_vec = np.cross(tangents[:-1], tangents[1:])
    all_vec_norm = np.linalg.norm(all_vec, axis=1)

    # Normalise vectors if necessary
    where = all_vec_norm > epsilon
    all_vec[where, :] /= all_vec_norm[where].reshape((sum(where), 1))

    # Precompute inner dot product
    dp = np.sum(tangents[:-1] * tangents[1:], axis=1)
    # Clip
    cl = np.clip(dp, -1, 1)
    # Get theta
    th = np.arccos(cl)

    # Compute normal and binormal vectors along the path
    for i in range(1, len(points)):
        normals[i] = normals[i - 1]

        vec_norm = all_vec_norm[i - 1]
        vec = all_vec[i - 1]
        if vec_norm > epsilon:
            normals[i] = rotate(-np.degrees(th[i - 1]), vec)[:3, :3].dot(normals[i])

    binormals = np.cross(tangents, normals)

    return tangents, normals, binormals


def rotate(angle, axis, dtype=None):
    """Generate a 4x4 rotation matrix for rotation about a vector.

    Modified from `vispy.utils.transforms`.

    Parameters
    ----------
    angle :     float
                The angle of rotation, in degrees.
    axis :      ndarray
                The x, y, z coordinates of the axis direction vector.

    Returns
    -------
    M :     ndarray
            Transformation matrix describing the rotation.

    """
    angle = np.radians(angle)
    assert len(axis) == 3
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    M = np.array(
        [
            [cx * x + c, cy * x - z * s, cz * x + y * s, 0.0],
            [cx * y + z * s, cy * y + c, cz * y - x * s, 0.0],
            [cx * z - y * s, cy * z + x * s, cz * z + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype,
    ).T
    return M
