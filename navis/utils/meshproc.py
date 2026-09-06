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

"""Fast stand-ins for what trimesh does when a mesh is built with `process=True`.

That pass drops non-finite vertices and merges duplicate ones, and on a
million-vertex neuron the merge is ~95% of it. Almost all of *that* is a single
`np.unique` over rows viewed as raw bytes.

Trimesh does try to avoid this: `grouping.hashable_rows` packs a row into one
64-bit integer when it fits, which needs ~21 bits per column. But
`merge_vertices` first quantizes the coordinates to `tol.merge` (1e-8), i.e.
multiplies them by 1e8, which for a mesh in nanometres overshoots that budget by
a factor of ~1e6. The fast path would need coordinates below 0.01, so in
practice every mesh in real units falls through to a `np.void` view and a sort
driven by a generic element-by-element comparator.

The way out is to hand that sort one integer per row after all. Coordinates
cannot be packed into 64 bits at 1e-8 precision, but their *ranks* can - what
then bounds the packing is how many distinct values a column has, not how far
the mesh extends. The map stays injective, so this is a packing and not a hash:
two vertices can never be merged by accident.

Everything here returns exactly what trimesh returns - same vertices, same
faces, same order - only faster: ~8x on the full `process=True` pass for a
500k-vertex mesh, ~7x for a 1.4M-vertex one.
"""

import numpy as np
import pandas as pd

from trimesh import util
from trimesh.constants import tol

__all__ = ["unique_rows", "merge_vertices", "process"]

_INT64_MAX = int(np.iinfo(np.int64).max)


def unique_rows(rows):
    """Group identical rows, numbered in order of first occurrence.

    Equivalent to `trimesh.grouping.unique_rows(rows, keep_order=True)`, but
    without the sort: rows are folded into one integer each and grouped by hash.

    Parameters
    ----------
    rows :      (N, M) array
                Integers. Floats would make the packing lossy and are rejected
                by the callers here, which quantize before calling.

    Returns
    -------
    index :     (K, ) int array
                `rows[index]` are the distinct rows, in the order they first
                appear.
    inverse :   (N, ) int array
                Group of each row, such that `rows[index][inverse] == rows`.

    """
    rows = np.asarray(rows)
    n = len(rows)
    if not n:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)

    # `sort=False` numbers the groups in order of first appearance, which is the
    # order asked for here - so no renumbering is needed, and `index` below comes
    # out ascending on its own.
    group, uniq = pd.factorize(_row_key(rows), sort=False)
    group = group.astype(np.intp, copy=False)

    # Lowest row index per group. Scattering back to front means the last write
    # wins, and the last write is the earliest row.
    index = np.empty(len(uniq), dtype=np.intp)
    index[group[::-1]] = np.arange(n - 1, -1, -1)
    return index, group


def _row_key(rows):
    """One int64 per row, equal exactly where two rows are equal.

    Each column is replaced by its rank code before being packed, so what has to
    fit into 64 bits is the number of *distinct* values in a column rather than
    its extent. Coordinates quantized to 1e-8 span ~1e12 per axis on a neuron,
    which no packing could hold; the ~1.5e5 distinct values they take fit with
    room to spare.

    """
    key, span = np.zeros(len(rows), dtype=np.int64), 1
    for col in range(rows.shape[1]):
        code, uniq = pd.factorize(rows[:, col], sort=False)
        dim = len(uniq)
        if span * dim - 1 > _INT64_MAX:
            # No room for another column. Re-code what we have, which bounds the
            # span by the row count again - so this can never run out, only cost
            # an extra pass on a mesh with very many distinct values.
            key, uniq = pd.factorize(key, sort=False)
            key, span = key.astype(np.int64), len(uniq)
        key = key * dim + code
        span *= dim
    return key


def merge_vertices(
    mesh,
    merge_tex=None,
    merge_norm=None,
    digits_vertex=None,
    digits_norm=None,
    digits_uv=None,
):
    """Drop-in for `trimesh.grouping.merge_vertices`.

    Same signature, same semantics, same result - see this module's docstring
    for what is different underneath. Operates in place via
    `mesh.update_vertices`, so visuals, normals and vertex attributes are
    carried along exactly as they are by trimesh's version.

    Parameters
    ----------
    mesh :              trimesh.Trimesh
    merge_tex :         bool, optional
                        If True, merge vertices regardless of UV coordinates.
    merge_norm :        bool, optional
                        If True, merge vertices regardless of vertex normals.
    digits_vertex :     int, optional
                        Decimal digits to consider for vertex positions.
                        Defaults to trimesh's `tol.merge`.
    digits_norm :       int, optional
                        Decimal digits to consider for unit normals.
    digits_uv :         int, optional
                        Decimal digits to consider for UV coordinates.

    """
    if len(mesh.vertices) == 0:
        return

    merge_tex = False if merge_tex is None else merge_tex
    merge_norm = False if merge_norm is None else merge_norm
    digits_norm = 2 if digits_norm is None else digits_norm
    digits_uv = 4 if digits_uv is None else digits_uv
    if digits_vertex is None:
        digits_vertex = util.decimal_to_digits(tol.merge)

    # Unreferenced vertices are dropped rather than grouped - they cost work and
    # no face names them.
    if hasattr(mesh, "faces") and len(mesh.faces) > 0:
        referenced = np.zeros(len(mesh.vertices), dtype=bool)
        referenced[mesh.faces] = True
    else:
        referenced = np.ones(len(mesh.vertices), dtype=bool)

    stacked = [mesh.vertices * (10**digits_vertex)]

    # A textured mesh has to keep vertices that differ only in UV apart, and
    # likewise for vertex normals - so those ride along as extra columns.
    if (
        not merge_tex
        and mesh.visual.defined
        and mesh.visual.kind == "texture"
        and mesh.visual.uv is not None
        and len(mesh.visual.uv) == len(mesh.vertices)
    ):
        stacked.append(mesh.visual.uv * (10**digits_uv))

    normals = mesh._cache["vertex_normals"]
    if not merge_norm and np.shape(normals) == mesh.vertices.shape:
        stacked.append(normals * (10**digits_norm))

    # Usually just the coordinates, and `column_stack` would copy them for
    # nothing.
    stacked = stacked[0] if len(stacked) == 1 else np.column_stack(stacked)
    stacked = stacked.round().astype(np.int64)

    keep, groups = unique_rows(stacked if referenced.all() else stacked[referenced])

    inverse = np.zeros(len(mesh.vertices), dtype=np.int64)
    inverse[referenced] = groups
    mask = np.flatnonzero(referenced)[keep]
    mesh.update_vertices(mask=mask, inverse=inverse)


def process(vertices, faces):
    """Trimesh's `process=True`, on bare arrays.

    Drops non-finite vertices, then merges duplicates - the two passes
    `trimesh.Trimesh(..., process=True, validate=False)` runs, with the same
    result but without building a mesh object to throw away afterwards. Where
    there *is* an object to build, install :func:`merge_vertices` on it as
    `navis.Volume` and `navis.utils.TrimeshPlus` do - the merge then follows the
    object rather than happening once on the way in.

    Note that `validate=True` adds a face-level clean-up which this does not
    do; callers that want it should go through trimesh (see `navis.Mesh`).

    Parameters
    ----------
    vertices :  (N, 3) array
    faces :     (M, 3) array | None

    Returns
    -------
    vertices :  (K, 3) float64 array
    faces :     (M, 3) int64 array

    """
    # Same coercion trimesh's own setters do, so that what comes out of here
    # does not depend on what went in.
    vertices = np.asarray(vertices, order="C", dtype=np.float64)
    if faces is None:
        faces = np.zeros((0, 3), dtype=np.int64)
    else:
        faces = np.asarray(faces, order="C", dtype=np.int64)

    n = len(vertices)
    if not n:
        return vertices, faces

    # --- non-finite vertices. The scalar check first: it is an order of
    # magnitude cheaper than the per-row one and answers "no" for every mesh
    # that is not broken.
    if not np.isfinite(vertices).all():
        finite = np.isfinite(vertices).all(axis=1)
        # A face naming a dropped vertex is left pointing at vertex 0, which is
        # what trimesh's `update_vertices` does with an unmapped index.
        inverse = np.zeros(n, dtype=np.int64)
        inverse[finite] = np.arange(finite.sum())
        vertices = vertices[finite]
        faces = inverse[faces]
        n = len(vertices)
        if not n:
            return vertices, faces

    # --- duplicate vertices
    referenced = np.ones(n, dtype=bool)
    if len(faces):
        referenced = np.zeros(n, dtype=bool)
        referenced[faces] = True

    # Skipping the copy where every vertex is referenced is worth a branch; the
    # rest of the bookkeeping below is already a no-op in that case.
    subset = vertices if referenced.all() else vertices[referenced]
    quantized = np.round(subset * 10 ** util.decimal_to_digits(tol.merge))
    keep, groups = unique_rows(quantized.astype(np.int64))

    # `keep` indexes the referenced subset, so this can only hold if every
    # vertex is referenced and none of them merged.
    if len(keep) == n:
        return vertices, faces

    inverse = np.zeros(n, dtype=np.int64)
    inverse[referenced] = groups
    return vertices[np.flatnonzero(referenced)[keep]], inverse[faces]
