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

"""Closing the holes a subset cuts into a mesh.

Subsetting a `Mesh` drops every face that loses a corner, which leaves the
cut cross-sections standing open. This module works out where those new
openings are and triangulates them shut.

Two things make this cheap enough to run unconditionally:

- Nothing here looks at the mesh as a whole. Both the search for the new
  openings and the triangulation are confined to the collar of faces around the
  cut. A global edge-grouping pass (what `trimesh.repair.fill_holes` does) costs
  about as much as the entire prune it is meant to tidy up after.
- No vertices are added, only faces. That keeps every vertex index - in
  `faces`, in `extra_edges`, in `connectors`, in a tracked subset's
  provenance - pointing at what it pointed at before.

"""

import numpy as np

from .. import config, core, utils

logger = config.get_logger(__name__)

__all__ = ["fill_holes"]

try:
    import mapbox_earcut
except ImportError:  # pragma: no cover
    mapbox_earcut = None
    logger.debug(
        "mapbox_earcut not available - holes cut into meshes will be closed "
        "with triangle fans, which are wonky on non-convex openings."
    )


def _directed_edges(faces):
    """The three edges of each face, in winding order."""
    return faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)


def _edge_keys(edges, n_vertices):
    """One int64 per edge, the same whichever way the edge is pointing."""
    e = edges.astype(np.int64, copy=False)
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    return lo * n_vertices + hi


def find_new_boundary(faces, dropped):
    """Find the edges a subset is about to expose.

    Call this with the *original* faces, before subsetting.

    A face survives only if all three of its corners do, so an edge ends up on
    a new boundary exactly when it loses a face to the cut but keeps one. Both
    halves of that test are local to the cut, so this never has to group the
    edges of the whole mesh.

    Edges that were already boundary are left out: they belong to openings the
    mesh came with (a neurite truncated at the edge of the dataset, say), and
    sealing those is not this function's business.

    Parameters
    ----------
    faces :     (N, 3) array
                Faces of the mesh *before* subsetting.
    dropped :   (M, ) bool array
                For each vertex, whether the subset drops it.

    Returns
    -------
    (K, 2) array
                Directed half-edges, wound the way the one face they have left
                winds them. Indices are into the *original* vertices.

    """
    n_vertices = len(dropped)
    empty = np.empty((0, 2), dtype=np.int64)

    corners = (faces[:, 0], faces[:, 1], faces[:, 2])
    lost = [dropped[c] for c in corners]
    removed = lost[0] | lost[1] | lost[2]
    if not removed.any() or removed.all():
        return empty

    # Only a face that loses *exactly one* corner can leave an edge behind:
    # lose two and there is no edge left with both ends still standing. On a
    # real prune that is a percent or so of the faces that go, which is what
    # keeps everything downstream small.
    exposing = (lost[0].view(np.int8) + lost[1].view(np.int8) + lost[2].view(np.int8)) == 1
    if not exposing.any():
        return empty

    # The edge each of those faces exposes is the one opposite its lost corner.
    stumps = faces[exposing]
    gone = np.column_stack([side[exposing] for side in lost]).argmax(axis=1)
    row = np.arange(len(stumps))
    candidates = np.column_stack(
        (stumps[row, (gone + 1) % 3], stumps[row, (gone + 2) % 3])
    )
    cand_keys = np.unique(_edge_keys(candidates, n_vertices))

    # Both ends of a candidate survive, so every surviving face carrying one is
    # in this collar. Counting occurrences here gives the same answer as
    # counting across the whole mesh, for a small fraction of the work.
    collar = np.zeros(n_vertices, dtype=bool)
    collar[candidates] = True
    kept_faces = faces[
        ~removed & (collar[corners[0]] | collar[corners[1]] | collar[corners[2]])
    ]
    if not len(kept_faces):
        return empty

    edges = _directed_edges(kept_faces)
    keys = _edge_keys(edges, n_vertices)
    is_cand = np.isin(keys, cand_keys)
    keys, edges = keys[is_cand], edges[is_cand]
    if not len(keys):
        return empty

    # Exactly one surviving face left on the edge means it is now a boundary.
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return edges[counts[inv.ravel()] == 1]


def find_boundary(faces, n_vertices):
    """Find every edge of a mesh that has only one face on it.

    Unlike [`find_new_boundary`][] this has to group the edges of the whole
    mesh, which is the expensive way round - use the other one where you know
    which vertices are going away.

    Parameters
    ----------
    faces :         (N, 3) array
    n_vertices :    int

    Returns
    -------
    (K, 2) array
                Directed half-edges, wound the way their one face winds them.

    """
    if not len(faces):
        return np.empty((0, 2), dtype=np.int64)

    edges = _directed_edges(faces)
    _, inv, counts = np.unique(
        _edge_keys(edges, n_vertices), return_inverse=True, return_counts=True
    )
    return edges[counts[inv.ravel()] == 1]


def trace_loops(halfedges):
    """Walk directed half-edges into closed rings.

    Greedy: at a non-manifold boundary vertex several half-edges leave at once,
    so we take whichever is still free. Every half-edge lands in exactly one
    ring, which is what makes this cover the whole boundary. `nx.cycle_basis` -
    which is what `trimesh.repair.fill_holes` uses - quietly drops the edges
    that are not part of a simple cycle.
    """
    following = {}
    for a, b in halfedges.tolist():
        following.setdefault(a, []).append(b)

    loops = []
    for start in list(following):
        while following.get(start):
            loop = [start]
            v = following[start].pop()
            while v != start:
                onward = following.get(v)
                if not onward:  # ran into a dead end - give up on this ring
                    loop = None
                    break
                loop.append(v)
                v = onward.pop()
            if loop is not None and len(loop) >= 3:
                loops.append(loop)
    return loops


def _signed_area(flat):
    """Twice the signed area of a 2d ring; positive means counter-clockwise."""
    x, y = flat[:, 0], flat[:, 1]
    return float(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def _basis(normals):
    """An orthonormal pair spanning the plane perpendicular to each normal."""
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 0
    normals = normals / np.where(usable, lengths, 1)[:, None]
    # Any vector not parallel to the normal will do to get started.
    other = np.where(
        np.abs(normals[:, :1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    )
    u = np.cross(normals, other)
    u /= np.linalg.norm(u, axis=1)[:, None]
    return u, np.cross(normals, u), usable


def _fan(ring):
    """Fan from the ring's first vertex, wound against the ring.

    Wonky on a non-convex ring, but it always closes the hole and it always gets
    the winding right, which is what everything downstream actually depends on.
    """
    return np.column_stack(
        (np.full(len(ring) - 2, ring[0]), ring[2:], ring[1:-1])
    )


def _earcut(ring, flat):
    """Ear-clip a flattened ring. None if the flattening self-intersects."""
    n = len(ring)
    tri = mapbox_earcut.triangulate_float64(flat, np.array([n]))
    # Short of n - 2 triangles means earcut ran out of ears part way through,
    # which is what happens when the projection crosses itself.
    if len(tri) != 3 * (n - 2):
        return None
    faces = ring[tri.reshape(-1, 3)]
    # The ring runs the way its remaining faces wind it, so the cap has to run
    # the other way or the two will disagree about which side is out. earcut
    # emits counter-clockwise triangles whichever way the ring itself goes, so
    # it is only a counter-clockwise ring that needs flipping.
    return faces[:, ::-1] if _signed_area(flat) > 0 else faces


def _retry(ring, points):
    """Second attempt for a ring the area-weighted normal could not flatten.

    Falls back through the best-fit plane to a plain fan.
    """
    if mapbox_earcut is not None and len(ring) > 3:
        centred = points - points.mean(axis=0)
        # eigh on the 3x3 scatter matrix beats an SVD of the (n, 3) points, and
        # its eigenvectors come out ascending - so the last two span the plane.
        _, vectors = np.linalg.eigh(centred.T @ centred)
        faces = _earcut(ring, np.ascontiguousarray(centred @ vectors[:, :0:-1]))
        if faces is not None:
            return faces
    return _fan(ring)


def triangulate_rings(rings, vertices):
    """Triangulate boundary rings, wound against the direction they run in.

    The flattening every ring needs before it can be ear-clipped is done for all
    of them at once - per-ring numpy calls dominate the runtime otherwise, and
    a cut mesh has hundreds of these.

    Parameters
    ----------
    rings :     list of lists
                Boundary rings as returned by [`trace_loops`][].
    vertices :  (N, 3) array
                Vertices of the mesh.

    Returns
    -------
    (M, 3) array
                New faces, indices into `vertices`.

    """
    counts = np.array([len(r) for r in rings])
    flat_ring = np.concatenate([np.asarray(r, dtype=np.int64) for r in rings])
    starts = np.zeros(len(rings), dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])

    points = vertices[flat_ring]
    centred = points - np.repeat(
        np.add.reduceat(points, starts, axis=0) / counts[:, None], counts, axis=0
    )

    # Index of the next point in the same ring, wrapping at the ring's end.
    following = np.arange(1, len(flat_ring) + 1)
    following[starts + counts - 1] = starts

    if mapbox_earcut is not None:
        # Newell's area-weighted normal. Cheaper than a best-fit plane and, on
        # the rings a cut actually produces, it fails slightly less often too.
        u, w, usable = _basis(
            np.add.reduceat(np.cross(centred, centred[following]), starts, axis=0)
        )
        flat = np.column_stack(
            (
                np.einsum("ij,ij->i", centred, np.repeat(u, counts, axis=0)),
                np.einsum("ij,ij->i", centred, np.repeat(w, counts, axis=0)),
            )
        )
    else:
        usable = np.zeros(len(rings), dtype=bool)

    caps = []
    for i, start in enumerate(starts.tolist()):
        n = int(counts[i])
        ring = flat_ring[start : start + n]
        faces = None
        if usable[i] and n > 3:
            faces = _earcut(ring, np.ascontiguousarray(flat[start : start + n]))
        if faces is None:
            faces = _retry(ring, points[start : start + n]) if n > 3 else _fan(ring)
        caps.append(faces)

    return np.vstack(caps)


def cap_boundary(vertices, halfedges):
    """Triangulate the openings bounded by `halfedges`.

    Parameters
    ----------
    vertices :  (N, 3) array
                Vertices of the mesh, after subsetting.
    halfedges : (K, 2) array
                Directed half-edges as returned by [`find_new_boundary`][], with
                indices into `vertices`.

    Returns
    -------
    (M, 3) array
                New faces. Empty if there was nothing to close.

    """
    rings = trace_loops(halfedges)
    if not rings:
        return np.empty((0, 3), dtype=np.int64)

    return triangulate_rings(rings, vertices)


@utils.map_neuronlist(desc="Filling", allow_parallel=True)
def fill_holes(x: "core.Mesh", inplace: bool = False) -> "core.Mesh":
    """Triangulate the holes in a mesh.

    Every opening is closed, whether the mesh came with it or something like
    [`navis.prune_twigs`][] cut it. If you only want to close the latter, ask
    for it at the time of the cut instead - see
    [`navis.subset_neuron`][]'s `cap_holes` parameter, which is both cheaper
    and able to tell the two apart.

    Only faces are added, never vertices, so vertex indices - and with them
    connectors, `extra_edges` and any tracked provenance - are untouched.

    Parameters
    ----------
    x :         Mesh | NeuronList
    inplace :   bool, optional
                If False, will fill holes on and return a copy.

    Returns
    -------
    Mesh/List

    Examples
    --------
    >>> import navis
    >>> m = navis.example_neurons(1, kind='mesh')
    >>> pruned = navis.prune_twigs(m, size='5 microns')
    >>> filled = navis.fill_holes(pruned)
    >>> # The twig stumps are closed, so the surface encloses less
    >>> filled.volume < pruned.volume
    True
    >>> # ... and no vertex moved or was added in the process
    >>> filled.n_vertices == pruned.n_vertices
    True

    See Also
    --------
    [`navis.subset_neuron`][]
                Its `cap_holes` closes only the openings the subset itself
                made, and does not have to scan the whole mesh to find them.
    [`navis.heal_mesh`][]
                Reconnects disconnected *fragments*, which is a different kind
                of hole entirely.

    """
    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, core.Mesh):
        raise TypeError(f'Expected Mesh(s), got "{type(x)}"')

    if not inplace:
        x = x.copy()

    faces = np.asarray(x.faces)
    halfedges = find_boundary(faces, len(x.vertices))
    if len(halfedges):
        new_faces = cap_boundary(np.asarray(x.vertices), halfedges)
        if len(new_faces):
            x.faces = np.vstack((faces, new_faces.astype(faces.dtype)))

    return x
