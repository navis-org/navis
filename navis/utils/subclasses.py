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


import trimesh as tm
import numpy as np


__all__ = ["TrimeshPlus", "validate_extra_edges"]


def validate_extra_edges(edges, n_vertices=None):
    """Coerce edges into the canonical form used for extra (non-face) edges.

    Canonical means: `(M, 2)` int64, each row sorted (orientation is
    meaningless for an undirected edge), rows sorted and deduplicated, and no
    self-loops.

    Parameters
    ----------
    edges :         (M, 2) array-like | (2, ) array-like | None
                    Edges to validate. `None` and empty inputs both produce an
                    empty `(0, 2)` array.
    n_vertices :    int, optional
                    If provided, will check that all indices are within
                    `[0, n_vertices)`.

    Returns
    -------
    edges :         (M, 2) int64 array

    """
    if edges is None:
        return np.zeros((0, 2), dtype=np.int64)

    edges = np.asarray(edges)

    if edges.shape == (2,):  # a single edge
        edges = edges[np.newaxis, :]

    if not edges.size:
        return np.zeros((0, 2), dtype=np.int64)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"Edges must be a (N, 2) array, got {tuple(edges.shape)}"
        )

    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError(f"Edges must be integers, got dtype {edges.dtype}")

    edges = edges.astype(np.int64, copy=False)

    if edges.min() < 0:
        raise ValueError("Edges must not contain negative vertex indices.")

    if n_vertices is not None and edges.max() >= n_vertices:
        raise ValueError(
            f"Edge references vertex {edges.max()} but mesh only has "
            f"{n_vertices} vertices."
        )

    # Sort within each row: (a, b) and (b, a) are the same edge
    edges = np.sort(edges, axis=1)

    # Drop self-loops - they add nothing to the connectivity
    edges = edges[edges[:, 0] != edges[:, 1]]

    # Sort and deduplicate rows
    return np.unique(edges, axis=0)


def _edge_keys(edges):
    """Collapse sorted edges into scalar keys for set operations.

    Requires the edges to be sorted within each row (see
    [`validate_extra_edges`][]). Note we can't just use the max vertex index as
    base because the two edge arrays being compared may have different maxima -
    so we use a base large enough for both.
    """
    if not len(edges):
        return np.zeros(0, dtype=np.int64)
    # This stays injective (and inside int64) up to ~2.1e9 vertices, which is
    # well beyond anything that would fit in memory anyway
    return edges[:, 0] * np.int64(2**32) + edges[:, 1]


class TrimeshPlus(tm.Trimesh):
    """Trimesh object with additional features.

    Currently, this includes:
      - `extra_edges` property: edges that are not part of any face
      - `graph_edges` property: unique face edges plus the extra edges

    Extra edges exist to express connectivity that the surface itself does not
    have - e.g. bridges between disconnected mesh fragments. They deliberately
    do *not* show up in `.edges`,
    `.edges_unique` or anything else derived from the faces: trimesh's own
    machinery assumes `.edges` is `3 * len(faces)` long and aligned with
    `.edges_face`, so injecting them there silently breaks `faces_unique_edges`,
    laplacian smoothing, subdivision and friends. Consumers that want the
    connectivity - rather than the surface - must ask for `.graph_edges`
    explicitly (`navis.utils.mesh_unique_edges` does this for you).

    """

    def __repr__(self):
        s = super().__repr__()
        if len(self.extra_edges):
            s = s[:-2]  # remove last bracket
            s += f", extra_edges.shape={self.extra_edges.shape})>"
        return s

    @property
    def extra_edges(self):
        """Edges that are not part of any face.

        Always a `(M, 2)` int64 array - empty if there are none.
        """
        edges = getattr(self, "_extra_edges", None)
        if edges is None:
            return np.zeros((0, 2), dtype=np.int64)
        return edges

    @extra_edges.setter
    def extra_edges(self, edges):
        self._extra_edges = validate_extra_edges(edges, n_vertices=len(self.vertices))

    @property
    def graph_edges(self):
        """Unique edges of the *graph*: face edges plus any extra edges.

        This is what you want when treating the mesh as a graph (connectivity)
        rather than as a surface (geometry).
        """
        if not len(self.extra_edges):
            return self.edges_unique
        return np.vstack((self.edges_unique, self.extra_edges))

    def add_extra_edges(self, edges, validate=True, replace=False):
        """Add non-face edges to the mesh.

        You can directly set the `extra_edges` property, but this function
        does some additional checks such as ensuring that the edges are
        not already present among the faces.

        Parameters
        ----------
        edges :     (N, 2) iterable of vertex indices | (2, ) array for a single edge
                    Edges to add. Note that orientation does not matter.
        validate :  bool
                    Whether to silently drop edges that are already present in
                    the faces. Note that edges are always deduplicated and
                    bounds-checked - this only controls the (more expensive)
                    check against the faces.
        replace :   bool
                    Whether to add to or replace any existing extra edges.

        """
        edges = validate_extra_edges(edges, n_vertices=len(self.vertices))

        if not replace and len(self.extra_edges):
            edges = validate_extra_edges(np.vstack((self.extra_edges, edges)))

        if validate and len(edges):
            # Drop edges that already exist among the faces. N.B. `edges_unique`
            # is sorted within rows just like our edges, so we can compare them
            # as scalar keys.
            is_dupe = np.isin(_edge_keys(edges), _edge_keys(self.edges_unique))
            edges = edges[~is_dupe]

        self._extra_edges = edges

    def copy(self, *args, **kwargs):
        # N.B. trimesh's `copy` hard-codes `Trimesh()` as the new object, so we
        # have to re-class it. That's safe because we add no state that isn't
        # optional.
        copy = super().copy(*args, **kwargs)
        copy.__class__ = type(self)

        if len(self.extra_edges):
            copy._extra_edges = self.extra_edges.copy()

        return copy
