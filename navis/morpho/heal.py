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

"""Healing of fragmented meshes.

The mesh counterpart to [`navis.heal_skeleton`][]. The hard part - finding the
set of bridges that connects the fragments with the least added length - is the
same problem in both cases and uses the same machinery (`_stitch_edges` in
`navis.morpho.manipulation`). What differs is how the result is applied: a
skeleton's topology *is* its `parent_id` column, whereas a mesh's
is implied by its faces, and a bridge between two vertices cannot be expressed
as a face without inventing geometry. Bridges therefore go into the neuron's
`.extra_edges` - part of the graph, not of the surface.

"""

from typing import Optional, Sequence

import numpy as np

from . import subset
from .manipulation import _stitch_edges
from .. import graph, utils, core

# N.B. `navis.graph.graph_utils` imports `morpho` at module level, so its helpers
# (`_mesh_component_labels`, `_fastcore_has`) have to be reached through the
# `graph.graph_utils.` attribute path - resolved at call time - rather than
# imported up here.

__all__ = ["heal_mesh"]


@utils.map_neuronlist(desc="Healing", allow_parallel=True)
def heal_mesh(
    x: "core.Mesh",
    max_dist: Optional[float] = None,
    min_size: Optional[int] = None,
    drop_disc: bool = False,
    mask: Optional[Sequence] = None,
    inplace: bool = False,
) -> "core.Mesh":
    """Heal fragmented mesh(es).

    Meshes often consist of several disconnected fragments - e.g. because the
    underlying segmentation had a gap, or because meshing produced separate
    closed surfaces where the neuron is actually continuous. This function
    reconnects those fragments by adding the set of bridges that minimises the
    total added length (a minimum spanning tree over the fragments).

    Note that this is a purely *topological* repair: the bridges are added to
    the neuron's [`extra_edges`][navis.Mesh.extra_edges] and no vertices
    or faces are touched. Vertices, faces, surface area and volume are all
    unchanged - only anything reading the mesh's *connectivity* (e.g.
    [`navis.geodesic_matrix`][], [`navis.break_fragments`][],
    [`navis.drop_fluff`][], `.igraph`) will see the difference.

    Parameters
    ----------
    x :         Mesh | NeuronList
                Fragmented mesh(es).
    max_dist :  float | str, optional
                This effectively sets the max length for newly added edges. Use
                it to prevent far away fragments from being forcefully connected.
                If the neurons have `.units` set, you can also pass a string
                such as e.g. "2 microns".
    min_size :  int, optional
                Minimum size in vertices for fragments to be reattached.
                Fragments smaller than `min_size` will be ignored during
                healing and hence remain disconnected.
    drop_disc : bool
                If True and the mesh remains fragmented after healing (i.e.
                `max_dist` or `min_size` prevented a full connect), we will keep
                only the largest (by number of vertices) connected component
                and discard all other fragments.
    mask :      list-like, optional
                Either a boolean mask or a list of vertex indices. If provided,
                only these vertices may be used as bridge endpoints. Note that
                a fragment without a single unmasked vertex can not be
                connected.
    inplace :   bool, optional
                If False, will perform healing on and return a copy.

    Returns
    -------
    Mesh/List

    See Also
    --------
    [`navis.heal_skeleton`][]
                The equivalent for skeletons.
    [`navis.break_fragments`][]
                Use to produce individual neurons from disconnected fragments.
    [`navis.drop_fluff`][]
                Use to drop small disconnected fragments instead of connecting
                them.

    Examples
    --------
    >>> import navis
    >>> m = navis.example_neurons(1, kind='mesh')
    >>> # This mesh consists of a main body plus a bunch of small fragments
    >>> len(navis.break_fragments(m))
    70
    >>> healed = navis.heal_mesh(m)
    >>> len(navis.break_fragments(healed))
    1
    >>> # Healing is topological only - the surface is untouched
    >>> healed.volume == m.volume
    True

    Only connect fragments that are close to one another:

    >>> partial = navis.heal_mesh(m, max_dist='500 nm')
    >>> len(navis.break_fragments(partial))
    13

    """
    # The decorator makes sure that at this point we have single neurons
    if not isinstance(x, core.Mesh):
        raise TypeError(f'Expected Mesh(s), got "{type(x)}"')

    if max_dist is not None:
        max_dist = float(x.map_units(max_dist, on_error="raise"))

    if not inplace:
        x = x.copy()

    labels, n_frags = graph.graph_utils._mesh_component_labels(x)
    connected = n_frags <= 1

    if not connected:
        # Collect the restrictions on which vertices may be used to bridge
        # fragments in a single mask
        keep = np.ones(len(labels), dtype=bool)

        if min_size:
            sizes = np.bincount(labels, minlength=n_frags)
            keep &= sizes[labels] >= min_size

        if mask is not None:
            keep &= _vertex_mask(x, mask)

        bridges = _bridge_fragments(x.vertices, labels, n_frags, keep, max_dist)

        # Each bridge merges exactly two fragments, so this many means we
        # connected everything - and saves us re-labelling below
        connected = len(bridges) == n_frags - 1

        if len(bridges):
            # N.B. bridges connect vertices that are (by definition) in
            # different fragments, so they can duplicate neither an existing
            # extra edge nor a face edge
            if x.n_extra_edges:
                bridges = np.vstack((x.extra_edges, bridges))
            x.extra_edges = bridges

    # See if we need to drop remaining disconnected fragments
    if drop_disc and not connected:
        labels, n_frags = graph.graph_utils._mesh_component_labels(x)
        sizes = np.bincount(labels, minlength=n_frags)
        _ = subset.subset_neuron(x, labels == np.argmax(sizes), inplace=True)

    return x


def _vertex_mask(x: "core.Mesh", mask: Sequence) -> np.ndarray:
    """Turn `mask` (boolean mask or vertex indices) into a boolean mask."""
    mask = np.asarray(mask)

    if mask.dtype == bool:
        if len(mask) != len(x.vertices):
            raise ValueError(
                "Length of boolean mask must match number of vertices in the "
                f"mesh ({len(mask)} != {len(x.vertices)})"
            )
        return mask

    out = np.zeros(len(x.vertices), dtype=bool)
    out[mask.astype(np.int64, copy=False)] = True
    return out


def _bridge_fragments(
    vertices: np.ndarray,
    labels: np.ndarray,
    n_frags: int,
    keep: np.ndarray,
    max_dist: Optional[float],
) -> np.ndarray:
    """Find the minimal-length set of edges connecting a mesh's fragments.

    Parameters
    ----------
    vertices :  (V, 3) array
    labels :    (V, ) array
                Contiguous component label for each vertex.
    n_frags :   int
                Number of components.
    keep :      (V, ) bool array
                Which vertices may be used as bridge endpoints.
    max_dist :  float, optional
                Maximum length for any single new edge. `None` = no limit.

    Returns
    -------
    (M, 2) int64 array
                Pairs of vertex indices to connect.

    """
    empty = np.zeros((0, 2), dtype=np.int64)

    if keep.all():
        # The common case - no point copying the whole mesh just to index it
        cand, cand_labels, coords = np.arange(len(labels)), labels, vertices
    else:
        cand = np.where(keep)[0]
        cand_labels, coords = labels[cand], vertices[cand]

    if len(cand) < 2:
        return empty

    # One arbitrary candidate per fragment, `-1` where the restrictions left a
    # fragment without any. This doubles as our "is there still more than one
    # fragment to connect?" check.
    root_of_frag = np.full(n_frags, -1, dtype=np.int64)
    root_of_frag[cand_labels] = cand

    if np.count_nonzero(root_of_frag >= 0) < 2:
        return empty

    coords = np.ascontiguousarray(coords, dtype=np.float64)

    if graph.graph_utils._fastcore_has("stitch_fragments"):
        # `stitch_fragments` takes a *skeleton* but only ever looks at which
        # fragment each node belongs to - so rather than building a spanning
        # tree per fragment we simply point every candidate at one arbitrary
        # candidate of its own fragment. That is the same partition at no cost.
        parents = root_of_frag[cand_labels]
        parents[parents == cand] = -1  # the local roots themselves

        bridges, _ = utils.fastcore.stitch_fragments(
            cand.astype(np.int64), parents, coords, max_dist=max_dist
        )
        return np.asarray(bridges, dtype=np.int64).reshape(-1, 2)

    return _stitch_edges(
        coords, cand, cand_labels, np.inf if max_dist is None else max_dist
    )
