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

Subsetting a `Mesh` drops every face that loses a corner, which leaves the cut
cross-sections standing open. Finding those openings and triangulating them
shut is `navis-fastcore`'s `caps` module; what is left here is the
[`navis.fill_holes`][] entry point and the one step both callers share.

There are two callers and they enter at different points. `fill_holes` closes
every opening the mesh has, so it has to group the edges of the whole mesh
(`boundary_halfedges`); [`navis.subset_neuron`][]'s `cap_holes` already knows
which vertices are going away, and `exposed_halfedges` gets its answer out of
the collar of faces around the cut without looking at the rest. Both hand their
half-edges to the same `trace_loops` -> `triangulate_rings` pair, which is what
`cap_boundary` below is.

No vertices are added, only faces. That keeps every vertex index - in `faces`,
in `extra_edges`, in `connectors`, in a tracked subset's provenance - pointing
at what it pointed at before, which is what lets the cap go on *after* a subset
rather than during it.

"""

import numpy as np

from .. import core, utils

__all__ = ["fill_holes"]


def cap_boundary(vertices, halfedges):
    """Triangulate the openings bounded by `halfedges`.

    Parameters
    ----------
    vertices :  (N, 3) array
                Vertices of the mesh, after subsetting.
    halfedges : (K, 2) array
                Directed half-edges, wound the way the one face they have left
                winds them, with indices into `vertices`.

    Returns
    -------
    (M, 3) array
                New faces. Empty if there was nothing to close.

    """
    rings, offsets = utils.fastcore.trace_loops(halfedges)
    return utils.fastcore.triangulate_rings(rings, offsets, vertices)


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
    halfedges = utils.fastcore.boundary_halfedges(faces)
    if len(halfedges):
        new_faces = cap_boundary(np.asarray(x.vertices), halfedges)
        if len(new_faces):
            x.faces = np.vstack((faces, new_faces.astype(faces.dtype)))

    return x
