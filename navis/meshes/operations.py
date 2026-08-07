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

from importlib.util import find_spec

try:
    from pykdtree.kdtree import KDTree
except ModuleNotFoundError:
    from scipy.spatial import cKDTree as KDTree

from .. import core, config, utils
from ..core import schema

from .b3d import smooth_mesh_blender
from .o3d import smooth_mesh_open3d
from .mesh_utils import smooth_mesh_trimesh


def smoothing_backends():
    """Backends available for smoothing, in order of preference.

    Blender comes last, behind a backend that is always there, and deliberately:
    it is reached by writing the mesh out to STL and reading the result back,
    which welds and reorders the vertices. Smoothing moves vertices without
    replacing them - which is why `smooth_mesh` is not a rebuild and everything
    indexing the vertices is simply left alone - and that stops being true on the
    way through a file. Still selectable by name for anyone who wants Blender's
    smoothing specifically and does not index the vertices.
    """
    backends = ['open3d'] if find_spec('open3d') is not None else []
    backends.append('trimesh')
    if tm.interfaces.blender.exists:
        backends.append('blender')
    return backends


@utils.map_neuronlist(desc='Simplifying', allow_parallel=True)
@utils.rebuilds('vertices')
def simplify_mesh(x, F, backend=None, inplace=False, **kwargs):
    """Simplify meshes (TriMesh, Mesh, Volume).

    Uses quadric error metric decimation (Garland & Heckbert) as implemented in
    `navis-fastcore`: the edge whose collapse costs least is contracted, over and
    over, until the target is met.

    Unlike other implementations this one tracks where every vertex went, which
    is what lets everything hanging off the vertices come through with them:
    connectors, extra edges, the skeleton correspondence and anything you
    attached yourself (see [`Neuron.attach`][navis.BaseNeuron.attach]).
    A vertex that was merged into another follows it there, and one the
    decimation could not place at all is read as the place on the surface it
    named - see `_place_orphans`.

    Parameters
    ----------
    x :         navis.Mesh/List | navis.Volume | trimesh.Trimesh
                Mesh(es) to simplify.
    F :         float | int
                Determines how much the mesh is simplified:
                Floats (0-1) are interpreted as ratio. For example, an F of
                0.5 will reduce the number of faces to 50%.
                Integers (>1) are intepreted as target face count. For example,
                an F of 5000 will attempt to reduce the number of faces to 5000.
    backend :   str, optional
                Deprecated and ignored - simplification always runs on
                `navis-fastcore` now. Passing anything raises a
                `DeprecationWarning`.
    inplace :   bool
                If True, will perform simplication on `x`. If False, will
                simplify and return a copy.
    **kwargs
                Keyword arguments are passed through to
                `navis_fastcore.simplify_mesh`, e.g. `aggressiveness`,
                `preserve_border` or `lock`. The last is worth knowing about:
                it pins the given vertices so they are never merged away or
                moved, which is how you keep, say, synapse-bearing vertices
                exactly where they are.

    Returns
    -------
    simplified
                Simplified object.

    See Also
    --------
    [`navis.downsample_neuron`][]
                Downsample all kinds of neurons.

    """
    if backend is not None:
        warnings.warn(
            "`simplify_mesh(backend=...)` is deprecated and ignored - "
            "simplification now always runs on `navis-fastcore`. Drop the "
            "argument.",
            DeprecationWarning,
            # Past `@rebuilds` and `@map_neuronlist`, so that this lands on the
            # caller: Python's default filters only show a `DeprecationWarning`
            # attributed to `__main__`, and one blamed on navis' own decorators
            # is a warning nobody ever sees.
            stacklevel=4,
        )

    if not utils.is_mesh(x):
        raise TypeError(f'Expected mesh-like, got "{type(x)}"')
    if F <= 0:
        raise ValueError(f'`F` must be greater than 0, got {F}')

    if not inplace:
        x = x.copy()

    target = dict(ratio=float(F)) if F < 1 else dict(n_faces=int(F))

    # Extra edges go out of the `.vertices` setter's way, exactly as
    # `_subset_meshneuron` does - it is the caller that knows better which that
    # setter's comment refers to. Left in place they would be dropped outright
    # the moment the vertex count changed, and there would be nothing left for
    # the rebuild below to move. They go back in un-remapped and are repaired
    # along with everything else. `Volume` and bare trimeshes have none to hold.
    extra_edges = getattr(x, '_extra_edges', None)
    if extra_edges is not None:
        x._extra_edges = None

    # `preserve_border=True` is navis' own default rather than fastcore's (or
    # pyfqmr's): meshes here are routinely fragments cut out of a larger volume,
    # and letting a cut face collapse inwards eats the neuron from its ends.
    defaults = dict(preserve_border=True)
    defaults.update(kwargs)

    # Kept past the assignment below: where a reference to a vertex the
    # decimation could not place should point can only be worked out against the
    # vertices it names. Deliberately not cast - fastcore takes its own float64
    # copy, so doing it here would only mean holding two.
    old_vertices = np.asarray(x.vertices)

    vertices, faces, vertex_map = utils.fastcore.simplify_mesh(
        np.asarray(x.faces), old_vertices, **target, **defaults
    )

    x.vertices = vertices
    x.faces = np.asarray(faces, dtype=np.int64)

    if extra_edges is not None:
        x._extra_edges = extra_edges

    # Decimation merges rather than selects - two vertices become one, so no new
    # vertex *is* an old one and `Rebuild.kept` would be a lie - but every old
    # vertex has somewhere it ended up, and that is the whole account.
    return x, schema.Rebuild(
        merged=vertex_map, snap=_place_orphans(x, old_vertices, vertex_map)
    )


def _place_orphans(x, old_vertices, vertex_map):
    """Where references to the vertices decimation could not place should point.

    `vertex_map` is `-1` for a vertex that ended up somewhere no surviving face
    references - typically one whose incident faces were all consumed, which
    leaves it sitting on the remaining surface rather than anywhere else. On the
    example neuron simplified to 20% that is 6% of the vertices, a median of 99
    units from the nearest survivor on a neuron 24,000 units across.

    Nothing *aligned* to such a vertex is invented here: no new vertex came from
    it, so it has no values to give and `merged` says exactly that. But a
    connector or an extra edge names a *place* on the surface, and the nearest
    surviving vertex is the honest reading of that place - which beats dropping
    the connector for want of an index.

    Goes back as `Rebuild.snap`, i.e. as the exceptions to `merged` rather than
    a map in its own right - `_relocation` lays one over the other. `None` when
    there is nothing to place, and `merged` then says where everything went on
    its own. Only the vertices something actually names are looked up: the query
    is priced per point, and a mesh whose connectors have never been snapped to
    a vertex has nothing to ask.
    """
    # `Volume` and bare `trimesh.Trimesh` come through here too, and have no
    # schema to carry
    if not isinstance(x, core.BaseNeuron):
        return None

    named = schema.referenced_values(x, schema.get_axis(x, 'vertices'))
    named = named[(named >= 0) & (named < len(vertex_map))]
    orphaned = named[vertex_map[named] < 0]
    if not len(orphaned):
        return None
    return orphaned, x.snap(old_vertices[orphaned])[0]


def combine_meshes(meshes, max_dist='auto', progress=True):
    """Try combining (partially overlapping) meshes.

    This function effectively works on the vertex graph and will not produce
    meaningful faces.
    """
    # Sort meshes by size
    meshes = sorted(meshes, key=lambda x: len(x.vertices), reverse=True)

    comb = tm.Trimesh(meshes[0].vertices.copy(), meshes[0].faces.copy())
    comb.remove_unreferenced_vertices()

    if max_dist == 'auto':
        max_dist = utils.mesh_unique_edges(comb, return_lengths=True)[1].mean()

    for m in config.tqdm(meshes[1:], desc='Combining',
                         disable=config.pbar_hide or not progress,
                         leave=config.pbar_leave):
        # Generate a new up-to-date tree
        tree = KDTree(comb.vertices)

        # Offset faces
        vertex_offset = comb.vertices.shape[0]

        # Find vertices that can be merged - note that we are effectively
        # zippig the two meshes by making sure that each vertex can only be
        # merged once
        dist, ix = tree.query(m.vertices, distance_upper_bound=max_dist)

        # Build a per-vertex remap (local index -> global index) rather than
        # scanning the entire face array once per merged vertex (was O(V*F))
        remap = np.arange(m.vertices.shape[0]) + vertex_offset
        merged = set()
        # Merge closest vertices first
        for i in np.argsort(dist):
            # Skip if no more within-distance
            if dist[i] >= np.inf:
                break
            # Skip if target vertex has already been merged
            if ix[i] in merged:
                continue

            # Remap this vertex onto the existing (comb) vertex
            remap[i] = ix[i]

            # Track that target vertex has already been seen
            merged.add(ix[i])

        # Apply the remap to all faces in a single vectorized pass
        new_faces = remap[m.faces]

        # Merge vertices and faces
        comb.vertices = np.append(comb.vertices, m.vertices, axis=0)
        comb.faces = np.append(comb.faces, new_faces, axis=0)

        # Drop unreferenced vertices (i.e. those that were remapped)
        comb.remove_unreferenced_vertices()

    return comb


@utils.map_neuronlist(desc='Smoothing', allow_parallel=True)
def smooth_mesh(x, iterations=5, L=.5, backend='auto', inplace=False):
    """Smooth meshes (TriMesh, Mesh, Volume).

    Uses Laplacian smoothing. Not necessarily because that is always the best
    approach but because there are three backends (see below) that offer similar
    interfaces.

    Parameters
    ----------
    x :             navis.Mesh/List | navis.Volume | trimesh.Trimesh
                    Mesh(es) to simplify.
    iterations :    int
                    Round of smoothing to apply.
    L :             float [0-1]
                    Diffusion speed constant lambda. Larger = more aggressive
                    smoothing.
    backend :       "auto" | "open3d" | "blender" | "trimesh"
                    Which backend to use. Currenly we support `open3d`,
                    Blender 3D or `trimesh`.
    inplace :       bool
                    If True, will perform simplication on `x`. If False, will
                    simplify and return a copy.

    Returns
    -------
    smoothed
                    Smoothed object.

    """
    if not isinstance(backend, str):
        raise TypeError(f'`backend` must be string, got "{type(backend)}"')

    backend = backend.lower()
    backends = smoothing_backends()

    if backend == 'auto':
        backend = backends[0]
    elif backend not in backends:
        raise ValueError(f'Backend "{backend}" appears to not be available. '
                         'Please choose one of the available backends: '
                         f'{", ".join(backends)}')

    if not inplace:
        x = x.copy()

    if backend == 'open3d':
        _ = smooth_mesh_open3d(x, iterations=iterations, L=L, inplace=True)
    elif backend == 'blender':
        _ = smooth_mesh_blender(x, iterations=iterations, L=L, inplace=True)
    elif backend == 'trimesh':
        _ = smooth_mesh_trimesh(x, iterations=iterations, L=L, inplace=True)

    return x
