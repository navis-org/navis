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

from .b3d import simplify_mesh_blender, smooth_mesh_blender
from .pyml import simplify_mesh_pyml
from .o3d import simplify_mesh_open3d, smooth_mesh_open3d
from .fqmr import simplify_mesh_fqmr
from .mesh_utils import smooth_mesh_trimesh


def available_backends(only_first=False):
    """Search for available backends."""
    backends = []

    try:
        if find_spec('pyfqmr') is not None:
            backends.append('pyfqmr')
    except ModuleNotFoundError:
        pass
    except BaseException:
        raise

    if only_first and len(backends):
        return backends

    try:
        if find_spec('open3d') is not None:
            backends.append('open3d')
    except ModuleNotFoundError:
        pass
    except BaseException:
        raise

    if only_first and len(backends):
        return backends

    try:
        if find_spec('pymeshlab') is not None:
            backends.append('pymeshlab')
    except ModuleNotFoundError:
        pass
    except BaseException:
        raise

    if tm.interfaces.blender.exists:
        backends.append('blender')

    return backends


@utils.map_neuronlist(desc='Simplifying', allow_parallel=True)
@utils.rebuilds('vertices')
def simplify_mesh(x, F, backend='auto', inplace=False, **kwargs):
    """Simplify meshes (TriMesh, Mesh, Volume).

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
    backend :   "auto" | "pyfqmr" | "open3d" | "blender" | "pymeshlab"
                Which backend to use. Currenly we support `pyfqmr`, `open3d`,
                Blender 3D and `pymeshlab`.
    inplace :   bool
                If True, will perform simplication on `x`. If False, will
                simplify and return a copy.
    **kwargs
                Keyword arguments are passed through to the respective backend's
                functions (see below).

    Returns
    -------
    simplified
                Simplified object.

    See Also
    --------
    [`navis.downsample_neuron`][]
                Downsample all kinds of neurons.
    [`navis.meshes.simplify_mesh_fqmr`][]
                pyfqmr implementation for mesh simplification.
    [`navis.meshes.simplify_mesh_open3d`][]
                Open3D implementation for mesh simplification.
    [`navis.meshes.simplify_mesh_pyml`][]
                PyMeshLab implementation for mesh simplification.
    [`navis.meshes.simplify_mesh_blender`][]
                Blender 3D implementation for mesh simplification.

    """
    if not isinstance(backend, str):
        raise TypeError(f'`backend` must be string, got "{type(backend)}"')

    backend = backend.lower()
    backends = available_backends(only_first=backend == 'auto')

    if not backends:
        raise BaseException("None of the supported backends appear to be "
                            "available. Please install either `pyfqmr`, `open3d` "
                            "or `pymeshlab` via `pip`, or install Blender 3D.")
    elif backend == 'auto':
        backend = backends[0]
    elif backend not in backends:
        raise ValueError(f'Backend "{backend}" appears to not be available. '
                         'Please choose one of the available backends: '
                         f'{", ".join(backends)}')

    if not inplace:
        x = x.copy()

    # Taken before a backend replaces them: where a reference to an old vertex
    # should now point can only be worked out against the vertices it names.
    old_vertices = np.asarray(x.vertices)

    # Extra edges go out of the `.vertices` setter's way, exactly as
    # `_subset_meshneuron` does - it is the caller that knows better which that
    # setter's comment refers to. Left in place they would be dropped outright
    # the moment the vertex count changed, and there would be nothing left for
    # the snap below to move. They go back in un-remapped and are repaired along
    # with everything else. `Volume` and bare trimeshes have none to hold.
    extra_edges = getattr(x, '_extra_edges', None)
    if extra_edges is not None:
        x._extra_edges = None

    if backend == 'pyfqmr':
        # This expects a target face count
        if F < 1:
            F = int(F * len(x.faces))
        _ = simplify_mesh_fqmr(x, F=F, inplace=True, **kwargs)
    elif backend == 'open3d':
        # This expects a target face count
        if F < 1:
            F = int(F * len(x.faces))
        _ = simplify_mesh_open3d(x, F=F, inplace=True, **kwargs)
    elif backend == 'blender':
        # This expects a ratio
        if F > 1:
            F = F / len(x.faces)
        _ = simplify_mesh_blender(x, F=F, inplace=True)
    elif backend == 'pymeshlab':
        # This expects a ratio
        if F > 1:
            F = F / len(x.faces)
        _ = simplify_mesh_pyml(x, F=F, inplace=True, **kwargs)

    if extra_edges is not None:
        x._extra_edges = extra_edges

    return x, schema.Rebuild(snap=_snap_to_new_vertices(x, old_vertices))


def _snap_to_new_vertices(x, old_vertices):
    """Where a reference to each old vertex should point after simplification.

    Decimation invents vertex positions rather than choosing among the old ones,
    so there is no identity here to claim - which is why `Rebuild.kept` is left
    unset and anything aligned to the vertices is dropped. But a reference to an
    old vertex still names a *place* on the surface, and the new vertex nearest
    that place is the honest reading of it: a connector stays where it was
    instead of pointing at a vertex that no longer exists, or at whatever
    happens to sit at its old index now.

    Only the vertices something actually names are looked up - the query is
    priced per point, and a mesh whose connectors have never been snapped to a
    vertex has nothing to ask about.
    """
    # `Volume` and bare `trimesh.Trimesh` come through here too, and have no
    # schema to carry
    if not isinstance(x, core.BaseNeuron):
        return None

    named = schema.referenced_values(x, schema.get_axis(x, 'vertices'))
    named = named[(named >= 0) & (named < len(old_vertices))]
    if not len(named):
        return None
    return named, x.snap(old_vertices[named])[0]


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
    backends = available_backends() + ['trimesh']

    # Drop pymeshlab from backend
    if 'pymeshlab' in backends:
        backends.remove('pymeshlab')

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
