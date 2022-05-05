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

try:
    from pykdtree.kdtree import KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree

from .. import core, config, utils

from .b3d import simplify_mesh_blender, smooth_mesh_blender
from .pyml import simplify_mesh_pyml
from .o3d import simplify_mesh_open3d, smooth_mesh_open3d
from .fqmr import simplify_mesh_fqmr
from .mesh_utils import smooth_mesh_trimesh


def available_backends(only_first=False):
    """Search for available backends."""
    backends = []

    try:
        import pyfqmr
        backends.append('pyfqmr')
    except ImportError:
        pass
    except BaseException:
        raise

    if only_first and len(backends):
        return backends

    try:
        import open3d
        backends.append('open3d')
    except ImportError:
        pass
    except BaseException:
        raise

    if only_first and len(backends):
        return backends

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pymeshlab
        backends.append('pymeshlab')
    except ImportError:
        pass
    except BaseException:
        raise

    if tm.interfaces.blender.exists:
        backends.append('blender')

    return backends


@utils.map_neuronlist(desc='Simplifying', allow_parallel=True)
def simplify_mesh(x, F, backend='auto', inplace=False, **kwargs):
    """Simplify meshes (TriMesh, MeshNeuron, Volume).

    Parameters
    ----------
    x :         navis.MeshNeuron/List | navis.Volume | trimesh.Trimesh
                Mesh(es) to simplify.
    F :         float | int
                Determines how much the mesh is simplified:
                Floats (0-1) are interpreted as ratio. For example, an F of
                0.5 will reduce the number of faces to 50%.
                Integers (>1) are intepreted as target face count. For example,
                an F of 5000 will attempt to reduce the number of faces to 5000.
    backend :   "auto" | "pyfqmr" | "open3d" | "blender" | "pymeshlab"
                Which backend to use. Currenly we support ``pyfqmr``, ``open3d``,
                Blender 3D and ``pymeshlab``.
    inplace :   bool
                If True, will perform simplication on ``x``. If False, will
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
    :func:`navis.downsample_neuron`
                Downsample all kinds of neurons.
    :func:`navis.meshes.simplify_mesh_fqmr`
                pyfqmr implementation for mesh simplification.
    :func:`navis.meshes.simplify_mesh_open3d`
                Open3D implementation for mesh simplification.
    :func:`navis.meshes.simplify_mesh_pyml`
                PyMeshLab implementation for mesh simplification.
    :func:`navis.meshes.simplify_mesh_blender`
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

    if backend == 'pyfqmr':
        # This expects a target face count
        if F < 1:
            F = F * len(x.faces)
        _ = simplify_mesh_fqmr(x, F=F, inplace=True, **kwargs)
    elif backend == 'open3d':
        # This expects a target face count
        if F < 1:
            F = F * len(x.faces)
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

    return x


def combine_meshes(meshes, max_dist='auto'):
    """Try combining (partially overlapping) meshes.

    This function effectively works on the vertex graph and will not produce
    meaningful faces.
    """
    # Sort meshes by size
    meshes = sorted(meshes, key=lambda x: len(x.vertices), reverse=True)

    comb = tm.Trimesh(meshes[0].vertices.copy(), meshes[0].faces.copy())
    comb.remove_unreferenced_vertices()

    if max_dist == 'auto':
        max_dist = comb.edges_unique_length.mean()

    for m in config.tqdm(meshes[1:], desc='Combining',
                         disable=config.pbar_hide,
                         leave=config.pbar_leave):
        # Generate a new up-to-date tree
        tree = KDTree(comb.vertices)

        # Offset faces
        vertex_offset = comb.vertices.shape[0]
        new_faces = m.faces + vertex_offset

        # Find vertices that can be merged - note that we are effectively
        # zippig the two meshes by making sure that each vertex can only be
        # merged once
        dist, ix = tree.query(m.vertices, distance_upper_bound=max_dist)

        merged = set()
        # Merge closest vertices first
        for i in np.argsort(dist):
            # Skip if no more within-distance
            if dist[i] >= np.inf:
                break
            # Skip if target vertex has already been merged
            if ix[i] in merged:
                continue

            # Remap this vertex
            new_faces[new_faces == (i + vertex_offset)] = ix[i]

            # Track that target vertex has already been seen
            merged.add(ix[i])

        # Merge vertices and faces
        comb.vertices = np.append(comb.vertices, m.vertices, axis=0)
        comb.faces = np.append(comb.faces, new_faces, axis=0)

        # Drop unreferenced vertices (i.e. those that were remapped)
        comb.remove_unreferenced_vertices()

    return comb


@utils.map_neuronlist(desc='Smoothing', allow_parallel=True)
def smooth_mesh(x, iterations=5, L=.5, backend='auto', inplace=False):
    """Smooth meshes (TriMesh, MeshNeuron, Volume).

    Uses Laplacian smoothing. Not necessarily because that is always the best
    approach but because there are three backends (see below) that offer similar
    interfaces.

    Parameters
    ----------
    x :             navis.MeshNeuron/List | navis.Volume | trimesh.Trimesh
                    Mesh(es) to simplify.
    iterations :    int
                    Round of smoothing to apply.
    L :             float [0-1]
                    Diffusion speed constant lambda. Larger = more aggressive
                    smoothing.
    backend :       "auto" | "open3d" | "blender" | "trimesh"
                    Which backend to use. Currenly we support ``open3d``,
                    Blender 3D or ``trimesh``.
    inplace :       bool
                    If True, will perform simplication on ``x``. If False, will
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
