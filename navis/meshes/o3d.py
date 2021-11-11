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

import numpy as np
import trimesh as tm

from .. import core, utils


def simplify_mesh_open3d(x, F, method='quadric', inplace=False, **kwargs):
    """Simplify mesh using open3d.

    Parameters
    ----------
    x :         MeshNeuron | Volume | Trimesh
                Mesh object to simplify.
    F :         float | int
                For method `quadric` this is the target face count (integer).
                For method `cluster` this is the size of the voxel within which
                vertices are pooled (larger t = coarser mesh).
    method :    "quadric" | "cluster"
                Which method to use for simplification: either Quadric Error
                Metric Decimation (by Garland and Heckbert) or vertex clustering.
                Note that the intepretation of ``F`` depends on the method.
    inplace :   bool
                If True, will perform simplication on ``x``. If False, will
                simplify and return a copy.
    **kwargs
                Keyword arguments are passed through to open3d's
                ``simplify_quadric_decimation`` and ``simplify_vertex_clustering``,
                respectively.

    Returns
    -------
    simp
                Simplified mesh object.

    """
    if not utils.is_mesh(x):
        raise TypeError(f'Expected mesh-like, got "{type(x)}"')

    mesh_o3d = make_o3d_mesh(x)

    if method == 'quadric':
        result = mesh_o3d.simplify_quadric_decimation(int(F), **kwargs)
    elif method == 'cluster':
        result = mesh_o3d.simplify_vertex_clustering(F,  **kwargs)
    else:
        raise ValueError(f'Unknown simplification scheme "{method}"')

    if not inplace:
        x = x.copy()

    x.vertices = np.asarray(result.vertices)
    x.faces = np.asarray(result.triangles)

    return x


def smooth_mesh_open3d(x, iterations=5, L=0.5, inplace=False):
    """Smooth mesh using open3d's Laplacian smoothing.

    Parameters
    ----------
    x :             MeshNeuron | Volume | Trimesh
                    Mesh object to simplify.
    iterations :    int
                    Round of smoothing to apply.
    L :             float [0-1]
                    Diffusion speed constant lambda. Larger = more aggressive
                    smoothing.
    inplace :       bool
                    If True, will perform simplication on ``x``. If False, will
                    simplify and return a copy.

    Returns
    -------
    smoothed
                    Smoothed mesh object.

    """
    if not isinstance(x, (core.MeshNeuron, core.Volume, tm.Trimesh)):
        raise TypeError('Expected MeshNeuron, Volume or trimesh.Trimesh, '
                        f'got "{type(x)}"')

    mesh_o3d = make_o3d_mesh(x)

    if L > 1 or L < 0:
        raise ValueError(f'`L` (lambda) must be between 0 and 1, got "{L}"')

    result = mesh_o3d.filter_smooth_laplacian(iterations, L)

    if not inplace:
        x = x.copy()

    x.vertices = np.asarray(result.vertices)
    x.faces = np.asarray(result.triangles)

    return x


def make_o3d_mesh(x):
    """Turn mesh-like object into an open3d mesh."""
    try:
        import open3d
    except ImportError:
        raise ImportError('Please install open3d: pip3 install open3d')
    except BaseException:
        raise

    return open3d.geometry.TriangleMesh(
                vertices=open3d.utility.Vector3dVector(x.vertices),
                triangles=open3d.utility.Vector3iVector(x.faces))
