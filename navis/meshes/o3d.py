#    This script is part of navis (http://www.github.com/schlegelp/navis).
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

from .. import core


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
    try:
        import open3d
    except ImportError:
        raise ImportError('Please install open3d: pip3 install open3d')
    except BaseException:
        raise

    if not isinstance(x, (core.MeshNeuron, core.Volume, tm.Trimesh)):
        raise TypeError('Expected MeshNeuron, Volume or trimesh.Trimesh, '
                        f'got "{type(x)}"')

    mesh_o3d = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(x.vertices),
            triangles=open3d.utility.Vector3iVector(x.faces))

    if method == 'quadric':
        simple = mesh_o3d.simplify_quadric_decimation(int(F), **kwargs)
    elif method == 'cluster':
        simple = mesh_o3d.simplify_vertex_clustering(F,  **kwargs)
    else:
        raise ValueError(f'Unknown simplification scheme "{method}"')

    if not inplace:
        x = x.copy()

    x.vertices = np.asarray(simple.vertices)
    x.faces = np.asarray(simple.triangles)

    if not inplace:
        return x
