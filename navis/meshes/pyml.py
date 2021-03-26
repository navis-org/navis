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

"""Interfaces with PyMeshLab."""

# Notes:
# - there is some odd behaviour when the PyQT version installed does not match
#   the one bundled (?) with pymeshlab -> should investigate before rolling out

import warnings

import trimesh as tm

from .. import utils, core


def simplify_mesh_pyml(x, F, method='quadric', inplace=False, **kwargs):
    """Simplify mesh using pymeshlab.

    Parameters
    ----------
    x :         MeshNeuron | Volume | Trimesh
                Mesh object to simplify.
    F :         float [0-1]
                For method "quadric" this is the target number of faces as
                fraction of the original face count. For method "cluster" this
                is the size of the cells used for clustering: larger values =
                coarser mesh.
    method :    "quadric" | "cluster"
                Which method to use for simplification: quadratic mesh
                decimation or vertex clustering.
    inplace :   bool
                If True, will perform simplication on ``x``. If False, will
                simplify and return a copy.
    **kwargs
                Passed to pymeshlab filter functions:
                `simplification_quadric_edge_collapse_decimation` or
                `simplification_clustering_decimation` depending on method.

    Returns
    -------
    simp
                Simplified mesh-like object.

    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pymeshlab
    except ImportError:
        raise ImportError('Please install pymeshlab: pip3 install pymeshlab')
    except BaseException:
        raise

    utils.eval_param(method,
                     name='method',
                     allowed_values=('quadric', 'cluster'))

    if not isinstance(x, (core.MeshNeuron, tm.Trimesh, core.Volume)):
        raise TypeError(f'Expected MeshNeuron, Volume or Trimesh, got "{type(x)}"')

    if (F <= 0) or (F >= 1):
        raise ValueError(f'`t` must be between 0-1, got {F}')

    verts, faces = x.vertices, x.faces

    # Create mesh from vertices and faces
    m = pymeshlab.Mesh(verts, faces)

    # Create a new MeshSet
    ms = pymeshlab.MeshSet()

    # Add the mesh to the MeshSet
    ms.add_mesh(m, "mymesh")

    # Apply filter
    if method == 'quadric':
        defaults = {'targetperc': F}
        defaults.update(kwargs)
        ms.simplification_quadric_edge_collapse_decimation(**defaults)
    else:
        # Threshold is for some reason in percent, not fraction
        defaults = {'thresholds': F * 100}
        defaults.update(kwargs)
        ms.simplification_clustering_decimation(**defaults)

    # Get update mesh
    m2 = ms.current_mesh()

    # Get new vertices and faces
    new_verts = m2.vertex_matrix()
    new_faces = m2.face_matrix()

    # Make copy of the original mesh and assign new vertices + faces
    if not inplace:
        x = x.copy()

    x.vertices = new_verts
    x.faces = new_faces

    if isinstance(x, core.MeshNeuron):
        x._clear_temp_attr()

    return x
