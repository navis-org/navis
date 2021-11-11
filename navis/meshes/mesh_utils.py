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

import networkx as nx
import numpy as np
import trimesh as tm

from typing import Union

from .. import core


def fix_mesh(mesh: Union[tm.Trimesh, 'core.MeshNeuron'],
             fill_holes: bool = False,
             remove_fragments: bool = False,
             inplace: bool = False):
    """Try to fix some common problems with mesh.

     1. Remove infinite values
     2. Merge duplicate vertices
     3. Remove duplicate and degenerate faces
     4. Fix normals
     5. Remove unreference vertices
     6. Remove disconnected fragments (Optional)
     7. Fill holes (Optional)

    Parameters
    ----------
    mesh :              trimesh.Trimesh | navis.MeshNeuron
    fill_holes :        bool
                        If True will try to fix holes in the mesh.
    remove_fragments :  False | int
                        If a number is given, will iterate over the mesh's
                        connected components and remove those consisting of less
                        than the given number of vertices. For example,
                        ``remove_fragments=5`` will drop parts of the mesh
                        that consist of five or less connected vertices.
    inplace :           bool
                        If True, will perform fixes on the input mesh. If False,
                        will make a copy and leave the original untouched.

    Returns
    -------
    fixed object :      trimesh.Trimesh or navis.MeshNeuron

    """
    if not inplace:
        mesh = mesh.copy()

    if isinstance(mesh, core.MeshNeuron):
        m = mesh.trimesh
    else:
        m = mesh

    assert isinstance(m, tm.Trimesh)

    if remove_fragments:
        to_drop = []
        for c in nx.connected_components(m.vertex_adjacency_graph):
            if len(c) <= remove_fragments:
                to_drop += list(c)

        # Remove dropped vertices
        remove = np.isin(np.arange(m.vertices.shape[0]), to_drop)
        m.update_vertices(~remove)

    if fill_holes:
        m.fill_holes()

    m.remove_infinite_values()
    m.merge_vertices()
    m.remove_duplicate_faces()
    m.remove_degenerate_faces()
    m.fix_normals()
    m.remove_unreferenced_vertices()

    # If we started with a MeshNeuron, map back the verts/faces
    if isinstance(mesh, core.MeshNeuron):
        mesh.vertices, mesh.faces = m.vertices, m.faces
        mesh._clear_temp_attr()

    return mesh


def smooth_mesh_trimesh(x, iterations=5, L=0.5, inplace=False):
    """Smooth mesh using Trimesh's Laplacian smoothing.

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
    simp
                Simplified mesh object.

    """
    if L > 1 or L < 0:
        raise ValueError(f'`L` (lambda) must be between 0 and 1, got "{L}"')

    if isinstance(x, core.MeshNeuron):
        mesh = x.trimesh.copy()
    elif isinstance(x, core.Volume):
        mesh = tm.Trimesh(x.vertices, x.faces)
    elif isinstance(x, tm.Trimesh):
        mesh = x.copy()
    else:
        raise TypeError('Expected MeshNeuron, Volume or trimesh.Trimesh, '
                        f'got "{type(x)}"')

    assert isinstance(mesh, tm.Trimesh)

    # Smooth mesh
    # This always happens in place, hence we made a copy earlier
    tm.smoothing.filter_laplacian(mesh, lamb=L, iterations=iterations)

    if not inplace:
        x = x.copy()

    x.vertices = mesh.vertices
    x.faces = mesh.faces

    return x
