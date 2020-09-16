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
                        will make a copy first.

    Returns
    -------
    fixed object :      trimesh.Trimesh or navis.MeshNeuron
                        Only if ``inplace=False``.

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

    if not inplace:
        return mesh
