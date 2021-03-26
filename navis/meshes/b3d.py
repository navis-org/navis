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

import os

import trimesh as tm

from .. import core


def simplify_mesh_blender(x, F, inplace=False):
    """Simplify mesh using Blender.

    Parameters
    ----------
    x :         MeshNeuron | Volume | Trimesh
                Mesh object to simplify.
    F :         float [0-1]
                Ratio to which to reduce the mesh. For example, ``F=0.5``
                should reduce number of vertices to half that of the original.
    inplace :   bool
                If True, will perform simplication on ``x``. If False, will
                simplify and return a copy.

    Returns
    -------
    simp
                Simplified mesh object.

    """
    if not tm.interfaces.blender.exists:
        raise ImportError('No Blender 3D unavailable (executable not found).')
    _blender_executable = tm.interfaces.blender._blender_executable

    if F > 1 and F < 0:
        raise ValueError(f'`F` must be between 0 and 1, got "{F}"')

    if isinstance(x, core.MeshNeuron):
        mesh = x.trimesh
    elif isinstance(x, core.Volume):
        mesh = tm.Trimesh(x.vertices, x.faces)
    elif isinstance(x, tm.Trimesh):
        mesh = x
    else:
        raise TypeError('Expected MeshNeuron, Volume or trimesh.Trimesh, '
                        f'got "{type(x)}"')

    assert isinstance(mesh, tm.Trimesh)

    # Load the template
    temp_name = 'blender_decimate.py.template'
    if temp_name in _cache:
        template = _cache[temp_name]
    else:
        with open(os.path.join(_pwd, 'templates', temp_name), 'r') as f:
            template = f.read()
        _cache[temp_name] = template

    # Replace placeholder with actual ratio
    script = template.replace('$RATIO', str(F))

    # Let trimesh's MeshScript take care of exectution and clean-up
    with tm.interfaces.generic.MeshScript(meshes=[mesh],
                                          script=script,
                                          debug=False) as blend:
        result = blend.run(_blender_executable
                           + ' --background --python $SCRIPT')

    # Blender apparently returns actively incorrect face normals
    result.face_normals = None

    if not inplace:
        x = x.copy()

    x.vertices = result.vertices
    x.faces = result.faces

    if not inplace:
        return x


# find the current absolute path to this directory
_pwd = os.path.expanduser(os.path.abspath(os.path.dirname(__file__)))

# Use to cache templates
_cache = {}
