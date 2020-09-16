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

from .. import core, config


def simplify_mesh(x, ratio, inplace=False):
    """Simplify mesh using Blender 3D.

    Uses Blender's "decimate" modifier in "collapse" mode.

    Parameters
    ----------
    x :         navis.MeshNeuron | navis.Volume | trimesh.Trimesh
                Object to simplify.
    ratio :     float
                Factor to which to reduce faces. For example, a ratio of 0.5 will
                reduce the number of faces to 50%.
    inplace :   bool
                If True, will perform simplication on ``x``. If False, will
                simplify and return a copy.

    Returns
    -------
    simplified
            Simplified object.

    """
    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()

        _ = [simplify_mesh(n, ratio=ratio, inplace=True)
             for n in config.tqdm(x,
                                  desc='Simplifying',
                                  leave=config.pbar_leave,
                                  disable=len(x) == 1 or config.pbar_hide)]

        if not inplace:
            return x
        return

    if isinstance(x, core.MeshNeuron):
        mesh = x.trimesh
    elif isinstance(x, core.Volume):
        mesh = tm.Trimesh(x.vertices, x.faces)
    elif isinstance(x, tm.Trimesh):
        mesh = x
    else:
        raise TypeError(f'Expected MeshNeuron, Volume or trimesh.Trimesh, got "{type(x)}"')

    if not tm.interfaces.blender.exists:
        raise ImportError('No Blender available (executable not found).')
    _blender_executable = tm.interfaces.blender._blender_executable

    assert ratio < 1 and ratio > 0, 'ratio must be between 0 and 1'
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
    script = template.replace('$RATIO', str(ratio))

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
