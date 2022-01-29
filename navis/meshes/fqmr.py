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


def simplify_mesh_fqmr(x, F, inplace=False, **kwargs):
    """Simplify mesh using pyfqmr.

    Parameters
    ----------
    x :         MeshNeuron | Volume | Trimesh
                Mesh object to simplify.
    F :         int
                Target face count (integer).
    inplace :   bool
                If True, will perform simplication on ``x``. If False, will
                simplify and return a copy.
    **kwargs
                Keyword arguments are passed through to pyfqmr's
                ``pyfqmr.Simplify.simplify_mesh``.

    Returns
    -------
    simp
                Simplified mesh object.

    """
    if not utils.is_mesh(x):
        raise TypeError(f'Expected mesh-like, got "{type(x)}"')

    try:
        import pyfqmr
    except ImportError:
        raise ImportError('Please install pyfqmr: pip3 install pyfqmr')
    except BaseException:
        raise

    defaults = dict(aggressiveness=7, preserve_border=True, verbose=False)
    defaults.update(kwargs)

    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(x.vertices, x.faces)
    mesh_simplifier.simplify_mesh(target_count=F, **defaults)
    vertices, faces, normals = mesh_simplifier.getMesh()

    if not inplace:
        x = x.copy()

    x.vertices = vertices
    x.faces = faces

    if hasattr(x, 'face_normals'):
        x.face_normals = normals

    return x
