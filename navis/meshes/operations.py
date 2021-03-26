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

import warnings

import trimesh as tm

from .. import core, config

from .b3d import simplify_mesh_blender
from .pyml import simplify_mesh_pyml
from .o3d import simplify_mesh_open3d


def available_backends():
    """Search for available backends."""
    backends = []
    try:
        import open3d
        backends.append('open3d')
    except ImportError:
        pass
    except BaseException:
        raise

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


def simplify_mesh(x, F, backend='auto', inplace=False, **kwargs):
    """Simplify meshes (TriMesh, MeshNeuron, Volume).

    Parameters
    ----------
    x :         navis.MeshNeuron/List | navis.Volume | trimesh.Trimesh
                Mesh to simplify.
    F :         float | int
                Determines how much the mesh is simplified:
                Floats (0-1) are interpreted as ratio. For example, an F of
                0.5 will reduce the number of faces to 50%.
                Integers (>1) are intepreted as target face count. For example,
                an F of 5000 will attempt to reduce the number of faces to 5000.
    backend :   "auto" | "open3d" | "blender" | "pymeshlab"
                Which backend to use. Currenly we support ``open3d``, Blender 3D
                and ``pymeshlab`.
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
    backends = available_backends()

    if not backends:
        raise BaseException("None of the supported backends appear to be "
                            "available. Please install either `open3d` or "
                            "`pymeshlab` via `pip`, or install Blender 3D.")
    elif backend == 'auto':
        backend = backends[0]
    elif backend not in backends:
        raise ValueError(f'Backend "{backend}" appears to not be available. '
                         'Please choose one of the available backends: '
                         f'{", ".join(backends)}')

    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()

        _ = [simplify_mesh(n, F=F, inplace=True, backend=backend, **kwargs)
             for n in config.tqdm(x,
                                  desc='Simplifying',
                                  leave=config.pbar_leave,
                                  disable=len(x) == 1 or config.pbar_hide)]

        if not inplace:
            return x
        return

    if not inplace:
        x = x.copy()

    if backend == 'open3d':
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

    if not inplace:
        return x
