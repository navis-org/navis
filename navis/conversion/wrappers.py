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

import trimesh as tm
import numpy as np

from typing import Union, Optional
from typing_extensions import Literal

from .. import core, config, utils
from .converters import (neuron2voxels, mesh2skeleton, tree2meshneuron,
                         points2skeleton)
from .meshing import voxels2mesh

logger = config.get_logger(__name__)


@utils.map_neuronlist(desc='Skeletonizing', allow_parallel=True)
def skeletonize(x: Union['core.MeshNeuron', 'core.Dotprops', np.ndarray],
                **kwargs):
    """Turn neuron into skeleton.

    Currently, we can only skeletonize meshes, dotprops and point clouds but
    are looking into ways to also do it for ``VoxelNeurons``.

    For meshes, this function is a thin-wrapper for `skeletor`. It uses sensible
    defaults for neurons but if you want to fine-tune your skeletons you should
    look into using `skeletor` directly.

    Parameters
    ----------
    x :         MeshNeuron | trimesh.Trimesh | Dotprops
                Mesh(es) to skeletonize. Note that the quality of the results
                very much depends on the mesh, so it might be worth doing some
                pre-processing (see below).
    **kwargs
                Keyword arguments are passed through to the respective
                converters:
                    - meshes: :func:`navis.conversion.mesh2skeleton`
                    - dotprops and point clouds: :func:`navis.conversion.points2skeleton`

    Returns
    -------
    skeleton :  navis.TreeNeuron
                For meshes, this has a `.vertex_map` attribute that maps each
                vertex in the input mesh to a skeleton node ID.

    See Also
    --------
    :func:`navis.drop_fluff`
                Use this if your mesh has lots of tiny free floating bits to
                reduce noise and speed up skeletonization.

    Examples
    --------
    # Skeletonize a mesh
    >>> import navis
    >>> # Get a mesh neuron
    >>> n = navis.example_neurons(1, kind='mesh')
    >>> # Convert to skeleton
    >>> sk = navis.skeletonize(n)
    >>> # Mesh vertex indices to node IDs map
    >>> sk.vertex_map                                           # doctest: +SKIP
    array([938, 990, 990, ...,  39, 234, 234])

    # Skeletonize dotprops (i.e. point-clouds)
    >>> import navis
    >>> # Get a skeleton and turn into dotprops
    >>> dp = navis.make_dotprops(navis.example_neurons(1))
    >>> # Turn back into a skeleton
    >>> sk = navis.skeletonize(dp)

    """
    if isinstance(x, (core.MeshNeuron, tm.Trimesh)):
        return mesh2skeleton(x, **kwargs)
    elif isinstance(x, (core.Dotprops, )):
        sk = points2skeleton(x.points, **kwargs)
        for attr in ('id', 'units', 'name'):
            if hasattr(x, attr):
                setattr(sk, attr, getattr(x, attr))
        return sk
    elif isinstance(x, np.ndarray):
        return points2skeleton(x.points, **kwargs)

    raise TypeError(f'Unable to skeletonize data of type {type(x)}')


@utils.map_neuronlist(desc='Voxelizing', allow_parallel=True)
def voxelize(x: 'core.BaseNeuron',
             pitch: Union[list, tuple, float],
             bounds: Optional[list] = None,
             counts: bool = False,
             vectors: bool = False,
             alphas: bool = False,
             smooth: int = 0) -> 'core.VoxelNeuron':
    """Turn neuron into voxels.

    Parameters
    ----------
    x :             TreeNeuron | MeshNeuron | Dotprops
                    Neuron(s) to voxelize. Uses the neurons' nodes, vertices and
                    points, respectively.
    pitch :         float | iterable thereof
                    Side length(s) voxels. Can be isometric (float) or an
                    iterable of dimensions in (x, y, z).
    bounds :        (3, 2)  or (2, 3) array, optional
                    Boundaries [in units of `x`] for the voxel grid. If not
                    provided, will use ``x.bbox``.
    counts :        bool
                    If True, voxel grid will have point counts for values
                    instead of just True/False.
    vectors :       bool
                    If True, will also attach a vector field as `.vectors`
                    property.
    alphas :        bool
                    If True, will also return a grid with alpha values as
                    `.alpha` property.
    smooth :        int
                    If non-zero, will apply a Gaussian filter with ``smooth``
                    as ``sigma``.

    Returns
    -------
    VoxelNeuron
                    Has the voxel grid as `.grid` and (optionally) `.vectors`
                    and `.alphas` properties. `.grid` data type depends
                    on settings:
                     - default = bool (i.e. True/False)
                     - if ``counts=True`` = integer
                     - if ``smooth=True`` = float
                    Empty voxels will have vector (0, 0, 0) and alpha 0. Also
                    note that data tables (e.g. `connectors`) are not carried
                    over from the input neuron.

    Examples
    --------
    >>> import navis
    >>> # Get a skeleton
    >>> n = navis.example_neurons(1)
    >>> # Convert to voxel neuron
    >>> vx = navis.voxelize(n, pitch='5 microns')

    """
    if isinstance(x, (core.TreeNeuron, core.MeshNeuron, core.Dotprops)):
        return neuron2voxels(x,
                             pitch=pitch,
                             bounds=bounds,
                             counts=counts,
                             vectors=vectors,
                             alphas=alphas,
                             smooth=smooth)

    raise TypeError(f'Unable to voxelize data of type {type(x)}')


@utils.map_neuronlist(desc='Meshing', allow_parallel=True)
def mesh(x: Union['core.VoxelNeuron', np.ndarray, 'core.TreeNeuron'],
         **kwargs) -> Union[tm.Trimesh, 'core.MeshNeuron']:
    """Generate mesh from object(s).

    VoxelNeurons or (N, 3) arrays of voxel coordinates will be meshed using
    a marching cubes algorithm. TreeNeurons will be meshed by creating
    cylinders using the radii.

    Parameters
    ----------
    x :             VoxelNeuron | (N, 3) np.array | TreeNeuron
                    Object to mesh. See notes above.
    **kwargs
                    Keyword arguments are passed through to the respective
                    converters: :func:`navis.conversion.voxels2mesh` and
                    :func:`navis.conversion.tree2meshneuron`, respectively.

    Returns
    -------
    mesh :          trimesh.Trimesh | MeshNeuron
                    Returns a trimesh or MeshNeuron depending on the input.
                    Data tables (e.g. `connectors`) are not carried over from
                    input neuron.

    """
    if isinstance(x, core.VoxelNeuron) or (isinstance(x, np.ndarray) and x.ndims == 2 and x.shape[1] == 3):
        return voxels2mesh(x, **kwargs)
    elif isinstance(x, core.TreeNeuron):
        return tree2meshneuron(x, **kwargs)

    raise TypeError(f'Unable to create mesh from data of type {type(x)}')
