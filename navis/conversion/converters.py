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
import skeletor as sk
import trimesh as tm

from numbers import Number
from scipy.ndimage import gaussian_filter
from typing import Union, Optional

from .. import core, config, utils, morpho

logger = config.logger


@utils.map_neuronlist(desc='Skeletonizing', allow_parallel=True)
def mesh2skeleton(x: 'core.MeshNeuron',
                  method: str = 'wavefront',
                  fix_mesh: bool = False,
                  heal: bool = False,
                  inv_dist: Union[int, float] = None,
                  **kwargs):
    """Turn mesh neuron into skeleton.

    This function is a thin-wrapper for `skeletor`. It uses sensible defaults
    for neurons but if you want to fine-tune your skeletons you should look
    into using `skeletor` directly.

    Parameters
    ----------
    x :         MeshNeuron | trimesh.Trimesh
                Mesh(es) to skeletonize.
    method :    'wavefront' | 'teasar'
                Method to use for skeletonization. The quality of the results
                very much depends on the mesh but broadly speaking:
                 - "wavefront": fast but noisier, skeletons will be ~centered
                   within the neuron
                 - "teasar": slower but smoother, skeletons follow the
                   surface of the mesh, requires the `inv_dist` parameter to be
                   set
                "wavefront" also produces radii, "teasar" doesn't.
    fix_mesh :  bool
                Whether to try to fix some common issues in the mesh before
                skeletonization. Note that this might compromise the
                vertex-to-node-ID mapping.
    heal :      bool
                Whether to heal the resulting skeleton if it is fragmented.
                For more control over the stitching set `heal=False` and use
                :func:`navis.heal_fragmented_neuron` directly.
    inv_dist :  int | float
                Only required foor method "teasar": invalidation distance for
                the traversal. Smaller ``inv_dist`` captures smaller features
                but is slower and vice versa. A good starting value is around
                2-5 microns.
    **kwargs
                Additional keyword arguments are passed through to the respective
                function in `skeletor` - i.e. `by_wavefront` or `by_teasar`.

    Returns
    -------
    skeleton :  navis.TreeNeuron
                Has a `.vertex_map` attribute that maps each vertex in the
                input mesh to a skeleton node ID. Note that data tables
                (e.g. `connectors`) are not carried over from the input neuron.

    Examples
    --------
    >>> import navis
    >>> # Get a mesh neuron
    >>> n = navis.example_neurons(1, kind='mesh')
    >>> # Convert to skeleton
    >>> sk = navis.conversion.mesh2skeleton(n)
    >>> # Mesh vertex indices to node IDs map
    >>> sk.vertex_map
    array([938, 990, 990, ...,  39, 234, 234])

    """
    utils.eval_param(x, name='x', allowed_types=(core.MeshNeuron, tm.Trimesh))
    utils.eval_param(method, name='method', allowed_values=('wavefront', 'teasar'))

    if method == 'teasar' and inv_dist is None:
        raise ValueError('Must set `inv_dist` parameter when using method '
                         '"teasar". A good starting value is around 2-5 microns.')

    props = {'soma': None}
    if isinstance(x, core.MeshNeuron):
        props.update({'id': x.id, 'name': x.name, 'units': x.units})
        if x.has_soma:
            props['soma_pos'] = x.soma_pos
        x = x.trimesh

    if fix_mesh:
        x = sk.pre.fix_mesh(x, drop_disconnected=False)

    kwargs['progress'] = not config.pbar_hide
    if method == 'wavefront':
        skeleton = sk.skeletonize.by_wavefront(x, **kwargs)
    elif method == 'teasar':
        skeleton = sk.skeletonize.by_teasar(x, inv_dist=inv_dist, **kwargs)

    props['vertex_map'] = skeleton.mesh_map

    skeleton = core.TreeNeuron(skeleton.swc, **props)

    if heal:
        _ = morpho.heal_fragmented_neuron(skeleton, inplace=True, method='ALL')

    return skeleton


def _make_voxels(x: 'core.BaseNeuron',
                 pitch: Union[list, tuple, float],
                 strip: bool = False):
    """Turn neuron into voxels.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | Dotprops
                Neuron(s) to voxelize. Uses the neurons' nodes, vertices and
                points, respectively.
    pitch :     float | iterable thereof
                Side length(s) of voxels. Can be isometric (float) or an
                iterable of dimensions in (x, y, z).
    strip :     bool
                Whether to strip empty leading voxels.

    Returns
    -------
    voxels :    numpy array
                Array of voxel indices.
    offset :    (3, ) numpy array
                Offset for voxels. Will be (0, 0, 0) if ``strip=False``.

    See Also
    --------
    :func:`navis.neuron2voxelgrid`
                Use this function to create a voxel neuron covering a specific
                volume. Useful e.g. when wanting to directly compare two
                neurons.

    """
    utils.eval_param(x, name='x', allowed_types=(core.BaseNeuron, tm.Trimesh))

    if not utils.is_iterable(pitch):
        if not isinstance(pitch, Number):
            raise TypeError('Expected `pitch` to be a number (or list thereof)'
                            f', got {type(pitch)}')
        pitch = [pitch] * 3
    elif len(pitch) != 3:
        raise ValueError('`pitch` must be single number or a list of three')

    if isinstance(x, core.TreeNeuron):
        pts = x.nodes[['x', 'y', 'z']].values
    elif isinstance(x, core.Dotprops):
        pts = x.points
    elif isinstance(x, (core.MeshNeuron, tm.Trimesh)):
        pts = np.array(x.vertices)

    # Convert points to voxel indices
    ix = (pts / pitch).round().astype(int)

    # Make neuron
    if strip:
        offset = ix.min(axis=0) * pitch
        ix = ix - ix.min()
    else:
        offset = (0, 0, 0)

    return ix, offset


@utils.map_neuronlist(desc='Voxelizing', allow_parallel=True)
def neuron2voxels(x: 'core.BaseNeuron',
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
    >>> vx = navis.conversion.neuron2voxels(n, pitch='5 microns')

    """
    if not utils.is_iterable(pitch):
        # Map units (non-str are just passed through)
        pitch = x.map_units(pitch, on_error='raise')
        if not isinstance(pitch, Number):
            raise TypeError('Expected `pitch` to be a number (or list thereof)'
                            f', got {type(pitch)}')
        pitch = [pitch] * 3
    elif len(pitch) != 3:
        raise ValueError('`pitch` must be single number or a list of three')
    else:
        pitch = np.array([x.map_units(p, on_error='raise') for p in pitch])

    # Convert to voxel indices
    ix, offset = _make_voxels(x=x, pitch=pitch, strip=False)

    if isinstance(bounds, type(None)):
        bounds = x.bbox
    else:
        bounds = np.asarray(bounds)

    if bounds.shape == (2, 3):
        bounds = bounds.T

    # Shape of grid
    dim = np.ceil(bounds[:, 1]) - np.floor(bounds[:, 0])
    shape = np.ceil(dim / pitch).astype(int) + 1

    # Get unique voxels
    if not counts:
        vxl = np.unique(ix, axis=0)
    else:
        vxl, cnt = np.unique(ix, axis=0, return_counts=True)

    # Substract lower bounds
    vxl = vxl - (bounds[:, 0] / pitch).round().astype(int)
    ix = ix - (bounds[:, 0] / pitch).round().astype(int)

    # Drop voxels outside the defined bounds
    vxl = vxl[vxl.min(axis=1) >= 0]
    vxl = vxl[np.all(vxl < shape, axis=1)]

    # Generate grid
    grid = np.zeros(shape=shape, dtype=bool)

    # Populate grid
    if not counts:
        grid[vxl[:, 0], vxl[:, 1], vxl[:, 2]] = True
    else:
        grid = grid.astype(int)
        grid[vxl[:, 0], vxl[:, 1], vxl[:, 2]] = cnt

    if smooth:
        grid = gaussian_filter(grid.astype(float), sigma=smooth)

    # Generate neuron
    n = core.VoxelNeuron(grid, id=x.id, name=x.name)

    # If no vectors required, we can just return now
    if not vectors and not alphas:
        return n

    if isinstance(x, core.TreeNeuron):
        pts = x.nodes[['x', 'y', 'z']].values
    elif isinstance(x, core.Dotprops):
        pts = x.points
    elif isinstance(x, core.MeshNeuron):
        pts = np.array(x.vertices)

    # Generate an empty vector field
    vects = np.zeros((grid.shape[0], grid.shape[1], grid.shape[2], 3),
                     dtype=np.float32)
    alph = np.zeros(grid.shape, dtype=np.float32)

    # Get unique voxels
    uni, inv = np.unique(ix, axis=0, return_inverse=True)

    # Go over each voxel
    for i in range(len(uni)):
        # Get points in this voxel
        pt = pts[inv == i]

        # Reshape
        pt = pt.reshape(1, -1, 3)

        # Generate centers for each cloud of k nearest neighbors
        centers = np.mean(pt, axis=1)

        # Generate vector from center
        cpt = pt - centers.reshape((pt.shape[0], 1, 3))

        # Get inertia (N, 3, 3)
        inertia = cpt.transpose((0, 2, 1)) @ cpt

        # Extract vector and alpha
        u, s, vh = np.linalg.svd(inertia)
        vect = vh[:, 0, :]

        # No alpha if only one point
        if pt.shape[1] > 1:
            alpha = (s[:, 0] - s[:, 1]) / np.sum(s, axis=1)
        else:
            alpha = [0]

        vects[uni[i][0], uni[i][1], uni[i][2]] = vect.flatten()
        alph[uni[i][0], uni[i][1], uni[i][2]] = alpha[0]

    if vectors:
        n.vectors = vects
    if alpha:
        n.alphas = alpha

    return n


@utils.map_neuronlist(desc='Skeletonizing', allow_parallel=True)
def tree2meshneuron(x: 'core.TreeNeuron',
                    tube_points: int = 8,
                    use_normals: bool = True) -> 'core.MeshNeuron':
    """Convert TreeNeuron to MeshNeuron.

    Uses the ``radius`` to convert skeleton to 3D tube mesh. Missing radii are
    treated as zeros.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
                    Neuron to convert.
    tube_points :   int
                    Number of points making up the circle of the cross-section
                    of the tube.
    use_normals :   bool
                    If True will rotate tube along its curvature.

    Returns
    -------
    TreeNeuron
                    Data tables (e.g. `connectors`) are not carried over from
                    the input neuron.

    Examples
    --------
    >>> import navis
    >>> # Get a skeleton
    >>> n = navis.example_neurons(1)
    >>> # Convert to mesh neuron
    >>> m = navis.conversion.tree2meshneuron(n)

    """
    # Delay to avoid circular imports
    from ..plotting.plot_utils import make_tube

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron, got "{type(x)}"')

    # Note that we are treating missing radii as "0"
    radii_map = x.nodes.set_index('node_id').radius.fillna(0)

    if (radii_map <= 0).any():
        logger.warning('At least some radii are missing or <= 0. Mesh will '
                       'look funny.')

    # Map radii onto segments
    radii = [radii_map.loc[seg].values for seg in x.segments]
    co_map = x.nodes.set_index('node_id')[['x', 'y', 'z']]
    seg_points = [co_map.loc[seg].values for seg in x.segments]

    vertices, faces = make_tube(seg_points,
                                radii=radii,
                                tube_points=tube_points,
                                use_normals=use_normals)

    return core.MeshNeuron({'vertices': vertices, 'faces': faces},
                           units=x.units, name=x.name, id=x.id)
