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
import skeletor as sk
import trimesh as tm
import networkx as nx

from numbers import Number
from scipy.ndimage import gaussian_filter
from typing import Union, Optional

from .. import core, config, utils, morpho, graph

logger = config.get_logger(__name__)


@utils.map_neuronlist(desc='Skeletonizing', allow_parallel=True)
def points2skeleton(x: Union['core.Dotprops', np.ndarray],
                    k: int = 10,
                    max_dist: Optional[float] = None):
    """Turn points into skeleton.

    This function works by:
     1. Compute the ``k`` nearest neighbors for each point
     2. Generate a graph from the nearest-neighbor edges
     3. Extract a minimum-spanning tree (MST) from the graph
     4. Process the MST into a skeleton

    Parameters
    ----------
    x :         (N, 3) array | Dotprops
                Points to skeletonize.
    k :         int
                Number of nearest neighbors to consider. Too low values of `k`
                can lead to disconnected skeletons.
    max_dist :  float, optional
                Edges longer than this will be ignored. This can lead to a
                fragmented (i.e. multi-root) skeleton!

    Returns
    -------
    skeleton :  navis.TreeNeuron

    Examples
    --------
    >>> import navis
    >>> # Get a mesh neuron
    >>> n = navis.example_neurons(1)
    >>> # Get the points
    >>> pts = n.nodes[['x', 'y', 'z']].values
    >>> # Convert points back into skeleton
    >>> sk = navis.conversion.points2skeleton(pts)

    """
    utils.eval_param(x, name='x', allowed_types=(core.Dotprops, np.ndarray))

    if isinstance(x, core.Dotprops):
        pts = x.points
    else:
        if (x.ndim != 2) and (x.shape[1] != 3):
            raise ValueError(f'Points must be shape (N, 3), got {x.shape}')
        pts = x

    # Get the list of nearest neighbours
    tree = core.dotprop.KDTree(pts)

    defaults = {}
    if max_dist is not None:
        # We have to avoid passing `None` because scipy's KDTree does not like
        # that (pykdtree does not care)
        defaults['distance_upper_bound'] = max_dist
    dists, NN = tree.query(pts, k=k + 1, **defaults)

    # Drop self-hits
    dists, NN = dists[:, 1:], NN[:, 1:]

    # Turn into edges
    edges = []
    ix1 = np.arange(len(dists))
    for i in range(k):
        ix2 = NN[:, i]
        le = dists[:, i]
        # If a max dist was set we have to remove NN that have dist np.inf
        if max_dist is None:
            edges += list(zip(ix1, ix2, le))
        else:
            not_inf = le != np.inf
            edges += list(zip(ix1[not_inf], ix2[not_inf], le[not_inf]))

    # Generate graph
    G = nx.Graph()
    G.add_nodes_from(ix1)
    G.add_weighted_edges_from(edges)

    # Extract minimum spanning tree
    G_mst = nx.minimum_spanning_tree(G)

    # Add the coordinates as node properties
    nx.set_node_attributes(G_mst, dict(zip(G.nodes, pts[:, 0])), name='x')
    nx.set_node_attributes(G_mst, dict(zip(G.nodes, pts[:, 1])), name='y')
    nx.set_node_attributes(G_mst, dict(zip(G.nodes, pts[:, 2])), name='z')

    return graph.nx2neuron(G_mst)


@utils.map_neuronlist(desc='Skeletonizing', allow_parallel=True)
def mesh2skeleton(x: 'core.MeshNeuron',
                  method: str = 'wavefront',
                  fix_mesh: bool = False,
                  shave: bool = True,
                  heal: bool = False,
                  connectors: bool = False,
                  inv_dist: Union[int, float] = None,
                  **kwargs):
    """Turn mesh neuron into skeleton.

    This function is a thin-wrapper for `skeletor`. It uses sensible defaults
    for neurons but if you want to fine-tune your skeletons you should look
    into using `skeletor` directly.

    Parameters
    ----------
    x :         MeshNeuron | trimesh.Trimesh
                Mesh(es) to skeletonize. Note that the quality of the results
                very much depends on the mesh, so it might be worth doing some
                pre-processing (see below).
    method :    'wavefront' | 'teasar'
                Method to use for skeletonization:
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
    shave :     bool
                Whether to "shave" the resulting skeleton to reduce bristles
                on the backbone.
    heal :      bool | "LEAFS" | "ALL"
                Whether to heal the resulting skeleton if it is fragmented.
                For more control over the stitching set `heal=False` and use
                :func:`navis.heal_skeleton` directly. Note that this
                can be fairly costly if the mesh as many tiny fragments.
    connectors : bool
                Whether to carry over existing connector tables. This will
                attach connectors by first snapping them to the closest mesh
                vertex and then to the skeleton node corresponding to that
                vertex.
    inv_dist :  int | float | str
                Only required for method "teasar": invalidation distance for
                the traversal. Smaller ``inv_dist`` captures smaller features
                but is slower and more noisy, and vice versa. A good starting
                value is around 2-5 microns. Can be a unit string - e.g.
                "5 microns" - if your neuron has its units set.
    **kwargs
                Additional keyword arguments are passed through to the respective
                function in `skeletor` - i.e. `by_wavefront` or `by_teasar`.

    Returns
    -------
    skeleton :  navis.TreeNeuron
                Has a `.vertex_map` attribute that maps each vertex in the
                input mesh to a skeleton node ID.

    See Also
    --------
    :func:`navis.drop_fluff`
                Use this if your mesh has lots of tiny free floating bits to
                reduce noise and speed up skeletonization.

    Examples
    --------
    >>> import navis
    >>> # Get a mesh neuron
    >>> n = navis.example_neurons(1, kind='mesh')
    >>> # Convert to skeleton
    >>> sk = navis.conversion.mesh2skeleton(n)
    >>> # Mesh vertex indices to node IDs map
    >>> sk.vertex_map                                           # doctest: +SKIP
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
        if x.has_soma_pos:
            props['soma_pos'] = x.soma_pos

        if not isinstance(inv_dist, type(None)):
            inv_dist = x.map_units(inv_dist)

        mesh = x.trimesh
    else:
        mesh = x
        x = core.MeshNeuron(x)

    if fix_mesh:
        mesh = sk.pre.fix_mesh(mesh, remove_disconnected=False)

    kwargs['progress'] = False
    if method == 'wavefront':
        skeleton = sk.skeletonize.by_wavefront(mesh, **kwargs)
    elif method == 'teasar':
        skeleton = sk.skeletonize.by_teasar(x, inv_dist=inv_dist, **kwargs)

    props['vertex_map'] = skeleton.mesh_map

    s = core.TreeNeuron(skeleton.swc, **props)

    if s.has_soma:
        s.reroot(s.soma, inplace=True)

    if heal:
        _ = morpho.heal_skeleton(s, inplace=True, method='ALL')

    if shave:
        # Find single node bristles
        leafs = s.leafs.node_id.values

        # Make sure we keep the soma
        if s.has_soma:
            leafs = leafs[~np.isin(leafs, s.soma)]

        bp = s.branch_points.node_id.values
        bristles = s.nodes[s.nodes.node_id.isin(leafs)
                           & s.nodes.parent_id.isin(bp)]

        # Subset neuron
        keep = s.nodes[~s.nodes.node_id.isin(bristles.node_id)].node_id.values
        s = morpho.subset_neuron(s, keep, inplace=True)

        # Fix vertex map
        for b, p in zip(bristles.node_id.values, bristles.parent_id.values):
            s.vertex_map[s.vertex_map == b] = p

    # In particular with method wavefront, some nodes (mostly leafs) can have
    # a radius of 0. We will fix this here by giving them 1/2 the radius of
    # their parent nodes'
    to_fix = (s.nodes.radius == 0) & (s.nodes.parent_id >= 0)
    if any(to_fix):
        radii = s.nodes.set_index('node_id').radius
        new_radii = radii.loc[s.nodes.loc[to_fix].parent_id].values / 2
        s.nodes.loc[to_fix, 'radius'] = new_radii

    # Last but not least: map connectors
    if connectors and x.has_connectors:
        cn_table = x.connectors.copy()

        # A connector/id column is currently required for skeletons but not
        # meshes
        if not any(np.isin(('id', 'connector_id'), cn_table.columns)):
            cn_table.insert(0, 'connector_id', np.arange(len(cn_table)))

        cn_table['node_id'] = x.snap(cn_table[['x', 'y', 'z']].values)[0]
        node_map = dict(zip(np.arange(len(s.vertex_map)), s.vertex_map))
        cn_table['node_id'] = cn_table.node_id.map(node_map)
        s.connectors = cn_table

    return s


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
    ix, _ = _make_voxels(x=x, pitch=pitch, strip=False)

    if isinstance(bounds, type(None)):
        bounds = x.bbox
    else:
        bounds = np.asarray(bounds)

    if bounds.shape == (2, 3):
        bounds = bounds.T

    # Shape of grid
    dim = np.ceil(bounds[:, 1] / pitch) - np.floor(bounds[:, 0] / pitch)
    shape = np.ceil(dim).astype(int) + 1

    # Get unique voxels
    if not counts:
        vxl = np.unique(ix, axis=0)
    else:
        vxl, cnt = np.unique(ix, axis=0, return_counts=True)

    # Substract lower bounds
    offset = (bounds[:, 0] / pitch)
    vxl = vxl - offset.round().astype(int)
    ix = ix - offset.round().astype(int)

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

    # Apply Gaussian filter
    if smooth:
        grid = gaussian_filter(grid.astype(np.float32), sigma=smooth)

    # Generate neuron
    units = [f'{p * u} {x.units.units}' for p, u in zip(utils.make_iterable(pitch),
                                                        x.units_xyz.magnitude)]
    offset = offset * pitch * x.units_xyz.magnitude
    n = core.VoxelNeuron(grid, id=x.id, name=x.name, units=units, offset=offset)

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


@utils.map_neuronlist(desc='Converting', allow_parallel=True)
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
