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

import warnings

import numpy as np
import pandas as pd
import skeletor as sk
import sparsecubes
import trimesh as tm

from numbers import Number
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import Union, Optional

from .. import core, config, utils, morpho, graph

logger = config.get_logger(__name__)


@utils.map_neuronlist(desc='Skeletonizing', allow_parallel=True)
def points2skeleton(x: Union['core.Dotprops', np.ndarray],
                    k: int = 10,
                    max_dist: Optional[float] = None):
    """Turn points into skeleton.

    This function works by:
     1. Compute the `k` nearest neighbors for each point
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
        if (x.ndim != 2) or (x.shape[1] != 3):
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

    # Extract the minimum spanning tree. The kNN edges are already a weighted sparse
    # structure, so we can hand them straight to scipy instead of detouring through
    # a networkx graph.
    edges = np.asarray(edges, dtype=float)
    src = edges[:, 0].astype(np.int64)
    tgt = edges[:, 1].astype(np.int64)
    weights = edges[:, 2]

    # Collapse (a, b) and (b, a) into a single undirected edge and drop self-loops:
    # a COO matrix *sums* duplicate entries, which would otherwise double the weight
    # of every reciprocal nearest-neighbour pair.
    lo = np.minimum(src, tgt)
    hi = np.maximum(src, tgt)
    keep = lo != hi
    lo, hi, weights = lo[keep], hi[keep], weights[keep]

    _, uniq = np.unique(np.stack([lo, hi], axis=1), axis=0, return_index=True)
    lo, hi, weights = lo[uniq], hi[uniq], weights[uniq]

    # N.B. navis-fastcore has a `minimum_spanning_tree` that would slot in here
    # (and returns row indices, so it needs no weight offset), but measured
    # end-to-end it is a wash - the k-NN and the edge-building loop above
    # dominate, and the MST itself is a rounding error. Not worth a second path.
    #
    # csgraph returns the tree as a sparse matrix, in which a zero-weight edge is
    # indistinguishable from an absent one - and duplicate points sit at distance
    # exactly 0. Shifting every weight by a constant keeps them off zero without
    # changing the tree: all spanning trees have the same number of edges, so a
    # constant per-edge offset shifts every candidate's total by the same amount.
    adj = coo_matrix((weights + 1, (lo, hi)), shape=(len(pts), len(pts)))
    mst = minimum_spanning_tree(adj).tocoo()

    mst_edges = np.stack([mst.row, mst.col], axis=1)

    return graph.edges2neuron(mst_edges, vertices=pts)


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
                [`navis.heal_skeleton`][] directly. Note that this
                can be fairly costly if the mesh as many tiny fragments.
    connectors : bool
                Whether to carry over existing connector tables. This will
                attach connectors by first snapping them to the closest mesh
                vertex and then to the skeleton node corresponding to that
                vertex.
    inv_dist :  int | float | str
                Only required for method "teasar": invalidation distance for
                the traversal. Smaller `inv_dist` captures smaller features
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
    [`navis.drop_fluff`][]
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

    # Warm-up the trimesh cache (this will use fastcore if available)
    _ = utils.mesh_unique_edges(mesh, return_lengths=True)

    kwargs['progress'] = kwargs.get("progress", False)
    if method == 'wavefront':
        skeleton = sk.skeletonize.by_wavefront(mesh, **kwargs)
    elif method == 'teasar':
        skeleton = sk.skeletonize.by_teasar(mesh, inv_dist=inv_dist, **kwargs)

    props['vertex_map'] = skeleton.mesh_map

    s = core.TreeNeuron(skeleton.swc, **props)

    if s.has_soma:
        s.reroot(s.soma, inplace=True)

    if heal:
        _ = morpho.heal_skeleton(s, inplace=True, method='ALL')

    if shave:
        # --- alternative approach: drop all terminal segments that are more or less straight ---
        # # First node of all small segments
        # n1 = np.array([seg[0] for seg in s.small_segments])
        # n2 = np.array([seg[-1] for seg in s.small_segments])

        # # Which segments are terminal
        # leafs = s.leafs.node_id.values
        # bp = s.branch_points.node_id.values
        # is_term = np.isin(n1, leafs) & np.isin(n2, bp)
        # term_seg = np.array(s.small_segments, dtype=object)[is_term]

        # # Calculate geodesic length of terminal segments
        # seg_lengths = np.array([graph.segment_length(s, seg) for seg in term_seg])

        # # Calculate Euclidean tip->end distance
        # nodes = s.nodes.set_index('node_id')
        # start = nodes.loc[[seg[0] for seg in term_seg], ["x", "y", "z"]].values
        # end = nodes.loc[[seg[-1] for seg in term_seg], ["x", "y", "z"]].values
        # L = np.sqrt(((start - end) ** 2).sum(axis=1))

        # # Calculate tortuosity
        # tort = seg_lengths / L

        # # Drop all terminal segments that are more ore less straight
        # seg_to_drop = term_seg[tort <= 1.2]
        # nodes_to_drop = np.concatenate([t[:-1] for t in seg_to_drop])
        # keep = s.nodes[~s.nodes.node_id.isin(nodes_to_drop)].node_id.values

        # # Subset neuron
        # s = morpho.subset_neuron(s, keep, inplace=True)

        # # Fix vertex map
        # for seg in seg_to_drop:
        #     s.vertex_map[np.isin(s.vertex_map, seg[:-1])] = seg[-1]

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
        mapping = dict(zip(bristles.node_id.values, bristles.parent_id.values))
        vm = pd.Series(s.vertex_map)
        s.vertex_map = vm.map(mapping).fillna(vm).astype(s.vertex_map.dtype).values

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


@utils.map_neuronlist(desc='Skeletonizing', allow_parallel=True)
def voxels2skeleton(vox: Union['core.VoxelNeuron', np.ndarray],
                    method: str = 'wavefront',
                    spacing: Union[str, np.ndarray] = 'auto',
                    heal: bool = False,
                    **kwargs) -> 'core.TreeNeuron':
    """Turn voxels into a skeleton.

    Uses [`sparsecubes`][], which works directly off the sparse voxels instead
    of densifying them to a 3D grid.

    Parameters
    ----------
    vox :           VoxelNeuron | (N, 3) np.array
                    Object to skeletonize. Can be a VoxelNeuron or an (N, 3)
                    array of x, y, z voxel coordinates.
    method :        "wavefront" | "teasar" | "thin"
                    Which algorithm to use:
                      - "wavefront" (default) propagates a geodesic wave and
                        collapses each ring of equidistant voxels to its
                        centroid. Needs neither a thinning pass nor a distance
                        transform, which makes it the fastest of the three, and
                        node positions are sub-voxel. Radii come for free from
                        the rings (volume-preserving by default).
                      - "teasar" produces well-centered, medial-axis skeletons.
                        This is a sparse reimplementation of `kimimaro`.
                      - "thin" peels the voxels down to a one-voxel-wide medial
                        curve and extracts the centerline from that. Tends to
                        produce more (and shorter) branches.
                    Note this mirrors [`navis.conversion.mesh2skeleton`][],
                    which also defaults to "wavefront".
    spacing :       "auto" | (3, ) array
                    Voxel size. If "auto" and input is a `VoxelNeuron` we use
                    the neuron's `.units`, else spacing will be `(1, 1, 1)`.
    heal :          bool
                    Whether to heal the resulting skeleton if it has multiple
                    connected components. Note that voxel data is often
                    fragmented, so this can make a big difference.
    **kwargs
                    Keyword arguments are passed through to the underlying
                    `sparsecubes` skeletonizer, e.g. `min_branch_length`, or
                    `radius_agg`/`step_size` for "wavefront".

    Returns
    -------
    skeleton :      navis.TreeNeuron
                    Note that data tables (e.g. `connectors`) are not carried
                    over from the input neuron.

    See Also
    --------
    [`navis.skeletonize`][]
                    The high-level wrapper you would normally use.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind='mesh')
    >>> vx = navis.voxelize(n, pitch='2 microns')
    >>> sk = navis.conversion.voxels2skeleton(vx)
    >>> sk.n_nodes > 0
    True

    """
    utils.eval_param(vox, 'vox', allowed_types=(core.VoxelNeuron, np.ndarray))
    utils.eval_param(method, 'method',
                     allowed_values=('wavefront', 'teasar', 'thin'))

    if isinstance(spacing, str) and spacing == 'auto':
        if not isinstance(vox, core.VoxelNeuron):
            spacing = np.array([1, 1, 1])
        else:
            spacing = vox.units_xyz.magnitude

    voxels = vox.voxels if isinstance(vox, core.VoxelNeuron) else vox

    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise ValueError(f'Voxels must be shape (N, 3), got {voxels.shape}')

    if not len(voxels):
        raise ValueError('Unable to skeletonize empty voxel data.')

    if method == 'wavefront':
        # Radii fall out of the ring contraction, so nothing extra to ask for
        skel = sparsecubes.wavefront_skeletonize(voxels, spacing=spacing,
                                                 **kwargs)
    elif method == 'teasar':
        skel = sparsecubes.teasar_skeletonize(voxels, spacing=spacing, **kwargs)
    else:
        # `radii=True` so the SWC gets meaningful radii instead of zeros
        kwargs['radii'] = kwargs.get('radii', True)
        skel = sparsecubes.thin_skeletonize(voxels, spacing=spacing, **kwargs)

    swc = pd.DataFrame(skel.to_swc(), columns=['node_id', 'label', 'x', 'y', 'z',
                                               'radius', 'parent_id'])
    swc['node_id'] = swc.node_id.astype(int)
    swc['parent_id'] = swc.parent_id.astype(int)
    swc['label'] = swc.label.astype(int)

    props = {}
    if isinstance(vox, core.VoxelNeuron):
        # Skeleton coordinates are in the voxel grid's own frame - shift them
        # into the same space as the neuron's bounding box and connectors
        swc[['x', 'y', 'z']] += vox.offset
        # `spacing` has already been applied, so one unit is now one `units`
        props = {'units': f'1 {vox.units.units}', 'id': vox.id, 'name': vox.name}

    s = core.TreeNeuron(swc, **props)

    if heal:
        _ = morpho.heal_skeleton(s, inplace=True, method='ALL')

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
                Offset for voxels. Will be (0, 0, 0) if `strip=False`.

    See Also
    --------
    [`navis.neuron2voxelgrid`][]
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
    else:
        raise TypeError(f'Expected TreeNeuron, Dotprops or MeshNeuron, got '
                        f'"{type(x)}"')

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
                  fill: bool = True,
                  fill_cavities: bool = False,
                  depth: bool = False,
                  smooth: int = 0) -> 'core.VoxelNeuron':
    """Turn neuron into voxels.

    Parameters
    ----------
    x :             TreeNeuron | MeshNeuron | Dotprops
                    Neuron(s) to voxelize. TreeNeurons and Dotprops are
                    voxelized by binning their nodes and points, respectively.
                    MeshNeurons are voxelized properly via
                    [`sparsecubes.voxelize`][]: their surface is walked and the
                    interior filled, which - unlike binning the vertices -
                    does not miss faces larger than a voxel. Note that
                    `counts`, `vectors` and `alphas` are per-point quantities
                    and fall back to binning the mesh's vertices.
    pitch :         float | iterable thereof
                    Side length(s) voxels. Can be isometric (float) or an
                    iterable of dimensions in (x, y, z).
    bounds :        (3, 2)  or (2, 3) array, optional
                    Boundaries [in units of `x`] for the voxel grid. If not
                    provided, will use `x.bbox`.
    counts :        bool
                    If True, voxel grid will have point counts for values
                    instead of just True/False.
    vectors :       bool
                    If True, will also attach a vector field as `.vectors`
                    property.
    alphas :        bool
                    If True, will also return a grid with alpha values as
                    `.alpha` property.
    fill :          bool
                    Only for MeshNeurons voxelized via `sparsecubes` (i.e. when
                    `counts`, `vectors` and `alphas` are all False): if True
                    (default), fill the mesh interior; if False, keep only the
                    surface shell. The shell is a robust fallback for meshes
                    where the interior fill misbehaves (e.g. badly
                    non-watertight meshes).
    fill_cavities : bool
                    Only for MeshNeurons voxelized via `sparsecubes`: if True,
                    fill enclosed cavities via
                    [`sparsecubes.binary.fill_cavities`][]. Useful to patch
                    voids left by a non-watertight surface.
    depth :         bool
                    Only for MeshNeurons voxelized via `sparsecubes`: if True,
                    weigh each voxel by its distance to the surface (via
                    [`sparsecubes.measure.distance_transform`][]) instead of a
                    plain True/False occupancy, producing a float grid in which
                    deep/thick regions (e.g. the soma) have larger values than
                    thin neurites. Mutually exclusive with `counts`, `vectors`
                    and `alphas`.
    smooth :        int
                    If non-zero, will apply a Gaussian filter with `smooth`
                    as `sigma`.

    Returns
    -------
    VoxelNeuron
                    Has the voxel grid as `.grid` and (optionally) `.vectors`
                    and `.alphas` properties. `.grid` data type depends
                    on settings:
                     - default = bool (i.e. True/False)
                     - if `counts=True` = integer
                     - if `smooth=True` = float
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

    # Meshes have faces and can therefore be voxelized properly - walking the
    # surface and filling the interior - instead of just binning their vertices,
    # which misses any face larger than a voxel. `counts`, `vectors` and
    # `alphas` are per-point quantities though, so those still need the points.
    is_mesh = isinstance(x, (core.MeshNeuron, tm.Trimesh))
    mesh_voxelize = is_mesh and not counts and not vectors and not alphas

    if depth and not is_mesh:
        raise ValueError("`depth=True` is only available for MeshNeurons.")
    if depth and (counts or vectors or alphas):
        raise ValueError(
            "`depth=True` is mutually exclusive with `counts`, `vectors` and "
            "`alphas`."
        )

    if isinstance(bounds, type(None)):
        bounds = x.bbox
    else:
        bounds = np.asarray(bounds)

    if bounds.shape == (2, 3):
        bounds = bounds.T

    # Shape of grid
    dim = np.ceil(bounds[:, 1] / pitch) - np.floor(bounds[:, 0] / pitch)
    shape = np.ceil(dim).astype(int) + 1

    # `counts` produces an integer grid and `smooth` a float32 one - check
    # against whichever dtype we will actually end up holding.
    # Note this has to happen *before* voxelizing: work scales with the number
    # of voxels, so a pitch fine enough to blow up the grid would grind away for
    # a very long time before we ever got here to reject it.
    if counts:
        grid_dtype = np.int64
    elif depth or smooth:
        grid_dtype = np.float32
    else:
        grid_dtype = bool
    utils.check_grid_size(
        shape, grid_dtype, hint="Try a coarser `pitch` or tighter `bounds`."
    )

    # Convert to voxel indices. Note `sparsecubes` uses the same convention as
    # `_make_voxels` - voxel `i` is centred on `i * pitch` - so the two produce
    # indices in the same space and everything downstream is shared.
    dist = None
    if mesh_voxelize:
        ix = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vxl = sparsecubes.voxelize(
                x.trimesh if isinstance(x, core.MeshNeuron) else x,
                spacing=pitch,
                solid=fill,
            )

        # Neuron meshes are routinely not watertight, so `sparsecubes` warning
        # about unfilled columns is expected rather than exceptional - demote it
        # to a debug log. Anything else is re-raised: we are only quietening the
        # one known-noisy case, not swallowing warnings wholesale.
        for w in caught:
            if "watertight" in str(w.message):
                logger.debug(f"Voxelizing {x.id}: {w.message}")
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

        # Optionally patch enclosed cavities left by a non-watertight surface
        if fill_cavities:
            vxl = sparsecubes.binary.fill_cavities(vxl)

        # Distance-to-surface weighting is computed on the full (absolute) voxel
        # set so the field reflects the whole object, then carried through the
        # out-of-bounds crop below.
        if depth:
            dist = sparsecubes.measure.distance_transform(vxl, spacing=pitch)
    else:
        ix, _ = _make_voxels(x=x, pitch=pitch, strip=False)

    # Get unique voxels (`sparsecubes` already returns them sorted + deduplicated)
    cnt = None
    if mesh_voxelize:
        pass
    elif counts:
        vxl, cnt = np.unique(ix, axis=0, return_counts=True)
    else:
        vxl = np.unique(ix, axis=0)

    # Substract lower bounds
    offset = (bounds[:, 0] / pitch)
    vxl = vxl - offset.round().astype(int)
    if ix is not None:
        ix = ix - offset.round().astype(int)

    # Drop voxels outside the defined bounds, carrying `cnt`/`dist` along so
    # they stay row-aligned with `vxl`
    inb = (vxl.min(axis=1) >= 0) & np.all(vxl < shape, axis=1)
    vxl = vxl[inb]
    if cnt is not None:
        cnt = cnt[inb]
    if dist is not None:
        dist = dist[inb]

    # Generate and populate grid
    if depth:
        grid = np.zeros(shape=shape, dtype=np.float32)
        grid[vxl[:, 0], vxl[:, 1], vxl[:, 2]] = dist
    elif counts:
        grid = np.zeros(shape=shape, dtype=int)
        grid[vxl[:, 0], vxl[:, 1], vxl[:, 2]] = cnt
    else:
        grid = np.zeros(shape=shape, dtype=bool)
        grid[vxl[:, 0], vxl[:, 1], vxl[:, 2]] = True

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
    inv = np.asarray(inv).ravel()

    # Pre-group points by voxel so we don't rescan all points each iteration
    # (the old `pts[inv == i]` was O(n_voxels * n_points))
    counts = np.bincount(inv, minlength=len(uni))
    groups = np.split(pts[np.argsort(inv, kind='stable')], np.cumsum(counts)[:-1])

    # Go over each voxel
    for i, pt in enumerate(groups):
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
    if alphas:
        n.alphas = alph

    return n


@utils.map_neuronlist(desc='Converting', allow_parallel=True)
def tree2meshneuron(x: 'core.TreeNeuron',
                    tube_points: int = 8,
                    radius_scale_factor: float = 1,
                    use_normals: bool = True,
                    warn_missing_radii: bool = True
                    ) -> 'core.MeshNeuron':
    """Convert TreeNeuron to MeshNeuron.

    Uses the `radius` to convert skeleton to 3D tube mesh. Missing radii are
    treated as zeros.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
                    Neuron to convert.
    tube_points :   int
                    Number of points making up the circle of the cross-section
                    of the tube.
    radius_scale_factor : float
                    Factor to scale radii by.
    use_normals :   bool
                    If True will rotate tube along its curvature.
    warn_missing_radii : bool
                    Whether to warn if radii are missing or <= 0.

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

    # Map segments of node IDs to segments of node indices
    id2ix = dict(zip(x.nodes.node_id, np.arange(len(x.nodes))))
    segments = [np.array([id2ix[n] for n in seg]) for seg in x.segments]

    # Note that we are treating missing radii as "0"
    radii_map = x.nodes.radius.fillna(0).values
    if warn_missing_radii and (radii_map <= 0).any():
        logger.warning('At least some radii are missing or <= 0. Mesh may look funny.')

    # Map radii onto segments
    radii = [radii_map[seg] * radius_scale_factor for seg in segments]
    co_map = x.nodes[['x', 'y', 'z']].values
    seg_points = [co_map[seg] for seg in segments]

    vertices, faces = make_tube(seg_points,
                                radii=radii,
                                tube_points=tube_points,
                                use_normals=use_normals)

    # Note: the `process=False` is necessary to not break correspondence
    # by e.g. merging duplicate vertices
    m = core.MeshNeuron({'vertices': vertices, 'faces': faces},
                        units=x.units, name=x.name, id=x.id, process=False)


    # For each vertex, track the original node: the first `tube_points` vertices
    # correspond to the first node of the first segment and so on.
    m.vertex_map = np.concatenate([np.repeat(seg, tube_points) for seg in segments])

    return m
