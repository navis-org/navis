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

import os

import multiprocessing as mp
import networkx as nx
import numpy as np
import pandas as pd
import trimesh as tm

from typing import Union
from scipy import ndimage, stats
from tqdm.auto import tqdm

try:
    import skimage
    from skimage import measure
except ImportError:
    skimage = None

from .. import core, config, intersection, graph, morpho


logger = config.get_logger(__name__)


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


def points_to_mesh(points, res, threshold=None, denoise=True):
    """Generate mesh from point cloud.

    Briefly, the workflow is this:
      1. Partition the point cloud into voxels of size ``res``.
      2. (Optional) Discard voxels with less than ``threshold`` points inside.
      3. Turn voxels into a (M, N, K) matrix.
      4. (Optional) Denoise the matrix by a round of binary erosion + dilation
         and fill holes.
      5. Use marching cubes to produce a mesh.

    Parameters
    ----------
    points :    (N, 3) array
                Point cloud.
    res :       int
                Resolution of the voxels.
    threshold : int, optional
                Use this to ignore voxels with very few points inside.
    denoise :   bool
                Whether to use binary filters to reduce noise and smoothen
                the mesh.


    Returns
    -------
    trimesh.Trimesh

    """
    if not skimage:
        raise ImportError('Meshing requires `skimage`:\n '
                          'pip3 install scikit-image')

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'Points must be of shape (N, 3), got {points.shape}')

    # Generate counts per voxel
    vxl, cnt = np.unique((points / res).round().astype(int),
                         return_counts=True, axis=0)

    # Turn into a DataFrame
    voxels = pd.DataFrame(np.vstack(vxl), columns=['x', 'y', 'z'])
    voxels['count'] = cnt

    if threshold:
        voxels = voxels[voxels['count'] > 1]

    # Generate empty matrix
    mat = np.zeros((voxels.x.max() + 1,
                    voxels.y.max() + 1,
                    voxels.z.max() + 1))

    # Fill matrix
    mat[voxels.x, voxels.y, voxels.z] = 1

    if denoise:
        # Denoise by a round of erosion...
        mat = ndimage.binary_erosion(mat)

        # ... followed by two rounds of dilation to smoothen things out...
        mat = ndimage.binary_dilation(mat, iterations=2)

        # ... followed by a round of fill holes
        mat = ndimage.binary_fill_holes(mat)

        # And a final round of erosion to get back to the correct scale
        mat = ndimage.binary_erosion(mat, iterations=1)

    # Run the marching cube algorithm
    # (newer versions of skimage have a "marching cubes" function and
    # the marching_cubes_lewiner is deprecreated)
    marching_cubes = getattr(measure, 'marching_cubes',
                             getattr(measure, 'marching_cubes_lewiner', None))
    verts, faces, normals, values = marching_cubes(mat.astype(float),
                                                   level=0,
                                                   gradient_direction='ascent',
                                                   allow_degenerate=False,
                                                   step_size=1)
    # Turn coordinates back into original units
    verts *= res

    # Somehow we seem to have introduced an offset equal to our resolution
    verts -= res

    mesh = tm.Trimesh(vertices=verts, faces=faces, normals=normals)

    # Need to fix normals
    mesh.fix_normals()

    return mesh


def pointlabels_to_meshes(points, labels, res, method='kde',
                          threshold=0.05, drop_fluff=True, volume=None,
                          n_cores=os.cpu_count() // 2, progress=True):
    """Generate non-overlapping meshes from a labelled point cloud.

    Briefly, the default workflow is this:

      1. Create a Gaussian KDE for each unique label.
      2. Tile the point's bounding box into voxels of size ``res``.
      3. Calculate the KDE's point density function (PDF) to assign a label to
         each voxel.
      4. (Optional) Denoise the matrix by a round of binary erosion + dilation
         and fill holes.
      5. Use marching cubes to produce a mesh.

    See `method` parameter for an alternative method.

    Parameters
    ----------
    points :    (N, 3) array
                Point cloud.
    labels :    (N, ) array
                A label for each point.
    res :       int
                Size of the voxels. Note that this also determines the
                gap between meshes: higher resolution = smaller, more precise gaps.
    method :    "kde" | "majority"
                Which method to use. "kde" (default) uses above described
                workflow. "majority" vote will simply ask, for each voxel, which
                labels are contained within it and which is the most numerous.
                The latter is much faster but may produce coarser meshes - in
                particular if the number of points is low.
    threshold : float [0-1], optional
                Threshold for dropping voxels that don't appear to belong to
                any of the original labels. This is the quantile! I.e. the
                default value of 0.05 means that we'll be dropping the bottom 5%.
    drop_fluff : bool
                Whether to drop small bits and pieces from meshes and only keep
                the largest contiguous pieces.
    volume :    Volume | Trimesh, optional
                Provide a mesh to contrain the sampled voxels to inside this
                volume.
    n_cores :   int
                Number of cores to use for parallel processing. Each unique
                label will be processed on a separate core.


    Returns
    -------
    meshes  :   list
                List of ``navis.Volume``. Their names correspond to unique
                ``labels``.

    """
    if not skimage:
        raise ImportError('Meshing requires `skimage`:\n '
                          'pip3 install scikit-image')

    if len(points) != len(labels):
        raise ValueError(f'Number of labels ({len(labels)}) must match number '
                         f'of points ({len(points)})')

    assert method in ('kde', 'majority')

    points = np.asarray(points)
    labels = np.asarray(labels)
    labels_unique = np.unique(labels)

    if method == 'kde':
        # Now create voxel coordinates for the volume we want to fill:
        # First the bounding box
        bbox = np.vstack((points.min(axis=0), points.max(axis=0)))

        # We'll pad the volume by 2x the resolution
        padding = res * 2

        xco = np.arange(bbox[0][0] - padding, bbox[1][0] + padding, res)
        yco = np.arange(bbox[0][1] - padding, bbox[1][1] + padding, res)
        zco = np.arange(bbox[0][2] - padding, bbox[1][2] + padding, res)

        i = np.arange(0, xco.shape[0], 1)
        j = np.arange(0, yco.shape[0], 1)
        k = np.arange(0, zco.shape[0], 1)

        ii, jj, kk = np.meshgrid(i, j, k)
        xx, yy, zz = np.meshgrid(xco, yco, zco)

        voxels = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
        voxels = pd.DataFrame(voxels, columns=['x', 'y', 'z'])

        voxels['i'] = ii.flatten()
        voxels['j'] = jj.flatten()
        voxels['k'] = kk.flatten()

        if not isinstance(volume, type(None)):
            in_vol = intersection.in_volume(voxels[['x', 'y', 'z']].values,
                                            volume)
            print(f'Dropping {(~in_vol).sum()}/{len(voxels)} voxels outside of provided volume.')
            voxels = voxels.loc[in_vol].copy()

        # For each label create a KDE
        kde = {}
        for l in tqdm(labels_unique,
                    desc='Generating KDEs',
                    disable=not progress,
                    leave=False):
            this_p = points[labels == l]

            kde[l] = stats.gaussian_kde(this_p.T)

        # For each point get the point density function for each KDE
        combinations = [(kde[l], [voxels[['x', 'y', 'z']].values.T], {}) for l in kde]
        with mp.Pool(n_cores) as pool:
            results = list(tqdm(pool.imap(_worker_wrapper,
                                        combinations,
                                        chunksize=1),
                                leave=False,
                                disable=not progress,
                                total=len(combinations),
                                desc=f'Assigning {len(voxels):,} voxels'))

        # Fill results
        for l, r in zip(kde, results):
            voxels[l] = r

        # Drop voxels that have a PDF of less than the given threshold
        if threshold:
            pdf = voxels[labels_unique].max(axis=1)
            keep = pdf >= np.quantile(pdf, threshold)
            print(f'Dropping {(~keep).sum()}/{len(voxels)} voxels with too low density.')
            voxels = voxels.loc[keep].copy()

        # Assign label to each voxel based on the max probability
        voxels['label'] = labels_unique[np.argmax(voxels[labels_unique].values, axis=1)]
    else:
        # Turn points into voxels of given size
        points_vxl = points // res

        # Turn labels into an array of integers
        labels_dict = dict(zip(labels_unique, np.arange(len(labels_unique))))
        labels_int = np.array([labels_dict[la] for la in labels])

        # Combine (N, 3) voxels and (N, ) labels into (N, 4) array
        point_labels = np.hstack((points_vxl, labels_int.reshape(-1, 1)))

        # Count unique voxel + label combinations
        point_labels_unique, cnt = np.unique(point_labels, axis=0, return_counts=True)

        # For each x/y/z coordinate (but not the label column) get a unique index
        unique_voxel, voxel_ix = np.unique(
            point_labels_unique[:, :3], axis=0, return_inverse=True
        )

        # Turn into DataFrame for tallying up
        point_labels_df = pd.DataFrame()
        point_labels_df["voxel_ix"] = voxel_ix
        point_labels_df["label_ix"] = point_labels_unique[:, -1]
        point_labels_df["cnt"] = cnt

        # Turn into a N_point x N_labels matrix
        adj = point_labels_df.groupby(["voxel_ix", "label_ix"]).cnt.sum().unstack()

        # Now generate the DataFrame we will use to create meshes
        voxels = pd.DataFrame()
        voxels['i'] = unique_voxel[:, 0]
        voxels['j'] = unique_voxel[:, 1]
        voxels['k'] = unique_voxel[:, 2]

        # Also re-generate x/y/z coordinates
        voxels['x'] = voxels['i'] * res
        voxels['y'] = voxels['j'] * res
        voxels['z'] = voxels['k'] * res

        # Make sure i/j/k start at zero (otherwise matrix further down might
        # end up really huge)
        voxels['i'] -= unique_voxel[:, 0].min()
        voxels['j'] -= unique_voxel[:, 1].min()
        voxels['k'] -= unique_voxel[:, 2].min()

        # For each voxel get the top label...
        voxels['label'] = labels_unique[np.nanargmax(adj, axis=1)]

        # Drop voxels that have less than given points inside
        if threshold:
            top_count = np.nanmax(adj, axis=1)
            keep = top_count >= np.quantile(top_count, threshold)
            print(f'Dropping {(~keep).sum()}/{len(voxels)} voxels with too low density.')
            voxels = voxels.loc[keep].copy()

        # Some settings for the meshing
        padding = 0

    meshes = []
    for l in tqdm(labels_unique,
                  desc='Creating meshes',
                  disable=not progress,
                  leave=False):
        # Generate empty matrix
        mat = np.zeros((voxels.i.max() + 2,
                        voxels.j.max() + 2,
                        voxels.k.max() + 2))

        # Get voxels belonging to this label
        this = voxels[voxels.label == l]

        if this.empty:
            logger.warning(f'Label {l} did not produce a mesh.')
            continue

        # Fill matrix
        mat[this.i, this.j, this.k] = 1

        # Remove binary holes
        mat = ndimage.binary_fill_holes(mat)

        # We need one round of erodes to make meshes non-overlapping
        mat = ndimage.binary_erosion(mat)

        if not np.any(mat):
            logger.warning(f'Label {l} did not produce a mesh.')
            continue

        # Use marching cubes to create surface model
        # (newer versions of skimage have a "marching cubes" function and
        # the marching_cubes_lewiner is deprecreated)
        marching_cubes = getattr(measure, 'marching_cubes',
                                 getattr(measure, 'marching_cubes_lewiner', None))
        verts, faces, normals, values = marching_cubes(mat.astype(float),
                                                       level=0,
                                                       allow_degenerate=False,
                                                       step_size=1)

        # Scale back to original units
        verts *= res

        # Add offset
        offset = voxels[['x', 'y', 'z']].min(axis=0).values  # keep .values !
        verts += offset

        # Somehow we seem to have introduced an offset
        verts -= padding
        verts += 0.5 * res

        if method == 'kde':
            verts[:, 1] -= 0.5 * res

        # Make a trimesh
        new_mesh = tm.Trimesh(vertices=verts, faces=faces, normals=normals)

        if drop_fluff:
            # Drop small stuff (anything that makes up less than 10% of the faces)
            cc = tm.graph.connected_components(edges=new_mesh.face_adjacency,
                                               nodes=np.arange(len(new_mesh.faces)),
                                               min_len=1,
                                               engine=None)
            if len(cc) > 1:
                min_faces = new_mesh.faces.shape[0] * 0.1
                to_keep = [c for c in cc if (len(c) >= min_faces)]
                if to_keep:
                    new_mesh = new_mesh.submesh([np.concatenate(to_keep)])[0]

        # Need to fix normals
        new_mesh.fix_normals()

        meshes.append(core.Volume(new_mesh, name=l))

    return meshes


def _worker_wrapper(x):
    f, args, kwargs = x
    return f(*args, **kwargs)



def face_dist_sorting(x, from_, strahler_weight=False, inplace=False):
    """Sort faces by distance from given point.

    This allows you to e.g. use Blender's "build" modifier to grow neurons
    from a point of origin.

    Parameters
    ----------
    x :         navis.MeshNeuron
                Mesh to sort faces for.
    from_ :     int | list of int
                Must be either a vertex index (single int) or an x/y/z coordinate.
    strahler_weight :   bool
                If True, will use Strahler index to grow twigs slower than
                backbone.
    inplace :   bool
                Whether to modify the input mesh or a copy thereof.

    Returns
    -------
    navis.MeshNeuron

    Examples
    --------

    >>> import navis
    >>> x = navis.example_neurons(1, kind='mesh')
    >>> x = navis.meshes.mesh_utils.face_dist_sorting(x, from_=x.soma_pos)

    """
    # Turn vertex indices to coordinates
    if isinstance(from_, (int, np.integer)):
        from_ = x.vertices[from_]

    # Generate the skeleton
    # (note we don't shave to avoid issues with vertex map)
    sk = x.skeletonize(heal=True, shave=False)

    # Get the node index for our from_
    seed = sk.snap(from_)[0]

    # Get distances from
    dists = graph.geodesic_matrix(sk, from_=seed).iloc[0]

    if strahler_weight:
        sk.reroot(seed, inplace=True)
        _ = morpho.strahler_index(sk)
        dists = dists / sk.nodes.strahler_index

    # Get sorting by distance
    srt = np.argsort(dists.values)

    # Map sorting back onto vertices
    verts_srt = srt[sk.vertex_map]

    # For each face get the skeleton nodes it maps to
    face_nodes = sk.vertex_map[x.faces]

    # For each face get the mean distance
    face_dist = dists.values[face_nodes].mean(axis=1)

    # Sort the faces
    faces_srt = np.array(x.faces)[np.argsort(face_dist)]

    if not inplace:
        x = x.copy()

    x.faces = faces_srt

    return x
