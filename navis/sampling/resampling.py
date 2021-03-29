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
import pandas as pd
import numpy as np
import scipy.spatial
import scipy.interpolate

from typing import Union, Optional, List, overload
from typing_extensions import Literal

from .. import config, core, utils, graph

# Set up logging
logger = config.logger

__all__ = ['resample_neuron', 'resample_along_axis']


@overload
def resample_neuron(x: 'core.NeuronObject',
                    resample_to: int,
                    inplace: Literal[False],
                    method: str = 'linear',
                    skip_errors: bool = True
                    ) -> 'core.NeuronObject': ...


@overload
def resample_neuron(x: 'core.NeuronObject',
                    resample_to: int,
                    inplace: Literal[True],
                    method: str = 'linear',
                    skip_errors: bool = True
                    ) -> None: ...


@overload
def resample_neuron(x: 'core.NeuronObject',
                    resample_to: int,
                    inplace: bool = False,
                    method: str = 'linear',
                    skip_errors: bool = True
                    ) -> None: ...


def resample_neuron(x: 'core.NeuronObject',
                    resample_to: int,
                    inplace: bool = False,
                    method: str = 'linear',
                    skip_errors: bool = True
                    ) -> Optional['core.NeuronObject']:
    """Resample neuron(s) to given resolution.

    Preserves root, leafs, branchpoints. Soma, connectors and node tags
    (if present) are mapped onto the closest node in the resampled neuron.

    Important
    ---------
    A few things to keep in mind:
      - This generates an entirely new set of node IDs! Those will be unique
        within a neuron, but you may encounter duplicates across neurons.
      - Any non-standard node table columns (e.g. "labels") will be lost.
      - Soma(s) will be pinned to the closest node in the resampled neuron.


    Also: be aware that high-resolution neurons will use A LOT of memory.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
                        Neuron(s) to resample.
    resample_to :       int | float
                        Target sampling resolution, i.e. one node every
                        N units of cable. Note that hitting the exact
                        sampling resolution might not be possible e.g. if
                        a branch is shorter than the target resolution.
    method :            str, optional
                        See ``scipy.interpolate.interp1d`` for possible
                        options. By default, we're using linear interpolation.
    inplace :           bool, optional
                        If True, will modify original neuron. If False, a
                        resampled copy is returned.
    skip_errors :       bool, optional
                        If True, will skip errors during interpolation and
                        only print summary.

    Returns
    -------
    TreeNeuron/List
                        Downsampled neuron(s). Only if ``inplace=False``.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> # Check sampling resolution (nodes/cable)
    >>> round(n.sampling_resolution)
    60
    >>> # Resample to 1 micron (example neurons are in nm space)
    >>> n_rs = navis.resample_neuron(n,
    ...                              resample_to=1000,
    ...                              inplace=False)
    >>> round(n_rs.sampling_resolution)
    195

    See Also
    --------
    :func:`navis.downsample_neuron`
                        This function reduces the number of nodes instead of
                        resample to certain resolution. Useful if you are
                        just after some simplification e.g. for speeding up
                        your calculations or you want to preserve node IDs.
    :func:`navis.resample_along_axis`
                        Resample neuron along a single axis such that nodes
                        align with given 1-dimensional grid.

    """
    if isinstance(x, core.NeuronList):
        if not inplace:
            x = x.copy()
        _ = [resample_neuron(x[i],
                             resample_to=resample_to,
                             method=method, inplace=True,
                             skip_errors=skip_errors)
             for i in config.trange(x.shape[0],
                                    desc='Resampl. neurons',
                                    disable=config.pbar_hide,
                                    leave=config.pbar_leave)]

        if not inplace:
            return x
        return None
    elif not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Unable to resample data of type "{type(x)}"')

    if not inplace:
        x = x.copy()

    # Collect some information for later
    nodes = x.nodes.set_index('node_id', inplace=False)
    locs = nodes[['x', 'y', 'z']]
    radii = nodes['radius'].to_dict()

    new_nodes: List = []
    max_tn_id = x.nodes.node_id.max() + 1

    errors = 0

    # Iterate over segments
    for i, seg in enumerate(x.small_segments):
        # Get coordinates
        coords = locs.loc[seg].values.astype(float)
        # Get radii
        rad = [radii[tn] for tn in seg]

        # Vecs between subsequently measured points
        vecs = np.diff(coords.T)

        # path: cum distance along points (norm from first to Nth point)
        dist = np.cumsum(np.linalg.norm(vecs, axis=0))
        dist = np.insert(dist, 0, 0)

        # If path is too short, just keep the first and last node
        if dist[-1] < resample_to or (method == 'cubic' and len(seg) <= 3):
            new_nodes += [[seg[0], seg[-1],
                           coords[0][0], coords[0][1], coords[0][2],
                           radii[seg[0]]]]
            continue

        # Distances (i.e. resolution) of interpolation
        n_nodes = int(dist[-1] / resample_to)
        new_dist = np.linspace(dist[0], dist[-1], n_nodes)

        try:
            sampleX = scipy.interpolate.interp1d(dist, coords[:, 0],
                                                 kind=method)
            sampleY = scipy.interpolate.interp1d(dist, coords[:, 1],
                                                 kind=method)
            sampleZ = scipy.interpolate.interp1d(dist, coords[:, 2],
                                                 kind=method)
            sampleR = scipy.interpolate.interp1d(dist, rad,
                                                 kind=method)
        except ValueError as e:
            if skip_errors:
                errors += 1
                new_nodes += x.nodes.loc[x.nodes.node_id.isin(seg[:-1]),
                                         ['node_id', 'parent_id',
                                          'x', 'y', 'z',
                                          'radius']].values.tolist()
                continue
            else:
                raise e

        # Sample each dim
        xnew = sampleX(new_dist)
        ynew = sampleY(new_dist)
        znew = sampleZ(new_dist)
        rnew = sampleR(new_dist)

        # Generate new coordinates
        new_coords = np.array([xnew, ynew, znew]).T

        # Generate new ids (start and end node IDs of this segment are kept)
        new_ids = seg[:1] + [max_tn_id +
                             i for i in range(len(new_coords) - 2)] + seg[-1:]

        # Increase max index
        max_tn_id += len(new_ids)

        # Keep track of new nodes
        new_nodes += [[tn, pn, co[0], co[1], co[2], r]
                      for tn, pn, co, r in zip(new_ids[:-1],
                                               new_ids[1:],
                                               new_coords,
                                               rnew)]

    if errors:
        logger.warning(f'{errors} ({errors/i:.0%}) segments skipped due to '
                       'errors')

    # Add root node(s)
    root = x.nodes.loc[x.nodes.node_id.isin(utils.make_iterable(x.root)),
                       ['node_id', 'parent_id', 'x', 'y', 'z', 'radius']]
    new_nodes += [list(r) for r in root.values]

    # Generate new nodes dataframe
    new_nodes = pd.DataFrame(data=new_nodes,
                             columns=['node_id', 'parent_id',
                                      'x', 'y', 'z', 'radius'])

    # Convert columns to appropriate dtypes
    dtypes = {k: x.nodes[k].dtype for k in ['node_id', 'parent_id', 'x', 'y', 'z', 'radius']}

    for cols in new_nodes.columns:
        new_nodes = new_nodes.astype(dtypes, errors='ignore')

    # Remove duplicate nodes (branch points)
    new_nodes = new_nodes[~new_nodes.node_id.duplicated()]

    # Generate KDTree
    tree = scipy.spatial.cKDTree(new_nodes[['x', 'y', 'z']].values)
    # Map soma onto new nodes if required
    # Note that if `._soma` is a soma detection function we can't tell
    # how to deal with it. Ideally the new soma node will
    # be automatically detected but it is possible, for example, that
    # the radii of nodes have changed due to interpolation such that more
    # than one soma is detected now. Also a "label" column in the node
    # table would be lost at this point.
    # We will go for the easy option which is to pin the soma at this point.
    if np.any(getattr(x, 'soma')):
        soma_nodes = utils.make_iterable(x.soma)
        old_pos = nodes.loc[soma_nodes, ['x', 'y', 'z']].values

        # Get nearest neighbours
        dist, ix = tree.query(old_pos)
        node_map = dict(zip(soma_nodes, new_nodes.node_id.values[ix]))

        # Map back onto neuron
        if utils.is_iterable(x.soma):
            # Use _soma to avoid checks - the new nodes have not yet been
            # assigned to the neuron!
            x._soma = [node_map[n] for n in x.soma]
        else:
            x._soma = node_map[x.soma]
    else:
        # If `._soma` was (read: is) a function but it didn't detect anything in
        # the original neurons, this makes sure that the resampled neuron
        # doesn't have a soma either:
        x.soma = None

    # Map connectors back if necessary
    if x.has_connectors:
        # Get position of old synapse-bearing nodes
        old_tn_position = nodes.loc[x.connectors.node_id, ['x', 'y', 'z']].values

        # Get nearest neighbours
        dist, ix = tree.query(old_tn_position)

        # Map back onto neuron
        x.connectors['node_id'] = new_nodes.node_id.values[ix]

    # Map tags back if necessary
    # Expects `tags` to be a dictionary {'tag': [node_id1, node_id2, ...]}
    if x.has_tags and isinstance(x.tags, dict):
        # Get nodes that need remapping
        nodes_to_remap = {n for l in x.tags.values() for n in l}

        # Get position of old tag-bearing nodes
        old_tn_position = nodes.loc[nodes_to_remap, ['x', 'y', 'z']].values

        # Get nearest neighbours
        dist, ix = tree.query(old_tn_position)

        # Map back onto tags
        node_map = dict(zip(nodes_to_remap, new_nodes.node_id.values[ix]))
        x.tags = {k: [node_map[n] for n in v] for k, v in x.tags.items()}

    # Set nodes
    x.nodes = new_nodes

    # Clear and regenerate temporary attributes
    x._clear_temp_attr()

    if not inplace:
        return x
    else:
        return None


def resample_along_axis(x: 'core.TreeNeuron',
                        interval: Union[int, float],
                        axis: int = 2,
                        old_nodes: Union[Literal['remove'],
                                         Literal['keep'],
                                         Literal['snap']] = 'remove',
                        inplace: bool = False
                        ) -> Optional['core.TreeNeuron']:
    """Resample neuron such that nodes lie exactly on given 1d grid.

    This function does not simply snap nodes to the closest grid line but
    instead adds new nodes where edges between existing nodes intersect
    with the planes defined by the grid.

    Parameters
    ----------
    x :             TreeNeuron | NeuronList
                    Neuron(s) to resample.
    interval :      float | int | None
                    Intervals defining a 1-dimensional grid along given axes
                    (see examples).
    axis :           0 | 1 | 2
                    Along which axes (x/y/z) to resample.
    old_nodes :     "remove" | "keep" | "snap"
                    Existing nodes are unlikely to intersect with the planes as
                    defined by the grid interval. There are three possible ways
                    to deal with them:
                     - "remove" (default) will simply drop old nodes: this
                       guarantees all remaining nodes will lie on a plane
                     - "keep" will keep old nodes without changing them
                     - "snap" will snap those nodes to the closest coordinate
                       on the grid without interpolation

    inplace :       bool
                    If False, will resample and return a copy of the original. If
                    True, will resample input neuron in place.

    Returns
    -------
    core.TreeNeuron
                Only if ``inplace=False``.

    See Also
    --------
    :func:`navis.resample_neuron`
                        Resample neuron such that edges between nodes have a
                        given length.
    :func:`navis.downsample_neuron`
                        This function reduces the number of nodes instead of
                        resample to certain resolution. Useful if you are
                        just after some simplification e.g. for speeding up
                        your calculations or you want to preserve node IDs.

    Examples
    --------
    Resample neuron such that we have one node in every 40nm slice along z axis

    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n.n_nodes
    4465
    >>> res = navis.resample_along_axis(n, interval=40,
    ...                                 axis=2, old_nodes='remove')
    >>> res.n_nodes < n.n_nodes
    True

    """
    utils.eval_param(axis, name='axis', allowed_values=(0, 1, 2))
    utils.eval_param(old_nodes, name='old_nodes',
                     allowed_values=("remove", "keep", "snap"))

    if not inplace:
        x = x.copy()

    if isinstance(x, core.NeuronList):
        res = [resample_along_axis(n, interval=interval,
                                   old_nodes=old_nodes,
                                   axis=axis, inplace=True)
               for n in config.tqdm(x,
                                    desc='Binning',
                                    disable=config.pbar_hide or len(x) == 1,
                                    leave=config.pbar_leave)]
        if not inplace:
            return core.NeuronList(res)

    # Collect coordinates of nodes and their parents
    nodes = x.nodes
    not_root = nodes.loc[nodes.parent_id >= 0]
    node_locs = not_root[['x', 'y', 'z']].values
    parent_locs = nodes.set_index('node_id').loc[not_root.parent_id.values,
                                                 ['x', 'y', 'z']].values

    # Get all vectors
    vecs = parent_locs - node_locs

    # Get coordinates along this axis
    loc1 = node_locs[:, axis]
    loc2 = parent_locs[:, axis]

    # This prevents runtime warnings e.g. from division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Find out which grid interval these are on
        int1 = (loc1 / interval).astype(int)
        int2 = (loc2 / interval).astype(int)

        # Difference in bin between both locs
        diff = int2 - int1
        sign = diff / np.abs(diff)

        # Figure out by how far we are from the gridline
        dist = np.zeros(diff.shape[0])
        dist[diff < 0] = loc1[diff < 0] % interval
        dist[diff > 0] = -loc1[diff > 0] % interval

        # Now we need to calculate the new position
        # Get other axes
        other_axes = list({0, 1, 2} - {axis})
        # Normalize other vectors by this vector
        other_vecs_norm = vecs[:, other_axes] / vecs[:, [axis]]

        # Get offset for other axis
        other_offset = other_vecs_norm * dist.reshape(dist.shape[0], 1)

        # Offset for this axis
        this_offset = dist * sign

    # Apply offsets
    new_coords = node_locs.copy()
    new_coords[:, other_axes] += other_offset * sign.reshape(sign.shape[0], 1)
    new_coords[:, [axis]] += this_offset.reshape(this_offset.shape[0], 1)

    # Now extract nodes that need to be inserted
    insert_between = not_root.loc[diff != 0, ['node_id', 'parent_id']].values
    new_coords = new_coords[diff != 0]

    # Insert nodes
    graph.insert_nodes(x, where=insert_between, coords=new_coords, inplace=True)

    # Figure out what to do with nodes that are not on the grid
    if old_nodes == 'remove':
        mod = x.nodes[['x', 'y', 'z'][axis]].values % interval
        not_lined_up = mod != 0
        to_remove = x.nodes.loc[not_lined_up, 'node_id'].values
    elif old_nodes == 'keep':
        to_remove = insert_between[:, 0]
    elif old_nodes == 'snap':
        not_lined_up = x.nodes[['x', 'y', 'z']].values[:, axis] % interval != 0
        to_snap = x.nodes.loc[not_lined_up, ['x', 'y', 'z'][axis]].values
        snapped = (to_snap / interval).round() * interval
        x.nodes.loc[not_lined_up, ['x', 'y', 'z'][axis]] = snapped
        to_remove = []

    if np.any(to_remove):
        graph.remove_nodes(x, which=to_remove, inplace=True)

    if not inplace:
        return x


def _make_grid(interval, axis, neuron):
    """Generate Volume visualizing 1d grid."""
    assert axis in (0, 1, 2)
    bounds = neuron.bbox

    # Generate a box for each plane - just a face won't render properly
    b = tm.primitives.Box()
    box_verts = np.array(b.vertices)
    box_faces = np.array(b.faces)
    for i in range(3):
        is_low = box_verts[:, i] < 0
        box_verts[is_low, i] = bounds[i][0]
        box_verts[~is_low, i] = bounds[i][1]

    is_low = b.vertices[:, axis] < 0

    start = (bounds[axis][0] / interval).astype(int) * interval
    end = ((bounds[axis][1] / interval).astype(int) + 1) * interval
    depth = np.arange(start, end + interval, interval)

    faces = []
    vertices = []
    for i, d in enumerate(depth):
        this_verts = box_verts.copy()
        this_faces = box_faces.copy()

        this_verts[is_low, axis] = d - 0.01 * interval
        this_verts[~is_low, axis] = d + 0.01 * interval

        this_faces += this_verts.shape[0] * i

        vertices.append(this_verts)
        faces.append(this_faces)
    faces = np.vstack(faces)
    vertices = np.vstack(vertices)

    return core.Volume(vertices=vertices, faces=faces, color=(1, 1, 1, .1))
