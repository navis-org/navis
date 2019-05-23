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

import pandas as pd
import numpy as np
import scipy.spatial
import scipy.interpolate

from .. import core, config

# Set up logging
logger = config.logger

__all__ = ['resample_neuron']


def resample_neuron(x, resample_to, method='linear', inplace=False,
                    skip_errors=True):
    """ Resamples neuron(s) to given NM resolution.

    Preserves root, leafs, branchpoints. Tags and connectors are mapped onto
    the closest new treenode. Columns "confidence" and "creator" of the
    treenode table are discarded.

    Important
    ---------
    This generates an entirely new set of treenode IDs! Those will be unique
    within a neuron, but you may encounter duplicates across neurons.

    Also: be aware that high-resolution neurons will use A LOT of memory.

    Parameters
    ----------
    x :                 CatmaidNeuron | CatmaidNeuronList
                        Neuron(s) to resample.
    resample_to :       int
                        New resolution in NANOMETERS.
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
    CatmaidNeuron/List
                        Downsampled neuron(s). Only if ``inplace=False``.

    See Also
    --------
    :func:`navis.downsample_neuron`
                        This function reduces the number of nodes instead of
                        resample to certain resolution. Useful if you are
                        just after some simplification e.g. for speeding up
                        your calculations or you want to preserve treenode IDs.
    """

    if isinstance(x, core.NeuronList):
        results = [resample_neuron(x[i], resample_to,
                                   method=method, inplace=inplace,
                                   skip_errors=skip_errors)
                   for i in config.trange(x.shape[0],
                                          desc='Resampl. neurons',
                                          disable=config.pbar_hide,
                                          leave=config.pbar_leave)]
        if not inplace:
            return core.NeuronList(results)
        return
    elif not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Unable to resample data of type "{type(x)}"')

    if not inplace:
        x = x.copy()

    # Collect some information for later
    nodes = x.nodes.set_index('node_id')
    locs = nodes[['x', 'y', 'z']]
    radii = nodes['radius'].to_dict()

    new_nodes = []
    max_tn_id = x.nodes.node_id.max() + 1

    errors = 0

    # Iterate over segments
    for i, seg in enumerate(config.tqdm(x.small_segments,
                                        desc='Proc. segments',
                                        disable=config.pbar_hide,
                                        leave=False)):
        # Get coordinates
        coords = locs.loc[seg].values.astype(float)
        # Get radii
        rad = [radii[tn] for tn in seg]

        # Vecs between subsequently measured points
        vecs = np.diff(coords.T)

        # path: cum distance along points (norm from first to ith point)
        path = np.cumsum(np.linalg.norm(vecs, axis=0))
        path = np.insert(path, 0, 0)

        # If path is too short, just keep the first and last treenode
        if path[-1] < resample_to or (method == 'cubic' and len(seg) <= 3):
            new_nodes += [[seg[0], seg[-1],
                           coords[0][0], coords[0][1], coords[0][2],
                           radii[seg[0]]]]
            continue

        # Coords of interpolation
        n_nodes = int(path[-1] / resample_to)
        interp_coords = np.linspace(path[0], path[-1], n_nodes)

        try:
            sampleX = scipy.interpolate.interp1d(path, coords[:, 0],
                                                 kind=method)
            sampleY = scipy.interpolate.interp1d(path, coords[:, 1],
                                                 kind=method)
            sampleZ = scipy.interpolate.interp1d(path, coords[:, 2],
                                                 kind=method)
            sampleR = scipy.interpolate.interp1d(path, rad,
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
        xnew = sampleX(interp_coords)
        ynew = sampleY(interp_coords)
        znew = sampleZ(interp_coords)
        rnew = sampleR(interp_coords)

        # Generate new coordinates
        new_coords = np.array([xnew, ynew, znew]).T

        # Generate new ids (start and end node IDs of this segment)
        new_ids = seg[:1] + [max_tn_id +
                             i for i in range(len(new_coords) - 2)] + seg[-1:]

        # Keep track of new nodes
        new_nodes += [[tn, pn, co[0], co[1], co[2], r]
                      for tn, pn, co, r in zip(new_ids[:-1],
                                               new_ids[1:],
                                               new_coords,
                                               rnew)]

        # Increase max index
        max_tn_id += len(new_ids)

    if errors:
        logger.warning(f'{errors} ({errors/i:.0%}) segments skipped due to '
                       'errors')

    # Add root node(s)
    root = x.root
    if not isinstance(root, (np.ndarray, list)):
        root = [x.root]
    root = x.nodes.loc[x.nodes.node_id.isin(root), ['node_id',
                                                    'parent_id',
                                                    'x', 'y', 'z',
                                                    'radius']]
    new_nodes += [list(r) for r in root.values]

    # Generate new nodes dataframe
    new_nodes = pd.DataFrame(data=new_nodes,
                             columns=['node_id', 'parent_id',
                                      'x', 'y', 'z', 'radius'],
                             dtype=object
                             )

    # Convert columns to appropriate dtypes
    dtypes = {'node_id': int, 'parent_id': int, 'x': float, 'y': float,
              'z': float, 'radius': float}

    for k, v in dtypes.items():
        new_nodes[k] = new_nodes[k].astype(v)

    # Remove duplicate treenodes (branch points)
    new_nodes = new_nodes[~new_nodes.node_id.duplicated()]

    if x.has_connectors:
        # Map connectors back:
        # 1. Get position of old synapse-bearing treenodes
        old_tn_position = x.nodes.set_index('node_id').loc[x.connectors.node_id,
                                                           ['x', 'y', 'z']].values
        # 2. Get closest neighbours
        distances = scipy.spatial.distance.cdist(
            old_tn_position, new_nodes[['x', 'y', 'z']].values)
        min_ix = np.argmin(distances, axis=1)
        # 3. Map back onto neuron
        x.connectors['node_id'] = new_nodes.iloc[min_ix].node_id.values

    # Set nodes
    x.nodes = new_nodes

    # Clear and regenerate temporary attributes
    x._clear_temp_attr()

    if not inplace:
        return x
