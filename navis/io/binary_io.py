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

import io
import os
import struct

import pandas as pd

from typing import Union, Optional

from .. import config, utils, core


def write_google_binary(x: Union['core.NeuronList', 'core.TreeNeuron'],
                        filename: Optional[str] = None) -> None:
    """Export TreeNeuron (skeletons) to Google's binary format.

    Follows the format specified
    `here <https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/skeletons.md#encoded-skeleton-file-format>`_.

    Parameters
    ----------
    x :                 TreeNeuron | NeuronList
                        If multiple neurons, will generate a file
                        for each neuron (see also ``filename``).
    filename :          None | str | list, optional
                        If ``None``, will return byte string or list of
                        thereof. If filepath will save to this file. If path
                        will save neuron(s) in that path using ``'{x.id}.bin'``
                        as filename(s). If list, input must be NeuronList and
                        a filepath must be provided for each neuron.

    Returns
    -------
    Nothing

    See Also
    --------
    :func:`navis.from_google_binary`
                        Import Google binary format.

    """
    if isinstance(x, core.NeuronList):
        if x.is_mixed:
            raise TypeError('NeuronList must only contain TreeNeurons.')
        res = []

        if not utils.is_iterable(filename):
            filename = [filename] * len(x)

        if len(filename) != len(x):
            raise ValueError('Must provide a filepath for every neuron')

        for n, f in config.tqdm(zip(x, filename),
                                desc='Exporting',
                                leave=config.pbar_leave,
                                total=len(x),
                                disable=config.pbar_hide):
            res.append(write_google_binary(n, filename=f))
        return res

    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Expected TreeNeuron, got "{type(x)}"')

    # Below code modified from:
    # https://github.com/google/neuroglancer/blob/master/python/neuroglancer/skeleton.py#L34
    result = io.BytesIO()
    vertex_positions = x.nodes[['x', 'y', 'z']].values
    # Map edges node IDs to node indices
    node_ix = pd.Series(x.nodes.reset_index(drop=True).index, index=x.nodes.node_id)
    edges = x.edges.copy()
    edges[:, 0] = node_ix.loc[edges[:, 0]].values
    edges[:, 1] = node_ix.loc[edges[:, 1]].values

    result.write(struct.pack('<II', vertex_positions.shape[0], edges.shape[0] // 2))
    result.write(vertex_positions.tobytes())
    result.write(edges.tobytes())

    if filename and os.path.isdir(filename):
        filename = os.path.join(filename, f'{x.id}.bin')

    if filename:
        with open(filename, 'wb') as f:
            f.write(result.getvalue())
    else:
        return result.getvalue()
