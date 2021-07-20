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

import numpy as np
import pandas as pd

from typing import Union, Optional

from .. import config, utils, core


def read_precomputed_mesh(f: Union[str, io.BytesIO],
                          include_subdirs: bool = False,
                          fmt: str = '{name}.bin',
                          **kwargs) -> 'core.NeuronObject':
    """Create Neuron/List from SWC file.

    This import is following format specified
    `here <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_

    Parameters
    ----------
    f :                 str | pandas.DataFrame | iterable
                        SWC string, URL, filename, folder or DataFrame.
                        If folder, will import all ``.swc`` files. If a
                        ``.zip`` file will read all SWC files in the file.
    include_subdirs :   bool, optional
                        If True and ``f`` is a folder, will also search
                        subdirectories for ``.swc`` files.
    fmt :               str
                        Formatter to specify what files to look for (when `f` is
                        directory) and how they are parsed into neuron
                        attributes. Some illustrative examples:

                          - ``{name}.bin`` (default) uses the filename
                            (minus the suffix) as the neuron's name property
                          - ``{name}.gbin`` looks for a different file extension
                          - ``{id}.bin`` uses the filename as the neuron's ID
                            property
                          - ``{name,id}.bin`` uses the filename as the neuron's
                            name and ID properties
                          - ``{name}.{id}.gbin`` splits the filename at a "."
                            and uses the first part as name and the second as ID
                          - ``{name,id:int}.bin`` same as above but converts
                            into integer for the ID
                          - ``{name}_{myproperty}.bin`` splits the filename at
                            "_" and uses the first part as name and as a
                            generic "myproperty" property
                          - ``{name}_{}_{id}.bin`` splits the filename at
                            "_" and uses the first part as name and the last as
                            ID. The middle part is ignored.

                        Throws a ValueError if pattern can't be found in
                        filename. Ignored for DataFrames.
    **kwargs
                        Keyword arguments passed to the construction of
                        ``navis.Tree/MeshNeuron``. You can use this to e.g. set
                        meta data.

    Returns
    -------
    navis.TreeNeuron
    navis.MeshNeuron
    navis.NeuronList

    See Also
    --------
    :func:`navis.write_precomputed`
                        Export neurons/volumes in precomputed format.

    """
    pass


def write_precomputed(x: Union['core.NeuronList', 'core.TreeNeuron', 'core.MeshNeuron', 'core.Volume'],
                      filename: Optional[str] = None) -> None:
    """Export skeletons or meshes to Google's (legacy) precomputed format.

    Follows the formats specified
    `here <https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed>`_.

    Parameters
    ----------
    x :                 TreeNeuron | MeshNeuron | Volume NeuronList
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
            res.append(write_precomputed(n, filename=f))
        return res

    if filename and os.path.isdir(filename):
        if isinstance(x, core.BaseNeuron):
            filename = os.path.join(filename, f'{x.id}.bin')
        elif isinstance(x, core.Volume):
            filename = os.path.join(filename, f'{x.name}.bin')
        else:
            raise ValueError(f'Unable to generate filename for {type(x)}')

    if isinstance(x, core.TreeNeuron):
        return _write_skeleton(x, filename)
    elif utils.is_mesh(x):
        return _write_mesh(x.vertices, x.faces, filename)
    else:
        raise TypeError(f'Unable to write data of type "{type(x)}"')


def _write_mesh(vertices, faces, filename):
    """Write mesh to Google binary format."""
    # Make sure we are working with the correct data types
    vertices = np.asarray(vertices, dtype='float32')
    faces = np.asarray(faces, dtype='uint32')
    n_vertices = np.uint32(vertices.shape[0])
    vertex_index_format = [n_vertices, vertices, faces]

    results = b''.join([array.tobytes('C') for array in vertex_index_format])

    if filename and os.path.isdir(filename):
        filename = os.path.join(filename, filename)

    if filename:
        with open(filename, 'wb') as f:
            f.write(results)
    else:
        return results


def _write_skeleton(x, filename):
    """Write skeleton to Google binary format."""
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

    if filename:
        with open(filename, 'wb') as f:
            f.write(result.getvalue())
    else:
        return result.getvalue()
