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

import io
import json
import os
import struct
import tempfile
import requests

import numpy as np
import pandas as pd

from pathlib import Path
from functools import lru_cache
from typing import Union, Dict, Optional, Any, IO, List
from typing_extensions import Literal
from zipfile import ZipFile, ZipInfo

from .. import config, utils, core
from . import base

try:
    import zlib
    import zipfile
    compression = zipfile.ZIP_DEFLATED
except ImportError:
    compression = zipfile.ZIP_STORED


DEFAULT_FMT = "{name}"


class PrecomputedReader(base.BaseReader):
    def is_valid_file(self, file):
        """Return True if file should be considered for reading."""
        if isinstance(file, zipfile.ZipInfo):
            file = str(file.filename)
        elif isinstance(file, Path):
            if not file.is_file():
                return False
            file = str(file.name)
        else:
            file = str(file)

        # Drop anything with a file extension or hidden files (e.g. ".DS_store")
        if '.' in file:
            return False
        # Ignore the info file
        if file == 'info':
            return False
        # Ignore manifests
        if file.endswith(':0'):
            return False
        return True


class PrecomputedMeshReader(PrecomputedReader):
    def __init__(
        self,
        fmt: str = DEFAULT_FMT,
        attrs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(fmt=fmt,
                         attrs=attrs,
                         file_ext='',
                         name_fallback='mesh',
                         read_binary=True)

    def read_buffer(
        self, f: IO, attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.MeshNeuron':
        """Read buffer into a MeshNeuron.

        Parameters
        ----------
        f :         IO
                    Readable buffer - must be bytes.
        attrs :     dict | None
                    Arbitrary attributes to include in the MeshNeuron.

        Returns
        -------
        core.MeshNeuron
        """
        if not isinstance(f.read(0), bytes):
            raise ValueError(f'Expected bytes, got {type(f.read(0))}')

        num_vertices = np.frombuffer(f.read(4), np.uint32)[0]
        vertices = np.frombuffer(f.read(int(3 * 4 * num_vertices)),
                                 np.float32).reshape(-1, 3)
        faces = np.frombuffer(f.read(),
                              np.uint32).reshape(-1, 3)

        return core.MeshNeuron({'vertices': vertices, 'faces': faces},
                               **(self._make_attributes({'name': self.name_fallback,
                                                         'origin': 'DataFrame'}, attrs)))


class PrecomputedSkeletonReader(PrecomputedReader):
    def __init__(
        self,
        fmt: str = DEFAULT_FMT,
        attrs: Optional[Dict[str, Any]] = None,
        info: Dict[str, Any] = {}
    ):
        super().__init__(fmt=fmt,
                         attrs=attrs,
                         file_ext='',
                         name_fallback='skeleton',
                         read_binary=True)
        self.info = info

    def read_buffer(
        self,
        f: IO, attrs: Optional[Dict[str, Any]] = None
    ) -> 'core.TreeNeuron':
        """Read buffer into a TreeNeuron.

        Parameters
        ----------
        f :         IO
                    Readable buffer - must be bytes.
        attrs :     dict | None
                    Arbitrary attributes to include in the TreeNeuron.

        Returns
        -------
        core.TreeNeuron

        """
        if not isinstance(f.read(0), bytes):
            raise ValueError(f'Expected bytes, got {type(f.read(0))}')

        num_nodes = np.frombuffer(f.read(4), np.uint32)[0]
        num_edges = np.frombuffer(f.read(4), np.uint32)[0]
        nodes = np.frombuffer(f.read(int(3 * 4 * num_nodes)),
                              np.float32).reshape(-1, 3)
        edges = np.frombuffer(f.read(int(2 * 4 * num_edges)),
                              np.uint32).reshape(-1, 2)

        swc = self.make_swc(nodes, edges)

        # Check for malformed vertex attributes (should be list of dicts)
        if isinstance(self.info.get('vertex_attributes', None), dict):
            self.info['vertex_attributes'] = [self.info['vertex_attributes']]

        # Parse additional vertex attributes if specified as per the info file
        for attr in self.info.get('vertex_attributes', []):
            dtype = np.dtype(attr['data_type'])
            n_comp = attr['num_components']
            values = np.frombuffer(f.read(int(n_comp * dtype.itemsize * num_nodes)),
                                   dtype).reshape(-1, n_comp)
            if n_comp == 1:
                swc[attr['id']] = values.flatten()
            else:
                for i in range(n_comp):
                    swc[f"{attr['id']}_{i}"] = values[:, i]

        return core.TreeNeuron(swc,
                               **(self._make_attributes({'name': self.name_fallback,
                                                         'origin': 'DataFrame'}, attrs)))

    def make_swc(
        self, nodes: np.ndarray, edges: np.ndarray
    ) -> pd.DataFrame:
        """Make SWC table from nodes and edges.

        Parameters
        ----------
        nodes :     (N, 3) array
        edges :     (N, 2) array

        Returns
        -------
        pandas.DataFrame
        """
        swc = pd.DataFrame()
        swc['node_id'] = np.arange(len(nodes))
        swc['x'], swc['y'], swc['z'] = nodes[:, 0], nodes[:, 1], nodes[:, 2]

        edge_dict = dict(zip(edges[:, 1], edges[:, 0]))
        swc['parent_id'] = swc.node_id.map(lambda x: edge_dict.get(x, -1)).astype(np.int32)

        return swc


def read_precomputed(f: Union[str, io.BytesIO],
                     datatype: Union[Literal['auto'],
                                     Literal['mesh'],
                                     Literal['skeleton']] = 'auto',
                     include_subdirs: bool = False,
                     fmt: str = '{id}',
                     info: Union[bool, str, dict] = True,
                     limit: Optional[int] = None,
                     parallel: Union[bool, int] = 'auto',
                     **kwargs) -> 'core.NeuronObject':
    """Read skeletons and meshes from neuroglancer's precomputed format.

    Follows the formats specified
    `here <https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed>`_.

    Parameters
    ----------
    f :                 filepath | folder | zip file | bytes
                        Filename, folder or bytes. If folder, will import all
                        files. If a ``.zip``, ``.tar`` or ``.tar.gz`` file will
                        read all files in the archive. See also ``limit`` parameter.
    datatype :          "auto" | "skeleton" | "mesh"
                        Which data type we expect to read from the files. If
                        "auto", we require a "info" file in the same directory
                        as ``f``.
    include_subdirs :   bool, optional
                        If True and ``f`` is a folder, will also search
                        subdirectories for binary files.
    fmt :               str
                        Formatter to specify what files to look for (when `f` is
                        directory) and how they are parsed into neuron
                        attributes. Some illustrative examples:
                          - ``{name}`` (default) uses the filename
                            (minus the suffix) as the neuron's name property
                          - ``{id}`` (default) uses the filename as the neuron's ID
                            property
                          - ``{name,id}`` uses the filename as the neuron's
                            name and ID properties
                          - ``{name}.{id}`` splits the filename at a "."
                            and uses the first part as name and the second as ID
                          - ``{name,id:int}`` same as above but converts
                            into integer for the ID
                          - ``{name}_{myproperty}`` splits the filename at
                            "_" and uses the first part as name and as a
                            generic "myproperty" property
                          - ``{name}_{}_{id}`` splits the filename at
                            "_" and uses the first part as name and the last as
                            ID. The middle part is ignored.

                        Throws a ValueError if pattern can't be found in
                        filename. Ignored for DataFrames.
    info :              bool | str | dict
                        An info file describing the data:
                          - ``True`` = will look for `info` file in base folder
                          - ``False`` = do not use/look for `info` file
                          - ``str`` = filepath to `info` file
                          - ``dict`` = already parsed info file
    limit :             int, optional
                        If reading from a folder you can use this parameter to
                        read only the first ``limit`` files. Useful if
                        wanting to get a sample from a large library of
                        skeletons/meshes.
    parallel :          "auto" | bool | int
                        Defaults to ``auto`` which means only use parallel
                        processing if more than 200 files are imported. Spawning
                        and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the
                        number of cores (otherwise defaults to
                        ``os.cpu_count() // 2``).
    **kwargs
                        Keyword arguments passed to the construction of the
                        neurons. You can use this to e.g. set meta data such
                        as ``units``.

    Returns
    -------
    navis.MeshNeuron
    navis.NeuronList

    See Also
    --------
    :func:`navis.write_precomputed`
                        Export neurons/volumes to precomputed format.

    """
    utils.eval_param(datatype, name='datatype', allowed_values=('skeleton',
                                                                'mesh',
                                                                'auto'))

    # See if we can get the info file from somewhere
    if info is True and not isinstance(f, bytes):
        # Find info in zip archive
        if str(f).endswith('.zip'):
            with ZipFile(Path(f).expanduser(), 'r') as zip:
                if 'info' in [f.filename for f in zip.filelist]:
                    info = json.loads(zip.read('info').decode())
                elif datatype == 'auto':
                    raise ValueError('No `info` file found in zip file. Please '
                                     'specify data type using the `datatype` '
                                     'parameter.')
        # Try loading info from URL
        elif utils.is_url(str(f)):
            base_url = '/'.join(str(f).split('/')[:-1])
            info = _fetch_info_file(base_url, raise_missing=False)
        # Try loading info from parent path
        else:
            fp = Path(str(f))
            # Find first existing root
            while not fp.is_dir():
                fp = fp.parent
            fp = fp / 'info'
            if fp.is_file():
                with open(fp, 'r') as info_file:
                    info = json.load(info_file)

    # At this point we should have a dictionary - even if it's empty
    if not isinstance(info, dict):
        info = {}

    # Parse data type from info file (if required)
    if datatype == 'auto':
        if '@type' not in info:
            raise ValueError('Either no `info` file found or it does not specify '
                             'a data type. Please provide data type using the '
                             '`datatype` parameter.')

        if info.get('@type', None) == 'neuroglancer_legacy_mesh':
            datatype = 'mesh'
        elif info.get('@type', None) == 'neuroglancer_skeletons':
            datatype = 'skeleton'
        else:
            raise ValueError('Data type specified in `info` file unknown: '
                             f'{info.get("@type", None)}. Please provide data '
                             'type using the `datatype` parameter.')

    if isinstance(f, bytes):
        f = io.BytesIO(f)

    if datatype == 'skeleton':
        if not isinstance(info, dict):
            info = {}
        reader = PrecomputedSkeletonReader(fmt=fmt, attrs=kwargs, info=info)
    else:
        reader = PrecomputedMeshReader(fmt=fmt, attrs=kwargs)

    return reader.read_any(f, include_subdirs, parallel, limit=limit)


class PrecomputedWriter(base.Writer):
    """Writer class that also takes care of `info` files."""

    def write_any(self, x, filepath, write_info=True, **kwargs):
        """Write any to file. Default entry point."""
        # First write the actual neurons
        kwargs['write_info'] = False
        super().write_any(x, filepath=filepath, **kwargs)

        # Write info file to the correct directory/zipfile
        if write_info:
            add_props = {}
            if kwargs.get('radius', False):
                add_props['vertex_attributes'] = [{'id': 'radius',
                                                  'data_type': 'float32',
                                                  'num_components': 1}]

            if str(self.path).endswith('.zip'):
                with ZipFile(self.path, mode='a') as zf:
                    # Context-manager will remove temporary directory and its contents
                    with tempfile.TemporaryDirectory() as tempdir:
                        # Write info to zip
                        if write_info:
                            # Generate temporary filename
                            f = os.path.join(tempdir, 'info')
                            write_info_file(x, f, add_props=add_props)
                            # Add file to zip
                            zf.write(f, arcname='info', compress_type=compression)
            else:
                fp = self.path
                # Find the first existing root directory
                while not fp.is_dir():
                    fp = fp.parent

                write_info_file(x, fp, add_props=add_props)


def write_precomputed(x: Union['core.NeuronList', 'core.TreeNeuron', 'core.MeshNeuron', 'core.Volume'],
                      filepath: Optional[str] = None,
                      write_info: bool = True,
                      write_manifest: bool = False,
                      radius: bool = False) -> None:
    """Export skeletons or meshes to neuroglancer's (legacy) precomputed format.

    Note that you should not mix meshes and skeletons in the same folder!

    Follows the formats specified
    `here <https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed>`_.

    Parameters
    ----------
    x :                 TreeNeuron | MeshNeuron | Volume | Trimesh | NeuronList
                        If multiple neurons, will generate a file for each
                        neuron (see also ``filepath``). For use in neuroglancer
                        coordinates should generally be in nanometers.
    filepath :          None | str | list, optional
                        If ``None``, will return byte string or list of
                        thereof. If filepath will save to this file. If path
                        will save neuron(s) in that path using ``{x.id}``
                        as filename(s). If list, input must be NeuronList and
                        a filepath must be provided for each neuron.
    write_info :        bool
                        Whether to also write a JSON-formatted ``info`` file that
                        can be parsed by e.g. neuroglancer. This only works if
                        inputs are either only skeletons or only meshes!
    write_manifest :    bool
                        For meshes only: whether to also write manifests. For
                        each mesh we will create a JSON-encoded ``{id}:0`` file
                        that contains a "fragments" entry that maps to the
                        actual filename. Note that this will not work on Windows
                        because colons aren't allowed in file names and on OSX
                        the colon will show up as a ``/`` in the Finder.
    radius :            bool
                        For TreeNeurons only: whether to write radius as
                        additional vertex property.

    Returns
    -------
    None
                        If filepath is not ``None``.
    bytes
                        If filepath is ``None``.

    See Also
    --------
    :func:`navis.read_precomputed`
                        Import neurons from neuroglancer's precomputed format.
    :func:`navis.write_mesh`
                        Write meshes to generic mesh formats (obj, stl, etc).

    Examples
    --------

    Write skeletons:

    >>> import navis
    >>> n = navis.example_neurons(3, kind='skeleton')
    >>> navis.write_precomputed(n, tmp_dir)

    Write meshes:

    >>> import navis
    >>> n = navis.example_neurons(3, kind='mesh')
    >>> navis.write_precomputed(n, tmp_dir)

    Write directly to zip archive:

    >>> import navis
    >>> n = navis.example_neurons(3, kind='skeleton')
    >>> navis.write_precomputed(n, tmp_dir / 'precomputed.zip')

    """
    writer = PrecomputedWriter(_write_precomputed, ext=None)

    return writer.write_any(x,
                            filepath=filepath,
                            write_info=write_info,
                            write_manifest=write_manifest,
                            radius=radius)


def _write_precomputed(x: Union['core.TreeNeuron', 'core.MeshNeuron', 'core.Volume'],
                       filepath: Optional[str] = None,
                       write_info: bool = True,
                       write_manifest: bool = False,
                       radius: bool = False) -> None:
    """Write single neuron to neuroglancer's precomputed format."""
    if filepath and os.path.isdir(filepath):
        if isinstance(x, core.BaseNeuron):
            if not x.id:
                raise ValueError('Neuron(s) must have an ID when destination '
                                 'is a folder')
            filepath = os.path.join(filepath, f'{x.id}')
        elif isinstance(x, core.Volume):
            filepath = os.path.join(filepath, f'{x.name}')
        else:
            raise ValueError(f'Unable to generate filename for {type(x)}')

    if isinstance(x, core.TreeNeuron):
        return _write_skeleton(x, filepath, radius=radius)
    elif utils.is_mesh(x):
        return _write_mesh(x.vertices, x.faces, filepath,
                           write_manifest=write_manifest)
    else:
        raise TypeError(f'Unable to write data of type "{type(x)}"')


def write_info_file(data, filepath, add_props={}):
    """Write neuroglancer 'info' file for given neurons.

    Parameters
    ----------
    data :         navis.NeuronList | navis.Volumes | trimesh
    filepath :     str | Path
                   Path to write the file to.
    add_props :    dict
                   Additional properties to write to the file.

    """
    info = {}
    if utils.is_iterable(data):
        types = list(set([type(d) for d in data]))
        if len(types) > 1:
            raise ValueError('Unable to write info file for mixed data: '
                             f'{data.types}')
        data = data[0]

    if utils.is_mesh(data):
        info['@type'] = 'neuroglancer_legacy_mesh'
    elif isinstance(data, core.TreeNeuron):
        info['@type'] = 'neuroglancer_skeletons'

        # If we know the units add transform from "stored model"
        # to "model space" which is supposed to be nm
        if not data.units.dimensionless:
            u = data.units.to('1 nm').magnitude
        else:
            u = 1
        tr = np.zeros((4, 3), dtype=int)
        tr[:3, :3] = np.diag([u, u, u])
        info['transform'] = tr.T.flatten().tolist()

    else:
        raise TypeError(f'Unable to write info file for data of type "{type(data)}"')

    info.update(add_props)
    if not str(filepath).endswith('/info'):
        filepath = os.path.join(filepath, 'info')
    with open(filepath, 'w') as f:
        json.dump(info, f)


def _write_mesh(vertices, faces, filename, write_manifest=False):
    """Write mesh to precomputed binary format."""
    # Make sure we are working with the correct data types
    vertices = np.asarray(vertices, dtype='float32')
    faces = np.asarray(faces, dtype='uint32')
    n_vertices = np.uint32(vertices.shape[0])
    vertex_index_format = [n_vertices, vertices, faces]

    results = b''.join([array.tobytes('C') for array in vertex_index_format])

    if filename:
        filename = Path(filename)
        with open(filename, 'wb') as f:
            f.write(results)

        if write_manifest:
            with open(filename.parent / f'{filename.name}:0', 'w') as f:
                json.dump({'fragments': [filename.name]}, f)
    else:
        return results


def _write_skeleton(x, filename, radius=False):
    """Write skeleton to neuroglancers binary format."""
    # Below code modified from:
    # https://github.com/google/neuroglancer/blob/master/python/neuroglancer/skeleton.py#L34
    result = io.BytesIO()
    vertex_positions = x.nodes[['x', 'y', 'z']].values.astype('float32', order='C')
    # Map edges node IDs to node indices
    node_ix = pd.Series(x.nodes.reset_index(drop=True).index, index=x.nodes.node_id)
    edges = x.edges.copy().astype('uint32', order='C')
    edges[:, 0] = node_ix.loc[edges[:, 0]].values
    edges[:, 1] = node_ix.loc[edges[:, 1]].values
    edges = edges[:, [1, 0]]  # For some reason we have to switch direction

    result.write(struct.pack('<II', vertex_positions.shape[0], edges.shape[0]))
    result.write(vertex_positions.tobytes())
    result.write(edges.tobytes())

    if radius and 'radius' in x.nodes.columns:
        if any(pd.isnull(x.nodes['radius'])):
            raise ValueError('Unable to write radii with missing values.')
        result.write(x.nodes.radius.values.astype('float32').tobytes())

    if filename:
        with open(filename, 'wb') as f:
            f.write(result.getvalue())
    else:
        return result.getvalue()


@lru_cache
def _fetch_info_file(base_url, raise_missing=True):
    """Try and fetch `info` file for given base url."""
    if not base_url.endswith('/'):
        base_url += '/'
    r = requests.get(f'{base_url}info')

    try:
        r.raise_for_status()
    except requests.HTTPError:
        if raise_missing:
            raise
        else:
            return {}
    except BaseException:
        raise

    return r.json()