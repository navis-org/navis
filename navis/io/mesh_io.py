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
import trimesh as tm

from pathlib import Path
from typing import Union, Iterable, Optional, Dict, Any
from typing_extensions import Literal

from .. import config, utils, core
from . import base

# Set up logging
logger = config.get_logger(__name__)


def read_mesh(f: Union[str, Iterable],
              include_subdirs: bool = False,
              parallel: Union[bool, int] = 'auto',
              output: Union[Literal['neuron'],
                            Literal['volume'],
                            Literal['trimesh']] = 'neuron',
              errors: Union[Literal['raise'],
                            Literal['log'],
                            Literal['ignore']] = 'log',
              limit: Optional[int] = None,
              **kwargs) -> 'core.NeuronObject':
    """Create Neuron/List from mesh.

    This is a thin wrapper around `trimesh.load_mesh` which supports most
    common formats (obj, ply, stl, etc.).

    Parameters
    ----------
    f :                 str | iterable
                        Filename(s) or folder. If folder must include file
                        extension (e.g. `my/dir/*.ply`).
    include_subdirs :   bool, optional
                        If True and ``f`` is a folder, will also search
                        subdirectories for meshes.
    parallel :          "auto" | bool | int,
                        Defaults to ``auto`` which means only use parallel
                        processing if more than 100 mesh files are imported.
                        Spawning and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the number of
                        cores (otherwise defaults to ``os.cpu_count() - 2``).
    output :            "neuron" | "volume" | "trimesh"
                        Determines function's output. See Returns.
    errors :            "raise" | "log" | "ignore"
                        If "log" or "ignore", errors will not be raised.
    limit :             int, optional
                        If reading from a folder you can use this parameter to
                        read only the first ``limit`` files. Useful when
                        wanting to get a sample from a large library of
                        meshes.
    **kwargs
                        Keyword arguments passed to :class:`navis.MeshNeuron`
                        or :class:`navis.Volume`. You can use this to e.g.
                        set the units on the neurons.

    Returns
    -------
    navis.MeshNeuron
                        If ``output="neuron"`` (default).
    navis.Volume
                        If ``output="volume"``.
    trimesh.Trimesh
                        If ``output='trimesh'``.
    navis.NeuronList
                        If ``output="neuron"`` and import has multiple meshes
                        will return NeuronList of MeshNeurons.
    list
                        If ``output!="neuron"`` and import has multiple meshes
                        will return list of Volumes or Trimesh.

    Examples
    --------

    Read a single file into :class:`navis.MeshNeuron`:

    >>> m = navis.read_mesh('mesh.obj')                         # doctest: +SKIP

    Read all e.g. .obj files in a directory:

    >>> nl = navis.read_mesh('/some/directory/*.obj')           # doctest: +SKIP

    Sample first 50 files in folder:

    >>> nl = navis.read_mesh('/some/directory/*.obj', limit=50) # doctest: +SKIP

    Read single file into :class:`navis.Volume`:

    >>> nl = navis.read_mesh('mesh.obj', output='volume')       # doctest: +SKIP

    """
    utils.eval_param(output, name='output',
                     allowed_values=('neuron', 'volume', 'trimesh'))

    # If is directory, compile list of filenames
    if isinstance(f, str) and '*' in f:
        f, ext = f.split('*')
        f = Path(f).expanduser()

        if not f.is_dir():
            raise ValueError(f'{f} does not appear to exist')

        if not include_subdirs:
            f = list(f.glob(f'*{ext}'))
        else:
            f = list(f.rglob(f'*{ext}'))

        if limit:
            f = f[:limit]

    if utils.is_iterable(f):
        # Do not use if there is only a small batch to import
        if isinstance(parallel, str) and parallel.lower() == 'auto':
            if len(f) < 100:
                parallel = False

        if parallel:
            # Do not swap this as ``isinstance(True, int)`` returns ``True``
            if isinstance(parallel, (bool, str)):
                n_cores = os.cpu_count() - 2
            else:
                n_cores = int(parallel)

            with mp.Pool(processes=n_cores) as pool:
                results = pool.imap(_worker_wrapper, [dict(f=x,
                                                           output=output,
                                                           errors=errors,
                                                           include_subdirs=include_subdirs,
                                                           parallel=False) for x in f],
                                    chunksize=1)

                res = list(config.tqdm(results,
                                       desc='Importing',
                                       total=len(f),
                                       disable=config.pbar_hide,
                                       leave=config.pbar_leave))

        else:
            # If not parallel just import the good 'ole way: sequentially
            res = [read_mesh(x,
                             include_subdirs=include_subdirs,
                             output=output,
                             errors=errors,
                             parallel=parallel,
                             **kwargs)
                   for x in config.tqdm(f, desc='Importing',
                                        disable=config.pbar_hide,
                                        leave=config.pbar_leave)]

        if output == 'neuron':
            return core.NeuronList([r for r in res if r])

        return res

    try:
        # Open the file
        fname = '.'.join(os.path.basename(f).split('.')[:-1])
        mesh = tm.load_mesh(f)

        if output == 'trimesh':
            return mesh

        attrs = {'name': fname, 'origin': f}
        attrs.update(kwargs)
        if output == 'volume':
            return core.Volume(mesh.vertices, mesh.faces, **attrs)
        else:
            return core.MeshNeuron(mesh, **attrs)
    except BaseException as e:
        msg = f'Error reading file {fname}.'
        if errors == 'raise':
            raise ImportError(msg) from e
        elif errors == 'log':
            logger.error(f'{msg}: {e}')
        return


def _worker_wrapper(kwargs):
    """Helper for importing meshes using multiple processes."""
    return read_mesh(**kwargs)


def write_mesh(x: Union['core.NeuronList', 'core.MeshNeuron', 'core.Volume', 'tm.Trimesh'],
               filepath: Optional[str] = None,
               filetype: str = None,
               ) -> None:
    """Export meshes (MeshNeurons, Volumes, Trimeshes) to disk.

    Under the hood this is using trimesh to export meshes.

    Parameters
    ----------
    x :                 MeshNeuron | Volume | Trimesh | NeuronList
                        If multiple objects, will generate a file for each
                        neuron (see also ``filepath``).
    filepath :          None | str | list, optional
                        If ``None``, will return byte string or list of
                        thereof. If filepath will save to this file. If path
                        will save neuron(s) in that path using ``{x.id}``
                        as filename(s). If list, input must be NeuronList and
                        a filepath must be provided for each neuron.
    filetype :          stl | ply | obj, optional
                        If ``filepath`` does not include the file extension,
                        you need to provide it as ``filetype``.

    Returns
    -------
    None
                        If filepath is not ``None``.
    bytes
                        If filepath is ``None``.

    See Also
    --------
    :func:`navis.read_mesh`
                        Import neurons.
    :func:`navis.write_precomputed`
                        Write meshes to Neuroglancer's precomputed format.

    Examples
    --------

    Write ``MeshNeurons`` to folder:

    >>> import navis
    >>> nl = navis.example_neurons(3, kind='mesh')
    >>> navis.write_mesh(nl, tmp_dir, filetype='obj')

    Specify the filenames:

    >>> import navis
    >>> nl = navis.example_neurons(3, kind='mesh')
    >>> navis.write_mesh(nl, tmp_dir / '{neuron.name}.obj')

    Write directly to zip archive:

    >>> import navis
    >>> nl = navis.example_neurons(3, kind='mesh')
    >>> navis.write_mesh(nl, tmp_dir / 'meshes.zip', filetype='obj')

    """
    ALLOWED_FILETYPES = ('stl', 'ply', 'obj')
    if filetype is not None:
        utils.eval_param(filetype, name='filetype', allowed_values=ALLOWED_FILETYPES)
    else:
        # See if we can get filetype from filepath
        if filepath is not None:
            for f in ALLOWED_FILETYPES:
                if str(filepath).endswith(f'.{f}'):
                    filetype = f
                    break

        if not filetype:
            raise ValueError('Must provide mesh type either explicitly via '
                             '`filetype` variable or implicitly via the '
                             'file extension in `filepath`')

    writer = base.Writer(_write_mesh, ext=f'.{filetype}')

    return writer.write_any(x,
                            filepath=filepath)


def _write_mesh(x: Union['core.MeshNeuron', 'core.Volume', 'tm.Trimesh'],
                filepath: Optional[str] = None) -> None:
    """Write single mesh to disk."""
    if filepath and os.path.isdir(filepath):
        if isinstance(x, core.MeshNeuron):
            if not x.id:
                raise ValueError('Neuron(s) must have an ID when destination '
                                 'is a folder')
            filepath = os.path.join(filepath, f'{x.id}')
        elif isinstance(x, core.Volume):
            filepath = os.path.join(filepath, f'{x.name}')
        else:
            raise ValueError(f'Unable to generate filename for {type(x)}')

    if isinstance(x, core.MeshNeuron):
        mesh = x.trimesh
    elif isinstance(x, tm.Trimesh):
        mesh = x
    else:
        raise TypeError(f'Unable to write data of type "{type(x)}"')

    # Write to disk (will only return content if filename is None)
    content = mesh.export(filepath)

    if not filepath:
        return content
