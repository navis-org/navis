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

# Set up logging
logger = config.logger


def read_mesh(f: Union[str, Iterable],
              include_subdirs: bool = False,
              parallel: Union[bool, int] = 'auto',
              output: Union[Literal['neuron'],
                            Literal['volume'],
                            Literal['trimesh']] = 'neuron',
              errors: Union[Literal['raise'],
                            Literal['log'],
                            Literal['ignore']] = 'log',
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
                        processing if more than 10 NRRD files are imported.
                        Spawning and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the number of
                        cores (otherwise defaults to ``os.cpu_count() - 2``).
    output :            "neuron" | "volume" | "trimesh"
                        Determines function's output. See Returns.
    errors :            "raise" | "log" | "ignore"
                        If "log" or "ignore", errors will not be raised.
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

    """
    utils.eval_param(output, name='output',
                     allowed_values=('neuron', 'volume', 'trimesh'))

    # If is directory, compile list of filenames
    if isinstance(f, str) and '*' in f:
        f, ext = f.split('*')

        if not os.path.isdir(f):
            raise ValueError(f'{f} is not a path')

        if not include_subdirs:
            f = list(Path(f).glob(f'*{ext}'))
        else:
            f = list(Path(f).rglob(f'*{ext}'))

    if utils.is_iterable(f):
        # Do not use if there is only a small batch to import
        if isinstance(parallel, str) and parallel.lower() == 'auto':
            if len(f) < 10:
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
        fname = os.path.basename(f).split('.')[0]
        mesh = tm.load_mesh(f)

        if output == 'trimesh':
            return mesh

        attrs = {'name': fname, 'origin': f}
        attrs.update(kwargs)
        if output == 'volume':
            return core.Volume(mesh.vertices, mesh.faces, *attrs)
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
