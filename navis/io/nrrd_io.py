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

import nrrd
import os

import numpy as np

from glob import glob

import multiprocessing as mp

from typing import Union, Iterable
from typing_extensions import Literal

from .. import config, utils, core

# Set up logging
logger = config.logger


def read_nrrd(f: Union[str, Iterable],
              include_subdirs: bool = False,
              parallel: Union[bool, int] = 'auto',
              output: Union[Literal['dotprops'],
                            Literal['raw']] = 'dotprops',
              errors: Union[Literal['raise'],
                            Literal['log'],
                            Literal['ignore']] = 'log',
              **kwargs) -> 'core.NeuronObject':
    """Create Dotprops Neuron/List from NRRD file.

    See `here <http://teem.sourceforge.net/nrrd/format.html>`_ for specs of
    NRRD file format including description of the headers.

    Parameters
    ----------
    f :                 str | iterable
                        Filename(s) or folder. If folder, will import all
                        ``.nrrd`` files.
    include_subdirs :   bool, optional
                        If True and ``f`` is a folder, will also search
                        subdirectories for ``.nrrd`` files.
    parallel :          "auto" | bool | int,
                        Defaults to ``auto`` which means only use parallel
                        processing if more than 10 NRRD files are imported.
                        Spawning and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the
                        number of cores (otherwise defaults to
                        ``os.cpu_count() - 2``).
    output :            "dotprops" | "raw"
                        Determines function's output. See Returns.
    errors :            "raise" | "log" | "ignore"
                        If "log" or "ignore", errors will not be raised but
                        instead empty Dotprops will be returned.

    **kwargs
                        Keyword arguments passed to :func:`navis.make_dotprops`.
                        Use this to e.g. adjust the number of nearest neighbors
                        used for calculating the tangent vector.

    Returns
    -------
    navis.Dotprops
                        If output is "dotprops". Contains NRRD header as
                        ``.nrrd_header`` attribute.
    navis.NeuronList
                        If import of multiple NRRD will return NeuronList of
                        Dotprops.
    (image, header)     (np.ndarray, OrderedDict)
                        If ``output='raw'`` return raw data contained in NRRD
                        file.

    """
    # If is directory, compile list of filenames
    if isinstance(f, str) and os.path.isdir(f):
        if not include_subdirs:
            f = [os.path.join(f, x) for x in os.listdir(f) if
                 os.path.isfile(os.path.join(f, x)) and x.endswith('.nrrd')]
        else:
            f = [y for x in os.walk(f) for y in glob(os.path.join(x[0], '*.nrrd'))]

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
            res = [read_nrrd(x,
                             include_subdirs=include_subdirs,
                             output=output,
                             errors=errors,
                             parallel=parallel,
                             **kwargs)
                   for x in config.tqdm(f, desc='Importing',
                                        disable=config.pbar_hide,
                                        leave=config.pbar_leave)]

        if output == 'raw':
            return [r[0] for r in res], [r[1] for r in res]

        return core.NeuronList(res)

    # Open the file
    data, header = nrrd.read(f)

    if output == 'raw':
        return data, header

    # Data is in voxels - we have to convert it to x/y/z coordinates
    x, y, z = np.where(data)
    points = np.vstack((x, y, z)).T

    # Generate dotprops
    fname = os.path.basename(f).split('.')[0]
    try:
        dp = core.make_dotprops(points, **kwargs)
    except BaseException as e:
        msg = f'Error converting file {fname} to Dotprops'
        if errors == 'raise':
            raise ImportError(msg) from e
        elif errors == 'log':
            logger.error(f'{msg}: {e}')
        dp = core.Dotprops(None, None, None)

    # Add some additional properties
    dp.name = fname
    dp.origin = f
    dp.nrrd_header = header

    try:
        dp.units = header.get('space units', None)
    except BaseException:
        pass

    return dp


def _worker_wrapper(kwargs):
    """Helper for importing SWCs using multiple processes."""
    return read_nrrd(**kwargs)
