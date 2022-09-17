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
import numpy as np

from glob import glob
from pathlib import Path
from typing import Union, Iterable, Optional, Dict, Any
from typing_extensions import Literal

from .. import config, utils, core
from . import base

# Set up logging
logger = config.logger



def read_tiff(f: Union[str, Iterable],
              channel: int = 0,
              threshold: Optional[Union[int, float]] = None,
              include_subdirs: bool = False,
              parallel: Union[bool, int] = 'auto',
              output: Union[Literal['voxels'],
                            Literal['dotprops'],
                            Literal['raw']] = 'voxels',
              errors: Union[Literal['raise'],
                            Literal['log'],
                            Literal['ignore']] = 'log',
              **kwargs) -> 'core.NeuronObject':
    """Create Neuron/List from TIFF file.

    Requires ``tifffile`` library which is not automatically installed!

    Parameters
    ----------
    f :                 str | iterable
                        Filename(s) or folder. If folder, will import all
                        ``.tif`` files.
    channel :           int
                        Which channel to import. Ignored if file has only one
                        channel. Can use e.g. -1 to get the last channel.
    threshold :         int | float | None
                        For ``output='dotprops'`` only: a threshold to filter
                        low intensity voxels. If ``None``, no threshold is
                        applied and all values > 0 are converted to points.
    include_subdirs :   bool, optional
                        If True and ``f`` is a folder, will also search
                        subdirectories for ``.tif`` files.
    parallel :          "auto" | bool | int,
                        Defaults to ``auto`` which means only use parallel
                        processing if more than 10 TIFF files are imported.
                        Spawning and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the number of
                        cores (otherwise defaults to ``os.cpu_count() - 2``).
    output :            "voxels" | "dotprops" | "raw"
                        Determines function's output. See Returns for details.
    errors :            "raise" | "log" | "ignore"
                        If "log" or "ignore", errors will not be raised but
                        instead empty neuron will be returned.

    **kwargs
                        Keyword arguments passed to :func:`navis.make_dotprops`
                        if ``output='dotprops'``. Use this to adjust e.g. the
                        number of nearest neighbors used for calculating the
                        tangent vector by passing e.g. ``k=5``.

    Returns
    -------
    navis.VoxelNeuron
                        If ``output="voxels"`` (default): requires TIFF data to
                        be 3-dimensional voxels. VoxelNeuron will have TIFF file
                        info as ``.tiff_header`` attribute.
    navis.Dotprops
                        If ``output="dotprops"``. Dotprops will contain TIFF
                        header as ``.tiff_header`` attribute.
    navis.NeuronList
                        If import of multiple TIFF will return NeuronList of
                        Dotprops/VoxelNeurons.
    (image, header)     (np.ndarray, OrderedDict)
                        If ``output='raw'`` return raw data contained in TIFF
                        file.

    """
    try:
        import tifffile
    except ImportError:
        raise ImportError('`navis.read_tiff` requires the `tifffile` library:\n'
                          '  pip3 install tifffile -U')

    utils.eval_param(output, name='output',
                     allowed_values=('raw', 'dotprops', 'voxels'))

    # If is directory, compile list of filenames
    if isinstance(f, (str, Path)) and Path(f).expanduser().is_dir():
        f = Path(f).expanduser()
        if not include_subdirs:
            f = [os.path.join(f, x) for x in os.listdir(f) if
                 os.path.isfile(os.path.join(f, x)) and x.endswith('.tif')]
        else:
            f = [y for x in os.walk(f) for y in glob(os.path.join(x[0], '*.tif'))]

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
                                                           channel=channel,
                                                           threshold=threshold,
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
            res = [read_tiff(x,
                             channel=channel,
                             threshold=threshold,
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

        return core.NeuronList([r for r in res if r])

    # Open the file
    f = str(Path(f).expanduser())
    fname = os.path.basename(f).split('.')[0]

    with tifffile.TiffFile(f) as tif:
        # The header contains some but not all the info
        if hasattr(tif, 'imagej_metadata'):
            header = tif.imagej_metadata
        else:
            header = {}

        # Read the x/y resolution from the first "page" (i.e. the first slice)
        res = tif.pages[0].resolution
        # Resolution to spacing
        header['xy_spacing'] = (1 / res[0], 1 / res[1])

        # Get the axes (this will be something like "ZCYX")
        axes = tif.series[0].axes

        # Generate volume
        data = tif.asarray()

    # Extract channel from volume - from what I've seen ImageJ always has the
    # "ZCYX" order
    data = data[:, channel, :, :]

    # And sort into x, y, z order
    data = np.transpose(data, axes=[2, 1, 0])

    if output == 'raw':
        return data, header

    # Try parsing units - this is modelled after the tif files you get from
    # ImageJ
    units = None
    su = None
    voxdim = np.array([1, 1, 1], dtype=np.float64)
    if 'spacing' in header:
        voxdim[2] = header['spacing']
    if 'xy_spacing' in header:
        voxdim[:2] = header['xy_spacing']
    if 'unit' in header:
        su = header['unit']
        units = [f'{m} {su}' for m in voxdim]
    else:
        units = voxdim

    try:
        if output == 'dotprops':
            # This really should be a 3D image but who knows
            if data.ndim == 3:
                if threshold:
                    data = data >= threshold

                # Convert data to x/y/z coordinates
                # Note we need to multiply units before creating the Dotprops
                # - otherwise the KNN will be wrong
                x, y, z = np.where(data)
                points = np.vstack((x, y, z)).T
                points = points * voxdim

                x = core.make_dotprops(points, **kwargs)
            else:
                raise ValueError('Data must be 2- or 3-dimensional to extract '
                                 f'Dotprops, got {data.ndim}')
            if su:
                x.units = f'1 {su}'
        else:
            x = core.VoxelNeuron(data, units=units)
    except BaseException as e:
        msg = f'Error converting file {fname} to neuron.'
        if errors == 'raise':
            raise ImportError(msg) from e
        elif errors == 'log':
            logger.error(f'{msg}: {e}')
        return

    # Add some additional properties
    x.name = fname
    x.origin = f
    x.tiff_header = header

    return x


def _worker_wrapper(kwargs):
    """Helper for importing TIFFs using multiple processes."""
    return read_tiff(**kwargs)
