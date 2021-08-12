#    This script is part of navis (http://www.github.com/schlegelp/navis).
#    Copyright (C) 2017 Philipp Schlegel
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

import functools

from .. import config, core, io

# Set up logging
logger = config.logger


def patch_cloudvolume():
    """Monkey patch cloudvolume to return navis neurons.

    This function must be run before initializing a `CloudVolume`.
    """
    global cv
    try:
        import cloudvolume as cv
    except ImportError:
        cv = None

    # If CV not installed do nothing
    if not cv:
        logger.info('cloudvolume appears to not be installed?')
        return

    for ds in [cv.datasource.graphene.mesh.sharded.GrapheneShardedMeshSource,
               cv.datasource.graphene.mesh.unsharded.GrapheneUnshardedMeshSource,
               cv.datasource.precomputed.mesh.unsharded.UnshardedLegacyPrecomputedMeshSource,
               cv.datasource.precomputed.mesh.multilod.UnshardedMultiLevelPrecomputedMeshSource,
               cv.datasource.precomputed.mesh.multilod.UnshardedMultiLevelPrecomputedMeshSource,
               cv.datasource.precomputed.mesh.multilod.ShardedMultiLevelPrecomputedMeshSource,]:
        ds.get = return_neurons(ds.get)

    logger.info('cloudvolume successfully patched!')


def return_neurons(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        neurons = []
        for k, v in res.items():
            if isinstance(v, cv.Mesh):
                n = core.MeshNeuron(v, id=k)
                neurons.append(n)
            elif isinstance(v, cv.Skeleton):
                swc_str = v.to_swc()
                n = io.read_swc(swc_str)
                n.id = k
                neurons.append(n)
            else:
                logger.warning(f'Skipped {k}: Unable to convert {type(v)} to '
                               'navis Neuron.')
        return core.NeuronList(neurons)
    return wrapper
