#    This script is part of navis (http://www.github.com/navis-org/navis).
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

import uuid
import functools

from .. import config, core, io

# Set up logging
logger = config.get_logger(__name__)


def patch_cloudvolume():
    """Monkey patch cloud-volume to return navis neurons.

    This function must be run before initializing the `CloudVolume`! Adds new
    methods/parameters to `CloudVolume.mesh.get` and `CloudVolume.skeleton.get`.
    See examples for details.

    Examples
    --------
    >>> import navis
    >>> import cloudvolume as cv
    >>> # Monkey patch cloudvolume
    >>> navis.patch_cloudvolume()
    >>> # Connect to the Google segmentation of FAFB
    >>> vol = cv.CloudVolume('precomputed://gs://fafb-ffn1-20200412/segmentation',
    ...                       use_https=True,
    ...                       progress=False)
    >>> ids = [2137190164, 2268989790]
    >>> # Fetch as navis neuron using newly added method or ...
    >>> nl = vol.mesh.get_navis(ids, lod=3)
    >>> # ... alternatively use `as_navis` keyword argument in original method
    >>> nl = vol.mesh.get(ids, lod=3, as_navis=True)
    >>> type(nl)
    <class 'navis.core.neuronlist.NeuronList'>
    >>> # The same works for skeletons
    >>> skels = vol.skeleton.get_navis(ids)

    """
    global cv
    try:
        import cloudvolume as cv
    except ImportError:
        cv = None

    # If CV not installed do nothing
    if not cv:
        logger.info('cloud-volume appears to not be installed?')
        return

    for ds in [cv.datasource.graphene.mesh.sharded.GrapheneShardedMeshSource,
               cv.datasource.graphene.mesh.unsharded.GrapheneUnshardedMeshSource,
               cv.datasource.precomputed.mesh.unsharded.UnshardedLegacyPrecomputedMeshSource,
               cv.datasource.precomputed.mesh.multilod.UnshardedMultiLevelPrecomputedMeshSource,
               cv.datasource.precomputed.mesh.multilod.ShardedMultiLevelPrecomputedMeshSource,
               cv.datasource.precomputed.skeleton.sharded.ShardedPrecomputedSkeletonSource,
               cv.datasource.precomputed.skeleton.unsharded.UnshardedPrecomputedSkeletonSource]:
        ds.get_navis = return_navis(ds.get, only_on_kwarg=False)
        ds.get = return_navis(ds.get, only_on_kwarg=True)

    logger.info('cloud-volume successfully patched!')


def return_navis(func, only_on_kwarg=False):
    """Wrap cloud-volume mesh and skeleton sources.

    Parameters
    ----------
    func :          callable
                    Function/method to wrap.
    only_on_kwarg : bool
                    If True, will look for a `as_navis=True` (default=False)
                    keyword argument to determine if results should be converted
                    to navis neurons.

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ret_navis = kwargs.pop('as_navis', False)
        res = func(*args, **kwargs)

        if not only_on_kwarg or ret_navis:
            neurons = []
            if isinstance(res, (list, tuple)):
                res = {getattr(n, "id", uuid.uuid4()): n for n in res}
            if isinstance(res, (cv.Mesh, cv.Skeleton)):
                res = {getattr(res, "id", uuid.uuid4()): res}

            for k, v in res.items():
                if isinstance(v, cv.Mesh):
                    n = core.MeshNeuron(v, id=k, units='nm')
                    neurons.append(n)
                elif isinstance(v, cv.Skeleton):
                    swc_str = v.to_swc()
                    n = io.read_swc(swc_str)
                    n.id = k
                    n.units = 'nm'
                    neurons.append(n)
                else:
                    logger.warning(f'Skipped {k}: Unable to convert {type(v)} to '
                                   'navis Neuron.')

            return core.NeuronList(neurons)
        return res
    return wrapper
