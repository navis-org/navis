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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.

from . import cave_utils

DATASTACK = "h01_c3_flat"
SERVER_ADDRESS = "https://global.brain-wire-test.org"
NUCLEUS_TABLE = "nucleus"


def get_cave_client(datastack=DATASTACK):
    """Get caveclient for H01 dataset."""
    client = cave_utils.get_cave_client(datastack, SERVER_ADDRESS)
    client.materialize.nucleus_table = NUCLEUS_TABLE
    return client


def fetch_neurons(
    x,
    *,
    lod=2,
    with_synapses=True,
    datastack=DATASTACK,
    materialization="auto",
    parallel=True,
    max_threads=4,
    **kwargs,
):
    """Fetch neuron meshes.

    Notes
    -----
    Synapses will be attached to the closest vertex on the mesh.

    Parameters
    ----------
    x :             str | int | list-like
                    Segment ID(s). Multiple IDs can be provided as list-like.
    lod :           int
                    Level of detail. Higher ``lod`` = coarser. This parameter
                    is ignored if the data source does not support multi-level
                    meshes.
    with_synapses : bool, optional
                    If True will also attach synapses as ``.connectors``.
    datastack :     str
                    Datastack to use. Default to "h01_c3_flat".
    materialization : "auto" | int
                    Which materialization version to use to look up somas and synapses
                    (if applicable). If "auto" (default) will try to find the most
                    recent version that contains the given root IDs. If an
                    integer is provided will use that version.
    parallel :      bool
                    If True, will use parallel threads to fetch data.
    max_threads :   int
                    Max number of parallel threads to use.
    **kwargs
                    Keyword arguments are passed through to the initialization
                    of the ``navis.MeshNeurons``.

    Returns
    -------
    navis.Neuronlist
                    Containing :class:`navis.MeshNeuron`.

    """
    client = get_cave_client(datastack)
    return cave_utils.fetch_neurons(
        x,
        lod=lod,
        with_synapses=with_synapses,
        client=client,
        parallel=parallel,
        max_threads=max_threads,
        materialization=materialization,
        **kwargs,
    )


def get_voxels(x, mip=0, bounds=None, datastack=DATASTACK):
    """Fetch voxels making a up given root ID.

    Parameters
    ----------
    x :             int
                    A single root ID.
    mip :           int
                    Scale at which to fetch voxels.
    bounds :        list, optional
                    Bounding box [xmin, xmax, ymin, ymax, zmin, zmax] in voxel
                    space. For example, the voxel resolution for mip 0
                    segmentation is 8 x 8 x 40 nm.
    datastack :     str
                    DATASTACK.

    Returns
    -------
    voxels :        (N, 3) np.ndarray
                    In voxel space according to `mip`.

    """
    return cave_utils.get_voxels(x, mip=mip, bounds=bounds, client=get_cave_client(datastack))
