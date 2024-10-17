from . import cave_utils

DATASTACK = "h01_c3_flat"
SERVER_ADDRESS = "https://global.brain-wire-test.org"

def get_cave_client():
    """Get caveclient for H01 dataset.
    """
    return cave_utils.get_cave_client(DATASTACK, SERVER_ADDRESS)

def fetch_neurons(x, *, lod=2,
                  with_synapses=True,
                  datastack=DATASTACK,
                  parallel=True,
                  max_threads=4,
                  **kwargs):
    """Fetch neuron meshes.

    Notes
    -----
    Synapses will be attached to the closest vertex on the mesh.

    Parameters
    ----------
    x :             str | int | list-like
                    Segment ID(s). Multiple Ids can be provided as list-like.
    lod :           int
                    Level of detail. Higher ``lod`` = coarser. This parameter
                    is ignored if the data source does not support multi-level
                    meshes.
    with_synapses : bool, optional
                    If True will also attach synapses as ``.connectors``.
    datastack :     str
                    DATASTACK.
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
    return cave_utils.fetch_neurons(
        x,
        lod=lod,
        with_synapses=with_synapses,
        datastack=datastack,
        parallel=parallel,
        max_threads=max_threads,
        **kwargs
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
    return cave_utils.get_voxels(
        x,
        mip=mip,
        bounds=bounds,
        datastack=datastack
    )