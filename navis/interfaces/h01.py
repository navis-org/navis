from functools import lru_cache
from . import cave_utils

@lru_cache(None)
def get_cave_client():
    """Get caveclient for H01 dataset.
    """
    datastack = "h01_c3_flat"
    server_address = "https://global.brain-wire-test.org/"
    
    return cave_utils.get_cave_client(datastack, server_address)

def fetch_neurons(x, *, lod=2,
                  with_synapses=True,
                  datastack='h01_c3_flat',
                  parallel=True,
                  max_threads=4,
                  **kwargs):
    """Fetch neuron meshes.
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

def get_voxels(x, mip=0, bounds=None, datastack='h01_c3_flat'):
    """Fetch voxels making a up given root ID.
    """
    return cave_utils.get_voxels(
        x,
        mip=mip,
        bounds=bounds,
        datastack=datastack
    )