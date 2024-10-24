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

"""Interface with MICrONS datasets: https://www.microns-explorer.org/."""

from .. import config
from functools import lru_cache
from textwrap import dedent
from . import cave_utils

err_msg = dedent("""
      Failed to import `caveclient` library. Please install using pip:

            pip install caveclient -U

      """)

try:
    from caveclient import CAVEclient
    import cloudvolume as cv
except ModuleNotFoundError:
    config.logger.error(err_msg)
    CAVEclient = None
    cv = None
except BaseException:
    raise

NUCLEUS_TABLE = "nucleus_detection_v0"


@lru_cache(None)
def _translate_datastack(datastack):
    """Translate datastack to source."""
    ds = get_datastacks(microns_only=False)

    if datastack in ds:
        return datastack
    elif datastack in ("cortex65", "minnie65"):
        # Find the latest cortex65 datastack
        # "minnie65_public" is apparently the prefered datastack
        if "minnie65_public" in ds:
            return "minnie65_public"
        # If for some reason that stack is not available, just take the latest
        return sorted([d for d in ds if "minnie65_public" in d and "sandbox" not in d])[
            -1
        ]
    elif datastack in ("cortex35", "minnie35"):
        # Find the latest cortex35 datastack
        return sorted([d for d in ds if "minnie35_public" in d and "sandbox" not in d])[
            -1
        ]
    elif datastack == "layer 2/3":
        # The "pinky_sandbox" seems to be the latest layer 2/3 datastack
        return sorted([d for d in ds if "pinky" in d])[-1]
    raise ValueError(f"Datastack '{datastack}' not found.")


@lru_cache(None)
def get_datastacks(microns_only=True):
    """Get available datastacks.

    Parameters
    ----------
    microns_only : bool
        If True, only return MICrONS datastacks.

    Returns
    -------
    list
        List of available datastacks.

    """
    if not CAVEclient:
        raise ModuleNotFoundError(err_msg)

    stacks = CAVEclient().info.get_datastacks()

    if microns_only:
        stacks = [s for s in stacks if "minnie" in s or "pinky" in s]
    return stacks


def get_cave_client(datastack="cortex65"):
    """Get caveclient for given datastack.

    Parameters
    ----------
    datastack :     "cortex65" | "cortex35" | "layer 2/3" | str
                    Which dataset to query. "cortex65", "cortex35" and "layer 2/3"
                    are internally mapped to the corresponding sources: for example,
                    "minnie65_public" for "cortex65"

    """
    if not CAVEclient:
        raise ModuleNotFoundError(err_msg)

    # Try mapping, else pass-through
    datastack = _translate_datastack(datastack)
    client = cave_utils.get_cave_client(datastack)
    client.materialize.nucleus_table = NUCLEUS_TABLE
    return client


@lru_cache(None)
def list_annotation_tables(datastack="cortex65"):
    """Get available annotation tables for given datastack.

    Parameters
    ----------
    datastack :     "cortex65" | "cortex35" | "layer 2/3" | str
                    Which dataset to query. "cortex65", "cortex35" and "layer 2/3"
                    are internally mapped to the corresponding sources: for example,
                    "minnie65_public" for "cortex65"

    Returns
    -------
    list
        List of available annotation tables.

    """
    return get_cave_client(datastack).materialize.get_tables()


def fetch_neurons(
    x,
    *,
    lod=2,
    with_synapses=True,
    datastack="cortex65",
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
                    Segment ID(s). Multiple Ids can be provided as list-like.
    lod :           int
                    Level of detail. Higher ``lod`` = coarser. This parameter
                    is ignored if the data source does not support multi-level
                    meshes.
    with_synapses : bool, optional
                    If True will also attach synapses as ``.connectors``.
    datastack :     "cortex65" | "cortex35" | "layer 2/3" | str
                    Which dataset to query. "cortex65", "cortex35" and "layer 2/3"
                    are internally mapped to the corresponding sources: for example,
                    "minnie65_public" for "cortex65"
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
    client = get_cave_client(_translate_datastack(datastack))
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


def get_voxels(x, mip=0, bounds=None, datastack="cortex65"):
    """Fetch voxels making a up given root ID.

    Parameters
    ----------
    x :             int
                    A single root ID.
    mip :           int
                    Scale at which to fetch voxels. Lower = higher resolution.
    bounds :        list, optional
                    Bounding box [xmin, xmax, ymin, ymax, zmin, zmax] in voxel
                    space. For example, the voxel resolution for mip 0
                    segmentation is 8 x 8 x 40 nm.
    datastack :     "cortex65" | "cortex35" | "layer 2/3" | str
                    Which dataset to query. "cortex65", "cortex35" and "layer 2/3"
                    are internally mapped to the corresponding sources: for example,
                    "minnie65_public" for "cortex65"

    Returns
    -------
    voxels :        (N, 3) np.ndarray
                    In voxel space according to `mip`.

    """
    return cave_utils.get_voxels(
        x, mip=mip, bounds=bounds, client=get_cave_client(datastack)
    )
