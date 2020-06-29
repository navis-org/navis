#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

""" Set of functions to interface with the www.insectbraindb.org database of
insect brains and neurons.
"""

import os
import requests
import tempfile

import pandas as pd

from tqdm import tqdm

from typing import Union, Optional, List

from ..core import Volume
from ..utils import make_url
from ..config import logger
from ..plotting.colors import hex_to_rgb


baseurl = 'http://www.insectbraindb.org'


def get_brain_meshes(species: Union[str, int],
                     save_to: Optional[str] = None,
                     combine_vols: bool = False
                     ) -> Optional[List[Volume]]:
    """Fetch brain meshes for given species.

    Parameters
    ----------
    species:        str | int
                    Species for which to fetch brain volumes. Strings are
                    interpreted as names, integers as IDs.
    save_to:        str, optional
                    Folder to save .obj files to. If provided, will NOT return
                    navis.Volume.
    combine_vols :  bool, optional
                    If True, will combine subvolumes (i.e. neuropils) into
                    a single navis.Volume - else will return list with volumes.

    Returns
    -------
    None
                    If ``save_to`` is folder.
    list of navis.Volume
                    If ``combine_vols`` is False.

    Examples
    --------
    >>> import navis
    >>> import navis.interfaces.insectbrain_db as ibd
    >>> v = ibd.get_brain_meshes('Desert Locust', combine_vols=True)
    >>> navis.plot3d(v)

    """
    if isinstance(save_to, str):
        if not os.path.isdir(save_to):
            raise ValueError('"save_to" must be None or valid path.')

    obj_url = 'https://s3.eu-central-1.amazonaws.com/ibdb-file-storage/'

    # Get info with all available neuropils
    sp_info = get_species_info(species)

    # Go over all brains
    n_brains = len(sp_info.reconstructions)  # type: ignore
    n_reconstr = len([r for r in sp_info.reconstructions if r.get('viewer_files')])  # type: ignore
    logger.info(f'{n_reconstr} reconstruction(s) from {n_brains} brain(s) found')

    volumes: List[Volume] = []
    for brain in tqdm(sp_info.reconstructions, desc='Brains'):  # type: ignore
        this_v = []
        # If no reconstructions, continue
        if not brain.get('viewer_files'):  # type: ignore
            continue
        for file in tqdm(brain['viewer_files'], desc='Meshes'):
            filename = file['p_file']['file_name']
            path = file['p_file']['path']
            url = make_url(obj_url, path)

            if file.get('neuropil'):
                color = hex_to_rgb(file['neuropil'][0]['color'])
            else:
                color = (1, 1, 1, .1)

            resp = requests.get(url)
            resp.raise_for_status()

            if save_to:
                fp = os.path.join(save_to, filename)
            else:
                # Save temporary file
                fp = os.path.join(tempfile.gettempdir(), 'temp.obj')

            with open(fp, 'w') as f:
                f.write(resp.content.decode())

            if not save_to:
                v = Volume.from_file(fp, name=filename, color=color)
                this_v.append(v)

        # Combine all volumes in this brain
        if combine_vols:
            this_v = [Volume.combine(this_v)]

        volumes += this_v

    if not save_to:
        return volumes
    else:
        return None


def get_species_info(species: Union[str, int]) -> pd.Series:
    """Get all info for given species.

    Parameters
    ----------
    species :       str | int
                    Species to get info for.

    Returns
    -------
    pandas.Series
                    Pandas Series with info on given species.

    Examples
    --------
    >>> import navis.interfaces.insectbrain_db as ibd
    >>> info = ibd.get_species_info()

    """
    # First get species ID
    if not isinstance(species, int):
        all_species = get_available_species()
        if species in all_species.common_name.values:
            sid = all_species.set_index('common_name',
                                        inplace=False).loc[species, 'id']
        elif species in all_species.scientific_name.values:
            sid = all_species.set_index('scientific_name',
                                        inplace=False).loc[species, 'id']
        else:
            raise ValueError(f'Species "{species}" not found.')
    else:
        sid = species

    url = make_url(baseurl, '/archive/species/most_current_permitted/',
                   species_id=sid)

    resp = requests.get(url)

    resp.raise_for_status()

    return pd.Series(resp.json())


def get_available_species() -> pd.DataFrame:
    """Get all info for given species.

    Returns
    -------
    pandas.DataFrame
            DataFrame with available species.

    Examples
    --------
    >>> import navis.interfaces.insectbrain_db as ibd
    >>> species = ibd.get_available_species()

    """
    url = make_url(baseurl, 'api', 'species', 'min')

    resp = requests.get(url)

    resp.raise_for_status()

    return pd.DataFrame.from_records(resp.json())
