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

import io
import os
import requests

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.colors as mcl
import trimesh as tm

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from typing import Union, Optional, List

from .. import config

from ..core import Volume, TreeNeuron, NeuronList
from ..utils import make_url, make_iterable

logger = config.logger
baseurl = 'https://www.insectbraindb.org'


class Session:
    """Client to manage Insect Brain DB session.

    Parameters
    ----------
    token :         str
                    API token. See `authenticate()`.
    created_at :    str
                    Time and date the token was generated. Iso-formatted.

    """

    def __init__(self, username=None, password=None, token=None, created_at=None):
        self._session = requests.Session()

        self.username = username
        self.password = password

        self.token = token
        self.created_at = created_at

    @property
    def token_expired(self):
        """Check if token is expired."""
        if self._token_created_at:
            now = dt.datetime.now()
            expires_in = dt.timedelta(days=1)
            if now - self._token_created_at >= expires_in:
                return True
        return False

    @property
    def token(self):
        return self._token

    @token.setter
    def token(self, token):
        if token and not token.startswith('Token'):
            token = f'Token {token}'
        self._token = token
        self._session.headers['Authorization'] = token

    @property
    def token_created_at(self):
        return self._token_created_at

    @token_created_at.setter
    def token_created_at(self, value):
        if value:
            self._token_created_at = dt.datetime.fromisoformat(value[:-1])
        else:
            self._token_created_at = None

    def fetch_token(self):
        """Fetch fresh token."""
        username = self.username
        if not username:
            username = os.environ.get('INSECT_BRAIN_DB_USER', None)
        password = self.password
        if not password:
            password = os.environ.get('INSECT_BRAIN_DB_PASSWORD', None)

        if not username or not password:
            msg = """\
            You must provide username + password, or an API token. Please see
            `navis.interfaces.insectbrian_db.authenticate()` for details.
            """
            raise ValueError(msg)

        creds = {'username': username, 'password': password}

        # Note: do NOT remove the trailing '/' here
        url = make_url(baseurl, 'api', 'v2', 'token/')

        resp = requests.post(url, data=creds)
        resp.raise_for_status()

        global session
        self.token = resp.json()['token']
        self.token_created_at = resp.json()['created']

        logger.info('Successfully retrieved 24h Insect Brain DB API token!')

    def preflight(self):
        """Check if we're ready to make requests."""
        if not self.token or self.token_expired:
            self.fetch_token()

    def get(self, *args, **kwargs):
        """Make GET request."""
        self.preflight()

        r = self._session.get(*args, **kwargs)
        r.raise_for_status()

        return r.json()

    def post(self, *args, **kwargs):
        """Make POST request."""
        self.preflight()

        r = self._session.post(*args, **kwargs)
        r.raise_for_status()

        return r.json()


def authenticate(username=None, password=None, token=None):
    """Authenticate against Insect Brain DB.

    You can either provide username + password, or a token. Each token is only
    valid for 24h though. The better alternative is to provide your
    username + password as environment variables `INSECT_BRAIN_DB_USER` and
    `INSECT_BRAIN_DB_PASSWORD` password respectively in which case you don't
    need to use `authenticate()` at all.

    Parameters
    ----------
    username :      str, optional
                    Your username on Insect Brain DB.
    password :      str, optional
                    Your password on Insect Brain DB.
    token :         str, optional
                    A token. If provided you don't need to provide username +
                    password.

    """
    if not token and (not username and not password):
        raise ValueError('Must provide either username + password, or token '
                         '(or both).')

    if username:
        session.username = username
    if password:
        session.password = password

    if token:
        session.token = token
    else:
        session.fetch_token()


def get_brain_meshes(species: Union[str, int],
                     combine: bool = False
                     ) -> Optional[List[Volume]]:
    """Fetch brain meshes for given species.

    Parameters
    ----------
    species:        str | int
                    Species for which to fetch brain volumes. Strings are
                    interpreted as names (scientific or common), integers as IDs.
    combine :       bool, optional
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
    >>> import navis.interfaces.insectbrain_db as ibdb
    >>> v = ibdb.get_brain_meshes('Desert Locust', combine_vols=True)
    >>> navis.plot3d(v)

    """
    obj_url = 'https://s3.eu-central-1.amazonaws.com/ibdb-file-storage/'

    # Get info with all available neuropils
    sp_info = get_species_info(species)

    # Go over all brains
    n_brains = len(sp_info.reconstructions)  # type: ignore
    n_reconstr = len([r for r in sp_info.reconstructions if r.get('viewer_files')])  # type: ignore
    logger.info(f'{n_reconstr} reconstruction(s) from {n_brains} brain(s) found')

    volumes: List[Volume] = []
    for brain in config.tqdm(sp_info.reconstructions,
                             disable=config.pbar_hide,
                             leave=config.pbar_leave,
                             desc='Brains'):  # type: ignore
        this_v = []
        # If no reconstructions, continue
        if not brain.get('viewer_files'):  # type: ignore
            continue
        for file in config.tqdm(brain['viewer_files'],
                                desc='Neuropils',
                                disable=config.pbar_hide,
                                leave=config.pbar_leave):
            filename = file['p_file']['file_name']
            path = file['p_file']['path']
            url = make_url(obj_url, path)

            resp = requests.get(url)
            resp.raise_for_status()

            f = io.BytesIO(resp.content)
            mesh = tm.load_mesh(f, file_type='obj')

            structures = file.get('structures')
            if structures:
                structure = structures[0].get('structure')
                hemisphere = structures[0].get('hemisphere')
                if structure:
                    color = structure.get('color')
                    if color:
                        color = mcl.to_rgba(color, alpha=.5)
                    else:
                        color = (.85, .85, .85, .5)
                name = structure.get('name', filename)

                if hemisphere:
                    name = f'{name} ({hemisphere})'

            v = Volume(mesh, name=name, color=color)
            this_v.append(v)

        # Combine all volumes in this brain
        if combine:
            this_v = [Volume.combine(this_v)]
            this_v[0].color = (.85, .85, .85, .5)

        volumes += this_v

    return volumes


@lru_cache()
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
    >>> import navis.interfaces.insectbrain_db as ibdb
    >>> info = ibdb.get_species_info()

    """
    # First get species ID
    if isinstance(species, str):
        species = _get_species_id(species)

    url = make_url(baseurl, '/archive/species/most_current_permitted/',
                   species_id=species)

    resp = requests.get(url)

    resp.raise_for_status()

    return pd.Series(resp.json())


@lru_cache()
def get_available_species() -> pd.DataFrame:
    """Get all info for given species.

    Returns
    -------
    pandas.DataFrame
            DataFrame with available species.

    Examples
    --------
    >>> import navis.interfaces.insectbrain_db as ibdb
    >>> species = ibdb.get_available_species()

    """
    url = make_url(baseurl, 'api', 'v2', 'species')

    return _sort_columns(pd.DataFrame.from_records(session.get(url)))


def get_skeletons_species(species, max_threads=4):
    """Fetch all skeletons for given species.

    Note that some neurons might have multiple reconstructions. They will
    show up with the same ID with different names.

    Parameters
    ----------
    species :       str | int
                    Name or ID of a species to fetch skeletons for.
    max_threads :   int
                    Number of parallel threads to use for fetching skeletons.

    Returns
    -------
    navis.NeuronList

    Examples
    --------
    >>> import navis.interfaces.insectbrain_db as ibdb
    >>> neurons = ibdb.get_skeletons_species('Desert Locust')

    """
    if isinstance(species, str):
        species = _get_species_id(species)

    # First fetch URLs for all neurons
    url = make_url(baseurl, 'api', 'v2', 'neuron', 'reconstruction',
                   neuron__species=species)
    meta = session.get(url)

    meta = [e for e in meta if e['viewer_files']]

    return _get_skeletons(meta, max_threads=max_threads)


def get_skeletons(x, max_threads=4):
    """Fetch skeletons for given neuron(s).

    Parameters
    ----------
    x :             str | int | list thereof
                    Name(s) or ID(s) of neurons you want to fetch.
    max_threads :   int
                    Number of parallel threads to use for fetching skeletons.

    Returns
    -------
    navis.NeuronList

    Examples
    --------
    >>> import navis.interfaces.insectbrain_db as ibdb
    >>> neurons = ibdb.get_skeletons('TUps2-2')

    """
    if isinstance(x, (int, str, np.int32, np.int64)):
        neurons = [x]
    else:
        neurons = x

    # First fetch URLs for all neurons
    meta = []
    for x in neurons:
        if isinstance(x, str):
            q = search_neurons(name=x, partial_match=False)
            if q.empty:
                raise ValueError(f'No neuron with name "{x}" found')
            ids = q.id.values
        else:
            ids = x

        for i, id in enumerate(make_iterable(ids)):
            url = make_url(baseurl, 'api', 'v2', 'neuron', 'reconstruction',
                           neuron=id)
            info = session.get(url)

            if (not info
                or 'viewer_files' not in info[0]
                or not info[0]['viewer_files']):
                raise ValueError(f'Neuron {x} ({id}) has no skeleton.')

            meta.append(info[0])

    return _get_skeletons(meta, max_threads=max_threads)


def _get_skeletons(meta, max_threads=4):
    """Fetch skeleton(s) from info."""
    nl = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {}
        for inf in meta:
            id = inf['neuron']
            desc = inf.get('description', '')
            for file in inf['viewer_files']:
                url = file['url']
                fn = file['file_name']
                f = executor.submit(_fetch_single_neuron,
                                    url,
                                    name=fn,
                                    description=desc,
                                    id=id)
                futures[f] = fn

        with config.tqdm(desc='Fetching',
                         total=len(futures),
                         leave=config.pbar_leave,
                         disable=len(futures) == 1 or config.pbar_hide) as pbar:
            for f in as_completed(futures):
                name = futures[f]
                pbar.update(1)
                try:
                    nl.append(f.result())
                except Exception as exc:
                    print(f'{name} generated an exception:', exc)

    return NeuronList(nl)


def _fetch_single_neuron(url, **kwargs):
    """Load and parse SWC from given URL."""
    resp = requests.get(url)
    resp.raise_for_status()

    s = io.StringIO(resp.content.decode())

    swc = pd.read_csv(s,
                      delimiter=' ', comment='#',
                      header=None, skipinitialspace=True)

    swc.columns = ['node_id', 'label', 'x', 'y', 'z', 'radius', 'parent_id']
    swc['radius'] /= 2

    return TreeNeuron(swc, units='um', soma=None, **kwargs)


def search_neurons(name=None, short_name=None, species=None, sex=None,
                   arborization=None, partial_match=True) -> pd.DataFrame:
    """Search for neurons matching given parameters.

    Parameters
    ----------
    name :          str, optional
                    Name of the neuron.
    short_name :    str, optional
                    Short name of the neuron.
    species :       str | int, optional
                    Name or ID of the species. Can be common or scientific name.
    sex :           "FEMALE" | "MALE" | "UNKNOWN", optional
                    Sex of the neuron.
    arborization :  str, optional
                    Restrict to neurons having arborizations in given neuropil.
    partial_match : bool
                    Whether to allow partial matches (does not apply for species).

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> import navis.interfaces.insectbrain_db as ibdb
    >>> neurons = ibdb.search_neurons(species='Desert Locust')

    """
    # Construct query
    options = {}
    if species:
        if not isinstance(species, int):
            species = _get_species_id(species)
        options['species'] = species

    for key, value in zip(['name', 'short_name', 'sex',
                           'arborization_region__structure'],
                          [name, short_name, sex, arborization]):
        if not value:
            continue
        if partial_match:
            key += '__icontains'
        options[key] = value

    url = make_url(baseurl, 'api', 'v2', 'neuron', **options)

    resp = requests.get(url)

    resp.raise_for_status()

    return _sort_columns(pd.DataFrame.from_records(resp.json()))


def _get_species_id(species):
    """Map species name to its ID."""
    spec = get_available_species()
    if species in spec.scientific_name.values:
        id = spec.set_index('scientific_name').loc[species, 'id']
    elif species in spec.common_name.values:
        id = spec.set_index('common_name').loc[species, 'id']
    else:
        raise ValueError(f'Unable to find an ID for species "{species}"')

    return id


def _sort_columns(df):
    """Sort DataFrame columns such that irrelevant columns are in the middle."""
    # Some hard-coded priorities
    prio = {}
    prio['id'] = 0
    prio.update({c: 2 for c in df.columns if 'name' in c})
    prio['name'] = 1
    prio['description'] = 100

    cols = sorted(df.columns, key=lambda x: prio.get(x, 10))

    return df[cols]


session = Session()
