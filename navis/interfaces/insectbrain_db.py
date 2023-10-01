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
from urllib.parse import urlparse

from typing import Union, Optional, List

from .. import config

from ..core import Volume, TreeNeuron, NeuronList
from ..utils import make_url, make_iterable

logger = config.get_logger(__name__)
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
        if self.token and self.token_expired:
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
    username + password as environment variables: `INSECT_BRAIN_DB_USER` and
    `INSECT_BRAIN_DB_PASSWORD`, respectively. If you are using these environment
    you don't need to bother with `authenticate()` at all.

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
                     combine: bool = False,
                     max_threads: int = 4
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
    max_threads :   int
                    Number of parallel threads to use for fetching meshes.

    Returns
    -------
    list of navis.Volume

    Examples
    --------
    >>> import navis
    >>> import navis.interfaces.insectbrain_db as ibdb
    >>> v = ibdb.get_brain_meshes('Desert Locust', combine_vols=True)
    >>> navis.plot3d(v)

    """
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

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {}
            for file in brain['viewer_files']:
                # If no file UUID, continue
                if not file['p_file']['uuid']:
                    continue
                filename = file['p_file']['file_name']
                f = executor.submit(_get_neuropil_mesh, file,)
                futures[f] = filename

            with config.tqdm(desc='Fetching',
                            total=len(futures),
                            leave=config.pbar_leave,
                            disable=len(futures) == 1 or config.pbar_hide) as pbar:
                for f in as_completed(futures):
                    name = futures[f]
                    pbar.update(1)
                    try:
                        this_v.append(f.result())
                    except Exception as exc:
                        print(f'{name} generated an exception:', exc)

        # Combine all volumes in this brain
        if combine:
            this_v = [Volume.combine(this_v)]
            this_v[0].color = (.85, .85, .85, .5)
            this_v[0].name = sp_info.scientific_name

        volumes += this_v

    return volumes


def _get_neuropil_mesh(file):
    filename = file['p_file']['file_name']
    # Get the AWS URL (with all the required headers) for this object
    url = _get_download_url(file['p_file']['uuid'])

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
    else:
        name = file['p_file']['file_name']

    return Volume(mesh, name=name, color=color)


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


# Note to self: do not cache as the URLs expire eventually
def list_experiment_files(id) -> pd.DataFrame:
    """List files associated with given experiment.

    Parameters
    ----------
    id :    int
            The experiment ID. See e.g. ``list_datasets``.

    Returns
    -------
    pandas.DataFrame
            DataFrame with files.

    Examples
    --------
    >>> import navis.interfaces.insectbrain_db as ibdb
    >>> files = ibdb.list_experiment_files(61)

    """
    url = make_url(baseurl, 'api', 'v2', 'experiment', id, 'file')

    return _sort_columns(pd.DataFrame.from_records(session.get(url)))


def list_datasets() -> pd.DataFrame:
    """List publication datasets and associated experiments.

    Returns
    -------
    pandas.DataFrame
            DataFrame with available datasets.

    Examples
    --------
    >>> import navis.interfaces.insectbrain_db as ibdb
    >>> datasets = ibdb.list_datasets()

    """
    url = make_url(baseurl, 'api', 'publications', 'experiments?offset=0&limit=500')

    return _sort_columns(pd.DataFrame.from_records(session.get(url)['results']))


def get_skeletons_experiment(id) -> 'NeuronList':
    """Fetch all skeletons for given experiment.

    Parameters
    ----------
    id :    int
            The experiment ID. See e.g. ``list_datasets``.

    Returns
    -------
    NeuronList

    Examples
    --------
    >>> import navis.interfaces.insectbrain_db as ibdb
    >>> nl = ibdb.get_skeletons_experiment(61)

    """
    # Make sure ID is integer
    id = int(id)

    # Get files associated with experiment
    files = list_experiment_files(id)

    # Figure out which files are skeletons
    sk_files = files[files.file_name.str.contains('skeleton') | files.file_name.str.endswith('.gz')]

    if sk_files.empty:
        raise ValueError('Did not find any skeleton files associated with '
                         f'experiment {id}')

    skeletons = []
    for f in sk_files.itertuples():
        logger.info(f'Downloading {f.file_name}')
        # Load the file
        r = requests.get(f.url)
        r.raise_for_status()

        # Files appear to be json-formatted and not compressed
        data = r.json()

        for i, neuron in enumerate(data['data']):
            for sk in neuron['skeletons']:
                # Load SWC table
                swc = pd.DataFrame(sk['data'],
                                   columns=['node_id', 'skeleton_id',
                                            'x', 'y', 'z', 'radius',
                                            'parent_id'])
                # Some cleaning up
                swc.drop('skeleton_id', axis=1, inplace=True)
                swc['parent_id'] = swc.parent_id.fillna(-1).astype(int)
                # Create neuron
                tn = TreeNeuron(swc,
                                id=sk.get('id', 1),
                                name=neuron.get('name', 'NA'),
                                annotations=neuron.get('annotations', []),
                                soma=None)
                skeletons.append(tn)
    logger.info(f'Done! Found {len(skeletons)} skeletons.')

    return NeuronList(skeletons)


def get_meshes_experiment(id) -> 'NeuronList':
    """Fetch volumes associated with given experiment.

    Parameters
    ----------
    id :    int
            The experiment ID. See e.g. ``list_datasets``.

    Returns
    -------
    list

    Examples
    --------
    >>> import navis.interfaces.insectbrain_db as ibdb
    >>> vols = ibdb.get_meshes_experiment(61)

    """
    # Make sure ID is integer
    id = int(id)

    # Get files associated with experiment
    files = list_experiment_files(id)

    # Figure out which files are skeletons
    me_files = files[files.file_name.str.endswith('.glb')]

    if me_files.empty:
        raise ValueError('Did not find any meshes associated with '
                         f'experiment {id}')

    volumes = []
    for f in config.tqdm(me_files.itertuples(),
                         desc='Downloading',
                         total=me_files.shape[0]):
        # Load the file
        r = requests.get(f.url)
        r.raise_for_status()

        name = '.'.join(f.file_name.split('.')[:-1])
        ext = f.file_name.split('.')[-1]

        file = io.BytesIO(r.content)
        scene = tm.load(file, file_type=ext)

        for obj in scene.geometry.values():
            v = Volume(obj.vertices, obj.faces, name=name)
            volumes.append(v)

    logger.info(f'Done! Found {len(volumes)} meshes.')

    return volumes


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


def _get_download_url(uuid):
    """Get AWS download URL for given object."""
    url = f"https://www.insectbraindb.org/filestore/download_url/?uuid={uuid}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()['url']


session = Session()
