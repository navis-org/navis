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

"""Set of functions to interface with the neuromorpho.org database of neurons.

See http://neuromorpho.org/apiReference.html for documentation.
"""

import requests

import pandas as pd
import numpy as np

from typing import List, Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core import TreeNeuron, NeuronList
from ..io import read_swc
from .. import utils, config


baseurl = 'http://neuromorpho.org'


def find_neurons(page_limit: Optional[int] = None,
                 parallel: bool = True,
                 max_threads: int = 4,
                 **filters) -> pd.DataFrame:
    """Find neurons matching by given criteria.

    Parameters
    ----------
    page_limit :    int | None, optional
                    Use this to limit the results if you are running a big query.
    **filters
                    Search criteria as ``field=value``. See
                    :func:`navis.interfaces.neuromorpho.get_neuron_fields` and
                    :func:`navis.interfaces.neuromorpho.get_available_field_values`
                    for available fields and values.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> import navis.interfaces.neuromorpho as nm
    >>> rat_neurons = nm.find_neurons(species='rat')
    >>> rat_or_mouse = nm.find_neurons(species=['rat', 'mouse'])

    """
    if not filters:
        answer = ""
        while answer not in ["y", "n"]:
            answer = input("No filters will list all neurons. Continue? [Y/N] ").lower()

        if answer != 'y':
            return  # type: ignore

    # Turn strings into lists
    filters = {k: list(utils.make_iterable(v)) for k, v in filters.items()}

    url = utils.make_url(baseurl, 'api', 'neuron', 'select')

    if isinstance(page_limit, type(None)):
        page_limit = float('inf')

    data: List[str] = []

    # Load the first page to get the total number of pages
    resp = requests.post(f'{url}?page=0', json=filters)
    content = resp.json()
    total_pages = content['page']['totalPages'] - 1
    page_limit = min(page_limit, total_pages)
    data += content['_embedded']['neuronResources']

    page = 1   # start with 1 because we already have 0

    with ThreadPoolExecutor(max_workers=1 if not parallel else max_threads) as executor:
        futures = {}
        while page < page_limit:
            f = executor.submit(requests.post, f'{url}?page={page}', json=filters)
            futures[f] = page
            page += 1

        with config.tqdm(desc='Fetching',
                         total=len(futures) + 1,
                         leave=config.pbar_leave,
                         disable=len(futures) == 1 or config.pbar_hide) as pbar:
            pbar.update(1)  # for the first page fetched
            for f in as_completed(futures):
                pbar.update(1)
                try:
                    resp = f.result()
                    resp.raise_for_status()
                    data += resp.json()['_embedded']['neuronResources']
                except Exception as exc:
                    print(f'Page {futures[f]} generated an exception:', exc)

    return pd.DataFrame.from_records(data)


def get_neuron_info(x: Union[str, int]) -> pd.Series:
    """Fetch neuron info by ID or by name.

    Parameters
    ----------
    x :         int | str
                Integer is intepreted as ID, string as neuron name. Will try
                to convert strings to integers first.

    Examples
    --------
    >>> import navis.interfaces.neuromorpho as nm
    >>> # Get info by ID
    >>> info = nm.get_neuron_info(1)
    >>> # Get info by Name
    >>> info = nm.get_neuron_info('cnic_001')

    """
    try:
        x = int(x)
    except BaseException:
        pass

    if isinstance(x, str):
        url = utils.make_url(baseurl, 'api', 'neuron', 'name', x)
    elif isinstance(x, int):
        url = utils.make_url(baseurl, 'api', 'neuron', 'id', str(x))
    else:
        raise TypeError(f'Expected string or int, got {type(x)}')

    resp = requests.get(url)

    resp.raise_for_status()

    return pd.Series(resp.json())


def get_neuron(x: Union[str, int, Dict[str, str]],
               parallel: bool = True,
               max_threads: int = 4,
               **kwargs) -> TreeNeuron:
    """Fetch neuron by ID or by name.

    Parameters
    ----------
    x :             int | str | dict | pandas.DataFrame
                    Integer is intepreted as ID, string as neuron name. Dictionary
                    and DataFrame must contain 'archive' (e.g. "Wearne_Hof") and
                    'neuron_name' (e.g. "cnic_001").
    parallel :      bool
                    If True, will use threads to fetch data.
    max_threads :   int
                    Max number of parallel threads to use.
    **kwargs
                    Keyword arguments passed on to :func:`navis.read_swc`.

    Returns
    -------
    TreeNeuron

    Examples
    --------
    >>> import navis.interfaces.neuromorpho as nm
    >>> # Get a neuron by its ID
    >>> n = nm.get_neuron(1)
    >>> n
    type            TreeNeuron
    name                   SWC
    n_nodes               1274
    n_connectors             0
    n_branches              46
    n_leafs                 54
    cable_length       4792.21
    soma                  None

    """
    if isinstance(x, pd.DataFrame):
        nl = []
        with ThreadPoolExecutor(max_workers=1 if not parallel else max_threads) as executor:
            futures = {}
            for r in x.to_dict(orient='records'):
                f = executor.submit(get_neuron, r, **kwargs)
                futures[f] = r.get('neuron_id', r.get('neuron_name', 'NA'))

            with config.tqdm(desc='Fetching',
                             total=len(x),
                             leave=config.pbar_leave,
                             disable=len(x) == 1 or config.pbar_hide) as pbar:
                for f in as_completed(futures):
                    id = futures[f]
                    pbar.update(1)
                    try:
                        nl.append(f.result())
                    except Exception as exc:
                        print(f'{id} generated an exception:', exc)

        # Turn into neuronlist
        nl = NeuronList(nl)

        # Make sure we return in same order as input
        if 'neuron_id' in x.columns:
            ids = x.neuron_id.values
            ids = ids[np.isin(ids, nl.id)]  # drop failed IDs
            nl = nl.idx[ids]

        return nl

    if not isinstance(x, (pd.Series, dict)):
        info = get_neuron_info(x)
    else:
        info = x  # type: ignore

    archive: str = info['archive']
    name: str = info['neuron_name']

    url = utils.make_url(baseurl, 'dableFiles', archive.lower(), 'CNG version', name + '.CNG.swc')

    n = read_swc(url, **kwargs)

    n.id = info.get('neuron_id', n.id)
    n.name = info.get('neuron_name', getattr(n, 'name'))

    return n


def get_neuron_fields() -> Dict[str, List[str]]:
    """List all available neuron fields.

    Examples
    --------
    >>> import navis.interfaces.neuromorpho as nm
    >>> fields = nm.get_neuron_fields()
    >>> fields
    ['neuron_id',
     'neuron_name',
     'archive',
     'age_scale',
     ...

    """
    url = utils.make_url(baseurl, 'api', 'neuron', 'fields')
    resp = requests.get(url)

    resp.raise_for_status()

    return resp.json().get('Neuron Fields')


def get_available_field_values(field: str) -> List[str]:
    """List all possible values for given neuron field.

    Parameters
    ----------
    field :     str
                Field to search for.

    Examples
    --------
    >>> import navis.interfaces.neuromorpho as nm
    >>> # Get availalbe values for "species" field
    >>> species = nm.get_available_field_values('species')
    >>> species
    ['rat',
     'mouse',
     'drosophila melanogaster',
     'human',
     'monkey',
     ...

    """
    data: List[str] = []
    page = 0

    with config.tqdm(total=1,
                     disable=config.pbar_hide,
                     leave=config.pbar_leave,
                     desc='Fetching') as pbar:
        while True:
            url = utils.make_url(baseurl, 'api', 'neuron', 'fields', field, page=page)

            resp = requests.get(url)

            resp.raise_for_status()

            content = resp.json()

            data += content['fields']

            if page == content['page']['totalPages']:
                break

            pbar.total = content['page']['totalPages']
            pbar.update(1)

            page += 1

    return data
