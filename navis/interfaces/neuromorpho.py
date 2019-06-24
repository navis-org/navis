#    This script is part of navis (http://www.github.com/schlegelp/navis).
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

""" Set of functions to interface with the neuromorpho.org database of
neurons.

See http://neuromorpho.org/apiReference.html for documentation.
"""

import requests
import urllib

import pandas as pd

from tqdm import tqdm
from typing import List, Dict, Union

from ..core import TreeNeuron, NeuronList
from ..io.swc_io import from_swc
from .. import utils


baseurl = 'http://neuromorpho.org'


def list_neurons(**filters) -> pd.DataFrame:
    """ List neurons filtered by given criteria
    """

    if not filters:
        answer = ""
        while answer not in ["y", "n"]:
            answer = input("No filters will list all neurons. Continue? [Y/N] ").lower()

        if answer != 'y':
            return  # type: ignore

    # Turn lists into strings
    filters = {k: ','.join(v) if isinstance(v, list) else v for k, v in filters.items()}

    # Turn filters into str
    fstring = '&'.join([f'{k}:{v}' for k, v in filters.items()])

    url = utils.make_url(baseurl, 'neuron', 'select', q=str(filters))


def get_neuron_info(x: Union[str, int]) -> pd.Series:
    """ Fetch neuron info by ID or by name.


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


def get_neuron(x: Union[str, int, Dict[str, str]], **kwargs) -> TreeNeuron:
    """ Fetch neuron by ID or by name.


    Parameters
    ----------
    x :         int | str | dict
                Integer is intepreted as ID, string as neuron name. Dictionary
                must contain 'archive' (e.g. "Wearne_Hof") and 'neuron_name'
                (e.g. "cnic_001").
    **kwargs
                Keyword arguments passed on to :func:`navis.from_swc`.

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

    if not isinstance(x, (pd.Series, dict)):
        info = get_neuron_info(x)
    else:
        info = x  # type: ignore

    archive: str = info['archive']
    name: str = info['neuron_name']

    url = utils.make_url(baseurl, 'dableFiles', archive.lower(), 'CNG version', name + '.CNG.swc')

    return from_swc(url, **kwargs)


def get_neuron_fields() -> Dict[str, List[str]]:
    """ Returns a list of neuron fields avaialble.

    Examples
    --------
    >>> import navis.interfaces.neuromorpho as nm
    >>> fields = nm.get_neuron_fields()

    """

    url = utils.make_url(baseurl, 'api', 'neuron', 'fields')
    resp = requests.get(url)

    resp.raise_for_status()

    return resp.json().get('Neuron Fields')


def get_available_field_values(field: str) -> List[str]:
    """ Returns a list of values present in the repository for the neuron
    field requested.

    Parameters
    ----------
    field :     str
                Field to search for

    Examples
    --------
    >>> import navis.interfaces.neuromorpho as nm
    >>> # Get availalbe values for "species" field
    >>> species = nm.get_available_field_values('species')
    """

    data: List[str] = []
    page = 1

    with tqdm(total=1, desc='Fetching') as pbar:
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
