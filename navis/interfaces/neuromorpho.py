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

import os

import pandas as pd

from typing import List, Dict, Union, Optional

from ..core import Skeleton, NeuronList
from ..io import read_swc
from .. import utils, config
from .base import fetch_parallel, get_session


BASEURL = 'https://neuromorpho.org'

# In the past there were some issues with neuromorpho's SSL certificate
# This is not recommended but you can switch off verification here
VERIFY = str(os.environ.get('NAVIS_NEUROMORPHO_VERIFY', 'True')).lower() not in (
    'false', '0', 'no', 'off', ''
)


def find_neurons(page_limit: Optional[int] = None,
                 parallel: bool = True,
                 max_threads: int = 4,
                 errors=None,
                 **filters) -> pd.DataFrame:
    """Find neurons matching by given criteria.

    Parameters
    ----------
    page_limit :    int | None, optional
                    Use this to limit the results if you are running a big query.
                    Required if you pass no filters at all.
    parallel :      bool
                    If True, will use threads to fetch pages.
    max_threads :   int
                    Max number of parallel threads to use.
    errors :        "raise" | "log" | "ignore", optional
                    What to do if a page of results fails to fetch. Defaults to
                    "log", or to "raise" under `navis.config.strict`.
    **filters
                    Search criteria as `field=value`. See
                    [`navis.interfaces.neuromorpho.get_neuron_fields`][] and
                    [`navis.interfaces.neuromorpho.get_available_field_values`][]
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
    # An unfiltered search walks the entire NeuroMorpho index. That used to
    # prompt for confirmation, which a library must not do - there may well be
    # nobody at the other end. Make the caller say what they want instead.
    if not filters and page_limit is None:
        raise ValueError(
            "Searching without filters would list every neuron in NeuroMorpho. "
            "Either pass a filter (e.g. `species='rat'`) or bound the search "
            "with `page_limit=N`."
        )

    # Turn strings into lists
    filters = {k: list(utils.make_iterable(v)) for k, v in filters.items()}

    url = utils.make_url(BASEURL, 'api', 'neuron', 'select')

    # Load the first page to get the total number of pages
    first = _fetch_page(0, url=url, filters=filters)
    # `totalPages` is a count (pages 0 .. totalPages-1); we've already fetched
    # page 0 and fetch pages 1 .. page_limit-1 below
    total_pages = first['page']['totalPages']
    page_limit = total_pages if page_limit is None else min(page_limit, total_pages)

    data: List[str] = list(first['_embedded']['neuronResources'])

    pages = list(range(1, page_limit))
    rest = fetch_parallel(
        _fetch_page,
        pages,
        labels=[f'page {p}' for p in pages],
        errors=errors,
        parallel=parallel,
        max_threads=max_threads,
        url=url,
        filters=filters,
    )

    for page in rest:
        if page is not None:  # `None` = a page that failed; see `errors`
            data += page['_embedded']['neuronResources']

    return pd.DataFrame.from_records(data)


def _fetch_page(page: int, *, url: str, filters: dict) -> dict:
    """Fetch a single page of search results."""
    resp = get_session().post(f'{url}?page={page}', json=filters, verify=VERIFY)
    resp.raise_for_status()
    return resp.json()


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
        url = utils.make_url(BASEURL, 'api', 'neuron', 'name', x)
    elif isinstance(x, int):
        url = utils.make_url(BASEURL, 'api', 'neuron', 'id', str(x))
    else:
        raise TypeError(f'Expected string or int, got {type(x)}')

    resp = get_session().get(url, verify=VERIFY)

    resp.raise_for_status()

    return pd.Series(resp.json())


def get_neuron(x: Union[str, int, Dict[str, str]],
               parallel: bool = True,
               max_threads: int = 4,
               errors=None,
               **kwargs) -> Skeleton:
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
    errors :        "raise" | "log" | "ignore", optional
                    Only relevant when `x` is a DataFrame. What to do if an
                    individual neuron fails to fetch. Defaults to "log", or to
                    "raise" under `navis.config.strict`.
    **kwargs
                    Keyword arguments passed on to [`navis.read_swc`][].

    Returns
    -------
    Skeleton
                    Or a NeuronList if `x` is a DataFrame.

    Examples
    --------
    >>> import navis.interfaces.neuromorpho as nm
    >>> # Get a neuron by its ID
    >>> n = nm.get_neuron(1)
    >>> n
    type            Skeleton
    name                   SWC
    n_nodes               1274
    n_connectors             0
    n_branches              46
    n_leafs                 54
    cable_length       4792.21
    soma                  None

    """
    if isinstance(x, pd.DataFrame):
        records = x.to_dict(orient='records')

        # `fetch_parallel` hands results back in input order, so no re-sort here.
        nl = fetch_parallel(
            get_neuron,
            records,
            labels=[r.get('neuron_id', r.get('neuron_name', 'NA')) for r in records],
            errors=errors,
            parallel=parallel,
            max_threads=max_threads,
            **kwargs,
        )

        return NeuronList([n for n in nl if n is not None])

    if not isinstance(x, (pd.Series, dict)):
        info = get_neuron_info(x)
    else:
        info = x  # type: ignore

    archive: str = info['archive']
    name: str = info['neuron_name']

    url = utils.make_url(BASEURL, 'dableFiles', archive.lower(), 'CNG version', name + '.CNG.swc')
    r = get_session().get(url, verify=VERIFY)
    r.raise_for_status()

    n = read_swc(r.content.decode(), **kwargs)

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
    url = utils.make_url(BASEURL, 'api', 'neuron', 'fields')
    resp = get_session().get(url, verify=VERIFY)

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
            url = utils.make_url(BASEURL, 'api', 'neuron', 'fields', field, page=page)

            resp = get_session().get(url, verify=VERIFY)

            resp.raise_for_status()

            content = resp.json()

            data += content['fields']

            # Pages are 0-indexed and `totalPages` is a count, so the last
            # valid page is totalPages - 1
            if page >= content['page']['totalPages'] - 1:
                break

            pbar.total = content['page']['totalPages']
            pbar.update(1)

            page += 1

    return data
