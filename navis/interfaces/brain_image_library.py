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

"""Set of functions to interface with the Brain Image Library (BIL).

BIL (https://www.brainimagelibrary.org) is a public repository for brain
imaging data hosted at the Pittsburgh Supercomputing Center. Besides (very
large) imagery it also hosts thousands of single neuron reconstructions.

See https://www.brainimagelibrary.org/metadataapi.html for the metadata API
and https://www.brainimagelibrary.org/download.html for download options.

Note that BIL datasets can be *huge* (up to hundreds of terabytes). The
functions in this module therefore have guardrails that will stop you from
accidentally crawling or downloading such datasets. For bulk transfers use
Globus instead (see the download docs linked above).
"""

import os
import re
import requests

import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from urllib.parse import unquote, urljoin
from urllib3.util import Retry

from ..core import NeuronList
from ..io import read_swc
from .. import utils, config

__all__ = [
    "search",
    "query",
    "get_metadata",
    "get_dataset_url",
    "list_files",
    "download_files",
    "get_neurons",
    "clear_metadata_cache",
]

logger = config.get_logger(__name__)

BASEURL = "https://api.brainimagelibrary.org"
DOWNLOAD_URL = "https://download.brainimagelibrary.org"

# All BIL directories are of the form "/bil/data/<c1c2>/<c3c4>/<submission id>"
BIL_PREFIX = "/bil/data/"

HEADERS = requests.utils.default_headers()
HEADERS.update({"User-Agent": "github.com/navis-org/navis"})

TIMEOUT = 60

# Guardrails. These are module-level so that power users can adjust them.
# `list_files` refuses to crawl datasets larger than this without `force=True`:
SIZE_WARN_GB = 100
FILE_WARN_COUNT = 5_000
# `download_files` refuses to fetch more than this without raising `max_size`:
DEFAULT_MAX_DOWNLOAD = "10G"

_HELP_GLOBUS = (
    "For bulk transfers use Globus instead: "
    "https://www.brainimagelibrary.org/download.html"
)


class BILError(Exception):
    """Raised when the Brain Image Library API returns an error."""


def _make_session() -> requests.Session:
    """Session with retries. BIL is a public service - be a good citizen."""
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry, pool_maxsize=16))
    return s


# Note: creating a session does not open any connections, so importing this
# module stays free of side effects (and works offline).
session = _make_session()


#############################################
#                                           #
#              Search / query               #
#                                           #
#############################################

# Maps a metadata element to the division (i.e. API endpoint) it lives under.
# The API only accepts a single element per request, hence `search` fans out
# and intersects the results. Note that quite a few of these are undocumented
# but verified to work.
FIELDS: Dict[str, str] = {
    "text": "fulltext",
    "title": "dataset",
    "generalmodality": "dataset",
    "technique": "dataset",
    "localid": "specimen",
    "species": "specimen",
    "genotype": "specimen",
    "sex": "specimen",
    "doi": "submission",
    "method": "submission",
    "project": "submission",
    "consortium": "submission",
    "submission_uuid": "submission",
    "microscopetype": "instrument",
    "objectivename": "instrument",
    "creator": "contributors",
    "contributorname": "contributors",
    "contributortype": "contributors",
    "affiliation": "contributors",
    "affiliationidentifier": "contributors",
    "award_number": "funders",
    "relatedidentifier": "publication",
    "gbytes": "image",
    "xid": "specimen/linkage",
    "type": "specimen/linkage",
    "relationship": "specimen/linkage",
    "class": "specimen/linkage",
}


def _check_fields(fields: Iterable[str]) -> None:
    """Validate search fields *before* we hit the network."""
    unknown = [f for f in fields if f not in FIELDS]
    if not unknown:
        return

    by_div: Dict[str, List[str]] = {}
    for el, div in sorted(FIELDS.items()):
        by_div.setdefault(div, []).append(el)
    avail = "\n".join(f"  {d}: {', '.join(e)}" for d, e in sorted(by_div.items()))

    raise ValueError(
        f"Unknown search field(s): {', '.join(unknown)}.\n"
        f"Available fields (grouped by division):\n{avail}\n"
        "For other (undocumented) division/element combinations use "
        "`brain_image_library.query()`."
    )


def _get_json(url: str) -> dict:
    """GET and parse a BIL API response.

    BIL signals *both* "no results" and "invalid field" with a HTTP 404 and a
    JSON body, so we must inspect the message rather than the status code.
    """
    r = session.get(url, timeout=TIMEOUT)

    try:
        data = r.json()
    except ValueError:
        # Not JSON at all - e.g. an unknown division returns an HTML 404 page
        if r.status_code == 404:
            raise BILError(
                f"BIL API returned 404 for {url}. This usually means the "
                "division does not exist. See `brain_image_library.FIELDS` "
                "for valid divisions."
            )
        r.raise_for_status()
        raise BILError(f"BIL API returned a non-JSON response for {url}: {r.text[:200]}")

    # N.B. `success` is the *string* "true"/"false". Do not be tempted to write
    # `if not data['success']` - `bool("false")` is True and the check would
    # silently never fire.
    if str(data.get("success", "true")).lower() != "false":
        return data

    msg = str(data.get("message", ""))

    # An empty result set is not an error - it comes back as a 404 too.
    if "no entry found" in msg.lower():
        return data

    raise BILError(f"BIL API error for {url}: {msg}")


def query(division: str, element: str, value: str) -> List[str]:
    """Run a single raw query against the BIL metadata API.

    This is a low-level escape hatch for division/element combinations not
    covered by [`navis.interfaces.brain_image_library.search`][]. No validation
    is performed on `division`/`element`.

    Parameters
    ----------
    division :  str
                The metadata division, e.g. "dataset" or "specimen".
    element :   str
                The metadata element to search, e.g. "generalmodality".
    value :     str
                The value to search for. Note that BIL matches values
                *exactly* - there is no substring or fuzzy matching.

    Returns
    -------
    list of str
                The IDs ("bildids") of matching datasets. Empty if no match.

    Examples
    --------
    >>> import navis.interfaces.brain_image_library as bil
    >>> ids = bil.query('dataset', 'generalmodality', 'cell morphology')

    """
    url = utils.make_url(BASEURL, "query", *str(division).split("/"), **{element: value})
    return list(_get_json(url).get("bildids", []))


def search(
    *,
    limit: Optional[int] = None,
    metadata: bool = True,
    parallel: bool = True,
    max_threads: int = 4,
    **filters: Union[str, Iterable[str]],
) -> pd.DataFrame:
    """Search BIL for datasets matching the given criteria.

    Parameters
    ----------
    limit :         int, optional
                    Cap the number of datasets returned.
    metadata :      bool
                    If True (default), fetch full metadata for the hits and
                    return it as a DataFrame. If False, return just the IDs -
                    much faster if you only need the IDs.
    parallel :      bool
                    Whether to run the individual queries in parallel.
    max_threads :   int
                    Max number of parallel threads to use.
    **filters
                    Search criteria as `field=value`. See
                    [`navis.interfaces.brain_image_library.FIELDS`][] for the
                    available fields.

                    Note the semantics:

                    - across fields the filters are combined with **AND**
                    - within a field, multiple values are combined with **OR**
                      (e.g. `species=['mouse', 'rat']`)

                    Values must match **exactly** - BIL does no substring or
                    fuzzy matching. Use `text=...` for a full-text search.

    Returns
    -------
    pandas.DataFrame
                    One row per dataset. Feed this straight into
                    [`navis.interfaces.brain_image_library.get_neurons`][] or
                    [`navis.interfaces.brain_image_library.list_files`][].

    Examples
    --------
    >>> import navis.interfaces.brain_image_library as bil
    >>> # Find mouse single-neuron reconstructions
    >>> ds = bil.search(species='mouse', generalmodality='cell morphology')
    >>> # Full-text search
    >>> ds = bil.search(text='barrel cortex')

    Note that `class` is a Python keyword and hence can't be used as a keyword
    argument. Pass it as a dict instead:

    >>> ds = bil.search(**{'class': 'somevalue'})                # doctest: +SKIP

    """
    if not filters:
        raise ValueError(
            "`search` requires at least one filter, e.g. `search(species='mouse')`. "
            "See `brain_image_library.FIELDS` for available fields."
        )

    _check_fields(filters)

    # Flatten to a list of (division, element, value) jobs
    jobs = [
        (FIELDS[field], field, value)
        for field, values in filters.items()
        for value in utils.make_iterable(values)
    ]

    # OR within a field, AND across fields
    per_field: Dict[str, set] = {f: set() for f in filters}
    with ThreadPoolExecutor(max_workers=1 if not parallel else max_threads) as executor:
        futures = {executor.submit(query, div, el, val): el for div, el, val in jobs}
        with config.tqdm(
            desc="Querying",
            total=len(futures),
            leave=config.pbar_leave,
            disable=len(futures) == 1 or config.pbar_hide,
        ) as pbar:
            for f in as_completed(futures):
                pbar.update(1)
                per_field[futures[f]] |= set(f.result())

    ids = sorted(set.intersection(*per_field.values()))

    if not ids:
        logger.warning(
            "No datasets matched. Note that BIL matches values *exactly* - "
            "check spelling and capitalisation, or try a full-text search "
            "via `search(text=...)`."
        )
        return pd.DataFrame(columns=["bildid"])

    if limit:
        ids = ids[:limit]

    if not metadata:
        return pd.DataFrame({"bildid": ids})

    return get_metadata(ids, parallel=parallel, max_threads=max_threads)


#############################################
#                                           #
#                 Metadata                  #
#                                           #
#############################################

# In-process cache: {bildid: raw record}. Not persisted to disk.
_META_CACHE: Dict[str, dict] = {}


def clear_metadata_cache() -> None:
    """Clear the in-process metadata cache."""
    _META_CACHE.clear()


def _collapse(items: List[dict], key: str):
    """Collapse a list-of-dicts to a scalar if unanimous, else a tuple.

    This keeps the common case (e.g. 40 specimens that are all "mouse") a
    plain scalar so that `df.species == 'mouse'` still works, while never
    silently dropping information.
    """
    values = [i.get(key) for i in items if i.get(key) not in (None, "")]
    unique = list(dict.fromkeys(values))  # order-preserving unique
    if not unique:
        return None
    return unique[0] if len(unique) == 1 else tuple(unique)


def _to_number(value, dtype):
    """Cast to `dtype`, falling back to NaN. BIL returns numbers as strings."""
    try:
        return dtype(value)
    except (TypeError, ValueError):
        return np.nan


def _flatten_record(rec: dict) -> dict:
    """Flatten a nested BIL metadata record into a single flat row.

    `Dataset`, `Specimen` and `Image` are lists in the raw record. We keep one
    row per dataset (rather than exploding) and add `n_*` counts so that a
    length > 1 is visible rather than silent.

    Note that every division may be missing or empty on any given record, so
    this must never raise a KeyError.
    """
    dataset = (rec.get("Dataset") or [{}])
    specimens = rec.get("Specimen") or []
    images = rec.get("Image") or []
    instruments = rec.get("Instrument") or []
    contributors = rec.get("Contributors") or []
    funders = rec.get("Funders") or []
    publications = rec.get("Publication") or []
    assets = rec.get("Assets") or []
    submission = rec.get("Submission") or {}

    ds = dataset[0] if dataset else {}
    img = images[0] if images else {}
    inst = instruments[0] if instruments else {}

    bildirectory = ds.get("bildirectory")
    try:
        url = _bildirectory_to_url(bildirectory) if bildirectory else None
    except ValueError:
        url = None

    row = {
        # The record itself carries the ID only inside `Assets`.
        "bildid": assets[0].get("bildid") if assets else None,
        "title": ds.get("title"),
        "doi": ds.get("doi"),
        "generalmodality": ds.get("generalmodality"),
        "technique": ds.get("technique"),
        "species": _collapse(specimens, "species"),
        "genotype": _collapse(specimens, "genotype"),
        "sex": _collapse(specimens, "sex"),
        "organname": _collapse(specimens, "organname"),
        "age": _collapse(specimens, "age"),
        "url": url,
        "dataset_size_gb": _to_number(ds.get("dataset_size"), float),
        "number_of_files": _to_number(ds.get("number_of_files"), int),
        "n_datasets": len(dataset),
        "n_specimens": len(specimens),
        "n_images": len(images),
        "rights": ds.get("rights"),
    }

    # Everything else is namespaced to avoid collisions - note that `doi`
    # exists on both `Dataset` and `Submission`.
    for key in ("abstract", "methods", "technicalinfo", "bildirectory", "rightsuri"):
        row[f"dataset.{key}"] = ds.get(key)

    for key in ("project", "consortium", "doi", "method", "submission_uuid", "bildate"):
        row[f"submission.{key}"] = submission.get(key)

    for key in (
        "xsize", "ysize", "zsize", "stepsizex", "stepsizey", "stepsizez",
        "channels", "dimensionorder", "xaxis", "yaxis", "zaxis",
    ):
        row[f"image.{key}"] = img.get(key)

    for key in ("microscopetype", "objectivename"):
        row[f"instrument.{key}"] = inst.get(key)

    row["contributors.names"] = [c.get("contributorname") for c in contributors]
    row["funders.award_numbers"] = [f.get("award_number") for f in funders]
    row["publication.relatedidentifiers"] = [
        p.get("relatedidentifier") for p in publications
    ]
    row["tags"] = rec.get("Tags")

    return row


def _retrieve(bildids: List[str], parallel: bool, max_threads: int, chunk_size: int) -> None:
    """Fetch records for `bildids` into `_META_CACHE`."""
    missing = [i for i in bildids if i not in _META_CACHE]
    if not missing:
        return

    chunks = [missing[i:i + chunk_size] for i in range(0, len(missing), chunk_size)]

    def _post(chunk):
        r = session.post(
            utils.make_url(BASEURL, "retrieve"), json={"bildids": chunk}, timeout=TIMEOUT
        )
        try:
            data = r.json()
        except ValueError:
            r.raise_for_status()
            raise BILError(f"BIL API returned a non-JSON response: {r.text[:200]}")
        return data

    with ThreadPoolExecutor(max_workers=1 if not parallel else max_threads) as executor:
        futures = {executor.submit(_post, c): c for c in chunks}
        with config.tqdm(
            desc="Fetching metadata",
            total=len(futures),
            leave=config.pbar_leave,
            disable=len(futures) == 1 or config.pbar_hide,
        ) as pbar:
            for f in as_completed(futures):
                pbar.update(1)
                data = f.result()

                for bildid, ok in (data.get("status") or {}).items():
                    if not ok:
                        logger.warning(f"No metadata found for '{bildid}'.")

                # IMPORTANT: the order of `retjson` does NOT match the order of
                # the request - we must key records by their own ID.
                for rec in data.get("retjson") or []:
                    assets = rec.get("Assets") or []
                    bildid = assets[0].get("bildid") if assets else None
                    if not bildid:
                        logger.warning("Skipping a record without an identifiable ID.")
                        continue
                    _META_CACHE[bildid] = rec


def _extract_ids(x) -> List[str]:
    """Coerce `x` (ID, list of IDs or DataFrame) into a list of bildids."""
    if isinstance(x, pd.DataFrame):
        if "bildid" not in x.columns:
            raise ValueError(
                "DataFrame must have a 'bildid' column. Did you mean to pass "
                "the output of `search()` or `get_metadata()`?"
            )
        return list(pd.unique(x.bildid.dropna()))
    if isinstance(x, pd.Series):
        if "bildid" in x.index:
            return [x["bildid"]]
        return list(pd.unique(x.dropna()))
    return [str(i) for i in utils.make_iterable(x)]


def get_metadata(
    x,
    *,
    raw: bool = False,
    parallel: bool = True,
    max_threads: int = 4,
    chunk_size: int = 100,
) -> Union[pd.DataFrame, List[dict]]:
    """Fetch metadata for one or more BIL datasets.

    Parameters
    ----------
    x :             str | list of str | pandas.DataFrame
                    The dataset ID(s) ("bildid"). Can also be a DataFrame with
                    a `bildid` column, e.g. the output of
                    [`navis.interfaces.brain_image_library.search`][].
    raw :           bool
                    If True, return the raw (nested) JSON records instead of a
                    DataFrame. Use this to get at the `Assets`, `Contributors`
                    and `Publication` divisions which do not survive
                    flattening intact.
    parallel :      bool
                    Whether to fetch in parallel.
    max_threads :   int
                    Max number of parallel threads to use.
    chunk_size :    int
                    Number of datasets to request per call.

    Returns
    -------
    pandas.DataFrame
                    One row per dataset. `Specimen` entries are collapsed: a
                    field is a scalar if all specimens agree and a tuple
                    otherwise. `n_specimens` always tells you how many there
                    actually were.
    list of dict
                    If `raw=True`.

    Examples
    --------
    >>> import navis.interfaces.brain_image_library as bil
    >>> meta = bil.get_metadata('ace-boo-van')
    >>> meta.title.values[0]
    'Single neuron reconstruction from fMOST images'

    """
    ids = _extract_ids(x)
    if not ids:
        raise ValueError("Got no dataset IDs to fetch metadata for.")

    _retrieve(ids, parallel=parallel, max_threads=max_threads, chunk_size=chunk_size)

    records = [_META_CACHE[i] for i in ids if i in _META_CACHE]

    if not records:
        raise BILError(f"No metadata found for: {', '.join(ids)}")

    if raw:
        return records

    return pd.DataFrame.from_records([_flatten_record(r) for r in records])


#############################################
#                                           #
#              Files / download             #
#                                           #
#############################################


def _bildirectory_to_url(bildirectory: str) -> str:
    """Turn a BIL directory into its https download URL.

    Note we always enforce a trailing slash: without it `urljoin` would drop
    the last path segment when we crawl.
    """
    path = str(bildirectory).strip()

    if BIL_PREFIX not in path:
        raise ValueError(
            f'Unexpected BIL directory "{bildirectory}": expected it to '
            f'contain "{BIL_PREFIX}".'
        )

    # Everything after the prefix is kept verbatim - this transparently handles
    # directories with an extra sub-directory such as
    # "/bil/data/cd/db/cddb33924966b07f/735467"
    rel = path.split(BIL_PREFIX, 1)[1].strip("/")

    if not rel:
        raise ValueError(f'BIL directory "{bildirectory}" is empty.')

    return f"{DOWNLOAD_URL}/{rel}/"


def get_dataset_url(x) -> Union[str, List[str]]:
    """Return the download URL(s) for the given dataset(s).

    Parameters
    ----------
    x :         str | list of str | pandas.DataFrame
                Dataset ID(s) ("bildid").

    Returns
    -------
    str
                If a single dataset was requested.
    list of str
                If multiple datasets were requested.

    Examples
    --------
    >>> import navis.interfaces.brain_image_library as bil
    >>> bil.get_dataset_url('ace-boo-van')
    'https://download.brainimagelibrary.org/22/5c/225c37cacfbd897c/'

    """
    ids = _extract_ids(x)
    meta = get_metadata(ids)

    urls = []
    for bildid, url in zip(meta.bildid, meta.url):
        if not url:
            raise BILError(
                f"Dataset '{bildid}' has no public download directory "
                "(it may be embargoed or not yet released)."
            )
        urls.append(url)

    return urls[0] if len(ids) == 1 else urls


_SIZE_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}

# The date/size in an nginx autoindex listing sit in the text node *after* the
# closing </a>, e.g.:
#   <a href="foo.swc">foo.swc</a>        31-Mar-2022 15:12        5245516
_META_RE = re.compile(
    r"\s*(?P<date>\d{2}-\w{3}-\d{4}\s+\d{2}:\d{2})?"
    r"\s*(?P<size>[\d.]+[KMGTP]?|-)?\s*\Z"
)


def _parse_size(size) -> Optional[int]:
    """Parse a size into bytes.

    Handles both nginx autoindex modes (rounded "3.4M" and exact "5245516")
    as well as the `max_size` arguments of this module (e.g. "10G").
    """
    if size is None:
        return None

    s = str(size).strip().upper()
    if not s or s == "-":
        return None

    if s.endswith("B"):  # "10GB" -> "10G"
        s = s[:-1]
    if not s:
        return None

    unit = s[-1] if s[-1] in "KMGTP" else ""
    number = s[:-1] if unit else s

    try:
        return int(float(number) * _SIZE_UNITS[unit])
    except ValueError:
        return None


class _AutoIndexParser(HTMLParser):
    """Parse an nginx autoindex directory listing."""

    def __init__(self):
        super().__init__()
        self.entries: List[dict] = []
        self._in_link = False
        self._expect_meta = False

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return

        self._expect_meta = False
        href = dict(attrs).get("href")

        # Skip the parent link and nginx's column-sort links - crawling the
        # latter would send us round in circles.
        if (
            not href
            or href in ("../", "./", "/")
            or href.startswith(("?", "#", "http://", "https://"))
        ):
            self._in_link = False
            return

        self.entries.append({"href": href, "date": None, "size": None})
        self._in_link = True

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link:
            self._in_link = False
            self._expect_meta = True

    def handle_data(self, data):
        if not self._expect_meta:
            return
        self._expect_meta = False

        match = _META_RE.match(data)
        if match and self.entries:
            self.entries[-1]["date"] = match.group("date")
            self.entries[-1]["size"] = match.group("size")


def _parse_autoindex(html: str, base_url: str) -> List[dict]:
    """Parse an nginx autoindex listing into a list of entries.

    Parameters
    ----------
    html :      str
                The raw HTML.
    base_url :  str
                URL of the directory. Must end in a "/".

    Returns
    -------
    list of dict
                With keys `name`, `url`, `is_dir`, `size` and `last_modified`.

    """
    parser = _AutoIndexParser()
    parser.feed(html)

    entries = []
    for entry in parser.entries:
        href = entry["href"]
        is_dir = href.endswith("/")
        entries.append(
            {
                "name": unquote(href).rstrip("/"),
                "url": urljoin(base_url, href),
                "is_dir": is_dir,
                "size": None if is_dir else _parse_size(entry["size"]),
                "last_modified": entry["date"],
            }
        )
    return entries


def _fetch_listing(url: str) -> str:
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def list_files(
    x,
    pattern: Optional[str] = None,
    *,
    recursive: bool = True,
    max_depth: int = 3,
    max_files: int = 10_000,
    max_requests: int = 500,
    force: bool = False,
    parallel: bool = True,
    max_threads: int = 4,
) -> pd.DataFrame:
    """List the files in a BIL dataset.

    This works by crawling the dataset's directory listing. Note that BIL
    hosts datasets with over a million files - see the guardrails below.

    Parameters
    ----------
    x :             str | list of str | pandas.DataFrame
                    Dataset ID(s) ("bildid"), or a DataFrame with a `bildid`
                    column. Note that BIL often splits a collection into one
                    small dataset per cell, so listing several at once is a
                    perfectly normal thing to do.
    pattern :       str, optional
                    Only return files matching this glob, e.g. `"*.swc"`.
                    Note this filters the *results*: we still have to visit
                    every directory to see what is in it, so a pattern does
                    not make crawling a huge dataset any cheaper.
    recursive :     bool
                    Whether to descend into subdirectories.
    max_depth :     int
                    How deep to descend. BIL layouts are shallow.
    max_files :     int
                    Stop after this many files (per dataset).
    max_requests :  int
                    Stop after this many directory listings (per dataset).
    force :         bool
                    BIL hosts datasets of hundreds of terabytes and millions of
                    files. By default we refuse to crawl anything larger than
                    `SIZE_WARN_GB`/`FILE_WARN_COUNT`. Set True to override.
    parallel :      bool
                    Whether to crawl in parallel.
    max_threads :   int
                    Max number of parallel threads to use.

    Returns
    -------
    pandas.DataFrame
                    Columns: `name`, `url`, `size` (bytes), `last_modified`,
                    `directory`, `depth` and `bildid`. Datasets are returned in
                    the order they were requested. Feed this straight into
                    [`navis.interfaces.brain_image_library.get_neurons`][] or
                    [`navis.interfaces.brain_image_library.download_files`][].

    Examples
    --------
    >>> import navis.interfaces.brain_image_library as bil
    >>> files = bil.list_files('ace-boo-van', pattern='*.swc')

    """
    ids = _extract_ids(x)
    if not ids:
        raise ValueError("Got no dataset IDs to list files for.")

    # Fetch metadata for all datasets up front (one round-trip) so that the
    # per-dataset crawls below hit the cache.
    get_metadata(ids, parallel=parallel, max_threads=max_threads)

    frames = []
    with config.tqdm(
        desc="Listing datasets",
        total=len(ids),
        leave=config.pbar_leave,
        disable=len(ids) == 1 or config.pbar_hide,
    ) as pbar:
        for bildid in ids:
            frames.append(
                _list_files_single(
                    bildid,
                    pattern=pattern,
                    recursive=recursive,
                    max_depth=max_depth,
                    max_files=max_files,
                    max_requests=max_requests,
                    force=force,
                    parallel=parallel,
                    max_threads=max_threads,
                )
            )
            pbar.update(1)

    return pd.concat(frames, ignore_index=True)


def _list_files_single(
    bildid: str,
    *,
    pattern,
    recursive,
    max_depth,
    max_files,
    max_requests,
    force,
    parallel,
    max_threads,
) -> pd.DataFrame:
    """Crawl a single dataset. See `list_files` for the parameters."""
    meta = get_metadata(bildid)
    n_files = meta.number_of_files.values[0]
    size_gb = meta.dataset_size_gb.values[0]

    # Guardrail. Note the metadata gives us the size *before* we touch the
    # download server, so this check is free.
    if not force and (
        (not np.isnan(n_files) and n_files > FILE_WARN_COUNT)
        or (not np.isnan(size_gb) and size_gb > SIZE_WARN_GB)
    ):
        raise ValueError(
            f"Dataset '{bildid}' contains {n_files:,.0f} files "
            f"({utils.sizeof_fmt(size_gb * 1024 ** 3)}). Crawling it would take "
            "a very long time and put unnecessary load on the BIL servers.\n"
            "Your options:\n"
            "  - only look at the top level: `list_files(x, recursive=False)`\n"
            "  - restrict the crawl with a smaller `max_depth`\n"
            "  - override this check with `force=True`\n"
            f"  - {_HELP_GLOBUS}"
        )

    root = get_dataset_url(bildid)

    queue = [(root, 0)]
    visited = {root}
    rows: List[dict] = []
    n_requests = 0
    truncated = None
    failed: List[str] = []

    with config.tqdm(
        desc="Crawling", leave=config.pbar_leave, disable=config.pbar_hide
    ) as pbar:
        while queue:
            if n_requests >= max_requests:
                truncated = "max_requests"
                break
            if len(rows) >= max_files:
                truncated = "max_files"
                break

            level, queue = queue, []

            with ThreadPoolExecutor(
                max_workers=1 if not parallel else max_threads
            ) as executor:
                futures = {
                    executor.submit(_fetch_listing, url): (url, depth)
                    for url, depth in level
                }
                for f in as_completed(futures):
                    url, depth = futures[f]
                    n_requests += 1
                    pbar.update(1)

                    try:
                        entries = _parse_autoindex(f.result(), url)
                    except Exception as exc:
                        # The root listing failing means we know nothing at all
                        # about this dataset - don't quietly return an empty
                        # table and let the caller draw the wrong conclusion.
                        if url == root:
                            raise BILError(
                                f"Failed to list dataset '{bildid}' at {url}: "
                                f"{exc}\nThe BIL download server may be down or "
                                "unreachable from this machine - check "
                                f"{DOWNLOAD_URL} and try again."
                            ) from exc
                        failed.append(url)
                        logger.warning(f"Failed to list {url}: {exc}")
                        continue

                    for entry in entries:
                        if entry["is_dir"]:
                            # `visited` guards against symlink loops
                            if (
                                recursive
                                and depth + 1 <= max_depth
                                and entry["url"] not in visited
                            ):
                                visited.add(entry["url"])
                                queue.append((entry["url"], depth + 1))
                        elif not pattern or fnmatch(entry["name"], pattern):
                            rows.append(
                                {
                                    **entry,
                                    "directory": url[len(root):],
                                    "depth": depth,
                                    "bildid": bildid,
                                }
                            )

    if truncated:
        logger.warning(
            f"Crawl stopped early (`{truncated}` reached): this file listing is "
            f"INCOMPLETE. Increase `{truncated}`, lower `max_depth`, or use Globus."
        )

    if failed:
        logger.warning(
            f"{len(failed)} of {n_requests} directory listings for '{bildid}' "
            "failed: this file listing is INCOMPLETE. This is typically a "
            "transient problem with the BIL download server - try again."
        )

    files = pd.DataFrame(
        rows,
        columns=["name", "url", "is_dir", "size", "last_modified", "directory",
                 "depth", "bildid"],
    )
    return files.drop(columns="is_dir").sort_values(["directory", "name"]).reset_index(
        drop=True
    )


# Keyword arguments that belong to `list_files` rather than to the reader
_LIST_FILES_KWARGS = {"recursive", "max_depth", "max_files", "max_requests", "force"}


def _is_file_table(x) -> bool:
    """Is `x` the output of `list_files`?"""
    return isinstance(x, pd.DataFrame) and {"url", "name"}.issubset(x.columns)


def _download_file(url: str, filepath: Path, pbar=None, chunk_size: int = 1024**2) -> Path:
    """Stream a single file to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temporary file and move it into place only once complete:
    # an interrupted download must not leave a truncated file behind that
    # `skip_existing` would happily hand back forever.
    tmp = filepath.with_name(filepath.name + ".part")

    with session.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=chunk_size):
                f.write(block)
                if pbar is not None:
                    pbar.update(len(block))

    os.replace(tmp, filepath)
    return filepath


def download_files(
    x,
    filepath: Union[str, Path],
    pattern: Optional[str] = None,
    *,
    max_size: Optional[str] = DEFAULT_MAX_DOWNLOAD,
    skip_existing: bool = True,
    parallel: bool = True,
    max_threads: int = 4,
    **kwargs,
) -> pd.DataFrame:
    """Download files from a BIL dataset.

    Parameters
    ----------
    x :             str | pandas.DataFrame
                    Dataset ID ("bildid"), or the output of
                    [`navis.interfaces.brain_image_library.list_files`][] - in
                    which case exactly those files are downloaded.
    filepath :      str | pathlib.Path
                    Directory to download to. The dataset's directory structure
                    is preserved underneath it.
    pattern :       str, optional
                    Only download files matching this glob, e.g. `"*.swc"`.
    max_size :      str, optional
                    Refuse to download more than this in total (e.g. `"10G"`).
                    BIL hosts datasets of hundreds of terabytes - this is here
                    to stop you pulling one by accident. Set to `None` to
                    disable the check.
    skip_existing : bool
                    Skip files that already exist locally.
    parallel :      bool
                    Whether to download in parallel.
    max_threads :   int
                    Max number of parallel threads to use.
    **kwargs
                    Passed to [`navis.interfaces.brain_image_library.list_files`][].

    Returns
    -------
    pandas.DataFrame
                    The file table with an added `filepath` column.

    Examples
    --------
    >>> import navis.interfaces.brain_image_library as bil
    >>> files = bil.download_files('ace-boo-van', '~/bil', pattern='*.swc')   # doctest: +SKIP

    """
    files = x if _is_file_table(x) else list_files(x, pattern=pattern, **kwargs)

    if _is_file_table(x) and pattern:
        files = files[[fnmatch(n, pattern) for n in files.name]]

    if not len(files):
        raise ValueError("No files to download.")

    # Guardrail. Note `list_files` may legitimately have been called with
    # `force=True`, so this is an independent check.
    total = files["size"].dropna().sum()
    if max_size is not None:
        limit = _parse_size(max_size)
        if limit is None:
            raise ValueError(f"Could not parse `max_size='{max_size}'`.")
        if total > limit:
            raise ValueError(
                f"Downloading these {len(files):,} files would fetch "
                f"~{utils.sizeof_fmt(total)}, which exceeds `max_size={max_size}`.\n"
                "Your options:\n"
                "  - narrow the selection with `pattern=...`\n"
                "  - raise `max_size` or set it to `None` to disable this check\n"
                f"  - {_HELP_GLOBUS}"
            )

    filepath = Path(filepath).expanduser()

    targets = []
    for row in files.itertuples():
        target = filepath / str(row.directory or "") / row.name
        targets.append(target)
    files = files.assign(filepath=targets)

    todo = [
        (row.url, Path(row.filepath), row.size)
        for row in files.itertuples()
        if not (skip_existing and Path(row.filepath).exists())
    ]

    n_skipped = len(files) - len(todo)
    if n_skipped:
        logger.info(f"Skipping {n_skipped} file(s) that already exist.")

    if todo:
        todo_bytes = sum(s for _, _, s in todo if not pd.isnull(s))
        with config.tqdm(
            desc="Downloading",
            total=todo_bytes if todo_bytes else None,
            unit="B",
            unit_scale=True,
            leave=config.pbar_leave,
            disable=config.pbar_hide,
        ) as pbar:
            with ThreadPoolExecutor(
                max_workers=1 if not parallel else max_threads
            ) as executor:
                futures = {
                    executor.submit(_download_file, url, target, pbar): url
                    for url, target, _ in todo
                }
                for f in as_completed(futures):
                    f.result()  # raise any exception

    return files


#############################################
#                                           #
#                  Neurons                  #
#                                           #
#############################################


def get_neurons(
    x,
    pattern: str = "*.swc",
    *,
    max_neurons: Optional[int] = None,
    parallel: bool = True,
    max_threads: int = 4,
    **kwargs,
) -> NeuronList:
    """Fetch neuron reconstructions from a BIL dataset.

    Skeletons are streamed straight into memory - nothing is written to disk.
    Use [`navis.interfaces.brain_image_library.download_files`][] if you want
    the files themselves.

    Parameters
    ----------
    x :             str | pandas.DataFrame
                    Dataset ID ("bildid"), or the output of
                    [`navis.interfaces.brain_image_library.list_files`][] - in
                    which case exactly those files are loaded. The latter lets
                    you inspect what you are about to fetch first.
    pattern :       str
                    Which files to read. Defaults to `"*.swc"`.
    max_neurons :   int, optional
                    Cap the number of neurons fetched.
    parallel :      bool
                    Whether to fetch in parallel.
    max_threads :   int
                    Max number of parallel threads to use.
    **kwargs
                    Passed to [`navis.read_swc`][]. Note that BIL does not
                    reliably record the units of its reconstructions - pass
                    e.g. `units='um'` if you know them. The `image.stepsizex`
                    field in the metadata (e.g. "0.35 micron/pixel") tells you
                    the voxel size if the coordinates are in voxels.

    Returns
    -------
    navis.NeuronList

    Examples
    --------
    >>> import navis.interfaces.brain_image_library as bil
    >>> # Look before you leap
    >>> files = bil.list_files('ace-boo-van', pattern='*.swc')
    >>> nl = bil.get_neurons(files, max_neurons=5)
    >>> len(nl)
    5

    """
    # Split off the kwargs meant for `list_files` - the rest go to `read_swc`
    list_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _LIST_FILES_KWARGS}

    # Note we deliberately crawl *without* the pattern and filter afterwards:
    # it costs no extra requests and means we can tell the user which file types
    # the dataset actually contains if nothing matches.
    available = x if _is_file_table(x) else list_files(x, **list_kwargs)
    files = available[[fnmatch(n, pattern) for n in available.name]]

    if not len(files):
        # An empty listing is a different problem from "no *.swc in here" and
        # warrants a different hint: the dataset may be empty, but far more
        # likely the crawl was blocked or the listing didn't get that deep.
        if not len(available):
            raise ValueError(
                "The file listing for this dataset is empty - there is nothing "
                "to load. If you passed a dataset ID, try inspecting it with "
                "`list_files()` first: the crawl may have been cut short (see "
                "`max_depth`/`max_files`/`max_requests`) or the BIL download "
                "server may be unreachable."
            )

        found = sorted({Path(n).suffix for n in available.name if Path(n).suffix})
        raise ValueError(
            f'No files matching "{pattern}" in this dataset.'
            + (f" File types present: {', '.join(found)}." if found else "")
            + " Note that BIL cell morphology datasets often ship Vaa3D "
            "reconstructions (.eswc, .ano, .apo) instead of plain SWC - navis "
            "cannot read those directly."
        )

    if max_neurons and len(files) > max_neurons:
        logger.warning(
            f"Restricting to the first {max_neurons} of {len(files)} files."
        )
        files = files.iloc[:max_neurons]

    urls = list(files.url)

    # Pass `max_threads` explicitly so that the caller controls the number of
    # threads we hit the BIL servers with. `read_swc` takes care of `name` and
    # `origin` (it parses them from the URL).
    nl = read_swc(urls, parallel=max_threads if parallel else False, **kwargs)

    # A dataset is one cell more often than not, so a bare file name is rarely
    # unique across datasets. Give each neuron a composite id and keep the
    # dataset it came from - neither is something navis can derive itself.
    # Order is preserved, but check to be safe.
    if len(nl) == len(files):
        for neuron, row in zip(nl, files.itertuples()):
            neuron.id = f"{row.bildid}/{Path(row.name).stem}"
            neuron.bildid = row.bildid
    else:
        logger.warning(
            "Could not match neurons back to their source files - `id` and "
            "`bildid` may be missing."
        )

    return nl
