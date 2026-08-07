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

"""Pooled `requests` sessions, shared by everything in navis that talks HTTP.

Two parts of navis fetch over the network - the URL readers in `navis.io` and
the data-source interfaces in `navis.interfaces` - and both want the same thing:
connections to a host pooled and kept alive (it matters a lot when reading a few
hundred files off one server), plus retries on the transient errors that public
academic services hand out under load.

Sessions are cached and shared. The cache is keyed on the pid as well as on the
configuration, so that a forked child never inherits - and corrupts - the
parent's live sockets.
"""

import os
import requests

from requests.adapters import HTTPAdapter
from urllib3.util import Retry

__all__ = ["get_session", "clear_sessions"]

# Identify ourselves. Some of the services navis talks to are small academic
# servers whose admins do look at their logs.
DEFAULT_HEADERS = {"User-Agent": "github.com/navis-org/navis"}

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.5
DEFAULT_POOL_MAXSIZE = 32

# 429 is rate limiting; the 5xx are the transient failures a busy server throws.
# Notably absent: 404 and the other 4xx, which retrying cannot fix.
RETRY_STATUS = (429, 500, 502, 503, 504)

# navis only ever POSTs *reads* - search endpoints that take a body too big for
# a query string (NeuroMorpho, the Brain Image Library) and token requests. None
# of those is unsafe to repeat, which is why POST is retried here even though it
# is not idempotent in general.
DEFAULT_METHODS = ("GET", "POST")

_SESSIONS = {}


def get_session(
    headers=None,
    *,
    retries=DEFAULT_RETRIES,
    backoff_factor=DEFAULT_BACKOFF,
    pool_maxsize=DEFAULT_POOL_MAXSIZE,
    allowed_methods=DEFAULT_METHODS,
) -> requests.Session:
    """Return a pooled `requests.Session`.

    Sessions are cached, so calling this per request is cheap and is in fact the
    intended usage - hold on to the return value only for as long as you need it.

    Parameters
    ----------
    headers :           dict, optional
                        Extra headers, applied on top of `DEFAULT_HEADERS`. Pass
                        an explicit `{}` to get only the defaults.
    retries :           int
                        Number of retries for the statuses in `RETRY_STATUS`.
                        Zero disables retrying.
    backoff_factor :    float
                        Exponential backoff between retries, in seconds.
    pool_maxsize :      int
                        Size of the connection pool. Must be at least as large
                        as the number of threads that will share the session.
    allowed_methods :   tuple
                        HTTP methods that may be retried.

    Returns
    -------
    requests.Session

    Examples
    --------
    >>> from navis.utils.http import get_session
    >>> session = get_session()
    >>> session is get_session()  # cached
    True

    """
    headers = dict(headers) if headers else {}

    # The key has to pin everything that shapes the session - two callers asking
    # for different retry behaviour must not be handed the same object.
    key = (
        os.getpid(),
        tuple(sorted(headers.items())),
        retries,
        backoff_factor,
        pool_maxsize,
        tuple(allowed_methods),
    )

    session = _SESSIONS.get(key)
    if session is not None:
        return session

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    session.headers.update(headers)

    adapter = HTTPAdapter(
        max_retries=Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=RETRY_STATUS,
            allowed_methods=list(allowed_methods),
        ),
        pool_maxsize=pool_maxsize,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    _SESSIONS[key] = session
    return session


def clear_sessions():
    """Close and forget every pooled session.

    Mainly useful in tests and when you want to be sure no sockets are held open.
    Sessions are re-created on demand, so this is always safe to call.
    """
    while _SESSIONS:
        _, session = _SESSIONS.popitem()
        try:
            session.close()
        except Exception:
            pass
