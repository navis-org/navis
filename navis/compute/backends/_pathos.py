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

"""The pathos backend - what `parallel=True` has always used."""

import importlib.util

from ... import config
from .base import ParallelBackend

logger = config.get_logger(__name__)

__all__ = ['PathosBackend']


class PathosBackend(ParallelBackend):
    """Run in separate processes using `pathos`.

    Serialises with `dill`, so it can ship lambdas, closures and functions
    defined in a notebook - which is why it is the highest-priority backend
    and why `NeuronList.apply(lambda ...)` works on it.

    Note that `multiprocess` (pathos' fork of `multiprocessing`) starts workers
    with `fork` even on macOS. That makes it markedly cheaper to start than a
    spawning pool, but forking a threaded process is exactly what CPython is
    moving away from.
    """

    name = 'pathos'
    priority = 20

    isolated = True
    pickles_by_value = True

    def available(self):
        # Spec lookup rather than an import - see the note in `_joblib.py`
        return importlib.util.find_spec('pathos') is not None

    def map(self, func, payloads, *, n_workers):
        # Note we use the private `_ProcessPool` rather than `ProcessingPool`
        # because the latter ignores `chunksize`, see
        # https://stackoverflow.com/questions/55611806/how-to-set-chunk-size-when-using-pathos-processingpools-map
        import pathos

        # Payloads are already chunked by the dispatcher, so pathos must not
        # chunk them a second time.
        with pathos.pools._ProcessPool(n_workers) as pool:
            yield from pool.imap_unordered(func, payloads, chunksize=1)
