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

"""The dask.distributed backend.

`Client.get_executor()` is a real `concurrent.futures.Executor`, so dask
already worked through the generic wrapper. This backend exists because it can
do three things the wrapper cannot:

- take the `Client` itself, and find one that already exists,
- read the cluster's actual layout instead of guessing whether workers share
  memory with us, and
- move the neurons to the workers directly rather than through the scheduler.

"""

import importlib.util

from ... import config
from .base import (ParallelBackend, apply_overrides, auto_chunksize,
                   REMOTE_CHUNKS_PER_WORKER)

logger = config.get_logger(__name__)

__all__ = ['DaskBackend']


class DaskBackend(ParallelBackend):
    """Run on a `dask.distributed` cluster.

    Never selected automatically - a cluster is something you set up, not
    something navis should conjure. Either hand over the client::

        client = Client('tcp://scheduler:8786')
        with navis.set_parallel_backend(client):
            navis.prune_twigs(nl, 5, parallel=True)

    or, if one is already the active client for this session, name the
    backend::

        navis.set_parallel_backend('dask')

    """

    name = 'dask'
    auto_select = False

    isolated = True
    pickles_by_value = True     # cloudpickle
    chunks_per_worker = REMOTE_CHUNKS_PER_WORKER

    def __init__(self, client=None, **overrides):
        self.client = client
        if client is not None:
            self.isolated = _workers_are_processes(client)
        apply_overrides(self, **overrides)

    def available(self):
        # A spec lookup rather than an import: `distributed` is slow to import
        # and this runs on every backend listing.
        return importlib.util.find_spec('distributed') is not None

    def adopt(self, obj, **overrides):
        # Recognise it without importing distributed - `adopt` is offered every
        # object handed to `set_parallel_backend`, including on machines where
        # dask isn't installed at all.
        if type(obj).__module__.split('.')[0] not in ('distributed', 'dask'):
            return None

        import distributed

        if isinstance(obj, distributed.Client):
            return DaskBackend(obj, **overrides)
        if isinstance(obj, distributed.cfexecutor.ClientExecutor):
            return DaskBackend(obj._client, **overrides)
        # A bare cluster (LocalCluster, SLURMCluster, ...) - a client for it is
        # cheap and is what actually submits work.
        if hasattr(obj, 'scheduler_address') and not isinstance(obj, type):
            return DaskBackend(distributed.Client(obj), **overrides)
        return None

    def get_client(self):
        """The client to submit to.

        Falls back to the session's active client, which is what `Client(...)`
        registers itself as - so `set_parallel_backend('dask')` works without
        having to plumb the object through.
        """
        if self.client is not None:
            return self.client

        import distributed

        try:
            return distributed.get_client()
        except ValueError:
            raise ValueError(
                "The 'dask' backend needs a cluster to talk to and no client "
                'is active in this session. Create one, or hand it over '
                'explicitly:\n'
                '    from dask.distributed import Client\n'
                '    client = Client(...)\n'
                "    navis.set_parallel_backend(client)"
            ) from None

    def chunksize(self, n_tasks, n_workers, requested=None, size_hint=None):
        """Size units against the cluster, not against this machine.

        `n_workers` defaults to half the *local* core count, which says nothing
        about how big the cluster is. Where we can see the real worker count,
        use it.
        """
        if requested is None and self.chunks_per_worker:
            n_workers = self._n_workers() or n_workers
            return auto_chunksize(n_tasks, n_workers,
                                  chunks_per_worker=self.chunks_per_worker,
                                  max_bytes=self.max_chunk_bytes,
                                  size_hint=size_hint)
        return super().chunksize(n_tasks, n_workers, requested=requested,
                                 size_hint=size_hint)

    def _n_workers(self):
        """Workers currently in the cluster, or None if we can't tell."""
        try:
            return len(self.get_client().scheduler_info()['workers']) or None
        except Exception as e:
            logger.debug(f'Could not read the dask cluster size ({e}).')
            return None

    def map(self, func, payloads, *, n_workers):
        import distributed

        client = self.get_client()

        # Scatter first: a payload is a slice of the neurons, and putting that
        # in the task graph would push all of it through the scheduler. This
        # sends it straight to the workers instead. `hash=False` because
        # hashing megabytes of neuron to deduplicate payloads that are unique
        # by construction is pure cost.
        scattered = client.scatter(list(payloads), hash=False)

        # `pure=False`: navis functions are not pure (`inplace=`, RNG-driven
        # resampling), so dask must not dedupe or cache calls by their inputs.
        futures = client.map(func, scattered, pure=False)
        try:
            for _, result in distributed.as_completed(futures,
                                                      with_results=True):
                yield result
        except BaseException:
            # Don't leave the cluster chewing on work nobody will collect
            client.cancel(futures, force=True)
            raise
        finally:
            del scattered, futures


def _workers_are_processes(client) -> bool:
    """Whether this client's workers have their own address space.

    `LocalCluster(processes=False)` runs its workers as threads in *our*
    interpreter, where turning a caller's `inplace=False` into an in-place
    operation would corrupt their neurons. Every other arrangement - a local
    cluster of processes, anything reached over the network - is isolated.
    """
    cluster = getattr(client, 'cluster', None)
    processes = getattr(cluster, 'processes', None)
    return True if processes is None else bool(processes)
