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

import concurrent.futures as cf

from ... import config
from .base import ParallelBackend, WrappedExecutorBackend, apply_overrides

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
    requires = 'distributed'
    adopts = ('distributed', 'dask')

    isolated = True
    pickles_by_value = True     # cloudpickle
    # Every unit is a scheduler round trip plus a transfer, so bundle. Eight
    # per worker still leaves plenty of room to even out stragglers.
    chunks_per_worker = 8

    def __init__(self, client=None, **overrides):
        self.client = client
        if client is not None:
            self.isolated = _workers_are_processes(client)
        apply_overrides(self, **overrides)

    def _adopt(self, obj, **overrides):
        import distributed

        if isinstance(obj, distributed.Client):
            return DaskBackend(obj, **overrides)
        if isinstance(obj, distributed.cfexecutor.ClientExecutor):
            return DaskBackend(obj._client, **overrides)
        # A bare cluster (LocalCluster, SLURMCluster, ...) - a client for it is
        # cheap and is what actually submits work.
        if hasattr(obj, 'scheduler_address'):
            return DaskBackend(distributed.Client(obj), **overrides)
        # Some other dask executor. We can't read its cluster, but we still
        # know more about it than the generic guess would: cloudpickle, and
        # units that are worth bundling. `None` means "not stated", so these
        # are defaults rather than overrides of the user's own values.
        if isinstance(obj, cf.Executor):
            for key, value in (('pickles_by_value', True),
                               ('chunks_per_worker', self.chunks_per_worker)):
                if overrides.get(key) is None:
                    overrides[key] = value
            return WrappedExecutorBackend(obj, **overrides)
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

    def worker_count(self, hint):
        """Size units against the cluster, not against this machine.

        `n_cores` defaults to half the *local* core count, which says nothing
        about how big the cluster is. Fall back to it only if the scheduler
        won't tell us.
        """
        try:
            # `n_workers=0` asks for the count without the per-worker state
            # dicts, which are ~1 KB each and all we would do is len() them.
            return self.get_client().scheduler_info(n_workers=0)['n_workers'] \
                or hint
        except Exception as e:
            logger.debug(f'Could not read the dask cluster size ({e}).')
            return hint

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
