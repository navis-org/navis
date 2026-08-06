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

"""Where navis runs its work.

`parallel=True` means "spread this over the neurons"; this module decides
*where* that happens. Point it at a different backend and the same call runs on
threads, on all your cores, or on another set of machines::

    navis.set_parallel_backend('joblib')
    navis.prune_twigs(nl, 5, parallel=True)

Scheduler configuration is deliberately not navis' business: configure your
executor with its own library's API and hand the object over, so navis never
has to grow a `slurm_partition` keyword::

    with navis.set_parallel_backend(dask_client):
        navis.prune_twigs(nl, 5, parallel=True)

"""

import atexit

from .threads import set_num_threads, limit_native_threads
from .dispatch import (map_tasks, imap_tasks, cpu_count, default_n_workers,
                       resolve_thread_cap, FailedRun, worker_init_hooks,
                       picklable_by_reference)
from .backends import (ParallelBackend, ExecutorBackend, register_backend,
                       get_backend, list_backends, available_backends,
                       resolve_backend, set_parallel_backend)


def shutdown():
    """Release resources held by the parallel backends.

    Backends that keep a pool of workers alive between calls tear it down here.
    Called automatically at interpreter exit - including for backends
    registered by third parties.
    """
    for backend in list(available_backends()):
        backend.shutdown()


atexit.register(shutdown)


#: Names exported to the top-level `navis` namespace.
__all__ = ['set_parallel_backend', 'list_parallel_backends', 'set_num_threads']


def list_parallel_backends() -> list:
    """List the parallel backends that can be used on this machine.

    Returns
    -------
    list of str

    Examples
    --------
    >>> import navis
    >>> 'serial' in navis.list_parallel_backends()
    True

    """
    return [b.name for b in available_backends()]
