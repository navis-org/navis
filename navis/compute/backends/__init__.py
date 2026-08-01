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

"""Pluggable backends for parallel processing.

See `navis.compute.backends.base` for the abstraction. Third-party libraries
can register their own::

    from navis.compute.backends import ParallelBackend, register_backend

    class MyClusterBackend(ParallelBackend):
        name = "mycluster"
        priority = 30
        pickles_by_value = True
        def available(self): ...
        def map(self, func, payloads, *, n_workers): ...

    register_backend(MyClusterBackend())

For anything that is already a `concurrent.futures.Executor` you don't need a
backend at all - hand it straight to `navis.set_parallel_backend()`.

"""

from .base import (ParallelBackend, ExecutorBackend, WrappedExecutorBackend,
                   register_backend, get_backend, list_backends,
                   available_backends, resolve_backend, set_parallel_backend,
                   auto_chunksize, adopt_object, apply_overrides)
from .local import SerialBackend, ThreadBackend, ProcessBackend, shutdown_pool
from ._pathos import PathosBackend
from ._joblib import JoblibBackend
from ._dask import DaskBackend
from ._submitit import SubmititBackend

# Register the backends shipped with navis. `serial` must always be present -
# it is what the dispatcher degrades to when there is nothing to parallelise.
# The cluster backends are registered too even where their dependencies are
# missing: they are what recognises a `Client`/`AutoExecutor` handed to
# `set_parallel_backend`, and `auto_select = False` keeps them out of the way.
register_backend(SerialBackend())
register_backend(ThreadBackend())
register_backend(ProcessBackend())
register_backend(JoblibBackend())
register_backend(PathosBackend())
register_backend(DaskBackend())
register_backend(SubmititBackend())

__all__ = ['ParallelBackend', 'ExecutorBackend', 'WrappedExecutorBackend',
           'register_backend', 'get_backend', 'list_backends',
           'available_backends', 'resolve_backend', 'set_parallel_backend',
           'SerialBackend', 'ThreadBackend', 'ProcessBackend',
           'PathosBackend', 'JoblibBackend', 'DaskBackend', 'SubmititBackend',
           'shutdown_pool', 'auto_chunksize', 'adopt_object',
           'apply_overrides']
