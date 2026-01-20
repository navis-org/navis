"""
Multiprocessing
===============

This notebook will show you how to use parallel processing with `navis`.

By default, most {{ navis }} functions use only a single thread/process (although some third-party functions
used under the hood might use more). Distributing expensive computations across multiple cores can speed things
up considerable.

Many {{ navis }} functions natively support parallel processing. This notebook will illustrate various ways
to use parallelism. Before we get started: by default, {{ navis }} uses `joblib` as backend for multiprocessing.
However, you can also use `pathos` as an alternative. If you installed {{ navis }} with `pip install navis[all]`
you should be all set to use either. If not, you can install the packages separately:

```shell
pip install joblib tqdm-joblib -U
pip install pathos -U
```

!!! tip
    Parallel processing incurs overhead: we have to spawn additional processes and move data between the main
    & the worker processes. If you have fast single-core performance and/or small tasks, that overhead might outweigh
    the benefits of parallelism. See also additional notes at the bottom of this tutorial.

## Running {{ navis }} functions in parallel

Since version `0.6.0` many {{ navis }} functions accept a `parallel=True` and an (optional) `n_cores` parameter:
"""

# %%
import time
import navis

def time_func(func, *args, **kwargs):
    """A function to time the execution of a function."""
    start = time.time()
    func(*args, **kwargs)
    print(f"Execution time: {round(time.time() - start, 2)}s")

# Load example neurons
nl = navis.example_neurons()

# %%
# !!! important
#     This documentation is built on Github Actions where the number of cores can be as low as 2. The speedup on
#     your machine should be more pronounced than the times you see below. That said: parallel processing has some
#     overhead and for small tasks the overhead can be larger than the speed-up.

# %%
# Without parallel processing:
time_func (
    navis.resample_skeleton,
    nl,
    resample_to=125
)

# %%
# With parallel processing:
time_func (
    navis.resample_skeleton,
    nl,
    resample_to=125,
    parallel=True
)

# %%
# The same also works for neuron methods!
#
# Without parallel processing:

time_func (
    nl.resample, 125
)

# %%
# With parallel processing:

time_func (
    nl.resample, 125, parallel=True
)

# %%
# By default `parallel=True` will use half the available CPU cores.
# You can adjust that behaviour using the `n_cores` parameter:

time_func (
    nl.resample, 125, parallel=True, n_cores=2
)

# %%
# !!! note
#     The name `n_cores` is actually a bit misleading as it determines the number of parallel processes
#     that {{ navis }} will spawn. There is nothing stopping you from setting `n_cores` to a number higher than
#     the number of available CPU cores. However, doing so will likely over-subscribe your CPU and end up
#     slowing things down.

# %%
# Additional parameters for controlling parallelism:
# - `backend`: either "auto" (default), "joblib", or "pathos". This determines which parallel processing
#   backend to use. "auto" will pick "joblib" if available, otherwise "pathos". Note: `joblib` is compatible
#   with Dask to run on clusters. See the joblib documentation for details on how to set that up and then
#   use `backend="joblib:dask"` in {{ navis }}.
# - `chunksize`: either "auto" (default) or an integer. This determines the number of neurons
#   that will be processed per worker in each batch. "auto" will pick a chunksize that tries to balance
#   load across workers while minimizing overhead. You can also set a fixed chunksize
#   (e.g. `chunksize=10`).

# %%
# ## Parallelizing generic functions
#
# For non-{{ navis }} function you can use [`NeuronList.apply`][navis.NeuronList.apply] to parallelize them.
#
# First, let's write a mock function that simply waits one second and then returns the number of nodes:

def my_func(x):
    import time
    time.sleep(1)
    return x.n_nodes

# %%

# Without parallel processing
time_func (
    nl.apply, my_func
)


# %%

# With parallel processing
time_func (
    nl.apply, my_func, parallel=True
)

"""
## A note on free-threading

With version 3.13, Python introduced a free-threading build where the Global Interpreter Lock (GIL) is removed.
In theory, this would allow true multi-threading in Python and make parallel processing with threads much more efficient.
However, as of late 2025, many of the libraries used by {{ navis }} (e.g. `igraph`) do not yet support free-threading.
"""