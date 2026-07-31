"""
Multiprocessing
===============

Speed up batch workloads with built-in parallel processing.

By default, most {{ navis }} functions use only a single thread/process (although some third-party functions
used under the hood might). Distributing expensive computations across multiple cores can speed things up considerably.

Many {{ navis }} functions natively support parallel processing. This notebook will illustrate various ways
to use parallelism. Nothing needs installing to get started - {{ navis }} falls back to the standard library's
process pool - though see [choosing a backend](#choosing-a-backend) at the bottom for when that matters.

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
# !!! note
#     This documentation is built on Github Actions where the number of cores can be as low as 2. The speedup on
#     your machine should be more pronounced than what you see below. That said: parallel processing has some
#     overhead and for small tasks the overhead can be larger than the speed-up.

# %%
# The same `parallel` switch works whether you call the module-level function or the
# neuron's own method - just add `parallel=True`:
#
# === "Serial"
#     ```python
#     navis.resample_skeleton(nl, resample_to=125)   # module-level function
#     nl.resample(125)                               # equivalent neuron method
#     ```
#
# === "Parallel"
#     ```python
#     navis.resample_skeleton(nl, resample_to=125, parallel=True)
#     nl.resample(125, parallel=True)
#     ```
#
# Let's time the function form to see it for real:

# %%
# Serial:
time_func(navis.resample_skeleton, nl, resample_to=125)

# %%
# Parallel:
time_func(navis.resample_skeleton, nl, resample_to=125, parallel=True)

# %%
# By default `parallel=True` uses half the available CPU cores. Adjust that with the
# `n_cores` parameter:

time_func(nl.resample, 125, parallel=True, n_cores=2)

# %%
# !!! note
#     The name `n_cores` is a bit misleading: it sets the number of parallel *processes*
#     that {{ navis }} spawns. Nothing stops you from setting it higher than your CPU count -
#     but doing so will likely over-subscribe the CPU and end up slowing things down.

# %%
# ## Parallelizing generic functions
#
# For non-{{ navis }} functions you can use [`NeuronList.apply`][navis.NeuronList.apply] to parallelize them.
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

# %%
# ## Choosing a backend
#
# `parallel=True` says *that* the work should be spread over the neurons;
# [`navis.set_parallel_backend`][] says *where* it runs. The default (`"auto"`) picks the
# best backend you have installed, so you rarely need to touch this:
#
# | Backend | Needs | Notes |
# |---|---|---|
# | `pathos` | `pip install pathos` | Default where installed. Serializes with `dill`, so it can run lambdas. |
# | `joblib` | `pip install joblib` | Also handles lambdas, and keeps workers alive between calls. |
# | `processes` | - | Standard library. No dependencies, but cannot ship lambdas. |
# | `threads` | - | Only helps for work that releases the GIL. |
# | `serial` | - | No parallelism at all - handy for debugging. |

print(navis.list_parallel_backends())

# %%
# Set one globally, or scope it to a block:
#
# ```python
# navis.set_parallel_backend("joblib")
#
# with navis.set_parallel_backend("threads"):
#     navis.resample_skeleton(nl, resample_to=125, parallel=True)
# ```
#
# You can also pass a single call its own backend:
#
# ```python
# navis.resample_skeleton(nl, resample_to=125, parallel=True, backend="processes")
# ```

# %%
# !!! tip "Running on a cluster"
#     [`navis.set_parallel_backend`][] also accepts any `concurrent.futures.Executor`,
#     which is how you point {{ navis }} at more than one machine. Configure the executor
#     with its own library's API - {{ navis }} deliberately has no `slurm_partition`-style
#     parameters of its own - and hand it over:
#
#     ```python
#     from dask.distributed import Client
#
#     client = Client("tcp://scheduler:8786")
#     with navis.set_parallel_backend(client.get_executor()):
#         navis.resample_skeleton(nl, resample_to=125, parallel=True)
#     ```
#
#     Note that neurons are sent to the workers, so for remote backends it pays to raise
#     `chunksize` - shipping one neuron per task is a lot of overhead for a short job.

# %%

# mkdocs_gallery_thumbnail_path = '_static/multiprocess.png'