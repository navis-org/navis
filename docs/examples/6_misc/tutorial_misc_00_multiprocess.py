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
# | `joblib` | `pip install joblib` | Default where installed. Handles lambdas, and keeps its workers alive between calls. |
# | `pathos` | `pip install pathos` | Also handles lambdas (via `dill`), but builds a fresh pool for every call. |
# | `processes` | - | Standard library. No dependencies, but cannot ship lambdas. |
# | `threads` | - | Only helps for work that releases the GIL. |
# | `serial` | - | No parallelism at all - handy for debugging. |
# | `dask` | `pip install navis[cluster]` | Another set of machines. See [below](#running-on-a-cluster). |
# | `submitit` | `pip install navis[cluster]` | Submits to a scheduler (SLURM). See [below](#running-on-a-cluster). |

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
# ## Running on a cluster
#
# Nothing above changes when the work leaves your machine. `parallel=True` still just
# means "spread this over the neurons"; you point [`navis.set_parallel_backend`][] at a
# cluster and the same calls run there.
#
# The scheduler is configured with *its* library's API, never with {{ navis }} arguments -
# there is deliberately no `slurm_partition=` anywhere in {{ navis }}. You build the
# object, {{ navis }} runs the work on it.
#
# === "dask"
#
#     ```python
#     from dask.distributed import Client
#
#     client = Client("tcp://scheduler:8786")   # or LocalCluster(), SLURMCluster(), ...
#
#     with navis.set_parallel_backend(client):
#         navis.resample_skeleton(nl, resample_to=125, parallel=True)
#     ```
#
#     Best when you already have a cluster up and want results back interactively.
#
# === "submitit"
#
#     ```python
#     import submitit
#
#     ex = submitit.AutoExecutor(folder="logs")
#     ex.update_parameters(slurm_partition="cpu", timeout_min=60, mem_gb=8)
#
#     with navis.set_parallel_backend(ex):
#         navis.resample_skeleton(nl, resample_to=125, parallel=True)
#     ```
#
#     Best where you would otherwise write a batch script: the work goes into the queue
#     as an array job. `cluster="local"` runs the same thing as subprocesses, which is
#     the cheap way to check a pipeline before submitting it.
#
# Both need `pip install navis[cluster]`. A `dask.distributed.Client`, a
# `submitit.Executor` and any plain `concurrent.futures.Executor` are all accepted
# directly - so ipyparallel, `mpi4py.futures` and friends work too, without {{ navis }}
# knowing anything about them.

# %%
# ### How the work gets split up
#
# Sending one neuron per task is right on one machine and wasteful across a network - a
# neuron is a few hundred kilobytes, and on a scheduler each task would be a whole *job*.
# So the cluster backends bundle neurons into fewer, larger units: enough units to keep
# every worker busy, but never more than ~128 MB of neurons in any one of them.
#
# You shouldn't need to tune this, but `chunksize` overrides it for a single call:
#
# ```python
# navis.resample_skeleton(nl, resample_to=125, parallel=True, chunksize=50)
# ```
#
# !!! note "What `n_cores` means on a cluster"
#     For `submitit` it only decides how the neurons are *divided up* - how many of those
#     jobs run at once is the scheduler's business, set on the executor (e.g.
#     `slurm_array_parallelism`). For `dask` it isn't used for sizing at all: {{ navis }}
#     reads the real worker count off the cluster.

# %%

# mkdocs_gallery_thumbnail_path = '_static/multiprocess.png'