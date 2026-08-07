"""
Multiprocessing
===============
<!-- difficulty: intermediate -->

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
# ## How many threads each worker gets
#
# `n_cores` is only half the story. {{ navis }} spreads work over *processes*, but several
# of the libraries underneath it - [navis-fastcore](https://github.com/schlegelp/fastcore-rs)
# above all, plus the BLAS/OpenMP pools under numpy - spread work over *threads*, and
# by default each takes every core it can see. Nothing tells a worker process that it
# is one of twenty, so without help the two multiply:
#
# ```python
# # 20 workers on a 224-core node, each building a 224-thread pool:
# # 4480 threads over 224 cores.
# navis.heal_skeleton(nl, parallel=True, n_cores=20)
# ```
#
# That is not a theoretical cost. Measured on exactly that node, healing 40 skeletons
# of 200k nodes each:
#
# | | wall | CPU |
# |---|---|---|
# | 1 worker, unlimited threads | 5.10 s | 1x |
# | 20 workers, unlimited threads | 6.71 s | 2.3x |
# | 20 workers, 1 thread each | 3.60 s | 0.4x |
#
# Parallelising made it *slower* than not parallelising at all, at more than double
# the CPU.
#
# {{ navis }} handles this for you: each worker is told how many threads it may use, and
# by default that is `cpu_count() // n_cores` - the machine divided up rather than
# handed to each worker whole. Override it with `inner_max_num_threads`:

# %%
# ```python
# # One thread per worker
# with navis.set_parallel_backend(inner_max_num_threads=1):
#     navis.heal_skeleton(nl, parallel=True, n_cores=20)
#
# # ... or globally, alongside the backend
# navis.set_parallel_backend("joblib", inner_max_num_threads=1)
# ```
#
# !!! tip "When *you* are the one running the pool"
#     {{ navis }} can only cap workers it started. If you are running your own
#     `multiprocessing.Pool` with {{ navis }} inside it, call
#     [`navis.set_num_threads`][] in each worker - it does the same thing for the
#     current process:
#
#     ```python
#     def work(neuron):
#         navis.set_num_threads(1)     # or once, in the pool's `initializer`
#         return navis.heal_skeleton(neuron)
#     ```
#
# !!! note "Cluster backends are left alone"
#     `cpu_count() // n_cores` is arithmetic about *your* machine and says nothing
#     about the node a SLURM job lands on. On the `dask` and `submitit` backends the
#     automatic cap is skipped - a worker's core budget was set by whatever allocated
#     it. An explicit `inner_max_num_threads` still applies.

# %%
# ## Is it worth parallelizing at all?
#
# Not always, and the answer is not a property of {{ navis }} - it depends on the
# function, on your data and on the machine. Three things to weigh:
#
# - **The profile shifts with the work.** Many small tasks want *fewer* threads per
#   worker: there is not enough parallel work inside one neuron to fill the threads it
#   claims, and the hand-off costs more than it saves. A few large tasks want more.
#   A 224-core node behaves nothing like an 8-core laptop, so no default is right
#   everywhere - which is why `inner_max_num_threads` exists.
# - **Some functions are not worth parallelizing, period.** If a function is already
#   threaded internally - anything fastcore-backed, which is most of the graph and
#   geodesic machinery - `parallel=True` adds a layer of concurrency on top of one
#   that was already using your cores. And if a function is fast relative to the cost
#   of moving a neuron in and out of a worker, that transfer *is* the runtime. The
#   timings at the top of this page show both effects on small example neurons.
# - **Measure before committing.** `%timeit` on a slice of your own data, at two or
#   three settings, will tell you more than any rule of thumb.
#
# The case that reliably pays is [`navis.Pipeline`][] - which is next.

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
# ## Composing pipelines
#
# `parallel=True` sends each neuron out to a worker and back again - once per function.
# Chain a few functions and you pay that toll every time:
#
# ```python
# sk  = navis.skeletonize(nl, parallel=True)             # neurons out and back
# res = navis.resample_skeleton(sk, 500, parallel=True)  # ... and again
# ```
#
# The old trick was to smuggle the whole chain into a single `apply` call, which works
# but does not scale past two or three functions:
#
# ```python
# res = nl.apply(lambda x: navis.resample_skeleton(navis.skeletonize(x), 500),
#                parallel=True)
# ```
#
# A [`navis.Pipeline`][] is that, made readable. It fuses consecutive per-neuron steps
# into a *single* task, so each neuron makes the trip once no matter how many steps
# there are.
#
# That is also why pipelines are the case where parallelism reliably pays: the transfer
# cost is fixed per neuron, so fusing steps buys you more work against the same
# overhead. The more steps you chain, the better the trade gets - which is the opposite
# of calling each function with `parallel=True` in turn.
#
# Build one by naming {{ navis }} functions directly:

pipe = (
    navis.Pipeline()
    .heal_skeleton()
    .prune_twigs(5000)
    .resample_skeleton(1000)
)
pipe

# %%
# Then call it on your neurons. It takes the same `parallel`, `n_cores`, `backend`,
# `progress` and `omit_failures` arguments you already know:

def three_calls(nl):
    x = navis.heal_skeleton(nl, parallel=True, n_cores=2)
    x = navis.prune_twigs(x, 5000, parallel=True, n_cores=2)
    return navis.resample_skeleton(x, 1000, parallel=True, n_cores=2)

# Warm the worker pool first, or we would just be timing its start-up
_ = navis.heal_skeleton(nl, parallel=True, n_cores=2)

# %%
# Three separate parallel calls - three round-trips per neuron:
time_func(three_calls, nl)

# %%
# One pipeline - one round-trip per neuron:
time_func(pipe, nl, parallel=True, n_cores=2)

# %%
# !!! note "Why the difference is small here"
#     What a pipeline saves is the cost of moving neurons in and out of the workers, so
#     the gap grows with how big your neurons are and how many steps you chain. Five
#     small example skeletons and three steps is about as unflattering as it gets.
#
# There is a second saving that does not show up in the task count. Normally every
# {{ navis }} function has to copy the neuron it is given, because it must not modify
# yours. A pipeline keeps track of *who owns* the neuron flowing through it: as soon as
# a step has handed back something the pipeline made itself, every step after it is
# allowed to work in place. Three steps cost one copy instead of three - and zero when
# the workers already hold their own copies.
#
# !!! tip "Reuse and compose"
#     Pipelines are immutable: [`add`][navis.Pipeline.add] and friends return a *new*
#     pipeline, so you can keep a base around and branch off it. `|` splices two
#     pipelines together.
#
#     ```python
#     clean = navis.Pipeline().heal_skeleton().prune_twigs(5000)
#     coarse = clean | navis.Pipeline().resample_skeleton(2000)
#     fine   = clean.resample_skeleton(200)
#     ```

# %%
# For a one-off chain, start from the [`NeuronList`][navis.NeuronList] itself and run it
# straight away:

res = nl.pipeline.heal_skeleton().prune_twigs(5000).run()
res

# %%
# ### Steps that aren't per-neuron
#
# By default a step is applied to each neuron individually if what reaches it is a
# `NeuronList`, and called once with the whole thing otherwise. Two methods override
# that:
#
# | Method | What it does |
# |--------|--------------|
# | [`add()`][navis.Pipeline.add] | Per neuron if the value is a `NeuronList`, else one call |
# | [`add_each()`][navis.Pipeline.add_each] | Always map over the elements of the value |
# | [`add_once()`][navis.Pipeline.add_once] | Always one call, with the whole value |
#
# Use `add_once` for functions that work across a whole list at a time -
# [`navis.xform_brain`][] pools all the coordinates and transforms them in one go, so
# handing it neurons one by one would be much slower. The fluent equivalents are
# `pipe.once.xform_brain(...)` and `pipe.each.<func>(...)`.
#
# ### Anything can be a step - and anything can be the input
#
# Steps do not have to be {{ navis }} functions: [`add()`][navis.Pipeline.add] takes any
# callable. And the input is simply whatever the first step accepts - it does not have
# to be neurons at all. Here a query object goes in, the first step turns it into
# neurons, and everything after that runs per neuron across all your cores:
#
# ```python
# import navis.interfaces.neuprint as neu
#
# client = neu.Client("https://neuprint-cns.janelia.org", "male-cns:v1.0")
# nc = neu.NeuronCriteria(class_="ALPN", somaSide="R")
#
# pipe = navis.Pipeline(neu.fetch_mesh_neuron).skeletonize().resample_skeleton(500)
# res = pipe(nc, parallel=True, n_cores=5)
# ```
#
# !!! note "What actually runs in parallel"
#     Only the per-neuron segments. The `fetch_mesh_neuron` call above is a single
#     query and runs in this process; `skeletonize` and `resample_skeleton` are fused
#     and spread over the five cores. If you want the fetching parallelized too, hand
#     the pipeline a list and use [`add_each()`][navis.Pipeline.add_each].

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
# ### Sending an NBLAST to a cluster
#
# [`navis.nblast`][] and its relatives have no `parallel` switch - they are always
# parallel, over `n_cores` - but they follow [`navis.set_parallel_backend`][] just the
# same. The only difference is what a unit of work *is*: {{ navis }} cuts the
# query :octicons-arrow-right-24: target matrix into blocks and sends those, rather than
# sending one neuron at a time. Blocks are sized by a runtime budget, so each is seconds
# to minutes of work however many neurons you have:
#
# ```python
# with navis.set_parallel_backend(client):
#     scores = navis.nblast(query, target, backend="builtin")
# ```
#
# How finely the matrix is cut is decided from the *cluster's* worker count where
# {{ navis }} can read it, so `n_cores` on your laptop doesn't cap it.
#
# !!! warning "navis-fastcore does not distribute"
#     The [navis-fastcore](https://github.com/schlegelp/fastcore-rs) NBLAST backend
#     computes the whole matrix in a single Rust call using its own threads, so it ignores
#     the parallel backend completely: select it, point {{ navis }} at a cluster, and all
#     the work still happens on the machine you are sitting at. It is not the default -
#     but it *is* what `navis.config.default_nblast_backend = "auto"` picks. Leave that at
#     `"builtin"`, or pass `backend="builtin"` per call, when you want the NBLAST spread
#     over the cluster.

# %%

# mkdocs_gallery_thumbnail_path = '_static/multiprocess.png'