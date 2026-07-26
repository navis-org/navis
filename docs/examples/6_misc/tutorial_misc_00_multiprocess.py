"""
Multiprocessing
===============

Speed up batch workloads with built-in parallel processing.

By default, most {{ navis }} functions use only a single thread/process (although some third-party functions
used under the hood might). Distributing expensive computations across multiple cores can speed things up considerably.

Many {{ navis }} functions natively support parallel processing. This notebook will illustrate various ways
to use parallelism. Before we get started: {{ navis }} uses `pathos` for multiprocessing - if you installed
{{ navis }} with `pip install navis[all]` you should be all set. If not, you can install `pathos` separately:

```shell
pip install pathos -U
```

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

# mkdocs_gallery_thumbnail_path = '_static/multiprocess.png'