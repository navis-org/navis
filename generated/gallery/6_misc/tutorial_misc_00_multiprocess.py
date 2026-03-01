"""
Multiprocessing
===============

This notebook will show you how to use parallel processing with `navis`.

By default, most {{ navis }} functions use only a single thread/process (although some third-party functions
used under the hood might). Distributing expensive computations across multiple cores can speed things up considerable.

Many {{ navis }} functions natively support parallel processing. This notebook will illustrate various ways
to use parallelism. Before we get start: {{ navis }} uses `pathos` for multiprocessing - if you installed
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
# !!! important
#     This documentation is built on Github Actions where the number of cores can be as low as 2. The speedup on
#     your machine should be more pronounced than what you see below. That said: parallel processing has some
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



