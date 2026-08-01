---
icon: material/tools
hide:
  - navigation
---

## Installing NAVis

{{ navis }} requires Python 3.10 or later. The instructions below assume that
you have already installed Python and its package manager [`pip`](https://pypi.org/project/pip/).

Open up a terminal and install {{ navis }} with:


=== "Full Install"

    This is the "batteries included" install that will install {{ navis }} plus a number of
    extra dependencies that are just nice to have.

    ``` sh
    pip install navis[all] -U
    ```

    If you run into issues, try the [minimal install](#__tabbed_1_2) instead.


=== "Minimal"

    If you're running into issues with the [full install](#__tabbed_1_1),
    you can try this minimal install instead:

    ``` sh
    pip install navis -U
    ```

    If you go down this route some functions in {{ navis }} might complain about missing dependencies.
    No worries though: they should also tell you how to install them. See also the section on
    [Optional Dependencies](#optional-dependencies) below.


=== "Dev"

    To install the latest version from Github:

    ``` sh
    pip install git+https://github.com/navis-org/navis@master
    ```

    To install the latest dev with extras:

    ``` sh
    pip install "navis[all] @ git+https://github.com/navis-org/navis@master"
    ```


!!! note

    MacOS (both Intel and the new ARM chips) and Linux should work off the bat without any problems.
    On Windows, you might run into issues with some of the dependencies. If that happens, we recommend you check
    out the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) (WSL).

!!! info "navis-fastcore is required"

    Since 2.0, [navis-fastcore](https://github.com/schlegelp/fastcore-rs) - the Rust
    engine behind {{ navis }}' graph, geodesic and NBLAST core - is a **core**
    dependency rather than an optional extra. Both installs above pull it in.

    It ships prebuilt wheels for macOS (Intel and ARM), Windows, and Linux on
    x86-64, aarch64, i686, armv7l, ppc64le and s390x, so there is nothing to
    compile. The exception is musl-based Linux (e.g. Alpine), which builds from
    the source distribution and therefore needs a Rust toolchain.


  [PyPI]: https://pypi.org/project/navis/
  [virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment
  [Markdown]: https://python-markdown.github.io/
  [Using Python's pip to Manage Your Projects' Dependencies]: https://realpython.com/what-is-pip/


!!! info "By the way"

    You can use NAVis without having to install anything on your local machine!
    Follow this [link](https://colab.research.google.com/github/navis-org/navis/blob/master/examples/colab.ipynb)
    to open an example notebook in Google's Colaboratory.


## Optional dependencies

If you installed {{ navis }} using the "batteries-included" `[all]` option, you can ignore this section.

If you opted for the minimal install, you might want to consider manually adding some of these
optional dependencies as they provide speed-boosts in certain situations or
are required for certain functions.

The user-facing extras, the dependencies they install, and how to install those dependencies
directly, are listed below:

??? tip "Performance"

    These dependencies aren't strictly necessary but will speed up certain operations:

    ---


    #### `kdtree`: [pykdtree](https://github.com/storpipfugl/pykdtree)

    Faster than scipy's cKDTree implementation. If available, will be used to
    speed up e.g. NBLAST.

    ``` shell
    pip install pykdtree
    ```

    ---

    #### `joblib`: [joblib](https://joblib.readthedocs.io)

    The parallel backend {{ navis }} prefers where it is installed (see
    [`navis.set_parallel_backend`][]). It serializes with `cloudpickle`, so it can
    run lambdas and functions defined in a notebook, and it keeps its worker
    processes alive between calls - which makes a sequence of `parallel=True` calls
    markedly faster than rebuilding a pool each time. `joblib.parallel_config` also
    lets you route the work through dask, ray or ipyparallel.

    Parallel processing works without it - {{ navis }} falls back to the standard
    library's process pool - but that one cannot ship lambdas.

    ``` shell
    pip install joblib
    ```

    ---

    #### `pathos`: [pathos](https://github.com/uqfoundation/pathos)

    An alternative parallel backend (see [`navis.set_parallel_backend`][]). Like
    `joblib` it serializes with `dill` and so can ship lambdas, but it builds a
    fresh worker pool for every call, which makes it the slower of the two.

    ``` shell
    pip install pathos
    ```

    ---

    #### `hash`: [xxhash](https://cyan4973.github.io/xxHash/)

    For speeding up some lookup tables.

    ``` shell
    pip install xxhash
    ```

    ---

    #### ``meshes``: [open3d](https://pypi.org/project/open3d/), [pyfqmr](https://github.com/Kramer84/pyfqmr-Fast-quadric-Mesh-Reduction)
      Assorted functionality associated with meshes. ``pyfqmr`` in particular is highly recommended if you want to downsample meshes.

    ``` shell
    pip install open3d pyfqmr
    ```

??? info "Cluster computing"

    For spreading `parallel=True` across more than one machine (see
    [`navis.set_parallel_backend`][]). Neither is part of `navis[all]` - install
    them explicitly with `pip install navis[cluster]` or individually:

    ---

    #### `dask`: [dask.distributed](https://distributed.dask.org)

    Runs the work on a dask cluster - anything from a `LocalCluster` on your
    laptop to a `SLURMCluster`/`KubeCluster` spanning a compute centre. Hand
    {{ navis }} the `Client` and it will size the units of work against the
    cluster and send the neurons straight to the workers.

    ``` shell
    pip install "dask[distributed]"
    ```

    ---

    #### `submitit`: [submitit](https://github.com/facebookincubator/submitit)

    Submits the work to a scheduler - SLURM, or your local machine for a dry
    run - as an array job. Use this where you would otherwise write a batch
    script: nothing has to stay connected while the jobs sit in the queue.

    ``` shell
    pip install submitit
    ```

??? example "Visualization"

    {{ navis }} supports various different backends for 2D and 3D visualization. For 2D visualizations we
    use `matplotlib` by default which is installed automatically. For 3D visualizations, you can use
    `octarine3d`, `plotly` or `k3d` backends.

    ---

    #### `octarine3d`: [octarine3d](https://schlegelp.github.io/octarine/)

    For 3D visualisation in terminal and Jupyter notebooks.

    Octarine3d is a modern, high-performance, WGPU-based viewer for interactive 3D visualisation of neurons
    and is the default 3D viewer for {{ navis }}. By default, `navis[all]` will install `octarine3d` with
    standard windows manager `pyside6` and Jupyter notebook manager `jupyter_rfb`. It will also
    install the `navis-octarine-plugin` which is required to use `octarine3d` as a viewer for {{ navis }}.
    This is equivalent to the following command:

    ``` shell
    pip install octarine3d[all] octarine-navis-plugin
    ```

    Please see `octarine3d` [installation instructions](https://schlegelp.github.io/octarine/install/)
    for information on how to choose a different backend.

    !!! note

        Older systems (pre ~2018) might not support WGPU. If you are running into issues try updating your
        operating system and/or your graphics drivers. Failing that, `plotly` works without a GPU.

    ---

    #### `plotly`: [plotly](https://plotly.com/python/)

    For 3D visualisation in Jupyter notebooks.

    ``` shell
    pip install plotly
    ```

    ---

    #### `k3d`: [k3d](https://k3d-jupyter.org/)

    For 3D visualisation in Jupyter notebooks.

    ``` shell
    pip install k3d
    ```

??? question "Miscellaneous"

    #### `shapely`: [Shapely](https://shapely.readthedocs.io/en/latest/)

    This is used to get 2D outlines of `navis.Volumes` when plotting in 2D
    with ``volume_outlines=True``.

    ``` shell
    pip install shapely
    ```

    ---

    #### `flybrains`: [flybrains](https://github.com/navis-org/navis-flybrains)

    Transforming data between some template *Drosophila* brains.

    ``` shell
    pip install flybrains
    ```

    ---

    #### `cloudvolume`: [cloud-volume](https://github.com/seung-lab/cloud-volume)

    Reading and writing images, meshes, and skeletons in Neuroglancer precomputed format.
    This is required for e.g. the MICrONs interface.

    ``` shell
    pip install cloud-volume
    ```

These extras can be installed directly (see instructions above), or alongside {{ navis }} with

``` shell
pip install navis[extra1,extra2]
```



## What next?

<div class="grid cards" markdown>

-   :octicons-feed-rocket-16:{ .lg .middle } __Quickstart__
    ---

    Check out the quickstart tutorial for an overview of basic concepts in {{ navis }}.

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

-   :material-help-box-multiple-outline:{ .lg .middle } __Tutorials__
    ---

    Check out the tutorials!

    [:octicons-arrow-right-24: Tutorials](../generated/gallery/)

</div>
