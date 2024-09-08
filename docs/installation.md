---
icon: material/tools
hide:
  - navigation
---

## Installing NAVis

{{ navis }} requires Python 3.9 or later. The instructions below assume that
you have already installed Python and its package manager [`pip`](https://pypi.org/project/pip/).

!!! info "By the way"

    You can use NAVis without having to install anything on your local machine!
    Follow this [link](https://colab.research.google.com/github/navis-org/navis/blob/master/examples/colab.ipynb)
    to open an example notebook in Google's Colaboratory.


{{ navis }} is published as a [Python package] and can be installed with `pip`, ideally by using a [virtual environment].
Open up a terminal and install {{ navis }} with:


=== "Full Install"

    This is the "batteries included" install that will install {{ navis }} plus a number of
    extra dependencies that are just nice to have.

    ``` sh
    pip install navis[all] -U
    ```

    If you run into issues, try the [minimal install](navis_setup.md#__tabbed_1_2) instead.


=== "Minimal"

    If you're running into issues with the [full install](navis_setup.md#__tabbed_1_1),
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

!!! note

    MacOS (both Intel and the new ARM chips) and Linux should work off the bat without any problems.
    On Windows, you might run into issues with some of the dependencies. If that happens, we recommend you check
    out the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) (WSL).


  [Python package]: https://pypi.org/project/navis/
  [virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment
  [Markdown]: https://python-markdown.github.io/
  [Using Python's pip to Manage Your Projects' Dependencies]: https://realpython.com/what-is-pip/


## Optional dependencies

If you installed {{ navis }} using the batteries-included `[all]` option, you can ignore this section.

If you opted for the minimal install, you might want to consider adding some of these
optional dependencies as they e.g. provide speed-boosts in certain situations or
are for certain functions.

These extras can be installed directly, or along with {{ navis }} with

``` shell
pip install navis[extra1,extra2]
```


The user-facing extras, the dependencies they install, and how to install those dependencies
directly, are listed below:

??? tip "Performance"

    These dependencies aren't strictly necessary but will speed up certain operations:

    ---

    #### `fastcore`: [navis-fastcore](https://github.com/schlegelp/fastcore-rs)

    `navis-fastcore` re-implements a bunch of low-level functions in Rust
    and wraps them in Python. {{ navis }} will use `fastcore` under-the-hood
    if it is available. This is a highly recommended extra as it can
    speed up operations such as geodesic distances, Strahler Index, pruning
    and other downstream functions by several orders of magnitude.

    ``` shell
    pip install navis-fastcore
    ```

    ---

    #### `kdtree`: [pykdtree](https://github.com/storpipfugl/pykdtree)

    Faster than scipy's cKDTree implementation. If available, will be used to
    speed up e.g. NBLAST.

    ``` shell
    pip install pykdtree
    ```

    ---

    #### `pathos`: [pathos](https://github.com/uqfoundation/pathos)

    Pathos is a multiprocessing library. {{ navis }} uses it to parallelize functions
    across lists of neurons.

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

??? example "Visualization"

    {{ navis }} supports various different backends for 2D and 3D visualization. For 2D visualizations we
    use `matplotlib` by default which is installed automatically. For 3D visualizations, you can use
    `octarine3d`, `vispy`, `plotly` or `k3d` backends.

    ---

    #### `octarine3d`: [octarine3d](https://schlegelp.github.io/octarine/)

    For 3D visualisation in terminal and Jupyter notebooks.

    Octarine3d is a modern, high-performance, WGPU-based viewer for 3D visualisation of neurons.
    It is the default 3D viewer for {{ navis }}. It is recommended to install it as it is the fastest, most
    feature-rich viewer available. By default, `navis[all]` will install `octarine3d` with
    standard windows manager `pyside6` and Jupyter notebook manager `jupyter_rfb`. It will also
    install the `navis-octarine-plugin` which is required to use octarine3d as a viewer for {{ navis }}.
    This is equivalent to the following command:

    ``` shell
    pip install octarine3d[all] octarine-navis-plugin
    ```

    Please see `octarine3d` [installation instructions](https://schlegelp.github.io/octarine/install/)
    for information on how to choose a different backend.

    !!! note

        Older systems (pre ~2018) might not support WGPU, in which case you might want to fall back
        to the ``vispy`` backend.

    ---

    #### `vispy-*` backends: [vispy](https://vispy.org)

    For 3D visualisation in terminal and Jupyter notebooks.

    Vispy provides a high-performance, OpenGL-based viewer for 3D data visualisation.
    Vispy itself has a choice of window managers: the one which works for you will depend on
    your operating system, hardware, other installed packages, and how you're using navis.
    The default, supplied with {{ navis }}' `vispy-default` extra, is `pyside6` (for use from the console)
    and `jupyter_rfb` (for use in Jupyter notebooks).
    Each of vispy's backends, listed [here](https://vispy.org/installation.html#backend-requirements)
    can be installed through vispy and its extras, or {{ navis }}' `vispy-*` extras.

    ``` shell
    pip install navis[vispy-pyqt5]
    # or
    pip install vispy[pyqt5]
    ```

    !!! note

        The Vispy backend is deprecated in favor of Octarine. We might still decide to keep it if people end
        up having problems with Octarine. Please get in touch on Github if that's the case.

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

    #### `r`: [Rpy2](https://rpy2.readthedocs.io/en/version_2.8.x/overview.html#installation)

    Provides interface with R. This allows you to use e.g. the [natverse](https://natverse.org)
    R packages. Note that this package is not installed automatically as it would fail if R is
    not already installed on the system. You have to install Rpy2 manually!

    ``` shell
    pip install rpy2
    ```

    ---

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
    This is required required for e.g. the MICrONs interface.

    ``` shell
    pip install cloud-volume
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