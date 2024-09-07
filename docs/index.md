---
icon: material/home
hide:
  - navigation
  - toc
---

![logo](_static/logo_new_banner.png)

# <span style="color:rgb(255,190,40);font-weight:bold">N</span>euron <span style="color:rgb(255,190,40);font-weight:bold">A</span>nalysis and <span style="color:rgb(255,190,40);font-weight:bold">Vis</span>ualization

[![Docs](https://github.com/navis-org/navis/actions/workflows/build-docs.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/build-docs.yml) [![Tests](https://github.com/navis-org/navis/actions/workflows/test-package.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/test-package.yml) [![Test tutorials](https://github.com/navis-org/navis/actions/workflows/test-tutorials.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/test-tutorials.yml) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navis-org/navis/blob/master/examples/colab.ipynb) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8191725.svg)](https://zenodo.org/doi/10.5281/zenodo.4699382) [![Downloads](https://pepy.tech/badge/navis)](https://pepy.tech/project/navis)

{{ navis }} is a Python library for analysis and visualization of neuron
morphology. It stands on the shoulders of the excellent
[`natverse`](http://natverse.org) for R.

---

**[Features](#features)** - **[Quickstart](quickstart.md)** - **[Installation](installation.md)**


## Features

<div class="grid cards" markdown>

-   :simple-databricks:{ .lg .middle } __Polgyglot__

    ---

    Support for all kinds of [neuron types](generated/gallery/plot_01_neurons_intro): skeletons, meshes, dotprops and images.

-   :material-eye:{ .lg .middle } __Exploration__

    ---

    Designed to let you explore your data interactively from Jupyter notebooks,
    terminal or via scripts.

-   :fontawesome-solid-circle-notch:{ .lg .middle } __Analysis__

    ---

    Calculate Strahler indices, cable length, volume, tortuosity, NBLAST
    and many other [morphometrics](generated/gallery/2_morpho/plot_01_morpho_analyze).

-   :fontawesome-solid-brush:{ .lg .middle } __Visualization__

    ---

    Generate beautiful publication-ready 2D (matplotlib) and 3D (octarine,
    vispy or plotly) [figures](generated/gallery/#plotting).

-   :material-progress-wrench:{ .lg .middle } __Processing__

    ---

    Smoothing, resampling, skeletonization, meshing and [more](api.md#neuron-morphology)!

-   :fontawesome-solid-computer:{ .lg .middle } __Fast__

    ---

    Uses compiled Rust code under-the-hood. Also scale thanks to
    out-of-the-box support for [multiprocessing](generated/gallery/6_misc/plot_00_misc_multiprocess).

-   :material-lightbulb-group:{ .lg .middle } __Clustering__

    ---

    Cluster your neurons by e.g. morphology using [NBLAST](generated/gallery/5_nblast/plot_00_nblast_intro).

-   :material-move-resize:{ .lg .middle } __Transforms__

    ---

    Fully featured [transform system](generated/gallery/5_transforms/plot_00_transforms) to move neurons between brain spaces.
    We support e.g. CMTK or Elastix.

-   :octicons-file-directory-symlink-24:{ .lg .middle } __Import/Export__

    ---

    Read and write from/to SWC, NRRD, Neuroglancer's precomputed format,
    OBJ, STL and [more](generated/gallery/#import-export)!

-   :octicons-globe-24:{ .lg .middle } __Connected__

    ---

    Load neurons straight from Allen's
    [MICrONS](generated/gallery/4_remote/plot_02_remote_microns) datasets,
    [neuromorpho](http://neuromorpho.org), [neuPrint](generated/gallery/4_remote/plot_00_remote_neuprint)
    or any NeuroGlancer source.

-   :material-connection:{ .lg .middle } __Interfaces__

    ---

    Load neurons into [Blender 3D](generated/gallery/3_interfaces/plot_01_interfaces_blender), simulate neurons and networks using
    [NEURON](generated/gallery/3_interfaces/plot_00_interfaces_neuron), or use the R natverse library via `rpy2`.

-   :material-google-circles-extended:{ .lg .middle } __Extensible__

    ---

    Write your own library built on top of NAVis functions. See
    our [ecosystem](ecosystem.md) for examples.

</div>

Check out the [Tutorials](generated/gallery/) and [API reference](api.md) to see
what you can do with {{ navis }}.

Need help? Use [discussions](https://github.com/navis-org/navis/discussions)
on Github to ask questions!

{{ navis }} is licensed under the GNU GPL v3+ license. The source code is hosted
at [Github](https://github.com/navis-org/navis). Feedback, feature requests
and bug reports are very welcome and best placed in a
[Github issue](https://github.com/navis-org/navis/issues)