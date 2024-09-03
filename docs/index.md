---
icon: material/home
---

![logo](_static/logo_new.png)

# <span style="color:rgb(255,190,40);font-weight:bold">N</span>euron <span style="color:rgb(255,190,40);font-weight:bold">A</span>nalysis and <span style="color:rgb(255,190,40);font-weight:bold">Vis</span>ualization

[![Documentation Status](https://readthedocs.org/projects/navis/badge/?version=latest)](http://navis.readthedocs.io/en/latest/?badge=latest) [![Tests](https://github.com/navis-org/navis/actions/workflows/test-package.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/test-package.yml) [![Run notebooks](https://github.com/navis-org/navis/actions/workflows/notebooktest-package.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/notebooktest-package.yml) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navis-org/navis/blob/master/examples/colab.ipynb) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8191725.svg)](https://zenodo.org/doi/10.5281/zenodo.4699382) [![Downloads](https://pepy.tech/badge/navis)](https://pepy.tech/project/navis)

{{ navis }} is a Python library for analysis and visualization of neuron
morphology. It stands on the shoulders of the excellent
[`natverse`](http://natverse.org) for R.

---

**[Features](#features)** - **[Quickstart](quickstart.md)** - **[Installation](installation.md)**


## Features

<div class="grid cards" markdown>

-   :material-progress-wrench:{ .lg .middle } __Polgyglot__

    ---

    We support all kinds of neuron data: skeletons, meshes, dotprops, (confocal) images.

-   :material-progress-wrench:{ .lg .middle } __Interactive__

    ---

    Designed to let you explore your data interactively from Jupyter notebooks,
    terminal or via scripts.

-   :material-progress-wrench:{ .lg .middle } __Analysis__

    ---

    Calculate Strahler indices, cable length, volume, tortuosity and other morphometrics.
    Run NBLASTs and other

-   :material-progress-wrench:{ .lg .middle } __Visualization__

    ---

    Generate beautiful publication-ready 2D (matplotlib) and 3D (octarine,
    vispy or plotly) figures.

-   :material-progress-wrench:{ .lg .middle } __Processing__

    ---

    Smoothing, resampling, skeletonization, meshing and more!

-   :material-progress-wrench:{ .lg .middle } __Fast__

    ---

    Scalable thanks to out-of-the-box support for multiprocessing.

-   :material-progress-wrench:{ .lg .middle } __Clustering__

    ---

    Cluster your neurons by e.g. morphology using NBLAST.

-   :material-progress-wrench:{ .lg .middle } __Transform__

    ---

    Fully featured transform system to move neurons between brain spaces.
    We support e.g. CMTK or Elastix.

-   :material-progress-wrench:{ .lg .middle } __Import/Export__

    ---

    Read and write from/to SWC, NRRD, Neuroglancer's precomputed format,
    OBJ, STL and more!

-   :material-progress-wrench:{ .lg .middle } __Online__

    ---

    Download neurons straight from Allen's
    [MICrONS](https://www.microns-explorer.org/) datasets,
    [neuromorpho](http://neuromorpho.org) or [neuPrint](neuprint_intro.md).

-   :material-progress-wrench:{ .lg .middle } __Interfaces__

    ---

    Load neurons into Blender 3D, simulate neurons and networks using
    NEURON, or use the R natverse library via `rpy2`.

-   :material-progress-wrench:{ .lg .middle } __Extensible__

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