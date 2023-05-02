[![Documentation Status](https://readthedocs.org/projects/navis/badge/?version=latest)](http://navis.readthedocs.io/en/latest/?badge=latest) [![Tests](https://github.com/navis-org/navis/actions/workflows/test-package.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/test-package.yml) [![Run notebooks](https://github.com/navis-org/navis/actions/workflows/notebooktest-package.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/notebooktest-package.yml) [![Coverage Status](https://coveralls.io/repos/github/navis-org/navis/badge.svg?branch=master)](https://coveralls.io/github/navis-org/navis?branch=master) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navis-org/navis/blob/master/examples/colab.ipynb) [![DOI](https://zenodo.org/badge/168142416.svg)](https://zenodo.org/badge/latestdoi/168142416) [![Downloads](https://pepy.tech/badge/navis)](https://pepy.tech/project/navis)

<img src="https://github.com/navis-org/navis/raw/master/docs/_static/favicon.png" height="60">

NAVis is a Python 3 (3.8 or later) library for **N**euron **A**nalysis and **Vis**ualization.

## Documentation
NAVis is on [ReadTheDocs](http://navis.readthedocs.io/ "NAVis ReadTheDocs").

## Features
* works as Jupyter notebook, script or from terminal
* support for various neuron types: **skeletons**, **meshes**, **dotprops**, **voxels**
* 2D (matplotlib) and 3D (vispy, plotly or k3d) **plotting**
* neuron **surgery**: cutting, stitching, pruning, rerooting, intersections, ...
* **morphometrics**: Strahler analysis, cable length, volume, tortuosity, ...
* compare & cluster by morphology (e.g. **NBLAST**, persistence, form factor) and connectivity
* **transform** data between template brains (support for e.g. HDF5, CMTK, Elastix and thin plate spline transforms)
* load neurons directly from [neuPrint](https://neuprint.janelia.org), [neuromorpho.org](http://neuromorpho.org) and others
* simulate neurons and networks using the **NEURON** simulator
* interface with **Blender 3D** for high quality [renderings](https://youtu.be/wl3sFG7WQJc)
* interface with **R** neuron libraries (e.g. [nat](https://github.com/jefferis/nat), [rcatmaid](https://github.com/jefferis/rcatmaid), [elmr](https://github.com/jefferis/elmr))
* import-export from/to **SWC**, neuroglancer's ["**precomputed**"](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) format and more
* scalable thanks to out-of-the-box support for multiprocessing
* designed to be **extensible** - see for example [pymaid](https://pymaid.readthedocs.io/en/latest/)

## Getting started
See the [documentation](http://navis.readthedocs.io/ "NAVis ReadTheDocs") for detailed installation instructions, tutorials and examples. For the impatient:

```sh
pip3 install 'navis[all]'
```

which includes all optional extras providing features and/or performance improvements.
Currently, this is
`igraph`,
`pathos`,
`shapely`,
`kdtree`,
`hash`,
`flybrains`,
`cloudvolume`,
`meshes`,
and `vispy-default`.

3D plotting from a python REPL is provided by `vispy`, which has a choice of backends.
Different backends work best on different combinations of hardware, OS, python distribution, and REPL, so there may be some trial and error involved.
`vispy`'s backends are [listed here](https://vispy.org/installation.html#backend-requirements), and each can be installed as a navis extra, e.g. `pip3 install 'navis[vispy-pyqt6]'`.

![movie](https://user-images.githubusercontent.com/7161148/114312307-28a72700-9aea-11eb-89a6-ee1d72bfa730.mov)

## Changelog

A summary of changes can be found
[here](https://navis.readthedocs.io/en/latest/source/whats_new.html).

## NAVis & friends
<p align="center">
<img src="https://github.com/navis-org/navis/blob/master/docs/_static/navis_ecosystem.png?raw=true" width="700">
</p>

NAVis comes with batteries included but is also highly extensible. Some
libraries built on top of NAVis:
* [flybrains](https://github.com/navis-org/navis-flybrains) provides templates and transforms for *Drosophila* brains to use with navis
* [pymaid](https://pymaid.readthedocs.io/en/latest/) pulls and pushes data from/to CATMAID servers
* [fafbseg](https://fafbseg-py.readthedocs.io/en/latest/index.html) contains tools to work with auto-segmented data for the FAFB EM dataset

## License
This code is under GNU GPL V3

## Acknowledgments
NAVis is inspired by and inherits much of its design from the excellent
[natverse](http://natverse.org) R packages by
[Greg Jefferis](https://github.com/jefferis), [Alex Bates](https://github.com/alexanderbates),
[James Manton](https://github.com/ajdm) and others.

## References
NAVis implements or provides interfaces with algorithms described in:

1. **Comparison of neurons based on morphology**: Neuron. 2016 doi: 10.1016/j.neuron.2016.06.012
*NBLAST: Rapid, Sensitive Comparison of Neuronal Structure and Construction of Neuron Family Databases.*
Costa M, Manton JD, Ostrovsky AD, Prohaska S, Jefferis GSXE.
[link](https://www.cell.com/neuron/fulltext/S0896-6273(16)30265-3?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627316302653%3Fshowall%3Dtrue)
2. **Comparison of neurons based on connectivity**: Science. 2012 Jul 27;337(6093):437-44. doi: 10.1126/science.1221762.
*The connectome of a decision-making neural network.*
Jarrell TA, Wang Y, Bloniarz AE, Brittin CA, Xu M, Thomson JN, Albertson DG, Hall DH, Emmons SW.
[link](http://science.sciencemag.org/content/337/6093/437.long)
3. **Comparison of neurons based on synapse distribution**: eLife. doi: 10.7554/eLife.16799
*Synaptic transmission parallels neuromodulation in a central food-intake circuit.*
Schlegel P, Texada MJ, Miroschnikow A, Schoofs A, Hückesfeld S, Peters M, … Pankratz MJ.
[link](https://elifesciences.org/content/5/e16799)
4. **Synapse flow centrality and segregation index**: eLife. doi: 10.7554/eLife.12059
*Quantitative neuroanatomy for connectomics in Drosophila.*
Schneider-Mizell CM, Gerhard S, Longair M, Kazimiers T, Li, Feng L, Zwart M … Cardona A.
[link](https://elifesciences.org/articles/12059)

## Contributing

1. Fork this repository
2. `git clone` it to your local machine
3. Install the full development dependencies with `pip install -r requirements.txt`
4. Install the package in editable mode with `pip install -e ".[all]"`
5. Create, `git add`, `git commit`, `git push`, and pull request your changes.

Run the tests locally with `pytest -v`.

Docstrings should use the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format,
and make sure you include any relevant links and citations.
Unit tests should be [doctests](https://docs.python.org/3/library/doctest.html)
and/or use [pytest](https://docs.pytest.org/en/stable/) in the `./tests` directory.

Doctests have access to the `tmp_dir: pathlib.Path` variable,
which should be used if any files need to be written.
