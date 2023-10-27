[![Documentation Status](https://readthedocs.org/projects/navis/badge/?version=latest)](http://navis.readthedocs.io/en/latest/?badge=latest) [![Tests](https://github.com/navis-org/navis/actions/workflows/test-package.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/test-package.yml) [![Run notebooks](https://github.com/navis-org/navis/actions/workflows/notebooktest-package.yml/badge.svg)](https://github.com/navis-org/navis/actions/workflows/notebooktest-package.yml) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navis-org/navis/blob/master/examples/colab.ipynb) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8191725.svg)](https://zenodo.org/doi/10.5281/zenodo.4699382) [![Downloads](https://pepy.tech/badge/navis)](https://pepy.tech/project/navis)

<img src="https://github.com/navis-org/navis/raw/master/docs/_static/logo_new.png" height="120">

NAVis is a Python 3 library for **N**euron **A**nalysis and **Vis**ualization.

## Documentation
NAVis is on [ReadTheDocs](http://navis.readthedocs.io/ "NAVis ReadTheDocs").

## Features
* **polyglot**: ``navis`` works with skeletons, meshes, dotprops and image data
* **visualize**: 2D (matplotlib) and 3D (vispy, plotly or k3d)
* **process**: skeletonization, meshing, smoothing, repairing, downsampling, etc.
* **edit**: e.g. cutting, stitching, pruning, rerooting or intersecting
* **morphometrics**: Strahler analysis, cable length, volume, tortuosity and more
* **similarity**: compare & cluster by morphology (e.g. NBLAST, persistence or form factor) or connectivity metrics
* **transform**: move data between template brains (built-in support for HDF5, CMTK, Elastix and landmark-based transforms)
* **interface**: load neurons directly from [neuPrint](https://neuprint.janelia.org), [neuromorpho.org](http://neuromorpho.org) and other data sources
* **simulate** neurons and networks using the *NEURON* simulator
* **render**: use Blender 3D for high quality [visualizations](https://youtu.be/wl3sFG7WQJc)
* **R** neuron libraries: interfaces with [nat](https://github.com/jefferis/nat), [rcatmaid](https://github.com/jefferis/rcatmaid), [elmr](https://github.com/jefferis/elmr) and more
* **import-export**: read/write SWCs, neuroglancer's ["*precomputed*"](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) format, NMX/NML, NRRD, mesh-files and more
* **scalable**: out-of-the-box support for multiprocessing
* **extensible**: build your own package on top of navis - see [pymaid](https://pymaid.readthedocs.io/en/latest/) for example

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

## Questions?
Questions on how to use `navis` are best placed in [discussions](https://github.com/navis-org/navis/discussions). Same goes for cool projects or analyses you made using `navis` -
we'd love to hear from you!

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
* [fafbseg](https://fafbseg-py.readthedocs.io/en/latest/index.html) contains tools to work with auto-segmented data for the FAFB EM dataset including FlyWire

## Who uses NAVis?
NAVis has been used in a range of neurobiological publications. Here's a semi-random sample
of recent papers:

```
BIFROST: a method for registering diverse imaging datasets, Brezovec et al., bioRxiv (2023); doi: https://doi.org/10.1101/2023.06.09.544408

Vimo: Visual Analysis of Neuronal Connectivity Motifs, Troidl et al., bioRxiv (2022); doi: https://doi.org/10.1101/2022.12.09.519772

Information flow, cell types and stereotypy in a full olfactory connectome, Schlegel, Bates et al., eLife (2021); doi: https://doi.org/10.7554/eLife.66018

Synaptic wiring motifs in posterior parietal cortex support decision-making, Kuan et al., bioRxiv (2022); doi: https://doi.org/10.1101/2022.04.13.488176
```

Want to see your work listed here? Open an [Issue](https://github.com/navis-org/navis/issues) and tell us about it!

## Citing NAVis
We'd love to know if you found NAVis useful for your research! You can help us
spread the word by citing the DOI provided by Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8191725.svg)](https://zenodo.org/doi/10.5281/zenodo.4699382)

## License
This code is under [GNU GPL V3](LICENSE).

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

Want to contribute? Great, here is how!

### Report bugs or request features
Open an [issue](https://github.com/navis-org/navis/issues). For bug reports
please make sure to include some code/data with a minimum example for us to
reproduce the bug.

### Contribute code
We're always happy for people to contribute code - be it a small bug fix, a
new feature or improved documentation/tutorials.

Here's how you'd do it in a nutshell:

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

Feel free to get in touch either through an [issue](https://github.com/navis-org/navis/issues) or [discussion](https://github.com/navis-org/navis/discussions) if you need
pointers or input on how to implement an idea.
