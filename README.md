[![Documentation Status](https://readthedocs.org/projects/navis/badge/?version=latest)](http://navis.readthedocs.io/en/latest/?badge=latest) [![Tests](https://github.com/schlegelp/navis/actions/workflows/test-package.yml/badge.svg)](https://github.com/schlegelp/navis/actions/workflows/test-package.yml) [![Run notebooks](https://github.com/schlegelp/navis/actions/workflows/notebooktest-package.yml/badge.svg)](https://github.com/schlegelp/navis/actions/workflows/notebooktest-package.yml) [![Coverage Status](https://coveralls.io/repos/github/schlegelp/navis/badge.svg?branch=master)](https://coveralls.io/github/schlegelp/navis?branch=master) [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/schlegelp/navis/master?urlpath=tree)

<img src="https://github.com/schlegelp/navis/raw/master/docs/_static/favicon.png" height="60">


NAVis is a Python 3 (3.7 or later) library for **N**euron **A**nalysis and **Vis**ualization.

## Documentation
NAVis is on [ReadTheDocs](http://navis.readthedocs.io/ "NAVis ReadTheDocs").

## Features
* work with various neuron types: **skeletons**, **meshes**, **dotprops**
* 2D (matplotlib) and 3D (vispy or plotly) **plotting**
* neuron **surgery**: cutting, stitching, pruning, rerooting, intersections, ...
* analyze morphology (e.g. **NBLAST**) and connectivity
* **transform** data between template brains (support for e.g. HDF5, CMTK and thin plate spline transforms)
* load neurons directly from [neuPrint](https://neuprint.janelia.org) and [neuromorpho.org](http://neuromorpho.org)
* interface with **Blender 3D**
* interface with **R** neuron libraries (e.g. [nat](https://github.com/jefferis/nat), [rcatmaid](https://github.com/jefferis/rcatmaid), [elmr](https://github.com/jefferis/elmr))
* import-export from/to **SWC**
* designed to be **extensible** - see for example [pymaid](https://pymaid.readthedocs.io/en/latest/)

## Getting started
See the [documentation](http://navis.readthedocs.io/ "NAVis ReadTheDocs") for detailed installation instructions, tutorials and examples. For the impatient:

```sh
pip3 install navis[all]
```

which includes all optional extras providing features and/or performance improvements.
Currently, this is just `igraph`.

Alternatively click on the *launch binder* badge above to try out navis hosted by [mybinder](https://mybinder.org)!

![movie](https://user-images.githubusercontent.com/7161148/114312307-28a72700-9aea-11eb-89a6-ee1d72bfa730.mov)

## NAVis & friends
<p align="center">
<img src="https://github.com/schlegelp/navis/blob/master/docs/_static/navis_ecosystem.png?raw=true" width="700">
</p>

NAVis comes with batteries included but is also highly extensible. Some
libraries built on top of NAVis:
* [flybrains](https://github.com/schlegelp/navis-flybrains) provides templates and transforms to use with navis
* [pymaid](https://pymaid.readthedocs.io/en/latest/) pulls and pushes data from/to CATMAID servers
* [fafbseg](https://fafbseg-py.readthedocs.io/en/latest/index.html) contains tools to work with autosegmented data for the FAFB EM dataset

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
4. Install the package in editable mode with `pip install -e .[all]`
5. Create, `git add`, `git commit`, `git push`, and pull request your changes.

Run the tests locally with `pytest -v`.

Docstrings should use the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format,
and make sure you include any relevant links and citations.
Unit tests should be [doctests](https://docs.python.org/3/library/doctest.html)
and/or use [pytest](https://docs.pytest.org/en/stable/) in the `./tests` directory.

Doctests have access to the `tmp_dir: pathlib.Path` variable,
which should be used if any files need to be written.
