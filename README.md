[![Documentation Status](https://readthedocs.org/projects/navis/badge/?version=latest)](http://navis.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/schlegelp/navis.svg?branch=master)](https://travis-ci.org/schlegelp/navis) [![Coverage Status](https://coveralls.io/repos/github/schlegelp/navis/badge.svg?branch=master)](https://coveralls.io/github/schlegelp/navis?branch=master) [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/schlegelp/navis/master?urlpath=tree)

<img src="https://github.com/schlegelp/navis/raw/master/docs/_static/favicon.png" height="60">


NAVis is a Python 3 (3.6 or later) library for **N**euron **A**nalysis and **Vis**ualization
with focus on hierarchical tree-like neuron data.

## Documentation
NAVis is on [ReadTheDocs](http://navis.readthedocs.io/ "NAVis ReadTheDocs").

## Features
* fetch data directly from [neuprint](https://neuprint.janelia.org/), [insectbraindb](https://insectbraindb.org/) or [neuromorpho](http://neuromorpho.org)
* SWC import
* interactive 2D (matplotlib) and 3D (vispy or plotly) plotting of neurons
* virtual neuron surgery: cutting, pruning, rerooting
* clustering (e.g. by connectivity or synapse placement)
* Python bindings for R neuron libraries (e.g. [nat](https://github.com/jefferis/nat), [rcatmaid](https://github.com/jefferis/rcatmaid), [elmr](https://github.com/jefferis/elmr))
* interfaces with [Blender3d](https://www.blender.org) and [Cytoscape](https://cytoscape.org)

## Getting started
See the [documentation](http://navis.readthedocs.io/ "NAVis ReadTheDocs") for detailed installation instructions, tutorials and examples. For the impatient:

`pip3 install navis`

Alternatively click on the *launch binder* badge above to try out navis hosted by [mybinder](https://mybinder.org)!

## License
This code is under GNU GPL V3


## TO-DOs
NAVis is a generalization of [pymaid](https://github.com/schlegelp/pyMaid) and
some of the code is still being refactored. Basic functionality is implemented
but there are still some TO-DOs:

- [x] update example notebooks and docs
- [ ] update Cytoscape interface
- [x] update Blender interface to Blender 2.8
- [ ] write new tests

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
