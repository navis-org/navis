---
icon: material/heart
---

# {{ navis }} & Friends

{{ navis }} comes with batteries included but is also highly extensible. Here are
some libraries that are built directly on top of {{ navis }}.

![NAVis ecosystem](_static/navis_ecosystem.png)

## flybrains

[flybrains](https://github.com/navis-org/navis-flybrains) is a package that
bundles fly template brains and transforms that {{ navis }} can use to map spatial
data (e.g. neurons) from one brain space to another. If you installed {{ navis }}
via ``pip`` with the ``[flybrains]`` option, you should already have this package.

```python
import navis
import flybrains  # importing registers the transforms with NAVis

# Plot one of the template brains
navis.plot2d(flybrains.JRC2018U)

# Transform neurons to another brain space
n = navis.example_neurons(3, kind='skeleton')
xf = navis.xform_brain(n, source='JRCFIB2018F', target='JRC2018F')
```

## pymaid

[pymaid](https://pymaid.readthedocs.io/en/latest/) provides an interface with
[CATMAID](https://catmaid.readthedocs.io/en/stable/) servers. It allows
you to pull data (neurons, connectivity) that can be directly plugged into
{{ navis }}. Conversely, you can also take {{ navis }} neurons and push them to a
CATMAID server. `pymaid` is a great example of how to extend {{ navis }}.

```python
import navis
import pymaid

# Connect to a public CATMAID server
rm = pymaid.CatmaidInstance(server="https://fafb.catmaid.virtualflybrain.org/", api_token=None)

# Fetch some neurons
nl = pymaid.get_neurons('annotation:Paper: Engert et al 2022')

# CATMAID neurons can be directly used in all NAVis functions
navis.plot2d(nl, radius=False)
```

## fafbseg

[fafbseg](https://fafbseg-py.readthedocs.io/en/latest/index.html) contains
tools to work with autosegmented data for the
[FAFB](https://www.temca2data.org) (full adult fly brain)
EM dataset. It brings together data from [FlyWire](https://flywire.ai/),
[Google's](http://fafb-ffn1.storage.googleapis.com/landing.html) segmentation
of FAFB and [synapse predictions](https://github.com/funkelab/synful) by
Buhmann et al. (2019).

```python
from fafbseg import flywire

# Grab a neuron mesh by its ID
n = flywire.get_mesh_neuron(720575940613091290)

# Skeletonize using NAVis
s navis.skeletonize(n)
```

## natverse

The [natverse](http://natverse.org/) is {{ navis }}'s equivalent in R. While we
are aiming for feature parity, it can be useful to access ``natverse`` functions
from Python. For this, {{ navis }} offers some convenience functions using the
R-Python interface ``rpy2``. Check out the [tutorial](../generated/gallery/).
