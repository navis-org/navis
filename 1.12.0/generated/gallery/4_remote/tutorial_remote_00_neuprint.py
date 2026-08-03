"""
neuPrint
========

This tutorial shows how to fetch neurons from a neuPrint server.

[NeuPrint](https://www.biorxiv.org/content/10.1101/2020.01.16.909465v1) is a service for presenting and analyzing connectomics data.
It is used to host, for example, the Janelia EM reconstructions from a *Drosophila* hemibrain at <https://neuprint.janelia.org/>.

[neuprint-python](https://github.com/connectome-neuprint/neuprint-python) is a Python library that lets you query data directly
from a neuPrint server. You can install it from PyPI:

```shell
pip3 install neuprint-python
```

`navis.interfaces.neuprint` wraps `neuprint-python` and adds a few new functions to fetch and convert data into {{ navis }} objects.

"""

# Import navis
import navis

# Import neuprint wrapper by navis
import navis.interfaces.neuprint as neu

# %%
# First set up the connection:
# You can either pass your API token directly or store as `NEUPRINT_APPLICATION_CREDENTIALS` environment variable.
# The latter is the recommended way and we will use it here:

client = neu.Client(
    "https://neuprint.janelia.org/",
    # token="MYLONGTOKEN"  # use this to instead pass your token directly
    dataset="hemibrain:v1.2.1",
)

# %%
# You can use all of neuprint's functions:

mbons, roi_info = neu.fetch_neurons(
    neu.SegmentCriteria(instance=".*MBON.*", regex=True)
)
mbons.head(3)

# %%
# {{ navis }} has added three functions to `neu`:
#
#  - [`navis.interfaces.neuprint.fetch_roi`][]: returns a [`navis.Volume`][] from a ROI
#  - [`navis.interfaces.neuprint.fetch_skeletons`][]: returns fully fledged [`navis.TreeNeurons`][navis.TreeNeuron] - nodes, synapses, soma and all
#  - [`navis.interfaces.neuprint.fetch_mesh_neuron`][]: returns [`navis.MeshNeurons`][navis.MeshNeuron] - including synapses
#
# Let's start by fetching the mesh for the right mushroom body ROI:
mb = neu.fetch_roi("MB(R)")
mb

# %%
# Next, let's fetch the skeletons of all right MBONs:
mbon_skeletons = neu.fetch_skeletons(
    neu.SegmentCriteria(instance=".*MBON.*_R", regex=True), with_synapses=True
)
mbon_skeletons.head()

# %%
# Co-visualize the MBONs and the MB volume:
navis.plot3d(
    [mbon_skeletons[0], mb],
    legend=False,  # Hide the legend (more space for the plot)
)

# %%
# Last (but not least), let's make a 2d plot for the tutorial's thumbnail:
import matplotlib.pyplot as plt

fig, ax = navis.plot2d(
    [mbon_skeletons[0], mb],
    c=(0, 0, 0, 1),  # Make the neuron black
    method="3d",
    connectors=True,
    linewidth=0.5,  # Make neuron a bit thinner to emphasize the synapses
    view=("x", "-z"),
)

plt.tight_layout()

# %%
# All {{ navis }} functions for analysis & visualization should work on these neurons. If not, please open an issue on Github.

# mkdocs_gallery_thumbnail_path = '_static/neuprint_logo.png'
