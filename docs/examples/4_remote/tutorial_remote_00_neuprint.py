"""
neuPrint
========
<!-- difficulty: intermediate -->

Query and fetch neurons and connectivity from a neuPrint server.

[NeuPrint](https://www.biorxiv.org/content/10.1101/2020.01.16.909465v1) is a service for presenting and analyzing connectomics data.
It is used to host, for example, the Janelia EM reconstructions from a *Drosophila* hemibrain at <https://neuprint.janelia.org/>.

[neuprint-python](https://github.com/connectome-neuprint/neuprint-python) is a Python library that lets you query data directly
from a neuPrint server. You can install it from PyPI:

```shell
pip3 install neuprint-python
```

`navis.interfaces.neuprint` wraps `neuprint-python` and adds a few new functions to fetch and convert data into {{ navis }} objects.

!!! warning "Requires network access and a token"
    This tutorial talks to a live neuPrint server, so it needs an internet connection. You will also need a
    neuPrint account and an API token - see the "Authentication" note below for how to set it.

"""
# %%
# Import navis
import navis

# Import neuprint wrapper by navis
import navis.interfaces.neuprint as neu

# %%
# ## Set up the connection
#
# !!! note "Authentication"
#     Pass your API token directly via `token=...`, or store it as a `NEUPRINT_APPLICATION_CREDENTIALS`
#     environment variable. The latter is the recommended approach and the one we use here.

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
# On top of neuprint-python's own functions, {{ navis }} adds three that return {{ navis }} objects:
#
# | Function | Returns |
# |----------|---------|
# | [`fetch_roi`][navis.interfaces.neuprint.fetch_roi] | a [`navis.Volume`][] from a ROI |
# | [`fetch_skeletons`][navis.interfaces.neuprint.fetch_skeletons] | fully-fledged [`navis.Skeletons`][navis.Skeleton] - nodes, synapses, soma and all |
# | [`fetch_mesh_neuron`][navis.interfaces.neuprint.fetch_mesh_neuron] | [`navis.Meshes`][navis.Mesh] - including synapses |
#
# Start by fetching the mesh for the right mushroom body ROI:
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
#
# *[EM]: Electron Microscopy.
# *[ROI]: Region of interest.

# mkdocs_gallery_thumbnail_path = '_static/neuprint_logo.png'
