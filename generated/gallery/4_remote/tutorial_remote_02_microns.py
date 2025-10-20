"""
The MICrONS Datasets
====================

In this tutorial we will explore the MICrONS datasets.

The [Allen Institute for Brain Science](https://alleninstitute.org/) in collaboration with Princeton University,
and Baylor College of Medicine released two large connecotmics dataset:

1. A "Cortical mm<sup>3</sup>" of mouse visual cortex. This one is broken into two portions: "65" and "35"
2. A smaller "Layer 2/3" dataset of mouse visual cortex.

All of these can be browsed via the [MICrONS Explorer](https://www.microns-explorer.org/) using neuroglancer.
These data are public and thanks to the excellent [`cloud-volume`](https://github.com/seung-lab/cloud-volume)
and [`caveclient`](https://github.com/CAVEconnectome/CAVEclient) libraries, developed by
William Silversmith, Forrest Collman, Sven Dorkenwald, Casey Schneider-Mizell and others, we can easily fetch
neurons and their connectivity.

For easier interaction, {{ navis }} ships with a small interface to these datasets. To use it, we will have to
make sure `caveclient` (and with it `cloud-volume`) is installed:

```shell
pip install caveclient cloud-volume -U
```

The first time you run below code, you might have to get and set a client secret. Simply follow the instructions
in the terminal and when in doubt, check out the section about authentication in the
[`caveclient` docs](https://caveconnectome.github.io/CAVEclient/tutorials/authentication/).

Let's get started:
"""
# %%
import navis
import navis.interfaces.microns as mi

# %%
# You will find that most functions in the interface accept a `datastack` parameter. At the time of writing, the available stacks are:
#
#  - `cortex65` (also called "minnie65") is the anterior portion of the cortical mm<sup>3</sup> dataset
#  - `cortex35` (also called "minnie35") is the (smaller) posterior portion of the cortical mm<sup>3</sup> dataset
#  - `layer 2/3` (also called "pinky") is the earlier, smaller cortical dataset
#
# If not specified, the default is `cortex65`. Both `cortex65` and `cortex35` always map to the most recent version of that dataset.
# You can use [`get_datastacks`](navis.interfaces.microns.get_datastacks) to see all available datastacks:
mi.get_datastacks()

# %%
# Let's start with some basic queries using the `caveclient` directly:

# Initialize the client for the 65 part of cortical mm^3 (i.e. "Minnie")
client = mi.get_cave_client(datastack="cortex65")

# Fetch available annotation tables
client.materialize.get_tables()

# %%
# These are the available public tables which we can use to fetch meta data. Let's check out `baylor_log_reg_cell_type_coarse_v1`.
# Note that there is also a `baylor_gnn_cell_type_fine_model_v2` table which contains more detailed cell types.

# %%

# Get cell type table
ct = client.materialize.query_table("baylor_log_reg_cell_type_coarse_v1")
ct.head()

# %%
ct.cell_type.value_counts()

# %%
# !!! important
#     Not all neurons in the dataset have been proofread. In theory, you can check if a neuron has been proofread using the
#     corresponding annotation table:
#     ```python
#     table = client.materialize.query_table('proofreading_status_public_release')#
#     fully_proofread = table[
#           table.status_dendrite.isin(['extented', 'clean']) &
#           table.status_axon.isin(['extented', 'clean'])
#       ].pt_root_id.values
#     ```
#     However, it appears that the proofreading status table may be outdated at the moment.
#
# Let's fetch one of the excitatory neurons:

n = mi.fetch_neurons(
    ct[ct.cell_type == "excitatory"].pt_root_id.values[0], with_synapses=False
)[0]
n

# %%
# !!! note "Neuron IDs"
#     The neuron IDs in MICrONS are called "root IDs" because they represent collections of supervoxels - or rather
#     hierarchical layers of chunks of which the lowest layer are supervoxel IDs.

# %%
# MICrONS neurons can be fairly large, i.e. have lots of faces. You can try using using a higher `lod` ("level of detail", higher = coarser)
# but not all datastacks actually support multi-resolution meshes. If they don't (like this one) the `lod` parameter is silently ignored.
#
# For visualization in this documentation we will simplify the neuron a little. For this, you need either
# `open3d` (`pip3 install open3d`), `pymeshlab` (`pip3 install pymeshlab`) or Blender 3D on your computer.

# Reduce face counts to 1/3 of the original
n_ds = navis.simplify_mesh(n, F=1 / 3)

# Inspect (note the lower face/vertex counts)
n_ds

# %%
# Plot the downsample neuron (again: the downsampling is mostly for the sake of this documentation)

navis.plot3d(
    n_ds,
    radius=False,
    color="r",
    legend=False,  # hide the legend (more space for the plot)
)

# %%
# Nice! Now let's run a bit of analysis.
#
# ## Sholl Analysis
#
# Sholl analysis is a simple way to quantify the complexity of a neuron's arbor. It counts the number of intersections
# a neuron's arbor makes with concentric spheres around a center (typically the soma). The number of intersections is
# then plotted against the radius of the spheres.

import numpy as np

# The neuron mesh will automatically be skeletonized for this analysis
# Note: were defining radii from 0 to 160 microns in 5 micron steps
sha = navis.sholl_analysis(n, center="soma", radii=np.arange(0, 160_000, 5_000))

# %%
# Plot the results

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))

sha.intersections.plot(c="r")

ax.set_xlabel("radius [nm]")
ax.set_ylabel("# of intersections")
ax.patch.set_color((0, 0, 0, 0))  # Make background transparent
fig.patch.set_color((0, 0, 0, 0))

plt.tight_layout()

# %%
# See [`navis.sholl_analysis`][] for ways to fine tune the analysis. Last but not least a quick visualization with the neuron:

# %%
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Plot one of the excitatory neurons
fig, ax = navis.plot2d(n, view=("x", "y"), figsize=(10, 10), c="k", method="2d")

cmap = plt.get_cmap("viridis")

# Plot Sholl circles and color by number of intersections
center = n.soma_pos
# Drop the outer Sholl circles where there are no intersections
norm = Normalize(vmin=0, vmax=(sha.intersections.max() + 1))
for r in sha.index.values:
    ints = sha.loc[r, "intersections"]
    ints_norm = norm(ints)
    color = cmap(ints_norm)

    c = plt.Circle(center[:2], r, ec=color, fc="none")
    ax.add_patch(c)

# Add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
_ = plt.colorbar(
    ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="# of intersections"
)

# %%
#
# ## Render Videos
#
# Beautiful data like the MICrONS datasets lend themselves to visualizations. For making high quality videos (and renderings)
# I recommend you check out the tutorial on navis' [Blender interface](../../../gallery/3_interfaces/tutorial_interfaces_02_blender).
# Here's a little taster:
#
#  <iframe width="560" height="315" src="https://www.youtube.com/embed/wl3sFG7WQJc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
