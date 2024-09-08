"""
Dotprops
========

This tutorial will show you have to work with Dotprops.

[`navis.Dotprops`][] are point clouds with associated principal vectors which are mostly used for
NBLASTing. They are typically derivatives of skeletons or meshes but you can load them straight from
confocal data using [`navis.read_nrrd`][]:
"""

# %%
import navis

# %%
# ## From image data
#
# For this example I downloaded one of Janelia's Fly Light confocal stacks ([link](https://splitgal4.janelia.org/))
# and converted it to NRRD format using [ImageJ](https://imagej.net/ij/).
#
# Load NRRD file into Dotprops instead of VoxelNeuron:
# ```python
# dp = navis.read_nrrd(
#    "~/Downloads/JRC_SS86025_JRC_SS86025-20211112_49_B6.nrrd",
#    output="dotprops",
#    threshold=3000,
# )
# ```

# %%
# !!! note
#     Note the threshold parameter? It determines which voxels (by brightness) are used and which are ignored!
#
# ## From other neurons
#
# Let's say you have a bunch of skeletons and you need to convert them to dotprops for NBLAST. For that you
# [`navis.make_dotprops`][]:

sk = navis.example_neurons(3, kind="skeleton")
dp = navis.make_dotprops(sk, k=5)

# Plot one of the dotprops
fig, ax = navis.plot2d(dp[0], view=("x", "-z"), method="2d", color="red")

# Add a zoom-in
axins = ax.inset_axes([0.03, 0.03, 0.47, 0.47], xticklabels=[], yticklabels=[])
_ = navis.plot2d(dp[0], view=("x", "-z"), method="2d", color="red", ax=axins)
axins.set_xlim(17e3, 19e3)
axins.set_ylim(15e3, 13e3)
ax.indicate_inset_zoom(axins, edgecolor="black")

# %%
# !!! note
#     The `k` parameter in [`make_dotprops`][navis.make_dotprops] determines how many neighbours are considered to
#     generated the tangent vector for a given point.
#     Higher `k` = smoother. Lower `k` = more detailed but also more noisy. If you have clean data such as these
#     connectome-derived skeletons, you can go with a low `k`. For confocal data, you might want to go with a higher `k`
#     (e.g. 20) to smooth out the noise. You can pass `k` to [`navis.read_nrrd`][] as well.
#
# ## Manual construction
#
# If not loaded from file, you would typically create [`Dotprops`][navis.Dotprops] via [`navis.make_dotprops`][] but just
# like all other neuron types, [`Dotprops`][navis.Dotprops] can be constructed manually:

# %%
import numpy as np

# Create some x/y/z coordinates
points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

# Create vectors for each point
# You can skip this point and just provide the `k` parameter
vect = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])

dp = navis.Dotprops(points, k=None, vect=vect)
dp

# %%
# There is no established format to store dotprops. But like all other neuron types in navis, you can pickle data for later (re)use
# - see the [pickling tutorial](../plot_04_io_pickle). See also the [I/O API reference](../../../api.md#importexport).
