"""
Dotprops
========

This tutorial will show you how to load/create Dotprops.

[`navis.Dotprops`][] are point clouds with associated principal vectors which are mostly used for
NBLASTing. They are typically derivatives of skeletons or meshes but you can load them straight from
e.g. confocal image stacks using [`navis.read_nrrd`][] or [`navis.read_tiff`][].

![dotprops](../../../../_static/dotprops.png)
"""

# %%
import navis
import matplotlib.pyplot as plt

# %%
# ## From image data
#
# For this example we will use a stack from [Janelia's split Gal4 collection](https://splitgal4.janelia.org/).
# This `LH2094` line is also available from [Virtual Fly Brain](https://v2.virtualflybrain.org/org.geppetto.frontend/geppetto?id=VFB_00102926&i=VFB_00101567,VFB_00102926)
# where, conveniently, they can be downloaded in NRRD format which we can directly read into {{ navis }}.
#
# Let's do this step-by-step first:

# Load raw NRRD image
im, header = navis.read_nrrd(
    "https://v2.virtualflybrain.org/data/VFB/i/0010/2926/VFB_00101567/volume.nrrd",
    output="raw"
)

# Plot a maximum projection
max_proj = im.max(axis=2)
plt.imshow(
    max_proj.T,
    extent=(0, int(0.5189 * 1210), (0.5189 * 566), 0),  # extent is calculated from the spacing (see `header`) times the no of x/y pixels
    cmap='Greys_r',
    vmax=10  # make it really bright so we can see neurons + outline of the brain
    )

# %%
# At this point we could threshold the image, extract above-threshold voxels and convert them to a Dotprops object.
# However, the easier option is to use [`navis.read_nrrd`][] with the `output="dotprops"` parameter:

dp = navis.read_nrrd(
    "https://v2.virtualflybrain.org/data/VFB/i/0010/2926/VFB_00101567/volume.nrrd",
    output="dotprops",
    threshold=5,  # threshold to determine which voxels are used for the dotprops
    thin=True,   # see note below on this parameter!
    k=10  # number of neighbours to consider when calculating the tangent vector
)

# %%
# !!! note "Thinning"
#     In the above [`read_nrrd`][navis.read_nrrd] call we used `thin=True`. This is a post-processing step that
#     thins the image to a single pixel width. This will produce "cleaner" dotprops but can also remove denser
#     neurites thus emphasizing the backbone of the neuron. This option requires the `scikit-image` package:
#
#     ```bash
#     pip install scikit-image
#     ```
#
#  Let's overlay the dotprops on the maximum projection:

fig, ax = plt.subplots()
ax.imshow(
    max_proj.T,
    extent=(0, int(0.5189 * 1210), (0.5189 * 566), 0),
    cmap='Greys_r',
    vmax=10
    )
navis.plot2d(dp, ax=ax, view=("x", "-y"), method="2d", color="r", linewidth=1.5)

# %%
# This looks pretty good but we have a bit of little fluff around the brain which we may want to get rid off:

# Drop everything but the two largest connected components
dp = navis.drop_fluff(dp, n_largest=2)

# Plot again
fig, ax = plt.subplots()
ax.imshow(
    max_proj.T,
    extent=(0, int(0.5189 * 1210), (0.5189 * 566), 0),
    cmap='Greys_r',
    vmax=10
    )
navis.plot2d(dp, ax=ax, view=("x", "-y"), method="2d", color="r", linewidth=1.5)

# %%
# !!! note
#     To extract the connected components, [`navis.drop_fluff`][] treats all pairs of points within a certain distance
#     as connected. The distance is determined by the `dp_dist` parameter which defaults to 5 x the average distance
#     between points. This is a good value ehre but you may need adjust it for your data.
#
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
# - see the [pickling tutorial](../tutorial_io_04_pickle). See also the [I/O API reference](../../../api.md#importexport).
