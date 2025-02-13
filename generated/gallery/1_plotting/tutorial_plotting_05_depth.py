"""
Depth-coloring
==============

This example shows how to color neurons by depth.

The obvious issue with 2d plots is that they are... well, 2d. This means that you can't easily convey depth information.
What we can do, however, is color the neuron by depth - that is by the distance to the camera.
This is a simple way to give a sense of the neuron's 3d structure.

Note that this currently works only for [`navis.plot2d`][], i.e. `matplotlib`:

"""

import navis
import matplotlib.pyplot as plt

n = navis.example_neurons(1, kind="skeleton")

fig, ax = navis.plot2d(
    n,
    depth_coloring=True,
    method='2d',
    view=("x", "-z"),
)
plt.tight_layout()

# %%
# The `depth_coloring` parameter will color the neuron by distance from the camera. For this neuron, the ventral dendrites
# are closest to the camera whereas the dorsal axon is further away.
#
# By default, this will use the `jet` colormap. You can change this to any of
# [`matplotlib`'s colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
# using the `palette` parameter:

fig, ax = navis.plot2d(
    n,
    depth_coloring=True,
    palette="hsv",
    method='2d',
    view=("x", "-z"),
)

# %%
# This should work with both [`TreeNeurons`][navis.TreeNeuron] and [`MeshNeurons`][navis.MeshNeuron] and methods `2d` and `3d`.