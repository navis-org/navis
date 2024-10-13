"""
XKCD Style
==========

This example demonstrates how to plot neurons in xkcd style.

If you don't already know: `matplotlib` has a [xkcd mode](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xkcd.html)
that produces plots that look like they were drawn by hand. This can be a fun way to visualize neurons:
"""

# %%

import navis
import matplotlib.pyplot as plt

n = navis.example_neurons(1, kind="skeleton")

# Plot in xkcd style
with plt.xkcd(scale=5, randomness=10, length=200):
    fig, ax = navis.plot2d(
        n, method="2d", c="k", view=("x", "-z"), radius=False, lw=1.5
    )

plt.tight_layout()


# %%
# Get a few more example neurons and a volume
nl = navis.example_neurons()
neuropil = navis.example_volume("neuropil")

# Make the neuropil mostly transparent
neuropil.color = (0, 0, 0, 0.02)

# Plot in xkcd style
with plt.xkcd(scale=5, randomness=10, length=200):
    fig, ax = navis.plot2d(
        [nl, neuropil],
        method="2d",
        c="k",
        view=("x", "-z"),
        lw=1.2,
        volume_outlines="both",
        radius=False,
    )

    ax.grid(False)

plt.tight_layout()
