"""
XKCD Style
==========

Render neurons in the hand-drawn XKCD sketch style, just for fun.

If you don't already know: `matplotlib` has an [xkcd mode](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xkcd.html)
that makes plots look like they were drawn by hand - a fun way to visualize neurons.

!!! tip "Dial in the wobble"
    `plt.xkcd()` takes three knobs: `scale` (how big the wiggles are), `randomness` (how often the line
    wiggles) and `length` (the length of each wiggle). Tweak them to taste.
"""

# %%
# The recipe is the same whether you sketch a single neuron or a whole scene - just wrap your
# [`navis.plot2d`][] call in a `with plt.xkcd(...)` block:
#
# === "Single neuron"
#     ```python
#     with plt.xkcd(scale=5, randomness=10, length=200):
#         navis.plot2d(n, method="2d", c="k", view=("x", "-z"), radius=False, lw=1.5)
#     ```
#
# === "Neurons + neuropil"
#     ```python
#     with plt.xkcd(scale=5, randomness=10, length=200):
#         navis.plot2d(
#             [nl, neuropil], method="2d", c="k", view=("x", "-z"),
#             lw=1.2, volume_outlines="both", radius=False,
#         )
#     ```

import navis
import matplotlib.pyplot as plt

# Get a few example neurons and a volume
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
