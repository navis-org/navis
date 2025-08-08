"""
Cortical Neurons
================

This tutorial demonstrates how to plot cortical neurons.


In this exercise we will visualize morphological data from ["Integrated Morphoelectric and Transcriptomic Classification of Cortical GABAergic Cells"](https://www.cell.com/cell/pdf/S0092-8674(20)31254-X.pdf)
by Gouwens, Sorensen _et al._, Cell (2020). Specifically, we will re-create a plot similar to their
[Figure 4A](https://www.cell.com/cms/10.1016/j.cell.2020.09.057/asset/c684cc3f-ee17-4a36-98c9-e464a7ce8063/main.assets/gr4_lrg.jpg).

For brevity, we will use some fixed cell IDs and properties from the dataset. These were taken from the
[`20200711_patchseq_metadata_mouse.csv`](https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/5e/2a/5e2a5936-61da-4e09-b6da-74ab97ce1b02/20200711_patchseq_metadata_mouse.csv)
file provided alongside the supplementary material of the paper:

"""

# %%

# The cell IDs we will use (it's the first 5 in the meta data file)
ids = [601506507, 601790961, 601803754, 601808698, 601810307]

# The normalized soma depths for these cells (also from the meta data file)
soma_depths = [0.36101451, 0.62182935, 0.16423996, 0.48303029, 0.2956563]

# %%
# ## Part I: Loading and Aligning Neurons
#
# First we need to load the neurons. Here, we will take them straight from their FTP server
# but you can of course download them first and then load from disk!

import navis

nl = navis.read_swc(
    "ftp://download.brainimagelibrary.org/biccn/zeng/pseq/morph/200526/",
    limit=[f"{i}_transformed.swc" for i in ids],  #  Load only the files we need
    fmt="{name,id:int}_transformed.swc",  # Parse the name and id from the file name
)

# %%
# To make our lives a bit easier, we will attach the soma depth to the neurons as metadata:

nl.set_neuron_attributes(
    soma_depths,
    name="cell_soma_normalized_depth",
    register=True
    )

nl

# %%
# Next, we need to align the neurons according to their soma depth! The normalized `cell_soma_normalized_depth` should
# map to a physical range of `0` to `922.5861720311` microns.
#
# Let's demo with one neuron before we run this for all neurons:

# Grab one of the neurons
n = nl[0]

# This is the normalized soma depth:
print(f"Normalized soma depth: {n.cell_soma_normalized_depth}")

# %%
# The physical soma depth is simply the normalized depth multiplied by the total depth of the cortex.
# Note that we're positioning from the bottom - i.e. 922.586 will be at the surface and 0 at the bottom!
# This is to make our lifes easier when it comes to plotting since the origin in `matplotlib`
# figures is in the bottom left corner.

phys_y = (1 - n.cell_soma_normalized_depth) * 922.5861720311
print(f"Physical soma depth: {phys_y}")

# Current soma
print(f"Current soma coordinates: {n.soma_pos[0]}")

# %%
# We will now offset the neuron such that the soma is at `(0, 589.519, 0)`:

# %%
offset = [0, phys_y, 0] - n.soma_pos[0]
offset

# %%
# Moving or scaling neurons in {{ navis }} is super straight forward: adding, subtracting, dividing or multiplying neurons by a number or an
# `[x, y, z]` vector will change their coordinates:

# Move the neuron to the new centered position
n += offset

# Check the that the soma is now in the correct position
n.soma_pos[0]

# %%
# That looks good! Let's do it for all neurons:

# %%
for n in nl:
    phys_y = (1 - n.cell_soma_normalized_depth) * 922.5861720311
    offset = [0, phys_y, 0] - n.soma_pos[0]
    n += offset

# %%
# Check that all soma positions are correct:
nl.soma_pos.reshape(-1, 3)

# %% [markdown]
# ## Part II: Plotting
#
# Now that we have loaded and aligned the neurons, let's recreate a plot similar to those in Figure 4A:

def plot_neurons(to_plot, color="purple", axon_color="magenta", offset=500):
    """Plot all neurons of a given transcriptomic type.

    Parameters
    ----------
    neurons : NeuronList
        The aligned neurons to plot.
    color : str
        The color of the dendrites.
    axon_color : str
        The color of the axon.
    offset : int
        The offset between neurons along the x-axis.

    Returns
    -------
    fig, ax
        The matplotlib figure and axis.

    """
    # Offset the neurons along the x-axis so that they don't overlap
    to_plot = [n + [offset * i, 0, 0] for i, n in enumerate(to_plot)]

    # The SWC files for this dataset include a `label` column which
    # indicates the compartment type:
    # 1 = soma
    # 2 = axon
    # 3 = dendrites
    # We will use this `label` to color the neurons' compartments.

    # Here we define a color palette for the compartments:
    compartment_palette = {1: color, 2: axon_color, 3: color}

    # Plot the neuron
    fig, ax = navis.plot2d(
        to_plot,
        radius=False,
        lw=1.5,
        soma=dict(
            fc="black",  # soma fill color
            ec="white",  # highlight the soma with a white outline
            radius=10,   # override the default soma radius
        ),
        color_by="label",  # color by `label` column in node table
        palette=compartment_palette,
        figsize=(
            len(to_plot) * 2,
            10,
        ),  # scale the figure size with the number of neurons
        method="2d",
    )

    # Add the layer boundaries (top bound for each layer in microns)
    layer_bounds = {
        "L1": 0,
        "L2/3": 115.1112491335,
        "L4": 333.4658190171,
        "L5": 453.6227158132,
        "L6": 687.6482650269,
        "L6b": 883.1308910545,
    }

    for layer, y in layer_bounds.items():
        y = 922.5861720311 - y  # flip the y-axis
        # Add a dashed line
        ax.axhline(y, color="gray", ls="--", lw=1)
        # Add the layer name
        ax.text(-300, y - 25, layer, color="gray", va="center", size=10)
    # Add the bottom bound
    ax.axhline(0, color="gray", ls="--", lw=1)

    # Set the axis y limits according to the layers
    ax.set_ylim(-10, 930)

    # Hide axes
    ax.axis("off")

    return fig, ax


fig, ax = plot_neurons(nl)

# %% [markdown]
# That looks close enough. The last bit is to add the little KDE plots for the depth-distribution of
# cable length!
#
# We're going to be cheap here and simply generate a histogram over the node positions.
# To make this representative, we should make sure that the number of nodes per unit of cable
# is homogeneous across neurons. For that we will resample the neurons:
print(
    f"Sampling rate (nodes per micron of cable) before resampling: {nl.sampling_resolution.mean():.2f}"
)

# Resample to 2 nodes per micron
resampled = navis.resample_skeleton(
    nl,
    resample_to=0.5,
    map_columns="label",  # make sure label column is carried over
)

print(
    f"Sampling rate (nodes per micron of cable) after resampling: {resampled.sampling_resolution.mean():.2f}"
)

# %%
# Get the combined nodes table:
nodes = resampled.nodes
nodes.head()

# %%
# Now we can plot the distribution of cable lengths for our neurons:

import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Plot the neurons again, re-using the function we defined above
fig, ax = plot_neurons(nl)

# Add a new axis to the right of the main plot
divider = make_axes_locatable(ax)
ax_hist = divider.append_axes("right", size=0.75, pad=0.05)

# Add histograms
# For axon:
sns.kdeplot(
    data=nodes[nodes.label == 2], y="y", ax=ax_hist, color="magenta", linewidth=1.5
)
# For the rest:
sns.kdeplot(
    data=nodes[nodes.label != 2], y="y", ax=ax_hist, color="purple", linewidth=1.5
)

# Add soma positions
soma_pos = nl.soma_pos.reshape(-1, 3)
ax_hist.scatter([0] * len(soma_pos), soma_pos[:, 1], color="black", s=10, clip_on=False)

# Set same axis limits as the main plot
ax_hist.set_ylim(-10, 930)

# Hide axes
ax_hist.set_axis_off()

# %%
# ## Acknowledgements
#
# We thank Staci Sorensen and Casey Schneider-Mizell from the Allen Institute for Brain Science
# for helping with extra information and data for this tutorial!