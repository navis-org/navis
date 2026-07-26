"""
Insect Brain DB
===============

Fetch neurons and brain meshes from the Insect Brain Database.

The insect brain database (<https://insectbraindb.org>) is an online repository for neuron morphologies, brain regions and experimental
data across various insect species. At the time of writing Insect Brain DB features close to 400 neuronal cell types from well over 30
insect species. Check out [Heinze et al. (2021)](https://elifesciences.org/articles/65376) for details!

While the website features a comprehensive search and some nifty analyses, it can be useful to download these data to run your own
analyses or compare to other data sets. For that purpose, {{ navis }} provides an interface to Insect Brain DB that wraps parts of their API:

!!! note "Network access"
    This tutorial downloads data from the Insect Brain Database over the network, so it needs an internet
    connection. No account or token is required.
"""

# mkdocs_gallery_thumbnail_path = '_static/insect_brain_db_thumbnail.png'

# %%
# Import navis
import navis

# Import the actual Insect Brain DB interface
import navis.interfaces.insectbrain_db as ibdb

# %%
# This tutorial walks through three kinds of data you can pull from the Insect Brain Database:
#
# <div class="grid cards" markdown>
#
# -   :material-information-outline:{ .lg .middle } __Species metadata__
#
#     ---
#
#     List the available species and pull metadata for one of them.
#
#     [:octicons-arrow-right-24: Fetching meta data](#fetching-meta-data)
#
# -   :material-cube-outline:{ .lg .middle } __Neuropil meshes__
#
#     ---
#
#     Download brain region meshes for a species.
#
#     [:octicons-arrow-right-24: Fetching meshes](#fetching-meshes)
#
# -   :material-graph-outline:{ .lg .middle } __Neuron skeletons__
#
#     ---
#
#     Search for cell types and fetch their reconstructions.
#
#     [:octicons-arrow-right-24: Fetch neurons](#fetch-neurons)
#
# </div>

# %%
# ## Fetching meta data
#
# First, fetch a list of available species:

species = ibdb.get_available_species()
species.head()

# %%
# Fetch info for a given species (you can use the scientific or common name, or an ID):

spec_info = ibdb.get_species_info('Schistocerca gregaria')
spec_info

# %%
# ## Fetching meshes
#
# Fetch neuropil meshes for the Locust brain:

# `combine=True` would produce a single combined mesh but here we want a list of individual neuropils
locust_brain = ibdb.get_brain_meshes('Desert Locust', combine=False)
locust_brain[:2]

# %%
# Plot neuropils
navis.plot3d(locust_brain, volume_legend=True)

# %%

# This is for the tutorial thumbnail:
import matplotlib.pyplot as plt
fig, ax = navis.plot2d(locust_brain, method='2d')
ax.set_axis_off()
ax.grid(False)
plt.tight_layout()

# %%
# ## Fetch neurons
#
# First we need to know what neurons are available. Just like on the website you can set all kinds of different
# search parameters. Here we will stick with our Locust:

locust_neurons = ibdb.search_neurons(species='Desert Locust')
locust_neurons.head()

# %%
# Now fetch skeletons ("reconstructions") for some of the above neurons.
#
# !!! note "Not all neurons have skeletons"
#     Only some entries come with a reconstruction - check the `reconstruction_creator` column to see which ones do.

# You can use IDs or names, or a combination thereof to fetch skeletons
sk = ibdb.get_skeletons('CL1a-R2')
sk

# %%
# Plot the neuron - note that most neurons appear to have radii information
navis.plot3d(sk, radius=True)

# %%
# Check out the [API reference](../../../api.md#insectbrain-db-api) for further details.


