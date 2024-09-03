"""
Insect Brain DB
===============

In this example we will show you how to fetch data from the Insect Brain DB.

The insect brain database (<https://insectbraindb.org>) is an online repository for neuron morphologies, brain regions and experimental
data across various insect species. At the time of writing Insect Brain DB features close to 400 neuronal cell types from well over 30
insect species. Check out [Heinze et al. (2021)](https://elifesciences.org/articles/65376) for details!

While the website features a comprehensive search and some nifty analyses, it can be useful to download these data to run your own
analyses or compare to other data sets. For that purpose, {{ navis }} provides an interface to Insect Brain DB that wraps parts of their API:
"""

# %%
# Import navis
import navis

# Import the actual Insect Brain DB interface
import navis.interfaces.insectbrain_db as ibdb

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

# This is for the tutorial thumbail:
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
# Let's fetch skeletons ("reconstructions") for some of the above neurons. Note that not all neurons have skeletons
# (see the "reconstruction_creator" column)!

# You can use IDs or names, or a combination thereof to fetch skeletons
sk = ibdb.get_skeletons('CL1a-R2')
sk

# %%
# Plot the neuron - note that most neurons appear to have radii information
navis.plot3d(sk, radius=True)

# %%
# Check out the [API reference](../../../api.md#insectbrain-db-api) for further details.


