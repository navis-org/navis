"""
H01 Dataset
===========

In this notebook, you can learn how to work with the H01 dataset using NAVis.

The [H01 dataset](https://www.science.org/doi/10.1126/science.adk4858) contains 57,000 cells and 150 million synapses
from a cubic millimeter of the human temporal cortex, which is [proofread](https://h01-release.storage.googleapis.com/proofreading.html)
using the CAVE ecoystem.

With this interface, you can access both a snapshot of the proofread dataset and the latest dataset using `caveclient`:

```shell
pip install caveclient -U
```

!!! note "Authentication"
    If this is your first time using `CAVEclient` to access the `H01` dataset, you might have to get and set your authentication token:

    1. Go to: https://https//global.brain-wire-test.org/auth/api/v1/create_token to create a new token.
    2. Log in with your Google credentials and copy the token shown afterward.
    3. Save it to your computer with:
       ```python
       from caveclient import CAVEclient
       client = CAVEclient(server_address="https://global.brain-wire-test.org", datastack_name='h01_c3_flat', auth_token="PASTE_YOUR_TOKEN_HERE")
       client.auth.save_token(token="PASTE_YOUR_TOKEN_HERE")
       ```

    Note that the H01 dataset uses a server address that is different from the default `CAVEClient` server.
    Also be aware that creating a new token by finishing step 2 will invalidate the previous token!

"""

# %%
import navis
from navis.interfaces import h01

# Initialize the client
client = h01.get_cave_client()

# %%
# ### Query Tables

# %%
client.materialize.get_versions()

# %%
client.materialize.get_tables()

# %%
# ### Query Materialized Synapse Table
# Query the first few rows in the table

client.materialize.synapse_query(limit=10)

# %%
# Query specific pre- and/or postsynaptic IDs

syn = client.materialize.synapse_query(
    post_ids=[864691131861340864],
    # pre_ids=[ADD YOUR ROOT ID],
)
syn.head()

# %%
print(len(syn))

# %%
# ### Live Synapse Queries

import datetime as dt

# Check if root ID is the most recent root ID
root_id = 864691131861340864
now = dt.datetime.now(dt.timezone.utc)
is_latest = client.chunkedgraph.is_latest_roots([root_id], timestamp=now)
latest_id = client.chunkedgraph.get_latest_roots(root_id, timestamp=now)
print(is_latest, latest_id)

# %%
synapse_table = client.info.get_datastack_info()["synapse_table"]
df = client.materialize.query_table(
    synapse_table,
    timestamp=dt.datetime.now(dt.timezone.utc),
    filter_equal_dict={"post_pt_root_id": latest_id[0]},
)
df.head()

# %%
# ### Query Cells Table

ct = client.materialize.query_table(table="cells")
ct.head()

# %%
ct.cell_type.unique()

# %%
# ### Filter by cell type

# Get the first 50 interneurons
interneuron_ids = ct[ct.cell_type == "INTERNEURON"].pt_root_id.values[:50]

# Remove 0 IDs
interneuron_ids = interneuron_ids[interneuron_ids != 0]

# What's left?
interneuron_ids

# %%
# ### Fetch Neuron Meshes
interneurons = h01.fetch_neurons(interneuron_ids, lod=2, with_synapses=False)
interneurons_ds = navis.simplify_mesh(interneurons, F=1 / 3)
interneurons_ds

# %%

# Plot
import seaborn as sns

colors = {n.id: sns.color_palette("Reds", 7)[i] for i, n in enumerate(interneurons_ds)}
navis.plot3d([interneurons_ds], color=colors)


# %%
# ### Fetch Skeletons

interneurons_sk = navis.skeletonize(interneurons, parallel=True)
interneurons_sk


# %%

# Plot
navis.plot3d([interneurons_sk[0], interneurons[0]], color=[(1, 0, 0), (1, 1, 1, 0.5)])
