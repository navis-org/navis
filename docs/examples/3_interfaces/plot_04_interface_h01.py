"""
H01 Dataset
============

In this notebook, you can learn how to work with the [H01 dataset](https://www.science.org/doi/10.1126/science.adk4858) using Navis. The H01 dataset contains 57,000 cells and 150 million synapses from a cubic millimeter of the human temporal cortex, which is [proofreading](https://h01-release.storage.googleapis.com/proofreading.html) in the CAVE. With this interface, you can access both a snapshot of the proofread dataset and the latest dataset using CAVEclient.
"""


# %%
## For Dev, clear cache and reload module
# import navis
# import importlib
# import navis.interfaces.h01 as h01
# importlib.reload(h01)
import navis
import navis.interfaces.h01 as h01

# %%
## Examples in using Navis interface for H01 dataset


# %%
auth = h01.Authentication()

# %%
## Scenario 1: Completely new user of H01 dataset

# %%
auth.setup_token(make_new=True)

# %%
auth.save_token(token=, overwrite=True)

# %%
# %%
client = h01.get_cave_client()

# %%
## Scenario 2: Existing user, new computer

# %%
auth.setup_token(make_new=False)

# %%
auth.save_token(token=, overwrite=True)

# %%
client = h01.get_cave_client()

# %%
## Get CAVE client after setting up your token in your computer

# %%
client = h01.get_cave_client()

# %%
## Query Tables

# %%
client.materialize.get_versions()
print("Mat version: ", client.materialize.version)
client.materialize.get_tables()

# %%
### Query Materialized Synapse Table


# %%
client.materialize.synapse_query(limit=10)

# %%
syn = client.materialize.synapse_query(
    post_ids=[864691131861340864],
    # pre_ids=[ADD YOUR ROOT ID],
)
syn.head()
print(len(syn))

# %%
### Live Synapse Queries


# %%
import datetime
# check if root ID is the most recent root ID
root_id = 864691131861340864
now = datetime.datetime.now(datetime.timezone.utc)
is_latest = client.chunkedgraph.is_latest_roots([root_id], timestamp=now)
latest_id = client.chunkedgraph.get_latest_roots(root_id, timestamp=now)
print(is_latest) 
print(latest_id)

# %%
synapse_table = client.info.get_datastack_info()['synapse_table']
df=client.materialize.query_table(synapse_table,
                                  timestamp = datetime.datetime.now(datetime.timezone.utc),
                                  filter_equal_dict = {'post_pt_root_id': latest_id[0]})
print(len(df))
df.head()

# %%
### Query Cells Table

# %%
ct = client.materialize.query_table(table="cells")
ct.head()

# %%
ct.cell_type.unique()

# %%
### Filter by cell type

# %%
interneuron_ids = ct[ct.cell_type == "INTERNEURON"].pt_root_id.values[:50]

# %%
interneuron_ids = interneuron_ids[interneuron_ids != 0 ]
interneuron_ids

# %%
len(interneuron_ids)

# %%
interneurons = h01.fetch_neurons(interneuron_ids, lod=2, with_synapses=False)

# %%
interneurons_ds = navis.simplify_mesh(interneurons, F=1/3)

# %%
interneuron_ds

# %%
import seaborn as sns
colors = {n.id: sns.color_palette('Reds', 7)[i] for i, n in enumerate(interneurons_ds)}
fig = navis.plot3d([interneurons_ds], color=colors)

# %%
## Get Skeletons

# %%
interneurons_sk = navis.skeletonize(interneurons, parallel=True)

# %%
interneurons_sk

# %%
fig = navis.plot3d([interneurons_sk[0], interneurons[0]], color=[(1, 0, 0), (1, 1, 1, .5)])
