"""
NBLAST against FlyCircuit
=========================

This example demonstrates how to run an NBLAST against the entire FlyCircuit dataset.

!!! important "This example is not executed"
    In contrast to almost all other tutorials, this one is not executed when the documentation is built.
    Consequently, it also does not show any code output or figures. That's because this example requires
    downloading a large dataset (~850Mb) and running an NBLAST against it, it is simply not feasible to
    run this as part of the documentation build process.

If you work with _Drosophila_, chances are you have heard of [FlyCircuit](http://www.flycircuit.tw). It's a collection of ~16k
single neuron clones published by [Chiang et al. (2010)](https://www.cell.com/current-biology/fulltext/S0960-9822(10)01522-8).

For R, there is a package containing dotprops and meta data for these neurons: [`nat.flycircuit`](https://github.com/natverse/flycircuit).
This does not yet exist for Python. However, it's still pretty straight forward to run an NBLAST against the entire flycircuit dataset in Python!

First you need to download flycircuit dotprops from [Zenodo](https://zenodo.org/record/5205616) (~850Mb). Next, unzip the archive containing the
dotprops as CSV files. Now we we need to load them into {{ navis }} (adjust filepaths as required):
"""

# %%
import navis
import pandas as pd

from pathlib import Path
from tqdm import tqdm

def load_dotprops_csv(fp):
    """Load dotprops from CSV files in filepath `fp`."""
    # Turn into a path object
    fp = Path(fp).expanduser()

    # Go over each CSV file
    files = list(fp.glob('*.csv'))
    dotprops = []
    for f in tqdm(files, desc='Loading dotprops', leave=False):
        # Read file
        csv = pd.read_csv(f)

        # Each row is a point with associated vector
        pts = csv[['pt_x', 'pt_y', 'pt_z']].values
        vect = csv[['vec_x', 'vec_y', 'vec_z']].values

        # Turn into a Dotprops
        dp = navis.Dotprops(points=pts, k=20, vect=vect, units='1 micron')

        # Use filename as ID/name
        dp.name = dp.id = f.name[:-4]

        # Add this dotprop to the list before moving on to the next
        dotprops.append(dp)

    return navis.NeuronList(dotprops)

# %%
# Load dotprops from the filepath
fc_dps = load_dotprops_csv('flycircuit_dotprops')
fc_dps

# %%
# !!! note
#     To avoid having to re-generate these dotprops, you could consider pickling them:
#     ```python
#     import pickle
#     with open('flycircuit_dps.pkl', 'wb') as f:
#         pickle.dump(fc_dps, f)
#     ```
#
#     In the future you can then reload the dotprops like so (much faster than loading from CSV):
#     ```python
#     with open('flycircuit_dps.pkl', 'rb') as f:
#         fc_dps = pickle.load(f)
#     ```
#
# Note: The names/ids correspond to the flycircuit gene + clone names:

fc_dps[0]

# %%
# In case your query neurons are in a different brain space, you can use [flybrains](https://github.com/navis-org/navis-flybrains) to
# convert them to flycircuit's `FCWB` space.
#
# For demonstration purposes we will use the example neurons - olfactory DA1 projection neurons from the hemibrain connectome - that ship
# with {{ navis }} and try to find their closest match in the FlyCircuit dataset.

# Load some of the example neurons
n = navis.example_neurons(3)

# Convert from hemibrain (JRCFIB2018Fraw) to FCWB space
import flybrains

n_fcwb = navis.xform_brain(n, source='JRCFIB2018Fraw', target='FCWB')

# %%
# A sanity check to make sure the transform worked
fig, ax = navis.plot2d([n_fcwb, flybrains.FCWB])

# %%
# FlyCircuit neurons are all on the left side of the brain. We need
# mirror our neurons from the right to the left to match that.
n_fcwb_mirr = navis.mirror_brain(n_fcwb, template='FCWB')

# %%
# Convert our neurons to dotprops
n_dps = navis.make_dotprops(n_fcwb_mirr, resample=1, k=None)

# %%
# Run a "smart" NBLAST to get the top hits
scores = navis.nblast_smart(query=n_dps, target=fc_dps, scores='mean', progress=False)
scores.head()

# %%
# !!! note
#     If you get a warning about data not being in microns: that's a rounding error from the transform and can be safely ignored.

# %%
# Find the top hits for each of our query neurons
import numpy as np

for dp in n_dps:
    hit = scores.columns[np.argmax(scores.loc[dp.id].values)]
    sc = scores.loc[dp.id].values.max()
    print(f'Top hit for {dp.id}: {hit} ({sc:.3f})')

# %%
# All of our query neurons should have the same top match (they are all from the same cell type after all): `FruMARCM-F001496_seg001`

# %%
# Let's co-visualize:
# Queries in red, hit in black
fig, ax = navis.plot2d([n_fcwb_mirr, fc_dps.idx['FruMARCM-F001496_seg001']],
                       color=['r'] * len(n_fcwb_mirr) + ['k'])

# %%
# Looking good!