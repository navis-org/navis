"""
Pickling
========
<!-- difficulty: beginner -->

Quickly cache and reload neurons with Python's pickle module.

All {{ navis }} neurons - including whole [`NeuronLists`][navis.NeuronList] - can be "pickled" :cucumber:.
Pickling serialises the live Python object to a byte stream: it's extremely fast and ideal for
short-term caching, with a couple of important caveats.
"""

# %%
# !!! warning "Caveats"
#     Pickling is fast but is *not* a durable archival format:
#
#     1. Pickle files can only be re-opened in Python.
#     2. The pickled object is tied to your current environment. If you update Python, {{ navis }}
#        or even just `numpy` or `pandas`, you may no longer be able to open an old file.
#
# !!! danger "Never unpickle untrusted files"
#     A pickle file can contain arbitrary Python code that runs the moment it is loaded. **Only ever
#     unpickle files you created yourself or that come from a source you fully trust.**

# %%
# With that out of the way, pickling is straightforward:
#
# === "Save"
#     ```python
#     import navis
#     import pickle
#
#     # Load some example neurons
#     nl = navis.example_neurons(3, kind='mesh')
#
#     # Pickle the NeuronList to a file
#     with open('neurons.pkl', 'wb') as f:
#         pickle.dump(nl, f)
#     ```
#
# === "Load"
#     ```python
#     import pickle
#
#     # Read the neurons back from the pickle file
#     with open('neurons.pkl', 'rb') as f:
#         nl = pickle.load(f)
#     ```
#
# See also the [I/O API reference](../../../api.md#importexport).

# %%

# mkdocs_gallery_thumbnail_path = '_static/pickle.png'
