
"""
Pickling
========

This tutorial shows how to use Python's `pickle` module to quickly store and load neurons.

All neuron types including whole [`NeuronLists`][navis.NeuronList] can be "pickled" :cucumber:.
If you don't know what that is: pickling is storing the actual Python object as a
bytes stream. This is incredibly fast and works very well for short-term storage
but has a few downsides:

1. Pickle files can only be re-opened in Python
2. The pickled object is (sort of) specific to your current Python enviroment. If
   you e.g. update Python, {{ navis }} or even just `numpy` or `pandas` you may not
   be able to open the file again.
3. Pickle files can contain arbitrary Python code. Never open a pickle file from
   an untrusted source!

With that out of the way, pickling is incredibly easy:

```python
import navis
import pickle

# Load some example neurons
nl = navis.example_neurons(3, kind='mesh')

# Pickle neurons to file
with open('/Users/philipps/Downloads//meshes.pkl', 'wb') as f:
    pickle.dump(nl, f)
```

To "unpickle", i.e. read the file back into Python:

```python
# Read neurons back from pickle file
with open('/Users/philipps/Downloads//meshes.pkl', 'rb') as f:
    nl = pickle.load(f)
```

This tutorial has hopefully given you some entry points on how to load your data.
See also the [I/O API reference](../../../api.md#importexport).

"""

# %%

# mkdocs_gallery_thumbnail_path = '_static/pickle.png'