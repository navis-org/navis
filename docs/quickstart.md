---
icon: material/rocket-launch
hide:
  - navigation
---

# Quickstart

This short introduction will show you the basics of how to use {{ navis }}. This is not
supposed to be comprehensive but rather to give you an idea of the basic concepts.
For inspiration, explore the [example gallery](../generated/gallery/) and for detailed
explanations have a look at the [API documentation](api.md).

## Single Neurons

For demonstration purposes {{ navis }} comes with a bunch of fruit fly neurons from
the [Janelia hemibrain](https://neuprint.janelia.org) project:

```python exec="on" source="above" result="py" session="quickstart"
import navis

# Load a single example neuron
n = navis.example_neurons(n=1, kind='skeleton')
n
print(n)  # markdown-exec: hide
```

!!! example "Loading your own neurons"
    Almost all tutorials will use the example neurons shipped with {{ navis }} (see
    [`navis.example_neurons`][] for details).

    You will most likely want to load your own neuron data. {{ navis }} has dedicated
    functions such as [`navis.read_swc`][] for that . Check out the
    [I/O Tutorials](../generated/gallery#import-export) to learn more!

{{ navis }} represents neurons as [`navis.TreeNeuron`][], [`navis.MeshNeuron`][], [`navis.VoxelNeuron`][] or
[`navis.Dotprops`][] - see the tutorial on [Neuron Types](../generated/gallery/tutorial_basic_01_neurons/)
for details.

In above code we asked for a skeleton, so the neuron returned is a [`TreeNeuron`][navis.TreeNeuron].
Like all neuron types, this class is essentially a wrapper around the actual
neuron data (the node table in case of skeletons) and has some convenient features.

The skeleton's node data is stored as `pandas.DataFrame`:

```python exec="on" source="above" result="py" session="quickstart"
n.nodes.head()
print(n.nodes.head()) # markdown-exec: hide
```

!!! note "Pandas"
    [pandas](https://pandas.pydata.org) is *the* data science library for Python and will help you
    analyze and visualize your data. We highly recommend familiarizing yourself with pandas!
    There are plenty of good tutorials out there but pandas' own
    [10 Minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/10min.html) is a good
    place to start.

Once you have your neuron loaded in {{ navis }} things are as simple as passing it to
the function that does what you need:

```python exec="on" source="above" result="py" session="quickstart" html="1"
fig, ax = navis.plot2d(n, view=('x', '-z'), color="coral", method='2d')
from io import StringIO # markdown-exec: hide
import matplotlib.pyplot as plt # markdown-exec: hide
plt.tight_layout() # markdown-exec: hide
buffer = StringIO() # markdown-exec: hide
plt.savefig(buffer, format="svg") # markdown-exec: hide
print(buffer.getvalue()) # markdown-exec: hide
plt.close() # markdown-exec: hide
```

## Lists of Neurons

When working with multiple neurons, {{ navis }} uses [`navis.NeuronLists`][navis.NeuronList]:

```python exec="on" source="above" result="py" session="quickstart"
# Ask for 3 example neurons
nl = navis.example_neurons(n=3, kind='skeleton')
nl
print(nl) # markdown-exec: hide
```

[`navis.NeuronLists`][navis.NeuronList] are containers and behave like a `list`
(with some extras, of course):

```python exec="on" source="above" result="py" session="quickstart"
# Access the first neuron in the list
nl[0]
print(nl[0]) # markdown-exec: hide
```

[`navis.NeuronLists`][navis.NeuronList] provide quick and convenient access
to all functions (methods) and properties of the neurons it contains:

```python exec="on" source="above" result="py" session="quickstart"
# Get the cable length for all neurons in the list
nl.cable_length
print(nl.cable_length) # markdown-exec: hide
```

Most functions that accept single neurons, also happily work with
[`NeuronLists`][navis.NeuronList]:

```python exec="on" source="above" result="py" session="quickstart" html="1"
# Generate a plot of our neurons
fig, ax = navis.plot2d(nl, view=('x', '-z'), method='2d')
from io import StringIO # markdown-exec: hide
import matplotlib.pyplot as plt # markdown-exec: hide
plt.tight_layout() # markdown-exec: hide
buffer = StringIO() # markdown-exec: hide
plt.savefig(buffer, format="svg") # markdown-exec: hide
print(buffer.getvalue()) # markdown-exec: hide
plt.close() # markdown-exec: hide
```

See the [Lists of Neurons](../generated/gallery/tutorial_basic_02_neuronlists/)
tutorial for more information.

## Methods vs Functions

{{ navis }} neurons and neuron lists have methods that serve as shortcuts to main functions.

These code snippets are equivalent:

=== "Full function"
    ```python
    import navis

    s = navis.example_neuron(n=1, type='skeleton')
    ds = navis.downsample_neuron(s, 5)
    ```

=== "Shortcut"
    ```python
    import navis

    s = navis.example_neuron(n=1, type='skeleton')
    ds = s.downsample(5)
    # Under the hood, `s.downsample()` calls `navis.downsample_neuron(s)`
    ```

## The `inplace` Parameter

You may notice that many {{ navis }} functions that modify neurons (resampling, pruning, etc.) have an
`inplace` parameter. This is analogous to `pandas` where `inplace` defines whether we
modify the original (`inplace=True`) or operate on a copy (`inplace=False`, default).

Downsample a copy of our skeleton and leaving the original unchanged
(this is the default for almost all functions):

```python
n_ds = navis.downsample_neuron(neuron, 10, inplace=False)
```

Downsample the original neuron:

```python
navis.downsample_neuron(neuron, 10, inplace=True)
```

Using `inplace=True` can be useful if you work with lots of neurons and want avoid making
unnecessary copies to keep the memory footprint low!


## Getting Help

Feeling a bit lost? No worries! Check out the [Tutorials](../generated/gallery) or browse
the [API documentation](api.md).

{{ navis }} also supports autocompletion: typing `navis.` and pressing the TAB key should bring
up a list of available functions. This also works with neuron properties and methods!

If you don't know what a function does, try e.g. `help(navis.plot3d)` or find it in the
[API documentation](api.md) to get a nicely rendered docstring:

```python exec="on" source="above" result="py" session="quickstart"
help(navis.prune_twigs)
print(navis.prune_twigs.__doc__)  # markdown-exec: hide
```

Note that most functions have helpful `Examples`!

## What next?

<div class="grid cards" markdown>

-   :material-tree-outline:{ .lg .middle } __Neuron types__
    ---

    Find out more about the different neuron types in {{ navis }}.

    [:octicons-arrow-right-24: Neuron types tutorial](../generated/gallery/tutorial_basic_01_neurons)

-   :material-cube:{ .lg .middle } __Lists of Neurons__
    ---

    Check out the guide on lists of neurons.

    [:octicons-arrow-right-24: NeuronLists tutorial](../generated/gallery/tutorial_basic_02_neuronlists)

-   :octicons-file-directory-symlink-16:{ .lg .middle } __Neuron I/O__

    ---

    Learn about how to load your own neurons into {{ navis }}.

    [:octicons-arrow-right-24: I/O Tutorials](../generated/gallery#import-export)

</div>