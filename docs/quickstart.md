---
icon: material/rocket-launch
---

# Quickstart

This short introduction will show you the basics of how to use {{ navis }}. This is not
supposed to be comprehensive but rather to give you a flavor of the basic concepts.
For inspiration, explore the [example gallery](../generated/gallery/) and for detailed
explanations have a look at the [API documentation](api.md).

## Single Neurons

{{ navis }} lets you import neurons from a variety of local and remote sources.
For demonstration purposes {{ navis }} comes with a bunch of fruit fly neurons from
the [Janelia hemibrain](https://neuprint.janelia.org) project:

```python exec="on" source="above" result="py" session="quickstart"
import navis

n = navis.example_neurons(n=1, kind='skeleton')
n
print(n)  # markdown-exec: hide
```

??? example Loading your own neurons
    Almost all tutorials we will use the example neurons {{ navis }} with (see
    [`navis.example_neurons`][]). You will obviously want to load your own neuron
    data. Check out the [I/O Tutorials](../generated/gallery#import-export)!


In above code we loaded one of the example neurons. {{ navis }} represents neurons as
[`navis.TreeNeuron`][], [`navis.MeshNeuron`][], [`navis.VoxelNeuron`][] or
[`navis.Dotprops`][] - see the [Neuron Types](../generated/gallery/plot_01_neurons_intro/)
tutorial for details.

In this example we asked for a skeleton, so the neuron returned is a
[`TreeNeuron`][navis.TreeNeuron]. This class is essentially a wrapper around the actual
neuron data (the node table in case of skeletons) and has some convenient features.

The skeleton's node data is stored as `pandas.DataFrame`:

```python exec="on" source="above" result="py" session="quickstart"
n.nodes.head()
print(n.nodes.head()) # markdown-exec: hide
```

!!! note "Pandas"
    [pandas](https://pandas.pydata.org) is *the* data science library for Python and will help you
    analyze and visualize your data. *We highly recommend familiarizing yourself with pandas!*
    There are plenty of good tutorials out there but pandas' own
    [10 Minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/10min.html) is a good
    place to start.

## Getting Help

Lost? Try typing in "`n.`" and hitting tab: most neuron attributes and {{ navis }} functions are accessible
via autocompletion. If you don't know what a function does, check out the documentation using
`help()` or via the [API documentation](api.md):

```python exec="on" source="above" result="py" session="quickstart"
help(navis.TreeNeuron.reroot)
print(navis.TreeNeuron.reroot.__doc__)  # markdown-exec: hide
```

The same obviously works with any `navis.{function}` function such as [`help(navis.plot2d)`][navis.plot2d]!

## Methods vs Functions

{{ navis }} neurons have class methods that serve as shortcuts to main functions.

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
    ```

## Lists of Neurons

To work with multiple neurons, {{ navis }} uses [`navis.NeuronLists`][navis.NeuronList]:

```python exec="on" source="above" result="py" session="quickstart"
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

See the [Lists of Neurons](../generated/gallery/plot_02_neuronlists_intro/)
tutorial for details.

## The `inplace` Parameter

You may notice that many {{ navis }} functions that run some operation on neurons have an
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

## What next?

<div class="grid cards" markdown>

-   :material-tree-outline:{ .lg .middle } __Neuron types__
    ---

    Find out more about the different neuron types in {{ navis }}.

    [:octicons-arrow-right-24: Neuron types tutorial](../generated/gallery/plot_01_neurons_intro)

-   :material-cube:{ .lg .middle } __Lists of Neurons__
    ---

    Check out the guide on lists of neurons.

    [:octicons-arrow-right-24: NeuronLists tutorial](../generated/gallery/plot_02_neuronlists_intro)

-   :octicons-file-directory-symlink-16:{ .lg .middle } __Neuron I/O__

    ---

    Learn about how to load your own neurons into {{ navis }}.

    [:octicons-arrow-right-24: I/O Tutorials](../generated/gallery#import-export)

</div>