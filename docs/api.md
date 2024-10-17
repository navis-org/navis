---
icon: material/help-circle
---

# API Overview

{{ navis }} has grown a lot! Last I looked, there were over 120 functions and classes
exposed at top level (e.g. [`navis.plot3d`][]) and easily another 100 available via
submodules (e.g. [`navis.transforms.AffineTransform`][]). This can be a bit
daunting at first - especially if you don't exactly know what you are looking
for!

This document provides a curated, high-level summary. I recommend you either
just have a browse, use the search field (upper right) or simply search in
page (CONTROL/CMD-F). Failing that, please feel free to open a thread on
Github [Discussions](https://github.com/navis-org/navis/discussions) with
your question.

This API reference is a more or less complete account of the primary functions:

1. [Neuron- and NeuronList functions and methods](#neurons-neuronlists)
2. [Functions for visualization](#visualization)
3. [Manipulate or analyze neuron morphology](#neuron-morphology)
4. [Transforming and mirroring data](#transforming-and-mirroring)
5. [Analyze connectivity](#connectivity)
6. [Import/Export](#importexport)
7. [Utility functions](#utility)
8. [Which functions work with which neuron types?](#neuron-types-and-functions)

In addition {{ navis }} has interfaces to various external datasets and tools:

- [NEURON simulator](#neuron-simulator)
- [Neuromorpho](#neuromorpho-api)
- [neuPrint](#neuprint-api)
- [InsectBrain DB](#insectbrain-db-api)
- [Blender 3D](#blender-api)
- [Cytoscape](#cytoscape-api)
- [Allen MICrONS datasets](#allen-microns-datasets)
- [R and the natverse libraries](#r-interface)
<!-- - [Allen Cell Type Atlas]() -->

Most of these functions include examples of how to use them. Click on them to
learn more!

## Neurons & NeuronLists

``TreeNeurons``, ``MeshNeurons``, ``VoxelNeurons`` and ``Dotprops`` are neuron
classes. ``NeuronLists`` are containers thereof.

| Class | Description |
|------|------|
| [`navis.TreeNeuron`][] | Skeleton representation of a neuron. |
| [`navis.MeshNeuron`][] | Meshes with vertices and faces. |
| [`navis.VoxelNeuron`][] | 3D images (e.g. from confocal stacks). |
| [`navis.Dotprops`][] | Point cloud + vector representations, used for NBLAST. |
| [`navis.NeuronList`][] | Containers for neurons. |

### General Neuron methods

Despite being fundamentally different data types, all neurons share some common
methods (i.e. functions) which they inherit from their abstract parent
class [`BaseNeurons`][navis.BaseNeuron].

| Method | Description |
|--------|-------------|
| [`Neuron.copy()`][navis.BaseNeuron.copy] | {{ autosummary("navis.BaseNeuron.copy") }} |
| [`Neuron.plot3d()`][navis.BaseNeuron.plot3d] | {{ autosummary("navis.BaseNeuron.plot3d") }} |
| [`Neuron.plot2d()`][navis.BaseNeuron.plot2d] | {{ autosummary("navis.BaseNeuron.plot2d") }} |
| [`Neuron.summary()`][navis.BaseNeuron.summary] | {{ autosummary("navis.BaseNeuron.summary") }} |
| [`Neuron.convert_units()`][navis.BaseNeuron.convert_units] | {{ autosummary("navis.BaseNeuron.convert_units") }} |
| [`Neuron.map_units()`][navis.BaseNeuron.map_units] | {{ autosummary("navis.BaseNeuron.map_units") }} |
| [`Neuron.memory_usage()`][navis.BaseNeuron.memory_usage] | {{ autosummary("navis.BaseNeuron.memory_usage") }} |

In addition to methods, neurons also have properties. These properties common
to all neurons:


### General Neuron properties

| Property | Description |
|------|------|
| `Neuron.connectors` | {{ autosummary("navis.BaseNeuron.connectors") }} |
| `Neuron.postsynapses` | {{ autosummary("navis.BaseNeuron.postsynapses") }} |
| `Neuron.presynapses` | {{ autosummary("navis.BaseNeuron.presynapses") }} |
| `Neuron.datatables` | {{ autosummary("navis.BaseNeuron.datatables") }} |
| `Neuron.id` | {{ autosummary("navis.BaseNeuron.id") }} |
| `Neuron.name` | {{ autosummary("navis.BaseNeuron.name") }} |
| `Neuron.units` | {{ autosummary("navis.BaseNeuron.units") }} |
| `Neuron.type` | {{ autosummary("navis.BaseNeuron.type") }} |
| `Neuron.soma` | {{ autosummary("navis.BaseNeuron.soma") }} |
| `Neuron.bbox` | {{ autosummary("navis.BaseNeuron.bbox") }} |

!!! note

    Neurons _can_ have the above properties but that is not guaranteed. For example, a neuron might
    not have any associated synapses in which case `Neuron.connectors` will be `None`.


### Skeletons

A [`navis.TreeNeuron`][] represents a skeleton. These are class methods available specific for
this neuron type. Note that most of them are simply short-hands for the other
{{ navis }} functions:

| Method | Description |
|--------|-------------|
| [`TreeNeuron.convert_units()`][navis.TreeNeuron.convert_units] | {{ autosummary("navis.TreeNeuron.convert_units") }} |
| [`TreeNeuron.cell_body_fiber()`][navis.TreeNeuron.cell_body_fiber] | {{ autosummary("navis.TreeNeuron.cell_body_fiber") }} |
| [`TreeNeuron.downsample()`][navis.TreeNeuron.downsample] | {{ autosummary("navis.TreeNeuron.downsample") }} |
| [`TreeNeuron.get_graph_nx()`][navis.TreeNeuron.get_graph_nx] | {{ autosummary("navis.TreeNeuron.get_graph_nx") }} |
| [`TreeNeuron.get_igraph()`][navis.TreeNeuron.get_igraph] | {{ autosummary("navis.TreeNeuron.get_igraph") }} |
| [`TreeNeuron.prune_by_longest_neurite()`][navis.TreeNeuron.prune_by_longest_neurite] | {{ autosummary("navis.TreeNeuron.prune_by_longest_neurite") }} |
| [`TreeNeuron.prune_by_strahler()`][navis.TreeNeuron.prune_by_strahler] | {{ autosummary("navis.TreeNeuron.prune_by_strahler") }} |
| [`TreeNeuron.prune_by_volume()`][navis.TreeNeuron.prune_by_volume] | {{ autosummary("navis.TreeNeuron.prune_by_volume") }} |
| [`TreeNeuron.prune_distal_to()`][navis.TreeNeuron.prune_distal_to] | {{ autosummary("navis.TreeNeuron.prune_distal_to") }} |
| [`TreeNeuron.prune_proximal_to()`][navis.TreeNeuron.prune_proximal_to] | {{ autosummary("navis.TreeNeuron.prune_proximal_to") }} |
| [`TreeNeuron.prune_twigs()`][navis.TreeNeuron.prune_twigs] | {{ autosummary("navis.TreeNeuron.prune_twigs") }} |
| [`TreeNeuron.reload()`][navis.TreeNeuron.reload] | {{ autosummary("navis.TreeNeuron.reload") }} |
| [`TreeNeuron.reroot()`][navis.TreeNeuron.reroot] | {{ autosummary("navis.TreeNeuron.reroot") }} |
| [`TreeNeuron.resample()`][navis.TreeNeuron.resample] | {{ autosummary("navis.TreeNeuron.resample") }} |
| [`TreeNeuron.snap()`][navis.TreeNeuron.snap] | {{ autosummary("navis.TreeNeuron.snap") }} |

In addition, a [`navis.TreeNeuron`][] has a range of different properties:

| Method | Description |
|--------|-------------|
| [`TreeNeuron.adjacency_matrix`][navis.TreeNeuron.adjacency_matrix] | {{ autosummary("navis.TreeNeuron.adjacency_matrix") }} |
| [`TreeNeuron.cable_length`][navis.TreeNeuron.cable_length] | {{ autosummary("navis.TreeNeuron.cable_length") }} |
| [`TreeNeuron.cycles`][navis.TreeNeuron.cycles] | {{ autosummary("navis.TreeNeuron.cycles") }} |
| [`TreeNeuron.downsample`][navis.TreeNeuron.downsample] | {{ autosummary("navis.TreeNeuron.downsample") }} |
| [`TreeNeuron.edges`][navis.TreeNeuron.edges] | {{ autosummary("navis.TreeNeuron.edges") }} |
| [`TreeNeuron.edge_coords`][navis.TreeNeuron.edge_coords] | {{ autosummary("navis.TreeNeuron.edge_coords") }} |
| [`TreeNeuron.igraph`][navis.TreeNeuron.igraph] | {{ autosummary("navis.TreeNeuron.igraph") }} |
| [`TreeNeuron.is_tree`][navis.TreeNeuron.is_tree] | {{ autosummary("navis.TreeNeuron.is_tree") }} |
| [`TreeNeuron.n_branches`][navis.TreeNeuron.n_branches] | {{ autosummary("navis.TreeNeuron.n_branches") }} |
| [`TreeNeuron.n_leafs`][navis.TreeNeuron.n_leafs] | {{ autosummary("navis.TreeNeuron.n_leafs") }} |
| [`TreeNeuron.n_skeletons`][navis.TreeNeuron.n_skeletons] | {{ autosummary("navis.TreeNeuron.n_skeletons") }} |
| [`TreeNeuron.n_trees`][navis.TreeNeuron.n_trees] | {{ autosummary("navis.TreeNeuron.n_trees") }} |
| [`TreeNeuron.nodes`][navis.TreeNeuron.nodes] | {{ autosummary("navis.TreeNeuron.nodes") }} |
| [`TreeNeuron.root`][navis.TreeNeuron.root] | {{ autosummary("navis.TreeNeuron.root") }} |
| [`TreeNeuron.sampling_resolution`][navis.TreeNeuron.sampling_resolution] | {{ autosummary("navis.TreeNeuron.sampling_resolution") }} |
| [`TreeNeuron.segments`][navis.TreeNeuron.segments] | {{ autosummary("navis.TreeNeuron.segments") }} |
| [`TreeNeuron.simple`][navis.TreeNeuron.simple] | {{ autosummary("navis.TreeNeuron.simple") }} |
| [`TreeNeuron.soma_pos`][navis.TreeNeuron.soma_pos] | {{ autosummary("navis.TreeNeuron.soma_pos") }} |
| [`TreeNeuron.subtrees`][navis.TreeNeuron.subtrees] | {{ autosummary("navis.TreeNeuron.subtrees") }} |
| [`TreeNeuron.vertices`][navis.TreeNeuron.vertices] | {{ autosummary("navis.TreeNeuron.vertices") }} |
| [`TreeNeuron.volume`][navis.TreeNeuron.volume] | {{ autosummary("navis.TreeNeuron.volume") }} |


#### Skeleton utility functions

| Function | Description |
|----------|-------------|
| [`navis.rewire_skeleton()`][navis.rewire_skeleton] | {{ autosummary("navis.rewire_skeleton") }} |
| [`navis.insert_nodes()`][navis.insert_nodes] | {{ autosummary("navis.insert_nodes") }} |
| [`navis.remove_nodes()`][navis.remove_nodes] | {{ autosummary("navis.remove_nodes") }} |
| [`navis.graph.simplify_graph()`][navis.graph.simplify_graph] | {{ autosummary("navis.graph.simplify_graph") }} |
| [`navis.graph.skeleton_adjacency_matrix()`][navis.graph.skeleton_adjacency_matrix] | {{ autosummary("navis.graph.skeleton_adjacency_matrix") }} |



### Mesh neurons

Properties specific to [`navis.MeshNeuron`][]:

| Property | Description |
|------|------|
| [`MeshNeuron.faces`][navis.MeshNeuron.faces] | {{ autosummary("navis.MeshNeuron.faces") }} |
| [`MeshNeuron.vertices`][navis.MeshNeuron.vertices] | {{ autosummary("navis.MeshNeuron.vertices") }} |
| [`MeshNeuron.trimesh`][navis.MeshNeuron.trimesh] | {{ autosummary("navis.MeshNeuron.trimesh") }} |
| [`MeshNeuron.volume`][navis.MeshNeuron.volume] | {{ autosummary("navis.MeshNeuron.volume") }} |
| [`MeshNeuron.sampling_resolution`][navis.MeshNeuron.sampling_resolution] | {{ autosummary("navis.MeshNeuron.sampling_resolution") }} |

Methods specific to [`navis.MeshNeuron`][]:

| Method | Description |
|--------|-------------|
| [`MeshNeuron.skeletonize()`][navis.MeshNeuron.skeletonize] | {{ autosummary("navis.MeshNeuron.skeletonize") }} |
| [`MeshNeuron.snap()`][navis.MeshNeuron.snap] | {{ autosummary("navis.MeshNeuron.snap") }} |
| [`MeshNeuron.validate()`][navis.MeshNeuron.validate] | {{ autosummary("navis.MeshNeuron.validate") }} |


### Voxel neurons

[VoxelNeurons][navis.VoxelNeuron] (e.g. from confocal image stacks) are a relatively
recet addition to {{ navis }} and the interface might still change.

These are methods and properties specific to [VoxelNeurons][navis.VoxelNeuron]:

| Property | Description |
|------|------|
| [`VoxelNeuron.density`][navis.VoxelNeuron.density] | {{ autosummary("navis.VoxelNeuron.density") }} |
| [`VoxelNeuron.grid`][navis.VoxelNeuron.grid] | {{ autosummary("navis.VoxelNeuron.grid") }} |
| [`VoxelNeuron.max`][navis.VoxelNeuron.max] | {{ autosummary("navis.VoxelNeuron.max") }} |
| [`VoxelNeuron.min`][navis.VoxelNeuron.min] | {{ autosummary("navis.VoxelNeuron.min") }} |
| [`VoxelNeuron.nnz`][navis.VoxelNeuron.nnz] | {{ autosummary("navis.VoxelNeuron.nnz") }} |
| [`VoxelNeuron.offset`][navis.VoxelNeuron.offset] | {{ autosummary("navis.VoxelNeuron.offset") }} |
| [`VoxelNeuron.shape`][navis.VoxelNeuron.shape] | {{ autosummary("navis.VoxelNeuron.shape") }} |
| [`VoxelNeuron.strip()`][navis.VoxelNeuron.strip] | {{ autosummary("navis.VoxelNeuron.strip") }} |
| [`VoxelNeuron.threshold()`][navis.VoxelNeuron.threshold] | {{ autosummary("navis.VoxelNeuron.threshold") }} |
| [`VoxelNeuron.voxels`][navis.VoxelNeuron.voxels] | {{ autosummary("navis.VoxelNeuron.voxels") }} |


### Dotprops

[`navis.Dotprops`][] are typically indirectly generated from e.g. skeletons or
point clouds using [`navis.make_dotprops()`][navis.make_dotprops].

These are methods and properties specific to [Dotprops][navis.Dotprops]:

| Function | Description |
|----------|-------------|
| [`Dotprops.points`][navis.Dotprops.points] | {{ autosummary("navis.Dotprops.points") }} |
| [`Dotprops.vect`][navis.Dotprops.vect] | {{ autosummary("navis.Dotprops.vect") }} |
| [`Dotprops.alpha`][navis.Dotprops.alpha] | {{ autosummary("navis.Dotprops.alpha") }} |
| [`Dotprops.to_skeleton()`][navis.Dotprops.to_skeleton] | {{ autosummary("navis.Dotprops.to_skeleton") }} |
| [`Dotprops.snap()`][navis.Dotprops.snap] | {{ autosummary("navis.Dotprops.snap") }} |

### Converting between types

These functions will let you convert between neuron types:

| Function | Description |
|----------|-------------|
| [`navis.make_dotprops()`][navis.make_dotprops] | {{ autosummary("navis.make_dotprops") }} |
| [`navis.skeletonize()`][navis.skeletonize] | {{ autosummary("navis.skeletonize") }} |
| [`navis.mesh()`][navis.mesh] | {{ autosummary("navis.mesh") }} |
| [`navis.voxelize()`][navis.voxelize] | {{ autosummary("navis.voxelize") }} |
| [`navis.conversion.voxels2mesh()`][navis.conversion.voxels2mesh] | {{ autosummary("navis.conversion.voxels2mesh") }} |
| [`navis.conversion.tree2meshneuron()`][navis.conversion.tree2meshneuron] | {{ autosummary("navis.conversion.tree2meshneuron") }} |

See also [Utility](#utility) for functions to convert to/from basic data types.

### NeuronList methods

[`NeuronLists`][navis.NeuronList] let you access all the properties and methods of the neurons
they contain. In addition there are a few [`NeuronList`][navis.NeuronList]-specific methods and
properties.

Methods:

| Method | Description |
|--------|-------------|
| [`NeuronList.apply()`][navis.NeuronList.apply] | {{ autosummary("navis.NeuronList.apply") }} |
| [`NeuronList.head()`][navis.NeuronList.head] | {{ autosummary("navis.NeuronList.head") }} |
| [`NeuronList.itertuples()`][navis.NeuronList.itertuples] | {{ autosummary("navis.NeuronList.itertuples") }} |
| [`NeuronList.mean()`][navis.NeuronList.mean] | {{ autosummary("navis.NeuronList.mean") }} |
| [`NeuronList.remove_duplicates()`][navis.NeuronList.remove_duplicates] | {{ autosummary("navis.NeuronList.remove_duplicates") }} |
| [`NeuronList.sum()`][navis.NeuronList.sum] | {{ autosummary("navis.NeuronList.sum") }} |
| [`NeuronList.summary()`][navis.NeuronList.summary] | {{ autosummary("navis.NeuronList.summary") }} |
| [`NeuronList.tail()`][navis.NeuronList.tail] | {{ autosummary("navis.NeuronList.tail") }} |
| [`NeuronList.unmix()`][navis.NeuronList.unmix] | {{ autosummary("navis.NeuronList.unmix") }} |

Properties:

| Property | Description |
|----------|-------------|
| [`NeuronList.bbox`][navis.NeuronList.bbox] | {{ autosummary("navis.NeuronList.bbox") }} |
| [`NeuronList.empty`][navis.NeuronList.empty] | {{ autosummary("navis.NeuronList.empty") }} |
| [`NeuronList.id`][navis.NeuronList.id] | An array with the IDs of the neurons contained in the list. |
| [`NeuronList.idx`][navis.NeuronList.idx] | An indexer similar to pandas' `iloc` that accepts neuron IDs. |
| [`NeuronList.is_degenerated`][navis.NeuronList.is_degenerated] | {{ autosummary("navis.NeuronList.is_degenerated") }} |
| [`NeuronList.is_mixed`][navis.NeuronList.is_mixed] | {{ autosummary("navis.NeuronList.is_mixed") }} |
| [`NeuronList.shape`][navis.NeuronList.shape] | {{ autosummary("navis.NeuronList.shape") }} |
| [`NeuronList.types`][navis.NeuronList.types]| {{ autosummary("navis.NeuronList.types") }} |

Please see the [tutorial on ``NeuronList``](../generated/gallery/tutorial_basic_02_neuronlists_intro/) for more
information, including how to index them.

## Visualization

Various functions for plotting neurons and volumes.

| Function | Description |
|----------|-------------|
| [`navis.plot3d()`][navis.plot3d] | {{ autosummary("navis.plot3d") }} |
| [`navis.plot2d()`][navis.plot2d] | {{ autosummary("navis.plot2d") }} |
| [`navis.plot1d()`][navis.plot1d] | {{ autosummary("navis.plot1d") }} |
| [`navis.plot_flat()`][navis.plot_flat] | {{ autosummary("navis.plot_flat") }} |
| [`navis.clear3d()`][navis.clear3d] | {{ autosummary("navis.clear3d") }} |
| [`navis.close3d()`][navis.close3d] | {{ autosummary("navis.close3d") }} |
| [`navis.pop3d()`][navis.pop3d] | {{ autosummary("navis.pop3d") }} |
| [`navis.get_viewer()`][navis.get_viewer] | {{ autosummary("navis.get_viewer") }} |

### Plotting Volumes/Meshes

To plot meshes, you can pass ``trimesh.Trimesh`` objects directly to [`plot3d()`][navis.plot3d]
or [`plot2d()`][navis.plot2d]. However, {{ navis }} has a custom class to represent meshes that
has some useful perks: [`navis.Volume`][].

| Class | Description |
|-------|-------------|
| `navis.Volume` | {{ autosummary("navis.Volume") }} |
| `navis.Volume.combine` | {{ autosummary("navis.Volume.combine") }} |
| `navis.Volume.plot3d` | {{ autosummary("navis.Volume.plot3d") }} |
| `navis.Volume.validate` | {{ autosummary("navis.Volume.validate") }} |
| `navis.Volume.resize` | {{ autosummary("navis.Volume.resize") }} |

### Vispy 3D viewer

Using [`navis.plot3d()`][navis.plot3d] with `backend="vispy"` from a terminal will spawn
a Vispy 3D viewer object which has a bunch of useful methods. Note that this requires one of
navis' ``vispy-*`` extras to be installed, so that vispy has a backend.

| Function | Description |
|----------|-------------|
| [`navis.Viewer`][] | {{ autosummary("navis.Viewer") }} |
| [`navis.Viewer.add()`][navis.Viewer.add] | {{ autosummary("navis.Viewer.add") }} |
| [`navis.Viewer.clear()`][navis.Viewer.clear] | {{ autosummary("navis.Viewer.clear") }} |
| [`navis.Viewer.close()`][navis.Viewer.close] | {{ autosummary("navis.Viewer.close") }} |
| [`navis.Viewer.colorize()][navis.Viewer.colorize] | {{ autosummary("navis.Viewer.colorize") }} |
| [`navis.Viewer.set_colors()`][navis.Viewer.set_colors] | {{ autosummary("navis.Viewer.set_colors") }} |
| [`navis.Viewer.hide_neurons()`][navis.Viewer.hide_neurons] | {{ autosummary("navis.Viewer.hide_neurons") }} |
| [`navis.Viewer.unhide_neurons()`][navis.Viewer.unhide_neurons] | {{ autosummary("navis.Viewer.unhide_neurons") }} |
| [`navis.Viewer.screenshot()`][navis.Viewer.screenshot] | {{ autosummary("navis.Viewer.screenshot") }} |
| [`navis.Viewer.show()`][navis.Viewer.show] | {{ autosummary("navis.Viewer.show") }} |
| [`navis.Viewer.toggle_bounds()`][navis.Viewer.toggle_bounds] | {{ autosummary("navis.Viewer.toggle_bounds") }} |

### Octarine 3D viewer

Using [`navis.plot3d()`][navis.plot3d] with `backend="octarine"` from a terminal will return
an `octarine.Viewer` 3D viewer. Please see the `Octarine` [documentation](https://schlegelp.github.io/octarine/)
for details about the viewer.

## Neuron Morphology

Collection of functions to analyze and manipulate neuronal morphology.

### Analyse

Functions to analyze morphology.

| Function | Description |
|----------|-------------|
| [`navis.find_main_branchpoint()`][navis.find_main_branchpoint] | {{ autosummary("navis.find_main_branchpoint") }} |
| [`navis.form_factor()`][navis.form_factor] | {{ autosummary("navis.form_factor") }} |
| [`navis.persistence_points()`][navis.persistence_points] | {{ autosummary("navis.persistence_points") }} |
| [`navis.persistence_vectors()`][navis.persistence_vectors] | {{ autosummary("navis.persistence_vectors") }} |
| [`navis.strahler_index()`][navis.strahler_index] | {{ autosummary("navis.strahler_index") }} |
| [`navis.segment_analysis()`][navis.segment_analysis] | {{ autosummary("navis.segment_analysis") }} |
| [`navis.ivscc_features()`][navis.ivscc_features] | {{ autosummary("navis.ivscc_features") }} |
| [`navis.sholl_analysis()`][navis.sholl_analysis] | {{ autosummary("navis.sholl_analysis") }} |
| [`navis.tortuosity()`][navis.tortuosity] | {{ autosummary("navis.tortuosity") }} |
| [`navis.betweeness_centrality()`][navis.betweeness_centrality] | {{ autosummary("navis.betweeness_centrality") }} |

### Manipulate

Functions to edit morphology:

| Function | Description |
|----------|-------------|
| [`navis.average_skeletons()`][navis.average_skeletons] | {{ autosummary("navis.average_skeletons") }} |
| [`navis.break_fragments()`][navis.break_fragments] | {{ autosummary("navis.break_fragments") }} |
| [`navis.despike_skeleton()`][navis.despike_skeleton] | {{ autosummary("navis.despike_skeleton") }} |
| [`navis.drop_fluff()`][navis.drop_fluff] | {{ autosummary("navis.drop_fluff") }} |
| [`navis.cell_body_fiber()`][navis.cell_body_fiber] | {{ autosummary("navis.cell_body_fiber") }} |
| [`navis.combine_neurons()`][navis.combine_neurons] | {{ autosummary("navis.combine_neurons") }} |
| [`navis.cut_skeleton()`][navis.cut_skeleton] | {{ autosummary("navis.cut_skeleton") }} |
| [`navis.guess_radius()`][navis.guess_radius] | {{ autosummary("navis.guess_radius") }} |
| [`navis.heal_skeleton()`][navis.heal_skeleton] | {{ autosummary("navis.heal_skeleton") }} |
| [`navis.longest_neurite()`][navis.longest_neurite] | {{ autosummary("navis.longest_neurite") }} |
| [`navis.prune_by_strahler()`][navis.prune_by_strahler] | {{ autosummary("navis.prune_by_strahler") }} |
| [`navis.prune_twigs()`][navis.prune_twigs] | {{ autosummary("navis.prune_twigs") }} |
| [`navis.prune_at_depth()`][navis.prune_at_depth] | {{ autosummary("navis.prune_at_depth") }} |
| [`navis.reroot_skeleton()`][navis.reroot_skeleton] | {{ autosummary("navis.reroot_skeleton") }} |
| [`navis.split_axon_dendrite()`][navis.split_axon_dendrite] | {{ autosummary("navis.split_axon_dendrite") }} |
| [`navis.split_into_fragments()`][navis.split_into_fragments] | {{ autosummary("navis.split_into_fragments") }} |
| [`navis.stitch_skeletons()`][navis.stitch_skeletons] | {{ autosummary("navis.stitch_skeletons") }} |
| [`navis.subset_neuron()`][navis.subset_neuron] | {{ autosummary("navis.subset_neuron") }} |
| [`navis.smooth_skeleton()`][navis.smooth_skeleton] | {{ autosummary("navis.smooth_skeleton") }} |
| [`navis.smooth_mesh()`][navis.smooth_mesh] | {{ autosummary("navis.smooth_mesh") }} |
| [`navis.smooth_voxels()`][navis.smooth_voxels] | {{ autosummary("navis.smooth_voxels") }} |
| [`navis.thin_voxels()`][navis.thin_voxels] | {{ autosummary("navis.thin_voxels") }} |


### Resampling

Functions to down- or resample neurons.

| Function | Description |
|----------|-------------|
| [`navis.resample_skeleton()`][navis.resample_skeleton] | {{ autosummary("navis.resample_skeleton") }} |
| [`navis.resample_along_axis()`][navis.resample_along_axis] | {{ autosummary("navis.resample_along_axis") }} |
| [`navis.downsample_neuron()`][navis.downsample_neuron] | {{ autosummary("navis.downsample_neuron") }} |
| [`navis.simplify_mesh()`][navis.simplify_mesh] | {{ autosummary("navis.simplify_mesh") }} |


### Comparing morphology

NBLAST and related functions:

| Module | Description |
|--------|-------------|
| [`navis.nblast`][navis.nblast] | {{ autosummary("navis.nblast") }} |
| [`navis.nblast_smart`][navis.nblast_smart] | {{ autosummary("navis.nblast_smart") }} |
| [`navis.nblast_allbyall`][navis.nblast_allbyall] | {{ autosummary("navis.nblast_allbyall") }} |
| [`navis.nblast_align`][navis.nblast_align] | {{ autosummary("navis.nblast_align") }} |
| [`navis.synblast`][navis.synblast] | {{ autosummary("navis.synblast") }} |
| [`navis.persistence_distances`][navis.persistence_distances] | {{ autosummary("navis.persistence_distances") }} |

#### Utilities for creating your own score matrices for NBLAST:

| Function | Description |
|----------|-------------|
| [`navis.nbl.smat.Lookup2d`][navis.nbl.smat.Lookup2d] | {{ autosummary("navis.nbl.smat.Lookup2d") }} |
| [`navis.nbl.smat.Digitizer`][navis.nbl.smat.Digitizer] | {{ autosummary("navis.nbl.smat.Digitizer") }} |
| [`navis.nbl.smat.LookupDistDotBuilder`][navis.nbl.smat.LookupDistDotBuilder] | {{ autosummary("navis.nbl.smat.LookupDistDotBuilder") }} |

#### Utilities for NBLAST

| Function | Description |
|----------|-------------|
| [`navis.nbl.make_clusters()`][navis.nbl.make_clusters] | {{ autosummary("navis.nbl.make_clusters") }} |
| [`navis.nbl.update_scores()`][navis.nbl.update_scores] | {{ autosummary("navis.nbl.update_scores") }} |
| [`navis.nbl.compress_scores()`][navis.nbl.compress_scores] | {{ autosummary("navis.nbl.compress_scores") }} |
| [`navis.nbl.extract_matches()`][navis.nbl.extract_matches] | {{ autosummary("navis.nbl.extract_matches") }} |
| [`navis.nbl.nblast_prime()`][navis.nbl.nblast_prime] | {{ autosummary("navis.nbl.nblast_prime") }} |

### Polarity metrics

| Function | Description |
|----------|-------------|
| [`navis.bending_flow()`][navis.bending_flow] | {{ autosummary("navis.bending_flow") }} |
| [`navis.flow_centrality()`][navis.flow_centrality] | {{ autosummary("navis.flow_centrality") }} |
| [`navis.synapse_flow_centrality()`][navis.synapse_flow_centrality] | {{ autosummary("navis.synapse_flow_centrality") }} |
| [`navis.arbor_segregation_index()`][navis.arbor_segregation_index] | {{ autosummary("navis.arbor_segregation_index") }} |
| [`navis.segregation_index()`][navis.segregation_index] | {{ autosummary("navis.segregation_index") }} |

### Distances

Functions to calculate Euclidean and geodesic ("along-the-arbor") distances.

| Function | Description |
|----------|-------------|
| [`navis.cable_overlap()`][navis.cable_overlap] | {{ autosummary("navis.cable_overlap") }} |
| [`navis.distal_to()`][navis.distal_to] | {{ autosummary("navis.distal_to") }} |
| [`navis.dist_between()`][navis.dist_between] | {{ autosummary("navis.dist_between") }} |
| [`navis.dist_to_root()`][navis.dist_to_root] | {{ autosummary("navis.dist_to_root") }} |
| [`navis.geodesic_matrix()`][navis.geodesic_matrix] | {{ autosummary("navis.geodesic_matrix") }} |
| [`navis.segment_length()`][navis.segment_length] | {{ autosummary("navis.segment_length") }} |

## Intersection

Functions to intersect points and neurons with volumes. For example, if you'd
like to know which part of a neuron is inside a certain brain region.

| Function | Description |
|----------|-------------|
| [`navis.in_volume()`][navis.in_volume] | {{ autosummary("navis.in_volume") }} |
| [`navis.intersection_matrix()`][navis.intersection_matrix] | {{ autosummary("navis.intersection_matrix") }} |


## Transforming and Mirroring

Functions to transform spatial data, e.g. move neurons from one brain space to
another. Check out the [tutorials](../generated/gallery/6_misc/tutorial_misc_01_transforms/) for examples on how to
use them.

High-level functions:

| Function | Description |
|----------|-------------|
| [`navis.xform()`][navis.xform] | {{ autosummary("navis.xform") }} |
| [`navis.xform_brain()`][navis.xform_brain] | {{ autosummary("navis.xform_brain") }} |
| [`navis.symmetrize_brain()`][navis.symmetrize_brain] | {{ autosummary("navis.symmetrize_brain") }} |
| [`navis.mirror_brain()`][navis.mirror_brain] | {{ autosummary("navis.mirror_brain") }} |
| [`navis.transforms.mirror()`][navis.transforms.mirror] |
| [`navis.align.align_rigid()`][navis.align.align_rigid] | {{ autosummary("navis.align.align_rigid") }} |
| [`navis.align.align_deform()`][navis.align.align_deform] | {{ autosummary("navis.align.align_deform") }} |
| [`navis.align.align_pca()`][navis.align.align_pca] | {{ autosummary("navis.align.align_pca") }} |
| [`navis.align.align_pairwise()`][navis.align.align_pairwise] | {{ autosummary("navis.align.align_pairwise") }} |

{{ navis }} supports several types of transforms:


| Class | Description |
|-------|-------------|
| [`navis.transforms.AffineTransform`][] | {{ autosummary("navis.transforms.AffineTransform") }} |
| [`navis.transforms.ElastixTransform`][] | {{ autosummary("navis.transforms.ElastixTransform") }} |
| [`navis.transforms.CMTKtransform`][] | {{ autosummary("navis.transforms.CMTKtransform") }} |
| [`navis.transforms.H5transform`][] | {{ autosummary("navis.transforms.H5transform") }} |
| [`navis.transforms.TPStransform`][] | {{ autosummary("navis.transforms.TPStransform") }} |
| [`navis.transforms.AliasTransform`][] | {{ autosummary("navis.transforms.AliasTransform") }} |
| [`navis.transforms.MovingLeastSquaresTransform`][] | {{ autosummary("navis.transforms.MovingLeastSquaresTransform") }} |

The [`TemplateRegistry`][navis.transforms.templates.TemplateRegistry] keeps track of template brains, transforms and such:

| Class    | Description |
|----------|-------------|
| [`navis.transforms.templates.TemplateRegistry`][] | {{ autosummary("navis.transforms.templates.TemplateRegistry") }} |

The relevant instance of this class is ``navis.transforms.registry``.
So to register and use a new transform you would look something like this:

``` python
>>> transform = navis.transforms.AffineTransform(...)
>>> navis.transforms.registry.register_transform(transform,
...                                              source='brainA',
...                                              target='brainB')
>>> xf = navis.xform_brain(data, 'brainA', 'brainB')
```

You can check which transforms are registered like so:

``` python
>>> navis.transforms.registry.summary()  # this outputs a dataframe
```

These are the methods and properties of ``registry``:

| Method   | Description |
|----------|-------------|
| [`TemplateRegistry.register_transform()`][navis.transforms.templates.TemplateRegistry.register_transform] | {{ autosummary("navis.transforms.templates.TemplateRegistry.register_transform") }} |
| [`TemplateRegistry.register_transformfile()`][navis.transforms.templates.TemplateRegistry.register_transformfile] | {{ autosummary("navis.transforms.templates.TemplateRegistry.register_transformfile") }} |
| [`TemplateRegistry.register_templatebrain()`][navis.transforms.templates.TemplateRegistry.register_templatebrain] | {{ autosummary("navis.transforms.templates.TemplateRegistry.register_templatebrain") }} |
| [`TemplateRegistry.register_path()`][navis.transforms.templates.TemplateRegistry.register_path] | {{ autosummary("navis.transforms.templates.TemplateRegistry.register_path") }} |
| [`TemplateRegistry.scan_paths()`][navis.transforms.templates.TemplateRegistry.scan_paths] | {{ autosummary("navis.transforms.templates.TemplateRegistry.scan_paths") }} |
| [`TemplateRegistry.plot_bridging_graph()`][navis.transforms.templates.TemplateRegistry.plot_bridging_graph] | {{ autosummary("navis.transforms.templates.TemplateRegistry.plot_bridging_graph") }} |
| [`TemplateRegistry.find_mirror_reg()`][navis.transforms.templates.TemplateRegistry.find_mirror_reg] | {{ autosummary("navis.transforms.templates.TemplateRegistry.find_mirror_reg") }} |
| [`TemplateRegistry.find_bridging_path()`][navis.transforms.templates.TemplateRegistry.find_bridging_path] | {{ autosummary("navis.transforms.templates.TemplateRegistry.find_bridging_path") }} |
| [`TemplateRegistry.shortest_bridging_seq()`][navis.transforms.templates.TemplateRegistry.shortest_bridging_seq] | {{ autosummary("navis.transforms.templates.TemplateRegistry.shortest_bridging_seq") }} |
| [`TemplateRegistry.clear_caches()`][navis.transforms.templates.TemplateRegistry.clear_caches] | {{ autosummary("navis.transforms.templates.TemplateRegistry.clear_caches") }} |
| [`TemplateRegistry.summary()`][navis.transforms.templates.TemplateRegistry.summary] | {{ autosummary("navis.transforms.templates.TemplateRegistry.summary") }} |
| [`TemplateRegistry.transforms()`][navis.transforms.templates.TemplateRegistry.transforms] | {{ autosummary("navis.transforms.templates.TemplateRegistry.transforms") }} |
| [`TemplateRegistry.mirrors()`][navis.transforms.templates.TemplateRegistry.mirrors] | {{ autosummary("navis.transforms.templates.TemplateRegistry.mirrors") }} |
| [`TemplateRegistry.bridges()`][navis.transforms.templates.TemplateRegistry.bridges] | {{ autosummary("navis.transforms.templates.TemplateRegistry.bridges") }} |

## Connectivity

Collection of functions to work with graphs and adjacency matrices.

| Function | Description |
|----------|-------------|
| [`navis.NeuronConnector`][] | {{ autosummary("navis.NeuronConnector") }} |

### Connectivity metrics

Functions to analyse/cluster neurons based on connectivity.

| Function | Description |
|----------|-------------|
| [`navis.connectivity_similarity()`][navis.connectivity_similarity] | {{ autosummary("navis.connectivity_similarity") }} |
| [`navis.connectivity_sparseness()`][navis.connectivity_sparseness] | {{ autosummary("navis.connectivity_sparseness") }} |
| [`navis.cable_overlap()`][navis.cable_overlap] | {{ autosummary("navis.cable_overlap") }} |
| [`navis.synapse_similarity()`][navis.synapse_similarity] | {{ autosummary("navis.synapse_similarity") }} |


## Import/Export

Functions to import neurons.

| Function | Description |
|----------|-------------|
| [`navis.read_swc()`][navis.read_swc] | {{ autosummary("navis.read_swc") }} |
| [`navis.read_nrrd()`][navis.read_nrrd] | {{ autosummary("navis.read_nrrd") }} |
| [`navis.read_mesh()`][navis.read_mesh] | {{ autosummary("navis.read_mesh") }} |
| [`navis.read_tiff()`][navis.read_tiff] | {{ autosummary("navis.read_tiff") }} |
| [`navis.read_nmx()`][navis.read_nmx] | {{ autosummary("navis.read_nmx") }} |
| [`navis.read_nml()`][navis.read_nml] | {{ autosummary("navis.read_nml") }} |
| [`navis.read_rda()`][navis.read_rda] | {{ autosummary("navis.read_rda") }} |
| [`navis.read_json()`][navis.read_json] | {{ autosummary("navis.read_json") }} |
| [`navis.read_precomputed()`][navis.read_precomputed] | {{ autosummary("navis.read_precomputed") }} |
| [`navis.read_parquet()`][navis.read_parquet] | {{ autosummary("navis.read_parquet") }} |
| [`navis.scan_parquet()`][navis.scan_parquet] | {{ autosummary("navis.scan_parquet") }} |


Functions to export neurons.

| Function | Description |
|----------|-------------|
| [`navis.write_swc()`][navis.write_swc] | {{ autosummary("navis.write_swc") }} |
| [`navis.write_nrrd()`][navis.write_nrrd] | {{ autosummary("navis.write_nrrd") }} |
| [`navis.write_mesh()`][navis.write_mesh] | {{ autosummary("navis.write_mesh") }} |
| [`navis.write_json()`][navis.write_json] | {{ autosummary("navis.write_json") }} |
| [`navis.write_precomputed()`][navis.write_precomputed] | {{ autosummary("navis.write_precomputed") }} |
| [`navis.write_parquet()`][navis.write_parquet] | {{ autosummary("navis.write_parquet") }} |

## Utility

Various utility functions.

| Function | Description |
|----------|-------------|
| [`navis.health_check()`][navis.health_check] | {{ autosummary("navis.health_check") }} |
| [`navis.set_pbars()`][navis.set_pbars] | {{ autosummary("navis.set_pbars") }} |
| [`navis.set_loggers()`][navis.set_loggers] | {{ autosummary("navis.set_loggers") }} |
| [`navis.set_default_connector_colors()`][navis.set_default_connector_colors] | {{ autosummary("navis.set_default_connector_colors") }} |
| [`navis.config.remove_log_handlers()`][navis.config.remove_log_handlers] | {{ autosummary("navis.config.remove_log_handlers") }} |
| [`navis.patch_cloudvolume()`][navis.patch_cloudvolume] | {{ autosummary("navis.patch_cloudvolume") }} |
| [`navis.example_neurons()`][navis.example_neurons] | {{ autosummary("navis.example_neurons") }} |
| [`navis.example_volume()`][navis.example_volume] | {{ autosummary("navis.example_volume") }} |

### Conversion

Functions to convert between data types.

| Function | Description |
|----------|-------------|
| [`navis.neuron2nx()`][navis.neuron2nx] | {{ autosummary("navis.neuron2nx") }} |
| [`navis.neuron2igraph()`][navis.neuron2igraph] | {{ autosummary("navis.neuron2igraph") }} |
| [`navis.neuron2KDTree()`][navis.neuron2KDTree] | {{ autosummary("navis.neuron2KDTree") }} |
| [`navis.neuron2tangents()`][navis.neuron2tangents] | {{ autosummary("navis.neuron2tangents") }} |
| [`navis.network2nx()`][navis.network2nx] | {{ autosummary("navis.network2nx") }} |
| [`navis.network2igraph()`][navis.network2igraph] | {{ autosummary("navis.network2igraph") }} |
| [`navis.nx2neuron()`][navis.nx2neuron] | {{ autosummary("navis.nx2neuron") }} |
| [`navis.edges2neuron()`][navis.edges2neuron] | {{ autosummary("navis.edges2neuron") }} |

## Network Models

{{ navis }} comes with a simple network traversal model (used in [Schlegel, Bates et al., 2021](https://elifesciences.org/articles/66018)).

_Not imported at top level! Must be imported explicitly:_

``` python
from navis import models
```

| Class    | Description |
|----------|-------------|
| [`navis.models.TraversalModel`][] | {{ autosummary("navis.models.TraversalModel") }} |
| [`navis.models.BayesianTraversalModel`][] | {{ autosummary("navis.models.BayesianTraversalModel") }} |


## Interfaces

Interfaces with various external tools/websites. These modules have to be
imported explicitly as they are not imported at top level.


### NEURON simulator

Functions to facilitate creating models of neurons/networks. Please see
the [tutorials](../generated/gallery/3_interfaces/tutorial_interfaces_00_neuron/) for examples.

_Not imported at top level! Must be imported explicitly:_

``` python
import navis.interfaces.neuron as nrn
```

#### Compartment models

A single-neuron compartment model is represented by
[`CompartmentModel`][navis.interfaces.neuron.comp.CompartmentModel]:

| Class    | Description |
|----------|-------------|
| [`CompartmentModel`][navis.interfaces.neuron.comp.CompartmentModel] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel") }} |
| [`DrosophilaPN`][navis.interfaces.neuron.comp.DrosophilaPN] | {{ autosummary("navis.interfaces.neuron.comp.DrosophilaPN") }} |

The [`DrosophilaPN`][navis.interfaces.neuron.comp.DrosophilaPN]  class is a subclass
of [`CompartmentModel`][navis.interfaces.neuron.comp.CompartmentModel] with
pre-defined properties from Tobin et al. (2017).

| Method   | Description |
|----------|-------------|
| [`CompartmentModel.add_current_record()`][navis.interfaces.neuron.comp.CompartmentModel.add_current_record] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.add_current_record") }} |
| [`CompartmentModel.add_spike_detector()`][navis.interfaces.neuron.comp.CompartmentModel.add_spike_detector] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.add_spike_detector") }} |
| [`CompartmentModel.add_synaptic_current()`][navis.interfaces.neuron.comp.CompartmentModel.add_synaptic_current] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.add_synaptic_current") }} |
| [`CompartmentModel.add_synaptic_input()`][navis.interfaces.neuron.comp.CompartmentModel.add_synaptic_input] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.add_synaptic_input") }} |
| [`CompartmentModel.add_voltage_record()`][navis.interfaces.neuron.comp.CompartmentModel.add_voltage_record] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.add_voltage_record") }} |
| [`CompartmentModel.clear_records()`][navis.interfaces.neuron.comp.CompartmentModel.clear_records] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.clear_records") }} |
| [`CompartmentModel.clear_stimuli()`][navis.interfaces.neuron.comp.CompartmentModel.clear_stimuli] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.clear_stimuli") }} |
| [`CompartmentModel.connect()`][navis.interfaces.neuron.comp.CompartmentModel.connect] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.connect") }} |
| [`CompartmentModel.get_node_section()`][navis.interfaces.neuron.comp.CompartmentModel.get_node_section] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.get_node_section") }} |
| [`CompartmentModel.get_node_segment()`][navis.interfaces.neuron.comp.CompartmentModel.get_node_segment] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.get_node_segment") }} |
| [`CompartmentModel.inject_current_pulse()`][navis.interfaces.neuron.comp.CompartmentModel.inject_current_pulse] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.inject_current_pulse") }} |
| [`CompartmentModel.plot_results()`][navis.interfaces.neuron.comp.CompartmentModel.plot_results] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.plot_results") }} |
| [`CompartmentModel.insert()`][navis.interfaces.neuron.comp.CompartmentModel.insert] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.insert") }} |
| [`CompartmentModel.uninsert()`][navis.interfaces.neuron.comp.CompartmentModel.uninsert] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.uninsert") }} |


| Attribute | Description |
|-----------|-------------|
| [`CompartmentModel.Ra`][navis.interfaces.neuron.comp.CompartmentModel.Ra] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.Ra") }} |
| [`CompartmentModel.cm`][navis.interfaces.neuron.comp.CompartmentModel.cm] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.cm") }} |
| [`CompartmentModel.label`][navis.interfaces.neuron.comp.CompartmentModel.label] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.label") }} |
| [`CompartmentModel.n_records`][navis.interfaces.neuron.comp.CompartmentModel.n_records] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.n_records") }} |
| [`CompartmentModel.n_sections`][navis.interfaces.neuron.comp.CompartmentModel.n_sections] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.n_sections") }} |
| [`CompartmentModel.n_stimuli`][navis.interfaces.neuron.comp.CompartmentModel.n_stimuli] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.n_stimuli") }} |
| [`CompartmentModel.nodes`][navis.interfaces.neuron.comp.CompartmentModel.nodes] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.nodes") }} |
| [`CompartmentModel.records`][navis.interfaces.neuron.comp.CompartmentModel.records] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.records") }} |
| [`CompartmentModel.sections`][navis.interfaces.neuron.comp.CompartmentModel.sections] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.sections") }} |
| [`CompartmentModel.stimuli`][navis.interfaces.neuron.comp.CompartmentModel.stimuli] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.stimuli") }} |
| [`CompartmentModel.synapses`][navis.interfaces.neuron.comp.CompartmentModel.synapses] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.synapses") }} |
| [`CompartmentModel.t`][navis.interfaces.neuron.comp.CompartmentModel.t] | {{ autosummary("navis.interfaces.neuron.comp.CompartmentModel.t") }} |

#### Network models

A network of point-processes is represented by [`PointNetwork`][navis.interfaces.neuron.network.PointNetwork]:


| Class                                         | Description                                      |
|-----------------------------------------------|--------------------------------------------------|
| [`PointNetwork`][navis.interfaces.neuron.network.PointNetwork] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork") }}      |


| Methods  | Description |
|----------|-------------|
| [`PointNetwork.add_background_noise()`][navis.interfaces.neuron.network.PointNetwork.add_background_noise] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.add_background_noise") }} |
| [`PointNetwork.add_neurons()`][navis.interfaces.neuron.network.PointNetwork.add_neurons] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.add_neurons") }} |
| [`PointNetwork.add_stimulus()`][navis.interfaces.neuron.network.PointNetwork.add_stimulus] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.add_stimulus") }} |
| [`PointNetwork.connect()`][navis.interfaces.neuron.network.PointNetwork.connect] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.connect") }} |
| [`PointNetwork.from_edge_list()`][navis.interfaces.neuron.network.PointNetwork.from_edge_list] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.from_edge_list") }} |
| [`PointNetwork.get_spike_counts()`][navis.interfaces.neuron.network.PointNetwork.get_spike_counts] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.get_spike_counts") }} |
| [`PointNetwork.plot_raster()`][navis.interfaces.neuron.network.PointNetwork.plot_raster] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.plot_raster") }} |
| [`PointNetwork.plot_traces()`][navis.interfaces.neuron.network.PointNetwork.plot_traces] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.plot_traces") }} |
| [`PointNetwork.run_simulation()`][navis.interfaces.neuron.network.PointNetwork.run_simulation] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.run_simulation") }} |
| [`PointNetwork.set_labels()`][navis.interfaces.neuron.network.PointNetwork.set_labels] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.set_labels") }} |


| Attributes | Description |
|------------|-------------|
| [`PointNetwork.edges`][navis.interfaces.neuron.network.PointNetwork.edges] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.edges") }} |
| [`PointNetwork.ids`][navis.interfaces.neuron.network.PointNetwork.ids] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.ids") }} |
| [`PointNetwork.labels`][navis.interfaces.neuron.network.PointNetwork.labels] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.labels") }} |
| [`PointNetwork.neurons`][navis.interfaces.neuron.network.PointNetwork.neurons] | {{ autosummary("navis.interfaces.neuron.network.PointNetwork.neurons") }} |


### NeuroMorpho API

Set of functions to grab data from [NeuroMorpho](http://neuromorpho.org) which hosts thousands of neurons (see [tutorials](../generated/gallery/)).

_Not imported at top level! Must be imported explicitly:_

```python
from navis.interfaces import neuromorpho
```

| Function | Description |
|----------|-------------|
| [`neuromorpho.get_neuron_info()`][navis.interfaces.neuromorpho.get_neuron_info] | {{ autosummary("navis.interfaces.neuromorpho.get_neuron_info") }} |
| [`neuromorpho.get_neuron()`][navis.interfaces.neuromorpho.get_neuron] | {{ autosummary("navis.interfaces.neuromorpho.get_neuron") }} |
| [`neuromorpho.get_neuron_fields()`][navis.interfaces.neuromorpho.get_neuron_fields] | {{ autosummary("navis.interfaces.neuromorpho.get_neuron_fields") }} |
| [`neuromorpho.get_available_field_values()`][navis.interfaces.neuromorpho.get_available_field_values] | {{ autosummary("navis.interfaces.neuromorpho.get_available_field_values") }} |


### neuPrint API

{{ navis }} wraps [`neuprint-python`](https://github.com/connectome-neuprint/neuprint-python)
and adds a few navis-specific functions. You must have `neuprint-python` installed for this to work:

```shell
pip install neuprint-python -U
```

You can then import neuprint from {{ navis }} like so:

```python
from navis.interfaces import neuprint
```

These are the additional functions added by {{ navis }}:

| Function | Description |
|----------|-------------|
| [`neuprint.fetch_roi()`][navis.interfaces.neuprint.fetch_roi] | {{ autosummary("navis.interfaces.neuprint.fetch_roi") }} |
| [`neuprint.fetch_skeletons()`][navis.interfaces.neuprint.fetch_skeletons] | {{ autosummary("navis.interfaces.neuprint.fetch_skeletons") }} |
| [`neuprint.fetch_mesh_neuron()`][navis.interfaces.neuprint.fetch_mesh_neuron] | {{ autosummary("navis.interfaces.neuprint.fetch_mesh_neuron") }} |

Please also check out the [tutorials](../generated/gallery/4_remote/tutorial_remote_00_neuprint/) for examples of how to fetch and work with data from neuPrint.

### InsectBrain DB API

Set of functions to grab data from [InsectBrain](https://www.insectbraindb.org)
which hosts some neurons and standard brains (see [tutorials](../generated/gallery/4_remote/tutorial_remote_03_insect_db/)).

_Not imported at top level! Must be imported explicitly:_

```python
from navis.interfaces import insectbrain_db
```

| Function | Description |
|----------|-------------|
| [`insectbrain_db.authenticate()`][navis.interfaces.insectbrain_db.authenticate] | {{ autosummary("navis.interfaces.insectbrain_db.authenticate") }} |
| [`insectbrain_db.get_brain_meshes()`][navis.interfaces.insectbrain_db.get_brain_meshes] | {{ autosummary("navis.interfaces.insectbrain_db.get_brain_meshes") }} |
| [`insectbrain_db.get_species_info()`][navis.interfaces.insectbrain_db.get_species_info] | {{ autosummary("navis.interfaces.insectbrain_db.get_species_info") }} |
| [`insectbrain_db.get_available_species()`][navis.interfaces.insectbrain_db.get_available_species] | {{ autosummary("navis.interfaces.insectbrain_db.get_available_species") }} |
| [`insectbrain_db.get_skeletons()`][navis.interfaces.insectbrain_db.get_skeletons] | {{ autosummary("navis.interfaces.insectbrain_db.get_skeletons") }} |
| [`insectbrain_db.get_skeletons_species()`][navis.interfaces.insectbrain_db.get_skeletons_species] | {{ autosummary("navis.interfaces.insectbrain_db.get_skeletons_species") }} |
| [`insectbrain_db.search_neurons()`][navis.interfaces.insectbrain_db.search_neurons] | {{ autosummary("navis.interfaces.insectbrain_db.search_neurons") }} |


### Blender API

Functions to be run inside [Blender 3D](https://www.blender.org/) and import
CATMAID data (see Examples). Please note that this requires Blender >2.8 as
earlier versions are shipped with older Python versions not supported by {{ navis }}.
See the [tutorials](../generated/gallery/3_interfaces/tutorial_interfaces_02_blender/) for an introduction of how to use {{ navis }} in
Blender.

_Not imported at top level! Must be imported explicitly:_

```python
from navis.interfaces import blender
```

The interface is realised through a [`navis.interfaces.blender.Handler`][]
object. It is used to import objects and facilitate working with them
programmatically once they are imported.

| Class    | Description |
|----------|-------------|
| [`blender.Handler`][navis.interfaces.blender.Handler] | {{ autosummary("navis.interfaces.blender.Handler") }} |

#### Objects

| Method   | Description |
|----------|-------------|
| [`blender.Handler.add()`][navis.interfaces.blender.Handler.add] | {{ autosummary("navis.interfaces.blender.Handler.add") }} |
| [`blender.Handler.clear()`][navis.interfaces.blender.Handler.clear] | {{ autosummary("navis.interfaces.blender.Handler.clear") }} |
| [`blender.Handler.select()`][navis.interfaces.blender.Handler.select] | {{ autosummary("navis.interfaces.blender.Handler.select") }} |
| [`blender.Handler.hide()`][navis.interfaces.blender.Handler.hide] | {{ autosummary("navis.interfaces.blender.Handler.hide") }} |
| [`blender.Handler.unhide()`][navis.interfaces.blender.Handler.unhide] | {{ autosummary("navis.interfaces.blender.Handler.unhide") }} |

#### Materials

| Properties | Description |
|------------|-------------|
| [`blender.Handler.color()`][navis.interfaces.blender.Handler.color] | {{ autosummary("navis.interfaces.blender.Handler.color") }} |
| [`blender.Handler.colorize()`][navis.interfaces.blender.Handler.colorize] | {{ autosummary("navis.interfaces.blender.Handler.colorize") }} |
| [`blender.Handler.emit()`][navis.interfaces.blender.Handler.emit] | {{ autosummary("navis.interfaces.blender.Handler.emit") }} |
| [`blender.Handler.use_transparency()`][navis.interfaces.blender.Handler.use_transparency] | {{ autosummary("navis.interfaces.blender.Handler.use_transparency") }} |
| [`blender.Handler.alpha()`][navis.interfaces.blender.Handler.alpha] | {{ autosummary("navis.interfaces.blender.Handler.alpha") }} |
| [`blender.Handler.bevel()`][navis.interfaces.blender.Handler.bevel] | {{ autosummary("navis.interfaces.blender.Handler.bevel") }} |

#### Selections

You can use [`Handler.select()`][navis.interfaces.blender.Handler.select] to select a group of neurons e.g. by type. That method
then returns [`ObjectList`][navis.interfaces.blender.ObjectList] which can be used to modify just the selected objects:

| Methods  | Description |
|----------|-------------|
| [`blender.Handler.select()`][navis.interfaces.blender.Handler.select] | {{ autosummary("navis.interfaces.blender.Handler.select") }} |
| [`blender.ObjectList.select()`][navis.interfaces.blender.ObjectList.select] | {{ autosummary("navis.interfaces.blender.ObjectList.select") }} |
| [`blender.ObjectList.color()`][navis.interfaces.blender.ObjectList.color] | {{ autosummary("navis.interfaces.blender.ObjectList.color") }} |
| [`blender.ObjectList.colorize()`][navis.interfaces.blender.ObjectList.colorize] | {{ autosummary("navis.interfaces.blender.ObjectList.colorize") }} |
| [`blender.ObjectList.emit()`][navis.interfaces.blender.ObjectList.emit] | {{ autosummary("navis.interfaces.blender.ObjectList.emit") }} |
| [`blender.ObjectList.use_transparency()`][navis.interfaces.blender.ObjectList.use_transparency] | {{ autosummary("navis.interfaces.blender.ObjectList.use_transparency") }} |
| [`blender.ObjectList.alpha()`][navis.interfaces.blender.ObjectList.alpha] | {{ autosummary("navis.interfaces.blender.ObjectList.alpha") }} |
| [`blender.ObjectList.bevel()`][navis.interfaces.blender.ObjectList.bevel] | {{ autosummary("navis.interfaces.blender.ObjectList.bevel") }} |
| [`blender.ObjectList.hide()`][navis.interfaces.blender.ObjectList.hide] | {{ autosummary("navis.interfaces.blender.ObjectList.hide") }} |
| [`blender.ObjectList.unhide()`][navis.interfaces.blender.ObjectList.unhide] | {{ autosummary("navis.interfaces.blender.ObjectList.unhide") }} |
| [`blender.ObjectList.hide_others()`][navis.interfaces.blender.ObjectList.hide_others] | {{ autosummary("navis.interfaces.blender.ObjectList.hide_others") }} |
| [`blender.ObjectList.delete()`][navis.interfaces.blender.ObjectList.delete] | {{ autosummary("navis.interfaces.blender.ObjectList.delete") }} |

### Cytoscape API

!!! warning Deprecated
    The Cytoscape API is depcrecated and will be removed in a future version of {{ navis }}.

Functions to use [Cytoscape](https://cytoscape.org) via the cyREST API.

_Not imported at top level! Must be imported explicitly:_

```python
from navis.interfaces import cytoscape
```

| Function                            | Description                            |
|-------------------------------------|----------------------------------------|
| [`cytoscape.generate_network()`][navis.interfaces.cytoscape.generate_network] | {{ autosummary("navis.interfaces.cytoscapecytoscape.generate_network") }} |
| [`cytoscape.get_client()`][navis.interfaces.cytoscape.get_client] | {{ autosummary("navis.interfaces.cytoscapecytoscape.get_client") }} |


### Allen MICrONS datasets

Functions to fetch neurons (including synapses) from the Allen Institute's
[MICrONS](https://www.microns-explorer.org/) EM datasets.

Requires `caveclient` and `cloud-volume` as additional dependencies:

```shell
pip3 install caveclient cloud-volume -U
```

Please see [caveclient's docs](https://caveconnectome.github.io/CAVEclient/) for details on how to retrieve and set credentials.

_Not imported at top level! Must be imported explicitly:_

```python
from navis.interfaces import microns
```

| Function | Description |
|----------|-------------|
| [`microns.get_cave_client()`][navis.interfaces.microns.get_cave_client] | {{ autosummary("navis.interfaces.microns.get_cave_client") }} |
| [`microns.fetch_neurons()`][navis.interfaces.microns.fetch_neurons] | {{ autosummary("navis.interfaces.microns.fetch_neurons") }} |
| [`microns.get_somas()`][navis.interfaces.microns.get_somas] | {{ autosummary("navis.interfaces.microns.get_somas") }} |

Please also see the [MICrONS tutorial](../generated/gallery/4_remote/tutorial_remote_02_microns/).


### H01 dataset

Functions to fetch neurons (including synapses) from the
[H01](https://h01-release.storage.googleapis.com/landing.html) connectome dataset.

Requires `caveclient` and `cloud-volume` as additional dependencies:

```shell
pip3 install caveclient cloud-volume -U
```

_Not imported at top level! Must be imported explicitly:_

```python
from navis.interfaces import h01
```

| Function | Description |
|----------|-------------|
| [`h01.get_cave_client()`][navis.interfaces.h01.get_cave_client] | {{ autosummary("navis.interfaces.h01.get_cave_client") }} |
| [`h01.fetch_neurons()`][navis.interfaces.h01.fetch_neurons] | {{ autosummary("navis.interfaces.h01.fetch_neurons") }} |
| [`h01.get_somas()`][navis.interfaces.h01.get_somas] | {{ autosummary("navis.interfaces.h01.get_somas") }} |

Please also see the [H01 tutorial](../generated/gallery/4_remote/tutorial_remote_04_h01/).


### R interface

Bundle of functions to use R natverse libraries.

_Not imported at top level! Must be imported explicitly:_

```python
from navis.interfaces import r
```

| Function | Description |
|----------|-------------|
| [`r.data2py()`][navis.interfaces.r.data2py] | {{ autosummary("navis.interfaces.r.data2py") }} |
| [`r.get_neuropil()`][navis.interfaces.r.data2py] | {{ autosummary("navis.interfaces.r.get_neuropil") }} |
| [`r.init_rcatmaid()`][navis.interfaces.r.data2py] | {{ autosummary("navis.interfaces.r.init_rcatmaid") }} |
| [`r.load_rda()`][navis.interfaces.r.data2py] | {{ autosummary("navis.interfaces.r.load_rda") }} |
| [`r.nblast()`][navis.interfaces.r.data2py] | {{ autosummary("navis.interfaces.r.nblast") }} |
| [`r.nblast_allbyall()`][navis.interfaces.r.data2py] | {{ autosummary("navis.interfaces.r.nblast_allbyall") }} |
| [`r.NBLASTresults()`][navis.interfaces.r.data2py] | {{ autosummary("navis.interfaces.r.NBLASTresults") }} |
| [`r.neuron2py()`][navis.interfaces.r.data2py] | {{ autosummary("navis.interfaces.r.neuron2py") }} |
| [`r.neuron2r()`][navis.interfaces.r.data2py] | {{ autosummary("navis.interfaces.r.neuron2r") }} |
| [`r.xform_brain()`][navis.interfaces.r.data2py] | {{ autosummary("navis.interfaces.r.xform_brain") }} |
| [`r.mirror_brain()`][navis.interfaces.r.data2py] | {{ autosummary("navis.interfaces.r.mirror_brain") }} |

## Neuron types and functions

As you can imagine not all functions will work on all neuron types. For example
it is currently not possible to find the longest neurite via
[`longest_neurite()`][navis.longest_neurite] for a [`VoxelNeuron`][navis.VoxelNeuron].
Conversely, some functionality like "smoothing" makes sense for multiple neuron types but the
application is so vastly different between e.g. meshes and skeletons that
there are specicialized functions for every neuron type.

Below table has an overview for which functions work with which neuron types:

|                            | TreeNeuron | MeshNeuron | VoxelNeuron | Dotprops |
|----------------------------|------------|------------|-------------|----------|
| [`navis.plot2d`][]             | yes        | yes        | limited     | yes      |
| [`navis.plot3d`][]             | yes        | yes        | see backend | yes      |
| [`navis.plot1d`][]             | yes        | no         | no          | no       |
| [`navis.plot_flat`][]          | yes        | no         | no          | no       |
| [`navis.subset_neuron`][]      | yes        | yes        | yes         | yes      |
| [`navis.in_volume`][]          | yes        | yes        | yes         | yes      |
| smoothing                  | [`navis.smooth_skeleton`][] | [`navis.smooth_mesh`][] | [`navis.smooth_voxels`][] | no |
| [`navis.downsample_neuron`][]  | yes        | yes        | yes         | yes      |
| resampling (e.g. [`navis.resample_skeleton`][]) | yes | no |       no | no        |
| [`navis.make_dotprops`][]      | yes        | yes        | yes         | yes      |
| NBLAST ([`navis.nblast`][], etc.) | no[^1]       | no[^1]         | no[^1]          | yes      |
| [`navis.xform_brain`][]        | yes        | yes        | yes (slow!) | yes      |
| [`navis.mirror_brain`][]       | yes        | yes        | no          | yes      |
| [`navis.skeletonize`][]        | no         | yes        | no          | no       |
| [`navis.mesh`][]               | yes        | no         | yes         | no       |
| [`navis.voxelize`][]           | yes        | yes        | no          | yes      |
| [`navis.drop_fluff`][]         | yes        | yes        | no          | no       |
| [`navis.break_fragments`][]    | yes        | yes        | no          | no       |

[^1]: Use [`navis.make_dotprops`][] to turn these neurons into [`navis.Dotprops`][].