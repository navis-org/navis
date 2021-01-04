.. _api:

API Reference
=============

``navis`` has grown a lot: last I looked, there were ~100 functions exposed
at top level (e.g. ``navis.plot3d``) and easily another 100 secondary functions
available via submodules (e.g. ``navis.morpho.find_soma``).

This API reference is a more or less complete account of the primary functions:

1. :ref:`Neuron- and NeuronList functions and methods <api_neurons>`
2. :ref:`Functions for visualization<api_plot>`
3. :ref:`Manipulate or analyze neuron morphology<api_morph>`
4. :ref:`Analyze connectivity<api_con>`
5. :ref:`Import/Export<io>`
6. :ref:`Utility functions<api_utility>`

In addition ``navis`` has interfaces to various external APIs and softwares:

- :ref:`Neuromorpho<api_interfaces.neuromorpho>`
- :ref:`neuPrint<api_interfaces.neuprint>`
- :ref:`InsectBrain DB<api_interfaces.insectdb>`
- :ref:`Blender 3D<api_interfaces.blender>`
- :ref:`Cytoscape<api_interfaces.cytoscape>`
- :ref:`R and the natverse libraries<api_interfaces.r>`


Most of these functions include examples of how to use them. Click on them to
learn more!

.. _api_neurons:

Neuron/List
+++++++++++
``TreeNeurons``, ``MeshNeurons`` and ``Dotprops`` are neuron classes.
``NeuronLists`` are containers thereof.

.. autosummary::
    :toctree: generated/

    navis.TreeNeuron
    navis.MeshNeuron
    navis.Dotprops
    navis.make_dotprops
    navis.NeuronList

Neuron methods
--------------
Despite being fundamentally different data types, ``TreeNeurons``,
``MeshNeurons`` and ``Dotprops`` share some fundamental methods (i.e. functions).

.. autosummary::
    :toctree: generated/

    ~navis.TreeNeuron.copy
    ~navis.TreeNeuron.plot3d
    ~navis.TreeNeuron.plot2d
    ~navis.TreeNeuron.summary

In addition to these methods, neurons also have properties. These are
properties common to all neurons.

.. autosummary::
    :toctree: generated/

    ~navis.TreeNeuron.bbox
    ~navis.TreeNeuron.connectors
    ~navis.TreeNeuron.datatables
    ~navis.TreeNeuron.id
    ~navis.TreeNeuron.name
    ~navis.TreeNeuron.units
    ~navis.TreeNeuron.soma
    ~navis.TreeNeuron.type
    ~navis.TreeNeuron.volume


TreeNeuron-specific methods
---------------------------
These are class methods available only for ``TreeNeurons``. Most of them are simply
short-hands for the other navis functions.

.. autosummary::
    :toctree: generated/

    ~navis.TreeNeuron.convert_units
    ~navis.TreeNeuron.downsample
    ~navis.TreeNeuron.get_dps
    ~navis.TreeNeuron.get_graph_nx
    ~navis.TreeNeuron.get_igraph
    ~navis.TreeNeuron.prune_by_longest_neurite
    ~navis.TreeNeuron.prune_by_strahler
    ~navis.TreeNeuron.prune_by_volume
    ~navis.TreeNeuron.prune_distal_to
    ~navis.TreeNeuron.prune_proximal_to
    ~navis.TreeNeuron.prune_twigs
    ~navis.TreeNeuron.reload
    ~navis.TreeNeuron.reroot
    ~navis.TreeNeuron.resample

In addition ``TreeNeurons`` have a range of different attributes:

.. autosummary::
    :toctree: generated/


    ~navis.TreeNeuron.cable_length
    ~navis.TreeNeuron.created_at
    ~navis.TreeNeuron.cycles
    ~navis.TreeNeuron.downsample
    ~navis.TreeNeuron.igraph
    ~navis.TreeNeuron.is_tree
    ~navis.TreeNeuron.n_branches
    ~navis.TreeNeuron.n_leafs
    ~navis.TreeNeuron.n_skeletons
    ~navis.TreeNeuron.n_trees
    ~navis.TreeNeuron.nodes
    ~navis.TreeNeuron.postsynapses
    ~navis.TreeNeuron.presynapses
    ~navis.TreeNeuron.root
    ~navis.TreeNeuron.sampling_resolution
    ~navis.TreeNeuron.segments
    ~navis.TreeNeuron.simple
    ~navis.TreeNeuron.subtrees


MeshNeuron-specific methods
---------------------------
These are properties only for ``MeshNeurons``.

.. autosummary::
    :toctree: generated/

    ~navis.MeshNeuron.bbox
    ~navis.MeshNeuron.faces
    ~navis.MeshNeuron.trimesh
    ~navis.MeshNeuron.vertices

NeuronList methods
------------------
``NeuronLists`` let you access all the properties and methods of the neuron they
contain. In addition there are a few ``NeuronList``-specific methods and
properties.

Methods:

.. autosummary::
    :toctree: generated/

    ~navis.NeuronList.apply
    ~navis.NeuronList.head
    ~navis.NeuronList.itertuples
    ~navis.NeuronList.mean
    ~navis.NeuronList.remove_duplicates
    ~navis.NeuronList.sum
    ~navis.NeuronList.summary
    ~navis.NeuronList.tail
    ~navis.NeuronList.unmix

Properties:

.. autosummary::
    :toctree: generated/

    ~navis.NeuronList.bbox
    ~navis.NeuronList.empty
    ~navis.NeuronList.id
    ~navis.NeuronList.idx
    ~navis.NeuronList.is_degenerated
    ~navis.NeuronList.is_mixed
    ~navis.NeuronList.shape
    ~navis.NeuronList.types

.. _api_plot:

Visualization
+++++++++++++
Various functions for plotting neurons and volumes.

.. autosummary::
    :toctree: generated/

    navis.plot3d
    navis.plot2d
    navis.plot1d
    navis.clear3d
    navis.close3d
    navis.get_viewer
    navis.screenshot

Plotting Volumes/Meshes
-----------------------
To plot meshes, you can pass ``trimesh.Trimesh`` objects directly to ``plot3d``
or ``plot2d``. However, ``navis`` has a custom class to represent meshes that
has some useful perks: :class:`navis.Volume`.

.. autosummary::
    :toctree: generated/

    navis.Volume
    navis.Volume.combine
    navis.Volume.plot3d
    navis.Volume.validate

Vispy 3D viewer
---------------
Using :func:`navis.plot3d` from a terminal will spawn a Vispy 3D viewer object
which has a bunch of useful methods:

.. autosummary::
    :toctree: generated/

    navis.Viewer
    navis.Viewer.add
    navis.Viewer.clear
    navis.Viewer.close
    navis.Viewer.colorize
    navis.Viewer.set_colors
    navis.Viewer.hide_neurons
    navis.Viewer.unhide_neurons
    navis.Viewer.screenshot
    navis.Viewer.show


.. _api_morph:

Neuron Morphology
+++++++++++++++++
Collection of functions to analyze and manipulate neuronal morphology.

Manipulation
------------
Functions to edit morphology:

.. autosummary::
    :toctree: generated/

    navis.average_neurons
    navis.break_fragments
    navis.despike_neuron
    navis.cut_neuron
    navis.guess_radius
    navis.heal_fragmented_neuron
    navis.longest_neurite
    navis.prune_by_strahler
    navis.prune_twigs
    navis.reroot_neuron
    navis.split_axon_dendrite
    navis.split_into_fragments
    navis.stitch_neurons
    navis.subset_neuron
    navis.smooth_neuron
    navis.tortuosity

Resampling
----------
Functions to down- or resample neurons.

.. autosummary::
    :toctree: generated/

    navis.resample_neuron
    navis.resample_along_axis
    navis.downsample_neuron

Analysis
--------
Functions to analyze morphology.

.. autosummary::
    :toctree: generated/

    navis.classify_nodes
    navis.find_main_branchpoint
    navis.strahler_index
    navis.nblast
    navis.nblast_allbyall

Polarity metrics
----------------
.. autosummary::
    :toctree: generated/

    navis.bending_flow
    navis.flow_centrality
    navis.arbor_segregation_index
    navis.segregation_index

Distances
---------
Functions to calculate Euclidian and geodesic ("along-the-arbor") distances.

.. autosummary::
    :toctree: generated/

    navis.cable_overlap
    navis.distal_to
    navis.dist_between
    navis.geodesic_matrix
    navis.segment_length

Intersection
------------
Functions to intersect points and neurons with volumes.

.. autosummary::
    :toctree: generated/

    navis.in_volume
    navis.intersection_matrix

Transforming and Mirroring
--------------------------
Functions to transform spatial data between (e.g. neurons) template brains.
Check out the `tutorials<example_gallery>` for example on how to use them.

.. autosummary::
    :toctree: generated/

    navis.xform_brain
    navis.mirror_brain

.. _api_con:

Connectivity
++++++++++++
Collection of functions to work with graphs and adjacency matrices.

Graphs
------
Functions to convert neurons and networkx to iGraph or networkX graphs.

.. autosummary::
    :toctree: generated/

    navis.neuron2nx
    navis.neuron2igraph
    navis.neuron2KDTree
    navis.network2nx
    navis.network2igraph
    navis.rewire_neuron
    navis.insert_nodes
    navis.remove_nodes

Adjacency matrices
------------------
Functions to work with adjacency matrices.

.. autosummary::
    :toctree: generated/

    navis.group_matrix

Connectivity clustering
-----------------------
Functions to cluster neurons based on connectivity.

.. autosummary::
    :toctree: generated/

    navis.cluster_by_connectivity
    navis.cluster_by_synapse_placement
    navis.ClustResults

.. _io:

Import/Export
+++++++++++++
Functions to import/export neurons.

.. autosummary::
    :toctree: generated/

    navis.read_swc
    navis.write_swc
    navis.read_nrrd
    navis.neuron2json
    navis.json2neuron

.. _api_interfaces:

Interfaces
++++++++++
Interfaces with various external tools/websites. These modules have to be
imported explicitly as they are not imported at top level.

.. _api_interfaces.neuromorpho:

NeuroMorpho API
+++++++++++++++
Set of functions to grab data from `NeuroMorpho <http://neuromorpho.org>`_
which hosts thousands of neurons (see `tutorials<example_gallery>`).

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import neuromorpho

.. autosummary::
    :toctree: generated/

    navis.interfaces.neuromorpho.get_neuron_info
    navis.interfaces.neuromorpho.get_neuron
    navis.interfaces.neuromorpho.get_neuron_fields
    navis.interfaces.neuromorpho.get_available_field_values


.. _api_interfaces.neuprint:

nePrint API
++++++++++++
NAVis wraps `neuprint-python <https://github.com/connectome-neuprint/neuprint-python>`_
and adds a few navis-specific functions. You must have `neuprint-python`
installed for this to work::

    pip install neuprint-python

You can then import neuprint from navis like so::

    import navis.interfaces.neuprint as neu

.. autosummary::
    :toctree: generated/

    navis.interfaces.neuprint.fetch_roi
    navis.interfaces.neuprint.fetch_skeletons
    navis.interfaces.neuprint.fetch_mesh_neuron

Please also check out the :ref:`tutorials<example_gallery>` for examples of how
to fetch and work with data from neuPrint.

.. _api_interfaces.insectdb:

InsectBrain DB API
++++++++++++++++++
Set of functions to grab data from `InsectBrain <https://www.insectbraindb.org>`_
which hosts some neurons and standard brains (see `tutorials<example_gallery>`).

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import insectbrain_db

.. autosummary::
    :toctree: generated/

    navis.interfaces.insectbrain_db.get_brain_meshes
    navis.interfaces.insectbrain_db.get_species_info
    navis.interfaces.insectbrain_db.get_available_species


.. _api_interfaces.blender:

Blender API
+++++++++++
Functions to be run inside `Blender 3D <https://www.blender.org/>`_ and import
CATMAID data (see Examples). Please note that this requires Blender >2.8 as
earlier versions are shipped with Python <3.6. See the
`tutorials<example_gallery>` for an introduction of how to use ``navis`` in
Blender.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import blender

The interface is realised through a :class:`~navis.interfaces.blender.Handler`
object. It is used to import objects and facilitate working with them
programmatically once they are imported.

.. autosummary::
    :toctree: generated/

    navis.interfaces.blender.Handler

Objects
-------
.. autosummary::
    :toctree: generated/

    navis.interfaces.blender.Handler.add
    navis.interfaces.blender.Handler.clear
    navis.interfaces.blender.Handler.select
    navis.interfaces.blender.Handler.hide
    navis.interfaces.blender.Handler.unhide

Materials
---------
.. autosummary::
    :toctree: generated/

    navis.interfaces.blender.Handler.color
    navis.interfaces.blender.Handler.colorize
    navis.interfaces.blender.Handler.emit
    navis.interfaces.blender.Handler.use_transparency
    navis.interfaces.blender.Handler.alpha
    navis.interfaces.blender.Handler.bevel

Selections
----------
.. autosummary::
    :toctree: generated/

    navis.interfaces.blender.Handler.select

    navis.interfaces.blender.ObjectList.select
    navis.interfaces.blender.ObjectList.color
    navis.interfaces.blender.ObjectList.colorize
    navis.interfaces.blender.ObjectList.emit
    navis.interfaces.blender.ObjectList.use_transparency
    navis.interfaces.blender.ObjectList.alpha
    navis.interfaces.blender.ObjectList.bevel
    navis.interfaces.blender.ObjectList.hide
    navis.interfaces.blender.ObjectList.unhide
    navis.interfaces.blender.ObjectList.hide_others
    navis.interfaces.blender.ObjectList.delete
    navis.interfaces.blender.ObjectList.to_json


.. _api_interfaces.cytoscape:

Cytoscape API
+++++++++++++
Functions to use `Cytoscape <https://cytoscape.org/>`_ via the cyREST API.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import cytoscape

.. autosummary::
    :toctree: generated/

    navis.interfaces.cytoscape.generate_network
    navis.interfaces.cytoscape.get_client

.. _api_interfaces.r:

R interface
+++++++++++
Bundle of functions to use R natverse libraries.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import r

.. autosummary::
    :toctree: generated/

    navis.interfaces.r.data2py
    navis.interfaces.r.get_neuropil
    navis.interfaces.r.init_rcatmaid
    navis.interfaces.r.load_rda
    navis.interfaces.r.nblast
    navis.interfaces.r.nblast_allbyall
    navis.interfaces.r.NBLASTresults
    navis.interfaces.r.neuron2py
    navis.interfaces.r.neuron2r
    navis.interfaces.r.xform_brain
    navis.interfaces.r.mirror_brain

.. _api_utility:

Utility
+++++++
Various utility functions.

.. autosummary::
    :toctree: generated/

    navis.health_check
    navis.set_pbars
    navis.set_loggers
    navis.set_default_connector_colors
