.. _api:

API Reference
=============

Unless otherwise stated below functions are available at top level import,
i.e. after ``import navis``.

.. _api_neurons:

TreeNeuron/List
+++++++++++++++
TreeNeurons and NeuronLists are the classes representing neurons and
lists thereof respectively.

.. autosummary::
    :toctree: generated/

    ~navis.TreeNeuron
    ~navis.NeuronList

TreeNeuron/List methods
-----------------------
These are class methods available for both TreeNeurons and NeuronLists. They
are simply short-hands for the other navis functions.

.. autosummary::
    :toctree: generated/

    ~navis.TreeNeuron.downsample
    ~navis.TreeNeuron.copy
    ~navis.TreeNeuron.plot3d
    ~navis.TreeNeuron.plot2d
    ~navis.TreeNeuron.prune_by_strahler
    ~navis.TreeNeuron.prune_by_volume
    ~navis.TreeNeuron.prune_distal_to
    ~navis.TreeNeuron.prune_proximal_to
    ~navis.TreeNeuron.prune_by_longest_neurite
    ~navis.TreeNeuron.prune_twigs
    ~navis.TreeNeuron.reload
    ~navis.TreeNeuron.reroot
    ~navis.TreeNeuron.resample
    ~navis.TreeNeuron.summary

NeuronList-specific
-------------------
These functions are specific to NeuronLists.

.. autosummary::
    :toctree: generated/

    navis.NeuronList.apply
    navis.NeuronList.head
    navis.NeuronList.itertuples
    navis.NeuronList.mean
    navis.NeuronList.remove_duplicates
    navis.NeuronList.sample
    navis.NeuronList.summary
    navis.NeuronList.sum
    navis.NeuronList.sort_values
    navis.NeuronList.tail


.. _api_plot:

Plotting
++++++++
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
    navis.Volume

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
Collection of functions to analyse and manipulate neuronal morphology.

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
    navis.downsample_neuron

Analysis
--------
Functions to analyse morphology.

.. autosummary::
    :toctree: generated/

    navis.bending_flow
    navis.classify_nodes
    navis.find_main_branchpoint
    navis.flow_centrality
    navis.segregation_index
    navis.strahler_index

Distances
---------
Functions to calculate eucledian and geodesic ("along-the-arbor") distances.

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


Import/Export
+++++++++++++
Functions to import/export neurons.

.. autosummary::
    :toctree: generated/

    navis.from_swc
    navis.to_swc
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
which hosts thousands of neurons (see Tutorials).

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import neuromorpho

.. autosummary::
    :toctree: generated/

    navis.interfaces.neuromorpho.get_neuron_info
    navis.interfaces.neuromorpho.get_neuron
    navis.interfaces.neuromorpho.get_neuron_fields
    navis.interfaces.neuromorpho.get_available_field_values


.. _api_interfaces.neuprint:

neuprint API
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


.. _api_interfaces.neuromorpho:

InsectBrain DB API
++++++++++++++++++
Set of functions to grab data from `InsectBrain <https://www.insectbraindb.org>`_
which hosts some neurons and standard brains (see Tutorials).

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
earlier versions are shipped with Python <3.6.

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


Cytoscape API
+++++++++++++
Functions to use `Cytoscape <https://cytoscape.org/>`_ via the cyREST API.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import cytoscape

.. autosummary::
    :toctree: generated/

    navis.interfaces.cytoscape.generate_network
    navis.interfaces.cytoscape.get_client

R interface
+++++++++++
Bundle of functions to use R libraries.

Not imported at top level! Must be imported explicitly::

    from navis.interfaces import r

.. autosummary::
    :toctree: generated/

    navis.interfaces.r.data2py
    navis.interfaces.r.dotprops2py
    navis.interfaces.r.get_neuropil
    navis.interfaces.r.init_rcatmaid
    navis.interfaces.r.nblast
    navis.interfaces.r.nblast_allbyall
    navis.interfaces.r.NBLASTresults
    navis.interfaces.r.neuron2py
    navis.interfaces.r.neuron2r
    navis.interfaces.r.xform_brain
    navis.interfaces.r.mirror_brain


Utility
+++++++
Various utility functions.

.. autosummary::
    :toctree: generated/

    navis.health_check
    navis.set_pbars
    navis.set_loggers
    navis.set_default_connector_colors
