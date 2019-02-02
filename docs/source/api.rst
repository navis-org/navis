.. _api:

API Reference
=============

.. _api_neurons:

TreeNeuron/List
++++++++++++++++++

.. autosummary::
    :toctree: generated/

    navis.TreeNeuron
    navis.NeuronList

TreeNeuron/List methods
--------------------------

.. autosummary::
    :toctree: generated/

    navis.TreeNeuron.plot3d
    navis.TreeNeuron.plot2d
    navis.TreeNeuron.prune_by_strahler
    navis.TreeNeuron.prune_by_volume
    navis.TreeNeuron.prune_distal_to
    navis.TreeNeuron.prune_proximal_to
    navis.TreeNeuron.prune_by_longest_neurite
    navis.TreeNeuron.reroot
    navis.TreeNeuron.summary
    navis.TreeNeuron.resample
    navis.TreeNeuron.downsample
    navis.TreeNeuron.copy

NeuronList-specific
--------------------------
.. autosummary::
    :toctree: generated/

    navis.NeuronList.sample
    navis.NeuronList.remove_duplicates
    navis.NeuronList.head
    navis.NeuronList.tail
    navis.NeuronList.itertuples
    navis.NeuronList.summary
    navis.NeuronList.mean
    navis.NeuronList.sum
    navis.NeuronList.sort_values


.. _api_plot:

Plotting
++++++++

.. autosummary::
    :toctree: generated/

    navis.plot3d
    navis.plot2d
    navis.plot1d
    navis.plot_network
    navis.clear3d
    navis.close3d
    navis.get_viewer
    navis.screenshot
    navis.Volume

Vispy 3D viewer

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

Manipulation
------------
.. autosummary::
    :toctree: generated/

    navis.cut_neuron
    navis.reroot_neuron
    navis.stitch_neurons
    navis.split_axon_dendrite
    navis.split_into_fragments
    navis.longest_neurite
    navis.prune_by_strahler
    navis.subset_neuron
    navis.average_neurons
    navis.despike_neuron
    navis.smooth_neuron
    navis.guess_radius
    navis.tortuosity

Resampling
----------
.. autosummary::
    :toctree: generated/

    navis.resample_neuron
    navis.downsample_neuron

Analysis
--------
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
.. autosummary::
    :toctree: generated/

    navis.cable_overlap
    navis.geodesic_matrix
    navis.distal_to
    navis.dist_between

Intersection
------------
.. autosummary::
    :toctree: generated/

    navis.in_volume
    navis.intersection_matrix

.. _api_con:

Connectivity
++++++++++++

Graphs
------
.. autosummary::
    :toctree: generated/

    navis.neuron2nx
    navis.neuron2igraph
    navis.neuron2KDTree
    navis.network2nx
    navis.network2igraph

Predicting connectivity
-----------------------
.. autosummary::
    :toctree: generated/

    navis.predict_connectivity

Adjacency matrices
------------------
.. autosummary::
    :toctree: generated/

    navis.group_matrix

Connectivity clustering
-----------------------
.. autosummary::
    :toctree: generated/

    navis.cluster_by_connectivity
    navis.cluster_by_synapse_placement
    navis.ClustResults


Import/Export
+++++++++++++
.. autosummary::
    :toctree: generated/

    navis.from_swc
    navis.to_swc
    navis.neuron2json
    navis.json2neuron

.. _api_b3d:

Blender API
+++++++++++

Objects
-------
.. autosummary::
    :toctree: generated/

    navis.b3d.handler.add
    navis.b3d.handler.clear
    navis.b3d.handler.select
    navis.b3d.handler.hide
    navis.b3d.handler.unhide

Materials
---------
.. autosummary::
    :toctree: generated/

    navis.b3d.handler.color
    navis.b3d.handler.colorize
    navis.b3d.handler.emit
    navis.b3d.handler.use_transparency
    navis.b3d.handler.alpha
    navis.b3d.handler.bevel

Selections
----------
.. autosummary::
    :toctree: generated/

    navis.b3d.handler.select

    navis.b3d.object_list.select
    navis.b3d.object_list.color
    navis.b3d.object_list.colorize
    navis.b3d.object_list.emit
    navis.b3d.object_list.use_transparency
    navis.b3d.object_list.alpha
    navis.b3d.object_list.bevel
    navis.b3d.object_list.hide
    navis.b3d.object_list.unhide
    navis.b3d.object_list.hide_others
    navis.b3d.object_list.delete
    navis.b3d.object_list.to_json


Cytoscape API
+++++++++++++
.. autosummary::
    :toctree: generated/

    navis.cytoscape.generate_network
    navis.cytoscape.get_client
    navis.cytoscape.watch_network

.. _api_userstats:


R interface (rMAID)
+++++++++++++++++++

.. autosummary::
    :toctree: generated/

    navis.rmaid.init_rcatmaid
    navis.rmaid.data2py
    navis.rmaid.nblast
    navis.rmaid.nblast_allbyall
    navis.rmaid.neuron2py
    navis.rmaid.dotprops2py
    navis.rmaid.neuron2r
    navis.rmaid.NBLASTresults
    navis.rmaid.get_neuropil

Utility
+++++++
.. autosummary::
    :toctree: generated/

    navis.set_pbars
    navis.set_loggers

