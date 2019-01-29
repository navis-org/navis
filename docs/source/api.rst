.. _api:

API Reference
=============

.. _api_fetch:

Fetching data
+++++++++++++

Neurons
-------
.. autosummary::
    :toctree: generated/

    pymaid.get_neuron
    pymaid.delete_neuron
    pymaid.find_neurons
    pymaid.get_arbor
    pymaid.get_neurons_in_volume
    pymaid.get_neuron_list
    pymaid.get_skids_by_annotation
    pymaid.get_skids_by_name
    pymaid.rename_neurons
    pymaid.get_names

Annotations
-----------
.. autosummary::
    :toctree: generated/

    pymaid.add_annotations
    pymaid.add_meta_annotations
    pymaid.get_annotations    
    pymaid.get_annotation_details
    pymaid.get_annotated
    pymaid.get_user_annotations
    pymaid.remove_annotations
    pymaid.remove_meta_annotations

Treenodes
----------
.. autosummary::
    :toctree: generated/

    pymaid.get_treenode_table
    pymaid.get_treenode_info
    pymaid.get_treenodes_by_tag
    pymaid.get_skid_from_treenode
    pymaid.get_node_details
    pymaid.get_node_location

Tags
----
.. autosummary::
    :toctree: generated/

    pymaid.get_label_list
    pymaid.add_tags
    pymaid.delete_tags
    pymaid.get_node_tags

Connectivity
------------
.. autosummary::
    :toctree: generated/

    pymaid.get_connectors
    pymaid.get_connector_details
    pymaid.get_connectors_between
    pymaid.get_connector_links
    pymaid.get_edges
    pymaid.get_partners
    pymaid.get_partners_in_volume
    pymaid.get_nth_partners
    pymaid.get_paths
    pymaid.adjacency_from_connectors
    pymaid.cn_table_from_connectors

User stats
----------
.. autosummary::
    :toctree: generated/

    pymaid.get_user_list
    pymaid.get_history
    pymaid.get_time_invested
    pymaid.get_user_contributions
    pymaid.get_contributor_statistics
    pymaid.get_logs
    pymaid.get_transactions
    pymaid.get_team_contributions

Volumes
-------
.. autosummary::
    :toctree: generated/

    pymaid.get_volume

.. _api_misc:

Misc
----
.. autosummary::
    :toctree: generated/

    pymaid.url_to_coordinates
    pymaid.get_review
    pymaid.get_review_details
    pymaid.clear_cache
    pymaid.has_soma


CatmaidInstance
+++++++++++++++

.. autosummary::
    :toctree: generated/

    pymaid.CatmaidInstance
    pymaid.CatmaidInstance.fetch
    pymaid.CatmaidInstance.make_url
    pymaid.CatmaidInstance.setup_cache
    pymaid.CatmaidInstance.clear_cache
    pymaid.CatmaidInstance.load_cache
    pymaid.CatmaidInstance.save_cache
    pymaid.CatmaidInstance.copy
    pymaid.CatmaidInstance.make_url

.. _api_neurons:

CatmaidNeuron/List
++++++++++++++++++

.. autosummary::
    :toctree: generated/

    pymaid.CatmaidNeuron
    pymaid.CatmaidNeuronList

CatmaidNeuron/List methods
--------------------------

.. autosummary::
    :toctree: generated/

    pymaid.CatmaidNeuron.plot3d
    pymaid.CatmaidNeuron.plot2d
    pymaid.CatmaidNeuron.plot_dendrogram
    pymaid.CatmaidNeuron.prune_by_strahler
    pymaid.CatmaidNeuron.prune_by_volume
    pymaid.CatmaidNeuron.prune_distal_to
    pymaid.CatmaidNeuron.prune_proximal_to
    pymaid.CatmaidNeuron.prune_by_longest_neurite
    pymaid.CatmaidNeuron.reroot
    pymaid.CatmaidNeuron.reload
    pymaid.CatmaidNeuron.summary
    pymaid.CatmaidNeuron.resample
    pymaid.CatmaidNeuron.downsample
    pymaid.CatmaidNeuron.copy
    pymaid.CatmaidNeuron.from_swc
    pymaid.CatmaidNeuron.to_swc

CatmaidNeuronList-specific
--------------------------
.. autosummary::
    :toctree: generated/

    pymaid.CatmaidNeuronList.to_selection
    pymaid.CatmaidNeuronList.from_selection
    pymaid.CatmaidNeuronList.has_annotation
    pymaid.CatmaidNeuronList.sample
    pymaid.CatmaidNeuronList.remove_duplicates
    pymaid.CatmaidNeuronList.head
    pymaid.CatmaidNeuronList.tail
    pymaid.CatmaidNeuronList.itertuples
    pymaid.CatmaidNeuronList.summary
    pymaid.CatmaidNeuronList.mean
    pymaid.CatmaidNeuronList.sum
    pymaid.CatmaidNeuronList.sort_values


.. _api_plot:

Plotting
++++++++

.. autosummary::
    :toctree: generated/

    pymaid.plot3d
    pymaid.plot2d
    pymaid.plot1d
    pymaid.plot_network
    pymaid.clear3d
    pymaid.close3d
    pymaid.get_viewer
    pymaid.screenshot
    pymaid.Volume

Vispy 3D viewer

.. autosummary::
    :toctree: generated/

    pymaid.Viewer
    pymaid.Viewer.add
    pymaid.Viewer.clear
    pymaid.Viewer.close
    pymaid.Viewer.colorize
    pymaid.Viewer.set_colors
    pymaid.Viewer.hide_neurons
    pymaid.Viewer.unhide_neurons
    pymaid.Viewer.screenshot
    pymaid.Viewer.show


.. _api_morph:

Neuron Morphology
+++++++++++++++++

Manipulation
------------
.. autosummary::
    :toctree: generated/

    pymaid.cut_neuron
    pymaid.reroot_neuron
    pymaid.stitch_neurons
    pymaid.split_axon_dendrite
    pymaid.split_into_fragments
    pymaid.longest_neurite
    pymaid.prune_by_strahler
    pymaid.subset_neuron
    pymaid.average_neurons
    pymaid.remove_tagged_branches
    pymaid.despike_neuron
    pymaid.smooth_neuron
    pymaid.guess_radius
    pymaid.time_machine
    pymaid.tortuosity

Resampling
----------
.. autosummary::
    :toctree: generated/

    pymaid.resample_neuron
    pymaid.downsample_neuron

Analysis
--------
.. autosummary::
    :toctree: generated/

    pymaid.arbor_confidence
    pymaid.bending_flow
    pymaid.calc_cable
    pymaid.classify_nodes
    pymaid.find_main_branchpoint
    pymaid.flow_centrality
    pymaid.segregation_index
    pymaid.strahler_index

Distances
---------
.. autosummary::
    :toctree: generated/

    pymaid.cable_overlap
    pymaid.geodesic_matrix
    pymaid.distal_to
    pymaid.dist_between

Intersection
------------
.. autosummary::
    :toctree: generated/

    pymaid.in_volume
    pymaid.intersection_matrix

.. _api_con:

Connectivity
++++++++++++

Graphs
------
.. autosummary::
    :toctree: generated/

    pymaid.neuron2nx
    pymaid.neuron2igraph
    pymaid.neuron2KDTree
    pymaid.network2nx
    pymaid.network2igraph

Predicting connectivity
-----------------------
.. autosummary::
    :toctree: generated/

    pymaid.predict_connectivity
    pymaid.connection_density

Adjacency matrices
------------------
.. autosummary::
    :toctree: generated/

    pymaid.adjacency_matrix
    pymaid.group_matrix

Connectivity clustering
-----------------------
.. autosummary::
    :toctree: generated/

    pymaid.cluster_by_connectivity
    pymaid.cluster_by_synapse_placement
    pymaid.ClustResults

Plotting network
----------------
.. autosummary::
    :toctree: generated/

    pymaid.plot_network

Filtering
---------
.. autosummary::
    :toctree: generated/

    pymaid.filter_connectivity

Import/Export
+++++++++++++
.. autosummary::
    :toctree: generated/

    pymaid.from_swc
    pymaid.to_swc
    pymaid.neuron2json
    pymaid.json2neuron

.. _api_b3d:

Blender API
+++++++++++

Objects
-------
.. autosummary::
    :toctree: generated/

    pymaid.b3d.handler.add
    pymaid.b3d.handler.clear
    pymaid.b3d.handler.select
    pymaid.b3d.handler.hide
    pymaid.b3d.handler.unhide

Materials
---------
.. autosummary::
    :toctree: generated/

    pymaid.b3d.handler.color
    pymaid.b3d.handler.colorize
    pymaid.b3d.handler.emit
    pymaid.b3d.handler.use_transparency
    pymaid.b3d.handler.alpha
    pymaid.b3d.handler.bevel

Selections
----------
.. autosummary::
    :toctree: generated/

    pymaid.b3d.handler.select

    pymaid.b3d.object_list.select
    pymaid.b3d.object_list.color
    pymaid.b3d.object_list.colorize
    pymaid.b3d.object_list.emit
    pymaid.b3d.object_list.use_transparency
    pymaid.b3d.object_list.alpha
    pymaid.b3d.object_list.bevel
    pymaid.b3d.object_list.hide
    pymaid.b3d.object_list.unhide
    pymaid.b3d.object_list.hide_others
    pymaid.b3d.object_list.delete
    pymaid.b3d.object_list.to_json


Cytoscape API
+++++++++++++
.. autosummary::
    :toctree: generated/

    pymaid.cytoscape.generate_network
    pymaid.cytoscape.get_client
    pymaid.cytoscape.watch_network

.. _api_userstats:

User statistics
+++++++++++++++

.. autosummary::
    :toctree: generated/

    pymaid.get_user_contributions
    pymaid.get_time_invested
    pymaid.get_history
    pymaid.get_logs
    pymaid.get_contributor_statistics
    pymaid.get_user_list
    pymaid.get_user_actions
    pymaid.get_user_stats
    pymaid.get_transactions


Image data (tiles)
++++++++++++++++++

.. autosummary::
    :toctree: generated/

    pymaid.tiles.LoadTiles
    pymaid.tiles.crop_neuron


R interface (rMAID)
+++++++++++++++++++

.. autosummary::
    :toctree: generated/

    pymaid.rmaid.init_rcatmaid
    pymaid.rmaid.data2py
    pymaid.rmaid.nblast
    pymaid.rmaid.nblast_allbyall
    pymaid.rmaid.neuron2py
    pymaid.rmaid.dotprops2py
    pymaid.rmaid.neuron2r
    pymaid.rmaid.NBLASTresults
    pymaid.rmaid.get_neuropil

Utility
+++++++
.. autosummary::
    :toctree: generated/

    pymaid.set_pbars
    pymaid.set_loggers
    pymaid.eval_skids
    pymaid.shorten_name

