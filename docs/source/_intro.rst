.. _example:

Introduction
************
This section will teach you the basics of how to use PyMaid. If you are impatient check out the *Quickstart Guide* but I recommend having a look at the *Basics* too.

Quickstart Guide
================
At the beginning of each session, you have to initialise a :class:`~pymaid.CatmaidInstance` which holds the url and your credentials for the CATMAID server. In most examples, this instance is assigned to a variable called ``remote_instance`` or just ``rm``. Here we are requesting a list of two neurons from the server:

>>> import pymaid
>>> # HTTP_USER AND HTTP_PASSWORD are only necessary if your server requires a 
... #http authentification
>>> rm = pymaid.CatmaidInstance( 'www.your.catmaid-server.org' , 
...                              'HTTP_USER' , 
...                              'HTTP_PASSWORD', 
...                              'TOKEN' )
>>> neuron_list = pymaid.get_neuron ( ['12345','67890'] )
>>> # To access individual neurons, use neuron_list like a normal list object
>>> neuron_list[0]
type              <class 'pymaid.CatmaidNeuron'>
neuron_name                PN glomerulus DA1 27296 BH
skeleton_id                                     27295
n_nodes                                          9924
n_connectors                                      437
n_branch_nodes                                    207
n_end_nodes                                       214
cable_length                                  1479.81
review_status                                      NA
annotations                                     False
igraph                                          False
tags                                             True
dtype: object
>>> # Note how some entries are "False" or "NA"? These are still empty. 
>>> # They will be retrieved/computed on-demand upon first *explicit* request
>>> neuron_list[0].review_status
57

Here, ``neuron_list`` is an instance of the :class:`~pymaid.CatmaidNeuronList` class and holds two neurons, both of which are of the :class:`~pymaid.CatmaidNeuron` class. Check out their documentation for methods and attributes!

Plotting is easy and straight forward:

>>> neuron_list.plot3d()

This method simply calls :func:`pymaid.plot3d` - check out the docs for which parameters you can pass along.

The Basics
==========
Neuron data is (in most cases) stored as :class:`~pymaid.CatmaidNeuron`. Multiple :class:`~pymaid.CatmaidNeuron` are grouped into :class:`~pymaid.CatmaidNeuronList`. 

You can minimally create an neuron object with just its skeleton ID:

>>> import pymaid
>>> neuron = pymaid.CatmaidNeuron( 123456 )

Attributes of this neuron will be retrieved from the server on-demand. For this, you will have to assign a :class:`~pymaid.CatmaidInstance` to that neuron:

>>> neuron.set_remote_instance( server_url = 'url', http_user = 'user', http_pw = 'pw', auth_token = 'token' ) 
>>> # Retrieve the name of the neuron on-demand
>>> neuron.neuron_name
>>> # You can also just pass an existing instance 
>>> neuron = pymaid.CatmaidNeuron( 123456, remote_instance = rm )

Some functions already return partially completed neurons (e.g. :func:`~pymaid.get_neuron`)

>>> rm = pymaid.CatmaidInstance( 'server_url', 'http_user', 'http_pw', 'auth_token' )
>>> neuron = pymaid.get_neuron( 123456, remote_instance = rm )

All functions that explicitly require you to pass a ``skids`` parameter (e.g. :func:`~pymaid.get_neuron`) accept either:

1. skeleton IDs (int or str)
2. neuron name (str, exact match)
3. annotation: e.g. ``'annotation:PN right'``
4. CatmaidNeuron or CatmaidNeuronList object

Some examples:

>>> import pymaid
>>> rm = pymaid.CatmaidInstance( 'www.your.catmaid-server.org' , 
...                              'HTTP_USER' , 
...                              'HTTP_PASSWORD', 
...                              'TOKEN' )
>>> # Create neuron list from annotation
>>> neuron_list = pymaid.get_neuron( 'annotation:glomerulus DA1' )
>>> # Get partners of these neurons
>>> partners = pymaid.get_partners( neuron_list )
>>> # Use a neuron name when adding an annotation
>>> pymaid.add_annotation( ['neuron1_name','neuron_name2'], ['annotation1','annotation2'] )

Advanced Stuff
==============

Connection to the server: CatmaidInstance 
-----------------------------------------
As you instanciate :class:`~pymaid.CatmaidInstance`, it is made the default, "global" remote instance and you don't need to worry about it anymore.

>>> import pymaid
>>> rm = pymaid.CatmaidInstance( 'server_url', 'http_user', 'http_pw', 'auth_token')
2017-08-24 19:31:22,663 - pymaid - INFO - Global CATMAID instance set.
>>> partners = pymaid.get_partners( [12345,67890] )

However, if you - for example - are working with two separate servers or projects, you can pass a :class:`~pymaid.CatmaidInstance` explicitly to a function. This will overule any globally defined remote instance:

>>> import pymaid
>>> rm2 = pymaid.CatmaidInstance( 'server_url2', 'http_user2', 'http_pw2', 'auth_token2', set_global=False )
>>> partners = pymaid.get_partners( [12345,67890], remote_instance = rm2 )

You can manually make a remote instance the global one:

>>> rm2.set_global()

The project ID is part of the CatmaidInstance and defaults to 1. You can change this either when initializing or later on-the-fly:

>>> # Initialise with project_id 2 (default = 1)
>>> rm = pymaid.CatmaidInstance( 'server_url', 'http_user', 'http_pw', 'auth_token', project_id = 2 )
>>> # Change project_id on-the-fly
>>> rm.project_id = 1

:class:`~pymaid.CatmaidNeuron` and :class:`~pymaid.CatmaidNeuronList` objects will store a CatmaidInstance and use it to pull data from the server on-demand:

>>> import pymaid
>>> rm = pymaid.CatmaidInstance( 'www.your.catmaid-server.org' , 
...                              'HTTP_USER' , 
...                              'HTTP_PASSWORD', 
...                              'TOKEN' )
>>> # Initialise explicitely with a CatmaidInstance
>>> nl = pymaid.CatmaidNeuronList( [12345,67890], remote_instance = rm )
>>> # Initialise without and add later
>>> nl = pymaid.CatmaidNeuronList( [12345,67890] )
>>> nl.set_remote_instance(rm)
>>> # Alternatively
>>> nl.set_remote_instance( server_url = 'www.your.catmaid-server.org', 
...                         http_user = 'HTTP_USER', 
...                         http_pw = 'HTTP_PASSWORD', 
...                         auth_token = 'TOKEN' ) 


CatmaidNeuron and CatmaidNeuronList objects
-------------------------------------------

Accessing data
++++++++++++++

As laid out in the Quickstart, :class:`~pymaid.CatmaidNeuron` can be initialised with just a skeleton ID and the rest will then be requested/calculated on-demand:

>>> import pymaid
>>> # Initialize a new neuron
>>> n = pymaid.CatmaidNeuron( 123456 ) 
>>> # Initialize Catmaid connections
>>> rm = pymaid.CatmaidInstance(server_url, http_user, http_pw, token) 
>>> # Add CatmaidInstance to the neuron for convenience    
>>> n.set_remote_instance(rm) 

To access any of the data stored in a CatmaidNeuron simply use:

>>> # Retrieve node data from server on-demand
>>> n.nodes 
CatmaidNeuron - INFO - Retrieving skeleton data...
    treenode_id  parent_id  creator_id  x  y  z radius confidence
0   ...

You might have noticed that nodes are stored as pandas.DataFrame. That allows some fancy indexing and processing!

>>> # Get all nodes with radius larger than -1
>>> n.nodes[ n.nodes.radius > 1 ]

Other data, such as annotations are stored as simple lists.

>>> n.annotations
[ 'annotation1', 'annotation2' ]

All this data is loaded once upon the first explicit request and then stored in the CatmaidNeuron object. You can force updates by using the ``get`` functions:

>>> n.get_annotations()
>>> n.annotations
[ 'annotation1', 'annotation2', 'new_annotation' ]

Attributes in :class:`~pymaid.CatmaidNeuronList` work much the same way but instead you will get that data for all neurons that are within that neuron list.

>>> nl = pymaid.CatmaidNeuronList( [ 123456, 456789, 123455 ], remote_instance = rm ) 
>>> nl.skeleton_id
[ 123456, 456789, 123455 ]
>>> nl.review_status
[ 10, 99, 12 ]

Indexing CatmaidNeuronLists
+++++++++++++++++++++++++++

:class:`~pymaid.CatmaidNeuron` is much like pandas DataFrames in that it allows some fancing indexing

>>> # Initialize with just a Skeleton ID 
>>> nl = pymaid.CatmaidNeuronList( [ 123456, 45677 ] )
>>> # Add CatmaidInstance to neurons in neuronlist
>>> rm = pymaid.CatmaidInstance(server_url, http_user, http_pw, token)
>>> nl.set_remote_instance( rm )
>>> # Index using node count
>>> subset = nl [ nl.n_nodes > 6000 ]
>>> # Index by skeleton ID 
>>> subset = nl.skid [ '123456' ]
>>> # Index by neuron name
>>> subset = nl [ 'name1' ]
>>> # Index by list of skeleton IDs
>>> subset = nl.skid [ [ '12345', '67890' ] ]
>>> # Index by annotation
>>> subset = nl.has_annotation( ['AN1', 'AN2'], intersect=False )
>>> # Concatenate lists
>>> nl += pymaid.get_neuron( [ 912345 ], remote_instance = rm )
>>> # Remove item(s)
>>> subset = nl - [ 45677 ]