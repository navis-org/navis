.. _rmaid_link:

R and PyMaid
************

Python over R? R over Python? Why not R *and* Python!? Turns out, they play
together very nicely - see this brilliant 
`blog post <https://blog.jupyter.org/i-python-you-r-we-julia-baf064ca1fb6>`_.

This section will teach you the basics of how to use R and pymaid. But first,
we have to make sure you are all set:

Setting up
==========

Using R from within Python requires `rpy2 <https://rpy2.readthedocs.io>`_. 
`rpy2 <https://rpy2.readthedocs.io>`_ is **not** automatically installed
alongside pymaid. That's because it fails to install if R is not already
installed on your system. Here is what you need to do:

.. raw:: html

    <ol type="1">
      <li>
        <strong>Install R</strong>.<br>
        You can either install just <a href="https://www.r-project.org">R</a>
        or install it along with <a href="https://www.rstudio.com">R Studio</a>
        (recommended).
      </li>      
      <li>
        <strong>Install rpy2</strong><br>
        This should do the trick:
        <pre>pip3 install rpy2</pre>
        Check out rpy2's <a href="https://rpy2.github.io/doc/v2.9.x/html/index.html">documentation</a> if you are running into issues. Word of advice: don't run the most recent versions of Python/R - your best bet is Python 3.5 and R 3.3.3
      </li>
      <li>
        <strong>Install R packages</strong>.<br>
        Pymaid has wrappers for the <a href="http://jefferis.github.io/nat/">nat</a> (NeuroAnatomy Toolbox)
        ecosystem by <a href="https://github.com/jefferis">Greg Jefferis</a>.
        Please make sure to install:
          <ol type="i">
              <li>                
                <a href="http://jefferis.github.io/nat/">nat</a> - core package for morphological analysis of neurons
              </li>
              <li>
                <a href="http://jefferislab.github.io/nat.nblast/">nat.nblast</a> - morphological similarity
              </li>
              <li>
                <a href="http://jefferis.github.io/elmr/">elmr</a> - bridging between EM and light level data
              </li>
              <li>
                <a href="http://jefferis.github.io/rcatmaid/">rcatmaid</a> - interface with CATMAID
              </li>
              <li>
                <a href="http://jefferis.github.io/flycircuit/">flycircuit</a> - interface with the
                flycircuit <a href="http://www.flycircuit.tw">database</a> for fly neurons
              </li>
              <li>
                <a href="http://jefferislab.github.io/nat.templatebrains/">nat.templatebrains</a>
                and <a href="http://jefferislab.github.io/nat.flybrains/">nat.flybrains</a>
                - bridging between different light level template brains
              </li>
          </ol>
      </li>
    </ol>


You ar ready to go! On a fundamental level, you can now use every single
R function from within Python - check out the rpy2
`documentation <https://rpy2.readthedocs.io>`_. Let's explore some
more CATMAID specific examples.

Quickstart
==========

>>> import pymaid
>>> from pymaid import rmaid
>>> import matplotlib.pyplot as plt
>>> from rpy2.robjects.packages import importr

>>> # Load nat as module
>>> nat = importr('nat')

>>> # Initialize connection to Catmaid server
>>> rm = pymaid.CatmaidInstance('server_url', 'http_user', 'http_pw', 'token')

>>> # Fetch a neuron in Python CATMAID
>>> skeleton_id = 123456
>>> n = pymaid.get_neuron(skeleton_id)

>>> # Convert pymaid neuron to R neuron (works with neuron + neuronlist objects)
>>> n_R = rmaid.neuron2r(n.ix[0])

>>> # Use nat to prune the neuron
>>> n_pruned = nat.prune_by_strahler(n_R)

>>> # Convert back to pymaid object
>>> n_Py = rmaid.neuron2py(n_pruned, rm)

>>> # Nblast pruned neuron (assumes FlyCircuit database is saved locally)
>>> results = rmaid.nblast(n_pruned)

>>> # Sort results by mu score
>>> results.sort('mu_score')

>>> # Plot top 3 hits (in Jupyter notebook)
>>> import plotly.offline
>>> fig = results.plot3d(hits=3)
>>> plotly.offline.iplot(fig)

Data conversion
===============
:mod:`pymaid.rmaid` provides functions to convert data from Python to R:

1. :func:`pymaid.rmaid.data2py` converts general data from R to Python
2. :func:`pymaid.rmaid.neuron2py` converts R neuron or neuronlist objects to Python :class:`pymaid.CatmaidNeuron` and :class:`pymaid.CatmaidNeuronList`, respectively
3. :func:`pymaid.rmaid.neuron2r` converts :class:`pymaid.CatmaidNeuron` or :class:`pymaid.CatmaidNeuronList` to R neuron or neuronlist objects
4. :func:`pymaid.rmaid.dotprops2py` converts R dotprop objects to pandas DataFrame that can be passed to :func:`pymaid.plot.plot3d`

R catmaid
=========
:func:`rmaid.init_rcatmaid` is a wrapper to initialise R catmaid (https://github.com/jefferis/rcatmaid)

>>> import pymaid
>>> from pymaid import rmaid

>>> # Initialize connection to Catmaid server
>>> rm = pymaid.CatmaidInstance('server_url', 'http_user', 'http_pw', 'token')

>>> # Initialize R's rcatmaid with Python instance
>>> rcat = rmaid.init_rcatmaid(rm)

>>> # Check contents of that module
>>> dir(rcat)
['*_catmaidneuron', '+_catmaidneuron', '___NAMESPACE___', '___S3MethodsTable___', '__doc__', '__loader__', '__name__', '__package__', '__rdata__', '__rname__', '__spec__', '__version__', '_env', '_exported_names', '_packageName', '_package_statevars', '_rpy2r', '_symbol_check_after', '_symbol_r2python', '_translation', 'as_catmaidmesh', 'as_catmaidmesh_catmaidmesh',
...
'read_neuron_catmaid', 'read_neurons_catmaid', 'server', 'somapos_catmaidneuron', 'summary_catmaidneuron', 'token', 'xform_catmaidneuron']

>>> #Get neurons as R catmaidneuron
>>> n = rcat.read_neurons_catmaid('annotation:glomerulus DA1')

You can use other packages such as nat (https://github.com/jefferis/nat) to process that neuron

>>> from rpy2.robjects.packages import importr
>>> # Load nat as module
>>> nat = importr('nat')

>>> # Use nat to prune the neuron
>>> n_pruned = nat.prune_strahler(n[0])

Now convert to PyMaid :class:`pymaid.CatmaidNeuron`

>>> # Convert to Python
>>> n_py = rmaid.neuron2py(n_pruned, remote_instance=rm)

>>> # Plot
>>> n_py.plot3d()

Nblasting
=========
:func:`pymaid.rmaid.nblast` provides a wrapper to nblast neurons.

>>> from pymaid import rmaid, CatmaidInstance
>>> # Initialize connection to Catmaid server
>>> rm = CatmaidInstance('url', 'http_user', 'http_pw', 'token')

>>> # Blast a neuron against default (FlyCircuit) database
>>> skeleton_id = 16
>>> nbl = rmaid.nblast(skeleton_id, remote_instance=rm)

:func:`pymaid.rmaid.nblast` returns nblast results as instance of the :class:`pymaid.rmaid.NBLASTresults` class.

>>> # See contents of nblast_res object
>>> help(nbl)

>>> # Get results as Pandas Dataframe
>>> nbl.res

>>> # Plot histogram of results
>>> nbl.res.plot.hist(alpha=.5)

>>> # Sort and plot the first hits
>>> nbl.sort('mu_score')
>>> nbl.plot(hits=4)


Reference
=========

.. autosummary::
    :toctree: generated/

	pymaid.rmaid.init_rcatmaid
	pymaid.rmaid.data2py
	pymaid.rmaid.nblast
	pymaid.rmaid.nblast_allbyall
	pymaid.rmaid.neuron2py
	pymaid.rmaid.neuron2r
    pymaid.rmaid.NBLASTresults

