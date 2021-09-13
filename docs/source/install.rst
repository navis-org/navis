.. _installing:

Install
=======

Installation instructions come in two flavours:

1. **Quick install**: if you know what you are doing
2. **Step-by-step instructions** : if you are new to Python

.. topic:: By the way

   You can try navis without having to install **anything**! Simply follow this
   link to `Binder <https://mybinder.org/v2/gh/navis-org/navis/master?urlpath=tree>`_:
   they are kindly hosting a Jupyter notebook server with the most up-to-date version
   of navis. Just navigate and open ``examples/start_here.ipynb`` to have
   a crack at it!


Quick install
-------------

Requires Python 3.6 or later.

If you don't already have it, get the Python package manager `PIP <https://pip.pypa.io/en/stable/installing/>`_.

To get the minimal most recent version from PyPI (see below for optional extras) use:

::

   pip3 install navis -U

To get the most recent development version from
`Github <https://github.com/navis-org/navis>`_ use:

::

   pip3 install git+git://github.com/navis-org/navis@master


**Installing from source**

Instead of using PIP to install from Github, you can also install manually:

1. Download the source (e.g a ``tar.gz`` file from
   https://github.com/navis-org/navis/tree/master/dist)

2. Unpack and change directory to the source directory
   (the one with ``setup.py``).

3. Run ``python setup.py install`` to build and install


Step-by-step instructions
-------------------------

.. raw:: html

    <ol type="1">
      <li><strong>Check if Python 3 is installed and install if
                  necessary</strong>.<br> Linux and Mac should already come
                  with Python distribution(s) but you need to figure out if
                  you have Python 2, Python 3 or both:
                  <br>
                  Open a terminal, type in
                  <pre>python3 --version</pre>
                  and press enter. You should get something similar to either of
                  this:
      </li>
          <ol type="a">
              <li>
                  <pre>python3: command not found</pre>
                  No Python 3 installed. See below box on how to install Python 3.
              </li>
              <li>
                  <pre>Python 3.7.4 :: Anaconda, Inc.</pre>
                  Python 3 is already installed. Nice! Proceed with step 2.
              </li>
          </ol>
      </li>
      <li>
        <strong>Get the Python package manager <a href="https://pip.pypa.io">PIP</a>.</strong><br>
        Try running this in a terminal:
        <pre>pip3 install --upgrade pip</pre>
        If you already have PIP, this should update it to the most recent version.
        If you get: <pre>pip3: command not found</pre> follow this
        <a href="https://pip.pypa.io/en/stable/installing/">link</a> to download
        and install PIP.
      </li>
      <li>
        <strong>Install navis and its dependencies</strong>.<br>
        Open a terminal and run:
        <pre>pip3 install navis -U</pre>
        to install the most recent version of navis and all of its
        <em>mandatory</em> dependencies. <strong>You can also use this command
        to update an existing install of navis!</strong>
      </li>
      <li>
        <strong>Done!</strong> Go to <em>Tutorial</em> to get started.
      </li>
    </ol>

.. topic:: Installing Python 3

   On **Linux** and **OSX (Mac)**, simply go to https://www.python.org to
   download + install Python3 (version 3.7 or later).

   On **Windows** things are bit more tricky. While NAVis is written in pure
   Python, some of its dependencies don't have pre-compiled binaries for Windows
   and hence need to be compiled - which a pain on Windows. You have two
   options:

   1. Install `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/>`_
      which runs a whole Linux inside your Windows. From my (admittedly limited)
      experience this seems to work very well.
   2. Use a scientific Python distribution that comes with "batteries included".
      `Anaconda <https://www.continuum.io/downloads>`_ is a widespread solution
      that comes with its own package manager ``conda`` which often has
      precompiled Windows binaries where ``pip`` doesn't.


Optional Dependencies
---------------------

If you installed navis using ``pip``, mandatory dependencies should have been
installed automatically. There are a few optional dependencies that e.g. provide
speed-boosts in certain situations or are required only in certain functions.

These extras can be installed directly, or along with navis with

::

   pip3 install navis[extra1,extra2]


The user-facing extras, the dependencies they install,
and how to install those dependencies directly, are below.
You can install all of them with the ``all`` extra.


.. _pykd:

``kdtree``: `pykdtree <https://github.com/storpipfugl/pykdtree>`_
  Faster than scipy's cKDTree implementation. If available, will be used to
  speed up e.g. NBLAST. **Important**: on Linux I found that I need to set
  a ``OMP_NUM_THREADS=4`` environment variable (see also ``pykdtree`` docs).
  Otherwise it's actually slower than scipy's KDTree.

  ::

    pip3 install pykdtree

.. _pyoc:

``octree``: `PyOctree <https://pypi.python.org/pypi/pyoctree/>`_
  Slower alternative to ncollpyde.

  ::

    pip3 install pyoctree

.. _rpy:

``r``: `Rpy2 <https://rpy2.readthedocs.io/en/version_2.8.x/overview.html#installation>`_ (``rpy2``)
  Provides interface with R. This allows you to use e.g. the
  `natverse <https://natverse.org>`_  R packages. Note that
  this package is not installed automatically as it would fail
  if R is not already installed on the system. You have to
  install Rpy2 manually!

  ::

    pip3 install rpy2

.. _shapely:

``shapely``: `Shapely <https://shapely.readthedocs.io/en/latest/>`_ (``shapely``)
  This is used to get 2D outlines of navis.Volumes.

  ::

    pip3 install shapely

.. _igraph:

``igraph``: `iGraph <http://igraph.org/>`_
  For advanced users.

  By default navis uses the `NetworkX <https://networkx.github.io>`_ graph
  library for most of the computationally expensive functions. NetworkX is
  written in pure Python, well maintained and easy to install.

  If you need that extra bit of speed, there is iGraph.
  It is written in C and therefore very fast.
  If available, navis will try using iGraph over NetworkX.
