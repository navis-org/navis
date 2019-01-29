.. _installing:

Install
=======

Installation instructions come in two flavors:

1. **Quick install**: if you know what you are doing
2. **Step-by-step instructions** : if you are new to Python

.. topic:: By the way

   You can try pymaid without having to install **anything**! Simply follow this
   link to `Binder <https://mybinder.org/v2/gh/schlegelp/pyMaid/master?urlpath=tree>`_:
   they are kindly hosting a Jupyter notebook server with the most up-to-date version
   of pymaid. Just navigate and open ``examples/start_here.ipynb`` to have
   a crack at it!


Quick install
-------------

If you don't already have it, get the Python package manager `PIP <https://pip.pypa.io/en/stable/installing/>`_.

Pymaid is **NOT** listed in the Python Packaging Index (PyPI). There is a
`pymaid` package on PyPI but that is something else! Hence, you will have to
install from `Github <https://github.com/schlegelp/PyMaid>`_. To get the
most recent version use:

::

   pip3 install git+git://github.com/schlegelp/pymaid@master


**Installing from source**

Instead of using PIP to install from Github, you can also install manually:

1. Download the source (e.g a ``tar.gz`` file from
   https://github.com/schlegelp/PyMaid/tree/master/dist)

2. Unpack and change directory to the source directory
   (the one with ``setup.py``).

3. Run ``python setup.py install`` to build and install

.. note::
   There are two optional dependencies that you might want to install manually:
   :ref:`pyoctree <pyoc>` and :ref:`rpy2 <rpy>`. The latter is only relevant if
   you intend to use pymaid's R wrappers.


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
                  <pre>Python 3.6.4 :: Anaconda, Inc.</pre>
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
        <strong>Make sure <a href="https://git-scm.com/">git</a> is installed.</strong><br>
        In 99% of the cases you should be fine but just to make sure, try running
        this in a terminal:
        <pre>git --version</pre>
        If you get: <pre>git: command not found</pre> follow this
        <a href="https://git-scm.com/">link</a> for instructions how to install
        git on your OS.
      </li>
      <li>
        <strong>Install pymaid and its dependencies</strong>.<br>
        Open a terminal and run:
        <pre>pip3 install git+git://github.com/schlegelp/pymaid@master</pre>
        to install the most recent version of pymaid and all of its
        <em>mandatory</em> dependencies. <strong>You can also use this command
        to update an existing install of pymaid!</strong>
      </li>
      <li>
        <strong>Done!</strong> Go to <em>Tutorial</em> to get started.
      </li>
    </ol>

.. raw:: html

    <div class="alert alert-danger alert-trim" role="alert">
      Missing permissions to write can mess up
      installations using <strong>PIP</strong>. If you get a
      <code>"..permission denied.."</code> error, try running the same command
      as admin: <code>sudo pip3 install ...</code>
    </div>

.. topic:: Installing Python 3

   On **Linux** and **OSX (Mac)**, simply go to https://www.python.org to
   download + install Python3. I recommend getting Python 3.5 or 3.6 as newer
   versions may still have compatibility problems with some of pymaid's
   dependencies.

   On **Windows**, things are bit more tricky. While pymaid is written in pure
   Python, some of its dependencies are written in C for speed and need to be
   compiled - which a pain on Windows. I strongly recommend installing a
   scientific Python distribution that comes with "batteries included".
   `Anaconda <https://www.continuum.io/downloads>`_ is a widespread solution
   that comes with its own package manager ``conda``.

.. note::
   There are two optional dependencies that you might want to install manually:
   :ref:`pyoctree <pyoc>` and :ref:`rpy2 <rpy>`. The latter is only relevant if
   you intend to use pymaid's R bindings.


Dependencies
------------

Mandatory
+++++++++

If you installed pymaid using ``PIP``, mandatory dependencies should have been
installed automatically.

`NumPy <http://www.numpy.org/>`_
  Provides matrix representation of graphs and is used in some graph
  algorithms for high-performance matrix computations.

`Pandas <http://pandas.pydata.org/>`_
  Provides advanced dataframes and indexing.

`Vispy <http://vispy.org/>`_
  Used to visualise neurons in 3D. This requires you to have *one* of
  the supported `backends <http://vispy.org/installation.html#backend-requirements>`_
  installed. During automatic installation pymaid will try installing the
  `PyQt5 <http://pyqt.sourceforge.net/Docs/PyQt5/installation.html>`_ backend
  to fullfil this requirement.

`Plotly <https://plot.ly/python/getting-started/>`_
  Used to visualise neurons in 3D. Alternative to Vispy based on WebGL.

`NetworkX <https://networkx.github.io>`_
  Graph analysis library written in pure Python. This is the standard library
  used by pymaid.

`SciPy <http://scipy.org>`_
  Provides tons of scientific computing tools: sparse matrix representation
  of graphs, pairwose distance computation, hierarchical clustering, etc.

`Matplotlib <http://matplotlib.sourceforge.net/>`_
  Essential for all 2D plotting.

`Seaborn <https://seaborn.pydata.org>`_
  Used e.g. for its color palettes.

`tqdm <https://pypi.python.org/pypi/tqdm>`_
  Neat progress bars.

`PyPNG <https://pythonhosted.org/pypng/>`_
  Generates PNG images. Used for taking screenshot from 3D viewer. Install
  from PIP: ``pip3 install pypng``.


Optional
++++++++

.. _pyoc:

`PyOctree <https://pypi.python.org/pypi/pyoctree/>`_ (highly recommended)
  Provides octrees from meshes to perform ray casting. Used to check e.g. if
  objects are within volume.

  ::

    pip3 install pyoctree

.. _rpy:

`Rpy2 <https://rpy2.readthedocs.io/en/version_2.8.x/overview.html#installation>`_
  Provides interface with R. This allows you to use e.g. R packages from
  https://github.com/jefferis and https://github.com/alexanderbates. Note that
  this package is not installed automatically as it would fail if R is not
  already installed on the system. You have to install Rpy2 manually!

  ::

    pip3 install rpy2

`Shapely <https://shapely.readthedocs.io/en/latest/>`_
  This is used to get 2D outlines of CATMAID volumes.

  ::

    pip3 install shapely


Advanced users: more speed with iGraph
--------------------------------------

By default pymaid uses the `NetworkX <https://networkx.github.io>`_ graph
library for most of the computationally expensive function. NetworkX is
written in pure Python, well maintained and easy to install.

If you need that extra bit of speed, consider manually installing
`iGraph <http://igraph.org/>`_. It is written in C and therefore very fast. If
available, pymaid will try using iGraph over NetworkX. iGraph is difficult to
install though because you have to install the C core first and then its
Python bindings, ``python-igraph``.

