Troubleshooting
===============

You've encountered a problem? See if you can find it in below table. If not,
please get in touch via `issues <https://github.com/schlegelp/PyMaid/issues>`_
on Github.

.. list-table:: 
   :widths: 40 60
   :header-rows: 1

   * - Problem
     - Solution
   * - **Installation**
     -
   * - PyOctree fails compiling because of ``fopenmp``.
     -  1. Download and extract the PyOctree Github `repository <https://github.com/mhogg/pyoctree>`_. 
        2. Open ``setup.py`` and set ``BUILD_ARGS['mingw32'] = [ ]`` and ``LINK_ARGS['unix'] = [ ]``
        3. Open a terminal, navigate to the directory containing ``setup.py`` and run ``python setup.py install`` (if your default Python is 2.7, use ``python3``)
   * - **Fetching data**
     -        
   * - Fetching any data throws an exception ``<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate very failed>``.
     - Have a look at this `blog post <http://www.cdotson.com/2017/01/sslerror-with-python-3-6-x-on-macos-sierra/>`_.
   * - **Plotting**
     -
   * - 3D plots with VisPy as backend use only one quarter of the canvas.
     - Try installing the developer version from GitHub (https://github.com/vispy/vispy). As one-liner::

         git clone https://github.com/vispy/vispy.git && cd vispy && python setup.py install --user

   * - 3D plots using Plotly are too small and all I can see is a chunk of legend.
     - Sometimes plotly does not scale the plot correctly. The solution is to play around with the ``width`` parameter::

         fig = pymaid.plot3d(neurons, backend='plotly', width=1200)

   * - Opening a vispy 3D viewer, throws a long list of errors ending with something like this::

         RuntimeError: Error while fetching file http://github.com/vispy/demo-data/raw/master/fonts/OpenSans-Regular.ttf.
         Dataset fetching aborted (<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)>)

     - For reasons beyond me, vispy does not include the font to render text so it has to download it on first use. If this fails with an ``SSL`` error, do the following once::

         import pymaid
         import ssl
         ssl._create_default_https_context = ssl._create_unverified_context
         v = pymaid.Viewer()

       This temporarily disables SSL verification to allow download of the font. I recommend restarting the Python session afterwards!

   * - **Jupyter**
     -
   * - Instead of a progress bar, I get some odd message (e.g. ``Hbox(children=...``) when using pymaid in a Jupyter notebook.
     - You probably have `ipywidgets <ipywidgets.readthedocs.io>`_ not installed or not configured properly. One work-around is to force pymaid to use standard progress bars using :func:`pymaid.set_pbars`::
        
         pymaid.set_pbars(jupyter=False)
