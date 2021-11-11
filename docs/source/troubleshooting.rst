Troubleshooting
===============

You've encountered a problem? See if you can find it in below table. If not,
please get in touch via `issues <https://github.com/navis-org/navis/issues>`_
on Github.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Problem
     - Solution
   * - **Installation**
     -
   * - Installing navis for Blender fails with "Python.h" not found.
     - Manually download and copy required files into Blender's Python directory. Follow the instructions `here <https://blender.stackexchange.com/questions/81740/python-h-missing-in-blender-python>`_.
   * - **Plotting**
     -
   * - 3D plots with VisPy as backend use only one quarter of the canvas.
     - Try installing the developer version from GitHub (https://github.com/vispy/vispy). As one-liner::

         git clone https://github.com/vispy/vispy.git && cd vispy && python setup.py install --user

   * - 3D plots using Plotly are too small and all I can see is a chunk of legend.
     - Sometimes plotly does not scale the plot correctly. The solution is to play around with the ``width`` parameter::

         fig = navis.plot3d(neurons, backend='plotly', width=1200)

   * - Opening a vispy 3D viewer, throws a long list of errors ending with something like this::

         RuntimeError: Error while fetching file http://github.com/vispy/demo-data/raw/master/fonts/OpenSans-Regular.ttf.
         Dataset fetching aborted (<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)>)

     - For reasons beyond me, vispy does not include the font to render text so it has to download it on first use. If this fails with an ``SSL`` error, do the following once::

         import navis
         import ssl
         ssl._create_default_https_context = ssl._create_unverified_context
         v = navis.Viewer()

       This temporarily disables SSL verification to allow download of the font. I recommend restarting the Python session afterwards!

   * - **Jupyter**
     -
   * - Instead of a progress bar, I get some odd message (e.g. ``Hbox(children=...``) when using navis in a Jupyter notebook.
     - You probably have `ipywidgets <ipywidgets.readthedocs.io>`_ not installed or not configured properly. One work-around is to force navis to use standard progress bars using :func:`navis.set_pbars`::

         navis.set_pbars(jupyter=False)

   * - **Output**
     -
   * - Navis starts viewers for plots when I don't want it to.
     - Set the environment variable ``NAVIS_HEADLESS`` to ``"True"`` before navis is first imported to disable viewers (good for use on servers).
   * - Navis interferes with my logging configuration.
     - By default, navis configures sensible defaults for logging (helpful for jupyter notebooks, scripting, and exploratory REPL use). Set the environment variable ``NAVIS_SKIP_LOG_SETUP`` to ``"True"`` before navis is first imported to disable this (good when using navis as a library). Use :func:`navis.config.default_logging()` to manually run the log setup.
