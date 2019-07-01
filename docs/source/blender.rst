.. _blender3d:

.. role:: red

Blender 3D
----------


NAVis comes with an interface to import neurons into
`Blender 3D <https://www.blender.org>`_: :mod:`navis.interfaces.blender`

Because NAVis requires Python at least 3.6 and only the most recent version of
Blender comes with Python >=3.5, we require Blender *2.8*!

Installation
============

Blender 2.8 comes with its own Python 3.7 distribution! So you need to install
NAVis explicitly for this distribution in order to use it within Blender.

There are several ways to install additional packages for Blender's
built-in Python. IMHO the easiest way is this:

1. Find out where Blender's Python lives (this depends on your OS). In
   :red:`Blender's Python console` run this::

    >>> bpy.app.binary_path_python
    '[..]/blender.app/Contents/Resources/2.80/python/bin/python3.7m'

    The same directory also has Blender's PIP: simply replace `python3.7m`
    with `pip` in the next steps.

2. Get pip by downloading ``get-pip.py`` from <here `https://pip.pypa.io/en/stable/installing/`>_
   and install by executing with your Python distribution::

    [..]/blender.app/Contents/Resources/2.80/python/bin/python3.7m get-pip.py

3. Use PIP to install NAVis (or any other package for that
   matter). Please note we have to - again - specify that we want to install
   for Blender's Python::

    [..]/blender.app/Contents/Resources/2.80/python/bin/pip install navis

4. You should now be all set to use NAVis in Blender. Check out Quickstart!

.. note::
   If any of the above steps fails with a *Permission* error, try the same
   command but prefixed with "``sudo``".

Quickstart
==========

:mod:`navis.interfaces.blender` provides a simple interface that lets you add,
select an manipulate neurons from within :red:`Blender's Python console`:

First, import and set up NAVis like you are used to.

>>> import navis
>>> # Get example neurons
>>> nl = navis.example_neurons()

Now initialise the interface with Blender and import the neurons.

>>> # The blender interface has be imported explicitly
>>> import navis.interfaces.blender as b3d
>>> # Initialise handler
>>> handler = b3d.handler()
>>> # Load neurons into scene
>>> handler.add(nl)

The interface lets you manipulate neurons in Blender too.

>>> # Colorise neurons
>>> handler.colorize()
>>> # Change thickness of all neurons
>>> handler.neurons.bevel(.02)
>>> # Select subset
>>> subset = handler.select(nl[:2])
>>> # Make subset red
>>> subset.color(1, 0, 0)
>>> # Clear all objects
>>> handler.clear()

.. note::
   Blender's Python console does not show all outputs. Please check the terminal
   if you experience issues. In Windows simply go to `Help` >> `Toggle System
   Console`. In MacOS, right-click Blender in Finder >> `Show Package Contents`
   >> `MacOS` >> double click on `blender`.

Reference
~~~~~~~~~

Objects
+++++++
.. autosummary::
    :toctree: generated/

    navis.interfaces.blender.Handler.add
    navis.interfaces.blender.Handler.clear
    navis.interfaces.blender.Handler.select
    navis.interfaces.blender.Handler.hide
    navis.interfaces.blender.Handler.unhide

Materials
+++++++++
.. autosummary::
    :toctree: generated/

    navis.interfaces.blender.Handler.color
    navis.interfaces.blender.Handler.colorize
    navis.interfaces.blender.Handler.emit
    navis.interfaces.blender.Handler.use_transparency
    navis.interfaces.blender.Handler.alpha
    navis.interfaces.blender.Handler.bevel


Selections
++++++++++
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
