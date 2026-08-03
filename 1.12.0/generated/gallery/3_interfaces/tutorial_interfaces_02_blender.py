"""
Blender 3D
==========

This tutorial shows how to use NAVis within Blender 3D.

{{ navis }} comes with an interface to import neurons into [Blender 3D](https://www.blender.org)
for high quality renderings and videos: `navis.interfaces.blender`.

## Installation

Blender comes with its own Python 3.X distribution! So you need to install {{ navis }} explicitly
for this distribution in order to use it within Blender.

There are several ways to install additional packages for Blender's built-in Python. The easiest
way is probably this:

1. Find out where Blender's Python lives (this depends on your OS). In
   _Blender's Python console_ run this:

    ```python
    >>> import sys
    >>> sys.executable
    [..]/Blender 4.1.app/Contents/Resources/4.1/python/bin/python3.11
    ```

    ![Blender Python console](../../../_static/blender_console.png)

2. Now that we know the Python path we open a normal terminal and check if Blender's Python
   already came with the package manager `pip`.

    ```shell
    [..]/Blender\ 4.1.app/Contents/Resources/4.1/python/bin/python3.11 -m pip --version
    ```

    ![Blender PIP](../../../_static/blender_pip.png)

    !!! warning
        You may have to escape whitespaces in the path to Blender's Python executable
        like we did above! On OSX this is done with a backslash `\`. On Windows you
        have to wrap the path in quotes `"` if it contains spaces.

    If the above command throws an error along the lines of `"No module named pip"`:
    get `pip` by downloading ``get-pip.py`` from
    [here](https://pip.pypa.io/en/stable/installing/) and install by executing
    with your Python distribution:

    ```shell
    [..]/Blender\ 4.1.app/Contents/Resources/4.1/python/bin/python3.11 get-pip.py
    ```

    If `pip` is there but horrendously outdated (the current version is `24.4`),
    you can update it like so:

    ```shell
    [..]/Blender\ 4.1.app/Contents/Resources/4.1/python/bin/python3.11 -m pip install pip -U
    ```

3. Use `pip` to install {{ navis }} (or any other package for that matter). Please note
   we have to - again - specify that we want to install for Blender's Python:

    ```shell
    [..]/Blender\ 4.1.app/Contents/Resources/4.1/python/bin/python3.11 -m pip install navis
    ```

    !!! important
        It's possible that this install fails with an error message along the lines
        of `'Python.h' file not found`. The reason for this is that Blender
        ships with a "Python light" and you have to manually provide the Python
        header files:

        First, find out the *exact* Blender Python version:

        ```shell
        [..]/Blender\ 4.1.app/Contents/Resources/4.1/python/bin/python3.11 -V
        ```

        Next point your browser at https://www.python.org/downloads/source/ and
        download the Gzipped source tarball from the exact same Python version,
        i.e. `Python-3.X.X.tgz` and save it to your Downloads directory.

        Finally you need to copy everything in the `Include` folder inside that
        tarball into the corresponding `include` folder in your Blender's Python.
        In a terminal run::

        ```shell
        cd ~/Downloads/
        tar -xzf Python-3.X.X.tgz
        cp Python-3.X.X/Include/* [..]/Blender\ 4.1.app/Contents/Resources/4.1/python/bin/python3.11
        ```

        If the above fails you have one more option: figure out which dependency fails
        to compile and compile it on your system's Python.

        a) Install the *exact* same version of Python as Blender is running on your
        system
        b) Download the source code for the offending dependency either from PyPI
        where it'll likely be some `tar.gz` file under "Download files" or
        from the Github repository
        c) Run `python setup.py bdist_wheel` to compile the dependency into a wheel
        file (will appear as `.whl` file in a `/dist` subdirectory)
        d) Go back to Blender's Python and install the dependency from that wheel:
        ```shell
        [..]/Blender\ 4.1.app/Contents/Resources/4.1/python/bin/python3.11 -m pip install <full file name of wheel file with .whl extension>
        ```

4. You should now be all set to use {{ navis }} in Blender. Check out Quickstart!

## Quickstart

`navis.interfaces.blender` provides a simple interface that lets you add,
select and manipulate neurons from within _Blender's Python console_:

First, import and set up {{ navis }} like you are used to.

```python
>>> import navis
>>> # Get example neurons
>>> nl = navis.example_neurons()
```

Now initialise the interface with Blender and import the neurons.

```python
>>> # The blender interface has to be imported explicitly
>>> import navis.interfaces.blender as b3d
>>> # Initialise handler
>>> h = b3d.Handler()
>>> # Load neurons into scene
>>> h.add(nl)
```

![b3d_screenshot](../../../_static/b3d_screenshot.jpg)


The interface lets you manipulate neurons in Blender too.

```python
>>> # Colorize neurons
>>> h.colorize()
>>> # Change thickness of all neurons
>>> h.neurons.bevel(.02)
>>> # Select subset
>>> subset = h.select(nl[:2])
>>> # Make subset red
>>> subset.color(1, 0, 0)
>>> # Clear all objects
>>> h.clear()
```

!!! note
    Blender's Python console does not show all outputs. Please check the terminal
    if you experience issues. In Windows simply go to `Help` >> `Toggle System
    Console`. In MacOS, right-click Blender in
    Finder >> `Show Package Contents` >> `MacOS` >> double click on `blender`.

Last but not least, here's a little taster of what you can do with Blender:

<iframe width="560" height="315" src="https://www.youtube.com/embed/wl3sFG7WQJc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Reference

The [`navis.interfaces.blender.Handler`][] is providing the interface between {{ navis }} and Blender.

"""

# %%

# mkdocs_gallery_thumbnail_path = '_static/blender_logo.png'