r"""
Blender 3D
==========
<!-- difficulty: intermediate -->

Drive Blender 3D from NAVis for high-quality neuron renders.

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

    You may have to escape whitespace in the path to Blender's Python executable, like we did above:

    === "macOS/Linux"
        Escape each space with a backslash, e.g. `Blender\ 4.1.app/.../bin/python3.11`.

    === "Windows"
        Wrap the path in double quotes if it contains spaces, e.g. `"C:\...\Blender\python.exe"`.

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

    ??? warning "If the install fails to compile"
        If the install fails with `'Python.h' file not found`, Blender's "Python light" is missing its
        header files and you have to supply them manually:

        1. Find the *exact* Blender Python version:

            ```shell
            [..]/Blender\ 4.1.app/Contents/Resources/4.1/python/bin/python3.11 -V
            ```

        2. Download the matching *Gzipped source tarball* (`Python-3.X.X.tgz`) from
           <https://www.python.org/downloads/source/> into your Downloads directory.

        3. Copy the headers from the tarball's `Include` folder into Blender's Python `include` folder:

            ```shell
            cd ~/Downloads/
            tar -xzf Python-3.X.X.tgz
            cp Python-3.X.X/Include/* [..]/Blender\ 4.1.app/Contents/Resources/4.1/python/bin/python3.11
            ```

        If that still fails, compile the offending dependency on your system's Python and install the wheel:

        1. Install the *exact* same version of Python that Blender uses.
        2. Download the dependency's source (a `tar.gz` from PyPI's "Download files", or the GitHub repo).
        3. Build a wheel with `python setup.py bdist_wheel` (it appears as a `.whl` in the `/dist` subdirectory).
        4. Install that wheel into Blender's Python:

            ```shell
            [..]/Blender\ 4.1.app/Contents/Resources/4.1/python/bin/python3.11 -m pip install <wheel-file>.whl
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
h.colorize()               # (1)!
h.neurons.bevel(0.02)      # (2)!
subset = h.select(nl[:2])  # (3)!
subset.color(1, 0, 0)      # (4)!
h.clear()                  # (5)!
```

1.  Give every neuron a different color.
2.  Set the thickness (bevel) of all neurons.
3.  Select a subset of neurons - here the first two.
4.  Color that subset red (values are `R, G, B`).
5.  Remove all objects from the scene.

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