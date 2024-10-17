#    This script is part of navis (http://www.github.com/navis-org/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

import os
import io

import trimesh as tm

from typing import Union, Iterable, Optional, Dict, Any
from typing_extensions import Literal
from urllib3 import HTTPResponse

from .. import config, utils, core
from . import base

# Set up logging
logger = config.get_logger(__name__)

# Mesh files can have all sort of extensions
DEFAULT_FMT = "{name}.{file_ext}"

# Mesh extensions supported by trimesh
MESH_LOAD_EXT = tuple(tm.exchange.load.mesh_loaders.keys())
MESH_WRITE_EXT = tuple(tm.exchange.export._mesh_exporters.keys())


class MeshReader(base.BaseReader):
    def __init__(
        self,
        output: str,
        fmt: str = DEFAULT_FMT,
        attrs: Optional[Dict[str, Any]] = None,
        errors: str = "raise",
    ):
        super().__init__(
            fmt=fmt,
            attrs=attrs,
            file_ext=MESH_LOAD_EXT,
            name_fallback="MESH",
            read_binary=True,
            errors=errors,
        )
        self.output = output

    def format_output(self, x):
        # This function replaces the BaseReader.format_output()
        # This is to avoid trying to convert multiple (image, header) to NeuronList
        if self.output == "trimesh":
            return x
        elif x:
            return core.NeuronList(x)
        else:
            return core.NeuronList([])

    @base.handle_errors
    def read_buffer(
        self, f, attrs: Optional[Dict[str, Any]] = None
    ) -> Union[tm.Trimesh, "core.Volume", "core.MeshNeuron"]:
        """Read buffer into mesh.

        Parameters
        ----------
        f :         IO
                    Readable buffer (must be bytes).
        attrs :     dict | None
                    Arbitrary attributes to include in the neurons.

        Returns
        -------
        Trimesh | MeshNeuron | Volume

        """
        if isinstance(f, HTTPResponse):
            f = io.StringIO(f.content)

        if isinstance(f, bytes):
            f = io.BytesIO(f)

        # We need to tell trimesh what file type we are reading
        if "file" not in attrs:
            raise KeyError(
                f'Unable to parse file type. "file" not in attributes: {attrs}'
            )

        file_type = attrs["file"].split(".")[-1]

        mesh = tm.load_mesh(f, file_type=file_type)

        if self.output == "trimesh":
            return mesh
        elif self.output == "volume":
            return core.Volume(mesh.vertices, mesh.faces, **attrs)

        # Turn into a MeshNeuron
        n = core.MeshNeuron(mesh)

        # Try adding properties one-by-one. If one fails, we'll keep track of it
        # in the `.meta` attribute
        meta = {}
        for k, v in attrs.items():
            try:
                n._register_attr(k, v)
            except (AttributeError, ValueError, TypeError):
                meta[k] = v

        if meta:
            n.meta = meta

        return n


def read_mesh(
    f: Union[str, Iterable],
    include_subdirs: bool = False,
    parallel: Union[bool, int] = "auto",
    output: Union[Literal["neuron"], Literal["volume"], Literal["trimesh"]] = "neuron",
    errors: Literal["raise", "log", "ignore"] = "raise",
    limit: Optional[int] = None,
    fmt: str = "{name}.",
    **kwargs,
) -> "core.NeuronObject":
    """Load mesh file into Neuron/List.

    This is a thin wrapper around `trimesh.load_mesh` which supports most
    commonly used formats (obj, ply, stl, etc.).

    Parameters
    ----------
    f :                 str | iterable
                        Filename(s) or folder. If folder should include file
                        extension (e.g. `my/dir/*.ply`) otherwise all
                        mesh files in the folder will be read.
    include_subdirs :   bool, optional
                        If True and `f` is a folder, will also search
                        subdirectories for meshes.
    parallel :          "auto" | bool | int,
                        Defaults to `auto` which means only use parallel
                        processing if more than 100 mesh files are imported.
                        Spawning and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the number of
                        cores (otherwise defaults to `os.cpu_count() - 2`).
    output :            "neuron" | "volume" | "trimesh"
                        Determines function's output - see `Returns`.
    errors :            "raise" | "log" | "ignore"
                        If "log" or "ignore", errors will not be raised and the
                        mesh will be skipped. Can result in empty output.
    limit :             int | str | slice | list, optional
                        When reading from a folder or archive you can use this parameter to
                        restrict the which files read:
                         - if an integer, will read only the first `limit` mesh files
                           (useful to get a sample from a large library of meshes)
                         - if a string, will interpret it as filename (regex) pattern
                           and only read files that match the pattern; e.g. `limit='.*_R.*'`
                           will only read files that contain `_R` in their filename
                         - if a slice (e.g. `slice(10, 20)`) will read only the files in
                           that range
                         - a list is expected to be a list of filenames to read from
                           the folder/archive
    **kwargs
                        Keyword arguments passed to [`navis.MeshNeuron`][]
                        or [`navis.Volume`][]. You can use this to e.g.
                        set the units on the neurons.

    Returns
    -------
    MeshNeuron
                        If `output="neuron"` (default).
    Volume
                        If `output="volume"`.
    Trimesh
                        If `output="trimesh"`.
    NeuronList
                        If `output="neuron"` and import has multiple meshes
                        will return NeuronList of MeshNeurons.
    list
                        If `output!="neuron"` and import has multiple meshes
                        will return list of Volumes or Trimesh.

    See Also
    --------
    [`navis.read_precomputed`][]
                        Read meshes and skeletons from Neuroglancer's precomputed format.

    Examples
    --------

    Read a single file into [`navis.MeshNeuron`][]:

    >>> m = navis.read_mesh('mesh.obj')                         # doctest: +SKIP

    Read all e.g. .obj files in a directory:

    >>> nl = navis.read_mesh('/some/directory/*.obj')           # doctest: +SKIP

    Sample first 50 files in folder:

    >>> nl = navis.read_mesh('/some/directory/*.obj', limit=50) # doctest: +SKIP

    Read single file into [`navis.Volume`][]:

    >>> nl = navis.read_mesh('mesh.obj', output='volume')       # doctest: +SKIP

    """
    utils.eval_param(
        output, name="output", allowed_values=("neuron", "volume", "trimesh")
    )

    reader = MeshReader(fmt=fmt, output=output, errors=errors, attrs=kwargs)
    return reader.read_any(f, include_subdirs, parallel, limit=limit)


def write_mesh(
    x: Union["core.NeuronList", "core.MeshNeuron", "core.Volume", "tm.Trimesh"],
    filepath: Optional[str] = None,
    filetype: str = None,
) -> None:
    """Export meshes (MeshNeurons, Volumes, Trimeshes) to disk.

    Under the hood this is using trimesh to export meshes.

    Parameters
    ----------
    x :                 MeshNeuron | Volume | Trimesh | NeuronList
                        If multiple objects, will generate a file for each
                        neuron (see also `filepath`).
    filepath :          None | str | list, optional
                        If `None`, will return byte string or list of
                        thereof. If filepath will save to this file. If path
                        will save neuron(s) in that path using `{x.id}`
                        as filename(s). If list, input must be NeuronList and
                        a filepath must be provided for each neuron.
    filetype :          stl | ply | obj, optional
                        If `filepath` does not include the file extension,
                        you need to provide it as `filetype`.

    Returns
    -------
    None
                        If filepath is not `None`.
    bytes
                        If filepath is `None`.

    See Also
    --------
    [`navis.read_mesh`][]
                        Import neurons.
    [`navis.write_precomputed`][]
                        Write meshes to Neuroglancer's precomputed format.

    Examples
    --------

    Write `MeshNeurons` to folder:

    >>> import navis
    >>> nl = navis.example_neurons(3, kind='mesh')
    >>> navis.write_mesh(nl, tmp_dir, filetype='obj')

    Specify the filenames:

    >>> import navis
    >>> nl = navis.example_neurons(3, kind='mesh')
    >>> navis.write_mesh(nl, tmp_dir / '{neuron.name}.obj')

    Write directly to zip archive:

    >>> import navis
    >>> nl = navis.example_neurons(3, kind='mesh')
    >>> navis.write_mesh(nl, tmp_dir / 'meshes.zip', filetype='obj')

    """
    if filetype is not None:
        utils.eval_param(filetype, name="filetype", allowed_values=MESH_WRITE_EXT)
    else:
        # See if we can get filetype from filepath
        if filepath is not None:
            for f in MESH_WRITE_EXT:
                if str(filepath).endswith(f".{f}"):
                    filetype = f
                    break

        if not filetype:
            raise ValueError(
                "Must provide mesh type either explicitly via "
                "`filetype` variable or implicitly via the "
                "file extension in `filepath`"
            )

    writer = base.Writer(_write_mesh, ext=f".{filetype}")

    return writer.write_any(x, filepath=filepath)


def _write_mesh(
    x: Union["core.MeshNeuron", "core.Volume", "tm.Trimesh"],
    filepath: Optional[str] = None,
) -> None:
    """Write single mesh to disk."""
    if filepath and os.path.isdir(filepath):
        if isinstance(x, core.MeshNeuron):
            if not x.id:
                raise ValueError(
                    "Neuron(s) must have an ID when destination " "is a folder"
                )
            filepath = os.path.join(filepath, f"{x.id}")
        elif isinstance(x, core.Volume):
            filepath = os.path.join(filepath, f"{x.name}")
        else:
            raise ValueError(f"Unable to generate filename for {type(x)}")

    if isinstance(x, core.MeshNeuron):
        mesh = x.trimesh
    elif isinstance(x, tm.Trimesh):
        mesh = x
    else:
        raise TypeError(f'Unable to write data of type "{type(x)}"')

    # Write to disk (will only return content if filename is None)
    content = mesh.export(filepath)

    if not filepath:
        return content
