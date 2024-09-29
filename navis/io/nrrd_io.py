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

import nrrd
import io

import numpy as np

from pathlib import Path
from typing import Union, Iterable, Optional, Dict, Any
from typing_extensions import Literal
from urllib3 import HTTPResponse

from .. import config, utils, core
from . import base

# Set up logging
logger = config.get_logger(__name__)

DEFAULT_FMT = "{name}.nrrd"


class NrrdReader(base.ImageReader):
    def __init__(
        self,
        output: Literal["voxels", "dotprops", "raw"] = "voxels",
        threshold: Optional[Union[int, float]] = None,
        thin: bool = False,
        dotprop_kwargs: Dict[str, Any] = {},
        fmt: str = DEFAULT_FMT,
        attrs: Optional[Dict[str, Any]] = None,
        errors: str = "raise",
    ):
        if not fmt.endswith(".nrrd"):
            raise ValueError('`fmt` must end with ".nrrd"')

        super().__init__(
            fmt=fmt,
            attrs=attrs,
            file_ext=".nrrd",
            name_fallback="NRRD",
            read_binary=True,
            output=output,
            threshold=threshold,
            thin=thin,
            dotprop_kwargs=dotprop_kwargs,
            errors=errors,
        )

    def format_output(self, x):
        # This function replaces the BaseReader.format_output()
        # This is to avoid trying to convert multiple (image, header) to NeuronList
        if self.output == "raw":
            return [n for n in x if n]
        elif x:
            return core.NeuronList([n for n in x if n])
        else:
            return core.NeuronList([])

    @base.handle_errors
    def read_buffer(
        self, f, attrs: Optional[Dict[str, Any]] = None
    ) -> Union[np.ndarray, "core.Dotprops", "core.VoxelNeuron"]:
        """Read buffer into (image, header) or a neuron.

        Parameters
        ----------
        f :         IO
                    Readable buffer (must be bytes).
        attrs :     dict | None
                    Arbitrary attributes to include in the neuron.

        Returns
        -------
        core.Dotprops | core.VoxelNeuron | np.ndarray

        """
        if isinstance(f, HTTPResponse):
            f = io.StringIO(f.content)

        if isinstance(f, bytes):
            f = io.BytesIO(f)

        header = nrrd.read_header(f)
        data = nrrd.read_data(header, f)

        if self.output == "raw":
            return data, header

        # Try parsing units - this is modelled after the nrrd files you get from
        # Virtual Fly Brain (VFB)
        units = None
        space_units = None
        voxdim = np.array([1, 1, 1])
        if "space directions" in header:
            sd = np.asarray(header["space directions"])
            if sd.ndim == 2:
                voxdim = np.diag(sd)[:3]
        if "space units" in header:
            space_units = header["space units"]
            if len(space_units) == 3:
                units = [f"{m} {u}" for m, u in zip(voxdim, space_units)]
        else:
            units = voxdim

        return self.convert_image(data, attrs, header, voxdim, units, space_units)


def write_nrrd(
    x: "core.NeuronObject",
    filepath: Union[str, Path],
    compression_level: int = 3,
    attrs: Optional[Dict[str, Any]] = None,
) -> None:
    """Write VoxelNeurons or Dotprops to NRRD file(s).

    Parameters
    ----------
    x :                 VoxelNeuron | Dotprops | NeuronList
                        If multiple neurons, will generate a NRRD file
                        for each neuron (see also `filepath`).
    filepath :          str | pathlib.Path | list thereof
                        Destination for the NRRD files. See examples for options.
                        If `x` is multiple neurons, `filepath` must either
                        be a folder, a "formattable" filename (see Examples) or
                        a list of filenames (one for each neuron in `x`).
                        Existing files will be overwritten!
    compression_level : int 1-9
                        Lower = faster writing but larger files. Higher = slower
                        writing but smaller files.
    attrs :             dict
                        Any additional attributes will be written to NRRD header.

    Returns
    -------
    Nothing

    Examples
    --------
    Save a single neuron to a specific file:

    >>> import navis
    >>> n = navis.example_neurons(1, kind='skeleton')
    >>> vx = navis.voxelize(n, pitch='2 microns')
    >>> navis.write_nrrd(vx, tmp_dir / 'my_neuron.nrrd')

    Save multiple neurons to a folder (must exist). Filenames will be
    autogenerated as "{neuron.id}.nrrd":

    >>> import navis
    >>> nl = navis.example_neurons(5, kind='skeleton')
    >>> dp = navis.make_dotprops(nl, k=5)
    >>> navis.write_nrrd(dp, tmp_dir)

    Save multiple neurons to a folder but modify the pattern for the
    autogenerated filenames:

    >>> import navis
    >>> nl = navis.example_neurons(5, kind='skeleton')
    >>> vx = navis.voxelize(nl, pitch='2 microns')
    >>> navis.write_nrrd(vx, tmp_dir / 'voxels-{neuron.name}.nrrd')

    Save multiple neurons to a zip file:

    >>> import navis
    >>> nl = navis.example_neurons(5, kind='skeleton')
    >>> vx = navis.voxelize(nl, pitch='2 microns')
    >>> navis.write_nrrd(vx, tmp_dir / 'neuronlist.zip')

    Save multiple neurons to a zip file but modify the filenames:

    >>> import navis
    >>> nl = navis.example_neurons(5, kind='skeleton')
    >>> vx = navis.voxelize(nl, pitch='2 microns')
    >>> navis.write_nrrd(vx, tmp_dir / 'voxels-{neuron.name}.nrrd@neuronlist.zip')

    See Also
    --------
    [`navis.read_nrrd`][]
                        Import VoxelNeuron from NRRD files.

    """
    compression_level = int(compression_level)

    if (compression_level < 1) or (compression_level > 9):
        raise ValueError("`compression_level` must be 1-9, got " f"{compression_level}")

    writer = base.Writer(_write_nrrd, ext=".nrrd")

    return writer.write_any(
        x, filepath=filepath, compression_level=compression_level, **(attrs or {})
    )


def _write_nrrd(
    x: Union["core.VoxelNeuron", "core.Dotprops"],
    filepath: Optional[str] = None,
    compression_level: int = 1,
    **attrs,
) -> None:
    """Write single neuron to NRRD file."""
    if not isinstance(x, (core.VoxelNeuron, core.Dotprops)):
        raise TypeError(f'Expected VoxelNeuron or Dotprops, got "{type(x)}"')

    header = getattr(x, "nrrd_header", {})
    header["space dimension"] = 3
    header["space directions"] = np.diag(x.units_xyz.magnitude)
    header["space units"] = [str(x.units_xyz.units)] * 3
    header.update(attrs or {})

    if isinstance(x, core.VoxelNeuron):
        data = x.grid
        if data.dtype == bool:
            data = data.astype("uint8")
    else:
        # For dotprops make a horizontal stack from points + vectors
        data = np.hstack((x.points, x.vect))
        header["k"] = x.k

    nrrd.write(
        str(filepath), data=data, header=header, compression_level=compression_level
    )


def read_nrrd(
    f: Union[str, Iterable],
    output: Union[Literal["voxels"], Literal["dotprops"], Literal["raw"]] = "voxels",
    threshold: Optional[Union[int, float]] = None,
    thin: bool = False,
    include_subdirs: bool = False,
    parallel: Union[bool, int] = "auto",
    fmt: str = "{name}.nrrd",
    limit: Optional[int] = None,
    errors: str = "raise",
    **dotprops_kwargs,
) -> "core.NeuronObject":
    """Create Neuron/List from NRRD file.

    See [here](http://teem.sourceforge.net/nrrd/format.html) for specs of
    NRRD file format including description of the headers.

    Parameters
    ----------
    f :                 str | list thereof
                        Filename, folder or URL:
                         - if folder, will import all `.nrrd` files
                         - if a `.zip`, `.tar` or `.tar.gz` archive will read all
                           NRRD files from the file
                         - if a URL (http:// or https://), will download the
                           file and import it
                         - FTP address (ftp://) can point to a folder or a single
                           file
                        See also `limit` parameter to read only a subset of files.
    output :            "voxels" | "dotprops" | "raw"
                        Determines function's output. See Returns for details.
    threshold :         int | float | None
                        For `output='dotprops'` only: a threshold to filter
                        low intensity voxels.
                          - if `None`, all values > 0 are converted to points
                          - if >=1, all values >= threshold are converted to points
                          - if <1, all values >= threshold * max(data) are converted
    thin :              bool
                        For `output='dotprops'` only: if True, will thin the
                        point cloud using `skimage.morphology.skeletonize`
                        after thresholding. Requires `scikit-image`.
    include_subdirs :   bool, optional
                        If True and `f` is a folder, will also search
                        subdirectories for `.nrrd` files.
    parallel :          "auto" | bool | int,
                        Defaults to `auto` which means only use parallel
                        processing if more than 10 NRRD files are imported.
                        Spawning and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the number of
                        cores (otherwise defaults to `os.cpu_count() - 2`).
    fmt :               str
                        Formatter to specify how filenames are parsed into neuron
                        attributes. Some illustrative examples:
                          - `{name}` (default) uses the filename
                            (minus the suffix) as the neuron's name property
                          - `{id}` (default) uses the filename as the neuron's ID
                            property
                          - `{name,id}` uses the filename as the neuron's
                            name and ID properties
                          - `{name}.{id}` splits the filename at a "."
                            and uses the first part as name and the second as ID
                          - `{name,id:int}` same as above but converts
                            into integer for the ID
                          - `{name}_{myproperty}` splits the filename at
                            "_" and uses the first part as name and as a
                            generic "myproperty" property
                          - `{name}_{}_{id}` splits the filename at
                            "_" and uses the first part as name and the last as
                            ID. The middle part is ignored.

                        Throws a ValueError if pattern can't be found in
                        filename.
    limit :             int | str | slice | list, optional
                        When reading from a folder or archive you can use this parameter to
                        restrict the which files read:
                         - if an integer, will read only the first `limit` NMX files
                           (useful to get a sample from a large library of skeletons)
                         - if a string, will interpret it as filename (regex) pattern
                           and only read files that match the pattern; e.g. `limit='.*_R.*'`
                           will only read files that contain `_R` in their filename
                         - if a slice (e.g. `slice(10, 20)`) will read only the files in
                           that range
                         - a list is expected to be a list of filenames to read from
                           the folder/archive
    errors :            "raise" | "log" | "ignore"
                        If "log" or "ignore", errors will not be raised and the
                        mesh will be skipped. Can result in empty output.
    **dotprops_kwargs
                        Keyword arguments passed to [`navis.make_dotprops`][]
                        if `output='dotprops'`. Use this to adjust e.g. the
                        number of nearest neighbors used for calculating the
                        tangent vector by passing e.g. `k=5`.

    Returns
    -------
    navis.VoxelNeuron
                        If `output="voxels"` (default): requires NRRD data to
                        be 3-dimensional voxels. VoxelNeuron will have NRRD file
                        header as `.nrrd_header` attribute.
    navis.Dotprops
                        If `output="dotprops"`: requires NRRD data to be
                        either:
                          - `(N, M, K)` (i.e. 3D) in which case we will turn
                            voxels into a point cloud (see also `threshold`
                            parameter)
                          - `(N, 3)` = x/y/z points
                          - `(N, 6)` = x/y/z points + x/y/z vectors
                          - `(N, 7)` = x/y/z points + x/y/z vectors + alpha

                        Dotprops will contain NRRD header as `.nrrd_header`
                        attribute.
    navis.NeuronList
                        If import of multiple NRRD will return NeuronList of
                        Dotprops/VoxelNeurons.
    (image, header)     (np.ndarray, OrderedDict)
                        If `output='raw'` return raw data contained in NRRD
                        file.

    """
    if thin:
        try:
            from skimage.morphology import skeletonize
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "The 'thin' option requires 'scikit-image' to be installed:\n"
                "    pip install scikit-image -U"
            )

    utils.eval_param(
        output, name="output", allowed_values=("raw", "dotprops", "voxels")
    )

    if parallel == "auto":
        # Set a lower threshold of 10 on parallel processing for NRRDs (default is 200)
        parallel = ("auto", 10)

    reader = NrrdReader(
        output=output, threshold=threshold, thin=thin, fmt=fmt, errors=errors, dotprop_kwargs=dotprops_kwargs
    )
    return reader.read_any(f, include_subdirs, parallel, limit=limit)
