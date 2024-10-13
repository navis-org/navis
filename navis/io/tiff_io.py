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

import io

import numpy as np

from typing import Union, Iterable, Optional, Dict, Any
from typing_extensions import Literal
from urllib3 import HTTPResponse

from .. import config, utils, core
from . import base

# Set up logging
logger = config.logger

DEFAULT_FMT = "{name}.tif"


class TiffReader(base.ImageReader):
    def __init__(
        self,
        output: Literal["voxels", "dotprops", "raw"] = "voxels",
        channel: int = 0,
        threshold: Optional[Union[int, float]] = None,
        thin: bool = False,
        dotprop_kwargs: Dict[str, Any] = {},
        fmt: str = DEFAULT_FMT,
        errors: str = "raise",
        attrs: Optional[Dict[str, Any]] = None,
    ):
        if not fmt.endswith(".tif") and not fmt.endswith(".tiff"):
            raise ValueError('`fmt` must end with ".tif" or ".tiff"')

        super().__init__(
            fmt=fmt,
            attrs=attrs,
            file_ext=(".tif", ".tiff"),
            name_fallback="TIFF",
            read_binary=True,
            output=output,
            threshold=threshold,
            thin=thin,
            dotprop_kwargs=dotprop_kwargs,
            errors=errors,
        )
        self.channel = channel

    def format_output(self, x):
        # This function replaces the BaseReader.format_output()
        # This is to avoid trying to convert multiple (image, header) to NeuronList
        if self.output == "raw":
            return x
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
        import tifffile

        if isinstance(f, HTTPResponse):
            f = io.StringIO(f.content)

        if isinstance(f, bytes):
            f = io.BytesIO(f)

        with tifffile.TiffFile(f) as tif:
            # The header contains some but not all the info
            if hasattr(tif, "imagej_metadata") and tif.imagej_metadata is not None:
                header = tif.imagej_metadata
            else:
                header = {}

            # Read the x/y resolution from the first "page" (i.e. the first slice)
            res = tif.pages[0].resolution
            # Resolution to spacing
            header["xy_spacing"] = (1 / res[0], 1 / res[1])

            # Get the axes; this will be something like "ZCYX" where:
            # Z = slices, C = channels, Y = rows, X = columns, S = color(?), Q = empty(?)
            axes = tif.series[0].axes

            # Generate volume
            data = tif.asarray()

        if self.output == "raw":
            return data, header

        # Drop "Q" axes if they have dimenions of 1 (we're assuming these are empty)
        while "Q" in axes and data.shape[axes.index("Q")] == 1:
            data = np.squeeze(data, axis=axes.index("Q"))
            axes = axes.replace("Q", "", 1)  # Only remove the first occurrence
        if "C" in axes:
            # Extract the requested channel from the volume
            data = data.take(self.channel, axis=axes.index("C"))
            axes = axes.replace("C", "")

        # At this point we expect 3D data
        if data.ndim != 3:
            raise ValueError(f'Expected 3D greyscale data, got {data.ndim} ("{axes}").')

        # Swap axes to XYZ order
        order = []
        for a in ("X", "Y", "Z"):
            if a not in axes:
                logger.warning(
                    f'Expected axes to contain "Z", "Y", and "X", got "{axes}". '
                    "Axes will not be automatically reordered."
                )
                order = None
                break
            order.append(axes.index(a))
        if order:
            data = np.transpose(data, order)

        # Try parsing units - this is modelled after the tif files you get from ImageJ
        units = None
        space_units = None
        voxdim = np.array([1, 1, 1], dtype=np.float64)
        if "spacing" in header:
            voxdim[2] = header["spacing"]
        if "xy_spacing" in header:
            voxdim[:2] = header["xy_spacing"]
        if "unit" in header:
            space_units = header["unit"]
            units = [f"{m} {space_units}" for m in voxdim]
        else:
            units = voxdim

        return self.convert_image(data, attrs, header, voxdim, units, space_units)


def read_tiff(
    f: Union[str, Iterable],
    output: Union[Literal["voxels"], Literal["dotprops"], Literal["raw"]] = "voxels",
    channel: int = 0,
    threshold: Optional[Union[int, float]] = None,
    thin: bool = False,
    include_subdirs: bool = False,
    parallel: Union[bool, int] = "auto",
    fmt: str = "{name}.tif",
    limit: Optional[int] = None,
    errors: str = "raise",
    **dotprops_kwargs,
) -> "core.NeuronObject":
    """Create Neuron/List from TIFF file.

    Requires `tifffile` library which is not automatically installed!

    Parameters
    ----------
    f :                 str | iterable
                        Filename(s) or folder. If folder, will import all
                        `.tif` files.
    output :            "voxels" | "dotprops" | "raw"
                        Determines function's output. See Returns for details.
    channel :           int
                        Which channel to import. Ignored if file has only one
                        channel or when `output="raw". Can use e.g. -1 to
                        get the last channel.
    threshold :         int | float | None
                        For `output='dotprops'` only: a threshold to filter
                        low intensity voxels. If `None`, no threshold is
                        applied and all values > 0 are converted to points.
    thin :              bool
                        For `output='dotprops'` only: if True, will thin the
                        point cloud using `skimage.morphology.skeletonize`
                        after thresholding. Requires `scikit-image`.
    include_subdirs :   bool, optional
                        If True and `f` is a folder, will also search
                        subdirectories for `.tif` files.
    parallel :          "auto" | bool | int,
                        Defaults to `auto` which means only use parallel
                        processing if more than 10 TIFF files are imported.
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
                        If `output="voxels"` (default): requires TIFF data to
                        be 3-dimensional voxels. VoxelNeuron will have TIFF file
                        info as `.tiff_header` attribute.
    navis.Dotprops
                        If `output="dotprops"`. Dotprops will contain TIFF
                        header as `.tiff_header` attribute.
    navis.NeuronList
                        If import of multiple TIFF will return NeuronList of
                        Dotprops/VoxelNeurons.
    (image, header)     (np.ndarray, OrderedDict)
                        If `output='raw'` return raw data contained in TIFF
                        file.

    """
    try:
        import tifffile
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "`navis.read_tiff` requires the `tifffile` library:\n"
            "  pip3 install tifffile -U"
        )

    utils.eval_param(
        output, name="output", allowed_values=("raw", "dotprops", "voxels")
    )

    if parallel == "auto":
        # Set a lower threshold of 10 on parallel processing for TIFFs (default is 200)
        parallel = ("auto", 10)

    reader = TiffReader(
        channel=channel,
        output=output,
        threshold=threshold,
        thin=thin,
        fmt=fmt,
        dotprop_kwargs=dotprops_kwargs,
        errors=errors,
    )
    return reader.read_any(f, include_subdirs, parallel, limit=limit)
