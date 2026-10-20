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

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from typing import Union, Dict, Optional, Any, IO, Iterable
from zipfile import ZipFile

from .. import config, core, utils
from . import base


__all__ = ["read_nmx", "read_nml"]

# Set up logging
logger = config.get_logger(__name__)

NODE_COLUMNS = ("node_id", "label", "x", "y", "z", "radius", "parent_id")
DEFAULT_PRECISION = 32
DEFAULT_FMT = "{name}.nmx"


class NMLReader(base.BaseReader):
    def __init__(
        self,
        precision: int = DEFAULT_PRECISION,
        attrs: Optional[Dict[str, Any]] = None,
        errors: str = "raise",
    ):
        super().__init__(
            fmt="",
            attrs=attrs,
            file_ext=".nml",
            read_binary=False,
            errors=errors,
            name_fallback="NML",
        )

        int_, float_ = base.parse_precision(precision)
        self._dtypes = {
            "node_id": int_,
            "parent_id": int_,
            "label": "category",
            "x": float_,
            "y": float_,
            "z": float_,
            "radius": float_,
        }

    @base.handle_errors
    def read_buffer(
        self, f: IO, attrs: Optional[Dict[str, Any]] = None
    ) -> "core.NeuronObject":
        """Read .nml buffer into Skeleton(s). See `read_nml`."""
        return self.read_nml(f.read(), attrs=attrs)

    def read_nml(
        self, f: Union[str, bytes], attrs: Optional[Dict[str, Any]] = None
    ) -> "core.NeuronObject":
        """Read .nml data into Skeleton(s).

        NML files are XML-encoded files containing one or more trees
        (`<thing>` elements). Each tree becomes a separate Skeleton.

        Parameters
        ----------
        f :         str | bytes
                    NML data.
        attrs :     dict | None
                    Arbitrary attributes to include in the Skeleton.

        Returns
        -------
        core.Skeleton | core.NeuronList
                    A single Skeleton if the file contains one tree, else a
                    NeuronList with one Skeleton per tree.
        """
        root = ET.fromstring(f)

        things = root.findall("thing")
        if not things:
            raise ValueError("NML file does not contain any trees (<thing> elements)")

        # Comments are stored globally and reference nodes by ID
        comments = {
            int(c.get("node")): c.get("content", "")
            for c in root.iterfind("comments/comment")
        }

        base_props = {"name": "NML", "origin": "nml"}
        units = self._parse_units(root)
        if units is not None:
            base_props["units"] = units

        neurons = []
        for thing in things:
            nodes = self._parse_nodes(thing)

            # The tree's own name/id are defaults that the filename or user
            # can override - unless there are multiple trees in the file in
            # which case they are the only thing telling the trees apart
            tree_props = {}
            if thing.get("name"):
                tree_props["name"] = thing.get("name")
            if thing.get("id") is not None:
                tree_props["id"] = int(thing.get("id"))
            props = {**base_props, **tree_props}
            if thing.get("groupId"):
                props["group_id"] = thing.get("groupId")

            override = tree_props if len(things) > 1 else {}

            tags = {}
            for i in nodes.node_id.tolist():
                if i in comments:
                    tags.setdefault(comments[i], []).append(i)
            if tags:
                override = {**override, "tags": tags}

            neurons.append(
                core.Skeleton(
                    nodes, **(self._make_attributes(props, attrs, override))
                )
            )

        if len(neurons) == 1:
            return neurons[0]
        return core.NeuronList(neurons)

    def _parse_units(self, root: ET.Element) -> Optional[list]:
        """Extract voxel size from the <parameters><scale> element."""
        scale = root.find("parameters/scale")
        if scale is None:
            return None
        unit = scale.get("unit", "nanometer")
        return [f"{float(scale.get(d))} {unit}" for d in ("x", "y", "z")]

    def _parse_nodes(self, thing: ET.Element) -> pd.DataFrame:
        """Parse a single <thing> into a node table with parent IDs."""
        records = [n.attrib for n in thing.iterfind("nodes/node")]
        if records:
            nodes = pd.DataFrame.from_records(records)
        else:
            nodes = pd.DataFrame(columns=["id", "x", "y", "z", "radius"])
        nodes.rename({"id": "node_id"}, axis=1, inplace=True)
        nodes = nodes.astype(
            {k: v for k, v in self._dtypes.items() if k in nodes.columns}
        )
        nodes.sort_values("node_id", inplace=True)
        nodes.reset_index(drop=True, inplace=True)

        # Translate edges into index space, dropping any that reference nodes
        # not in this tree
        ids = nodes.node_id.values
        edges = np.array(
            [
                (int(e.get("source")), int(e.get("target")))
                for e in thing.iterfind("edges/edge")
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        edges = edges[np.isin(edges, ids).all(axis=1)]
        edges = np.searchsorted(ids, edges).astype(np.int32)

        # Root each connected component at its lowest node ID (typically the
        # first node placed) and orient edges away from it
        parents = utils.fastcore.parents_from_edges(edges, len(ids))[0]
        nodes["parent_id"] = np.where(parents >= 0, ids[parents], -1).astype(
            self._dtypes["parent_id"]
        )

        return nodes


class NMXReader(NMLReader):
    """This is a version of the NML file reader that reads from zipped archives."""

    def __init__(
        self,
        precision: int = DEFAULT_PRECISION,
        attrs: Optional[Dict[str, Any]] = None,
        errors: str = "raise",
    ):
        super().__init__(precision=precision, errors=errors, attrs=attrs)

        # Overwrite some of the settings
        self.read_binary = True
        self.file_ext = ".nmx"
        self.name_fallback = "NMX"

    @base.handle_errors
    def read_buffer(
        self, f: IO, attrs: Optional[Dict[str, Any]] = None
    ) -> "core.NeuronObject":
        """Read .nmx buffer into Skeleton(s).

        NMX files are zip files containing XML-encoded .nml files.

        Parameters
        ----------
        f :         IO
                    Readable buffer (must be bytes).
        attrs :     dict | None
                    Arbitrary attributes to include in the Skeleton.

        Returns
        -------
        core.Skeleton | core.NeuronList
        """
        if not isinstance(f.read(0), bytes):
            raise ValueError(f'Expected bytes, got "{type(f.read(0))}"')

        zip = ZipFile(f)
        for f in zip.filelist:
            if f.filename.endswith(".nml") and "skeleton" in f.filename:
                attrs["file"] = f.filename
                attrs["id"] = f.filename.split("/")[0]
                return self.read_nml(zip.read(f), attrs=attrs)
        logger.warning(
            f'Skipped "{f.filename.split("/")[0]}.nmx": failed to ' 'import skeleton.'
        )


def read_nmx(
    f: Union[str, pd.DataFrame, Iterable],
    include_subdirs: bool = False,
    parallel: Union[bool, int] = "auto",
    precision: int = 32,
    limit: Optional[int] = None,
    errors: str = "raise",
    **kwargs,
) -> "core.NeuronObject":
    """Read NMX files into Neuron/Lists.

    NMX is an xml-based format used by pyKNOSSOS.
    See e.g. [here](https://doi.org/10.5281/zenodo.58985) for a data dump
    of neurons from Wanner et al. (2016).

    Parameters
    ----------
    f :                 str
                        Filename or folder. If folder, will import all `.nmx`
                        files.
    include_subdirs :   bool, optional
                        If True and `f` is a folder, will also search
                        subdirectories for `.nmx` files.
    parallel :          "auto" | bool | int
                        Defaults to `auto` which means only use parallel
                        processing if more than 200 files are imported. Spawning
                        and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the
                        number of cores (otherwise defaults to
                        half the available cores).
    precision :         int [8, 16, 32, 64] | None
                        Precision for data. Defaults to 32 bit integers/floats.
                        If `None` will let pandas infer data types - this
                        typically leads to higher than necessary precision.
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
    **kwargs
                        Keyword arguments passed to the construction of
                        `navis.Skeleton`. You can use this to e.g. set
                        meta data.

    Returns
    -------
    navis.NeuronList

    See Also
    --------
    [`navis.read_nml`][]
                        Read NML file(s).

    """
    reader = NMXReader(precision=precision, errors=errors, attrs=kwargs)
    # Read neurons
    neurons = reader.read_any(
        f, parallel=parallel, limit=limit, include_subdirs=include_subdirs
    )

    # Failed reads will produce empty neurons which we need to remove
    if isinstance(neurons, core.NeuronList):
        neurons = neurons[neurons.has_nodes]

    return neurons


def read_nml(
    f: Union[str, pd.DataFrame, Iterable],
    include_subdirs: bool = False,
    parallel: Union[bool, int] = "auto",
    precision: int = 32,
    limit: Optional[int] = None,
    **kwargs,
) -> "core.NeuronObject":
    """Read xml-based NML files into Neuron/Lists.

    Parameters
    ----------
    f :                 str
                        Filename or folder. If folder, will import all `.nml`
                        files.
    include_subdirs :   bool, optional
                        If True and `f` is a folder, will also search
                        subdirectories for `.nml` files.
    parallel :          "auto" | bool | int
                        Defaults to `auto` which means only use parallel
                        processing if more than 200 files are imported. Spawning
                        and joining processes causes overhead and is
                        considerably slower for imports of small numbers of
                        neurons. Integer will be interpreted as the
                        number of cores (otherwise defaults to
                        half the available cores).
    precision :         int [8, 16, 32, 64] | None
                        Precision for data. Defaults to 32 bit integers/floats.
                        If `None` will let pandas infer data types - this
                        typically leads to higher than necessary precision.
    limit :             int | str | slice | list, optional
                        When reading from a folder or archive you can use this
                        parameter to restrict the which files read:
                         - if an integer, will read only the first `limit` NML files
                           (useful to get a sample from a large library of skeletons)
                         - if a string, will interpret it as filename (regex) pattern
                           and only read files that match the pattern; e.g. `limit='.*_R.*'`
                           will only read files that contain `_R` in their filename
                         - if a slice (e.g. `slice(10, 20)`) will read only the files in
                           that range
                         - a list is expected to be a list of filenames to read from
                           the folder/archive
    **kwargs
                        Keyword arguments passed to the construction of
                        `navis.Skeleton`. You can use this to e.g. set
                        meta data.

    Returns
    -------
    navis.NeuronList

    See Also
    --------
    [`navis.read_nmx`][]
                        Read NMX files (collections of NML files).

    """
    reader = NMLReader(precision=precision, attrs=kwargs)
    # Read neurons
    neurons = reader.read_any(
        f, parallel=parallel, limit=limit, include_subdirs=include_subdirs
    )

    return neurons
