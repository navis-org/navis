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

"""Read and write R data (.rda/.rds) files.

This lets us exchange data with R's [natverse](https://natverse.org): navis
objects are translated to and from the representations used by `nat` (neurons,
dotprops, neuronlists, im3d) and `rgl` (mesh3d). All of it goes through
[rdata](https://github.com/vnmabus/rdata), so neither direction needs a running
R session or `rpy2`.

"""

import rdata
import warnings

import matplotlib.colors as mcl
import numpy as np
import pandas as pd

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .. import config, utils, core

__all__ = ['read_rda', 'read_rds', 'write_rda', 'write_rds']

# Set up logging
logger = config.get_logger(__name__)


###############################################################################
#                                  Reading                                    #
###############################################################################

def read_rda(f: str,
             combine: bool = True,
             neurons_only: bool = True) -> 'core.NeuronList':
    """Read objects from nat R data (.rda) file.

    Currently supports parsing neurons, dotprops and mesh3d. Note that this
    does not require a working R installation or `rpy2`.

    Parameters
    ----------
    f :                 str
                        Filepath.
    combine :           bool
                        What to do if there are multiple neuronlists contained
                        in the RDA files. By default, we will combine them into
                        a single NeuronList but you can also choose to keep them
                        as separate neuronlists.
    neurons_only :      bool
                        Whether to only parse and return neurons and dotprops
                        found in the RDA file.

    Returns
    -------
    navis.NeuronList
                        If `combine=True` and `neurons_only=True` returns
                        a single NeuronList with the parsed neurons.
    dict
                        If `combine=False` or `neurons_only=False` returns
                        a dictionary with the original R object name as key and
                        the parsed object as value.

    See Also
    --------
    [`navis.read_rds`][]
                        Read `.rds` files (a single, unnamed object).
    [`navis.write_rda`][]
                        Write navis objects to `.rda` files.

    """
    return _clean_up(_parse(f), combine=combine, neurons_only=neurons_only)


def read_rds(f: str,
             combine: bool = True,
             neurons_only: bool = True):
    """Read objects from nat R data (.rds) file.

    In contrast to `.rda` files, `.rds` files contain a single (unnamed)
    object - typically a neuron, a neuronlist or a mesh.

    Parameters
    ----------
    f :                 str
                        Filepath.
    combine :           bool
                        Only relevant if the file contains a plain (i.e.
                        non-neuronlist) list of neurons: if True, these are
                        combined into a single NeuronList.
    neurons_only :      bool
                        Whether to only return neurons and dotprops found in
                        the file.

    Returns
    -------
    object
                        The parsed object - typically a `NeuronList`.

    See Also
    --------
    [`navis.read_rda`][]
                        Read `.rda` files (multiple named objects).
    [`navis.write_rds`][]
                        Write navis objects to `.rds` files.

    """
    converted = _parse(f)

    # Single neurons are wrapped into a NeuronList for consistency
    if isinstance(converted, core.BaseNeuron):
        return core.NeuronList(converted)

    if isinstance(converted, dict):
        return _clean_up(converted, combine=combine, neurons_only=neurons_only)

    return converted


def _parse(f: str):
    """Parse an R data file and convert it to Python objects."""
    parsed = rdata.parser.parse_file(f)

    with warnings.catch_warnings():
        # rdata warns about every R class it doesn't have a constructor for
        warnings.simplefilter("ignore")
        return rdata.conversion.convert(parsed, CLASS_MAP_EXT)


def _clean_up(converted: dict, combine: bool, neurons_only: bool):
    """Post-process the objects parsed from an R data file."""
    for k, v in converted.items():
        # Convert single neurons to neuronlist
        if isinstance(v, core.BaseNeuron):
            converted[k] = core.NeuronList(v)
        # Give volumes a name
        elif isinstance(v, core.Volume):
            converted[k].name = k

    if combine:
        nl = core.NeuronList([n for n in converted.values() if isinstance(n, core.NeuronList)])
        if nl:
            converted = {k: v for k, v in converted.items() if not isinstance(v, core.NeuronList)}
            converted['neurons'] = nl

    if neurons_only:
        if combine:
            converted = converted.get('neurons', core.NeuronList([]))
        else:
            converted = {k: v for k, v in converted.items() if isinstance(v, core.NeuronList)}

    return converted


def neuronlist_constructor(obj: Any,
                           attrs: Mapping[Union[str, bytes], Any],
                           ) -> 'core.NeuronList':
    """Convert nat neuronlists to navis NeuronLists."""
    # Set IDs
    neurons = []
    for k, n in obj.items():
        if isinstance(n, (core.BaseNeuron, core.NeuronList)):
            n.id = k
            neurons.append(n)
        else:
            logger.warning(f'Unexpected object in neuronlist: {type(n)}. '
                           'Possible parsing error.')

    # Turn into NeuronList
    nl = core.NeuronList(neurons)

    # Now parse extra attributes DataFrame
    df = attrs.get('df', None)
    if isinstance(df, pd.DataFrame):
        # Make sure we have still the correct order
        nl = nl.idx[attrs['names']]

        for col in df:
            # Skip non-string columns
            if not isinstance(col, str):
                continue

            # Skip some columns
            if col.lower() in ['type', 'idx']:
                continue
            if col.lower() in nl[0].__dict__.keys():
                continue

            for n, v in zip(nl, df[col].values):
                # Register
                n._register_attr(col.lower(), v)

    return nl


def dotprops_constructor(obj: Any,
                         attrs: Mapping[Union[str, bytes], Any],
                         ) -> 'core.Dotprops':
    """Convert nat dotprops to navis Dotprops."""
    pts = np.asarray(obj.pop('points'))
    vect = np.asarray(obj.pop('vect'))
    alpha = np.asarray(obj.pop('alpha'))
    k = int(attrs.get('k', 1)[0])
    file = attrs.get('file', [None])[0]

    return core.Dotprops(points=pts, k=k, alpha=alpha, vect=vect, file=file)


def volume_constructor(obj: Any,
                       attrs: Mapping[Union[str, bytes], Any],
                       ) -> 'core.Volume':
    """Convert e.g. mesh3d to navis Volume."""
    if 'vb' in obj and 'it' in obj:
        verts = np.asarray(obj.pop('vb'))[:3, :].T
        faces = np.asarray(obj.pop('it')).T - 1
        return core.Volume(vertices=verts, faces=faces)
    elif 'Vertices' in obj and "Regions" in obj:
        verts = obj['Vertices'][['X', 'Y', 'Z']].values

        # If only one region
        if len(obj['Regions']) == 1:
            region = list(obj['Regions'].keys())[0]
            faces = obj['Regions'][region][['V1', 'V2', 'V3']].values - 1
            return core.Volume(vertices=verts, faces=faces)
        else:
            volumes = []
            for r in obj['Regions']:
                faces = obj['Regions'][r][['V1', 'V2', 'V3']].values - 1
                volumes.append(core.Volume(vertices=verts, faces=faces, name=r))
            return volumes
    else:
        logger.warning('Unable to construct Volume from R object of type '
                       f'"{attrs["class"]}". Returning raw data')
        return obj


def _fix_parents(d: pd.DataFrame, obj: Mapping[str, Any]) -> pd.DataFrame:
    """Make sure a nat node table has a proper parent column.

    nat's authoritative topology is the `SegList`, not `d$Parent`, and the two
    do not always agree: some neurons (e.g. several of the ones in nat's own
    `Cell07PNs`) have the root pointing back at its child which makes the graph
    cyclic and leaves the neuron without a root. Where that's the case, we
    re-derive the parents from the seglists.

    """
    if 'Parent' not in d.columns or 'PointNo' not in d.columns:
        return d

    # A NaN parent is never `isin` the point numbers, so this covers it too
    if (~d.Parent.isin(d.PointNo)).any():
        # We have at least one root - assume all is well
        return d

    # `SubTrees` (only present for fragmented neurons) supersedes `SegList`
    subtrees = obj.get('SubTrees') or ([obj['SegList']] if 'SegList' in obj else None)
    if not subtrees:
        logger.warning('Neuron has no root and no seglist to derive one from - '
                       'the node table will be cyclic.')
        return d

    point_no = d.PointNo.values
    new_parents = np.full(len(d), -1, dtype=np.int64)
    for seglist in subtrees:
        for seg in seglist:
            # Seglists are 1-based vertex indices ordered proximal -> distal
            seg = np.asarray(seg, dtype=np.int64) - 1
            new_parents[seg[1:]] = point_no[seg[:-1]]

    d = d.copy()
    d['Parent'] = new_parents
    return d


def neuron_constructor(obj: Any,
                       attrs: Mapping[Union[str, bytes], Any],
                       ) -> 'core.TreeNeuron':
    """Convert nat neuron/catmaidneuron to navis TreeNeuron."""
    # Data to skip: nat's topology fields, all of which navis derives itself
    DO_NOT_USE = ['nTrees', 'SegList', 'SubTrees', 'NumPoints', 'StartPoint',
                  'EndPoints', 'BranchPoints', 'NumSegs']
    # nat/CATMAID names for things navis calls something else
    RENAME = {'NeuronName': 'name', 'skid': 'id'}

    # Construct neuron from just the nodes
    n = core.TreeNeuron(_fix_parents(obj.pop('d'), obj))

    # R uses diameter, not radius - let's fix that
    if 'radius' in n.nodes.columns:
        has_rad = n.nodes.radius.fillna(0) > 0
        n.nodes.loc[has_rad, 'radius'] = n.nodes.loc[has_rad, 'radius'] / 2

    # If this is a CATMAID neuron, we assume it's in nanometers
    if 'catmaidneuron' in attrs.get('class', []):
        n.units = 'nm'

    # Try attaching other data
    for k, v in obj.items():
        if k in DO_NOT_USE:
            continue
        # R has no scalars, so meta data arrives as length-1 vectors
        if utils.is_iterable(v) and len(v) == 1:
            v = v[0]
        try:
            setattr(n, RENAME.get(k, k), v)
        except BaseException:
            pass

    return n


CLASS_MAP_EXT = {**rdata.conversion.DEFAULT_CLASS_MAP,
                 "neuronlist": neuronlist_constructor,
                 "neuron": neuron_constructor,
                 "mesh3d": volume_constructor,
                 "hxsurf": volume_constructor,
                 "dotprops": dotprops_constructor}


###############################################################################
#                                  Writing                                    #
###############################################################################

# R only knows 32 bit integers
INT32 = np.iinfo(np.int32)

# Class vectors used by nat/rgl
NAT_CLASSES = {
    "neuron": ("neuron", "list"),
    "neuronlist": ("neuronlist", "list"),
    "dotprops": ("dotprops", "list"),
    "seglist": ("seglist", "list"),
    "mesh3d": ("mesh3d", "shape3d"),
    "im3d": ("im3d", "array"),
    "boundingbox": ("boundingbox",),
}


class _RObj:
    """Thin wrapper attaching additional R attributes to a Python object.

    Parameters
    ----------
    value :         dict | list | numpy array
                    The data itself.
    **attributes
                    Extra R attributes (e.g. `class`) to attach to the
                    resulting R object. These are added on top of the
                    attributes `rdata` generates by itself (`names` for dicts,
                    `dim` for multi-dimensional arrays).

    """

    def __init__(self, value: Any, **attributes: Any):
        self.value = value
        self.attributes = attributes


def _robj_constructor(data: _RObj, converter) -> Any:
    """Convert an `_RObj` into an R object, merging in the extra attributes."""
    from rdata.parser import RObjectType
    from rdata.conversion.to_r import build_r_list

    r_obj = converter.convert_to_r_object(data.value)

    if not data.attributes:
        return r_obj

    # `rdata` generates a closed set of attributes by itself: `names` for dicts
    # and `dim` for arrays with more than one dimension. We have to *reuse* the
    # pairlist it built rather than deriving those again: the pairs carry the
    # symbol definitions for those names, and a freshly derived one would be
    # written as a reference to a symbol that never makes it into the file.
    pairs = []
    node = r_obj.attributes
    while node is not None and node.info.type == RObjectType.LIST:
        car, cdr = node.value
        pairs.append((node.tag, car))
        node = cdr

    # ... which means we depend on how `rdata` encodes them. Fail loudly if that
    # ever changes rather than silently writing e.g. an unnamed list.
    expected = int(
        isinstance(data.value, dict)
        or (isinstance(data.value, np.ndarray) and data.value.ndim > 1)
    )
    if len(pairs) != expected:
        raise RuntimeError(
            f"Expected `rdata` to generate {expected} attribute(s) for "
            f"{type(data.value).__name__}, found {len(pairs)}. The installed "
            f"`rdata` ({rdata.__version__}) is likely incompatible."
        )

    for key, value in data.attributes.items():
        pairs.append(
            (converter.convert_to_r_sym(key), converter.convert_to_r_object(value))
        )

    return replace(
        r_obj,
        info=replace(
            r_obj.info,
            object="class" in data.attributes or r_obj.info.object,
            attributes=True,
        ),
        attributes=build_r_list(pairs),
    )


def _classed(value: Any, nat_class: str, **attributes: Any) -> _RObj:
    """Wrap `value` and give it one of nat's class vectors."""
    return _RObj(value, **attributes, **{"class": _str_array(NAT_CLASSES[nat_class])})


def _str_array(x) -> np.ndarray:
    """Turn `x` into a numpy string array (R character vector)."""
    return np.asarray(list(x) if utils.is_iterable(x) else [x], dtype=np.dtype("U"))


def _fits_int32(array: np.ndarray) -> bool:
    """Check whether an integer array fits into R's 32 bit integers."""
    return not array.size or (array.min() >= INT32.min and array.max() <= INT32.max)


def _int_array(x) -> np.ndarray:
    """Turn `x` into an int32 array (R integer vector).

    Values that don't fit into R's 32 bit integers become doubles.

    """
    array = np.asarray(x).reshape(-1)
    return array.astype(np.int32) if _fits_int32(array) else _float_array(array)


def _float_array(x) -> np.ndarray:
    """Turn `x` into a float64 array (R numeric vector).

    `NaN`s are turned into R's `NA_real_` - in navis, `NaN` is used to denote
    missing data which in R is `NA`.

    """
    from rdata.missing import R_FLOAT_NA

    array = np.asarray(x, dtype=np.float64)
    is_nan = np.isnan(array)
    if np.any(is_nan):
        array = array.copy()
        array[is_nan] = R_FLOAT_NA
    return array


def _masked(array: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Mask an int/bool array so the unparser writes `NA` where required."""
    if mask is None or not mask.any():
        return array
    return np.ma.array(array, mask=mask)


def _column_to_array(values) -> np.ndarray:
    """Coerce a column of data into a dtype the R unparser can handle.

    R only knows 32-bit integers and 64-bit doubles, so we have to downcast
    (or, for large integers, upcast) accordingly. Everything that isn't
    numeric or boolean ends up as a character vector. Missing values are
    turned into the corresponding `NA`.

    """
    # Pandas' extension dtypes (nullable ints, arrow-backed strings,
    # categoricals, ...) don't survive `np.asarray` in a useful way, so we
    # unpack them first
    mask = None
    if isinstance(values, (pd.Series, pd.Index)) and isinstance(
        values.dtype, pd.api.extensions.ExtensionDtype
    ):
        kind = getattr(values.dtype, "kind", "O")
        if kind in "iu":
            mask = np.asarray(values.isna())
            values = values.to_numpy(dtype=np.int64, na_value=0)
        elif kind == "b":
            mask = np.asarray(values.isna())
            values = values.to_numpy(dtype=bool, na_value=False)
        elif kind == "f":
            values = values.to_numpy(dtype=np.float64, na_value=np.nan)
        elif not values.isna().any():
            # Strings and categoricals without NAs go straight to a character
            # vector; the slower element-wise path below is only for NAs
            return np.asarray(values, dtype=np.dtype("U"))
        else:
            values = np.asarray(values.astype(object))

    array = np.asarray(values)

    if array.dtype.kind == "f":
        return _float_array(array)
    if array.dtype.kind in "iu":
        if not _fits_int32(array):
            array = array.astype(np.float64)
            return _float_array(np.where(mask, np.nan, array) if mask is not None else array)
        return _masked(array.astype(np.int32), mask)
    if array.dtype.kind == "b":
        return _masked(array, mask)
    if array.dtype.kind in "US":
        return array.astype(np.dtype("U"))

    # Anything else (objects, dates, ...) becomes a character vector
    return np.array([None if pd.isna(x) else str(x) for x in array], dtype=object)


def _dataframe_constructor(data: pd.DataFrame, converter) -> Any:
    """Convert a pandas DataFrame into an R `data.frame`.

    We roll our own instead of using `rdata`'s because it raises on pandas
    extension dtypes (nullable ints, arrow-backed strings, ...). Note that
    pre-sanitizing the frame and delegating does *not* work: under pandas 3
    `future.infer_string` turns object columns back into `ArrowStringArray` on
    DataFrame construction, so the dtypes have to be fixed on the way out.

    """
    from rdata.missing import R_INT_NA
    from rdata.parser import RObjectType
    from rdata.conversion.to_r import build_r_object

    r_value = [
        converter.convert_to_r_object(_column_to_array(data[col]))
        for col in data.columns
    ]

    index = data.index
    if isinstance(index, pd.RangeIndex) and index.step == 1 and index.start in (0, 1):
        # R data.frames are 1-indexed, so a plain range index is simply
        # dropped in favour of R's "default" row names, stored as [NA, -nrow]
        row_names = np.ma.array(
            data=[R_INT_NA, -data.shape[0]], mask=[True, False], fill_value=R_INT_NA
        )
    else:
        row_names = _column_to_array(index)

    r_attributes = converter.convert_to_r_attributes(
        {
            "names": np.array([str(c) for c in data.columns], dtype=np.dtype("U")),
            "class": "data.frame",
            "row.names": row_names,
        }
    )

    return build_r_object(
        RObjectType.VEC, value=r_value, is_object=True, attributes=r_attributes
    )


def _seglist(root, children: Mapping[Any, list], ix: Mapping[Any, int]) -> list:
    """Break one connected component into nat segments.

    Segments run proximal -> distal, are broken at branch points (which two
    consecutive segments share) and are given as 1-based vertex indices.

    """
    segs = []
    stack = [root]
    while stack:
        start = stack.pop()
        for child in children[start]:
            seg = [start, child]
            current = child
            # Extend until we hit a branch point or a leaf
            while len(children[current]) == 1:
                current = children[current][0]
                seg.append(current)
            segs.append([ix[n] for n in seg])
            if len(children[current]) > 1:
                stack.append(current)

    # Isolated node: nat represents this as a single-vertex segment
    return segs or [[ix[root]]]


def _nat_topology(node_ids: np.ndarray, parent_ids: np.ndarray):
    """Compute the topology of a nat neuron.

    This is close to (but not the same as) navis' own `.small_segments` /
    `.subtrees`: nat wants 1-based vertex indices running proximal -> distal,
    grouped per connected component and sorted by size. Building it here in one
    pass measured faster than translating navis' segments, and it keeps nat's
    branch/end point conventions (which differ from navis' node types - nat
    calls a root with several children a branch point) in one place.

    Parameters
    ----------
    node_ids :      (N, ) array
    parent_ids :    (N, ) array
                    Parent for each node; negative for roots.

    Returns
    -------
    subtrees :      list of seglists
                    One seglist per connected component, sorted by number of
                    vertices (descending) as in nat. Indices are 1-based.
    n_children :    dict
                    Number of children for each (1-based) vertex index.

    """
    # `tolist()` because hashing numpy scalars is markedly slower
    node_ids = node_ids.tolist()
    parent_ids = parent_ids.tolist()

    # Map node ID -> 1-based vertex index
    ix = {n: i + 1 for i, n in enumerate(node_ids)}

    children = defaultdict(list)
    roots = []
    for node, parent in zip(node_ids, parent_ids):
        if parent in ix:
            children[parent].append(node)
        else:
            roots.append(node)

    # nat orders subtrees by size (largest = "master" tree). The number of
    # vertices in a component is 1 + the number of edges its segments describe.
    subtrees = sorted(
        (_seglist(root, children, ix) for root in roots),
        key=lambda segs: sum(len(seg) - 1 for seg in segs),
        reverse=True,
    )

    return subtrees, {ix[node]: len(cn) for node, cn in children.items()}


def _neuron2r(x: "core.TreeNeuron") -> _RObj:
    """Convert a TreeNeuron into a nat `neuron`."""
    nodes = x.nodes

    node_ids = nodes.node_id.values
    parent_ids = nodes.parent_id.values

    subtrees, n_children = _nat_topology(node_ids, parent_ids)

    if not subtrees:
        raise ValueError(f"Unable to convert neuron {x.id}: neuron has no nodes.")

    # The master subtree is the first (largest) one and starts at its root
    seglist = subtrees[0]
    root = seglist[0][0]

    # Branch/end points refer to the master subtree only (as in nat)
    master = {i for seg in seglist for i in seg}
    branch_points = sorted(i for i in master if n_children.get(i, 0) > 1)
    ends = [i for i in master if n_children.get(i, 0) == 0]
    # nat also counts the root as an end point if it has a single child
    if n_children.get(root, 0) == 1:
        ends.append(root)
    ends = sorted(ends)

    # Assemble the node table
    d = pd.DataFrame()
    d["PointNo"] = node_ids
    d["Label"] = (
        np.asarray(nodes.label, dtype=np.int32)
        if "label" in nodes.columns
        else np.zeros(len(nodes), dtype=np.int32)
    )
    d["X"] = nodes.x.values
    d["Y"] = nodes.y.values
    d["Z"] = nodes.z.values
    # nat's W is a *diameter*, navis stores radii
    if "radius" in nodes.columns:
        radii = nodes.radius.values.astype(np.float64)
        radii[radii < 0] = np.nan
        d["W"] = radii * 2
    else:
        d["W"] = np.nan
    parents = parent_ids.astype(np.int64)
    parents[parents < 0] = -1
    d["Parent"] = parents

    obj = {
        "NumPoints": _int_array(len(nodes)),
        "StartPoint": _int_array(root),
        "BranchPoints": _int_array(branch_points),
        "EndPoints": _int_array(ends),
        "nTrees": _int_array(len(subtrees)),
        "NumSegs": _int_array(len(seglist)),
        "SegList": _classed([_int_array(seg) for seg in seglist], "seglist"),
    }

    if len(subtrees) > 1:
        obj["SubTrees"] = [
            _classed([_int_array(seg) for seg in st], "seglist") for st in subtrees
        ]

    obj["d"] = d

    # Some meta data - nat uses `NeuronName`
    if getattr(x, "name", None):
        obj["NeuronName"] = _str_array(x.name)
    if x.id is not None:
        obj["id"] = _str_array(str(x.id))
    if getattr(x, "soma", None) is not None:
        soma = utils.make_non_iterable(x.soma)
        if soma is not None:
            obj["soma"] = _int_array(soma)

    if x.has_connectors:
        cn = x.connectors
        # Use the column names the `catmaid` R package expects
        cn_df = pd.DataFrame()
        if "node_id" in cn.columns:
            cn_df["treenode_id"] = cn.node_id.values
        cn_df["connector_id"] = cn.connector_id.values
        if "type" in cn.columns:
            cn_df["prepost"] = cn.type.values
        cn_df["x"] = cn.x.values
        cn_df["y"] = cn.y.values
        cn_df["z"] = cn.z.values
        obj["connectors"] = cn_df

    return _classed(obj, "neuron")


def _dotprops2r(x: "core.Dotprops") -> _RObj:
    """Convert Dotprops into a nat `dotprops` object."""
    points = _float_array(x.points)

    if x._vect is None and not x.k:
        raise ValueError(
            f"Dotprops {x.id} has neither tangent vectors nor a `k` to "
            "calculate them from. Please run `recalculate_tangents()` first."
        )

    # nat's alpha is the "colinearity" of the local neighborhood. If we have
    # neither the values nor a `k` to compute them from (e.g. because the
    # tangent vectors were provided ready-made) we default to 1.
    if x._alpha is not None:
        alpha = x._alpha
    elif x.k:
        alpha = x.alpha
    else:
        alpha = np.ones(len(points))

    xyz = [None, _str_array(["X", "Y", "Z"])]
    obj = {
        "points": _RObj(points, dimnames=xyz),
        "alpha": _float_array(alpha),
        "vect": _RObj(_float_array(x.vect), dimnames=xyz),
    }

    attrs = {"k": _float_array(x.k if x.k else np.nan)}
    if getattr(x, "name", None):
        attrs["NeuronName"] = _str_array(x.name)

    return _classed(obj, "dotprops", **attrs)


def _mesh2r(x) -> _RObj:
    """Convert a MeshNeuron/Volume/trimesh into an rgl `mesh3d` object."""
    vertices = np.asarray(x.vertices, dtype=np.float64)
    faces = np.asarray(x.faces)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Expected an (N, 3) array of triangular faces, got {faces.shape}."
        )

    # rgl stores vertices as a 4 x N matrix of homogeneous coordinates
    vb = np.ones((4, vertices.shape[0]), dtype=np.float64)
    vb[:3, :] = vertices.T
    # ... and faces as a 3 x M matrix of 1-based indices
    it = faces.T.astype(np.int32) + 1

    obj = {
        "vb": vb,
        "it": it,
        "primitivetype": _str_array("triangle"),
        "material": _material(getattr(x, "color", None)),
        "normals": None,
        "texcoords": None,
        "meshColor": _str_array("vertices"),
    }

    return _classed(obj, "mesh3d")


def _material(color) -> Union[list, dict]:
    """Turn a navis color into an rgl material list."""
    if color is None:
        return []

    # Imported here because `navis.plotting` isn't available yet at import time
    from ..plotting.colors import eval_color

    # `eval_color` normalises names ("red"), hex and 0-255 ints to 0-1 RGB(A)
    rgba = eval_color(color, color_range=1)
    if not utils.is_iterable(rgba) or len(rgba) not in (3, 4):
        return []

    material = {"color": _str_array(mcl.to_hex(rgba[:3]))}
    if len(rgba) == 4:
        material["alpha"] = _float_array(rgba[3])

    return material


def _voxels2r(x: "core.VoxelNeuron") -> _RObj:
    """Convert a VoxelNeuron into a nat `im3d` object."""
    # Note that `im3d` is dense, so this materialises the whole grid
    grid = x.grid
    dtype = np.int32 if grid.dtype.kind in "biu" else np.float64
    grid = grid.astype(dtype, copy=False)

    offset = np.asarray(x.offset, dtype=np.float64)
    voxdims = np.asarray(x.units_xyz.magnitude, dtype=np.float64).reshape(-1)
    if voxdims.size == 1:
        voxdims = np.repeat(voxdims, 3)

    dims = np.asarray(grid.shape, dtype=np.float64)
    # nat's bounding box is the centre of the first and last voxel
    bbox = np.vstack([offset, offset + (dims - 1) * voxdims])

    attrs = {
        "BoundingBox": _classed(bbox, "boundingbox"),
        **{
            ax: offset[i] + np.arange(grid.shape[i], dtype=np.float64) * voxdims[i]
            for i, ax in enumerate("xyz")
        },
    }

    return _classed(grid, "im3d", **attrs)


def _neuronlist2r(x: "core.NeuronList", add_metadata: bool = True) -> _RObj:
    """Convert a NeuronList into a nat `neuronlist`."""
    names = _unique_names(x)

    obj = {name: _any2r(n) for name, n in zip(names, x)}

    attrs = {"df": _neuronlist_df(x, names)} if add_metadata else {}

    return _classed(obj, "neuronlist", **attrs)


def _unique_names(x: "core.NeuronList") -> list:
    """Generate unique names for the neurons in a NeuronList."""
    names = [str(n.id) for n in x]

    if len(set(names)) == len(names):
        return names

    logger.warning("Neuron IDs are not unique - appending suffixes to make them so.")
    seen: dict = defaultdict(int)
    unique = []
    for name in names:
        seen[name] += 1
        unique.append(name if seen[name] == 1 else f"{name}.{seen[name] - 1}")
    return unique


def _neuronlist_df(x: "core.NeuronList", names: list) -> pd.DataFrame:
    """Assemble the data.frame attached to a nat neuronlist."""
    # ID and name, plus anything the user registered on the neurons (which is
    # what makes an instance's SUMMARY_PROPS differ from its class')
    extra = [p for n in x for p in n.SUMMARY_PROPS if p not in type(n).SUMMARY_PROPS]
    props = list(dict.fromkeys(["id", "name", *extra]))

    return pd.DataFrame(
        [[getattr(n, p, None) for p in props] for n in x],
        columns=props,
        index=pd.Index(names),
    )


def _any2r(x, add_metadata: bool = True) -> Any:
    """Convert a navis object into its R representation.

    Anything that is not a navis object is passed through untouched - `rdata`
    will take care of dicts, lists, numpy arrays and pandas DataFrames.

    """
    if isinstance(x, core.NeuronList):
        return _neuronlist2r(x, add_metadata=add_metadata)
    elif isinstance(x, core.TreeNeuron):
        return _neuron2r(x)
    elif isinstance(x, core.Dotprops):
        return _dotprops2r(x)
    elif isinstance(x, (core.MeshNeuron, core.Volume)):
        return _mesh2r(x)
    elif isinstance(x, core.VoxelNeuron):
        return _voxels2r(x)
    elif isinstance(x, dict):
        return {str(k): _any2r(v, add_metadata=add_metadata) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [_any2r(v, add_metadata=add_metadata) for v in x]
    return x


def _write(x, filepath, file_type, compression, compresslevel, add_metadata):
    """Convert `x` and unparse it into an R data file.

    We drive `rdata`'s converter and unparser directly rather than going
    through its `write_rds`/`write_rda` helpers: those hand the file to
    `gzip.open` at its default level 9, which for our data costs ~8x the time
    of level 6 to save ~1% of the file size.

    """
    if not hasattr(rdata, "unparser"):
        raise ImportError(
            "Writing R data files requires `rdata` >= 1.0.0, you have "
            f"{rdata.__version__}. Please upgrade: `pip install -U rdata`."
        )

    filepath = Path(filepath).expanduser()
    if filepath.is_dir():
        raise ValueError("`filepath` must be a file, not a directory.")

    r_data = rdata.conversion.convert_python_to_r_data(
        _any2r(x, add_metadata=add_metadata),
        file_type=file_type,
        constructor_dict={
            **rdata.conversion.to_r.DEFAULT_CLASS_MAP,
            pd.DataFrame: _dataframe_constructor,
            _RObj: _robj_constructor,
        },
    )

    if compression is None:
        opener, kwargs = open, {}
    elif compression == "gzip":
        import gzip

        opener, kwargs = gzip.open, {"compresslevel": compresslevel}
    elif compression == "bzip2":
        import bz2

        opener, kwargs = bz2.open, {"compresslevel": compresslevel}
    elif compression == "xz":
        import lzma

        opener, kwargs = lzma.open, {"preset": compresslevel}
    else:
        raise ValueError(f'Unknown compression: "{compression}"')

    with opener(filepath, "wb", **kwargs) as f:
        rdata.unparser.unparse_fileobj(f, r_data, file_type=file_type)


def write_rds(
    x,
    filepath: Union[str, Path],
    compression: Optional[str] = "gzip",
    compresslevel: int = 6,
    add_metadata: bool = True,
) -> None:
    """Write navis object(s) to R data (.rds) file.

    The resulting file contains the corresponding natverse objects (`nat`
    neurons/dotprops/neuronlists and `rgl` mesh3d) and can be loaded in R
    using `readRDS()`. Note that this does not require a working R
    installation or `rpy2`.

    Two things to be aware of: R has no concept of units, so make sure to
    [`convert_units`][navis.BaseNeuron.convert_units] beforehand if the
    R side expects e.g. microns. And nat's `W` column is a *diameter* while
    navis stores radii - we double on the way out and halve on the way back
    in, so this is only relevant if you compare the two directly.

    Parameters
    ----------
    x :             Neuron | NeuronList | Volume | dict | list
                    Object(s) to write. Neurons are converted to their nat
                    equivalents:

                     - `TreeNeuron` -> `nat::neuron`
                     - `Dotprops` -> `nat::dotprops`
                     - `MeshNeuron`/`Volume` -> `rgl::mesh3d`
                     - `VoxelNeuron` -> `nat::im3d`
                     - `NeuronList` -> `nat::neuronlist`

                    Dicts, lists, DataFrames and numpy arrays are written as
                    the corresponding R types.
    filepath :      str | pathlib.Path
                    Destination file.
    compression :   "gzip" | "bzip2" | "xz" | None
                    Compression to use.
    compresslevel : int
                    Compression level, 1 (fastest) to 9 (smallest). The
                    default of 6 matches R's own `saveRDS`.
    add_metadata :  bool
                    Whether to attach a data.frame with neuron meta data
                    (IDs, names, etc.) to neuronlists.

    Returns
    -------
    None

    Notes
    -----
    `rdata` has no streaming API, so the whole dataset is built in memory
    before anything is written - budget roughly 2x the uncompressed file size
    on top of the neurons themselves.

    See Also
    --------
    [`navis.write_rda`][]
                    Write multiple named objects to an `.rda` file.
    [`navis.read_rda`][]
                    Read R data files into navis.

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(3)
    >>> navis.write_rds(nl, '/tmp/neurons.rds')                 # doctest: +SKIP

    In R:

    ``` r
    library(nat)
    nl <- readRDS('/tmp/neurons.rds')
    plot3d(nl)
    ```

    """
    _write(x, filepath, "rds", compression, compresslevel, add_metadata)


def write_rda(
    x,
    filepath: Union[str, Path],
    name: Optional[str] = None,
    compression: Optional[str] = "gzip",
    compresslevel: int = 6,
    add_metadata: bool = True,
) -> None:
    """Write navis object(s) to R data (.rda) file.

    In contrast to `.rds` files, `.rda` files contain *named* objects - i.e.
    they behave like an R workspace. The resulting file contains the
    corresponding natverse objects and can be loaded in R using `load()`.
    Note that this does not require a working R installation or `rpy2`.

    Parameters
    ----------
    x :             dict | Neuron | NeuronList | Volume
                    If dict, keys are used as names for the R objects.
                    Anything else is stored under `name`. See
                    [`navis.write_rds`][] for the type conversions.
    filepath :      str | pathlib.Path
                    Destination file.
    name :          str, optional
                    Name under which to store `x` if `x` is not a dict.
                    Defaults to "neurons".
    compression :   "gzip" | "bzip2" | "xz" | None
                    Compression to use.
    compresslevel : int
                    Compression level, 1 (fastest) to 9 (smallest). The
                    default of 6 matches R's own `save`.
    add_metadata :  bool
                    Whether to attach a data.frame with neuron meta data
                    (IDs, names, etc.) to neuronlists.

    Returns
    -------
    None

    See Also
    --------
    [`navis.write_rds`][]
                    Write a single (unnamed) object to an `.rds` file.
    [`navis.read_rda`][]
                    Read R data files into navis.

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(3)
    >>> lh = navis.example_volume('LH')
    >>> navis.write_rda({'neurons': nl, 'LH': lh},
    ...                 '/tmp/data.rda')                        # doctest: +SKIP

    In R:

    ``` r
    library(nat)
    load('/tmp/data.rda')
    plot3d(neurons)
    ```

    """
    if not isinstance(x, dict):
        x = {name if name else "neurons": x}
    elif name:
        raise ValueError("`name` must not be used when `x` is a dictionary.")

    if not all(isinstance(k, str) for k in x):
        raise ValueError("Keys must be strings when writing to .rda.")

    _write(x, filepath, "rda", compression, compresslevel, add_metadata)
