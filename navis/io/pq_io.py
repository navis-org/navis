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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
import uuid

import pandas as pd
import numpy as np

from collections import namedtuple
from pathlib import Path
from typing import List, Union, Optional

from .. import config, core, utils

__all__ = ["read_parquet", "write_parquet", "scan_parquet"]

# Set up logging
logger = config.get_logger(__name__)

# Node table columns we write. Anything else is dropped.
SKELETON_COLUMNS = (
    "node_id",
    "x",
    "y",
    "z",
    "radius",
    "parent_id",
    "label",
    "neuron",
)
# Columns a connector table must contain. Any additional columns (e.g. "roi" or
# "confidence") are written as-is.
CONNECTOR_COLUMNS = ("connector_id", "node_id", "type", "x", "y", "z", "neuron")
META_DATA = ("name", "units", "soma")  # meta data to write for each neuron

INT_TYPES = (int, np.int8, np.int16, np.int32, np.int64)

# Formats `write_parquet` knows how to produce
FORMATS = ("navis", "neurarrow")

# Version of the neurarrow spec (https://neurarrow.readthedocs.io) we write
NEURARROW_VERSION = "0.2.1"
# Extensions we use for data neurarrow itself does not model. Note that the
# `net.clbarnes.connector` extension names its metadata key after the schema it
# adds (`net.clbarnes.connectors`), not after the extension itself.
NEURARROW_CONNECTOR_SCHEMA = "net.clbarnes.connectors"
NEURARROW_CONNECTOR_EXT_VERSION = "0.2"
NEURARROW_SWC_EXT = "net.clbarnes.swc"
NEURARROW_SWC_EXT_VERSION = "0.1"

# Maps neurarrow fields onto their navis equivalents
NEURARROW_COLUMNS = {
    "sample_id": "node_id",
    "fragment_id": "neuron",
    "tangent_x": "vect_x",
    "tangent_y": "vect_y",
    "tangent_z": "vect_z",
    "colinearity": "alpha",
    f"{NEURARROW_SWC_EXT}:type_id": "label",
}
NAVIS_COLUMNS = {v: k for k, v in NEURARROW_COLUMNS.items()}

# Everything the neurarrow writers need that isn't derived from the neurons:
# the context all IDs are unique in, the file-level `unit` and the factor to
# bring coordinates into that unit.
NeurarrowSpec = namedtuple("NeurarrowSpec", ["context", "unit", "scale"])


def _import_pyarrow(action: str):
    """Import pyarrow or raise an informative error."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"{action} parquet files requires the pyarrow library:\n"
            " pip3 install pyarrow"
        )
    return pa, pq


def _connectors_filepath(filepath: Union[str, Path]) -> Path:
    """Path of the sidecar file holding the connector table.

    E.g. `neurons.parquet` -> `neurons.connectors.parquet`.
    """
    filepath = Path(filepath)
    return filepath.with_name(f"{filepath.stem}.connectors{filepath.suffix}")


def _decode_meta(metadata) -> dict:
    """Decode a parquet file's byte-encoded key/value metadata."""
    if not metadata:
        return {}
    try:
        return {k.decode(): v.decode() for k, v in metadata.items()}
    except BaseException:
        logger.warning("Unable to decode meta data for parquet file")
        return {}


def _is_neurarrow(meta: dict, columns=()) -> bool:
    """Check whether a file follows the neurarrow spec.

    `version` and `context` are required by neurarrow's base schema, and
    `fragment_id` by every concrete schema we can read.
    """
    if "version" in meta and "context" in meta:
        return True
    return "fragment_id" in columns


def _parse_meta(meta: dict, neurarrow: bool):
    """Split a file's metadata into per-neuron and file-level properties.

    Returns
    -------
    neuron_meta :   dict
                    Neuron properties, grouped as `{id: {property: value}}`.
                    Grouping here means the extractors don't have to scan the
                    whole file's meta data once per neuron.
    file_meta :     dict
                    Properties that apply to the file as a whole. For neurarrow
                    files this is where units and the dotprops `k` live.

    """
    neuron_meta = {}
    file_meta = {}
    for k, v in meta.items():
        # Skip private and pyarrow-internal properties
        if k.startswith("_") or k in ("ARROW:schema", "pandas"):
            continue
        # neurarrow namespaces per-fragment properties as `frag:{id}:{property}`
        if neurarrow:
            if not k.startswith("frag:"):
                file_meta[k] = v
                continue
            k = k[len("frag:") :]
        # Properties without a key are file-level
        if ":" not in k:
            file_meta[k] = v
            continue
        id, prop = k.split(":", 1)
        neuron_meta.setdefault(id, {})[prop] = v

    return neuron_meta, file_meta


def _id_column(columns) -> Optional[str]:
    """Which column holds the neuron IDs (if any)."""
    for col in ("fragment_id", "neuron"):
        if col in columns:
            return col
    return None


def _file_defaults(file_meta: dict) -> dict:
    """Turn file-level neurarrow metadata into per-neuron navis properties.

    neurarrow tracks units and the dotprops neighbourhood size for the whole
    file, navis tracks them per neuron.
    """
    defaults = {}
    for src, dest in (("unit", "units"), ("neighborhood_size", "k")):
        # Note that neurarrow uses an empty `unit` for arbitrary units
        if file_meta.get(src):
            defaults[dest] = file_meta[src]
    return defaults


def scan_parquet(file: Union[str, Path]):
    """Scan parquet file.

    Parameters
    ----------
    file :              str
                        File to be scan.

    Returns
    -------
    pd.DataFrame
                        Summary of file's content.

    See Also
    --------
    [`navis.write_parquet`][]
                        Export neurons as parquet files.
    [`navis.read_parquet`][]
                        Read parquet file into neurons.

    Examples
    --------
    See [`navis.write_parquet`][] for examples.

    """
    _, pq = _import_pyarrow("Reading")

    f = Path(file).expanduser()
    if not f.is_file():
        raise FileNotFoundError(f'File "{f}" does not exist.')

    metadata = pq.read_metadata(f)
    meta = _decode_meta(metadata.metadata)
    id_col = _id_column(metadata.schema.names)

    neuron_meta, file_meta = _parse_meta(
        meta, _is_neurarrow(meta, metadata.schema.names)
    )
    defaults = _file_defaults(file_meta)

    # Compile one record per neuron
    records = {id: {**defaults, **props, "id": id} for id, props in neuron_meta.items()}

    if not records and id_col:
        # No per-neuron meta data (e.g. a neurarrow file written by some other
        # tool). Fall back to reading just the ID column - that's a lot more
        # than we'd like to read here but still far less than the whole table.
        logger.debug(f"No neuron meta data in {f} - scanning `{id_col}` column.")
        ids = pq.read_table(f, columns=[id_col])[id_col].unique().to_pylist()
        records = {i: dict(defaults, id=i) for i in ids}

    # Turn into DataFrame
    df = pd.DataFrame.from_records(list(records.values()))

    if df.empty:
        return df

    # Move ID column to front
    ids = df["id"]
    df.drop(labels=["id"], axis=1, inplace=True)
    df.insert(0, "id", ids)

    # The IDs are always stored as strings but the column might be integers
    if id_col and not pd.api.types.is_numeric_dtype(df["id"]):
        schema = metadata.schema.column(metadata.schema.names.index(id_col))
        if schema.physical_type.lower() in ("int", "int64", "int32", "int16", "int8"):
            df["id"] = df["id"].astype(int)

    return df


def read_parquet(
    f: Union[str, Path],
    read_meta: bool = True,
    limit: Optional[int] = None,
    subset: Optional[List[Union[str, int]]] = None,
    read_connectors: bool = True,
    progress=True,
) -> "core.NeuronObject":
    """Read parquet file into Neuron/List.

    Reads both navis' own format (see
    [here](https://github.com/navis-org/navis/blob/master/navis/io/pq_io.md)
    for specifications) and files following the
    [neurarrow](https://neurarrow.readthedocs.io) spec - the format is detected
    from the file's metadata.

    Parameters
    ----------
    f :                 str
                        File to be read.
    read_meta :         bool
                        Whether to read neuron meta data stored in the parquet
                        file (e.g. name or units). Defaults to True but can be
                        switched off in case there are any issues.
    limit :             int, optional
                        If reading from a file containing multiple neurons you
                        can use this parameter to read only the first `limit`
                        neurons. Useful if wanting to get a sample from a large
                        library of neurons.
    subset :            str | int | list thereof
                        If the parquet file contains multiple neurons you can
                        use this to select the IDs of the neurons to load. Only
                        works if the parquet file actually contains multiple
                        neurons.
    read_connectors :   bool
                        Whether to also read the connector table from the
                        sidecar file (e.g. `neurons.connectors.parquet` next to
                        `neurons.parquet`) if it exists.

    Returns
    -------
    navis.Skeleton/Dotprops
                        If parquet file contains a single neuron.
    navis.NeuronList
                        If parquet file contains multiple neurons or if
                        `limit`/`subset` were used.

    See Also
    --------
    [`navis.write_parquet`][]
                        Export neurons as parquet files.
    [`navis.scan_parquet`][]
                        Scan parquet file for its contents.

    Examples
    --------
    See [`navis.write_parquet`][] for examples.

    """
    f = Path(f).expanduser()
    if not f.is_file():
        raise FileNotFoundError(f'File "{f}" does not exist.')

    _, pq = _import_pyarrow("Reading")

    # Peek at the schema to work out the format and which column holds the IDs
    schema = pq.read_schema(f)
    file_meta_raw = _decode_meta(schema.metadata)
    neurarrow = _is_neurarrow(file_meta_raw, schema.names)
    id_col = _id_column(schema.names)

    # Extract meta data (will be byte encoded)
    neuron_meta, file_meta = _parse_meta(file_meta_raw if read_meta else {}, neurarrow)

    # For neurarrow files, `frag:{fragment_id}:id` maps the (uint64) fragment ID
    # back onto the original neuron ID
    frag_to_id = (
        {k: v["id"] for k, v in neuron_meta.items() if "id" in v} if neurarrow else {}
    )
    id_to_frag = {v: k for k, v in frag_to_id.items()}

    if limit is not None:
        if subset not in (None, False):
            raise ValueError(
                "You can provide either a `subset` or a `limit` but not both."
            )
        scan = scan_parquet(f)
        if scan.empty:
            raise ValueError(
                f"Parquet file {f} either does not contain any neurons or meta data could not be read."
            )
        subset = scan.id.values[:limit]

    if isinstance(subset, (pd.Series)):
        subset = subset.values

    filtered = subset is not None and subset is not False

    # Read the table
    if not filtered:
        table = pq.read_table(f)
    elif isinstance(subset, (str, int)):
        table = pq.read_table(f, filters=[(id_col, "=", _to_frag(subset, id_to_frag))])
    elif isinstance(subset, (list, np.ndarray)):
        table = pq.read_table(
            f, filters=[(id_col, "in", [_to_frag(s, id_to_frag) for s in subset])]
        )
    else:
        raise TypeError(f'`subset` must be int, str or iterable, got "{type(subset)}')

    # Convert to pandas
    table = table.to_pandas()

    if neurarrow:
        table, neuron_meta = _neurarrow_to_navis(table, neuron_meta, frag_to_id)

    # The mapping of (neuron, node_id) -> sample_id we need to read neurarrow
    # connectors back in
    samples = table["sample_id"] if "sample_id" in table.columns else None
    table = table.drop(columns=["sample_id"], errors="ignore")

    # Check if we're doing skeletons or dotprops
    if "node_id" in table.columns:
        _extract_neuron = _extract_skeleton
    elif "vect_x" in table.columns:
        _extract_neuron = _extract_dotprops
    else:
        raise TypeError(
            f"Unable to extract neuron from parquet file with columns {table.columns}"
        )

    def _meta_for(id):
        """Meta data for a single neuron, on top of the file-wide defaults."""
        props = dict(_file_defaults(file_meta), **neuron_meta.get(str(id), {}))
        # Drop "Nones"
        return {k: v for k, v in props.items() if v != "None"}

    # If this is a single neuron
    if "neuron" not in table.columns:
        # Generic ID as fallback if we don't have metadata
        id = next((v["id"] for v in neuron_meta.values() if "id" in v), "0")
        neurons = [_extract_neuron(table, id, _meta_for(id))]
    else:
        neurons = []
        # Note: this could be done in threads
        for id, this_table in config.tqdm(
            table.groupby("neuron"),
            disable=not progress,
            leave=False,
            desc="Making nrn",
        ):
            this_table = this_table.drop("neuron", axis=1)
            neurons.append(_extract_neuron(this_table, id, _meta_for(id)))

    if read_connectors:
        _read_connectors(f, neurons, table, samples, filtered=filtered)

    # Return a single neuron only if that's all the file contains
    if len(neurons) == 1 and not filtered:
        return neurons[0]
    return core.NeuronList(neurons)


def _to_frag(id, id_to_frag):
    """Translate a neuron ID into the fragment ID used in a neurarrow file."""
    frag = id_to_frag.get(str(id))
    if frag is None:
        return id
    return _try_int(frag)


def _neurarrow_to_navis(table, neuron_meta, frag_to_id):
    """Translate a table read from a neurarrow file into navis' own layout.

    Note that we deliberately keep the `sample_id` column around: it's what
    connectors reference.
    """
    # Everything but `sample_id` maps 1:1 onto a navis column
    table = table.rename(
        columns={k: v for k, v in NEURARROW_COLUMNS.items() if k != "sample_id"}
    )

    # Only skeletons have parents - for dotprops the sample IDs are meaningless
    if "parent_id" in table.columns:
        if "attr:node_id" in table.columns:
            # We had to remap node IDs on write - restore the originals
            mapper = pd.Series(
                table["attr:node_id"].values, index=table["sample_id"].values
            )
            parents = mapper.reindex(table["parent_id"].values).values
            node_id = table["attr:node_id"].values
            table = table.drop(columns=["attr:node_id"])
        else:
            # Sample IDs double as node IDs
            parents = table["parent_id"].values
            node_id = table["sample_id"].values
        table["node_id"] = _to_int64(node_id)
        # Roots are encoded as a null parent
        table["parent_id"] = (
            pd.to_numeric(pd.Series(parents), errors="coerce")
            .fillna(-1)
            .astype(np.int64)
            .values
        )

    # Map fragment IDs back onto the original neuron IDs
    if "neuron" in table.columns:
        mapper = {_try_int(k): _try_int(v) for k, v in frag_to_id.items()}
        if mapper and any(k != v for k, v in mapper.items()):
            table["neuron"] = table["neuron"].map(mapper).fillna(table["neuron"])
        elif np.issubdtype(table["neuron"].dtype, np.unsignedinteger):
            # uint64 IDs are unwieldy downstream
            table["neuron"] = table["neuron"].astype(np.int64)

    # Re-key the metadata from fragment IDs onto neuron IDs
    neuron_meta = {frag_to_id.get(id, id): props for id, props in neuron_meta.items()}

    # Drop any remaining attribute/extension columns we don't understand
    drop = [c for c in table.columns if ":" in c]
    if drop:
        logger.debug(f"Dropping unsupported neurarrow columns: {', '.join(drop)}")
        table = table.drop(columns=drop)

    return table, neuron_meta


def _to_int64(x):
    """Cast a column of (potentially unsigned) IDs to int64."""
    return np.asarray(x).astype(np.int64)


def _extract_skeleton(nodes, id, meta):
    """Extract a single skeleton."""
    meta = dict(meta, id=id)
    meta.pop("k", None)  # `k` only applies to dotprops

    # The soma needs to be added separately because it is typically stored as
    # list (e.g. [0]) which the Skeleton initialisation doesn't like
    soma = _parse_soma(meta.pop("soma", None))

    # Make the neuron
    tn = core.Skeleton(nodes, **meta)
    tn.soma = soma if soma else None

    return tn


def _parse_soma(soma):
    """Parse the soma back out of its string representation."""
    if soma is None:
        return None
    # Parse a list string (e.g. "[1]") back into a list
    if soma.startswith("["):
        return [_try_int(i.strip()) for i in soma[1:-1].split(",")]
    return _try_int(soma)


def _extract_dotprops(table, id, meta):
    """Extract a single dotprop."""
    meta = dict(meta, id=id)
    meta["k"] = _try_int(meta.get("k", 5))  # <- set a default K of 5

    if "vect_x" in table:
        meta["vect"] = table[["vect_x", "vect_y", "vect_z"]].values
    if "alpha" in table:
        meta["alpha"] = table["alpha"].values

    return core.Dotprops(table[["x", "y", "z"]].values, **meta)


def _try_int(x):
    """Try converting `x` into an integer."""
    try:
        return int(x)
    except (ValueError, TypeError):
        return x


def write_parquet(
    x: "core.NeuronObject",
    filepath: Union[str, Path],
    write_meta: bool = True,
    write_connectors: bool = True,
    format: str = "navis",
    context: Optional[str] = None,
) -> None:
    """Write Skeleton(s) or Dotprops to parquet file.

    See [here](https://github.com/navis-org/navis/blob/master/navis/io/pq_io.md)
    for format specifications.

    Connectors are written to a sidecar file next to `filepath` - e.g.
    `neurons.parquet` is accompanied by `neurons.connectors.parquet`.
    [`navis.read_parquet`][] picks that file up automatically.

    Note that MeshNeurons and VoxelNeurons are not supported.

    Parameters
    ----------
    x :                 Skeleton | Dotprop | NeuronList thereof
                        Neuron(s) to save. If NeuronList must contain either
                        only Skeletons or only Dotprops.
    filepath :          str | pathlib.Path
                        Destination for the file.
    write_meta :        bool | list of str
                        Whether to also write neuron properties to file. By
                        default this is `.name`, `.units` and `.soma`. You can
                        change which properties are written by providing them as
                        list of strings.
    write_connectors :  bool
                        Whether to write the neurons' connector tables to the
                        sidecar file. If False, an existing sidecar file for
                        this `filepath` is removed so it can't go stale.
    format :            "navis" | "neurarrow"
                        Which format specs to write:
                          - `navis` (default) is navis' own format and the only
                            one that round-trips without loss
                          - `neurarrow` follows the
                            [neurarrow](https://neurarrow.readthedocs.io) spec
                            for interoperability with other tools. This requires
                            all neurons to share the same units (and, for
                            dotprops, the same `k`), and connectors are written
                            using the `net.clbarnes.connector` extension which
                            drops any extra columns (e.g. "roi").
    context :           str, optional
                        Only for `format="neurarrow"`: identifier for the
                        context in which all IDs in this file are unique. If not
                        provided, a random UUID is generated. Pass the same
                        context when writing files that belong to one dataset.

    See Also
    --------
    [`navis.read_parquet`][]
                        Import skeleton from parquet file.
    [`navis.scan_parquet`][]
                        Scan parquet file for its contents.

    Examples
    --------
    Save a bunch of skeletons:

    >>> import navis
    >>> nl = navis.example_neurons(3, kind='skeleton')
    >>> navis.write_parquet(nl, tmp_dir / 'skeletons.parquet')

    Inspect that file's content

    >>> import navis
    >>> contents = navis.scan_parquet(tmp_dir / 'skeletons.parquet')
    >>> contents                                                # doctest: +SKIP
               id        units       name    soma
    0   722817260  8 nanometer  DA1_lPN_R     NaN
    1  1734350908  8 nanometer  DA1_lPN_R     [6]
    2  1734350788  8 nanometer  DA1_lPN_R  [4177]

    Read the skeletons back in

    >>> import navis
    >>> nl = navis.read_parquet(tmp_dir / 'skeletons.parquet')
    >>> len(nl)
    3

    Connectors travel in a sidecar file and come back automatically

    >>> import navis
    >>> nl[0].n_connectors == navis.read_parquet(tmp_dir / 'skeletons.parquet')[0].n_connectors
    True

    Write to the neurarrow spec instead

    >>> import navis
    >>> navis.write_parquet(nl, tmp_dir / 'skeletons.na.parquet', format='neurarrow')
    >>> len(navis.read_parquet(tmp_dir / 'skeletons.na.parquet'))
    3

    """
    filepath = Path(filepath).expanduser()

    # Fail with an informative error here rather than on the bare imports below
    _import_pyarrow("Writing")

    if format not in FORMATS:
        raise ValueError(f'`format` must be one of {FORMATS}, got "{format}"')

    # Make sure inputs are only Skeletons or Dotprops. Each type comes with a
    # pair of converters: one for our own format, one for neurarrow.
    if isinstance(x, core.NeuronList):
        types = x.types
        if types == (core.Skeleton,):
            converters = (_skeletons_to_table, _skeletons_to_neurarrow)
        elif types == (core.Dotprops,):
            converters = (_dotprops_to_table, _dotprops_to_neurarrow)
        else:
            raise TypeError(
                "Can only write either Skeletons or Dotprops to "
                f"parquet but NeuronList contains {types}"
            )
        if x.is_degenerated:
            raise ValueError("NeuronList must not contain non-unique IDs")
    elif isinstance(x, core.Skeleton):
        converters = (_skeletons_to_table, _skeletons_to_neurarrow)
    elif isinstance(x, core.Dotprops):
        converters = (_dotprops_to_table, _dotprops_to_neurarrow)
    else:
        raise TypeError(
            f'Can only write Skeletons or Dotprops to parquet, got "{type(x)}"'
        )

    if format == "neurarrow":
        unit, scale = _neurarrow_unit(x)
        spec = NeurarrowSpec(context or uuid.uuid4().hex, unit, scale)
    else:
        spec = None

    samples = _write_neurons(x, filepath, write_meta, spec, *converters)

    _write_connectors(
        x,
        filepath=filepath,
        write_meta=write_meta,
        write_connectors=write_connectors,
        spec=spec,
        samples=samples,
    )


def _write_table(table, filepath, metadata):
    """Attach metadata to a pyarrow table and write it to file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [table.schema.field(i) for i in range(len(table.schema))], metadata=metadata
    )
    return pq.write_table(table.cast(schema), filepath)


def _write_neurons(x, filepath, write_meta, spec, to_table, to_neurarrow):
    """Write neurons to parquet file using the given pair of converters.

    Returns the (neuron, node_id) -> sample_id mapping when writing neurarrow,
    which is what the connector writer needs to reference nodes.
    """
    import pyarrow as pa

    if spec is not None:
        table, samples, metadata = to_neurarrow(x, write_meta, spec)
    else:
        table = pa.Table.from_pandas(to_table(x), preserve_index=False)
        metadata = _compile_meta(x, write_meta=write_meta)
        samples = None

    _write_table(table, filepath, metadata)

    return samples


def _skeletons_to_table(x):
    """Turn skeletons into a table we can write to parquet."""
    # Make sure we're working with a list, not a single neuron
    x = core.NeuronList(x)

    # Generate node table
    nodes = x.nodes[x.nodes.columns[np.isin(x.nodes.columns, SKELETON_COLUMNS)]]

    return nodes


def _dotprops_to_table(x):
    """Turn dotprops into a table we can write to parquet.

    Examples
    --------
    We test writing dotprops here instead of in the main function

    >>> import navis
    >>> nl = navis.example_neurons(3, kind='skeleton')
    >>> dp = navis.make_dotprops(nl, k=5)
    >>> navis.write_parquet(dp, tmp_dir / 'dotprops.parquet')
    >>> dp2 = navis.read_parquet(tmp_dir / 'dotprops.parquet')
    >>> assert len(dp) == len(dp2)
    >>> assert all([i in dp2.id for i in dp.id])

    """
    # Make sure we're working with a list, not a single neuron
    x = core.NeuronList(x)

    # Generate table
    table = pd.DataFrame(np.vstack(x.points), columns=["x", "y", "z"])

    if all(x.has_vect):
        table = pd.concat(
            (
                table,
                pd.DataFrame(np.vstack(x.vect), columns=["vect_x", "vect_y", "vect_z"]),
            ),
            axis=1,
        )

    if all(x.has_alpha):
        table["alpha"] = np.concatenate(x.alpha)

    # Add neuron ID
    table["neuron"] = np.repeat(x.id, x.n_points)

    return table


def _compile_meta(
    x: Union["core.BaseNeuron", "core.NeuronList"],
    write_meta: bool,
    keys: Optional[dict] = None,
    skip: tuple = (),
) -> dict:
    """Compile per-neuron meta data for writing to parquet file.

    Properties are mapped as `"{ID}:{property}"`, e.g.
    `{"1734350788:name": "DA1_lPN_R"}`. Pass `keys` to write them under a
    different ID than the neuron's own (neurarrow uses uint64 fragment IDs) and
    `skip` for properties tracked at file level instead.
    """
    if isinstance(write_meta, (list, np.ndarray, tuple)):
        attrs = write_meta
    else:
        attrs = META_DATA

    metadata = {}
    for n in core.NeuronList(x):
        key = n.id if keys is None else keys[n.id]

        # ID is always written to file and it has to be a string
        metadata[f"{key}:id"] = str(n.id)

        # If not write_meta, only ID is written to file
        if not write_meta:
            continue

        for p in attrs:
            if p in skip or not getattr(n, p, None):
                continue
            metadata[f"{key}:{p}"] = str(getattr(n, p, None))

    return metadata


###############################################################################
#                                 Connectors                                  #
###############################################################################


def _connectors_to_table(x):
    """Pool the neurons' connector tables into one. Returns None if empty."""
    x = core.NeuronList(x)

    if not any(x.has_connectors):
        return None

    connectors = x.connectors

    missing = [c for c in ("connector_id", "node_id", "type") if c not in connectors]
    if missing:
        logger.warning(
            "Connector table(s) are missing the following columns and may not "
            f"read back correctly: {', '.join(missing)}"
        )

    return connectors


def _write_connectors(x, filepath, write_meta, write_connectors, spec, samples):
    """Write the neurons' connectors to the sidecar file next to `filepath`.

    Every path through here either writes the sidecar or removes it - it must
    never be left describing a file we have since overwritten.
    """
    import pyarrow as pa

    cn_file = _connectors_filepath(filepath)
    table = _connectors_to_table(x) if write_connectors else None
    metadata = None

    if table is not None and not table.empty:
        if spec is None:
            metadata = _compile_meta(x, write_meta=write_meta)
            table = pa.Table.from_pandas(table, preserve_index=False)
        elif samples is None:
            # Only skeletons produce the sample IDs neurarrow connectors need
            logger.warning(
                "Connectors can only be written in neurarrow format alongside "
                "skeletons - skipping."
            )
            table = None
        else:
            table, metadata = _connectors_to_neurarrow(table, samples, spec)

    if table is None:
        if cn_file.is_file():
            logger.info(f"Removing stale connector file {cn_file}")
            cn_file.unlink()
        return

    _write_table(table, cn_file, metadata)


def _read_connectors(filepath, neurons, table, samples, filtered=False):
    """Read the sidecar connector file (if any) and attach to `neurons`."""
    cn_file = _connectors_filepath(filepath)
    if not cn_file.is_file():
        return

    _, pq = _import_pyarrow("Reading")

    # The sidecar describes its own format - don't infer it from the main file
    schema = pq.read_schema(cn_file)
    by_neuron = "neuron" in schema.names

    index = {n.id: n for n in neurons}
    if filtered and by_neuron:
        # Don't read connectors for neurons we didn't load. neurarrow connectors
        # reference samples rather than neurons, so there is nothing to push down
        connectors = pq.read_table(cn_file, filters=[("neuron", "in", list(index))])
    else:
        connectors = pq.read_table(cn_file)
    connectors = connectors.to_pandas()

    if not by_neuron:
        if samples is None or "neuron" not in table.columns:
            logger.warning(
                f"Unable to map connectors in {cn_file.name} back onto nodes - skipping."
            )
            return
        connectors = _neurarrow_to_connectors(connectors, table, samples)

    for id, this_cn in connectors.groupby("neuron"):
        neuron = index.get(id)
        if neuron is not None:
            neuron.connectors = this_cn.drop(columns=["neuron"]).reset_index(drop=True)


###############################################################################
#                                  neurarrow                                  #
###############################################################################


def _neurarrow_unit(x):
    """The file-level `unit` metadata plus the scale factor for coordinates.

    neurarrow has no place for a scale factor (navis' "8 nanometer"), so we
    convert coordinates into the base unit ("nanometer") on the way out.
    """
    x = core.NeuronList(x)

    # Compare the parsed units, not the strings they were set from, so that
    # e.g. "8 nm" and "8 nanometer" count as the same
    units = {str(n.units_xyz) for n in x}
    if len(units) > 1:
        raise ValueError(
            "neurarrow tracks units for the whole file, so all neurons must "
            f"share the same units, got: {', '.join(sorted(units))}. "
            "Use `.convert_units()` to bring them onto a common scale."
        )

    units = x[0].units_xyz

    if units is None or units.dimensionless:
        return "", np.ones(3)

    scale = np.asarray(units.magnitude, dtype=np.float64)
    if np.any(scale != 1):
        logger.warning(
            f'Converting coordinates from "{x[0].units}" into "{units.units}" - '
            "neurarrow has no place for a scale factor. Note that this changes "
            "the coordinates written to file (but not the neurons themselves)."
        )

    return str(units.units), scale


def _scaled(values, scale):
    """Scale coordinates into the file's base unit."""
    return np.asarray(values, dtype=np.float64) * scale


def _neurarrow_fragment_ids(x):
    """Map neuron IDs onto the uint64 fragment IDs neurarrow requires."""
    ids = list(core.NeuronList(x).id)
    if all(
        isinstance(i, INT_TYPES + (np.unsignedinteger,)) and int(i) >= 0 for i in ids
    ):
        return {i: int(i) for i in ids}
    logger.info(
        "Neuron IDs are not unsigned integers - remapping them onto neurarrow "
        "fragment IDs. The originals are kept in the file's metadata."
    )
    return {i: n for n, i in enumerate(ids)}


def _compile_neurarrow_meta(x, write_meta, spec, frag_ids, extra=None):
    """Compile schema metadata for a neurarrow file."""
    metadata = {
        "version": NEURARROW_VERSION,
        "context": spec.context,
        "unit": spec.unit,
    }
    metadata.update(extra or {})

    # Units are tracked for the whole file and the coordinates have been
    # converted to match - writing them per neuron as well would apply the
    # scale factor twice on read
    per_neuron = _compile_meta(x, write_meta, keys=frag_ids, skip=("units",))
    metadata.update({f"frag:{k}": v for k, v in per_neuron.items()})

    return metadata


def _neurarrow_samples(neuron, node_id):
    """Assign globally unique sample IDs to (neuron, node_id) pairs."""
    index = pd.MultiIndex.from_arrays([neuron, node_id])
    if not pd.Index(node_id).has_duplicates:
        # Node IDs are already unique within the context
        return pd.Series(np.asarray(node_id, dtype=np.uint64), index=index), False
    return pd.Series(np.arange(len(index), dtype=np.uint64), index=index), True


def _lookup_samples(samples, neuron, node_id):
    """Look up sample IDs for (neuron, node_id) pairs.

    Returns the sample IDs plus a mask flagging pairs we couldn't find (e.g.
    the `-1` parent of a root node). Note that we're going via `get_indexer`
    rather than `reindex` to avoid uint64 IDs taking a lossy detour via float.
    """
    query = pd.MultiIndex.from_arrays([np.asarray(neuron), np.asarray(node_id)])
    loc = samples.index.get_indexer(query)
    missing = loc < 0
    found = samples.values[loc]
    return np.where(missing, 0, found).astype(np.uint64), missing


def _neurarrow_table(columns):
    """Build a table from `[(name, array, nullable), ...]`.

    neurarrow is explicit about which fields may be null, and readers (e.g. the
    reference validator) check for it - so we can't rely on pyarrow's default
    of making everything nullable.
    """
    import pyarrow as pa

    schema = pa.schema(
        [pa.field(name, arr.type, nullable) for name, arr, nullable in columns]
    )
    return pa.Table.from_arrays([arr for _, arr, _ in columns], schema=schema)


def _skeletons_to_neurarrow(x, write_meta, spec):
    """Turn skeletons into a neurarrow `skeletons` table."""
    import pyarrow as pa

    x = core.NeuronList(x)
    nodes = _skeletons_to_table(x)

    frag_ids = _neurarrow_fragment_ids(x)
    samples, remapped = _neurarrow_samples(nodes.neuron.values, nodes.node_id.values)

    # Parents live in sample ID space and roots are encoded as null
    parents, is_root = _lookup_samples(
        samples, nodes.neuron.values, nodes.parent_id.values
    )

    columns = [
        ("sample_id", pa.array(samples.values, type=pa.uint64()), False),
        (
            "fragment_id",
            pa.array(nodes.neuron.map(frag_ids).values.astype(np.uint64)),
            False,
        ),
        ("x", pa.array(_scaled(nodes.x.values, spec.scale[0])), False),
        ("y", pa.array(_scaled(nodes.y.values, spec.scale[1])), False),
        ("z", pa.array(_scaled(nodes.z.values, spec.scale[2])), False),
        ("parent_id", pa.array(parents, mask=is_root, type=pa.uint64()), True),
    ]

    if "radius" in nodes.columns:
        # Radii have no single scale for anisotropic units - go with x
        radius = pa.array(_scaled(nodes.radius.values, spec.scale[0]))
        columns.append(("radius", radius, True))

    extra = {}
    if "label" in nodes.columns:
        # SWC structure identifiers are modelled by an extension
        label = pa.array(nodes.label.values.astype(np.int64))
        columns.append((NAVIS_COLUMNS["label"], label, False))
        extra[f"{NEURARROW_SWC_EXT}:version"] = NEURARROW_SWC_EXT_VERSION

    if remapped:
        # Keep the original node IDs so we can round-trip losslessly
        orig = pa.array(nodes.node_id.values.astype(np.uint64))
        columns.append(("attr:node_id", orig, False))

    metadata = _compile_neurarrow_meta(x, write_meta, spec, frag_ids, extra)

    return _neurarrow_table(columns), samples, metadata


def _dotprops_to_neurarrow(x, write_meta, spec):
    """Turn dotprops into a neurarrow `dotprops` table."""
    import pyarrow as pa

    x = core.NeuronList(x)

    if not all(x.has_vect):
        raise ValueError(
            "neurarrow requires tangent vectors for all dotprops. Use "
            "`navis.make_dotprops()` to (re-)generate them."
        )

    ks = {n.k for n in x}
    if len(ks) > 1:
        raise ValueError(
            "neurarrow tracks the dotprops neighbourhood size for the whole "
            f"file, so all dotprops must share the same `k`, got: {sorted(ks)}"
        )

    frag_ids = _neurarrow_fragment_ids(x)
    points = np.vstack(x.points)
    vect = np.vstack(x.vect)
    frag = np.repeat([frag_ids[i] for i in x.id], x.n_points).astype(np.uint64)

    columns = [
        ("sample_id", pa.array(np.arange(len(points), dtype=np.uint64)), False),
        ("fragment_id", pa.array(frag), False),
    ]
    for i, axis in enumerate(("x", "y", "z")):
        columns.append((axis, pa.array(_scaled(points[:, i], spec.scale[i])), False))
    for i, axis in enumerate(("x", "y", "z")):
        # Tangent vectors are normalised, i.e. they don't need scaling
        columns.append(
            (
                NAVIS_COLUMNS[f"vect_{axis}"],
                pa.array(vect[:, i].astype(np.float64)),
                False,
            )
        )

    if all(x.has_alpha):
        alpha = pa.array(np.concatenate(x.alpha).astype(np.float64))
        columns.append((NAVIS_COLUMNS["alpha"], alpha, False))

    extra = {"neighborhood_size": str(int(ks.pop()))}
    metadata = _compile_neurarrow_meta(x, write_meta, spec, frag_ids, extra)

    # Dotprops don't carry connectors, hence no sample mapping
    return _neurarrow_table(columns), None, metadata


def _split_pre_post(connectors):
    """Work out which connector `type` labels are pre- and postsynaptic."""
    types = connectors["type"].unique()
    pre = utils.guess_connector_type(types, "pre")
    post = utils.guess_connector_type(types, "post")

    other = [t for t in types if t not in (pre, post)]
    if other:
        logger.warning(
            "neurarrow only models pre- and postsynaptic connectors - dropping "
            f"connectors of type(s): {', '.join(str(t) for t in other)}"
        )

    return pre, post


def _connectors_to_neurarrow(connectors, samples, spec):
    """Pivot navis connectors into the `net.clbarnes.connectors` schema."""
    import pyarrow as pa

    dropped = [c for c in connectors.columns if c not in CONNECTOR_COLUMNS]
    if dropped:
        logger.warning(
            "The neurarrow connector extension has no place for the following "
            f"columns - they are dropped: {', '.join(dropped)}. Use "
            '`format="navis"` to write them.'
        )

    pre, post = _split_pre_post(connectors)
    known = [t for t in (pre, post) if t is not None]
    if not known:
        logger.warning(
            "None of the connectors are recognisably pre- or postsynaptic - "
            "not writing a connector file."
        )
        return None, None

    connectors = connectors[connectors["type"].isin(known)].copy()
    sample_id, orphaned = _lookup_samples(
        samples, connectors.neuron.values, connectors.node_id.values
    )
    connectors["sample_id"] = sample_id

    if orphaned.any():
        logger.warning(
            f"Dropping {orphaned.sum()} connector(s) that don't map onto a node."
        )
        connectors = connectors[~orphaned]

    # neurarrow connector IDs must be unique within the context - navis' are
    # only guaranteed to be unique within a neuron. If the same ID shows up at
    # more than one location we know they are per-neuron and have to remap.
    # The cheap duplicate check short-circuits the much costlier `nunique`.
    per_neuron = (
        connectors.connector_id.duplicated().any()
        and (
            connectors.groupby("connector_id")[["x", "y", "z"]].nunique().max(axis=1)
            > 1
        ).any()
    )
    key = ["neuron", "connector_id"] if per_neuron else ["connector_id"]
    if per_neuron:
        logger.info(
            "Connector IDs are not unique across neurons - assigning new IDs. "
            "The originals are kept in `attr:connector_id`."
        )

    # One row per connector
    locs = connectors.groupby(key, sort=True)[["x", "y", "z"]].first()

    def _samples_for(type_):
        """The sample IDs of each connector's `type_` side, as a list column."""
        if type_ is not None:
            subset = connectors[connectors["type"] == type_]
            if not subset.empty:
                grouped = subset.groupby(key, sort=True).sample_id.agg(list)
                # Connectors with no side of this type get an empty list
                return [
                    v if isinstance(v, list) else []
                    for v in grouped.reindex(locs.index)
                ]
        return [[] for _ in range(len(locs))]

    if per_neuron:
        connector_id = np.arange(len(locs), dtype=np.uint64)
        orig_id = locs.index.get_level_values("connector_id").values
    else:
        connector_id = locs.index.values.astype(np.uint64)
        orig_id = None

    columns = [("connector_id", pa.array(connector_id), False)]
    for i, axis in enumerate(("x", "y", "z")):
        loc = pa.array(_scaled(locs[axis].values, spec.scale[i]))
        columns.append((axis, loc, False))
    for name, type_ in (("src_sample_ids", pre), ("tgt_sample_ids", post)):
        ids = pa.array(_samples_for(type_), type=pa.list_(pa.uint64()))
        columns.append((name, ids, True))

    if orig_id is not None:
        columns.append(("attr:connector_id", pa.array(orig_id.astype(np.int64)), False))

    metadata = {
        "version": NEURARROW_VERSION,
        "context": spec.context,
        # `net.clbarnes.connectors` inherits from the spatial schema
        "unit": spec.unit,
        f"{NEURARROW_CONNECTOR_SCHEMA}:version": NEURARROW_CONNECTOR_EXT_VERSION,
    }

    return _neurarrow_table(columns), metadata


def _neurarrow_to_connectors(connectors, table, samples):
    """Explode a `net.clbarnes.connectors` table back into navis connectors."""
    # Map sample IDs back onto (neuron, node_id)
    lookup = pd.DataFrame(
        {"neuron": table["neuron"].values, "node_id": table["node_id"].values},
        index=np.asarray(samples.values if hasattr(samples, "values") else samples),
    )

    if "attr:connector_id" in connectors.columns:
        # We had to remap connector IDs on write - restore the originals
        connectors = connectors.drop(columns=["connector_id"]).rename(
            columns={"attr:connector_id": "connector_id"}
        )

    parts = []
    for col, type_ in (("src_sample_ids", "pre"), ("tgt_sample_ids", "post")):
        if col not in connectors.columns:
            continue
        part = connectors[["connector_id", "x", "y", "z", col]].explode(col)
        part = part[part[col].notnull()].rename(columns={col: "sample_id"})
        part["type"] = type_
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=CONNECTOR_COLUMNS)

    out = pd.concat(parts, axis=0, ignore_index=True)
    matched = lookup.reindex(out.sample_id.astype(np.uint64).values)
    out["neuron"] = matched.neuron.values
    out["node_id"] = matched.node_id.values
    out = out[out.node_id.notnull()]
    out["node_id"] = out.node_id.astype(np.int64)
    out["connector_id"] = out.connector_id.astype(np.int64)

    return out[list(CONNECTOR_COLUMNS)].reset_index(drop=True)
