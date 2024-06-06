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
import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Union, Optional

from .. import config, core

__all__ = ["read_parquet", "write_parquet", "scan_parquet"]

# Set up logging
logger = config.get_logger(__name__)

SKELETON_COLUMNS = ('node_id', 'x', 'y', 'z', 'radius', 'parent_id', 'neuron')
NA_VALUES = (None, 'None')
META_DATA = ('name', 'units', 'soma')  # meta data to write for each neuron

INT_TYPES = (int, np.int8, np.int16, np.int32, np.int64)


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
    :func:`navis.write_parquet`
                        Export neurons as parquet files.
    :func:`navis.read_parquet`
                        Read parquet file into neurons.

    Examples
    --------
    See :func:`navis.write_parquet` for examples.

    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError('Reading parquet files requires the pyarrow library:\n'
                          ' pip3 install pyarrow')

    f = Path(file).expanduser()
    if not f.is_file():
        raise FileNotFoundError(f'File "{f}" does not exist.')

    metadata = pq.read_metadata(f)

    try:
        meta = {k.decode(): v.decode() for k, v in metadata.metadata.items()}
    except BaseException:
        logger.warning(f'Unable to decode meta data for parquet file {f}')

    # Parse meta data
    ids = [v for k, v in meta.items() if k.endswith(':id') and not k.startswith('_')]
    records = {i: {} for i in ids}
    for k, v in meta.items():
        if k.startswith('_'):
            continue
        if ':' not in k:
            continue

        id, prop = k.split(':')

        if id not in records:  # there might be an "ARROW:schema" entry
            continue

        records[id][prop] = v

    # Turn into DataFrame
    df =  pd.DataFrame.from_records(list(records.values()))

    # Move ID column to front
    ids = df['id']
    df.drop(labels=['id'], axis=1, inplace=True)
    df.insert(0, 'id', ids)

    return df


def read_parquet(f: Union[str, Path],
                 read_meta: bool = True,
                 limit: Optional[int] = None,
                 subset: Optional[List[Union[str, int]]] = None,
                 progress=True
                 ) -> 'core.NeuronObject':
    """Read parquet file into Neuron/List.

    See `here <https://github.com/navis-org/navis/blob/master/navis/io/pq_io.md>`_
    for format specifications.

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
                        can use this parameter to read only the first ``limit``
                        neurons. Useful if wanting to get a sample from a large
                        library of neurons.
    subset :            str | int | list thereof
                        If the parquet file contains multiple neurons you can
                        use this to select the IDs of the neurons to load. Only
                        works if the parquet file actually contains multiple
                        neurons.

    Returns
    -------
    navis.TreeNeuron/Dotprops
                        If parquet file contains a single neuron.
    navis.NeuronList
                        If parquet file contains multiple neurons.

    See Also
    --------
    :func:`navis.write_parquet`
                        Export neurons as parquet files.
    :func:`navis.scan_parquet`
                        Scan parquet file for its contents.

    Examples
    --------
    See :func:`navis.write_parquet` for examples.

    """
    f = Path(f).expanduser()
    if not f.is_file():
        raise FileNotFoundError(f'File "{f}" does not exist.')

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError('Reading parquet files requires the pyarrow library:\n'
                          ' pip3 install pyarrow')

    if limit is not None:
        if subset not in (None, False):
            raise ValueError('You can provide either a `subset` or a `limit` but '
                             'not both.')
        scan = scan_parquet(f)
        subset = scan.id.values[:limit]

    if isinstance(subset, (pd.Series)):
        subset = subset.values

    # Read the table
    if subset is None or subset is False:
        table = pq.read_table(f)
    elif isinstance(subset, (str, int)):
        table = pq.read_table(f, filters=[("neuron", "=", subset)])
    elif isinstance(subset, (list, np.ndarray)):
        table = pq.read_table(f, filters=[("neuron", "in", subset)])
    else:
        raise TypeError(f'`subset` must be int, str or iterable, got "{type(subset)}')

    # Extract meta data (will be byte encoded)
    if read_meta:
        metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    else:
        metadata = {}

    # Extract neuron meta data once here instead of for every neuron individually
    # Meta data is encoded as {"{ID}_{PROPERTY}": VALUE}
    # Here we pre-emptively turn this into {(ID, PROPERTY): VALUE}
    # Note that we're dropping "private" properties where the key starts with "_"
    neuron_meta = {tuple(k.split(':')): v for k, v in metadata.items() if not k.startswith('_')}

    # Convert to pandas
    table = table.to_pandas()

    # Check if we're doing skeletons or dotprops
    if 'node_id' in table.columns:
        _extract_neuron = _extract_skeleton
    elif 'vect_x' in table.columns:
        _extract_neuron = _extract_dotprops
    else:
        raise TypeError('Unable to extract neuron from parquet file with '
                        f'columns {table.columns}')

    # If this is a single neuron
    if 'neuron' not in table.columns:
        if metadata:
            id = [v for k, v in metadata.items() if k[1] == 'id'][0]
        else:
            id = '0'  # <-- generic ID as fallback if we don't have metadata
        return _extract_neuron(table, id, neuron_meta)
    else:
        neurons = []
        # Note: this could be done in threads
        for i, (id, this_table) in enumerate(config.tqdm(table.groupby('neuron'),
                                             disable=not progress,
                                             leave=False,
                                             desc='Making nrn')):
            this_table = this_table.drop("neuron", axis=1)
            neurons.append(_extract_neuron(this_table, id, neuron_meta))
        return core.NeuronList(neurons)


def _extract_skeleton(nodes, id, metadata):
    """Extract a single skeleton."""
    # Meta data is encoded as "{ID}_{PROPERTY}"
    str_id = str(id)
    this_meta = {k[1]: v for k, v in metadata.items() if k[0] == str_id}
    # Drop "Nones"
    this_meta = {k: v for k, v in this_meta.items() if v != "None"}

    # The soma needs to be added separately because it is typically stored as
    # list (e.g. [0]) which the TreeNeuron initialisation doesn't like
    if "soma" in this_meta:
        soma = this_meta.pop("soma")
        # Parse a list string (e.g. "[1]") back into a list
        if soma.startswith('['):
            soma = [_try_int(i.strip()) for i in soma[1:-1].split(',')]
        else:
            soma = _try_int(soma)
    else:
        soma = None

    # Make the neuron
    this_meta['id'] = id
    tn = core.TreeNeuron(nodes, **this_meta)

    # Fix soma
    if soma:
        tn.soma = soma
    else:
        tn.soma = None

    return tn


def _extract_dotprops(table, id, metadata):
    """Extract a single dotprop."""
    # Meta data is encoded as "{ID}_{PROPERTY}"
    str_id = str(id)
    this_meta = {k[1]: v for k, v in metadata.items() if k[0] == str_id}
    # Drop "Nones"
    this_meta = {k: v for k, v in this_meta.items() if v != "None"}

    # Make the neuron
    this_meta['id'] = id
    this_meta['k'] = this_meta.get('k', 5)  # <- set a default K of 5

    if 'vect_x' in table:
        this_meta['vect'] = table[['vect_x', 'vect_y', 'vect_z']].values
    if 'alpha' in table:
        this_meta['alpha'] = table['alpha'].values

    return core.Dotprops(table[['x', 'y', 'z']].values,
                       **this_meta)


def _try_int(x):
    """Try converting `x` into an integer."""
    try:
        return int(x)
    except ValueError:
        return x


def _int_to_bytes(x, bits=64):
    """Convert integer to bytes."""
    return int(x).to_bytes(bits, 'big')


def _bytes_to_int(x):
    """Convert bytes to integer."""
    return int.from_bytes(x, "big")


def write_parquet(x: 'core.NeuronObject',
                  filepath: Union[str, Path],
                  write_meta: bool = True) -> None:
    """Write TreeNeuron(s) or Dotprops to parquet file.

    See `here <https://github.com/navis-org/navis/blob/master/navis/io/pq_io.md>`_
    for format specifications.

    Parameters
    ----------
    x :                 TreeNeuron | Dotprop | NeuronList thereof
                        Neuron(s) to save. If NeuronList must contain either
                        only TreeNeurons or only Dotprops.
    filepath :          str | pathlib.Path
                        Destination for the file.
    write_meta :        bool | list of str
                        Whether to also write neuron properties to file. By
                        default this is `.name`, `.units` and `.soma`. You can
                        change which properties are written by providing them as
                        list of strings.

    See Also
    --------
    :func:`navis.read_parquet`
                        Import skeleton from parquet file.
    :func:`navis.scan_parquet`
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

    """
    filepath = Path(filepath).expanduser()

    # Make sure inputs are only TreeNeurons or Dotprops
    if isinstance(x, core.NeuronList):
        types = x.types
        if types == (core.TreeNeuron,):
            _write_parquet = _write_parquet_skeletons
        elif types == (core.Dotprops, ):
            _write_parquet = _write_parquet_dotprops
        else:
            raise TypeError('Can only write either TreeNeurons or Dotprops to '
                            f'parquet but NeuronList contains {types}')
        if x.is_degenerated:
            raise ValueError('NeuronList must not contain non-unique IDs')
    else:
        if isinstance(x, (core.TreeNeuron, )):
            _write_parquet = _write_parquet_skeletons
        elif isinstance(x, (core.Dotprops, )):
            _write_parquet = _write_parquet_dotprops
        else:
            raise TypeError('Can only write TreeNeurons or Dotprops to parquet, '
                            f'got "{type(x)}"')

    return _write_parquet(x, filepath=filepath, write_meta=write_meta)


def _write_parquet_skeletons(x: 'core.TreeNeuron',
                             filepath: Union[str, Path],
                             write_meta: bool = True,
                             ) -> None:
    """Write TreeNeurons to parquet file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError('Writing parquet files requires the pyarrow library:\n'
                         ' pip3 install pyarrow')

    # Make sure we're working with a list, not a single neuron
    x = core.NeuronList(x)

    # Generate node table
    nodes = x.nodes[x.nodes.columns[np.isin(x.nodes.columns, SKELETON_COLUMNS)]]

    # Convert to pyarrow table
    table = pa.Table.from_pandas(nodes)

    # Compile metadata
    metadata = _compile_meta(x, write_meta=write_meta)

    # Generate a schema with the new meta data
    schema = pa.schema([table.schema.field(i) for i in range(len(table.schema))],
                       metadata=metadata)

    return pq.write_table(table.cast(schema), filepath)


def _write_parquet_dotprops(x: 'core.Dotprops',
                            filepath: Union[str, Path],
                            write_meta: bool = True,
                            ) -> None:
    """Write Dotprops to parquet file.

    Examples
    --------
    We will test writing dotprops here instead of the main function

    >>> import navis
    >>> nl = navis.example_neurons(3, kind='skeleton')
    >>> dp = navis.make_dotprops(nl, k=5)
    >>> navis.write_parquet(dp, tmp_dir / 'dotprops.parquet')
    >>> dp2 = navis.read_parquet(tmp_dir / 'dotprops.parquet')
    >>> assert len(dp) == len(dp2)
    >>> assert all([i in dp2.id for i in dp.id])

    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError('Writing parquet files requires the pyarrow library:\n'
                         ' pip3 install pyarrow')

    # Make sure we're working with a list, not a single neuron
    x = core.NeuronList(x)

    # Generate table
    table = pd.DataFrame(np.vstack(x.points), columns=['x', 'y', 'z'])

    if all(x.has_vect):
        table = pd.concat((table,
                           pd.DataFrame(np.vstack(x.vect),
                                        columns=['vect_x', 'vect_y', 'vect_z'])
                                        ),
                          axis=1)

    if all(x.has_alpha):
        table['alpha'] = np.concatenate(x.alpha)

    # Add neuron ID
    table['neuron'] = np.repeat(x.id, x.n_points)

    # Convert to pyarrow table
    table = pa.Table.from_pandas(table)

    # Compile metadata
    metadata = _compile_meta(x, write_meta=write_meta)

    # Generate a schema with the new meta data
    schema = pa.schema([table.schema.field(i) for i in range(len(table.schema))],
                       metadata=metadata)

    return pq.write_table(table.cast(schema), filepath)


def _compile_meta(x: Union['core.BaseNeuron', 'core.NeuronList'],
                  write_meta: bool
                  ) -> dict:
    """Compile meta data for writing to parquet file."""
    metadata = {}
    for n in core.NeuronList(x):
        # ID is always written to file and it has to be a string
        if isinstance(n.id, INT_TYPES):
            metadata[f'{n.id}:id'] = str(n.id) #_int_to_bytes(n.id)
        else:
            metadata[f'{n.id}:id'] = str(n.id)

        # If not write_meta, only ID is written to file
        if not write_meta:
            continue

        if isinstance(write_meta, (list, np.ndarray, tuple)):
            attrs = write_meta
        else:
            attrs = META_DATA

        for p in attrs:
            if not getattr(n, p, None):
                continue
            # We're mapping meta data as "{ID}_{property}"
            # e.g. {"1734350788_name": "DA1_lPN_R"}
            metadata[f'{n.id}:{p}'] = str(getattr(n, p, None))

    return metadata