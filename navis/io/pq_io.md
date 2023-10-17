## Storing Neurons in Parquet Files

Current formats for storing neuroanatomical data typically focus on one neuron per file. Unsurprisingly this doesn't scale well to tens of thousands of neurons: reading and writing a large number of small files quickly becomes painfully slow.

Here, we propose a file format that stores an arbitrary number of neurons in a single Parquet file. What is Parquet you ask? Why, Apache Parquet is a column-oriented data file format designed for efficient data storage and retrieval. It has two important properties for what we are trying to do:

1. Because it is column-oriented we can quickly search for a given neuron without having to load the entire file.
2. It allows storage of arbitrary meta-data which we can use to store neuron properties

### Skeletons

For skeletons the file contains the SWC table plus an additional column indicating the neuron ID.

Let's assume we want to store two neurons with IDs `12345` and `67890`, respectively. The table would look like this:

```
  node_id  parent_id      radius      x      y      z neuron
        1         -1   10.000000  15784  37250  28062  12345
        2          1   18.284300  15764  37230  28082  12345
        3          2   34.721401  15744  37190  28122  12345
        4          3   34.721401  15744  37150  28202  12345
        5          4   34.721401  15704  37130  28242  12345
      ...        ...         ...    ...    ...    ...  ...
     4843          9   62.111000  15450  35582  23284  67890
     4844          7   46.568501  15830  36182  23124  67890
     4845          6  287.321014  15450  35862  23244  67890
     4846          5  151.244995  15530  35622  22844  67890
     4847          3   30.000000  15970  36362  23044  67890
```

A node table must contain the following columns: `node_id`, `x`, `y`, `z`, `parent_id` and `neuron`. Additional columns (such as  `radius`) are allowed but may be ignored by the reader.

### Dotprops

Dotprops are point clouds with associated vectors indicating directionality.

The table for two dotprops with IDs `12345` and `67890` would look like this:

```
      x      y      z     vec_x     vec_y     vec_z  neuron
  15784  37250  28062 -0.300205 -0.393649  0.868860   12345
  15764  37230  28082 -0.108453 -0.211375  0.971369   12345
  15744  37190  28122 -0.043569 -0.455931  0.888948   12345
  15744  37150  28202 -0.307032 -0.533594  0.788041   12345
  15704  37130  28242 -0.143014 -0.318347  0.937124   12345
    ...    ...    ...       ...       ...       ...     ...
  15450  35582  23284 -0.199709 -0.936613  0.287877   67890
  15830  36182  23124 -0.154800 -0.952146 -0.263541   67890
  15450  35862  23244 -0.283306  0.042772  0.958075   67890
  15530  35622  22844 -0.020729  0.722610  0.690945   67890
  15970  36362  23044 -0.459681 -0.524251  0.716836   67890
```

The node table must contain the following columns: `x`, `y`, `z`, and `neuron`.
Additional columns such as `vec_x`, `vec_y`, `vec_z` or `alpha` are allowed but
may be ignored by the reader.

### Meta data

Meta data can be stored in Parquet files as `{key: value}` dictionary where both
`key` and `value` have to be either strings or bytes. Note that the former will be converted to bytes automatically.

This means that floats/integers need to be converted to bytes or strings.

To keep track of which neuron has which property, the meta data is encoded in
the dictionary as `{ID:PROPERTY: VALUE}`. For example, if our two neurons in the
examples above had names they would be encode as:

```
{"12345:name": "Humpty", "67890:name": "Dumpty"}
```

The datatype of the `ID` (i.e. whether ID is `12345` or `"12345"`) can be inferred
from the node table itself. In our example, the names (Humpty and Dumpty) are
quite obviously supposed to be strings. This may be less obvious for other
(byte-encoded) properties or values. It is on the reader to decide how to parse
them. In the future, we could add additional meta data to determine data
types e.g. via `{"_dtype:name": "str", "_dtype:id": "int"}`.

### Synapses

Synapses and other similar data typically associated with a neuron must be
stored in separate parquet files.

We propose using a simple zip archive where:

```bash
skeletons.parquet.zip
├── skeletons.parquet  <- contains the actual skeletons
└── synapses.parquet   <- contains the synapse data
```

## Benchmarks

Testcase: 1,000 skeletons on a 2018 MacBook Pro


| Writing           | Timing|
|-------------------|-------|
|Write to SWC files: | 2:37min|
|Write to Zip: | 2:55min |
|Write to Parquet: |0:25min|


| Size on disk      |       |
|-------------------|-------|
|SWC files: | 200.7MB |
|Zip archive: | 55.6MB|
|Parquet file: | 35.6MB|

| Reading           | Timing|
|-------------------|-------|
|SWC files (single thread): | 0:42min |
|SWC files (multi-thread): | 0:24min |
|Zip archive (single thread): | 1:01min |
|Zip archive (multi-thread): | 0:28min |
|Parquet file: | 0:35min |

As you can see, in these preliminary tests parquet is ahead in terms of
writing speed and size on disk. It beats reading if compared to single-threaded
reads but is slightly slower compared to multi-threaded.
