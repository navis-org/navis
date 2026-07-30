<!--
Prose block inlined at the top of llms.txt by gen_llms_txt.py.

This is the part of llms.txt that isn't generated from docstrings: what an
agent can't infer from signatures. Headings are demoted one level on the way
in, so `##` here becomes `###` under llms.txt's "Read this first". Keep it
self-contained - no repo-relative links, since it is served over HTTP.
-->

What follows is what you cannot infer from signatures and docstrings. The rest
of this file is the API index; the linked per-section files carry the full
docstrings.

## Installation

```bash
pip install navis[all]
```

## Core model

A neuron is one of four types:

| Type | Holds | Made from |
|---|---|---|
| `TreeNeuron` | skeleton: node table with `node_id`/`parent_id`/`x`/`y`/`z`/`radius` | SWC, `navis.skeletonize()` |
| `MeshNeuron` | vertices + faces | OBJ/STL, `navis.mesh()` |
| `VoxelNeuron` | 3D image | NRRD/TIFF, `navis.voxelize()` |
| `Dotprops` | points + local vectors | `navis.make_dotprops()` — **required for NBLAST** |

Explicit conversions go through `navis.skeletonize()`, `navis.mesh()`, `navis.voxelize()`
and `navis.make_dotprops()`. Not every pairing is supported — see the
"Neuron types and functions" compatibility matrix in the API index.

Some functions will implicitly convert a neuron to the required type and then map the result back to the original type. For example, `navis.strahler_index` works on a `MeshNeuron` by converting it to a `TreeNeuron`, computing the Strahler index, and mapping it back to the `MeshNeuron`.

`NeuronList` is the container. Most functions accept either a single neuron or
a `NeuronList` and return the same shape.

## A few potential pitfalls

### 1. Optional dependencies

navis has a few optional dependencies that are not installed by default but can greatly
improve performance or enable additional functionality. See installation instructions
in the documentation for details.

### 2. Brain spaces & units

**This is the most important item on this page.** Neurons from different datasets are likely
in different (template) spaces, and navis itself does not track that. Even within the same
template space you may encounter data in different units - e.g. nanometers, microns or voxels. Fortunately, navis neurons carry an optional `.units` attribute which is often set
by the loader. For example, the demo neurons returned by `navis.example_neurons()` are in **8 nanometer** voxel space, not microns.

The reason this is important is that navis *will not warn you* if you compare, overlay or
NBLAST neurons from different spaces — you may get a plausible, wrong answer.

To align neurons from different template spaces, use `navis.xform_brain()` with a registered transform. For example the `flybrains` package registers the transforms between hemibrain, FlyWire and JRC2018F:

```python
import navis, flybrains          # flybrains registers the transforms

hb = navis.xform_brain(hemibrain_neurons, source='JRCFIB2018Fraw', target='JRC2018F')
fw = navis.xform_brain(flywire_neurons,  source='FLYWIRE',        target='JRC2018F')
# only now are hb and fw comparable
```

For neurons in the same template space but different units, you can either use the `.convert_units()` method (requires the `.units` property to be set) or use simple scaling:

```python
n = navis.example_neurons(2)  # hemibrain skeletons in 8nm voxel space
n_um = n * 8 / 1000           # convert to microns
```

### 3. NBLAST has three preconditions; only two are checked

`navis.nblast()` requires its inputs to be (a) `Dotprops`, (b) in microns, and
(c) in the same template space.

- **(a) is a hard error** with the fix in the message:
  `TypeError: 'query' must be Dotprop(s), got "(<class '...TreeNeuron'>,)". Use 'navis.make_dotprops' to convert neurons.`
- **(b) is only a warning:**
  `NBLAST is optimized for data in microns and it looks like your queries may not be in microns.`
  Scores are still returned, and they are not comparable to published ones.
- **(c) is not checked at all.** See item 2.

The full, correct pipeline:

```python
nl = nl.convert_units('um')                    # (b) — do this BEFORE make_dotprops
dp = navis.make_dotprops(nl, k=5, resample=1)  # (a) — k=5 for dense data, k=20 default, resample only needed if neurons are much denses or much sparser than 1 micron
scores = navis.nblast(dp, dp)                  # navis.nblast_allbyall(dp) also works
```

`navis.nblast()` is **asymmetric**: `scores.loc[a, b] != scores.loc[b, a]`. The
default `scores='forward'` returns the raw query→target matrix. Use
`scores='mean'` (or `'min'`/`'max'`) when you want a symmetric matrix — which is
almost always what you want for clustering or a distance matrix.

### 4. Functions return, they don't mutate (by default)

Every navis function that takes `inplace` defaults to `inplace=False` (28 of
them, no exceptions). So this silently does nothing:

```python
navis.prune_twigs(nl, size=5)      # WRONG — result discarded
nl = navis.prune_twigs(nl, size=5) # right
navis.prune_twigs(nl, size=5, inplace=True) # also right and saves a copy
```

`NeuronList` also auto-dispatches methods and functions over its members, so
`nl.prune_twigs(size=5)` and `nl.convert_units('um')` work and return a new
`NeuronList`.

### 5. Names that look alike and aren't

- `navis.mesh()` converts *to* a mesh; `navis.MeshNeuron` is the class;
  `navis.read_mesh()` loads one from disk.
- `navis.smooth_skeleton()` / `smooth_mesh()` / `smooth_voxels()` are three
  different functions — there is no generic `smooth()`.
- `navis.mirror_brain()` takes neurons and is template-aware (needs registered
  transforms); `navis.mirror()` takes a raw `(N, 3)` point array and flips it
  about an axis. They are not interchangeable.
- Inconsistent casing: `navis.neuron2nx()` but `navis.neuron2KDTree()`.
- `navis.betweeness_centrality()` is spelled with one `n` — that is the real
  name, not a typo to correct.

## Verify before you run

`navis.example_neurons()` works offline and needs no credentials. Use it to
check that a snippet runs before pointing it at the user's data:

```python
import navis
nl = navis.example_neurons(2)      # skeletons; kind='mesh' for MeshNeurons
nl.summary()                       # cheap overview: type, cable_length, units, ...
```

`.summary()` on a neuron or `NeuronList` is the right way to inspect results —
prefer it over dumping `.nodes`, which is a full node table and will flood the
context window. For large results, write them out with
`navis.write_parquet()` and read back selectively with `navis.read_parquet()` (see the API reference).
