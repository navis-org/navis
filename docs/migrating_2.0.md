---
icon: material/sign-direction
hide:
  - navigation
---

# :material-sign-direction: Migrating to {{ navis }} 2.0

{{ navis }} 2.0 is the first major release since 1.0 and the first that is allowed to
break things. This page is three things: a summary of what the release is *about*, a checklist
of everything you may have to change in your own code, and - below the fold - the
complete list of changes, which is what the [changelog](changelog.md) points at
rather than repeats.

## What 2.0 is about

**Rust is the engine.** [navis-fastcore](https://github.com/schlegelp/fastcore-rs)
has been an optional dependency for several releases, with a second, pure-Python
implementation kept bit-for-bit in step behind it. That is over: fastcore is now
**required** and the fallbacks are gone. The same move retires the other
"install this and {{ navis }} will be faster" paths - the four mesh simplification
backends, the mesh smoothing backends, the CMTK/elastix binaries, `pycpd`,
`morphops`/`molesq` - each replaced by one implementation that everybody gets.
Most of the speed-ups in this release are that, and some of them are large:
`betweenness_centrality` 217 ms :octicons-arrow-right-24: 0.49 ms, `smooth_skeleton`
~200x, `smooth_mesh` ~50x, alignment 20-28x, `xform_brain` 4-30x.

**One neuron model.** Which attributes of a neuron are aligned to its nodes,
vertices, points or voxels - and which of them *reference* those elements - is now
declared once, per type, in `navis/core/schema.py`. [`subset_neuron`][navis.subset_neuron]
was rewritten onto it (three hand-written implementations became one, and four
silent bugs fell out), and three things that were not possible before follow from
it: [`navis.masked`][navis.masked] to work on part of a neuron and fold the edits
back in, [`Neuron.attach`][navis.BaseNeuron.attach] to carry *your* data through
every selection, and a [`Mesh`][navis.Mesh] that keeps its skeleton - with the same
node IDs - through a subset, a mask or a simplification.

**One name per thing.** The neuron classes are now
[`Skeleton`][navis.Skeleton], [`Mesh`][navis.Mesh], [`Voxels`][navis.Voxels] and
[`Dotprops`][navis.Dotprops]; a distance cap is `max_dist` everywhere; `min_size`
counts elements and `min_length` is a distance; connected components have one
function, one shape and one home; and every morphology function that takes a
single neuron has a method, without the type suffix. Old names mostly still work
with a `DeprecationWarning` - see the checklist below.

**New capabilities**, most of which close a hole rather than open a front:
[`plot_collage`][navis.plot_collage] and [`plot3d(snapshot=True)`][navis.plot3d]
for figures, [`nblast_knn`][navis.nblast_knn] for k-NN at connectome scale,
[`heal_mesh`][navis.heal_mesh]/[`fill_holes`][navis.fill_holes]/[`Mesh.extra_edges`][navis.Mesh.extra_edges]
and [`find_soma_mesh`][navis.find_soma_mesh] so meshes can do what skeletons could,
a proper [`Voxels`][navis.Voxels] toolkit that never materialises the grid,
`navis.ml` for machine-learning inputs, `.rds`/`.rda` I/O that replaces the `rpy2`
interface, [`Pipeline`][navis.Pipeline] to fuse a chain of operations into one
pass over the neurons, and parallel backends that reach a cluster.

**And a lot of fixes.** Rather more of them than usual return *different numbers*
than 1.12 did, because that is what a fix to a silent bug looks like - see
[Results that change](#results-that-change).

## Migration checklist

### Renamed - old name still works, with a `DeprecationWarning`

| Old | New |
|-----|-----|
| `navis.TreeNeuron` | [`navis.Skeleton`][] |
| `navis.MeshNeuron` | [`navis.Mesh`][] |
| `navis.VoxelNeuron` | [`navis.Voxels`][] |
| `navis.break_fragments()` | [`navis.split_components()`][navis.split_components] |
| `navis.split_into_fragments()` | [`navis.split_neurites()`][navis.split_neurites] |
| `navis.graph.connected_subgraph()` | [`navis.graph.connecting_nodes()`][navis.graph.connecting_nodes] |
| `Skeleton.subtrees` | `Skeleton.connected_components()` |
| `Skeleton.n_trees`, `Skeleton.n_skeletons` | `Skeleton.n_components` |
| `Skeleton.is_tree` | [`Skeleton.is_acyclic`][navis.Skeleton.is_acyclic] |
| `bridging_graph(reciprocal=)` & co. | `inverse_weight=` (and the default changed from `0.5` to `1`) |
| `smooth_mesh(L=)` | `lamb=` |
| `simplify_mesh(backend=)`, `smooth_mesh(backend=)` | gone; there is only one backend now (the argument is ignored) |

The class names are **aliases**, not subclasses: `isinstance(x, navis.TreeNeuron)`
and `class MyNeuron(navis.TreeNeuron)` behave exactly as before, and neurons
pickled by earlier versions still load. Note that `.type` follows the class, so it
now reads `"navis.Skeleton"` - code matching on that string, including anything
filtering a [`NeuronList.summary()`][navis.NeuronList.summary] table by `type`,
needs updating.

Python hides `DeprecationWarning`s by default outside `__main__`; run with
`python -W default::DeprecationWarning` (or `pytest -W default`) to find the old
names in your code.

### Renamed - no alias

| Old | New | Where |
|-----|-----|-------|
| `limit_dist=` | `max_dist=` | all five `nblast_*` |
| `limit=` | `max_dist=` | [`geodesic_matrix`][navis.geodesic_matrix], [`average_skeletons`][navis.average_skeletons] |
| `dist=` | `max_dist=` | [`cable_overlap`][navis.cable_overlap] |
| `min_size=` | `min_length=` | [`split_neurites`][navis.split_neurites] |
| `size=` | `min_length=` | [`prune_twigs`][navis.prune_twigs], `Skeleton.prune_twigs()` |
| `keep_size=` | `min_size=` | [`drop_fluff`][navis.drop_fluff] |
| `drop_disc=` | `keep_largest=` | [`heal_skeleton`][navis.heal_skeleton], [`heal_mesh`][navis.heal_mesh] |
| `dp.drop_fluff(500)` | `dp.drop_fluff(epsilon=500)` | `epsilon` is keyword-only on the shared method |
| `navis.betweeness_centrality` | [`navis.betweenness_centrality`][navis.betweenness_centrality] | the misspelling is gone |
| `navis.graph.connected_components_of(x, mask)` | `navis.connected_components(x, mask=...)` | |
| `break_fragments(..., labels_only=True)` | [`navis.connected_components`][] | |
| `prune_by_strahler(..., relocate_connectors=)` | removed; passing it raises `TypeError` | the default was `False`, so nothing changes unless you passed it |

### Removed

| Removed | Use instead |
|---------|-------------|
| `navis.interfaces.r` (and the `navis[r]` extra) | [`read_rds`][navis.read_rds]/[`write_rds`][navis.write_rds], [`read_rda`][navis.read_rda]/[`write_rda`][navis.write_rda], plus {{ navis }}' own `nblast`/`xform_brain` |
| `navis.interfaces.cytoscape` | [`navis.network2nx`][] + `networkx.write_graphml` |
| the `vispy` plotting backend, `navis.Viewer`, `navis.utils.check_vispy` | [octarine](https://schlegelp.github.io/octarine/); [`plot3d`][navis.plot3d] returns an `octarine.Viewer` |
| `plot3d` arguments `combine`, `shading`, `shininess`, `name` | vispy-only; no replacement |
| `navis.meshes.simplify_mesh_fqmr`/`_open3d`/`_pyml`/`_blender`, `navis.meshes.available_backends` | [`navis.simplify_mesh`][], [`navis.smooth_mesh`][] |
| `navis.find_soma(x)[0]`, `len(navis.find_soma(x))` | [`find_soma`][navis.find_soma] now returns a single node ID or `None` |
| the non-fastcore graph/geodesic/morphometric/NBLAST fallbacks | nothing to do - fastcore is required |
| the private stitching helpers `_stitch_edges`, `_segment_radii`, `_rewire_from_edges`, `_component_labels` | [`heal_skeleton`][navis.heal_skeleton], [`stitch_skeletons`][navis.stitch_skeletons] |

Deprecated but still working, and scheduled for removal in 3.0: the `"binary"`
CMTK/elastix transform backend and the `"python"` (`morphops`/`molesq`) landmark
backend.

### Results that change

These are not renames - the same call returns something different than it did in
1.12. Re-baseline anything pinned to exact output.

| What | Why |
|------|-----|
| NBLAST scores (~1e-5) | fastcore now runs on the `float32` coordinates {{ navis }} stores, cutting peak memory ~45% |
| [`smooth_skeleton`][navis.smooth_skeleton] | the window is centred now, not trailing, and branch points are pinned (median 54 nm on the example neuron) |
| [`smooth_mesh`][navis.smooth_mesh] | defaults to Taubin's λ&#124;μ filter rather than the plain Laplacian, which cost the example neuron 55% of its volume |
| [`smooth_voxels`][navis.smooth_voxels] | treats everything outside the neuron as empty instead of reflecting at the canvas edge |
| [`downsample_neuron`][navis.downsample_neuron] | fastcore counts from each segment's distal end, so a different ~7% of nodes survives |
| [`find_soma`][navis.find_soma] | returns one node ID (or `None`), scored by neighbourhood radius, not an array of candidates |
| [`betweenness_centrality`][navis.betweenness_centrality]`(from_=)` | sources within one hop of the root now count |
| `node_label_sorting`, [`rewire_skeleton`][navis.rewire_skeleton] | ties and leftover roots broken deterministically rather than arbitrarily |
| [`collapse_nodes`][navis.collapse_nodes] | no longer re-roots the neuron at the collapsed node |
| [`ivscc_features`][navis.ivscc_features] | neurons are **rows** now (add a `.T` for the old layout), and six features were wrong |
| bridging paths | ~29% of routes in `navis-flybrains` change - mostly {{ navis }} no longer inverting a registration when a purpose-built one exists |
| [`align_rigid`][navis.align.align_rigid]/[`align_deform`][navis.align.align_deform] | now `rcpd` registration objects, and four `pycpd` bugs are fixed |
| [`plot2d`][navis.plot2d] with meshes | real back-face culling and depth ordering; `plot2d(method="2d")` no longer collapses a mesh into a silhouette |
| `parallel=True` | defaults to the `joblib` backend rather than `pathos` |
| [`read_precomputed`][navis.read_precomputed] | edges are oriented by traversal, so skeletons written "child first" no longer shatter |
| [`geodesic_matrix`][navis.geodesic_matrix], `segment_analysis`, `sholl_analysis`, `make_dotprops`, [`Voxels`][navis.Voxels] | see the fixes below - each was returning something wrong |

### Requirements

- **`navis-fastcore >= 0.13.0` is required.** Prebuilt wheels exist for macOS
  (Intel/ARM), Windows and Linux on x86-64, aarch64, i686, armv7l, ppc64le and
  s390x; only musl-based Linux (Alpine) builds from source. `pip install navis[fastcore]`
  still resolves as a no-op alias
- **`sparse-cubes[skeleton] >= 0.5.0` is required** (it backs the sparse [`Voxels`][navis.Voxels])
- **`rcpd` is a new optional dependency** for alignment: `pip install navis[cpd]`
- **`pathos` is no longer needed** for `parallel=True`
- `six` and `pypng` are gone; `pip install navis[all]` no longer drags in `cloud-volume`

## Worth adopting

Nothing here breaks anything - but if you have been working around one of these,
stop:

- [**`navis.masked`**][navis.masked] instead of subsetting, working on the
  fragment and stitching it back by hand. Inside the block the neuron *is* the
  masked region, so every function, property and plot sees only that part; on the
  way out it is whole again, edits folded in ([tutorial](../generated/gallery/tutorial_basic_03_masking))
- [**`Neuron.attach`**][navis.BaseNeuron.attach] for per-node/per-vertex data you
  currently keep in a parallel array and re-index yourself. Attached data is
  subset, masked and (where the function can say what happened) carried through
  rebuilds ([tutorial](../generated/gallery/tutorial_basic_04_attach))
- [**`plot3d(..., snapshot=True)`**][navis.plot3d] when you want a real render
  *on a matplotlib axes*, in the neurons' own coordinates - figures you can
  annotate, arrange and save as PDF
- [**`plot_collage`**][navis.plot_collage] for a page of a few hundred neurons
  ([tutorial](../generated/gallery/1d_plotting_misc/zzz_tutorial_plotting_misc_04_collage))
- [**`nblast_knn`**][navis.nblast_knn] instead of an all-by-all you only wanted
  the top matches from - 164k neurons is a 107 GB matrix or a 26 MB k-NN graph
- [**`navis.Pipeline`**][navis.Pipeline] instead of calling three functions with
  `parallel=True` in turn, or smuggling the whole chain into one `apply` lambda -
  the neurons make the trip to the workers once for the chain rather than once
  per function, and the steps stop copying each other's intermediates
- [**`set_parallel_backend`**][navis.set_parallel_backend] to point `parallel=True`
  at `joblib`, a `dask.distributed` cluster, a SLURM array job or any
  `concurrent.futures.Executor` you own - and `inner_max_num_threads` so the
  processes and the threads underneath them stop fighting
- [**`downsample_neuron(..., method="rdp")`**][navis.downsample_neuron] to spend
  the node budget where the neuron curves rather than uniformly
- **`navis.config.strict`** (or `NAVIS_STRICT=1`) in servers and
  pipelines: remote fetches raise instead of returning a partial result, and
  nothing prompts for input
- [**`write_parquet(..., format="neurarrow")`**][navis.write_parquet] when the
  file has to be read by something that is not {{ navis }}

## All changes

Everything below is the release in full, grouped by what it touches.

## Naming: one name per thing

The changes that touch the most code are the ones that make {{ navis }} say the same thing the same way everywhere.

#### Breaking

- **one name for a distance cap (`max_dist`), and `min_size`/`min_length` now mean count and distance respectively.** The same "stop at this distance" argument was spelled five ways depending on which corner of {{ navis }} you were in, and `min_size` meant a node count in four functions and a cable length in a fifth.

    | Old | New | Functions |
    |-----|-----|-----------|
    | `limit_dist` | `max_dist` | all five `nblast_*` |
    | `limit` | `max_dist` | [`geodesic_matrix`][navis.geodesic_matrix], [`average_skeletons`][navis.average_skeletons] |
    | `dist` | `max_dist` | [`cable_overlap`][navis.cable_overlap] |
    | `min_size` | `min_length` | [`split_neurites`][navis.split_neurites] |
    | `size` | `min_length` | [`prune_twigs`][navis.prune_twigs] (and `Skeleton.prune_twigs()`) |

    The rule now holds throughout: **`min_size` counts elements** (nodes/vertices/voxels - [`drop_fluff`][navis.drop_fluff], [`split_components`][navis.split_components], [`heal_skeleton`][navis.heal_skeleton], [`heal_mesh`][navis.heal_mesh], [`stitch_skeletons`][navis.stitch_skeletons]), **`min_length` is a distance**, and **`max_dist` caps a distance**. `epsilon` is deliberately untouched: it does not cap a search but *defines* the edges of a neighbourhood graph, which is a different question.

    `limit` survives only where it never meant a distance - restricting which files a reader reads, and `guess_radius`' count of consecutive missing radii.

    **Every one of these now accepts a unit string** (e.g. `"5 microns"`) via [`map_units`][navis.BaseNeuron.map_units], where it did not already. Newly gained: [`stitch_skeletons`][navis.stitch_skeletons], [`navis.graph.geodesic_clusters`][navis.graph.geodesic_clusters] (only when `weight` is not `None` - with `weight=None` the radius is a number of hops, which has no units) and the `nblast_*` family (`"auto"` and `None` keep their own meaning and pass through).

- **every morphology function that takes one neuron now has a method, and the methods drop the type suffix.** Which operations were reachable as `neuron.do_this()` rather than `navis.do_this(neuron)` was arbitrary: `prune_twigs` had a method, `despike_skeleton` did not, and `drop_fluff` had one on `Dotprops` alone despite the function accepting all four neuron types - the same hole `connected_components` had on `Voxels`.

    Shared operations now live on [`BaseNeuron`][navis.BaseNeuron], so every type has them:

    | Method | Wraps |
    |--------|-------|
    | [`Neuron.drop_fluff()`][navis.BaseNeuron.drop_fluff] | [`navis.drop_fluff`][] |
    | [`Neuron.subset()`][navis.BaseNeuron.subset] | [`navis.subset_neuron`][] |
    | [`Neuron.split_axon_dendrite()`][navis.BaseNeuron.split_axon_dendrite] | [`navis.split_axon_dendrite`][] |

    Type-specific ones live on their class and **drop the suffix the function carries**, following `reroot_skeleton` → `.reroot()` and `resample_skeleton` → `.resample()`, which already worked this way:

    | Method | Wraps |
    |--------|-------|
    | [`Skeleton.heal()`][navis.Skeleton.heal] / [`Mesh.heal()`][navis.Mesh.heal] | [`navis.heal_skeleton`][] / [`navis.heal_mesh`][] |
    | [`Skeleton.smooth()`][navis.Skeleton.smooth] / [`Mesh.smooth()`][navis.Mesh.smooth] / [`Voxels.smooth()`][navis.Voxels.smooth] | [`navis.smooth_skeleton`][] / [`navis.smooth_mesh`][] / [`navis.smooth_voxels`][] |
    | [`Skeleton.despike()`][navis.Skeleton.despike] | [`navis.despike_skeleton`][] |
    | [`Skeleton.cut()`][navis.Skeleton.cut] | [`navis.cut_skeleton`][] |

    **Breaking:** `Dotprops.drop_fluff()` took `epsilon` as its first positional argument; the shared method takes it as a keyword, so `dp.drop_fluff(500)` must become `dp.drop_fluff(epsilon=500)`. Every method also takes `**kwargs` straight through to the function it wraps, so the two cannot drift apart.

    All methods now agree on `inplace`: `inplace=True` returns `None` and mutates, `inplace=False` returns a copy. `Mesh.heal()` and `Mesh.smooth()` were the odd ones out while being written and were corrected before landing - note that the older `Mesh.fill_holes()` and `Mesh.validate()` still hand the neuron back either way.

- **connected components have one name, one shape and one home.** {{ navis }} had six words for "a connected piece of a neuron" (*component*, *fragment*, *subtree*, *tree*, *skeleton*, *fluff*), three incompatible ways to ask for them, and coverage that depended on which neuron type you happened to hold: a `Skeleton` handed back sets of node IDs, a `Voxels` a label array, a `Mesh` nothing at all. All of it now goes through one function.

    **New: [`navis.connected_components`][]** - the primitive, public at last. It takes any neuron (`Skeleton`, `Mesh`, `Dotprops`, `Voxels` or a bare `Trimesh`) and returns an `(N, )` array of labels, one per node/vertex/point/voxel and aligned with it:

    ```python
    >>> labels = navis.connected_components(mesh)
    >>> np.bincount(labels)[:3]         # component sizes
    array([17058,   240,    12])
    >>> mesh.vertices[labels == 0]      # the largest component
    ```

    Labels are **sorted by size, largest first**, so `== 0` always selects the biggest piece and `np.bincount` gives the sizes in descending order (ties break by first element, so the labelling is a function of the neuron rather than of union-find's internals). Where the old function returned a list of sets, an array composes with `np.bincount`, `groupby` and boolean indexing without materialising anything, and matches `scipy.sparse.csgraph.connected_components`. It takes `connectivity` and `epsilon` (as [`drop_fluff`][navis.drop_fluff] always did) plus two new arguments:

    - `element` decides what the labels are *of*. Only meshes have a choice: `"vertex"` normally, but under a face-based `connectivity` a pinch vertex belongs to several components at once and only `"face"` is a partition - so that becomes the default there. Asking for `"vertex"` anyway is allowed; each such vertex takes the label of the largest component it touches.
    - `mask` restricts the answer to the *induced* sub-neuron: anything outside it neither belongs to a component nor connects two. This replaces the internal `graph.connected_components_of`.

    **New: every neuron type has [`.connected_components()`][navis.BaseNeuron.connected_components] and [`.n_components`][navis.BaseNeuron.n_components]**, off `BaseNeuron`, so a `Mesh` and a `Dotprops` can answer the question a `Skeleton` always could.

    **Renamed:**

    | Old | New | Note |
    |-----|-----|------|
    | `navis.break_fragments()` | [`navis.split_components()`][navis.split_components] | `labels_only=True` is gone - that is what `connected_components` is for. Gained `connectivity`/`epsilon` |
    | `navis.split_into_fragments()` | [`navis.split_neurites()`][navis.split_neurites] | never had anything to do with components: it cuts a *connected* arbor at longest-neurite branch points |
    | `Skeleton.subtrees` | `Skeleton.connected_components()` | |
    | `Skeleton.n_trees` | `Skeleton.n_components` | |
    | `Skeleton.n_skeletons` | `Skeleton.n_components` | was `len(.root)`, i.e. the component count only for an acyclic neuron - a cycle has no root and went uncounted |
    | `Skeleton.is_tree` | [`Skeleton.is_acyclic`][navis.Skeleton.is_acyclic] | it is a cycle check, not a connectivity check - it always returned `True` for a *forest*, which the old name made hard to believe |
    | `navis.graph.connected_subgraph()` | [`navis.graph.connecting_nodes()`][navis.graph.connecting_nodes] | returns the nodes needed to *connect* a subset; the old name collides with `connected_components` |
    | `navis.graph.connected_components_of()` | `navis.connected_components(mask=...)` | |
    | `Voxels.connected_components(connectivity=26)` | `Voxels.connected_components()` | same name, now the shared one; `connectivity` defaults through `None` |

    **Parameters normalised:** `drop_fluff(keep_size=)` is now `min_size=`, matching `split_components` and both healers. `heal_skeleton(drop_disc=)` / `heal_mesh(drop_disc=)` are now `keep_largest=`, which says what they do (and `heal_skeleton` annotated it `float` where `heal_mesh` had `bool`).

    The renamed top-level functions still resolve under their old names with a `DeprecationWarning`, as do the renamed `Skeleton` properties. Run with `python -W default::DeprecationWarning` to find them in your code.

- **the neuron classes have been renamed.** The old names said "neuron" three times over and buried the part that actually distinguishes them; the new ones just name the representation:

    | Old | New |
    |---------|-------------|
    | `navis.TreeNeuron` | [`navis.Skeleton`][] |
    | `navis.MeshNeuron` | [`navis.Mesh`][] |
    | `navis.VoxelNeuron` | [`navis.Voxels`][] |
    | `navis.Dotprops` | [`navis.Dotprops`][] (unchanged) |

    The old names still work and will be removed in a future version. They are **aliases**, not subclasses - `isinstance(x, navis.TreeNeuron)` and `class MyNeuron(navis.TreeNeuron)` behave exactly as before, so downstream packages keep working until they get around to it. Neurons pickled by earlier versions still load. `navis.TreeNeuron` raises a `DeprecationWarning` once per session; Python hides those by default outside `__main__`, so run with `python -W default::DeprecationWarning` (or `pytest -W default`) to find the old names in your code.

    The `.type` property follows the class, so it now reads `"navis.Skeleton"` rather than `"navis.TreeNeuron"`. Code matching on that string - including anything filtering a [`NeuronList.summary()`][navis.NeuronList.summary] table by `type` - needs updating.

## Neurons: one schema, masking and attached data

What a neuron *is* made of is now declared in one place (`navis/core/schema.py`) instead of being re-implemented per type. Masking, attached data and a mesh keeping its skeleton all fall out of that.

#### New

- **new [`navis.masked`][navis.masked]: work on part of a neuron, then put it back.** [`navis.subset_neuron`][] is a one-way street - you get a fragment, and its relationship to the neuron it came from is gone. A mask keeps that relationship: inside the block the neuron *is* the masked region, so every function, property and plot sees only that part with no special-casing anywhere; on the way out it becomes whole again, with any edits folded back in. Masking happens **in place** and masks **nest**. Works on `Skeletons`, `Meshes` and `Dotprops`, and on whole `NeuronLists`; the methods underneath ([`Neuron.mask()`][navis.BaseNeuron.mask], [`.unmask()`][navis.BaseNeuron.unmask], [`.apply_mask()`][navis.BaseNeuron.apply_mask]) are there for when a mask has to outlive a block.

    Two new primitives underneath, usable on their own: `subset_neuron(..., track=True)` records where each surviving element came from, and [`navis.merge_subset`][] folds an edited subset back in by joining on that record (and **raises** where the record can no longer be trusted, rather than guessing). Two caveats: merging re-points connectors, tags and the soma at whichever elements survived, so an edit touching none of the elements is silently dropped; and a mask that cuts across branches leaves nodes that look like the ends of the arbour but are not, which anything working from the tips ([`prune_twigs`][navis.prune_twigs], [`strahler_index`][navis.strahler_index], `.leafs`) will act on. See the new [masking tutorial](../generated/gallery/tutorial_basic_03_masking).

- **new [`navis.cast_neuron`][navis.cast_neuron]: convert a neuron's data to a given dtype.** What gets cast depends on the neuron type - node `x`/`y`/`z`/`radius` for skeletons, vertices for meshes, points (and, for float dtypes, tangent vectors and alpha) for `Dotprops`, voxel *values* for `Voxels` - plus connectors for all of them. Anything that *indexes* into those (mesh faces, voxel coordinates, node/parent IDs) is left alone. Handy for e.g. the float32/float64 NBLAST question below

#### Improvements

- **a [`Mesh`][navis.Mesh] now keeps its skeleton through a subset or a mask instead of throwing it away.** The vertex-to-node map is declared as a *link* in the schema (`navis/core/schema.py`) - one array aligned to the mesh's vertices whose values name skeleton nodes - so a selection carries it the same way it carries anything else, and subsets the skeleton to the nodes that still have vertices:

    ```python
    >>> nodes = mesh.skeleton.nodes.node_id.values
    >>> with navis.masked(mesh, lambda x: x.vertices[:, 0] > 0):
    ...     np.isin(mesh.skeleton.nodes.node_id, nodes).all()   # same nodes
    np.True_
    ```

    Previously the skeleton was a plain cache, dropped by any change to the vertices and re-derived from whatever was left - so it cost a full re-skeletonization (~10x the subset itself on the example neuron) and came back with **different node IDs**. Anything computed on a masked mesh's skeleton therefore could not be traced back to the whole neuron. Functions that route through the skeleton ([`navis.prune_twigs`][], [`navis.prune_by_strahler`][], [`navis.split_axon_dendrite`][], ...) get this for free.

    A skeleton is kept only while {{ navis }} can vouch for it: it is regenerated after any change to the vertices that did not go through the schema (assigning `.vertices`, transforming the neuron, [`navis.merge_subset`][]), and after any change to the skeleton that alters which nodes exist. Moving or rerooting the skeleton does *not* invalidate it, since links store node IDs. A skeleton attached by hand (`mesh.skeleton = ...`) behaves as before - there is no correspondence to carry, so it is dropped when the mesh changes.

    Links compose: [`Neuron.get_mapping(source, target)`][navis.BaseNeuron.get_mapping] walks the link graph, so a correspondence nobody declared directly is still available. Deliberately one-way - a vertex has one node but a node has many vertices - so use [`Neuron.select_across`][navis.BaseNeuron.select_across] to go the other way.

- **connectors are now elements in their own right**, on the same footing as nodes and vertices: they have an axis of their own and reach the thing they sit on through a link rather than a bare reference. Three things follow.

    Anything you align to them is carried with them, `get_mapping` composes across them - `connectors → vertices → nodes` resolves on a mesh without anyone declaring the shortcut - and one shared declaration (`schema.CONNECTOR_AXIS` / `schema.connector_link`) now serves every neuron type instead of three near-copies. Your connector table is left exactly as you handed it over: the axis identifies connectors by position, so {{ navis }} has no reason to write an id column into it.

- **new [`Neuron.attach`][navis.BaseNeuron.attach] and [`Neuron.attach_link`][navis.BaseNeuron.attach_link] carry your own data through selections.** Anything attached is subset, filtered and re-indexed by [`subset_neuron`][navis.subset_neuron], [`masked`][navis.masked] and everything built on them, exactly as the neuron's own tables are:

    ```python
    >>> n.attach('compartment', labels, axis='vertices')  # one per vertex
    >>> with navis.masked(n, lambda x: x.vertices[:, 0] > 0):
    ...     n.compartment                                 # masked to match
    ```

    `attach` also takes data that brings its own elements (`n.attach('mito', table, ids='mito_id')` declares an axis), and `attach_link` says which elements of another axis that data names - so a mitochondria table can follow the nodes it sits on, and be dropped when they are:

    ```python
    >>> n.attach('mito', table, ids='mito_id')
    >>> n.attach_link('nodes', 'mito', column='node_id',
    ...               source='mito', target_axis='nodes', cascade='keep')
    ```

    [`Neuron.attached()`][navis.BaseNeuron.attached] lists what a neuron carries beyond its type - attached data is otherwise a plain attribute that nothing in the summary mentions - and [`NeuronList.attached()`][navis.NeuronList.attached] summarises it over a list, counting how many neurons carry each entry:

    ```python
    >>> n.attached()
               name     kind   axis names      shape
    0     embedding  aligned  nodes        (4465, 8)
    1          mito     axis   mito          (30, 2)
    2   nodes->mito     link  nodes  mito    (4465,)
    ```

    Both write per-neuron the same declaration the neuron classes write per-class (`navis/core/schema.py`); `.connectors` is now itself a thin wrapper over `attach`. Note that assigning to an axis is not the same as selecting it - replacing `.connectors` wholesale drops anything aligned to the old connectors, with a warning, since nothing can say where they went. See the new [attaching data tutorial](../generated/gallery/tutorial_basic_04_attach), which also covers what no file format persists.

    Functions that **rebuild** the elements rather than select from them ([`navis.resample_skeleton`][], [`navis.downsample_neuron`][], [`navis.stitch_skeletons`][], [`navis.simplify_mesh`][], ...) drop attached data with a warning rather than leaving it at the old length. Two `on_rebuild` policies opt out of that where a rebuild can say enough:

    ```python
    >>> n.attach('score', values, axis='nodes', on_rebuild='carry')
    >>> navis.downsample_neuron(n, 10).score        # thinning keeps the nodes it keeps
    >>> navis.resample_skeleton(n, 1000).score      # ... re-sampling does not: dropped
    AttributeError
    ```

    `attach_link(..., on_rebuild='snap')` is the same choice for something whose values *name* elements: it does not need a value invented for it, only somewhere to point, so it follows the rebuild to the nearest node of the same branch. That is what connectors, tags and the soma have always done through `resample_skeleton` and `downsample_neuron`, and it is now the generic path rather than three hand-written blocks - so anything you attach can ask for it too.

    A rebuild that **merges** rather than selects can now say so, which is what makes the above work for [`navis.simplify_mesh`][]. It reports where every old element went (`schema.Rebuild(merged=...)`, one entry per old element) - several of them possibly to the same place. That is a weaker claim than "this new element *is* that old one" and a stronger one than "a reference to it should point here": no vertex survives decimation unchanged, but each new one stands for a known group of old ones, and the value of the first of that group is what it takes. So per-vertex data attached with `on_rebuild='carry'` now comes through simplification, and so - via the new `Link(on_rebuild_aligned='carry')` - does a mesh's **vertex-to-node map**: a simplified mesh keeps its skeleton and its node IDs instead of throwing them away and re-skeletonizing into a different set of nodes.

#### Fixes

- **[`navis.subset_neuron`][navis.subset_neuron] has been rewritten onto a declarative per-type schema** (`navis/core/schema.py`) that says which attributes are aligned to which axis - a skeleton's nodes, a mesh's vertices, a dotprops' points - and what references them. The three hand-written implementations had drifted apart in exactly the places nobody was looking; there is now one, plus a test that fails if any bulk field is undeclared. [`navis.masked`][navis.masked] (above) is built on the same schema. Four bugs came out of it:
    - `subset_neuron` **mis-mapped a `Mesh`'s connectors**: it built its old→new vertex index map from the vertex indices *requested*, but `submesh` returns the survivors in sorted order and additionally drops vertices left in no face - so unless the request happened to be sorted and complete, surviving connectors came back attached to the wrong vertex, silently
    - `subset_neuron` **left a `Dotprops`' soma pointing at whatever point had moved into that slot**. A dotprops soma is a point *index*, so a subset has to renumber it; it is now remapped, or dropped if the point did not survive
    - `subset_neuron` did not preserve the node table's **column order** - it moved `parent_id` to the end as a side effect of how it re-rooted the survivors
    - `Dotprops` never took part in the staleness check: the class attribute naming its core data was misspelled `_CORE_DATA`, so `Dotprops.core_md5` was always `None`

- **rebuilding a neuron left data attached to its old elements exactly where it was.** The `.nodes`, `.vertices` and `.points` setters predate the schema and write their private attribute directly, so anything you had attached came out of [`navis.resample_skeleton`][], [`navis.downsample_neuron`][], [`navis.stitch_skeletons`][], [`navis.simplify_mesh`][] and friends describing elements that no longer exist - at the old length, so it still indexed cleanly and nothing complained. Such data is now carried where the function can say which elements it kept, and dropped with a warning otherwise; see `on_rebuild` above.

- **[`navis.insert_nodes`][navis.insert_nodes] and `navis.graph.clinic.merge_duplicate_nodes` wrote the node table straight past everything that keeps a neuron consistent.** Both assigned to the private `_nodes` rather than going through the setter, so neither the attached-data handling nor the reference repair got a say. Anything attached to the nodes was left at the old length - `insert_nodes` on the example neuron gave back 4468 nodes and 4465 labels, and the misalignment survived every later operation - and `merge_duplicate_nodes` left connectors sitting on the duplicate it had just folded away. Inserting is a rebuild and now says so (nothing aligned to the nodes can come along: there is no value to give a node that was not there before, so it is dropped with a warning); merging duplicates is a selection and now goes through it, so attached data is carried - and because a folded duplicate is the *very same point in space* as the node it went into, a connector, tag or soma sitting on it is moved there rather than dropped along with it. A test now fails if a new direct write to `_nodes`/`_vertices`/`_points` appears outside the handful that are allowed

- **[`memory_usage`][navis.BaseNeuron.memory_usage] did not count data held inside something else, and cached a number that changes could not clear.** It walked the neuron's own attributes for arrays and tables, so a mesh's skeleton, the snapshot a masked neuron holds to be restored from and the arrays provenance keeps per axis were all free - on the example mesh, 239 KB of skeleton reported as 0. Attaching data or masking then did not invalidate the cached total either (masking runs with the neuron locked, which is exactly when `_clear_temp_attr` declines to act), so the size could be reported from before the change. Both fixed; caches built by other libraries (`trimesh`, igraph, networkx) are still not counted, being temporary and rebuilt on demand.

- **`memory_usage(estimate=True)` raised on any neuron with connectors, and `NeuronList` reported `0.0B` because of it.** The estimating path prices each column from its dtype using `dtype.itemsize` - which pandas' extension dtypes do not all have, and from pandas 3 a text column defaults to `StringDtype`. [`NeuronList.memory_usage`][navis.NeuronList.memory_usage] caught *everything* and returned 0, which is why this surfaced only as a `NeuronList` claiming to be `0.0B` in its own repr. Columns that can be sized from their dtype still are; the rest are now costed by pandas itself, and estimates are exact on the example neurons. **`NeuronList.memory_usage` now raises instead of returning 0** when it cannot size the neurons - printing a `NeuronList` still can't fail (it shows `?`), but a returned `0` was indistinguishable from a genuinely empty list

## Skeletons: morphology and graph

#### Breaking

- **[`navis.smooth_skeleton`][navis.smooth_skeleton] runs on [navis-fastcore](https://github.com/schlegelp/fastcore-rs), and its window is now centred.** It used to take a *trailing* mean (pandas' `rolling(window).mean()`) along each segment, which lagged the smoothed neurite half a window towards the segment's distal end. It also let branch points move - a branch point is the last node of its segment, so it took a full one-sided mean that the parent segment then read back, dragging the branch's three neurites apart. Both are fixed: the window is centred and shrinks symmetrically as it approaches a segment's ends, and roots, branch points and leafs are pinned. **Coordinates change** - on the example neuron by a median of 54 nm - so this is not a drop-in match for saved output. It is also ~200x faster (351 ms :octicons-arrow-right-24: 1.6 ms for a 4.5k-node skeleton).

    An even `window` now rounds down to the odd value below, since a centred window can only hold an odd number of nodes.

- new: [`navis.smooth_skeleton`][navis.smooth_skeleton] takes a `sigma` in place of `window` to smooth with a **Gaussian kernel whose width is a distance along the neurite** rather than a count of nodes - so the amount of smoothing does not change when the skeleton is resampled, which is usually what you want. The two are mutually exclusive; passing both raises. `to_smooth` works as before with either, and note that `sigma`'s kernel is always measured over the x/y/z coordinates whatever is being smoothed, since a radius is a value and not a geometry
    ```python
    sk_smoothed = navis.smooth_skeleton(sk, sigma=2000)  # these neurons are in nm
    ```

- **[`navis.find_soma`][navis.find_soma] now returns a single node ID (or `None`) instead of an array of candidates.** It used to hand back every node passing the radius/label filter and leave the choice to the caller, which meant a thick primary neurite could be returned alongside - or instead of - the actual soma. Candidates are now scored by the mean radius of their neighbourhood (within `dist_factor` times their own radius, new argument) so that the fattest *region* wins rather than the fattest single node, and the fattest node of that region is returned; the label-only path takes the most central node of the largest connected label component. Nodes whose radius is missing (`NaN` or `<= 0`, as `guess_radius` writes) are no longer treated as candidates. Code doing `find_soma(n)[0]` or `len(find_soma(n))` needs updating

- **skeleton graph functions**, where several results change - some genuinely, some merely from arbitrary to deterministic:

    - **`navis.betweenness_centrality(from_=...)` now counts sources within one hop of a root.** That branch never computed betweenness: it walked root→source paths and tallied every node but the source, which is simply "how many of `from_` lie below this node". It additionally discarded paths of two nodes or fewer, so a source sitting on or next to a root contributed nothing. Those sources now count like any other. In practice only the root and its immediate children move - on the example neuron exactly one node changes, by 3. [`navis.find_main_branchpoint`][navis.find_main_branchpoint] (the one caller) is unaffected, since roots are never branch points

    - **[`navis.collapse_nodes`][navis.collapse_nodes] no longer re-roots the neuron at the collapsed node.** That was a side effect of how it rewired, not a documented behaviour; roots are now left where they were. It also no longer raises on real node IDs - see Fixes

    - **[`navis.rewire_skeleton`][navis.rewire_skeleton] roots leftover components at their lowest node ID** rather than at whatever `set.pop()` returned. Same edge set, same tree; the docstring already promised nothing better than "arbitrary" for those roots, but the choice is now deterministic

    - **`navis.graph.node_label_sorting` breaks ties differently.** Where two branches have exactly equal sort keys their order now follows the node table, deterministically; it used to follow the edge-insertion order of the networkx graph underneath, which carried no stability guarantee at all. On the example neuron 14 of 1217 positions move; the node *set* and the keys themselves are unchanged. This feeds `skeleton_adjacency_matrix(sort=True)`

    - [`navis.betweenness_centrality`][navis.betweenness_centrality] is now the only spelling of the function. The old `navis.betweeness_centrality` is gone

- **[`navis.prune_by_strahler`][navis.prune_by_strahler]'s `relocate_connectors` parameter is gone.** It walked from each pruned node up the parent chain until it hit a survivor and re-attached the connector there - which, since pruning takes whole branches away, could park a synapse a long way from where it actually was. Pruning removes parts of the neuron, and connectors on the removed parts now simply go with them. This does not change any default behaviour: the parameter defaulted to `False`, so unless you passed it explicitly you already had this. Passing it now raises `TypeError`

- **[`navis.downsample_neuron`][] thins skeletons in fastcore now, and picks slightly different nodes.** The Python walk that used to do it counted rootwards from every fix point; fastcore counts from each segment's distal end. Roots, branch points and leafs still always survive and `preserve_nodes` is unchanged, so the result is the same neuron either way - but the surviving node set differs, by about 7% more nodes at `downsampling_factor=5` on the example neuron. Anything pinned to an exact node set needs re-baselining. It is also ~2.3x faster (3.8ms -> 1.7ms on the example neuron) and needs `navis-fastcore >= 0.11.0`.

    Where connectors, tags and the soma end up is unchanged in substance - fastcore hands back a complete map of where every dropped node's data goes, and on the example neuron it agrees with the old geodesic search on 99.7% of dropped connector nodes. The rest are cases where the two surviving nodes are **exactly** equidistant, which the old code resolved arbitrarily and fastcore now resolves towards the root.

- **[`navis.downsample_neuron`][]'s `method` argument now applies to every type of neuron, and an inapplicable one is an error.** It used to be read only by `Dotprops` and silently ignored by everything else. That was harmless while it named a way of picking *points*; it is not harmless now that it also names a way of picking *nodes*, since quietly ignoring `method="rdp"` on a mesh would hand back something simplified by face count and call it shape-aware. `navis.downsample_neuron(mesh, 5, method="uniform")` now raises a `ValueError` naming the methods that type does understand.

#### New

- **[`navis.downsample_neuron`][] can thin skeletons by *shape* rather than by counting**, via `method="rdp"` (Ramer-Douglas-Peucker) and `method="vw"` (Visvalingam-Whyatt), both from [navis-fastcore](https://github.com/schlegelp/fastcore-rs). RDP drops a node unless removing it would move the traced path by more than the tolerance, so long straight stretches collapse to their two ends while a tight curve keeps every node it needs; Visvalingam-Whyatt repeatedly removes whichever node adds least area, which sheds detail more evenly under aggressive simplification - RDP will happily keep one spike and flatten everything around it. Both spend the same node budget where the neuron actually curves, which buys a much better skeleton per node than a fixed factor does: on the example neuron `method="rdp"` at half a micron keeps 1362 nodes against `downsampling_factor=5`'s 1564, for a closer fit.

    For these, `downsampling_factor` is read as a **distance tolerance** in the neuron's own units rather than as a factor - roughly "how far the simplified neuron may stray from this one" - so it takes a unit string (`navis.downsample_neuron(n, "1 micron", method="rdp")`) and has no lower bound. Note this includes `method="vw"`, whose underlying threshold is an *area*: navis squares the tolerance for it so that `method` can be swapped without also rescaling the number.

    Worth knowing for both, and for `method="simple"`: downsampling a skeleton **shortens it**. Survivors keep their coordinates, so the edges replacing a dropped chain cut its corners and `.cable_length` falls with them - by 5-6% on the example neuron at the settings above. Use [`navis.resample_skeleton`][] if you need the cable length left intact.

- **new functions for analyzing the angles in a skeleton** (see the new section in the morphometrics tutorial): [`navis.branch_angles`][navis.branch_angles] (between child branches at each branch point), [`navis.path_angles`][navis.path_angles] (in- vs outgoing edge at each continuation node, i.e. how much the path bends), [`navis.root_angles`][navis.root_angles] (how far each edge deviates from pointing radially away from the root) and [`navis.soma_exit_angles`][navis.soma_exit_angles] (between the neurites emanating from the soma). All return a tidy per-node DataFrame, work on `Meshes` via their skeleton, and map over `NeuronLists`

- **new [`navis.graph.geodesic_clusters()`][navis.graph.geodesic_clusters]**: greedily partitions a skeleton or mesh into connected clusters of bounded geodesic radius. Please read its warning before using it for downsampling - the clusters are guaranteed connected and bounded, but they are *not* evenly sized and their centroids are not evenly spaced

#### Improvements

- **the graph internals now run on [navis-fastcore](https://github.com/schlegelp/fastcore-rs) instead of igraph/networkx.** Each of these was a general graph algorithm answering a question about a *rooted forest*, where the answer is a linear pass over the parent vector - so building a graph object cost more than the answer did. Measured on the example neuron (4465 nodes):

    | Function | Before | After |
    |---|---|---|
    | [`betweenness_centrality`][navis.betweenness_centrality] (`directed=True`) | 11.1 ms | 0.48 ms |
    | [`betweenness_centrality`][navis.betweenness_centrality] (`directed=False`) | 217 ms | 0.49 ms |
    | [`betweenness_centrality`][navis.betweenness_centrality] (`from_=...`) | 10.6 ms | 0.83 ms |
    | [`find_main_branchpoint`][navis.find_main_branchpoint] (`"longest_neurite"`) | 2.8 ms | 0.40 ms |
    | [`find_main_branchpoint`][navis.find_main_branchpoint] (`"betweenness"`) | 9.3 ms | 1.4 ms |
    | [`split_neurites`][navis.split_neurites] (`n=5`) | 20.1 ms | 7.6 ms |
    | [`reroot_skeleton`][navis.reroot_skeleton] | 2.3 ms | 0.75 ms |
    | [`cut_skeleton`][navis.cut_skeleton] | 7.4 ms | 3.6 ms |
    | [`collapse_nodes`][navis.collapse_nodes] | 10.8 ms | 1.9 ms |
    | [`rewire_skeleton`][navis.rewire_skeleton] | 9.5 ms | 2.9 ms |
    | [`edges2neuron`][navis.edges2neuron] (`validate=True`) | 3.5 ms | 0.60 ms |
    | [`cell_body_fiber`][navis.cell_body_fiber] | 12.3 ms | 3.0 ms |
    | `node_label_sorting` | 14.4 ms | 3.5 ms |
    | [`Skeleton.is_acyclic`][navis.Skeleton.is_acyclic] | 2.3 ms | 0.65 ms |

    `betweenness_centrality` gains the most because shortest paths in a tree are unique, so betweenness has a closed form (descendants × ancestors) and needs neither Brandes nor a graph. Values are unchanged - bit-identical to igraph's, counted in `int64` since an undirected 100k-node skeleton reaches ~5e9.

#### Fixes

- **`navis.graph.collapse_nodes` never updated the vertex map it meant to.** It guarded on `hasattr(x, "_vertex_map")`, but the map was stored under the public name, so the branch had never once run and a collapsed skeleton was left with vertices pointing at nodes it no longer had. `Skeleton.vertex_map` is now a validated property backed by `_vertex_map` and the branch fires (and no longer tries to write into skeletor's read-only array)

- **[`navis.downsample_neuron`][navis.downsample_neuron] left connectors and tags pointing at nodes it had just deleted.** It thinned the node table and never touched either, so on the example neuron a `downsampling_factor=10` produced a skeleton whose 2705 connectors included 1704 referring to node IDs no longer in the table. Both are now moved onto the geodesically nearest surviving node, which is the same stretch of the same branch. Pass `preserve_nodes="connectors"` to pin connectors exactly instead of letting them move

- **[`navis.prune_by_strahler`][navis.prune_by_strahler] left tags and the soma pointing at nodes it had just pruned** - the same class of bug. Tags now lose their pruned nodes (and go away entirely if that empties them) and a soma sitting on a pruned node is cleared

- **[`navis.collapse_nodes`][navis.collapse_nodes] raised `MemoryError` on real node IDs.** It built an igraph contraction mapping of vertex *indices* but wrote node IDs into it, so it only held together while IDs happened to run `1..N`. Given the 7e17-range IDs segmentation backends hand out, igraph tried to reserve a vector sized by the ID and died. Everything now happens in ID space

- **`navis.geodesic_matrix(directed=True)` reported a coincident child as reachable from its parent**, at distance 0, whenever both `from_` and `to_` were given. A navis-fastcore fix (`0.10.0`) that {{ navis }} inherits: the partial backend used depth as a proxy for ancestry, which only holds while every edge weight is strictly positive. Coincident nodes are routine in traced and resampled skeletons

- **weighted segment lengths were one edge too long.** `navis.graph.graph_utils._generate_segments(..., return_lengths=True)` summed the weight of every node in a segment including the terminal one, whose own child→parent edge belongs to the *parent* segment. Another navis-fastcore fix {{ navis }} inherits: lengths now measure first node to last, so they sum to exactly the neuron's cable length. [`segment_analysis`][navis.segment_analysis], `.segments` and [`persistence_points`][navis.persistence_points] are unaffected - the over-count never reached them

- **[`navis.sholl_analysis`][navis.sholl_analysis] was broken for most of its `center` options - including the default.** The `"centermass"` branch rebound `center` from a preset name to an x/y/z array *before* the remaining branches compared it against `"soma"`/`"root"`, and on numpy >= 1.25 `array == "soma"` is an elementwise comparison, so the next `if` raised. Separately, a node ID given as a *numpy* integer (e.g. `center=n.root[0]`) failed the `isinstance(center, int)` check, skipped the node → coordinate lookup and was broadcast as a scalar into the distance computation - **returning wrong numbers without raising**. `center` is now resolved by type before any string comparison. Also fixed in passing: `geodesic=True, center="soma"` raised `IndexError` and `radii` did not accept numpy integers. Note that `geodesic=True` now *requires* a center that lies on the arbor (`"root"`, `"soma"` or a node ID) and raises for the default `"centermass"`

## Meshes

#### Breaking

- **[`navis.simplify_mesh`][] runs on [navis-fastcore](https://github.com/schlegelp/fastcore-rs) and the other four backends are gone.** `pyfqmr`, `open3d`, `pymeshlab` and Blender 3D all did the same job, needed installing separately, disagreed about what `F` meant, and - the point - none of them could say what had become of a vertex. fastcore's implementation is the same Garland-Heckbert quadric decimation `pyfqmr` runs, and it returns a **vertex map**: for each old vertex, the new one it was merged into. That is what everything below is built on.

    fastcore is a hard requirement, so there is nothing to install and no backend to choose; `pyfqmr` has been dropped from the `meshes` extra. The `backend` argument is deprecated and ignored (passing anything raises a `DeprecationWarning`), and `navis.meshes.simplify_mesh_fqmr`, `simplify_mesh_open3d`, `simplify_mesh_pyml` and `simplify_mesh_blender` are gone. `**kwargs` now go to `navis_fastcore.simplify_mesh` - `aggressiveness`, `preserve_border` and `lock`, the last of which pins vertices so they are never merged away or moved.

    One thing found along the way and worth reporting on its own: [`navis.smooth_mesh`][] with `backend="auto"` was **silently doing nothing** on any machine with `pyfqmr` installed, since `pyfqmr` came first in the shared backend list and smoothing has no `pyfqmr` branch. Smoothing no longer has backends at all - see below - so the list is gone with them, and `navis.meshes.available_backends` with it.

- **[`navis.smooth_mesh`][] runs on fastcore too, and its default filter has changed.** `open3d`, Blender 3D and `trimesh` are gone the same way `simplify_mesh`'s backends went, and `backend` is likewise deprecated and ignored. Smoothing moves vertices and replaces none of them, so the faces, the vertex count and the vertex order all come back untouched - connectors, extra edges, the skeleton correspondence and anything you attached yourself stay attached to the vertex they were attached to, and unlike simplification there is nothing to repair afterwards.

    **The default is now Taubin's λ|μ filter rather than the plain Laplacian**, which is the part to know about. The Laplacian step removes high frequencies quickly and low ones slowly, and a closed surface's enclosed volume *is* a low frequency: at the settings {{ navis }} ships it costs the example neuron 55% of its volume, and a thin neurite proportionally more. `trimesh` papered over that with a rescale (its `volume_constraint`, on by default) anchored at the **origin** rather than at the mesh, which is not a shape operation - it displaced the example neuron's centroid by 1,963 units, on a neuron 24,442 units across. Taubin alternates a shrinking λ pass with an inflating μ pass tuned so the two cancel below a cut-off frequency, needs no such correction, and holds the same neuron to within 13% while moving its centroid by 0.06 units. `method="laplacian"` and `method="humphrey"` (the HC filter of Vollmer et al.) are there when you want them, and `volume_correction=True` gets the Laplacian back to 99.6% - this time scaled about the mesh's own centroid.

    `L` is deprecated in favour of `lamb`: that is what `navis_fastcore.smooth_mesh` calls it, and it leaves room for the `mu` it pairs with. The remaining `**kwargs` go straight there - `mu`, `alpha`/`beta`, `weights`, `preserve_border`, `lock`, `volume_correction` and `threads`. Two worth knowing about: `weights="cotangent"` is the discrete Laplace-Beltrami operator, which moves vertices along the surface normal instead of sliding them around *within* the surface, and is usually what you want on meshes out of EM segmentation; and `lock` pins vertices bitwise, the same argument `simplify_mesh` has. `preserve_border` defaults to `True` here rather than fastcore's `False`, for the same reason it does in `simplify_mesh` - {{ navis }}' meshes are routinely fragments cut out of a larger volume, and a boundary vertex's one-ring lies entirely to one side of it, so without this every iteration rolls the cut face a little further inwards.

    It is also, incidentally, ~50x faster: ten iterations on a 434k-vertex mesh take 2.26 s through `trimesh` and 0.04 s here.

#### New

- **new [`navis.find_soma_mesh`][navis.find_soma_mesh]: soma detection for [`Meshes`][navis.Mesh], straight off the mesh.** No skeletonization involved: it finds the thickest part of the neuron (the point of largest inscribed sphere) and fits an oriented ellipsoid to the surrounding surface, returning the new [`navis.SomaEllipsoid`][navis.SomaEllipsoid] - `center`, `radii`, principal `axes`, `inscribed_radius`, plus `volume`, `equiv_radius`, `contains()` and `distance_to_surface()`. With `inplace=True` it simply sets the neuron's `.soma_pos`. The approach is inspired by [skeliner](https://github.com/berenslab/skeliner); `min_soma_radius` (accepts e.g. `"1 micron"` if the neuron has units) is the main accept/reject knob and should be tuned to your data

- **new [`navis.heal_mesh`][navis.heal_mesh]: the mesh counterpart to [`navis.heal_skeleton`][navis.heal_skeleton].** Meshes often consist of several disconnected fragments - because the segmentation had a gap, or because meshing produced separate closed surfaces where the neuron is continuous. `heal_mesh` reconnects them with the set of bridges that minimises the total added length (a true minimum spanning tree over the fragments), subject to the same `max_dist`, `min_size`, `mask` and `keep_largest` knobs as its skeleton sibling. The repair is purely topological: bridges land in `.extra_edges` (see below) so vertices, faces, area and volume are all untouched. A 100k-vertex mesh takes ~15 ms

- **new [`Mesh.extra_edges`][navis.Mesh.extra_edges]: connectivity that the surface itself does not have.** A mesh's topology is implied by its faces, so there is no way to express "these two vertices are connected" without inventing geometry. Extra edges are exactly that: an `(N, 2)` array of vertex indices that is part of the *graph* but not of the *surface*. Everything that derives connectivity from a mesh now sees them - [`geodesic_matrix`][navis.geodesic_matrix], [`split_components`][navis.split_components], [`drop_fluff`][navis.drop_fluff], `.igraph`/`.graph` - while anything describing the surface (e.g. `.sampling_resolution`) does not. They are remapped through [`subset_neuron`][navis.subset_neuron] and [`combine_neurons`][navis.combine_neurons], and dropped whenever the number of vertices changes. Note that mesh file formats have no place for them: they are lost on export

#### Improvements

- **[`navis.drop_fluff`][]`(mesh, connectivity=...)`: a piece that only just touches the rest is now fluff you can drop.** Connected components of a mesh have always been components of the *vertex* graph, where a face joins its three corners - so two blobs pinched together at one vertex are one piece, however little that vertex holds them together. There are now two finer readings to pick from, each dropping a kind of junction: `"face"` joins two faces wherever they share an **edge**, which drops the pinch points; `"manifold"` joins them only across an edge carrying **exactly two** faces, which also drops the seams, and reproduces `trimesh.split(only_watertight=False)` exactly - the pieces that are *surfaces*, each with a well-defined inside. The example mesh has 14 pieces under the default, 24 under `"face"` and 502 under `"manifold"`; dropping everything but the largest leaves 17058, 16978 and 15039 vertices respectively. Sizes are counted in vertices whichever you pick, and a vertex that several pieces pinch together stays as long as one of them does.

    Under the hood this is [navis-fastcore](https://github.com/schlegelp/fastcore-rs) 0.13's `mesh_connected_components(..., connectivity=...)`; `graph_utils._connected_components` and `_mesh_component_labels` take the same argument, the latter returning one label per *face* for the two face readings - a pinch vertex belongs to several face components at once, so there is no per-vertex form of that answer. One wrinkle worth knowing if you use [`heal_mesh`][navis.heal_mesh]: an extra edge is a bridge between two *vertices*, so under a face reading it joins the faces at one end to those at the other - and where its endpoint is itself a junction, the pieces meeting there get welded to each other too. That is not a choice so much as arithmetic: they cannot stay apart while both connect to the far end

- **new [`navis.fill_holes`][] closes the holes in a [`Mesh`][navis.Mesh]** - the openings it was cut with, and any it came with. Cutting a mesh (via [`navis.prune_twigs`][], [`navis.prune_by_strahler`][], [`navis.subset_neuron`][], ...) drops every face that loses a corner, which used to leave each severed twig standing open. `fill_holes` triangulates those cross-sections shut:

    ```python
    >>> pruned = navis.prune_twigs(mesh, min_length='5 microns')
    >>> filled = navis.fill_holes(pruned)   # or pruned.fill_holes()
    ```

    Openings are ear-clipped in their own plane rather than filled with a triangle fan, which matters because roughly a third of the cross-sections a prune leaves behind are not convex - a fan spills outside them. This is [navis-fastcore](https://github.com/schlegelp/fastcore-rs) `>= 0.11.0`'s work, so there is nothing extra to install. No vertices are ever added, only faces, so vertex indices - and with them connectors, [`extra_edges`][navis.Mesh.extra_edges] and any tracked provenance - keep meaning what they meant.

    [`navis.subset_neuron`][] also takes a `cap_holes=True`, which closes only the openings that call itself made and leaves pre-existing ones alone. It is the cheaper of the two - it only inspects the collar of faces around the cut, where `fill_holes` has to group the edges of the whole mesh (3.8 ms against 25 ms on a 2.2M-face mesh) - but it is off by default, so nothing changes for existing code unless you ask. At that price capping adds about 2% to the subset it follows.

- a mesh's unique edges now come from navis-fastcore (new `navis.utils.mesh_unique_edges`) instead of `trimesh.edges_unique`, which sorts an `(n_faces * 3, 2)` array to find them. This sits underneath [`neuron2nx`][navis.neuron2nx]/[`neuron2igraph`][navis.neuron2igraph] for `Meshes` and hence everything built on a mesh graph. The results are seeded into trimesh's own cache, so a mesh that has already computed its edges pays nothing

#### Fixes

- **[`navis.simplify_mesh`][] left connectors pointing at vertices it had just deleted.** Decimation replaces a mesh's vertices wholesale, so on the example neuron simplifying to 20% left 83% of the connectors naming a vertex index that no longer existed - and the rest naming whatever had come to sit at that index. Connectors now follow the vertex they named to whatever it was merged into (see the rewrite under Breaking). **Extra edges** survive it too, where they used to be dropped outright with a warning: a bridge names a place on the surface just like a connector does, so it follows its two endpoints (13 fragment-bridging edges on the example neuron come out as 12 - the one that goes is the one whose ends decimation merged into a single vertex, which is no longer an edge at all).

- **[`navis.fix_mesh`][navis.fix_mesh] raised an `AttributeError` on `trimesh >= 4.10`.** `Trimesh.remove_duplicate_faces`/`.remove_degenerate_faces` were replaced by `.unique_faces()`/`.nondegenerate_faces()` in trimesh 3.23 and finally removed in 4.10; `fix_mesh` now picks the right pair based on the installed version. This also unbroke [`Mesh.validate()`][navis.Mesh.validate] and `Mesh(..., validate=True)`, which route through it

- **`Mesh(..., validate=True)` silently did nothing when `process=False`**: it fixed a *copy* of the mesh and threw it away

## Voxels

#### Breaking

- **[`Voxels`][navis.Voxels] now avoid the dense grid wherever possible**, which requires [sparse-cubes](https://github.com/navis-org/sparse-cubes) `>= 0.5.0` - now a **core** dependency, pulled in as `sparse-cubes[skeleton]` so that its [dijkstra3d-sparse](https://github.com/schlegelp/dijkstra3d-sparse) accelerator comes along too (skeletonization falls back to `scipy` without it, but TEASAR is ~11x slower at 100k voxels and the gap widens with size). Materialising a grid larger than `navis.config.max_grid_size` (4 GiB) now raises a `MemoryError` instead of being silently OOM-killed - a neuron's grid is sized by its *bounding box*, so a handful of far-apart voxels can imply terabytes. Raise or disable the limit if you hit it on data you know fits

- [`navis.smooth_voxels`][navis.smooth_voxels] treats everything outside the neuron as empty (scipy's `mode="constant"`). It previously used scipy's default, which *reflects* at the canvas boundary and invents signal outside the imaged volume; results change for neurons touching that edge

#### New

- **[`Voxels`][navis.Voxels] gained a proper toolkit**, all of it working straight off the sparse voxels:
    - morphology and set algebra: `dilate`, `erode`, `opening`, `closing`, `thin`, `fill_cavities`, `union`, `intersection`, `difference`, `symmetric_difference`. Per-voxel values are carried through; set operations align neurons onto a common lattice and refuse to combine ones that do not line up
    - measurements: `surface_area`, `centroid`, `distance_transform`, `connected_components`, `iou`, `dice`, `grid_nbytes`/`voxels_nbytes`
    - shorthands `.mesh()` and `.skeletonize()`

- **[`navis.skeletonize`][navis.skeletonize] now accepts [`Voxels`][navis.Voxels]** (via the new [`navis.conversion.voxels2skeleton`][navis.conversion.voxels2skeleton]), closing a gap its own docstring used to flag. Defaults to `method="wavefront"` - ~4x faster than `"teasar"` and radii come free from the ring contraction rather than being snapped to the voxel lattice; `"teasar"` and `"thin"` remain available

- **existing functions stopped densifying.** [`navis.drop_fluff`][navis.drop_fluff] and `graph_utils._connected_components` now handle `Voxels`; [`navis.smooth_voxels`][navis.smooth_voxels], [`navis.thin_voxels`][navis.thin_voxels] and [`navis.downsample_neuron`][navis.downsample_neuron] no longer allocate the grid (the latter could trip the new memory cap on exactly the sparse neurons worth downsampling). Voxel adjacency - behind `neuron2nx`/`neuron2igraph` - is ~100x faster and no longer needs the *undeclared* scikit-learn dependency. `smooth_voxels`/`thin_voxels` keep a `backend` argument if you want the old scipy/scikit-image route

#### Fixes

- **[`Voxels`][navis.Voxels]:**

    - **a batch of [`Voxels`][navis.Voxels] bugs, most of them on the sparse (voxels + values) backing**, which until now was barely exercised - values and coordinates were free to drift apart. `threshold()` filtered the coordinates but not the values; `normalize()` scaled the *coordinates* instead of the values, corrupting the geometry outright; the documented `(N, 4)` constructor input silently discarded its value column; and changing `.values` did not invalidate a cached `.grid`. Also fixed: `convert_units()` resized the neuron instead of re-labelling it (125x too small for 8 nm → µm), `.volume` squared the z voxel size and dropped y, `.density` crashed on numpy 2, `copy.deepcopy()` raised a `TypeError`, `flip()` moved the neuron and mirrored connectors in the wrong space, and `.bbox` disagreed between the two backings by one voxel

    - **[`Voxels`][navis.Voxels] with no filled voxels raised `ValueError: zero-size array to reduction` on `.shape`** - and hence on `.grid`, `.bbox`, `repr()` and `summary()`. Empty neurons are not exotic - an all-zero grid auto-sparsifies to nothing. `.shape` now falls back to the canvas the neuron was left on, and `strip()`/`normalize()` no-op instead of raising

    - **assigning `.voxels` a different number of voxels left the old `.values` in place**, so `.nnz`/`.volume` kept counting voxels that no longer existed and `.grid` raised a broadcasting error. Mismatched values are now dropped (with a warning); values that still line up row for row are kept. Latent in `1.12.0` but easy to hit now that grids auto-sparsify

    - [`navis.mesh`][navis.mesh] raised `AttributeError` on the `(N, 3)` voxel arrays it documents (it tested `.ndims`, which numpy spells `.ndim`)

## NBLAST and Dotprops

#### Breaking

- **NBLAST scores shift very slightly with [navis-fastcore](https://github.com/schlegelp/fastcore-rs) `>= 0.8.0`**, which now takes its internal coordinate precision from the dtype of the input rather than always widening to float64. {{ navis }} `Dotprops` store `points`/`vect` as **float32**, so NBLAST now runs on float32 coordinates - cutting peak memory on a large all-by-all by ~45%, at a cost of ~1e-5 on the scores. Nothing in {{ navis }} changed; upgrading fastcore is enough to see it.

    The scoring maths itself is untouched (it still accumulates in float64) and this does not change which neurons match - on the example neurons the k-nearest-neighbour identities are unchanged. But it is enough to break a bit-for-bit comparison against previously saved scores: cast `.points` and `.vect` to `float64` (see [`cast_neuron`][navis.cast_neuron] above) if you need the old numbers exactly.

#### New

- **new [`navis.nblast_knn`][navis.nblast_knn]: the `k` nearest neighbours of every neuron, without ever building the score matrix.** An all-by-all is the wrong shape for a k-NN question at scale - 164k neurons is 2.7e10 pairs and a 107 GB matrix, when what is wanted from it is a 26 MB k-NN graph (typically to feed a UMAP embedding). This computes that graph directly: each neuron is reduced to a coarse voxel-occupancy signature, the `n_candidates` most similar neurons per row are shortlisted from those signatures, and the *exact* NBLAST score is then computed for the shortlisted pairs only. Only the shortlisting is approximate - every returned score is a real NBLAST score. Measured on 163,976 neurons, recall@20 is 0.990 at the default `n_candidates`, having scored 0.16% of pairs. Returns a tidy `query`/`target`/`score`/`rank` frame by default; `format="wide"` gives the [`extract_matches`][navis.nbl.extract_matches] layout and `format="arrays"` the raw arrays UMAP's `precomputed_knn` wants. Unlike the other NBLAST functions this one is provided only by navis-fastcore

- **[`navis.make_dotprops`][navis.make_dotprops] is ~12x faster**: the tangent vectors and alpha values (96% of its runtime) now come from one parallel Rust pass instead of a `scipy.spatial.cKDTree` query plus N 3x3 SVDs. Same for [`navis.Dotprops.recalculate_tangents`][navis.Dotprops.recalculate_tangents]. The two agree exactly except where the k-nearest-neighbour search hits a *tied* distance, which grid-quantised coordinates produce readily - there the k-th neighbour is genuinely ambiguous and the two trees may pick differently (~0.3% of points on the example neurons)

- **[`navis.nbl.extract_matches`][navis.nbl.extract_matches] is much faster and gained a `max_matches` guard.** All three criteria now go through navis-fastcore: `N` is 6-118x faster (the gap widens with matrix size - 2.8 s to 23 ms on a 20k x 20k matrix), `threshold` 1.5-8x, `percentage` 1.3-1.9x. `max_matches` refuses to return more than a given number of matches for `threshold`/`percentage`, whose output size is not knowable in advance - an over-broad cutoff on a large matrix could previously take the machine down. The count is established before anything is allocated

- **NBLAST runs on the same backends, so a big one can go to a cluster.** The built-in NBLAST backend used to build its own `ProcessPoolExecutor`, which meant [`navis.set_parallel_backend`][] had no effect on it; it now dispatches through the same layer as everything else. The unit of work is a *block of the score matrix* rather than a neuron, sized from a per-block runtime budget so that each is seconds to minutes of work no matter how many neurons you have. Scores are unchanged - bit-identical, on every backend. On a single machine, where the default is now `joblib` rather than a private pool, a *repeated* NBLAST is **~1.75x faster** because the workers are no longer thrown away and rebuilt between calls (150 neurons all-by-all on 8 cores: 6.3s :octicons-arrow-right-24: 3.6s); they stay resident for a minute or so afterwards and `navis.compute.shutdown()` reclaims them at once.

    !!! warning "navis-fastcore does not distribute"
        The [navis-fastcore](https://github.com/schlegelp/fastcore-rs) NBLAST backend computes the whole matrix in one Rust call with its own threads, so it ignores the parallel backend entirely and runs everything locally. It is not the default (`navis.config.default_nblast_backend` is `"builtin"`), but it is what `"auto"` picks where it is installed - so leave that alone, or pass `backend="builtin"`, when you want a distributed NBLAST.

#### Improvements

- **NBLAST no longer pins its workers to a single thread.** It used to force `OMP_NUM_THREADS=1` in every worker, from a time when nothing else capped native threading and pykdtree's OpenMP would otherwise claim every core in every one of them. Dividing the machine (above) already prevents that, so the pin was leaving NBLAST at a fraction of the cores it had been asked for - 3 of 14 at `n_cores=3`, against 12 now. Scores are unchanged. Applies to [`navis.nblast`][] and relatives on the `builtin` backend, and to [`navis.nblast_align`][].

#### Fixes

- **NBLAST and [`Dotprops`][navis.Dotprops]:**

    - **[`navis.nblast`][] with `scores="both"` crashed on every multi-core run.** With `"both"`, each query occupies *two* rows of the result, but the code that reassembles the score matrix from its blocks assumed one row per query. Any NBLAST split into more than one block therefore died with `ValueError: setting an array element with a sequence`; only runs that happened to fit in a single block ever worked. While there: [`navis.nblast_smart`][] and [`navis.synblast`][] now **reject** `scores="both"` rather than accepting it and getting it wrong - neither ever implemented it

    - **[`navis.nbl.extract_matches`][navis.nbl.extract_matches] ranked `NaN` as the *best* possible score.** numpy sorts `NaN` to the end, so for a query that had been scored against some targets but not others - which is what [`navis.nbl.update_scores`][navis.nbl.update_scores] and any hand-assembled matrix produce - the unscored pair was returned as that query's top match, with a `NaN` score. `NaN`s are now skipped; a query with fewer than `N` valid scores gets empty `match_k`/`score_k` for the remainder. The `percentage` criterion was worse off still: one `NaN` anywhere in a row made that row's threshold `NaN`, so the query got *no* matches at all

    - **`extract_matches(..., percentage=...)` listed matches worst first for distance matrices** - the exact inverse of what it does for similarities, and of what the `N` criterion does for either. Matches are now always best first

    - **[`navis.make_dotprops`][navis.make_dotprops] silently produced wrong tangent vectors for point clouds containing duplicate coordinates.** Points whose `k` nearest neighbours are *all* at distance zero are dropped, but the neighbour indices were then offset by a flat `n_dropped` - only correct if every duplicate happens to sit at the *start* of the array. Anywhere else the indices ran past the end or went negative, and because numpy reads negative indices from the back this raised nothing: it just computed each tangent from an unrelated neighbourhood. On a 40-point cloud with a 4-point duplicate block in the middle, **39 of the 40 surviving points came back with the wrong tangent**

    - **[`navis.Dotprops.recalculate_tangents`][navis.Dotprops.recalculate_tangents] returned `NaN` alpha values** for points sitting on duplicate coordinates - it has no equivalent of `make_dotprops`' duplicate check and cannot drop points. Those `NaN`s then propagated into every NBLAST score the neuron took part in. Such points now get `alpha=0` and an arbitrary unit vector, matching navis-fastcore

## Transforms and alignment

#### Breaking

- **transforms:** how {{ navis }} chooses a route between two template brains has been reworked, and the non-fastcore backends are on their way out:

    - **bridging graph: edge weights now mean one thing, and lower always wins.** `weight` was doing two jobs at once - it set the cost `networkx` minimises when *choosing a route*, and it was also the tie-breaker between several registrations connecting the same two templates (where the old code took the *highest*-weight edge). The two uses want opposite things, so no weighting could satisfy both. Now `weight` only ever means "what this hop costs" and **lower weight = more likely to be used** everywhere; which transform serves a hop is decided separately, by the new `prefer_forward` argument (see below).

        This **changes ~29% of the bridging paths in `navis-flybrains`** (no routes gained or lost). Most of that is {{ navis }} no longer inverting a registration when a purpose-built one for that direction was sitting right next to it - e.g. `BANC`→`Cell07` used to invert `Cell07_IS2.list` and now simply uses `IS2_Cell07.list`, at the same path length. 360 fewer routes traverse *any* transform backwards.

    - transforms now declare how expensive they are to invert, via `BaseTransform.inverse_weight_factor`. It is `1` wherever the inverse is stored or exact (`AffineTransform`, `H5transform`, `TPStransform`), `2` for [`CMTKtransform`][navis.transforms.CMTKtransform] and `5` for [`ElastixTransform`][navis.transforms.ElastixTransform] - both of which have to *solve* for the inverse numerically. `register_transform`'s `weight_inv` now defaults to `weight * inverse_weight_factor`; passing it explicitly still overrides that

    - `reciprocal` is deprecated in favour of `inverse_weight` (`bridging_graph`, `find_bridging_path`, `find_all_bridging_paths`) - one name for one knob. Its default also changed from `0.5` to `1`: {{ navis }} no longer discounts inverse transforms across the board, since each transform now says for itself what inverting it costs. Passing `reciprocal` still works but warns

    - **the non-fastcore transform backends are deprecated and will be removed in 3.0.** The CMTK/elastix `"binary"` backend (shelling out to `streamxform`/`transformix`) and the `"python"` landmark backend (`morphops`/`molesq`) still work, but selecting either now emits a `DeprecationWarning`. `"auto"` - the default - has always preferred fastcore, so this only affects code that asked for them by name. Image transforms (`xform_image`, `to_dfield`) still need CMTK and are not affected

- **alignment runs on [rcpd](https://github.com/schlegelp/rcpd) instead of `pycpd`, which makes it 20-28x faster and fixes four long-standing bugs.** `pycpd` is the reference Python implementation of coherent point drift and has been unmaintained since 2021; `rcpd` is a Rust re-implementation of the same algorithms, and a new **optional** dependency (`pip install navis[cpd]`). On two ~4,500-node skeletons a rigid fit takes **0.3 s and +69 MB against `pycpd`'s 9.6 s and +1.4 GB**, and a deformable one **2.4 s and +96 MB against 48 s and +2.0 GB**. Asked for the same fit, the two agree on where the points land to ~1e-8 of a neuron's extent.

    [`navis.align.align_pairwise`][] fits its whole grid in one batch, one registration per core, for every coherent point drift method - a 5 x 5 alignment of the example neurons takes 6.8 s where `pycpd` would need twenty registrations at ~10 s apiece. A single pair goes the other way and threads *inside* the fit, worth another ~8x on the align-one-neuron-onto-another case. `method="rigid+deform"` and `method="pca"` also work at all now: both used to raise `TypeError`, as did [`navis.nblast_align`][]`(align_method="pca")`.

    **The registration objects you get back are different.** [`navis.align.align_rigid`][] returns `rcpd.RigidTransform` and [`navis.align.align_deform`][] `rcpd.DeformTransform`, in place of `pycpd`'s registration objects. Both move further points with `.apply(coords)` and carry the residual they left behind as `.rms` (in input units) and `.nrms` (as a fraction of the target's own radius, and so comparable between pairs: two neurons of the same type land around 0.06-0.11, two unrelated ones around 0.2). A rigid one also has `.scale`, `.rotation`, `.translation`, a 4x4 `.matrix` and an inverse, `~reg`. **`sample=` no longer runs a moving-least-squares step** over the points it did not fit on: both registrations come back as functions of position rather than as sets of moved points, so they apply to the rest exactly.

    Four behaviours changed, all of them fixes:

    - **`scale=False` is honoured.** `pycpd.RigidRegistration` accepts no `scale` argument in any released version, so the value {{ navis }} passed landed in `**kwargs` and a scale was fitted regardless. `align_rigid` also gains `scale_bounds=(lower, upper)` to hold a fitted scale inside a range you consider plausible
    - **convergence is tested relatively**, on the fractional change in the fitted variance, so the same neurons no longer converge differently in nanometres than in microns. This is what made `align_deform` stop after ~3 iterations: {{ navis }} normalised the neurons first, which put the variance below `pycpd`'s absolute `tolerance=1e-3` before the fit had done anything
    - **`w` is scale-free.** `pycpd` writes the uniform outlier density as a literal 1, which is only right for data spanning ~1 unit; on a neuron spanning 1e4 nm any `w > 0` collapsed the fit. The escalating-`w` retry loop in `align_rigid` is gone with it, along with the `LinAlgError` it worked around - the 3x3 SVD `rcpd` uses cannot fail to converge, and collinear points complete the rotation basis rather than raising
    - **`beta` is a fraction of the neuron, not a length**, so it means the same thing whatever units the neurons are in. This is what used to pull dense regions into a ball: `pycpd`'s default `beta=2`, against neurons {{ navis }} had normalised to ~1 unit, is a kernel several times wider than the whole neuron, so every point moved as one

    `align_deform` exposes `alpha`, `beta`, `w`, `tolerance`, `max_iterations` and `num_modes` as arguments, and defaults to `beta=0.2` where `rcpd` defaults to 1 - a neuron should be free to deform locally rather than as a block. Measured over the ten pairs of example neurons, mean nearest-neighbour distance to the target (as a fraction of its extent) is 0.0089 unaligned, 0.0085 after a rigid alignment and **0.0043** after a deformable one. The neuron normalisation {{ navis }} used to apply before handing anything to `pycpd` is gone - with `beta` and `w` scale-free, nothing needs it.

#### New

- various improvements to transforms:
    - **new [`navis.transforms.SimilarityTransform`][navis.transforms.SimilarityTransform]**: a rotation, a translation and a single uniform scale factor fitted to matched landmarks - the least-squares optimal solution in closed form via SVD (Umeyama, 1991), with optional per-landmark `weights`. Set `scale=False` for a pure rigid transform. Unlike a [`TPStransform`][navis.transforms.thinplate.TPStransform] it has only 7 degrees of freedom and so will generally *not* map the landmarks onto each other exactly - which is the point when you want the landmarks to constrain a global transform rather than warp space to fit them
    - **the landmark transforms now have a fastcore backend.** [`TPStransform`][navis.transforms.thinplate.TPStransform] and [`MovingLeastSquaresTransform`][navis.transforms.moving_least_squares.MovingLeastSquaresTransform] use navis-fastcore's Rust implementation; the deprecated `morphops`/`molesq` path agrees with it to ~1e-13. Neither materialises the `(n_points, n_landmarks)` intermediate those libraries build, so `batch_size` stops mattering and peak memory no longer scales with the landmark count - which is what made `MovingLeastSquaresTransform` impractical at the landmark counts real registrations use (3400 landmarks needed ~23 GB at the default batch size). Transforming points is ~10-15x faster. Note that `TPStransform` still *fits* its spline with `morphops` even on the fastcore backend, because numpy's LAPACK-backed solve is faster there; only the point transform switches
    - **CMTK and elastix point transforms can now run without the external binaries.** [`CMTKtransform`][navis.transforms.CMTKtransform] and [`ElastixTransform`][navis.transforms.ElastixTransform] now use navis-fastcore's in-process Rust implementation instead of shelling out to `streamxform`/`transformix`: no CMTK, no elastix, no subprocess and no temporary files. Results match the binaries to ~1e-6 (including which points fail), and `xform_brain` is 4-30x faster end-to-end. Image transforms (`xform_image`, `to_dfield`) still require CMTK. Control this with `navis.config.default_transform_backend` (`"auto"` by default, i.e. fastcore) or per transform via `backend=`
    - elastix transforms are now *invertible* when navis-fastcore is installed - something `transformix` cannot do at all. Note the bridging graph does not use this by default: set `navis.config.elastix_invertible = True` to let it (see the notes on [`ElastixTransform`][navis.transforms.ElastixTransform] for why it is off)
    - `MovingLeastSquaresTransform` gained a `.matrix_affine` property, analogous to `TPStransform.matrix_affine`: since moving least squares is a *locally* weighted affine transform, this returns the global (least-squares) affine that it converges to far away from the landmarks

- `find_bridging_path`, `find_all_bridging_paths` and `shortest_bridging_seq` gained a `prefer_forward` argument (default `True`): where two templates are connected by both a purpose-built registration and the inverse of its counterpart, use the purpose-built one - regardless of weight. Set it to `False` to have your graph's weights taken entirely at face value

- **new `mirror_axis` for [`navis.align.align_rigid`][]**, for when the neurons may be of either handedness - aligning a left-hand-side neuron onto a right-hand-side one, say, without knowing in advance which is which. Given an axis (`"x"`/`"y"`/`"z"` or `0`/`1`/`2`) each neuron is fitted twice, as given and mirrored along that axis, and whichever fit left the smaller residual (`RigidTransform.rms`) is the one you get back. Both fits go into the same batch as everything else, so across a `NeuronList` this costs twice the registrations but not twice the wall-clock. The reflection is folded into the returned transform where it won - `np.linalg.det(reg.rotation) < 0` says which neurons were flipped, and `.apply()`, `.matrix` and `~` all still hold - and a [`Mesh`][navis.Mesh] that gets mirrored has its face winding flipped with it, so it does not come back inside out.

#### Fixes

- `TransformSequence` was registered as invertible even if it contained a transform that was not

## Plotting

#### Breaking

- **the `vispy` backend is gone.** It was deprecated in `1.7.0` in favour of [octarine](https://schlegelp.github.io/octarine/), which does the same job on modern WGPU instead of OpenGL. `plot3d(..., backend="vispy")` now raises, as does `NAVIS_PLOT3D_BACKEND="vispy"`; the terminal auto-pick is octarine :material-arrow-right-thin: plotly. This removes ~2.3k lines - 28% of the plotting package - and with them the second `Viewer` implementation and the `vispy-*` install extras.

    Concretely, `navis.Viewer` **has been removed**: the viewer is now `octarine.Viewer`, which [`navis.plot3d`][] returns and whose [documentation](https://schlegelp.github.io/octarine/) covers its methods. [`navis.get_viewer`][], [`navis.clear3d`][], [`navis.pop3d`][] and [`navis.close3d`][] are unaffected - they were never vispy-specific. `navis.utils.check_vispy` is gone with no replacement, as are the vispy-only [`navis.plot3d`][] arguments `combine`, `shading`, `shininess` and `name` (`title` is unaffected - it was always plotly-only). `navis.matching.matching_pipeline` has been ported to octarine and behaves the same, except that the vispy-only "cycle through neurons by hiding them" mode is not available.

#### New

- **new [`navis.plot3d`][]`(..., snapshot=True)`: a rendered 3D scene on a matplotlib axes, in data coordinates.** Between [`navis.plot2d`][], which draws real matplotlib artists but has no renderer behind it, and [`navis.plot3d`][], which renders properly but hands back an interactive window, there was no way to get a *figure* you can annotate, arrange into a panel and save as PDF. `snapshot=True` renders the scene through octarine's offscreen canvas and returns `(fig, ax)` with the image already placed - and because the image extent is read back off the camera, `ax` is in the **neurons' own coordinates**, so `ax.annotate("soma", soma_pos[[0, 2]])` lands on the soma and distances are to scale.

    `view` takes the same `("x", "-y")` axis pairs as [`navis.plot2d`][], so the two can be mixed in one figure; pass a camera state dict from `viewer.get_view()` to reproduce a view you set up interactively, or `viewer=` to shoot an existing viewer. Only axis-aligned orthographic cameras can be expressed in data coordinates - for anything else the axes fall back to world units in the view plane. Also new alongside it: `margin`, `bgcolor` (the render is transparent by default), `size`/`pixel_ratio` and the usual `ax`, `figsize` and `dpi`. See the new [3D skeletons tutorial](../generated/gallery/1c_plotting_3d/tutorial_plotting_3d_00_skeletons).

- **new [`navis.plot_collage`][]: a few hundred neurons arranged on one page.** A plate of every cell type in a region, a poster, a supplementary "here is the whole dataset" figure - all of it has meant one subplot per neuron and a lot of fiddling with limits, and a regular grid spends most of the paper on the empty space around each arbor. `plot_collage` does the arranging: it scales and moves *copies* of the neurons onto a page (`page_size`, A4 by default) and plots them, leaving the originals alone. `layout="grid"` puts one neuron per cell (`cols`, `margin`, `uniform_scale`, `sort`, `drop_dangling`); `layout="dense"` packs them at one common scale, so their relative sizes survive. Skeletons, meshes and a mix of the two.

    The dense layout is where the interesting knobs are. By default it packs *bounding boxes* - fast, but a box is reserved for one neuron whether or not it fills it. `occupancy=True` packs the rasterised arbors instead, so a neuron may reach into another's empty space - even into the loop of another neuron - as long as no cable actually collides: on 200 central complex skeletons that draws each neuron **1.37x larger** for the same page, close to twice the area. The box packing runs first either way and seeds the occupancy search, so it can only improve on it. `backfill=` takes a second set of neurons to drop into whatever gaps are left - at the scale `x` settled on, without influencing it, and silently dropping those that find no room - and `mask=` confines the neurons to a shape, given either as a bool array covering the page or as a *picture* of the outline. The packing itself is [navis-fastcore](https://github.com/schlegelp/fastcore-rs)'s (`pack_rectangles`/`pack_masks`), which is what makes searching for the scale affordable.

    Either renderer will draw it: `backend="matplotlib"` gives vector paths you can edit afterwards, `backend="octarine"` renders the page offscreen via [`navis.plot3d`][]`(snapshot=True)` and places the image, which fixes the file size no matter how dense the page gets (160 neurons: 8.9 MB of SVG against 1.0 MB) and shades meshes the way the interactive viewer does. Both put the neurons in data coordinates, so a page looks the same whichever one drew it and overlays land in the same place. The third return value holds the placed copies: hand them back as `placed=` to re-draw that exact page - in different colours, or with the other backend - without packing it again. See the new [collage tutorial](../generated/gallery/1d_plotting_misc/zzz_tutorial_plotting_misc_04_collage).

#### Improvements

- **the matplotlib, plotly and k3d backends assemble their lines in one pass.** Drawing a whole skeleton as a single line wants the segments in one array with the breaks marked, so each backend took the list of per-segment arrays `segments_to_coords` hands back and concatenated it straight back together - using `[[None] * 3]` as the separator, which made the whole array object dtype and pushed the float conversion down to individual coordinates. `navis-fastcore` 0.13 can pad the array as it builds it, and `segments_to_coords` passes that through as `flat=True`. On 100 example neurons (447k nodes, 61.8k segments) assembling the coordinates goes **120 ms :octicons-arrow-right-24: 27 ms**. The breaks are `NaN` rather than `None` now; all three backends already read that as a gap, and plotly's payload shrinks as a side effect (146 KiB :octicons-arrow-right-24: 101 KiB for one example neuron) because a float array serialises to base64 where an object array becomes a JSON list. Renders are pixel-identical.

    Per-node colours ride along on the same call, so they come back already in segment order instead of being matched up afterwards through a dictionary lookup per vertex. That was the slower half of both 3-D backends: for 100 neurons k3d's coloured path goes **672 ms :octicons-arrow-right-24: 38 ms**, and plotly's sheds everything except the `rgb(...)` strings themselves, which is what it is now bound by.

    The paths that genuinely want per-segment arrays - depth sorting, per-edge colouring and `method="3d_complex"` - are unchanged and still get the list.

- **`plot3d(..., backend="plotly", hover_id=True)` no longer raises a `TypeError`.** It built its hover labels with `seg + [None]` over [`navis.Skeleton.segments`][], whose entries are numpy arrays rather than lists, so the addition was attempted element-wise against `None`.

- **[`navis.plot2d`][] (`method="2d"`) has a new renderer**, which between true surface rendering, shading and depth ordering closes much of the gap to a real 3d view:

    - **meshes are rendered as surfaces rather than as flat blobs.** Faces pointing away from the viewer are dropped and the rest painted furthest-first, so a [`Mesh`][navis.Mesh] or [`Volume`][navis.Volume] finally occludes itself. Previously faces went down in whatever order the mesh stored them, which meant a dense arbour collapsed into a single silhouette and the last neuron drawn covered everything under it. There is no switch to turn this off - it is a correctness fix, and because half the polygons never reach the renderer it is also considerably faster (build / draw on the example neurons):

        | | Before | After |
        |---|---|---|
        | one mesh neuron (13k faces) | 32 / 28 ms | 4.8 / 12 ms |
        | five mesh neurons | 241 / 107 ms | 12 / 19 ms |
        | one 209k-face mesh | 513 / 278 ms | 30 / 28 ms |

        A mesh in a single colour is now filled as **one path** instead of one polygon per face, which is where most of that speed-up comes from - and it fixes `alpha`, which used to composite every triangle separately, so a fold in the surface came out darker than a flat stretch.

    - **`mesh_shade` lights meshes and volumes in 2d.** It was a bool for the 3d methods only; with `method="2d"` it now takes a mode - `True`/`"lambert"`, `"cel"`, `"rim"` or `"ghost"` (opacity from the grazing angle instead of brightness, so a neuron inside a neuropil stays visible). Pass a dict to tune `light`, `ambient` and `strength`. Shading *multiplies into* whatever colour a face already has, so it composes with `color`, `color_by` and `depth_coloring` rather than replacing them. `style="publication"` turns it on - pass `mesh_shade=False` alongside it to keep the flat fill. See the new [mesh plotting tutorial](../generated/gallery/1b_plotting_2d/tutorial_plotting_2d_01_meshes)

    - **new `depth_sort` and `halo` arguments**, which between them stand in for the depth buffer matplotlib does not have. `depth_sort=True`/`<int>` buckets skeleton nodes *and* mesh faces into that many bins along the axis pointing into the screen and gives each bin its own z-order, so the two kinds of neuron interleave in one stack instead of every mesh landing on top of every skeleton; a negative number flips which end counts as nearest. `depth_sort="global"` skips the bins and sorts *exactly* - segment by segment, face by face - which for skeletons is cheaper than the default 10 bins but for meshes costs several times either, since sorting across neurons forces one polygon per face and gives up the fill-once `alpha`. `halo` draws each neuron with a background-coloured outline underneath so that crossings read as one neurite passing in front of another. The two compose everywhere except on the `"global"` path - a halo has to sit *between* two neurons of a type, which a single merged artist cannot express, so passing both there warns and falls back to bins

    - **`volume_outlines` draws its contour opaque.** A volume's alpha is a *fill* alpha - there is nothing to see through a line, so at the 10-20% volumes default to the contour was all but invisible. `volume_outlines="both"` already did this; `True` now matches it

    - [`Volume.to_2d`][navis.Volume.to_2d] warns when the alpha shape falls apart and it retries at a tenth of the alpha. The retry is silent no longer: it is why a too-large `volume_outlines_alpha` looks like it did nothing rather than like it was rolled back

- **connector plotting has been overhauled:**

    - **connectors are drawn in one artist per neuron, in a fixed shuffled order.** They used to go down one `type` at a time, which meant the type painted last won wherever markers overlap - so a rare type could bury a common one. On the example neuron that is not hypothetical: 232 presynapses sat on top of 1933 postsynapses and the antennal lobe read as an *output* region. Interleaving the draw order makes the visible mix a fair sample of the real one; the permutation is seeded, so the same data still gives the same figure

    - **`cn_color_by` colours connectors by any column of the connector table** (or by an array with one value per connector) instead of by their `type` - ROI, confidence, partner id, whatever the table carries. Numerical data gets a colormap, categorical data one colour per level, and `cn_palette` picks the palette. The scale is worked out across all neurons in the call, so the same value is the same colour throughout; missing values are drawn grey. Supported by [`navis.plot2d`][] and by `plot3d`'s plotly and k3d backends - the latter two draw markers rather than stalks when it is set, since a line there carries a single colour

    - **`cn_legend=True` explains the connector colours** - one entry per connector *type* for the whole axes rather than one per type per neuron, or a colorbar when `cn_color_by` is numerical. `navis.plot2d` only; call `ax.legend()` afterwards as usual

    - **[`navis.plot2d`][] now honours `cn_layout={"display": "lines"}`**, which draws a stalk from each connector back to the node it belongs to instead of a free-floating marker. That has always been the *default* in `navis.config.default_connector_colors` and has always been what plotly and k3d drew; `matplotlib` silently drew markers regardless. Pass `cn_layout={"display": "circles"}` for the old look. Meshes have no nodes to point at and still get markers

    - **there is a [connectors tutorial](../generated/gallery/1d_plotting_misc/tutorial_plotting_misc_00_connectors)** covering every `cn_*` parameter; connector plotting was previously a three-panel aside in the 3D skeletons page. `cn_zorder` is documented for the first time - it has always worked, and appeared in no docstring

- **the plotting tutorials have been restructured**: *General* (the modes and backends, plus one page on colouring that absorbs the old depth-colouring tutorial), *2D plotting* and *3D plotting* (one page each for skeletons, meshes and volumes), *Other plots* (connectors, barcodes, topology, collages, XKCD) and *Examples*. The styling pages compare settings side by side rather than one render at a time - and the 3D ones do it with `plot3d(..., snapshot=True, ax=...)`, so every panel is a real octarine render. Old URLs under `generated/gallery/1_plotting/` have moved to `1a_plotting_general/`, `1b_plotting_2d/`, `1c_plotting_3d/`, `1d_plotting_misc/` and `1e_plotting_examples/`

#### Fixes

- **the plotting backends now share the code that resolves connectors, connector colors, somata and `shade_by`** (new internal `navis/plotting/_common.py`) instead of keeping a copy each - ~240 lines of near-identical code across matplotlib, plotly and k3d, and, as usual, the copies had drifted:

    - **`connectors=<numpy array of types>` raised `ValueError: truth value of an array ... is ambiguous` in every backend**, even though all three explicitly listed `np.ndarray` among the accepted types. The array was being fed to `bool()` by the guard in front of the filtering
    - **plotly gave somata the wrong color when a per-node colormap was in play** (`color_by=...`). It indexed the color list built for the *line* trace - which is ordered by segment and padded with a black sentinel per segment - with a node-table index
    - **a partial `cn_colors` dict (e.g. `{"pre": "red"}`) broke `plot2d(..., method="3d")`**: the types the dict didn't cover kept their default rgb tuple, and matplotlib can't build an array from a mixed string/tuple sequence
    - **`norm_global=False` raised `TypeError: 'bool' object is not iterable`** for both `color_by` and `shade_by`, and plotly never forwarded `norm_global` to `shade_by` at all

    Two inconsistencies are resolved by construction rather than fixed: k3d ignored `cn_mesh_colors` entirely and checked a `cn_colors` dict *before* `"neuron"`, so it disagreed with the other two on which wins; and the cap on runaway soma detection was applied three different ways (all backends now use plotly/k3d's: 10 or more somata on one neuron are taken as a detection failure and skipped, with a warning). Still outstanding: `cn_colors="neuron"` combined with `color_by=<node property>` is broken in all three backends, because there is no single "neuron color" to give the connectors in that case.

- **more plotting fixes**, each of them independent of the shared-code rewrite above:

    - **[`navis.plot2d`][] and [`navis.plot3d`][] documented several defaults they don't actually use**: `plot2d` advertised `linewidth=.5` (really `1`), `alpha=1` (really `None`), `connectors=True` (really `False`) and `method='3d'` (really `'2d'`); `plot3d` advertised `connectors=True` (really `False`) and `fig_autosize=False` (really `True`). The defaults are now checked against the `Settings` dataclass fields in the test suite, so they can't drift again

    - **[`navis.plot3d`][navis.plot3d] raised for *any* skeleton plotted with `connectors=True` on the `plotly` backend.** {{ navis }} passed `opacity` into plotly's `scatter3d.Line`, which has no such property. Plotly's default connector layout is `display="lines"`, so the only way past it was `cn_layout={"display": "circles"}` or plotting a `Mesh`. Connector transparency now sits on the trace, and an alpha channel on an explicit `cn_colors` is honoured too instead of being emitted as a malformed `rgb(r,g,b,a)` string

    - **`cn_colors` was broken in [`navis.plot2d`][navis.plot2d]** for two of its three documented forms: `"neuron"` was tested against `cn_layout` (a dict, so never equal) and reached matplotlib as a literal color name, and a `{type: color}` dict replaced each whole per-type layout entry rather than just its color

    - **`color_by=<neuron property>` worked only in the `matplotlib` backend.** `plotly` and `k3d` skipped the "is this a neuron property or a per-node property?" resolution and went straight to the per-node path, so e.g. `plot3d(nl, color_by="cell_type", palette="viridis")` raised `ValueError: Column "cell_type" does not exist`. All backends now share one implementation, which also makes `color_by` require a `palette` consistently everywhere

    - **`radius="auto"` was decided once for the whole [`NeuronList`][navis.NeuronList] instead of per neuron.** The auto-detection wrote its verdict back onto the shared settings object, so as soon as one neuron had too few radii every *subsequent* neuron was forced onto lines regardless of its own radii

    - [`navis.plot_flat`][navis.plot_flat]: `normalize_distance=True` raised `KeyError` (it scaled a key that does not exist) and `connectors=True` combined with `highlight_connectors=[...]` raised `TypeError` (the connector angles shadowed the per-node angle lookup). Each worked on its own

    - [`navis.plot2d`][navis.plot2d] with `depth_coloring=True` raised when there were no neurons to normalize against - e.g. when plotting only a [`Volume`][navis.Volume]

    - [`navis.plot3d`][navis.plot3d] with `backend="octarine"` and an explicit `viewer=` raised `AttributeError` unless a {{ navis }}-created viewer already existed in the session

    - [`navis.plot3d`][navis.plot3d]'s `backend` is now case-insensitive: a capitalised but otherwise valid backend (e.g. `"Plotly"`) passed validation and then fell through to "unknown backend"

## Parallelism and threads

#### New

- **new [`navis.Pipeline`][]: chain the operations once, then run the chain.** Chaining {{ navis }} functions over a [`NeuronList`][navis.NeuronList] with `parallel=True` sends every neuron out to a worker and back *once per function*, and where the functions are cheap that transfer **is** the runtime. A pipeline fuses consecutive per-neuron steps into a single task, so each neuron makes the trip once no matter how many steps there are - the more you chain, the better the trade gets, which is the opposite of calling each function with `parallel=True` in turn. It also tracks *ownership*: once a step has handed back something the pipeline made itself, every step after it may modify it in place rather than taking its own defensive copy.

    Steps can be given up front, added one at a time, or simply named - any {{ navis }} function resolves as a method:

    ```python
    pipe = navis.Pipeline().heal_skeleton().prune_twigs(5000).resample_skeleton(1000)
    res = pipe(nl, parallel=True, n_cores=2)
    ```

    Pipelines are **immutable** - [`add`][navis.Pipeline.add] and friends return a new one, so a base can be kept around and branched off, and `|` splices two together. [`add_each`][navis.Pipeline.add_each] and [`add_once`][navis.Pipeline.add_once] override whether a step maps over the neurons or is called once with the whole value. Steps do not have to be {{ navis }} functions, and the input does not have to be neurons - it is whatever the first step accepts, so `navis.Pipeline(neu.fetch_skeletons).resample_skeleton(1000)` takes a query object and everything after it runs per neuron. Running one takes the same `parallel`, `n_cores`, `chunksize`, `backend`, `progress` and `omit_failures` arguments as everything else, and a step that raises comes back as a `PipelineStepError` naming the step and its position. [`NeuronList.pipeline`][navis.NeuronList.pipeline] builds one bound to that list for one-off chains (`nl.pipeline.heal_skeleton().prune_twigs(5000).run()`). See the [multiprocessing tutorial](../generated/gallery/6_misc/tutorial_misc_00_multiprocess).

- **`parallel=True` now runs on a backend you can choose - including one you supply.** Where per-neuron work runs used to be a hard-wired `pathos` pool; it is now pluggable via the new [`navis.set_parallel_backend`][] (also usable as a context manager) and a per-call `backend=` parameter. Ships `pathos`, `joblib`, the standard library's process pool (`processes`), `threads` and `serial`; [`navis.list_parallel_backends`][] shows what's installed and third parties can register their own with `navis.compute.register_backend`. [`NeuronList.apply()`][navis.NeuronList.apply] takes `backend` and `chunksize` alongside the existing `parallel`/`n_cores`/`omit_failures`, all of them explicit parameters rather than forwarded to the applied function. **`parallel=True` therefore no longer needs `pathos`** - though `pathos`/`joblib` are still needed to ship a lambda - and [`navis.set_parallel_backend`][] accepts any `concurrent.futures.Executor`, which is how you point {{ navis }} at a cluster without {{ navis }} ever needing scheduler options of its own.

    `backend="auto"` (the default) prefers `joblib`, then `pathos`, then the standard library's pool. **This changes which backend existing `parallel=True` calls use**: `pathos` was the only option before. The two behave the same, but `joblib` keeps its workers alive between calls where `pathos` builds a fresh pool each time, which makes a sequence of parallel calls measurably faster; pass `backend="pathos"` to keep the old one. Note also that `inplace=False` is only silently turned into an in-place run where the workers get *copies* of the neurons - a thread-based or in-process backend now honours it.

- **`parallel=True` can now run across a cluster**, via two new backends: `dask` (a [dask.distributed](https://distributed.dask.org) cluster, from a `LocalCluster` to a compute centre) and `submitit` (an array job on SLURM). Install with `pip install navis[cluster]` - deliberately *not* part of `navis[all]`. Neither backend is ever selected automatically.

    Nothing about how you call {{ navis }} changes: `parallel=True` still only means "spread this over the neurons", and all scheduler configuration stays on your own object - [`navis.set_parallel_backend`][] takes a `dask.distributed.Client` (or a cluster) and a `submitit.Executor` directly. Both bundle neurons into fewer, larger units of work, aiming for enough units to keep every worker busy while capping each at ~128 MB of neurons; `chunksize=` still overrides it per call. See the [multiprocessing tutorial](../generated/gallery/6_misc/tutorial_misc_00_multiprocess) for both in context.

#### Improvements

- **`parallel=True` no longer fights with the threads underneath it.** {{ navis }} spreads work over processes; [navis-fastcore](https://github.com/schlegelp/fastcore-rs) (and the BLAS/OpenMP pools under numpy) spread it over threads, and by default each takes every core it can see. Nothing told a worker process that it was one of twenty, so the two multiplied - `n_cores=20` on a 224-core node meant 20 x 224 = 4480 threads over 224 cores. Healing 40 skeletons of 200k nodes that way measured **slower than not parallelising at all** (6.71 s vs 5.10 s) while burning 2.3x the CPU; one thread per worker did it in 3.60 s at a sixth of the CPU.

    Each worker is now told what it may use. The default divides the machine up rather than handing all of it to everyone - `cpu_count() // n_cores` threads apiece - and the new `inner_max_num_threads` overrides it (the name is joblib's, for the parameter that does the same job there):

    ```python
    # work with little internal parallelism to spread over more than one thread
    with navis.set_parallel_backend(inner_max_num_threads=1):
        navis.heal_skeleton(nl, parallel=True, n_cores=20)
    ```

    Cluster backends are left alone: `cpu_count() // n_cores` is arithmetic about the submitting machine and says nothing about the node a SLURM job lands on. Pass an explicit value if you want one there.

- **new: [`navis.set_num_threads`][]**, which caps fastcore and BLAS in the *current* process. For the other direction from the above - when you are the one running the pool and {{ navis }} is the thing inside it, where {{ navis }} cannot help itself:

    ```python
    def work(neuron):
        navis.set_num_threads(1)      # or once, in the pool's `initializer`
        return navis.heal_skeleton(neuron)

    with mp.Pool(20) as pool:
        healed = pool.map(work, neurons)
    ```

- **`n_cores` defaults now follow the cores this *process* may use**, via `os.process_cpu_count()` (or the affinity mask below Python 3.13) rather than `os.cpu_count()`. The two differ under SLURM's `--cpus-per-task`, `taskset` and anything else that pins a process to a subset of the machine - i.e. on exactly the machines where claiming cores you do not have hurts most. The NBLAST family's `n_cores` also picks up [`navis.set_parallel_backend`][]`(n_workers=...)`, which it previously ignored: its default was baked in at import time.

#### Fixes

- **parallel processing:**

    - **`parallel=True` could hang forever once the process had done any real work.** `pathos` starts its workers with `fork` on every platform - including macOS, where the standard library does not - and forking is unsafe after native thread pools (BLAS, or Accelerate on macOS) have come up: only the forking thread survives into the child, so the first call that touches one blocks forever with no error, no output and no progress bar. Skeletonizing a handful of meshes was enough to trigger it. {{ navis }} now starts `pathos` workers with `forkserver`/`spawn`, as it already did for the standard-library backend

    - **parallel work printed `resource_tracker: There appear to be N leaked semaphore objects to clean up`.** Building a progress bar - even a disabled one, since the lock is taken in `tqdm.__new__` - makes `tqdm` allocate a cross-process lock, i.e. a named semaphore that nothing then unlinks. Workers now say up front that their bars aren't shared across processes, so no such lock is allocated

    - **`omit_failures=True` mislabelled dataframes with the wrong neuron.** Functions returning a dataframe per neuron (e.g. [`segment_analysis`][navis.segment_analysis]) paired the results with the input neurons by position. Since failed runs are dropped, every dataframe after the first failure got the wrong ID and the last neuron's results were discarded - silently, and on the serial path as much as the parallel one. Also returns an empty dataframe when everything failed, instead of raising from `pd.concat`

    - **`can_zip`/`must_zip` arguments were never actually distributed per neuron.** They were validated against the neuron count and then passed to every neuron whole. In practice this meant [`navis.prune_at_depth`][navis.prune_at_depth] - the only user - raised a broadcast error when given one `source` per neuron, rather than pruning each from its own root

## Morphometrics: IVSCC

#### Breaking

- **[`navis.ivscc_features`][] returns neurons as *rows***, like every other NeuronList-level function in {{ navis }}. It used to build its `DataFrame` straight out of a `{neuron_id: {feature: value}}` dict, which lands features on the index and neurons on the columns - so `.mean()` averaged across neurons per feature only after a transpose nobody remembered. The index is now named `id`, and neurons sharing an ID no longer collapse into one row. Add a `.T` to get the old layout.

    Several features changed meaning at the same time (see Fixes below for the ones that were simply wrong):

    | Old | New |
    |---------|-------------|
    | `mean_contraction` was tortuosity, `L / R >= 1` | contraction proper, `R / L <= 1` |
    | `max_branch_order` was the branch point *count* | the number of bifurcations on the longest root-to-tip path |
    | `basal_dendrite_calculate_number_of_stems` | `<compartment>_num_stems`, and computed for every compartment |
    | `max_euclidean_distance` was the *summed* distance | the maximum distance |

    Feature classes are now constructed from a shared `NeuronContext` rather than from a neuron - `feat(ctx)` instead of `feat(neuron, verbose=...)` - so that rerooting and the distance-to-soma pass happen once per neuron instead of once per feature class. Custom classes passed to `features=` need to subclass `Features` and take a context. `_check_compartments`, which was never called, is gone, and `from navis.morpho.ivscc import *` no longer raises (`__all__` named a function that does not exist).

#### New

- **[`navis.ivscc_features`][] gained the features it was missing**, mostly the ones that need a radius or a third dimension:

    | New feature | Description |
    |---------|-------------|
    | `extent_z` | The depth extent. Only `x` and `y` were measured before, so a neuron's third dimension went unrecorded |
    | `num_tips`, `num_branch_points` | Tip and branch point counts - `num_branches` (linear segments) was the only topology metric |
    | `bifurcation_angle_local`, `bifurcation_angle_remote` | Mean angle between child branches, measured at the branch point and to the next branch/tip respectively |
    | `mean_diameter`, `total_surface`, `total_volume` | Radius-derived size. `total_surface`/`total_volume` model the cable as tapered cylinders and match `Skeleton.surface_area`/[`.volume`][navis.Skeleton.volume] |
    | `parent_daughter_ratio` | Mean ratio of daughter to parent radius across branch points |
    | `early_branch_path` | Path length to the first branch point over the maximum path length - how early the arbor starts splitting |
    | `soma_percentile_x`, `soma_percentile_y` | Fraction of the compartment's nodes above the soma |
    | `soma_radius`, `soma_surface`, `num_stems` | Whole-cell features, from the new `SomaFeatures` |

    `exit_distance` and `exit_theta` are no longer axon-only - every compartment gets them. Features needing a radius are skipped (`NaN`, plus a warning under `verbose=True`) where the node table has none: {{ navis }} fills a missing `radius` column with zeros, which used to come back as `total_surface = 0.0` rather than "not measured".

    There is also a new [IVSCC tutorial](../generated/gallery/2_morpho/zzz_tutorial_morpho_05_ivscc) - the docstring used to point at one that did not exist. It works through 25 Patch-seq mouse cortical cells from the [Brain Image Library](https://www.brainimagelibrary.org), including the part that is easy to get wrong: these features read cortical depth off the `y` axis, and the same cells measured in their unaligned `_Raw` frame come back with `soma_percentile_y` of 0.07 instead of 0.97 and `bias_y` the wrong way round.

- **a second IVSCC tutorial covers [EM data](../generated/gallery/2_morpho/zzz_tutorial_morpho_06_ivscc_em)**, which the first one dismisses in a sentence. It preps 32 proofread MICrONS cells - two per m-type, `L2a` through `L6tall-c` - for [`navis.ivscc_features`][] using only existing {{ navis }} API. The two things a connectome will not hand you are an apical/basal split and a frame where `y` means cortical depth, and both are short: [`navis.subset_neuron`][] plus [`navis.split_components`][] to find the dendritic stem that climbs towards the pia, and one 4x4 matrix through [`navis.xform`][] to undo the ~5° tilt of the cortical column (worth 87 µm of apparent depth per mm of `x` - about half a layer across the dataset). Also covers the QC that catches a cell whose axon label is missing entirely, and what [`navis.split_axon_dendrite`][] does about it.

#### Fixes

- **[`navis.ivscc_features`][] had six features that were quietly wrong.** Most of them came from the same place: a compartment is measured on a *subset* of the neuron, and the subset does not know about the cell it was cut out of.
    - `max_euclidean_distance` was `.sum().max()` on the per-node distances. `.sum()` collapses the series to a scalar and `.max()` on a scalar is a no-op, so the feature was the **sum** of every node's distance to the soma - growing with node count rather than with reach
    - `calculate_number_of_stems` was **always 0**. It counted nodes whose parent is the soma, but the soma is not in the compartment subset and [`subset_neuron`][navis.subset_neuron] had already rewired those stems into roots. Stems are now counted on the full neuron
    - `max_branch_order` was `n_branch_points + 1`, which is a *count*, not an order: a balanced binary tree of depth 5 scored 32 instead of 5, and a 10-stem star 2 instead of 1
    - `mean_contraction` recorded [`navis.tortuosity`][], i.e. `L / R`. Contraction is the reciprocal, `R / L`, and is now computed as such - and note that `mean(R / L)` is not `1 / mean(L / R)` (0.940 vs 0.931 on the example neuron)
    - `exit_distance` and `exit_theta` could describe **different roots**: the distance took the `min` over all of the axon's roots, the angle whichever came first in the node table. Both now use the root closest to the soma. The distance is also clamped at 0 (it went negative for a root inside the soma sphere) and falls back to the soma centre where the radius is unknown, instead of returning `NaN`
    - branch points, branch order, bifurcation angles and `parent_daughter_ratio` **missed a compartment's most proximal fork**. Those read the node table's `type` column, which types the subset's own root as `root` even where it has two children - so a dendrite splitting as it leaves the soma counted zero branch points. They now count children directly, excluding only the soma

    Two crashes are gone with them: `BasicFeatures` on a neuron without a soma returned `None` from `extract_features` and blew up on `dict.update(None)`, and a neuron without a `label` column raised a bare `ValueError` that `missing_compartments` did not catch - so `"ignore"` and `"skip"` did not, in fact, ignore or skip it. `missing_compartments` and `x` are now validated up front rather than falling through to `raise`.

## Machine learning

#### New

- **new `navis.ml` module: helpers for preparing neurons as machine-learning inputs.** These live under their own `navis.ml.*` namespace (deliberately *not* lifted to the top level) and come with three tutorials:
    - **normalize** - [`navis.ml.normalize_neuron`][navis.ml.normalize_neuron] maps a neuron into a canonical frame with a single rigid-plus-uniform-scale transform: center on the centroid/bbox/soma/an explicit point, PCA-orient (signs disambiguated deterministically and handedness preserved - neurons are never mirrored), and scale to unit RMS / extent / enclosing sphere. `return_matrix=True` returns the 4x4 matrix so predictions made in the normalized frame can be mapped back
    - **augment** - geometry/sampling augmentations, each taking a `random_state` and returning a copy: [`jitter_neuron`][navis.ml.jitter_neuron], [`warp_neuron`][navis.ml.warp_neuron], [`rotate_neuron`][navis.ml.rotate_neuron], [`scale_neuron`][navis.ml.scale_neuron], [`translate_neuron`][navis.ml.translate_neuron] and [`drop_nodes`][navis.ml.drop_nodes]. [`augment_neuron`][navis.ml.augment_neuron] chains them in one seeded, reproducible call
    - **sample & chunk** - turn a variable-sized neuron into a fixed-size model input: [`sample_cable`][navis.ml.sample_cable] (arclength-uniform sampling along a skeleton), [`sample_surface`][navis.ml.sample_surface] (area-weighted mesh-surface sampling), both with per-point provenance, plus [`chunk_neuron`][navis.ml.chunk_neuron] (tile a neuron into evenly-sized fragments for batching)

## I/O and interfaces

#### Breaking

- **the `rpy2`-based R interface is gone.** It required a working R installation plus `rpy2` on top of it, and existed mainly to move data in and out of the [natverse](https://natverse.org). That job is now done by plain files - [`navis.read_rda`][], [`navis.read_rds`][], [`navis.write_rda`][] and [`navis.write_rds`][] (see below), which need neither R nor `rpy2`. `from navis.interfaces import r` now raises with a pointer to the replacement, and the `navis[r]` install extra no longer exists. See the new [natverse tutorial](../generated/gallery/0_io/tutorial_io_03_r) for the round trip.

    | Removed | Use instead |
    |---------|-------------|
    | `r.neuron2r()` / `r.neuron2py()` | [`navis.write_rds`][] / [`navis.read_rds`][] |
    | `r.load_rda()` | [`navis.read_rda`][] |
    | `r.nblast()` / `r.nblast_allbyall()` | [`navis.nblast`][] / [`navis.nblast_allbyall`][] |
    | `r.xform_brain()` / `r.mirror_brain()` | [`navis.xform_brain`][] / [`navis.mirror_brain`][] |
    | `r.get_neuropil()` / `r.get_brain_template_mesh()` | [flybrains](https://github.com/navis-org/navis-flybrains) |
    | `r.init_rcatmaid()` | [pymaid](https://pymaid.readthedocs.io) |

- **the Cytoscape interface is gone.** `navis.interfaces.cytoscape` pushed a network into a running [Cytoscape](https://cytoscape.org) desktop over cyREST, via [py2cytoscape](https://github.com/cytoscape/py2cytoscape) - a dependency that is no longer maintained and was never declared as an install extra. It was deprecated in the API docs and nothing in it was navis-specific: `generate_network()` handed a `networkx` Graph or a DataFrame straight to the cyREST client, and `get_navis_style()` set some colours. `from navis.interfaces import cytoscape` now raises with a pointer to the replacement - build the graph with [`navis.network2nx`][] and hand it to Cytoscape as a file (`networkx.write_graphml(g, "network.graphml")`), or drive cyREST yourself.

- **`navis.interfaces.neuromorpho.find_neurons` no longer prompts.** Called with no filters it used to ask `"No filters will list all neurons. Continue? [Y/N]"` on stdin - a library function blocking on a human who, in a script or a server, is not there. It now raises and tells you to pass either a filter or a `page_limit`. `navis.interfaces.vfb.get_skeletons` likewise returns an empty `NeuronList` rather than `None` when nothing matches

#### New

- **new [`navis.write_parquet`][]`(..., format="neurarrow")`: parquet files other tools can read.** [neurarrow](https://neurarrow.readthedocs.io) is a specification for neuroanatomical data on Apache Arrow which started from {{ navis }}' own parquet format and has since gone its own way - different column names, `uint64` IDs unique across the whole file, `float64` coordinates, null rather than `-1` for roots, and required metadata. `format="neurarrow"` writes all of that, and [`navis.read_parquet`][] detects and reads neurarrow files without being told - including those written by other tools, such as [swc2na](https://github.com/clbarnes/swc2na).

    Two things don't map cleanly: neurarrow tracks units for the **file**, not the neuron, so all neurons must share the same units (and dotprops the same `k`); and navis' connector table is a set of point annotations where neurarrow's `connections` schema describes *edges*, so connectors are written through the [`net.clbarnes.connector`](https://github.com/clbarnes/neurarrow-ext/blob/main/extensions/connector.md) extension, which has no room for extra columns like `roi`. `format="navis"` remains the default and the only lossless round-trip - see [the format spec](https://github.com/navis-org/navis/blob/master/navis/io/pq_io.md) for the full comparison.

- **new [`navis.write_rds`][navis.write_rds] and [`navis.write_rda`][navis.write_rda]: hand data to R's [natverse](https://natverse.org) without `rpy2`.** `Skeleton` :material-arrow-right-thin: `nat::neuron` (including nat's `SegList`/`SubTrees` topology, so `resample()`, `prune_*()` etc. work on the result), `Dotprops` :material-arrow-right-thin: `nat::dotprops`, `Mesh`/`Volume` :material-arrow-right-thin: `rgl::mesh3d`, `Voxels` :material-arrow-right-thin: `nat::im3d` and `NeuronList` :material-arrow-right-thin: `nat::neuronlist` (with its meta data attached); plain dicts, lists, arrays and `DataFrames` become the corresponding R types. The serialisation is done by [rdata](https://github.com/vnmabus/rdata), so no R installation and no `rpy2` are required on either end; read the files in R with `readRDS()`/`load()`

- **new [`navis.read_rds`][navis.read_rds]**, the counterpart to [`navis.read_rda`][navis.read_rda] for `.rds` files (a single, unnamed R object rather than a whole workspace)

- **`neuprint` interface: new `dedup` option for fetching synapses.** In insect connectomes synapses are polyadic - a single presynapse connects to multiple postsynapses - and `neuprint-python` de-duplicates presynapses so each is reported only once. [`fetch_synapses`][navis.interfaces.neuprint.fetch_synapses] (a new thin wrapper over `neuprint.fetch_synapses`), [`fetch_skeletons`][navis.interfaces.neuprint.fetch_skeletons] and [`fetch_mesh_neuron`][navis.interfaces.neuprint.fetch_mesh_neuron] now take `dedup` (default `True`, i.e. unchanged behaviour). With `dedup=False` each presynapse is instead reported once per postsynaptic site it connects to

- **new function: [`navis.patch_caveclient`][]** monkey patches `caveclient` to return {{ navis }} neurons, the same way [`navis.patch_cloudvolume`][] does for `cloud-volume`. It wraps the skeleton service - `client.skeleton.get_skeleton`, `get_bulk_skeletons` and `fetch_skeletons` - giving each an `as_navis=True` keyword argument plus a `*_navis` twin that always converts. Both `output_format`s are handled, which matters more than it sounds: `"dict"` is in **nanometres** and `"swc"` in **micrometres**, and the neurons come back with `.units` set accordingly. The service's per-vertex compartment (`1` soma, `2` axon, `3` dendrite) lands in the node table's `label` column, so [`navis.ivscc_features`][] and `color_by="label"` work on the result without further ado.

    Two differences to `patch_cloudvolume`: it patches the client *class*, so it can be called after the client exists and patching twice is a no-op; and because the bulk endpoints cap how many skeletons they return (ten, at time of writing) and drop the rest silently, {{ navis }} warns when it gets fewer skeletons back than you asked for - a truncated `NeuronList` looks a lot like a complete one.

#### Improvements

- **the remote data-source interfaces now share their plumbing** (new `navis/interfaces/base.py`). Each of `microns`, `h01`, `neuprint`, `neuromorpho`, `insectbrain_db`, `vfb`, `allen_celltypes` and `brain_image_library` had grown its own copy of "fan a fetch out over a thread pool", twelve in all - and eight of them ended in the *same* line, `print(f"{id} generated an exception:", exc)`. A failed neuron was written to stdout and dropped, so a partial result came back looking like a complete one. There is now one `fetch_parallel`, and with it:

    - **results come back in the order you asked for them.** The old loops appended as requests completed, so the order was whatever the network happened to do; two modules re-sorted afterwards to compensate and the rest simply returned neurons shuffled
    - **a failure is reported through `errors`**, not `print` - `"raise"`, `"log"` (default) or `"ignore"`, matching what [`navis.read_swc`][] and friends already take. Available on every fetch function
    - `cave_utils.fetch_neurons` treated a failure differently depending on whether `parallel` was `True` or `False` (swallowed vs raised). Both now follow `errors`

- **new `navis.config.strict` for server and pipeline contexts** (or `NAVIS_STRICT=1`). It does not change *what* {{ navis }} computes - only how it behaves when something goes wrong or when it would otherwise want a human: remote fetches default to `errors="raise"` so a partial result is loud rather than silent, and nothing prompts for input. Carried into worker processes like the other settings

- **importing an interface no longer requires its dependencies - or a network connection.** `navis.interfaces.vfb` built a live `VfbConnect` client *at import time*, so merely importing it phoned VirtualFlyBrain; `allen_celltypes` wrote an `allensdk` manifest into your working directory on import, and it and `vfb` both raised outright if their dependency was missing (while `microns`/`cave_utils` logged and left a `None` behind to trip over later). Every optional dependency now goes through one `optional_import`, which defers the failure to first *use* and reports it with the `pip install` line to fix it. `vfb.get_client()` and `vfb.vc` still give you the client, on demand

- **one HTTP session for the whole library** (new `navis.utils.http`). `navis.io`, the Brain Image Library and NeuroMorpho each built their own - or, in NeuroMorpho's and part of InsectBrain DB's case, used bare `requests.get` with no pooling at all. All now share a pooled, retrying session that identifies {{ navis }} in its User-Agent; URL reads pick up retries on 429/5xx as a side effect. Insect Brain DB keeps its own session, since its `Authorization` header must not leak onto anything else

- **every interface cache is now reachable from one place**: `navis.interfaces.clear_cache()`. There were nine - eight `lru_cache`s scattered across `cave_utils`, `microns` and `insectbrain_db` plus the Brain Image Library's metadata dict - and only the last of them had a way to clear it

#### Fixes

- **[`navis.write_parquet`][navis.write_parquet] silently dropped connectors.** It only ever wrote the node table, so round-tripping any neuron with synapses gave it back with `connectors=None` - no warning, and neither the docstring nor the format spec mentioned it. Connectors now go into a *sidecar* file next to the main one (`neurons.parquet` gets a `neurons.connectors.parquet`), which [`navis.read_parquet`][] picks up automatically; two files rather than a zip archive, so both tables keep parquet's column pruning and predicate pushdown. Pass `write_connectors=False` to opt out. The node table's `label` column was going missing the same way and is now written too

- **[`navis.read_parquet`][navis.read_parquet] returned a `NeuronList` for a single-neuron file**, contradicting its own docstring, because dotprops always carried the neuron-ID column. A file holding exactly one neuron now reads back as a single neuron for both skeletons and dotprops; `subset=`/`limit=` still always give you a `NeuronList`

- **reading with a list `limit` returned nothing at all** for most sources. `limit=["neuron.swc"]` is documented as a list of filenames to pick out of a folder or archive, but the readers filtered it with `f in limit` against whatever they happened to be holding - `Path` objects for a folder, `ZipInfo` objects for a zip, neither of which compares equal to a string - so [`navis.read_swc`][navis.read_swc]`(dir, limit=["neuron.swc"])` came back empty. Tar archives and Google Storage buckets kept *full* paths, so a plain filename missed there too; of the five call sites only FTP, which happens to store bare filenames, ever worked. A list `limit` now matches either a filename or a full path everywhere. Note that a filename is matched against the last path component, so one entry can pick up files of that name in several subdirectories - give the full path to disambiguate. The lookup is also hashed once rather than scanned per file, which takes filtering a large library from quadratic to linear

- **`neuprint` interface: [`fetch_skeletons`][navis.interfaces.neuprint.fetch_skeletons] shared one client - and hence one `requests.Session` - across all its worker threads**, which `requests` does not support. `neuprint-python` normally hands each thread its own deepcopy of the client, but passing an explicit client into a thread bypasses that; {{ navis }} now does the copying itself in the pool's initializer

- `navis.interfaces.brain_image_library`: a failing *root* directory listing now raises instead of quietly returning an empty file table, and partial listing failures further down warn that the result is incomplete rather than passing it off as the whole dataset

- [`navis.conversion.mesh2skeleton`][navis.conversion.mesh2skeleton] ignored the `progress` keyword argument it accepts

- **[`navis.read_precomputed`][] silently shattered skeletons whose edges are not in the column order it expected.** The [precomputed spec](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/skeletons.md) leaves edges *undirected* - neither column is promised to be the parent - but the reader read column 0 as the parent and column 1 as the child, and built the parent mapping with a `dict` keyed on column 1. In a file written the other way round, every branch point repeats in column 1 and each repeat overwrites the last, so those edges vanished without a word: the male CNS skeletons, for instance, came out with 378 roots where the graph actually has 4. Edges are now oriented by traversal, which also breaks any cycles. Where one column *can* be the child column - i.e. it holds each vertex at most once - the vertices missing from it are used as the roots, so a file with a consistent orientation (including anything {{ navis }} wrote) keeps the roots it meant to have

## Packaging, dependencies and internals

#### Breaking

- **[navis-fastcore](https://github.com/schlegelp/fastcore-rs) is now a required dependency (`>= 0.10.0`), and the pure-Python/igraph/scipy fallbacks are gone.** {{ navis }} has shipped two implementations of its graph, geodesic, morphometric and NBLAST core for several releases - a fastcore fast path and a fallback for installs without it - and kept them agreeing bit-for-bit. That is over: fastcore is the engine.

    For anyone installing with `pip`, nothing changes but the install: fastcore ships prebuilt wheels for macOS (Intel/ARM), Windows, and Linux on x86-64, aarch64, i686, armv7l, ppc64le and s390x. Only musl-based Linux (Alpine) builds from source and therefore needs a Rust toolchain. `pip install navis[fastcore]` still resolves - it is now a no-op alias.

    Two behavioural notes. `navis.geodesic_matrix` on a `Mesh` no longer accepts `directed=True` as anything but a no-op (it was already ignored for meshes). And the private helpers behind skeleton stitching - `_stitch_edges`, `_segment_radii`, `_rewire_from_edges`, `_component_labels` - have been removed along with the KDTree stitcher they made up; [`heal_skeleton`][navis.heal_skeleton] and [`heal_mesh`][navis.heal_mesh] are unaffected.

#### Improvements

- **two dependencies are gone and two are no longer imported up front.** `six` was a Python-2 compatibility shim and `pypng` had not been imported anywhere for several releases; both are dropped from the requirements. `morphops` and `molesq` are now imported on first use rather than at module level, taking ~280 ms off `import navis`. `molesq` is now only reached by the deprecated `"python"` backend; `morphops` is still needed by every thin plate spline transform, on either backend, since it does the *fit*

#### Fixes

- **{{ navis }}' logger could be left silenced for the rest of the session.** [`arbor_segregation_index`][navis.arbor_segregation_index], [`bending_flow`][navis.bending_flow], [`synapse_flow_centrality`][navis.synapse_flow_centrality] and [`flow_centrality`][navis.flow_centrality] quieten the logger while they build a throwaway downsampled copy, and a neuron's HTML thumbnail does the same while it plots; if any of those steps raised, the whole library went quiet with nothing to say why (and the thumbnail additionally left matplotlib's interactive mode off and a stray figure open). All now go through the new `navis.config.quiet_logger` context manager

- **`pip install navis[all]` pulled in `cloud-volume` and its dependency tree**, which was never the intention - it is a specialized dependency and is meant to be installed on its own (`navis[cloudvolume]`). The exclusion list named the *distribution* (`cloud-volume`) where it needed the *extra* (`cloudvolume`), so it silently matched nothing. `r`, `flybrains` and `cluster` were unaffected

