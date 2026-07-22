---
icon: simple/keepachangelog
hide:
  - navigation
---

# :simple-keepachangelog: Changelog

This is a selection of features added, changes made and bugs fixed with each version.
For a full list of changes please see the [commits history](https://github.com/navis-org/navis/commits/master) on
{{ navis }}' Github repository.

## dev
_Date: ongoing_

To install the current `dev` version of {{ navis }}:

```shell
pip uninstall navis -y
pip install git+https://github.com/navis-org/navis@master
```

##### Breaking
- **NBLAST scores shift very slightly with [navis-fastcore](https://github.com/schlegelp/fastcore-rs) `>= 0.8.0`.** fastcore now picks its internal coordinate precision from the dtype of the input rather than always widening to float64. {{ navis }} `Dotprops` store `points`/`vect` as **float32**, so NBLAST now runs on float32 coordinates - cutting peak memory on a large all-by-all by ~45%, at a cost of ~1e-5 on the scores. Nothing in {{ navis }} changed; upgrading fastcore is enough to see it.

    The scoring maths itself is untouched (it still accumulates in float64) and this does not change which neurons match: on the example neurons the k-nearest-neighbour identities are unchanged. But it is enough to break a bit-for-bit comparison against previously saved scores. Cast `.points` and `.vect` to `float64` before NBLASTing if you need the old numbers exactly.
- **bridging graph: edge weights now mean one thing, and lower always wins.** `weight` was doing two jobs at once - it set the cost `networkx` minimises when *choosing a route*, and it was also used to pick between several registrations connecting the same two templates (the old code took the *highest*-weight edge there). The two uses want opposite things, so no weighting could satisfy both: to stop an inverse transform pulling unrelated routes through it you had to weight it *up*, but to stop it being picked over a purpose-built registration you had to weight it *down*.

    Now `weight` only ever means "what this hop costs", and **lower weight = more likely to be used** everywhere - both when routing and when picking between parallel registrations. Which transform serves a hop is decided separately, by the new `prefer_forward` argument (see below).

    This **changes ~29% of the bridging paths in `navis-flybrains`** (no routes gained or lost). Most of that is {{ navis }} no longer inverting a registration when a purpose-built one for that direction was sitting right next to it - e.g. `BANC`→`Cell07` used to invert `Cell07_IS2.list` and now simply uses `IS2_Cell07.list`, at the same path length. 360 fewer routes traverse *any* transform backwards.
- transforms now declare how expensive they are to invert, via `BaseTransform.inverse_weight_factor`. It is `1` wherever the inverse is stored or exact (`AffineTransform`, `H5transform`, `TPStransform`), `2` for [`CMTKtransform`][navis.transforms.CMTKtransform] and `5` for [`ElastixTransform`][navis.transforms.ElastixTransform] - both of which have to *solve* for the inverse numerically. `register_transform`'s `weight_inv` now defaults to `weight * inverse_weight_factor`; passing it explicitly still overrides that.
- `reciprocal` is deprecated in favour of `inverse_weight` (`bridging_graph`, `find_bridging_path`, `find_all_bridging_paths`) - one name for one knob. Its default also changed from `0.5` to `1`: {{ navis }} no longer discounts inverse transforms across the board, since each transform now says for itself what inverting it costs. Passing `reciprocal` still works but warns.
- **[`VoxelNeurons`][navis.VoxelNeuron] now avoid the dense grid wherever possible**, which requires [sparse-cubes](https://github.com/navis-org/sparse-cubes) `>= 0.5.0` - now pulled in as `sparse-cubes[skeleton]`, so that its [dijkstra3d-sparse](https://github.com/schlegelp/dijkstra3d-sparse) accelerator comes along too (skeletonization falls back to `scipy` without it, but TEASAR is ~11x slower at 100k voxels and the gap widens with size). Materialising a grid larger than `navis.config.max_grid_size` (2 GiB) now raises a `MemoryError` instead of being silently OOM-killed - a neuron's grid is sized by its *bounding box*, so a handful of far-apart voxels can imply terabytes. Raise or disable the limit if you hit it on data you know fits.
- [`navis.smooth_voxels`][navis.smooth_voxels] treats everything outside the neuron as empty (scipy's `mode="constant"`). It previously used scipy's default, which *reflects* at the canvas boundary and invents signal outside the imaged volume; results change for neurons touching that edge.

##### Additions
- **[`navis.make_dotprops`][navis.make_dotprops] is ~12x faster** when [navis-fastcore](https://github.com/schlegelp/fastcore-rs) `>= 0.8.0` is installed: the tangent vectors and alpha values (96% of its runtime) now come from one parallel Rust pass instead of a `scipy.spatial.cKDTree` query plus N 3x3 SVDs. Same for [`navis.Dotprops.recalculate_tangents`][navis.Dotprops.recalculate_tangents]. The scipy route remains as a fallback.

    The two agree exactly except where the k-nearest-neighbour search hits a *tied* distance, which grid-quantised coordinates produce readily - there the k-th neighbour is genuinely ambiguous and the two trees may pick differently. That affects ~0.3% of points on the example neurons.
- **new [`navis.graph.geodesic_clusters()`][navis.graph.geodesic_clusters]**: greedily partitions a skeleton or mesh into connected clusters of bounded geodesic radius. Please read its warning before using it for downsampling - the clusters are guaranteed connected and bounded, but they are *not* evenly sized and their centroids are not evenly spaced. Uses navis-fastcore if available, else scipy.
- **new [`navis.nblast_knn`][navis.nblast_knn]: the `k` nearest neighbours of every neuron, without ever building the score matrix.** An all-by-all is the wrong shape for a k-NN question at scale - 164k neurons is 2.7e10 pairs and a 107 GB matrix, when what is wanted from it is a 26 MB k-NN graph (typically to feed a UMAP embedding). This computes that graph directly: each neuron is reduced to a coarse voxel-occupancy signature, the `n_candidates` most similar neurons per row are shortlisted from those signatures, and the *exact* NBLAST score is then computed for the shortlisted pairs only. Only which neurons get shortlisted is approximate - every returned score is a real NBLAST score. Measured on 163,976 neurons, recall@20 is 0.990 at the default `n_candidates`, having scored 0.16% of pairs.

    Returns a tidy `query`/`target`/`score`/`rank` frame by default; `format="wide"` gives the [`navis.nbl.extract_matches`][navis.nbl.extract_matches] layout and `format="arrays"` the raw index/score arrays UMAP's `precomputed_knn` wants.

    Note this requires [navis-fastcore](https://github.com/schlegelp/fastcore-rs) - unlike the other NBLAST functions there is no built-in Python implementation to fall back on, and it raises a pointed error if fastcore is missing or too old.
- **the landmark transforms now have a fastcore backend too.** [`TPStransform`][navis.transforms.thinplate.TPStransform] and [`MovingLeastSquaresTransform`][navis.transforms.MovingLeastSquaresTransform] use navis-fastcore's Rust implementation when it is available, falling back to `morphops`/`molesq` otherwise (the two agree to ~1e-13). Neither materialises the `(n_points, n_landmarks)` intermediate those libraries build, so `batch_size` stops mattering and peak memory no longer scales with the landmark count - which is what made `MovingLeastSquaresTransform` impractical at the landmark counts real registrations use (3400 landmarks needed ~23 GB at the default batch size). Transforming points is ~10-15x faster.

    `TPStransform` still *fits* its spline with `morphops` even on the fastcore backend, because numpy's LAPACK-backed solve is faster there than fastcore's; only the point transform switches. That keeps the fastcore backend from ever being slower end-to-end (fastcore's own fit is cubic and would regress small one-shot transforms).

    Steered by the same `navis.config.default_transform_backend` as CMTK/elastix, or per transform via `backend=`. The non-fastcore option is spelled `"python"` here and `"binary"` there; both names are accepted for either.
- **CMTK and elastix point transforms can now run without the external binaries.** If [navis-fastcore](https://github.com/schlegelp/fastcore-rs) is installed, [`CMTKtransform`][navis.transforms.CMTKtransform] and [`ElastixTransform`][navis.transforms.ElastixTransform] use its in-process Rust implementation instead of shelling out to `streamxform`/`transformix`: no CMTK, no elastix, no subprocess and no temporary files. Results match the binaries to ~1e-6 (including which points fail), and `xform_brain` is 4-30x faster end-to-end. Image transforms (`xform_image`, `to_dfield`) still require CMTK.

    Control this with `navis.config.default_transform_backend` (`"auto"` by default - use fastcore if available, else the binaries) or per transform via `backend=`.
- elastix transforms are now *invertible* when navis-fastcore is installed - something `transformix` cannot do at all. Note the bridging graph does not use this by default: set `navis.config.elastix_invertible = True` to let it (see the notes on [`ElastixTransform`][navis.transforms.ElastixTransform] for why it is off).
- `find_bridging_path`, `find_all_bridging_paths` and `shortest_bridging_seq` gained a `prefer_forward` argument (default `True`): where two templates are connected by both a purpose-built registration and the inverse of its counterpart, use the purpose-built one - regardless of weight. Set it to `False` to have your graph's weights taken entirely at face value.
- `MovingLeastSquaresTransform` gained a `.matrix_affine` property, analogous to `TPStransform.matrix_affine`: since moving least squares is a *locally* weighted affine transform, this returns the global (least-squares) affine that it converges to far away from the landmarks
- **[`VoxelNeurons`][navis.VoxelNeuron] gained a proper toolkit**, all of it working straight off the sparse voxels:
    - morphology and set algebra: `dilate`, `erode`, `opening`, `closing`, `thin`, `fill_cavities`, `union`, `intersection`, `difference`, `symmetric_difference`. Per-voxel values are carried through; set operations align neurons onto a common lattice and refuse to combine ones that do not line up.
    - measurements: `surface_area`, `centroid`, `distance_transform`, `connected_components`, `iou`, `dice`, `grid_nbytes`/`voxels_nbytes`.
    - shorthands `.mesh()` and `.skeletonize()`.
- **[`navis.skeletonize`][navis.skeletonize] now accepts [`VoxelNeurons`][navis.VoxelNeuron]** (via the new [`navis.conversion.voxels2skeleton`][navis.conversion.voxels2skeleton]), closing a gap its own docstring used to flag. Defaults to `method="wavefront"` - ~4x faster than `"teasar"` and radii come free from the ring contraction rather than being snapped to the voxel lattice; `"teasar"` and `"thin"` remain available.
- **existing functions stopped densifying.** [`navis.drop_fluff`][navis.drop_fluff] and `graph_utils._connected_components` now handle `VoxelNeurons`; [`navis.smooth_voxels`][navis.smooth_voxels], [`navis.thin_voxels`][navis.thin_voxels] and [`navis.downsample_neuron`][navis.downsample_neuron] no longer allocate the grid (the latter could trip the new memory cap on exactly the sparse neurons worth downsampling). Voxel adjacency - behind `neuron2nx`/`neuron2igraph` - is ~100x faster and no longer needs the *undeclared* scikit-learn dependency. `smooth_voxels`/`thin_voxels` keep a `backend` argument if you want the old scipy/scikit-image route.

##### Fixes
- **a batch of [`VoxelNeuron`][navis.VoxelNeuron] bugs, most of them on the sparse (voxels + values) backing**, which until now was barely exercised - values and coordinates were free to drift apart. `threshold()` filtered the coordinates but not the values, leaving the two different lengths; `normalize()` scaled the *coordinates* instead of the values, corrupting the geometry outright; the documented `(N, 4)` constructor input silently discarded its value column; and changing `.values` did not invalidate a cached `.grid`. Also fixed: `convert_units()` resized the neuron instead of re-labelling it (125x too small for 8 nm → µm), `.volume` squared the z voxel size and dropped y, `.density` crashed on numpy 2, `copy.deepcopy()` raised a `TypeError`, `flip()` moved the neuron and mirrored connectors in the wrong space, and `.bbox` disagreed between the two backings by one voxel.
- [`navis.mesh`][navis.mesh] raised `AttributeError` on the `(N, 3)` voxel arrays it documents (it tested `.ndims`, which numpy spells `.ndim`).
- **[`navis.make_dotprops`][navis.make_dotprops] silently produced wrong tangent vectors for point clouds containing duplicate coordinates.** Points whose `k` nearest neighbours are *all* at distance zero are dropped (they have no defined tangent), but the neighbour indices were then offset by a flat `n_dropped` - which is only correct if every duplicate happens to sit at the *start* of the array. Anywhere else the indices ran past the end or went negative, and because numpy reads negative indices from the back this raised nothing: it just computed each tangent from an unrelated neighbourhood. On a 40-point cloud with a 4-point duplicate block in the middle, **39 of the 40 surviving points came back with the wrong tangent** (worst case nearly perpendicular to the correct one) and alpha off by up to 0.55. The neighbour indices are no longer remapped at all.
- **[`navis.Dotprops.recalculate_tangents`][navis.Dotprops.recalculate_tangents] returned `NaN` alpha values** for points sitting on duplicate coordinates - it has no equivalent of `make_dotprops`' duplicate check and cannot drop points (it must keep a 1:1 correspondence with `.points`). Those `NaN`s then propagated into every NBLAST score the neuron took part in. Such points now get `alpha=0` and an arbitrary unit vector, matching navis-fastcore.
- `TransformSequence` was registered as invertible even if it contained a transform that was not

## Version `1.12.0` { data-toc-label="1.12.0" }
_Date: 13/06/26_

##### Breaking
- [`mirror_brain`][navis.mirror_brain] now defaults to `mirror_axis="auto"`, i.e. takes the mirror axis from the template brain's meta data (falling back to `x`). This can change results for templates whose mirror axis is not `x`
- `TPStransform.matrix_rigid` (added in 1.11.0) was renamed to `.matrix_affine`
- {{ navis }}' internal graph algorithms no longer fall back to `networkx` - consequently `navis.config.use_igraph` is gone. `TreeNeuron.graph`, [`neuron2nx`][navis.neuron2nx] & co. are unaffected
- `requests-futures` is no longer a dependency: it was only used for the parallel URL reader, which now uses a plain `ThreadPoolExecutor`
- note that a number of bug fixes below **change outputs**: see the entries for flow centrality on fragmented neurons, [`resample_skeleton`][navis.resample_skeleton], [`plot1d`][navis.plot1d], `classify_nodes` and `small_segments`

##### Additions
- new interface: `navis.interfaces.brain_image_library` provides access to the [Brain Image Library](https://www.brainimagelibrary.org) which hosts thousands of single neuron reconstructions (see new tutorial)
- new function: [`propagate_labels`][navis.propagate_labels] propagates sparse labels across a neuron (see new tutorial)
- new function: [`split_axon_dendrite_prop`][navis.split_axon_dendrite_prop] uses label propagation to split a neuron into axon and dendrite (see new tutorial)
- new function: [`sample_skeleton`][navis.sample_skeleton] draws a given number of points at equal geodesic spacing along a skeleton
- new function: [`collapse_nodes`][navis.collapse_nodes] collapses a group of nodes into a single node
- NBLAST gained pluggable backends: [`nblast`][navis.nblast], [`nblast_smart`][navis.nblast_smart], [`nblast_allbyall`][navis.nblast_allbyall] and [`synblast`][navis.synblast] have a new `backend` parameter accepting `"builtin"` (the default), `"fastcore"` (requires `navis-fastcore`) or `"auto"`. The default can be changed globally via `navis.config.default_nblast_backend` and third parties can register their own backend with `navis.nbl.backends.register_backend`
- [`geodesic_matrix`][navis.geodesic_matrix] has a new `to_` parameter which restricts the *columns* of the matrix, mirroring the existing `from_`. Previously, the only way to get a `from_` x `to_` block was to compute every column and subset afterwards (for the leaf-by-leaf matrix of a 45k node skeleton: 794ms/2.2GB -> 126ms/307MB)
- [`dist_between`][navis.dist_between] now accepts matched arrays of nodes and returns their pairwise distances instead of raising `"Can only process single nodes"`. With `navis-fastcore` this is ~750x faster than the loop it replaces (1000 pairs on a 45k node skeleton: 4.7s -> 6ms); a single pair still returns a single float
- [`stitch_skeletons`][navis.stitch_skeletons] now exposes `min_size` and `use_radius`, which the underlying stitcher already supported but which the signature dropped
- [`NeuronList`][navis.NeuronList] now supports in-place scaling (`nl *= 1000`, `nl /= 1000`) which - unlike `nl * 1000` - does not copy every neuron
- `face_dist_sorting` gained a `heal_method` parameter

##### Improvements
- {{ navis }} now requires `navis-fastcore` >= 0.6.1 (still an optional dependency)
- `classify_nodes` is 6-20x faster and uses up to 40x less memory (10.3ms/6.2MB -> 0.5ms/0.1MB for a 71k node skeleton) which matters because it runs on every neuron mutation: it now uses `navis-fastcore` if available and builds the `type` column from integer categorical codes instead of an array of strings (the latter also speeds up the non-fastcore path by ~5x)
- the subtree height (the geodesic distance from a node down to the farthest leaf below it) is now computed with `navis-fastcore` if available: 14-31x faster and ~10x less memory. This backs [`prune_twigs`][navis.prune_twigs] with `exact=True` (2.2x faster, 4x less memory) and `node_label_sorting`
- `node_label_sorting` no longer builds a directed geodesic matrix (**4.6 GB** for a 71k node skeleton - the single largest allocation in {{ navis }}): 4.6x faster and 31x less memory. The resulting order is unchanged. This also speeds up [`plot1d`][navis.plot1d] and [`skeleton_adjacency_matrix`][navis.graph.skeleton_adjacency_matrix] with `sort=True`
- [`ivscc_features`][navis.ivscc_features] no longer builds a leafs-by-all-nodes distance matrix (**8.5 GB** for a 71k node skeleton!) to compute a single number: `max_path_length` is now 185x faster and uses 350x less memory
- [`geodesic_matrix`][navis.geodesic_matrix] now uses `navis-fastcore` for `MeshNeurons` too: ~19-68x faster and ~30-60x less memory
- [`longest_neurite`][navis.longest_neurite] with `from_root=False` no longer builds a leafs-by-leafs distance matrix just to take its maximum: 29x faster and 125x less memory (722ms/785MB -> 25ms/6.3MB for a 71k node skeleton)
- [`distal_to`][navis.distal_to] now uses `navis-fastcore` if available: 13x faster and 5x less memory (it previously asked igraph for a source-by-target block, which igraph answers by running an all-sources search)
- [`arbor_segregation_index`][navis.arbor_segregation_index], [`bending_flow`][navis.bending_flow], [`synapse_flow_centrality`][navis.synapse_flow_centrality], [`flow_centrality`][navis.flow_centrality], [`longest_neurite`][navis.longest_neurite] and `node_label_sorting` now request only the geodesic distances they actually use (see `to_` above)
- major speed-up for [`heal_skeleton`][navis.heal_skeleton] and [`stitch_skeletons`][navis.stitch_skeletons]: they now use `navis-fastcore` if available (2.5-340x faster on real, fragmented skeletons, e.g. 85s -> 0.25s for a 640k node skeleton), and the built-in fallback was rewritten around a vectorized Borůvka algorithm (5-15x faster and ~7x less memory, e.g. 8s/640MB -> 0.5s/90MB for a 220k node skeleton with 5k fragments). Results are unchanged: both produce the same minimum spanning tree
- major speed-up for [`resample_skeleton`][navis.resample_skeleton]: ~15-20x faster with the default `method="linear"` (e.g. 100ms -> 7ms for the example neuron; 425ms -> 64ms when densifying it to 132k nodes, as [`xform_brain`][navis.xform_brain] does) by interpolating all segments and columns in one go instead of fitting one `scipy.interpolate.interp1d` per column *per segment*. Non-linear methods (e.g. `"cubic"`) can't share that trick but still gain ~4x. It also no longer builds a KDTree and an indexed copy of the node table when the neuron has no soma, connectors or tags to re-map
- [`reroot_skeleton`][navis.reroot_skeleton] builds a node ID -> vertex index map once instead of scanning all vertices for each root: much faster on heavily fragmented neurons
- [`split_axon_dendrite`][navis.split_axon_dendrite] no longer runs out of memory on very large (100k+ nodes) neurons - the assignment of orphan nodes used to build a full orphans-by-all-nodes geodesic matrix - and is faster (igraph instead of `networkx`)
- [`drop_fluff`][navis.drop_fluff], [`fix_mesh`][navis.fix_mesh] and everything else built on connected components are faster: `navis-fastcore` is now used for `MeshNeurons` too, igraph otherwise
- [`skeletonize`][navis.skeletonize] with `shave=True`: fixing up the vertex map was an O(n_bristles x n_vertices) Python loop and is now a single vectorized map - a major bottleneck on large meshes
- [`rewire_skeleton`][navis.rewire_skeleton] now skips the minimum spanning tree if the graph is already a forest (i.e. has no cycles)
- [`H5transform`][navis.transforms.H5transform] and [`GridTransform`][navis.transforms.GridTransform] use `scipy.ndimage.map_coordinates` (~2x faster), and copies of an `H5transform` now carry over the cache - previously [`xform_brain`][navis.xform_brain] copied the transform and hence never benefitted from caching
- [`skeletonize`][navis.skeletonize] for point clouds/[`Dotprops`][navis.Dotprops] uses `scipy`'s minimum spanning tree instead of `networkx` and now correctly handles duplicate points
- `betweeness_centrality`, [`plot_flat`][navis.plot_flat] and [`segment_analysis`][navis.segment_analysis] are faster
- reading from URLs with the default `parallel="auto"` now goes parallel from 5 files onwards instead of 200. The 200 was tuned for the process pool used to read local files; URLs are read in a *thread* pool and are network- rather than CPU-bound. Reading e.g. 100 neurons off a remote server no longer means 100 sequential blocking requests
- URL reads now share a single `requests.Session`, so connections to the same host are pooled and kept alive
- `read_*` functions can now read from Google buckets (`gs://...`) without `gcsfs` installed

##### Fixes
- neurons read from a list of URLs **in parallel** came back stripped of their identity: the parallel reader handed the downloaded *bytes* (instead of the URL) to the parser, so the filename was never parsed. Affected neurons had no `file`, an `origin` of `"string"`, a `name` of `"SWC"`/`"MESH"`/... and a random `id`, and any `fmt` was silently ignored - i.e. the same input produced different neurons depending on `parallel`. [`read_mesh`][navis.read_mesh] failed outright (`ReadError`) since it needs the filename to determine the file type
- [`read_swc`][navis.read_swc] & co. no longer choke on URLs with a query string (e.g. `.../neuron.swc?token=123`, whose file extension was previously parsed as `swc?token=123`) and now decode percent-encoded filenames (`%20` -> a space)
- [`flow_centrality`][navis.flow_centrality], [`synapse_flow_centrality`][navis.synapse_flow_centrality] and [`arbor_segregation_index`][navis.arbor_segregation_index] returned wrong values for **fragmented neurons**: all three work out how many leafs/synapses are *proximal* to a node as `total - distal`, which is only valid on a single-rooted neuron. Nodes in another fragment are neither distal nor proximal but were silently counted as proximal, inflating the flow. Totals are now counted within each node's own fragment. **This changes the output for fragmented neurons** (they were previously wrong); single-rooted neurons are unaffected. Note that [`synapse_flow_centrality`][navis.synapse_flow_centrality] was only affected without `navis-fastcore`, so the two backends used to disagree; `bending_flow` was never affected
- `TreeNeuron.small_segments` returned the segments in a different **order** depending on whether `navis-fastcore` was installed (without it, {{ navis }} walked a Python `set`, i.e. in arbitrary hash order). They are now always ordered by the node table position of their seed node, which is what `navis-fastcore` already did. Several functions `enumerate()` the segments, so the order was ending up in their output - most visibly the **row order of [`segment_analysis`][navis.segment_analysis]**
- `classify_nodes`: a node whose parent does not exist is now classified as `root` instead of `end`. Such a neuron is already broken (it raises in `neuron2igraph`)
- [`despike_skeleton`][navis.despike_skeleton]: nodes whose flanking nodes coincided were assigned a spike ratio read from uninitialised memory and were hence flagged as spikes (and removed) at random
- [`plot1d`][navis.plot1d]: the bars were drawn using the length of each segment's *first edge* rather than the length of the whole segment. The x-axis was therefore far too short and distorted per-segment; for the example neuron it spanned 74,934 instead of 266,477 units of cable. **This changes the rendered plot** (it was previously wrong)
- [`resample_skeleton`][navis.resample_skeleton]: the resampled neuron was consistently coarser than requested. A segment of length `L` sampled at `N` nodes spans `N - 1` intervals, but the node count was `round(L / resample_to)` instead of `round(L / resample_to) + 1`. **This changes the output**: neurons now come back with slightly more nodes and a sampling resolution much closer to `resample_to` (example neuron at `resample_to=125`: 2039 -> 2284 nodes, achieved resolution 140 -> 112)
- [`resample_skeleton`][navis.resample_skeleton]: `skip_errors=True` never actually skipped anything (the failing segment raised a `KeyError` instead); failing segments now keep their original nodes, as intended
- [`resample_skeleton`][navis.resample_skeleton]: fixed a typo (`pd.to_nunmeric`) that raised an `AttributeError` whenever the new node IDs overflowed the original `int32` node ID column, and a `NameError` in the "N segments skipped" warning for neurons without segments
- [`heal_skeleton`][navis.heal_skeleton]: the `use_radius` parameter was accepted but silently ignored; it is now honoured
- [`heal_skeleton`][navis.heal_skeleton]: with `use_radius`, isolated nodes (which belong to no segment) were given an unrelated node's radius and were not scaled by `use_radius`; they now correctly fall back to their own, scaled radius
- [`heal_skeleton`][navis.heal_skeleton]: fragments that remain disconnected (because of `max_dist`/`min_size`/`mask`) now keep their original root instead of being re-rooted arbitrarily
- [`stitch_skeletons`][navis.stitch_skeletons]: passing `method=<list of node IDs>` - documented as option (4) - raised a bare `AssertionError`. It now works and restricts the new edges to those nodes. Note that node IDs are remapped when the fragments have duplicate IDs, in which case a list of IDs is ambiguous
- [`dist_between`][navis.dist_between]: unreachable node pairs are now correctly reported as `inf` (`navis-fastcore` 0.5.1 returned a bogus `1.0`; fixed in 0.6.0, which {{ navis }} now requires)
- [`drop_fluff`][navis.drop_fluff] works on [`Dotprops`][navis.Dotprops] again
- `models.BayesianTraversalModel`: corrected the traversal-probability propagation so results now match the Monte-Carlo `TraversalModel` for reconvergent graphs (e.g. diamonds); previously an independence assumption across time caused nodes to appear traversed too early (#194)
- neuron math operators (`+`, `-`, `*`, `/`) no longer break on neurons with integer node/connector coordinates under modern `pandas`
- [`simplify_mesh`][navis.simplify_mesh]: passing a float ratio as `F` failed with the `pyfqmr` and `open3d` backends because the computed face count was not an integer
- [`plot2d`][navis.plot2d]: fixed an `AttributeError` with `matplotlib` 3.11 (which removed `Poly3DCollection._vec`)
- [`close3d`][navis.close3d]/[`pop3d`][navis.pop3d] no longer break when there is no active viewer, and `close3d` now actually releases `config.primary_viewer`
- transforms: [`GridTransform`][navis.transforms.GridTransform] and [`CMTKtransform`][navis.transforms.CMTKtransform] had broken `copy()` methods - for `GridTransform` this dropped `spacing`/`offset` and hence produced wrong coordinates for any copied (i.e. any [`xform_brain`][navis.xform_brain]'d) transform. Also fixed `GridTransform.from_warpfield` when the input is an actual warpmap
- `neuprint` interface: fixed synapses not being assigned to the correct neuron when `fetch_mesh_neuron` was called for multiple neurons with `with_synapses=True`
- `neuprint` interface: no longer errors on datasets whose meta data lacks fields such as `instance`, `size`, `status` or `somaLocation`, and now sets the correct units for datasets with **anisotropic** voxels (the x voxel size was previously assumed for all three axes)
- `neuromorpho` interface: works again - it now uses `https`, sends a user agent (the server was rejecting requests without one) and gained a `NAVIS_NEUROMORPHO_VERIFY` environment variable to disable SSL verification if the certificate is broken
- [`read_precomputed`][navis.read_precomputed]: fixed reading from Google buckets
- silenced `pandas` deprecation warnings in [`read_swc`][navis.read_swc] and [`NeuronConnector`][navis.NeuronConnector]

##### Notes
- [`heal_skeleton`][navis.heal_skeleton] with `use_radius` can give marginally different (but equally valid) results depending on whether `navis-fastcore` is installed: each node is weighted by the mean radius of *its* segment, but branch points belong to several segments and end up with whichever one is assigned last - and the two backends enumerate segments in a different order. Both still produce a true minimum spanning tree. This is not new but is easier to run into now that fastcore is used by default

## Version `1.11.0` { data-toc-label="1.11.0" }
_Date: 27/02/26_

#### Breaking
- dropped support for Python 3.9

##### Improvements
- [`split_axon_dendrite`][navis.split_axon_dendrite] now allows setting the in-/output ratio for the split (see `split` parameter)
- major speed-up for [`heal_skeleton`][navis.heal_skeleton]
- minor speed-up for [`resample_skeleton`][navis.resample_skeleton]
- add `progress` parameter to [`mirror_brain`][navis.mirror_brain], [`symmetrize_brain`][navis.symmetrize_brain] and [`combine_meshes`][navis.meshes.operations.combine_meshes]
- [`persistence_vectors`][navis.persistence_vectors] now accepts list of distances to be sampled as `samples`
- [`make_dotprops`][navis.make_dotprops] has a new `on_issue` parameter that determines what to do when issues with the inputs are encountered (e.g. NaNs)
- two new [`VoxelNeuron`][navis.VoxelNeuron] methods:
    - [`flip()`][navis.VoxelNeuron.flip] flips the neuron along specified axes
    - [`normalize()`][navis.VoxelNeuron.normalize] scales values to a 0-1 range
- `neuprint` interface:
    - [`fetch_skeletons`][navis.interfaces.neuprint.fetch_skeletons] and [`fetch_mesh_neuron`][navis.interfaces.neuprint.fetch_mesh_neuron] will now also look for `tosomaLocation` to set the root/soma if there is no `somaLocation`
    - avoid fetching unused ROI info in [`fetch_skeletons`][navis.interfaces.neuprint.fetch_skeletons] and [`fetch_mesh_neuron`][navis.interfaces.neuprint.fetch_mesh_neuron] (minor speed-up)
- [`pointlabels_to_meshes`][navis.meshes.mesh_utils.pointlabels_to_meshes] can now also output voxels instead of meshes
- transforms:
    - new transform type: [`GridTransform`][navis.transforms.GridTransform] is a class for generic deformation-field transforms
    - [`TPStransform`][navis.transforms.TPStransform] now has a `.matrix_rigid` property that extracts the rigid part of the TPS affine as a 4x4 matrix
    - [`TPStransforms`][navis.transforms.TPStransform] and `MovingLeastSquaresTransforms` transforms now transform in batches to avoid memory issues when transforming large numbers of points
    - new methods for [`CMTKtransform`][navis.transforms.CMTKtransform]: [`to_dfield`][navis.transforms.CMTKtransform.to_dfield] and [`to_grid_transform`][navis.transforms.CMTKtransform.to_grid_transform] can be used to sample the CMTK transform into a deformation field (this is experimental)
    - new [`H5transform`][navis.transforms.H5transform] method: [`xform_image`][navis.transforms.H5transform.xform_image] can be used to apply the transform to images (this is experimental)
    - [`TransformRegistry.register_transform`][navis.transforms.templates.TemplateRegistry] now accepts an optional `weight_inv` parameter; can be used to penalize expensive inverse transforms (e.g. CMTK)
- input/output:
    - `read_xxx` functions will now use threads instead of processes for parallelization when reading from URLs (much faster)
    - [`read_precomputed`][navis.read_precomputed] will now also look for `.ngmesh` files when given a folder to search
    - `read_xxx` function can now load data straight from Google buckets (`gs://...`, requires the optional `gcsfs`)

##### Fixes
- using `connectors="pre/postsynapses"` now actually works in `plot2d` and `plot3d`
- fixed an issue in [`resample_skeleton`][navis.resample_skeleton] where adding new nodes could cause an overflow error for node IDs
- subsetting neuron meshes with connectors will now correctly carry over vertex IDs
- plotting skeleton where the soma has no radius will not break anymore
- `write_xxx` functions do not break anymore when a neuron has id `0`
- plotting of connectors:
   - parameters `cn_alpha`, `cn_colors` and `cn_mesh_colors` now work across all 3d backends
   - `plot2d` now respects `cn_alpha`
- fixed an issue where checking for available mesh backends (pyfqmr, pymeshlab, etc) could cause a crash
- Blender interface: fixed an issue adding skeletons caused by pandas >= 3.0

**Full Changelog**: [v1.10.0...v1.11.0](https://github.com/navis-org/navis/compare/v1.10.0...v1.11.0)

## Version `1.10.0` { data-toc-label="1.10.0" }
_Date: 06/02/25_

##### Improvements
- made reading neurons from `.tar` archives much faster
- [`read_swc`][navis.read_swc] now works if additional columns are present

##### Fixes
- `opacity` parameter [`plot3d`][navis.plot3d] now works correctly when using the plotly backend
- fixed an issue with Elastix transforms on Windows machines
- fixes for [`navis.longest_neurite`][] when `from_root=False`
- fixed issues with neuPrint interface when using multiple clients
- fixed an issue with the MICrONS interface
- fixed an issue with mesh simplification and the pymeshlab and Blender backends (@floesche)
- fixed two (potential) issues in [navis.longest_neurite][] when `from_root=False`
- fixed various issues related to numpy 2.0 (@floesche)

**Full Changelog**: [v1.9.1...v1.10.0](https://github.com/navis-org/navis/compare/v1.9.1...v1.10.0)

## Version `1.9.1` { data-toc-label="1.9.1" }
_Date: 24/10/24_

##### Improvements
- MICrONS & H01 interfaces:
    - `fetch_neurons` now accepts a `materialization` parameter that determines which materialization version is used for synapse and nucleus queries; defaults to "auto" which means `navis` will try to find a materialization version matching the queried root IDs
    - `fetch_neurons` will now also assign soma positions for H01 neurons (see the `.soma_pos` neuron property)
- `CloudVolume.mesh.get_navis` (see [`navis.patch_cloudvolume`][]) now accepts a `process` (default is `False`) parameter that determines whether the NeuroGlancer mesh is processed (deduplication of vertices, etc.); contribute by Forrest Collman

##### Fixes
- fixed a bug in [`navis.subset_neuron`][] that caused branch points to not be re-calculated

**Full Changelog**: [v1.9.0...v1.9.1](https://github.com/navis-org/navis/compare/v1.9.0...v1.9.1)

## Version `1.9.0` { data-toc-label="1.9.0" }
_Date: 17/10/24_

This version brings a shiny new interface to the [H01 human cortex](https://h01-release.storage.googleapis.com/landing.html) dataset
and various other quality of life improvements.

##### Breaking
- The default for `radius` ([`navis.plot2d`][] and [`navis.plot3d`][]) was changed to `False` (from `"auto"`); this is to make sure that the defaults allow visualizing large numbers of skeletons, i.e. prioritizing performance over beauty

##### Additions
- New interface to the [H01](https://h01-release.storage.googleapis.com/landing.html) dataset (by Jinhan Choi and Jakob Troidl; see the new tutorial for details)

##### Improvements
- I/O:
    - [`read_nrrd`][navis.read_nrrd], [`read_tiff`][navis.read_tiff] and [`read_mesh`][navis.read_mesh] now use the same backend as e.g. [`read_swc`][navis.read_swc] which enables some niceties such as reading directly from URLs and archives, parallel processing, etc
    - all `read_*` functions now have an `error` parameter that can be used to skip errors
- Image data:
    - new function: [`navis.thin_voxels`][] can be used to thin images and `VoxelNeurons` to single-pixel width (see also below)
    - new `thin` parameter for [`read_nrrd`][navis.read_nrrd], [`read_tiff`][navis.read_tiff]
- [`TreeNeurons`][navis.TreeNeuron]:
    - skeletons can now be initialized from a `(vertices, edges)` tuple - see also [`navis.edges2neuron`][]
    - new property: `TreeNeuron.vertices` gives read-only to node (vertex) coordinates
- [`VoxelNeurons`][navis.VoxelNeuron]:
    - new properties: `VoxelNeuron.nnz` and `VoxelNeuron.density`
- [`navis.drop_fluff`][] and [`navis.neuron2nx`][] now also works with [`Dotprops`][navis.Dotprops]

##### Experimental
- setting `navis.config.add_units=True` (default is `False` for the time being) will add units to certain neuron properties such as `.cable_length` to make them human-readable

**Full Changelog**: [v1.8.0...v1.9.0](https://github.com/navis-org/navis/compare/v1.8.0...v1.9.0)

## Version `1.8.0` { data-toc-label="1.8.0" }
_Date: 22/09/24_

This version contains a major internal rework of both [`navis.plot2d`][] and [`navis.plot3d`][] to make them
more consistent and easier to use.

##### Breaking
- Plotting: the `synapse_layout` parameter was renamed to `cn_layout` (matching e.g. other parameters such as `cn_colors`)
- Negative views in [`navis.plot2d`][] (e.g. `view=("x", "-z")`) will now invert axes rather than changing the underlying data
- Minimum version of `matplotlib` is now `3.9` (was `3.6`)
- The `plotly` backend is not part of a minimal install anymore (still installed using `navis[all]`)
- The Vispy backend is now deprecated and will be removed in a future release
- Removed `navis.screenshot` - please use the Octarine/Vispy viewer's `.screenshot()` method instead
- [`navis.tortuosity`][] now calculates tortuosity as-is (i.e. without resampling) by default

##### Additions
- Added [Octarine](https://github.com/schlegelp/octarine) as the default backend for plotting from terminal
- New Function: [`navis.ivscc_features`][] computes some basic IVSCC features
- New function: [`navis.graph.skeleton_adjacency_matrix`][] computes the node adjacency for skeletons
- New function: [`navis.graph.simplify_graph`][] simplifies skeleton graphs to only root, branch and leaf nodes while preserving branch length (i.e. weights)
- New [`NeuronList`][navis.NeuronList] method: [`get_neuron_attributes`][navis.NeuronList.get_neuron_attributes] is analagous to `dict.get`
- [`NeuronLists`][navis.NeuronList] now implement the `|` (`__or__`) operator which can be used to get the union of two [`NeuronLists`][navis.NeuronList]
- [`navis.Volume`][] now have an (optional) `.units` property similar to neurons
- `Tree/MeshNeurons` and `Dotprops` now support addition/subtraction (similar to the already existing multiplication and division) to allow offsetting neurons

##### Improvements
- Plotting:
    - [`navis.plot3d`][]:
      - `legendgroup` parameter (plotly backend) now also sets the legend group's title
      - new parameters for the plotly backend:
          - `legend` (default `True`): determines whether legends is shown
          - `legend_orientation` (default `v`): determines whether legend is aranged vertically (`v`) or horizontally (`h`)
          - `linestyle` (default `-`): determines line style for skeletons
      - default for `radius` is now `"auto"`
    - [`navis.plot2d`][]:
      - the `view` parameter now also works with `methods` `3d` and `3d_complex`
      - the `color_by` and `shade_by` parameters now also work when plotting skeletons with `radius=True`
      - new defaults: `radius="auto"`, `alpha=1`, `figsize=None` (use matplotlib defaults)
      - new parameters for methods `3d` and `3d_complex`: `mesh_shade=False` and `non_view_axes3d`
      - the `scalebar` and `soma` parameters can now also be dictionaries to style (color, width, etc) the scalebar/soma
    - the `connectors` parameter can now be used to show specific connector types (e.g. `connectors="pre"`)
- I/O:
    - `read_*` functions are now able to read from FTP servers (`ftp://...`)
    - the `limit` parameter used in many `read_*` functions can now also be a regex pattern or a `slice`
- New parameter in [`navis.resample_skeleton`][]: use `map_column` to include arbitrary columns in the resampling
- [`navis.prune_twigs`][] and [`navis.morpho.cable_length`][] now accept a `mask` parameter
- General improvements to docs and tutorials

##### Fixes
- Memory usage of `Neuron/Lists` is now correctly re-calculated when the neuron is modified
- Various fixes and improvements for the MICrONS interface (`navis.interfaces.microns`)
- [`navis.graph.node_label_sorting`][] now correctly prioritizes total branch length
- [`navis.TreeNeuron.simple`][] now correctly drops soma nodes if they aren't root, branch or leaf points themselves

**Full Changelog**: [v1.7.0...v1.8.0](https://github.com/navis-org/navis/compare/v1.7.0...v1.8.0)

## Version `1.7.0` { data-toc-label="1.7.0" }
_Date: 25/07/24_

##### Breaking
- Plotting: dropped the `cluster` parameter in favor of an improved `color_by` logic (see below)

##### Additions
- {{ navis }} now uses `navis-fastcore` if present to dramatically speed up core functions (see updated install instructions)
- New method `navis.NeuronList.add_metadata` to quickly add metadata to neurons

##### Improvements
- `navis.find_soma` and `navis.graph.neuron2nx` (used under the hood) are now much faster
- All I/O functions such as `navis.read_swc` now show which file caused an error (if any); original filenames are tracked as `file` property
- `navis.NeuronList` will only search the first 100 neurons for autocompletion to avoid freezing with large lists
- Plotting functions: `color_by` now accepts either a list of labels (one per neuron) or the name of a neuron property
- `navis.subset_neuron` is now faster and more memory efficient when subsetting meshes
- `navis.TreeNeuron.cable_length` is now faster
- Fixed a bug in plotting when using vertex colors
- Fixed the progress bar in `navis.interfaces.neuprint.fetch_mesh_neuron`
- Fixed a bug in `navis.synblast` that caused multiprocessing to fail (pickling issue with `pykdtree`)
- `navis.interfaces.neuprint.fetch_mesh_neuron` will now ignore the `lod` parameter if the data source does not support it instead of breaking
- Fixed a number of deprecation warnings in the codebase

**Full Changelog**: [v1.6.0...v1.7.0](https://github.com/navis-org/navis/compare/v1.6.0...v1.7.0)

## Version `1.6.0` { data-toc-label="1.6.0" }
_Date: 07/04/24_

##### Breaking
- Dropped support for Python 3.8, per NEP 29
- `navis.write_swc` no longer supports writing Dotprops to SWC files

##### Additions
- New property `TreeNeuron.surface_area`
- New (experimental) functions `navis.read_parquet` and `navis.write_parquet` store skeletons and dotprops in parquet files (see [here](https://github.com/clbarnes/neurarrow) for format specs)
- New `navis.read_nml` function to read single NML file
- New `navis.NeuronConnector` class for creating connectivity graphs from groups neurons with consistent connector IDs
- New method for CMTKtransforms: `navis.transforms.CMTKTransform.xform_image`

##### Improvements
- Improved performance for adding recordings to `CompartmentModel`
- `navis.heal_skeleton` and `navis.resample_skeleton` are now faster
- Improved logic for splitting NBLASTs across multiple cores
- `navis.xform_brain` now allows specifying multiple intermediate template spaces through the `via` parameter and to ignore spaces through the `avoid` parameter
- I/O functions can now read directly from `.tar` or `.tar.gz` files
- `navis.read_precomputed` now accepts a `limit` parameter similar to `navis.read_swc`

##### Fixes
- Fixed interface to InsectBrainDB
- `navis.read_precomputed`:
    - now correctly parses the `info` file depending on the source
    - reading large files (i.e. meshes) directly from a URL should not break anymore
- Fixed writing vertex properties in `navis.write_precomputed`
- Fixed a bug in `navis.resample_skeleton`
- Fixed an occasional issue when plotting skeletons with radii
- Fixed bug in `navis.subset_neuron` that caused connectors to be dropped when using mask
- Fixed a bug in `navis.despike_skeleton` that caused the `reverse` argument to be ignored
- Fixed two small bugs in `navis.interfaces.neuprint.fetch_mesh_neuron`

**Full Changelog**: [v1.5.0...v1.6.0](https://github.com/navis-org/navis/compare/v1.5.0...v1.6.0)

## Version `1.5.0` { data-toc-label="1.5.0" }
_Date: 27/07/23_

##### Breaking
- Dropped support for Python 3.7

##### Additions
- New function: `navis.pop3d` removes the most recently added object from the vispy 3d viewer
- New experimental functions for (pairwise) alignment of neurons using the `pycpd` package: `navis.nblast_align`, `navis.align.align_deform`, `navis.align.align_rigid`, `navis.align.align_pca`, `navis.align.align_pairwise`
- New `NeuronList` method: `navis.NeuronList.set_neuron_attributes`
- New utility functions: `navis.nbl.compress_scores`, `navis.nbl.nblast_prime`

##### Improvements
- `navis.xform_brain` now recognizes the target template's units if available
- Improved persistence functions: `navis.persistence_distances`, `navis.persistence_vector`, `navis.persistence_diagram`
- `navis.longest_neurite` and `navis.cell_body_fiber` now also allow removing the longest neurite and CBF, respectively
- `navis.heal_skeleton` now accepts a `mask` parameter that allows restricting where fragments are stitched

##### Fixes
- Various other bugfixes

**Full Changelog**: [v1.4.0...v1.5.0](https://github.com/navis-org/navis/compare/v1.4.0...v1.5.0)

## Version `1.4.0` { data-toc-label="1.4.0" }
_Date: 21/12/22_

##### Breaking
- `navis.flow_centrality` was renamed to `navis.synapse_flow_centrality` and a new non-synaptic `navis.flow_centrality` function was added. This also impacts the `method` parameter in `navis.split_axon_dendrite`!
- `vispy` is now a soft dependency

##### Additions
- New function: `navis.read_tiff` to read image stacks from TIFF files
- New utility function: `navis.nbl.extract_matches`

##### Improvements
- NBLASTs: single progress bar instead of one for each process
- New `via` parameter for `navis.xform_brain`
- `navis.write_swc` can now save Dotprops to SWC files
- `navis.make_dotprops` can now downsample point cloud inputs
- Various improvements to `navis.split_axon_dendrite`, `navis.nblast_allbyall`, `navis.interfaces.neuprint.fetch_mesh_neuron`, `navis.interfaces.neuprint.fetch_skeletons`

##### Fixes
- Tons of bug fixes

**Full Changelog**: [v1.3.1...v1.4.0](https://github.com/navis-org/navis/compare/v1.3.1...v1.4.0)

## Version `1.3.1` { data-toc-label="1.3.1" }
_Date: 10/06/22_

##### Fixes
- Various bugs fixed

**Full Changelog**: [v1.3.0...v1.3.1](https://github.com/navis-org/navis/compare/v1.3.0...v1.3.1)

## Version `1.3.0` { data-toc-label="1.3.0" }
_Date: 10/05/22_

##### Breaking
- As of this version `pip install navis` won't install a vispy backend

##### Additions
- New interface to fetch data from Virtual Fly Brain: `navis.interfaces.vfb`
- Tools to build custom NBLAST score matrices
- Bayesian implementation of the network traversal model: `navis.models.network_models.BayesianTraversalModel`
- New morphometrics functions: `navis.segment_analysis` & `navis.form_factor`
- New function to write meshes: `navis.write_mesh`

##### Improvements
- NBLASTs: new `approx_nn` parameter
- Example neurons now come with some meta data

##### Fixes
- Lots of fixes and improvements in particular for I/O-related functions

**Full Changelog**: [v1.2.1...v1.3.0](https://github.com/navis-org/navis/compare/v1.2.1...v1.3.0)

## Version `1.2.1` { data-toc-label="1.2.1" }
_Date: 25/02/22_

##### Fixes
- Hot fix for `navis.split_axon_dendrite`

**Full Changelog**: [v1.2.0...v1.2.1](https://github.com/navis-org/navis/compare/v1.2.0...v1.2.1)

## Version `1.2.0` { data-toc-label="1.2.0" }
_Date: 24/02/22_

##### Additions
- New function: `navis.betweeness_centrality`
- New function: `navis.combine_neurons` to simply concatenate neurons
- New set of persistence functions: `navis.persistence_vectors`, `navis.persistence_points` and `navis.persistence_distances`
- Added a new interface with the Allen Cell Types Atlas

##### Improvements
- Improvements to various functions: e.g. `navis.bending_flow`, `navis.synapse_flow_centrality`, `navis.split_axon_dendrite`, `navis.longest_neurite`
- `navis.write_nrrd` and `navis.read_nrrd` can now be used to write/read Dotprops to/from NRRD files
- `navis.read_swc` now accepts a `limit` parameter that enables reading on the first N neurons
- `navis.nblast` (and variants) now accept a `precision` parameter
- `navis.simplify_mesh` (and therefore `navis.downsample_neuron` with skeletons) now uses the `pyfqmr` if present
- Improved the interface to Neuromorpho

##### Fixes
- Myriads of small and big bugfixes

**Full Changelog**: [v1.1.0...v1.2.0](https://github.com/navis-org/navis/compare/v1.1.0...v1.2.0)

## Version `1.1.0` { data-toc-label="1.1.0" }
_Date: 18/11/21_

##### Additions
- New function: `navis.sholl_analysis`
- Plotly is now correctly chosen as default backend in Google colab

##### Fixes
- Fixed a critical bug with plotting skeletons with plotly `5.4.0`

**Full Changelog**: [v1.0.0...v1.1.0](https://github.com/navis-org/navis/compare/v1.0.0...v1.1.0)

## Version `1.0.0` { data-toc-label="1.0.0" }
_Date: 11/11/21_

##### Breaking
- `navis.MeshNeuron`: `__getattr__` does not search `trimesh` representation anymore
- NBLASTs: queries/targets now MUST be `navis.Dotprops` (no more automatic conversion, use `navis.make_dotprops`)
- Renamed functions to make it clear they work only on `TreeNeurons`:
- `smooth_neuron` :octicons-arrow-right-24: `navis.smooth_skeleton`
- `reroot_neuron` :octicons-arrow-right-24: `navis.reroot_skeleton`
- `rewire_neuron` :octicons-arrow-right-24: `navis.rewire_skeleton`
- `despike_neuron` :octicons-arrow-right-24: `navis.despike_skeleton`
- `average_neurons` :octicons-arrow-right-24: `navis.average_skeletons`
- `heal_fragmented_neuron` :octicons-arrow-right-24: `navis.heal_skeleton`
- `stitch_neurons` :octicons-arrow-right-24: `navis.stitch_skeletons`
- `cut_neuron` :octicons-arrow-right-24: `navis.cut_skeleton`
- Removals and other renamings:
    - `navis.clustering` module was removed and with it `navis.cluster_xyz` and `ClustResult` class
    - renamed `cluster_by_synapse_placement` :octicons-arrow-right-24: `navis.synapse_similarity`
    - renamed `cluster_by_connectivity` :octicons-arrow-right-24: `navis.connectivity_similarity`
    - renamed `sparseness` :octicons-arrow-right-24: `navis.connectivity_sparseness`
    - renamed `navis.write_google_binary` :octicons-arrow-right-24: `navis.write_precomputed`
- `navis.geodesic_matrix` renamed parameter `tn_ids` :octicons-arrow-right-24: `from_`

##### Additions & Improvements
- `navis.NeuronList.apply()` now allows omitting failures
- `navis.VoxelNeuron`:
    - new class representing neurons as voxels
    - new (experimental) class representing neurons as voxels
    - `navis.read_nrrd` now returns `VoxelNeuron` instead of `Dotprops` by default
    - currently works with only a selection of functions
- `navis.TreeNeuron`:
    - can now be initialized directly with `skeletor.Skeleton`
    - new method: `navis.TreeNeuron.snap`
- `navis.MeshNeuron`:
    - `navis.in_volume`, `navis.subset_neuron` and `navis.break_fragments` now work on `MeshNeurons`
    - new properties: `.skeleton`, `.graph` and `.igraph`
    - new methods: `navis.MeshNeuron.skeletonize` and `navis.MeshNeuron.snap`
    - can now be initialized with `skeletor.Skeleton` and `(vertices, faces)` tuple
    - plotting: `color_by` parameter now works with `MeshNeurons`
- `navis.Dotprops`:
    - new property: `.sampling_resolution` (used e.g. for scaling vectors for plotting)
    - new method: `navis.Dotprops.snap`
- Experimental support for non-isometric `.units` for neurons
- NBLASTs:
    - new parameter `limit_dist` allows speeding up NBLASTs with minor precision loss
    - new experimental parameter `batch_size` to NBLAST neurons in batches
    - overall faster initialization with large lists of neurons
- SWC I/O (`navis.read_swc` & `navis.write_swc`):
    - by default we will now deposit neuron meta data (name, id, units) in the SWC header (see `write_meta` parameter)
    - meta data in SWC header can also be read back (see `read_meta` parameter)
    - filenames can now be parsed into specific neuron properties (see `fmt` parameter)
    - node IDs now start with 0 instead of 1 when writing SWC files
- I/O to/from Google neuroglancer's precomputed format:
    - total rework of this module
    - renamed `navis.write_google_binary` :octicons-arrow-right-24: `navis.write_precomputed`
    - new function: `navis.read_precomputed`
- Plotting:
    - new function `navis.plot_flat` plots neurons as dendrograms
    - `navis.plot3d` with plotly backend now returns a plotly `Figure` instead of a figure dictionary
    - new [k3d](https://k3d-jupyter.org) backend for plotting in Jupyter environments: try `navis.plot3d(x, backend='k3d')`
    - new parameter for `navis.plot2d` and `navis.plot3d`: use `clusters=[0, 0, 0, 1, 1, ...]` to assigns clusters and have them automatically coloured accordingly
    - `navis.plot2d` now allows `radius=True` parameter
- Transforms:
    - support for elastix (`navis.transforms.ElastixTransform`)
    - whether transforms are invertible is now determined by existence of `__neg__` method
- Most functions that work with `TreeNeurons` now also work with `MeshNeurons`
- New high-level wrappers to convert neurons: `navis.voxelize`, `navis.mesh` and `navis.skeletonize`
- `navis.make_dotprops` now accepts `parallel=True` parameter for parallel processing
- `navis.smooth_skeleton` can now be used to smooth arbitrary numeric columns in the node table
- New function `navis.drop_fluff` removes small disconnected bits and pieces from neurons
- New function `navis.patch_cloudvolume` monkey-patches `cloudvolume` (see the new tutorial)
- New function `navis.write_nrrd` writes `VoxelNeurons` to NRRD files
- New functions to read/write `MeshNeurons`: `navis.read_mesh` and `navis.write_mesh`
- New function `navis.read_nmx` reads pyKNOSSOS files
- New function `navis.smooth_mesh` smoothes meshes and `MeshNeurons`
- Improved/updated the InsectBrain DB interface (see the tutorial)
- Under-the-hood fixes and improvements to: `navis.plot2d`, `navis.split_axon_dendrite`, `navis.tortuosity`, `navis.resample_skeleton`, `navis.mirror_brain`
- First pass at a `NEURON` interface (see the new tutorial)
- First pass at interface with the Allen's MICrONS datasets (see the new tutorial)
- `NAVIS_SKIP_LOG_SETUP` environment variable prevents default log setup for library use
- Improved `navis.cable_overlap`

##### Fixes
- Under-the-hood fixes and improvements

**Full Changelog**: [v0.6.0...v1.0.0](https://github.com/navis-org/navis/compare/v0.6.0...v1.0.0)

## Version `0.6.0` { data-toc-label="0.6.0" }
_Date: 12/05/21_

##### Additions
- new functions: `navis.prune_at_depth`, `navis.read_rda`, `navis.cell_body_fiber`
- new functions to map units into neuron space: `BaseNeuron.map_units` and `navis.to_neuron_space`

##### Improvements
- many spatial parameters (e.g. in `navis.resample_skeleton`) can now be passed as unit string, e.g. `"5 microns"`
- many functions now accept a `parallel=True` parameter to use multiple cores (depends on `pathos`)
- `navis.read_swc` and `navis.write_swc` can now read/write directly from/to zip files
- reworked `navis.read_json`, and `navis.write_json`
- `nblast` functions now let you use your own scoring function (thanks to Ben Pedigo!)
- added `threshold` parameter to `navis.read_nrrd`
- `navis.nblast_smart`: drop `quantile` and add `score` criterion
- functions that manipulate neurons will now always return something (even if `inplace=True`)
- `navis.cut_skeleton` now always returns a single `NeuronList`
- `navis.mirror_brain` now works with `k=0/None` Dotprops
- all `reroot_to_soma` parameters have been renamed to `reroot_soma`
- `navis.TreeNeuron` now has a `soma_pos` property that can also be used to set the soma by position
- made transforms more robust against points outside deformation fields
- better deal if node ID of soma is `0` (e.g. during plotting)
- `navis.neuron2tangents` now drops zero-length vectors

##### Fixes
- fixed `navis.guess_radius`
- fixed NBLAST progress bars in notebook environments
- fixed a couple bugs with `CMTK` transforms

**Full Changelog**: [v0.5.3...v0.6.0](https://github.com/navis-org/navis/compare/v0.5.3...v0.6.0)

## Version `0.5.3` { data-toc-label="0.5.3" }
_Date: 10/04/21_

##### Additions
- new functions: `navis.nblast_smart`, `navis.synblast`, `navis.symmetrize_brain`
- `navis.plot2d`: `rasterize=True` will rasterize neurons (but not axes or labels) to help keep file sizes low
- `navis.plot3d` (plotly): `hover_name=True` will show neuron names on hover

##### Improvements
- `navis.simplify_mesh` now supports 3 backends: Blender3D, `open3d` or `pymeshlab`
- `navis.make_dotprops` can now produce `Dotprops` purely from skeleton edges (set `k=None`)
- reworked `navis.write_swc` (faster, easier to work with)
- a new type of landmark-based transform: moving least square transforms (thanks to Chris Barnes)
- vispy `navis.Viewer`: press B to show a bounding box
- moved tests from Travis to Github Actions (this now also includes testing tutorial notebooks)

##### Fixes
- a great many small and big bug fixes

**Full Changelog**: [v0.5.2...v0.5.3](https://github.com/navis-org/navis/compare/v0.5.2...v0.5.3)

## Version `0.5.2` { data-toc-label="0.5.2" }
_Date: 02/02/21_

##### Additions
- new functions: `navis.xform`, `navis.write_precomputed`

##### Improvements
- `navis.downsample_neuron` now also works on `Dotprops`
- Neurons: connectors are now included in bounding box calculations
- NeuronLists: added progress bar for division / multiplication

**Full Changelog**: [v0.5.1...v0.5.2](https://github.com/navis-org/navis/compare/v0.5.1...v0.5.2)

## Version `0.5.1` { data-toc-label="0.5.1" }
_Date: 10/01/21_

##### Fixes
- Various under-the-hood improvements and bugfixes

**Full Changelog**: [v0.5.0...v0.5.1](https://github.com/navis-org/navis/compare/v0.5.0...v0.5.1)

## Version `0.5.0` { data-toc-label="0.5.0" }
_Date: 05/01/21_

##### Additions
- new functions for transforming spatial data (locations, neurons, etc) between brain spaces:
    - `navis.xform_brain` transforms data from one space to another
    - `navis.mirror_brain` mirrors data about given axis
    - see the new tutorials for explanations
- low-level interfaces to work with affine, H5-, CMTK- and thin plate spline transforms

##### Improvements
- de-cluttered top level namespace: some more obscure functions are now only available through modules

**Full Changelog**: [v0.4.3...v0.5.0](https://github.com/navis-org/navis/compare/v0.4.3...v0.5.0)

## Version `0.4.3` { data-toc-label="0.4.3" }
_Date: 22/12/20_

##### Fixes
- Small bugfixes

**Full Changelog**: [v0.4.2...v0.4.3](https://github.com/navis-org/navis/compare/v0.4.2...v0.4.3)

## Version `0.4.2` { data-toc-label="0.4.2" }
_Date: 22/12/20_

##### Fixes
- Small bugfixes

**Full Changelog**: [v0.4.1...v0.4.2](https://github.com/navis-org/navis/compare/v0.4.1...v0.4.2)

## Version `0.4.1` { data-toc-label="0.4.1" }
_Date: 06/12/20_

##### Fixes
- Critical bugfix in NBLAST

**Full Changelog**: [v0.4.0...v0.4.1](https://github.com/navis-org/navis/compare/v0.4.0...v0.4.1)

## Version `0.4.0` { data-toc-label="0.4.0" }
_Date: 06/12/20_

##### Additions
- native implementation of NBLAST: `navis.nblast` and `navis.nblast_allbyall`!
- new parameter `navis.plot3d` (plotly backend) with `hover_id=True` will show node IDs on hover
- `navis.Volume.resize` has now `inplace=False` as default

**Full Changelog**: [v0.3.4...v0.4.0](https://github.com/navis-org/navis/compare/v0.3.4...v0.4.0)

## Version `0.3.4` { data-toc-label="0.3.4" }
_Date: 24/11/20_

##### Improvements
- improved `navis.Dotprops`:
- more control over generation in `navis.make_dotprops`
- `navis.Dotprops` now play nicely with R interface

**Full Changelog**: [v0.3.3...v0.3.4](https://github.com/navis-org/navis/compare/v0.3.3...v0.3.4)

## Version `0.3.3` { data-toc-label="0.3.3" }
_Date: 23/11/20_

##### Additions
- new module: `models` for modelling networks and neurons
- new functions `navis.resample_along_axis`, `navis.insert_nodes`, `navis.remove_nodes`
- full rework of `navis.Dotprops`:
- make them a subclass of BaseNeuron
- implement `nat:dotprops` in `navis.make_dotprops`
- added `navis.read_nrrd` and `navis.write_nrrd`
- side-effect: renamed `navis.from_swc` :octicons-arrow-right-24: `read_swc` and `navis.to_swc` :octicons-arrow-right-24: `write_swc`
- improved conversion between nat and {{ navis }} `Dotprops`
- full rework of topology-related functions:
- `navis.strahler_index`, `navis.segregation_index`, `navis.bending_flow`, `navis.synapse_flow_centrality` and `navis.split_axon_dendrite` now work better, faster and more accurately. See their docs for details.
- new function: `navis.arbor_segregation_index`
- new `color_by` and `shade_by` parameters for `plot3d` and `plot2d` that lets you color/shade a
neuron by custom properties (e.g. by Strahler index or compartment)

##### Improvements
- neurons are now more memory efficient:
    - pandas "categoricals" are used for connector and node "type" and "label" columns
    - add a `.memory_usage` method analogous to that of `pandas.DataFrames`
- `navis.NeuronList` can now be pickled!
- made `navis.Viewer` faster
- `navis.prune_twigs` can now (optionally) prune by `exactly` the desired length
- improved `navis.NeuronList.apply`

##### Fixes
- small bugfixes and improvements

**Full Changelog**: [v0.3.2...v0.3.3](https://github.com/navis-org/navis/compare/v0.3.2...v0.3.3)

## Version `0.3.2` { data-toc-label="0.3.2" }
_Date: 18/10/20_

##### Improvements
- `navis.plot2d` and `navis.plot3d` now accept `trimesh.Trimesh` directly
- `navis.in_volume` now works with any mesh-like object, not just `navis.Volumes`

##### Fixes
- lots of small bugfixes and improvements

**Full Changelog**: [v0.3.1...v0.3.2](https://github.com/navis-org/navis/compare/v0.3.1...v0.3.2)

## Version `0.3.1` { data-toc-label="0.3.1" }
_Date: 07/10/20_

##### Additions
- new function `navis.rewire_skeleton`

##### Improvements
- `navis.heal_skeleton` and `navis.stitch_skeletons` are now much much faster
- `navis.reroot_skeleton` can now reroot to multiple roots in one go
- `navis.plot3d` now accepts a `soma` argument
- improved caching for neurons
- improved multiplication/division of neurons
- faster `r.nblast` and `r.nblast_allbyall`
- `r.xform_brain` now also adjusts the soma radius
- `neuprint.fetch_skeletons` now returns correct soma radius

##### Fixes
- lots of small bugfixes

**Full Changelog**: [v0.3.0...v0.3.1](https://github.com/navis-org/navis/compare/v0.3.0...v0.3.1)

## Version `0.3.0` { data-toc-label="0.3.0" }
_Date: 06/10/20_

##### Additions
- Started module to manipulate mesh data (see e.g. `navis.simplify_mesh`)

##### Improvements
- Improved interfaces with R NBLAST and `xform_brain`
- Improved attribute caching for neurons

**Full Changelog**: [v0.2.3...v0.3.0](https://github.com/navis-org/navis/compare/v0.2.3...v0.3.0)

## Version `0.2.3` { data-toc-label="0.2.3" }
_Date: 06/09/20_

##### Additions
- New Neuron property `.label` that if present will be used for plot legends
- New function for R interface: `navis.interfaces.r.load_rda`

##### Improvements
- Blender interface: improved scatter plot generation

## Version `0.2.2` { data-toc-label="0.2.2" }
_Date: 15/08/20_

##### Additions
- New `plot3d` parameter: with plotly backend, use `fig` to add data to existing plotly figure
- New `plot3d` parameter: with vispy backend, use `center=False` to not re-center camera on adding new data
- New `r.mirror_brain` parameter: use e.g. `via='FCWB'` if source space does not have mirror transform
- New `NeuronList` method: `append()` works like `list.append()`
- First implementation of smarter (re-)calculation of temporary Neuron properties using `.is_stale` property
- Neurons can now be multiplied/divided by array/list of x/y/z coordinates for non-isometric transforms

##### Fixes
- Fix issues with newer rpy2 versions
- Various improvements and bug fixes

## Version `0.2.1` { data-toc-label="0.2.1" }
_Date: 20/04/20_

##### Additions
- New `plot3d` parameter: with plotly backend, use `radius=True` plots TreeNeurons with radius
- New `plot2d` parameter: `orthogonal=False` sets view to perspective

##### Improvements
- Various improvements to e.g. `nx2neuron`

## Version `0.2.0` { data-toc-label="0.2.0" }
_Date: 29/06/20_

##### Breaking
- `navis.nx2neuron` now returns a `navis.TreeNeuron` instead of a `DataFrame`

##### Additions
- New neuron class `navis.MeshNeuron`
- New `navis.TreeNeuron` property `.volume`
- New example data from the Janelia hemibrain data set

##### Improvements
- Clean-up in neuromorpho interface
- We now use [ncollpyde](https://pypi.org/project/ncollpyde) for ray casting (intersections)

##### Fixes
- Fix bugs in `navis.Volume` pickling

## Version `0.1.16` { data-toc-label="0.1.16" }
_Date: 26/05/20_

##### Fixes
- Many small bugfixes

## Version `0.1.15` { data-toc-label="0.1.15" }
_Date: 15/05/20_

##### Improvements
- Improvements to R and Blender interface
- Improved loading from SWCs (up to 2x faster)
- `TreeNeurons`: allow rerooting by setting the `.root` attribute

## Version `0.1.14` { data-toc-label="0.1.14" }
_Date: 05/05/20_

##### Fixes
- Emergency fixes for critical bugs

## Version `0.1.13` { data-toc-label="0.1.13" }
_Date: 05/05/20_

##### Additions
- new function: `navis.vary_color`

##### Improvements
- improvements to Blender interface and various other functions

## Version `0.1.12` { data-toc-label="0.1.12" }
_Date: 02/04/20_

##### Imnprovements
- `navis.Volume` is now sublcass of `trimesh.Trimesh`

## Version `0.1.11` { data-toc-label="0.1.11" }
_Date: 28/02/20_

##### Improvements
- improved `navis.stitch_neurons`: much faster now if you have iGraph

##### Fixes
- fixed errors when using multiprocessing (e.g. in `NeuronList.apply`)
- fixed bugs in `navis.downsample_neuron`

## Version `0.1.10` { data-toc-label="0.1.10" }
_Date: 24/02/20_

##### Fixes
- Fixed bugs in Blender interface introduced in 0.1.9

## Version `0.1.9` { data-toc-label="0.1.9" }
_Date: 24/02/20_

##### Fixes
- Removed hard-coded swapping and translation of axes in the Blender interface
- Fixed bugs in `navis.stitch_neurons`

## Version `0.1.8` { data-toc-label="0.1.8" }
_Date: 21/02/20_

##### Fixes
- Again lots of fixed bugs

## Version `0.1.0` { data-toc-label="0.1.0" }
_Date: 23/05/19_

##### Fixes
- Many small bugfixes

## Version `0.0.1` { data-toc-label="0.0.1" }
_Date: 29/01/19_

##### Fixes
- First commit, lots to fix.

