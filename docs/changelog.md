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
- **the neuron classes have been renamed.** The old names said "neuron" three times over and buried the part that actually distinguishes them; the new ones just name the representation:

    | Old | New |
    |---------|-------------|
    | `navis.TreeNeuron` | [`navis.Skeleton`][] |
    | `navis.MeshNeuron` | [`navis.Mesh`][] |
    | `navis.VoxelNeuron` | [`navis.Voxels`][] |
    | `navis.Dotprops` | [`navis.Dotprops`][] (unchanged) |

    The old names still work and will be removed in a future version. They are **aliases**, not subclasses - `isinstance(x, navis.TreeNeuron)` and `class MyNeuron(navis.TreeNeuron)` behave exactly as before, so downstream packages keep working until they get around to it. Neurons pickled by earlier versions still load. `navis.TreeNeuron` raises a `DeprecationWarning` once per session; Python hides those by default outside `__main__`, so run with `python -W default::DeprecationWarning` (or `pytest -W default`) to find the old names in your code.

    The `.type` property follows the class, so it now reads `"navis.Skeleton"` rather than `"navis.TreeNeuron"`. Code matching on that string - including anything filtering a [`NeuronList.summary()`][navis.NeuronList.summary] table by `type` - needs updating.

- **NBLAST scores shift very slightly with [navis-fastcore](https://github.com/schlegelp/fastcore-rs) `>= 0.8.0`**, which now takes its internal coordinate precision from the dtype of the input rather than always widening to float64. {{ navis }} `Dotprops` store `points`/`vect` as **float32**, so NBLAST now runs on float32 coordinates - cutting peak memory on a large all-by-all by ~45%, at a cost of ~1e-5 on the scores. Nothing in {{ navis }} changed; upgrading fastcore is enough to see it.

    The scoring maths itself is untouched (it still accumulates in float64) and this does not change which neurons match - on the example neurons the k-nearest-neighbour identities are unchanged. But it is enough to break a bit-for-bit comparison against previously saved scores: cast `.points` and `.vect` to `float64` (see [`cast_neuron`][navis.cast_neuron] below) if you need the old numbers exactly.

- **bridging graph: edge weights now mean one thing, and lower always wins.** `weight` was doing two jobs at once - it set the cost `networkx` minimises when *choosing a route*, and it was also the tie-breaker between several registrations connecting the same two templates (where the old code took the *highest*-weight edge). The two uses want opposite things, so no weighting could satisfy both. Now `weight` only ever means "what this hop costs" and **lower weight = more likely to be used** everywhere; which transform serves a hop is decided separately, by the new `prefer_forward` argument (see below).

    This **changes ~29% of the bridging paths in `navis-flybrains`** (no routes gained or lost). Most of that is {{ navis }} no longer inverting a registration when a purpose-built one for that direction was sitting right next to it - e.g. `BANC`→`Cell07` used to invert `Cell07_IS2.list` and now simply uses `IS2_Cell07.list`, at the same path length. 360 fewer routes traverse *any* transform backwards.

- transforms now declare how expensive they are to invert, via `BaseTransform.inverse_weight_factor`. It is `1` wherever the inverse is stored or exact (`AffineTransform`, `H5transform`, `TPStransform`), `2` for [`CMTKtransform`][navis.transforms.CMTKtransform] and `5` for [`ElastixTransform`][navis.transforms.ElastixTransform] - both of which have to *solve* for the inverse numerically. `register_transform`'s `weight_inv` now defaults to `weight * inverse_weight_factor`; passing it explicitly still overrides that
- `reciprocal` is deprecated in favour of `inverse_weight` (`bridging_graph`, `find_bridging_path`, `find_all_bridging_paths`) - one name for one knob. Its default also changed from `0.5` to `1`: {{ navis }} no longer discounts inverse transforms across the board, since each transform now says for itself what inverting it costs. Passing `reciprocal` still works but warns
- **[`Voxels`][navis.Voxels] now avoid the dense grid wherever possible**, which requires [sparse-cubes](https://github.com/navis-org/sparse-cubes) `>= 0.5.0` - now a **core** dependency, pulled in as `sparse-cubes[skeleton]` so that its [dijkstra3d-sparse](https://github.com/schlegelp/dijkstra3d-sparse) accelerator comes along too (skeletonization falls back to `scipy` without it, but TEASAR is ~11x slower at 100k voxels and the gap widens with size). Materialising a grid larger than `navis.config.max_grid_size` (4 GiB) now raises a `MemoryError` instead of being silently OOM-killed - a neuron's grid is sized by its *bounding box*, so a handful of far-apart voxels can imply terabytes. Raise or disable the limit if you hit it on data you know fits
- [`navis.smooth_voxels`][navis.smooth_voxels] treats everything outside the neuron as empty (scipy's `mode="constant"`). It previously used scipy's default, which *reflects* at the canvas boundary and invents signal outside the imaged volume; results change for neurons touching that edge
- **[`navis.find_soma`][navis.find_soma] now returns a single node ID (or `None`) instead of an array of candidates.** It used to hand back every node passing the radius/label filter and leave the choice to the caller, which meant a thick primary neurite could be returned alongside - or instead of - the actual soma. Candidates are now scored by the mean radius of their neighbourhood (within `dist_factor` times their own radius, new argument) so that the fattest *region* wins rather than the fattest single node, and the fattest node of that region is returned; the label-only path takes the most central node of the largest connected label component. Nodes whose radius is missing (`NaN` or `<= 0`, as `guess_radius` writes) are no longer treated as candidates. Code doing `find_soma(n)[0]` or `len(find_soma(n))` needs updating
- **the `rpy2`-based R interface is gone.** It required a working R installation plus `rpy2` on top of it, and existed mainly to move data in and out of the [natverse](https://natverse.org). That job is now done by plain files - [`navis.read_rda`][], [`navis.read_rds`][], [`navis.write_rda`][] and [`navis.write_rds`][] (see below), which need neither R nor `rpy2`. `from navis.interfaces import r` now raises with a pointer to the replacement, and the `navis[r]` install extra no longer exists. See the new [natverse tutorial](../generated/gallery/0_io/tutorial_io_03_r) for the round trip.

    | Removed | Use instead |
    |---------|-------------|
    | `r.neuron2r()` / `r.neuron2py()` | [`navis.write_rds`][] / [`navis.read_rds`][] |
    | `r.load_rda()` | [`navis.read_rda`][] |
    | `r.nblast()` / `r.nblast_allbyall()` | [`navis.nblast`][] / [`navis.nblast_allbyall`][] |
    | `r.xform_brain()` / `r.mirror_brain()` | [`navis.xform_brain`][] / [`navis.mirror_brain`][] |
    | `r.get_neuropil()` / `r.get_brain_template_mesh()` | [flybrains](https://github.com/navis-org/navis-flybrains) |
    | `r.init_rcatmaid()` | [pymaid](https://pymaid.readthedocs.io) |

- **the `vispy` backend is gone.** It was deprecated in `1.7.0` in favour of [octarine](https://schlegelp.github.io/octarine/), which does the same job on modern WGPU instead of OpenGL. `plot3d(..., backend="vispy")` now raises, as does `NAVIS_PLOT3D_BACKEND="vispy"`; the terminal auto-pick is octarine :material-arrow-right-thin: plotly. This removes ~2.3k lines - 28% of the plotting package - and with them the second `Viewer` implementation and the `vispy-*` install extras.

    Concretely, `navis.Viewer` **has been removed**: the viewer is now `octarine.Viewer`, which [`navis.plot3d`][] returns and whose [documentation](https://schlegelp.github.io/octarine/) covers its methods. [`navis.get_viewer`][], [`navis.clear3d`][], [`navis.pop3d`][] and [`navis.close3d`][] are unaffected - they were never vispy-specific. `navis.utils.check_vispy` is gone with no replacement, as are the vispy-only [`navis.plot3d`][] arguments `combine`, `shading`, `shininess` and `name` (`title` is unaffected - it was always plotly-only). `navis.matching.matching_pipeline` has been ported to octarine and behaves the same, except that the vispy-only "cycle through neurons by hiding them" mode is not available.

- **[navis-fastcore](https://github.com/schlegelp/fastcore-rs) is now a required dependency (`>= 0.10.0`), and the pure-Python/igraph/scipy fallbacks are gone.** {{ navis }} has shipped two implementations of its graph, geodesic, morphometric and NBLAST core for several releases - a fastcore fast path and a fallback for installs without it - and kept them agreeing bit-for-bit. That is over: fastcore is the engine.

    For anyone installing with `pip`, nothing changes but the install: fastcore ships prebuilt wheels for macOS (Intel/ARM), Windows, and Linux on x86-64, aarch64, i686, armv7l, ppc64le and s390x. Only musl-based Linux (Alpine) builds from source and therefore needs a Rust toolchain. `pip install navis[fastcore]` still resolves - it is now a no-op alias.

    Two behavioural notes. `navis.geodesic_matrix` on a `Mesh` no longer accepts `directed=True` as anything but a no-op (it was already ignored for meshes). And the private helpers behind skeleton stitching - `_stitch_edges`, `_segment_radii`, `_rewire_from_edges`, `_component_labels` - have been removed along with the KDTree stitcher they made up; [`heal_skeleton`][navis.heal_skeleton] and [`heal_mesh`][navis.heal_mesh] are unaffected.

- **the non-fastcore transform backends are deprecated and will be removed in 3.0.** The CMTK/elastix `"binary"` backend (shelling out to `streamxform`/`transformix`) and the `"python"` landmark backend (`morphops`/`molesq`) still work, but selecting either now emits a `DeprecationWarning`. `"auto"` - the default - has always preferred fastcore, so this only affects code that asked for them by name. Image transforms (`xform_image`, `to_dfield`) still need CMTK and are not affected
- **`navis.betweenness_centrality(from_=...)` now counts sources within one hop of a root.** That branch never computed betweenness: it walked root→source paths and tallied every node but the source, which is simply "how many of `from_` lie below this node". It additionally discarded paths of two nodes or fewer, so a source sitting on or next to a root contributed nothing. Those sources now count like any other. In practice only the root and its immediate children move - on the example neuron exactly one node changes, by 3. [`navis.find_main_branchpoint`][navis.find_main_branchpoint] (the one caller) is unaffected, since roots are never branch points
- **[`navis.collapse_nodes`][navis.collapse_nodes] no longer re-roots the neuron at the collapsed node.** That was a side effect of how it rewired, not a documented behaviour; roots are now left where they were. It also no longer raises on real node IDs - see Fixes
- **[`navis.rewire_skeleton`][navis.rewire_skeleton] roots leftover components at their lowest node ID** rather than at whatever `set.pop()` returned. Same edge set, same tree; the docstring already promised nothing better than "arbitrary" for those roots, but the choice is now deterministic
- **`navis.graph.node_label_sorting` breaks ties differently.** Where two branches have exactly equal sort keys their order now follows the node table, deterministically; it used to follow the edge-insertion order of the networkx graph underneath, which carried no stability guarantee at all. On the example neuron 14 of 1217 positions move; the node *set* and the keys themselves are unchanged. This feeds `skeleton_adjacency_matrix(sort=True)`
- [`navis.betweenness_centrality`][navis.betweenness_centrality] is now the only spelling of the function. The old `navis.betweeness_centrality` is gone

##### Additions
- **new [`navis.masked`][navis.masked]: work on part of a neuron, then put it back.** [`navis.subset_neuron`][] is a one-way street - you get a fragment, and its relationship to the neuron it came from is gone. A mask keeps that relationship: inside the block the neuron *is* the masked region, so every function, property and plot sees only that part with no special-casing anywhere; on the way out it becomes whole again, with any edits folded back in. Masking happens **in place** and masks **nest**. Works on `Skeletons`, `Meshes` and `Dotprops`, and on whole `NeuronLists`; the methods underneath ([`Neuron.mask()`][navis.BaseNeuron.mask], [`.unmask()`][navis.BaseNeuron.unmask], [`.apply_mask()`][navis.BaseNeuron.apply_mask]) are there for when a mask has to outlive a block.

    Two new primitives underneath, usable on their own: `subset_neuron(..., track=True)` records where each surviving element came from, and [`navis.merge_subset`][] folds an edited subset back in by joining on that record (and **raises** where the record can no longer be trusted, rather than guessing). Two caveats: merging re-points connectors, tags and the soma at whichever elements survived, so an edit touching none of the elements is silently dropped; and a mask that cuts across branches leaves nodes that look like the ends of the arbour but are not, which anything working from the tips ([`prune_twigs`][navis.prune_twigs], [`strahler_index`][navis.strahler_index], `.leafs`) will act on. See the new [masking tutorial](../generated/gallery/tutorial_basic_03_masking).

- **new [`navis.write_parquet`][]`(..., format="neurarrow")`: parquet files other tools can read.** [neurarrow](https://neurarrow.readthedocs.io) is a specification for neuroanatomical data on Apache Arrow which started from {{ navis }}' own parquet format and has since gone its own way - different column names, `uint64` IDs unique across the whole file, `float64` coordinates, null rather than `-1` for roots, and required metadata. `format="neurarrow"` writes all of that, and [`navis.read_parquet`][] detects and reads neurarrow files without being told - including those written by other tools, such as [swc2na](https://github.com/clbarnes/swc2na).

    Two things don't map cleanly: neurarrow tracks units for the **file**, not the neuron, so all neurons must share the same units (and dotprops the same `k`); and navis' connector table is a set of point annotations where neurarrow's `connections` schema describes *edges*, so connectors are written through the [`net.clbarnes.connector`](https://github.com/clbarnes/neurarrow-ext/blob/main/extensions/connector.md) extension, which has no room for extra columns like `roi`. `format="navis"` remains the default and the only lossless round-trip - see [the format spec](https://github.com/navis-org/navis/blob/master/navis/io/pq_io.md) for the full comparison.

- **new [`navis.plot3d`][]`(..., snapshot=True)`: a rendered 3D scene on a matplotlib axes, in data coordinates.** Between [`navis.plot2d`][], which draws real matplotlib artists but has no renderer behind it, and [`navis.plot3d`][], which renders properly but hands back an interactive window, there was no way to get a *figure* you can annotate, arrange into a panel and save as PDF. `snapshot=True` renders the scene through octarine's offscreen canvas and returns `(fig, ax)` with the image already placed - and because the image extent is read back off the camera, `ax` is in the **neurons' own coordinates**, so `ax.annotate("soma", soma_pos[[0, 2]])` lands on the soma and distances are to scale.

    `view` takes the same `("x", "-y")` axis pairs as [`navis.plot2d`][], so the two can be mixed in one figure; pass a camera state dict from `viewer.get_view()` to reproduce a view you set up interactively, or `viewer=` to shoot an existing viewer. Only axis-aligned orthographic cameras can be expressed in data coordinates - for anything else the axes fall back to world units in the view plane. Also new alongside it: `margin`, `bgcolor` (the render is transparent by default), `size`/`pixel_ratio` and the usual `ax`, `figsize` and `dpi`. See the new [3D skeletons tutorial](../generated/gallery/1c_plotting_3d/tutorial_plotting_3d_00_skeletons).

- **new [`navis.find_soma_mesh`][navis.find_soma_mesh]: soma detection for [`Meshes`][navis.Mesh], straight off the mesh.** No skeletonization involved: it finds the thickest part of the neuron (the point of largest inscribed sphere) and fits an oriented ellipsoid to the surrounding surface, returning the new [`navis.SomaEllipsoid`][navis.SomaEllipsoid] - `center`, `radii`, principal `axes`, `inscribed_radius`, plus `volume`, `equiv_radius`, `contains()` and `distance_to_surface()`. With `inplace=True` it simply sets the neuron's `.soma_pos`. The approach is inspired by [skeliner](https://github.com/berenslab/skeliner); `min_soma_radius` (accepts e.g. `"1 micron"` if the neuron has units) is the main accept/reject knob and should be tuned to your data
- **new [`navis.heal_mesh`][navis.heal_mesh]: the mesh counterpart to [`navis.heal_skeleton`][navis.heal_skeleton].** Meshes often consist of several disconnected fragments - because the segmentation had a gap, or because meshing produced separate closed surfaces where the neuron is continuous. `heal_mesh` reconnects them with the set of bridges that minimises the total added length (a true minimum spanning tree over the fragments), subject to the same `max_dist`, `min_size`, `mask` and `drop_disc` knobs as its skeleton sibling. The repair is purely topological: bridges land in `.extra_edges` (see below) so vertices, faces, area and volume are all untouched. A 100k-vertex mesh takes ~15 ms
- **new [`Mesh.extra_edges`][navis.Mesh.extra_edges]: connectivity that the surface itself does not have.** A mesh's topology is implied by its faces, so there is no way to express "these two vertices are connected" without inventing geometry. Extra edges are exactly that: an `(N, 2)` array of vertex indices that is part of the *graph* but not of the *surface*. Everything that derives connectivity from a mesh now sees them - [`geodesic_matrix`][navis.geodesic_matrix], [`break_fragments`][navis.break_fragments], [`drop_fluff`][navis.drop_fluff], `.igraph`/`.graph` - while anything describing the surface (e.g. `.sampling_resolution`) does not. They are remapped through [`subset_neuron`][navis.subset_neuron] and [`combine_neurons`][navis.combine_neurons], and dropped whenever the number of vertices changes. Note that mesh file formats have no place for them: they are lost on export
- **new functions for analyzing the angles in a skeleton** (see the new section in the morphometrics tutorial): [`navis.branch_angles`][navis.branch_angles] (between child branches at each branch point), [`navis.path_angles`][navis.path_angles] (in- vs outgoing edge at each continuation node, i.e. how much the path bends), [`navis.root_angles`][navis.root_angles] (how far each edge deviates from pointing radially away from the root) and [`navis.soma_exit_angles`][navis.soma_exit_angles] (between the neurites emanating from the soma). All return a tidy per-node DataFrame, work on `Meshes` via their skeleton, and map over `NeuronLists`
- **new [`navis.cast_neuron`][navis.cast_neuron]: convert a neuron's data to a given dtype.** What gets cast depends on the neuron type - node `x`/`y`/`z`/`radius` for skeletons, vertices for meshes, points (and, for float dtypes, tangent vectors and alpha) for `Dotprops`, voxel *values* for `Voxels` - plus connectors for all of them. Anything that *indexes* into those (mesh faces, voxel coordinates, node/parent IDs) is left alone. Handy for e.g. the float32/float64 NBLAST question above
- **new [`navis.transforms.SimilarityTransform`][navis.transforms.SimilarityTransform]**: a rotation, a translation and a single uniform scale factor fitted to matched landmarks - the least-squares optimal solution in closed form via SVD (Umeyama, 1991), with optional per-landmark `weights`. Set `scale=False` for a pure rigid transform. Unlike a [`TPStransform`][navis.transforms.thinplate.TPStransform] it has only 7 degrees of freedom and so will generally *not* map the landmarks onto each other exactly - which is the point when you want the landmarks to constrain a global transform rather than warp space to fit them
- **new [`navis.nblast_knn`][navis.nblast_knn]: the `k` nearest neighbours of every neuron, without ever building the score matrix.** An all-by-all is the wrong shape for a k-NN question at scale - 164k neurons is 2.7e10 pairs and a 107 GB matrix, when what is wanted from it is a 26 MB k-NN graph (typically to feed a UMAP embedding). This computes that graph directly: each neuron is reduced to a coarse voxel-occupancy signature, the `n_candidates` most similar neurons per row are shortlisted from those signatures, and the *exact* NBLAST score is then computed for the shortlisted pairs only. Only the shortlisting is approximate - every returned score is a real NBLAST score. Measured on 163,976 neurons, recall@20 is 0.990 at the default `n_candidates`, having scored 0.16% of pairs. Returns a tidy `query`/`target`/`score`/`rank` frame by default; `format="wide"` gives the [`extract_matches`][navis.nbl.extract_matches] layout and `format="arrays"` the raw arrays UMAP's `precomputed_knn` wants. Unlike the other NBLAST functions this one is provided only by navis-fastcore
- **new [`navis.graph.geodesic_clusters()`][navis.graph.geodesic_clusters]**: greedily partitions a skeleton or mesh into connected clusters of bounded geodesic radius. Please read its warning before using it for downsampling - the clusters are guaranteed connected and bounded, but they are *not* evenly sized and their centroids are not evenly spaced
- **[`navis.make_dotprops`][navis.make_dotprops] is ~12x faster**: the tangent vectors and alpha values (96% of its runtime) now come from one parallel Rust pass instead of a `scipy.spatial.cKDTree` query plus N 3x3 SVDs. Same for [`navis.Dotprops.recalculate_tangents`][navis.Dotprops.recalculate_tangents]. The two agree exactly except where the k-nearest-neighbour search hits a *tied* distance, which grid-quantised coordinates produce readily - there the k-th neighbour is genuinely ambiguous and the two trees may pick differently (~0.3% of points on the example neurons)
- **[`navis.nbl.extract_matches`][navis.nbl.extract_matches] is much faster and gained a `max_matches` guard.** All three criteria now go through navis-fastcore: `N` is 6-118x faster (the gap widens with matrix size - 2.8 s to 23 ms on a 20k x 20k matrix), `threshold` 1.5-8x, `percentage` 1.3-1.9x. `max_matches` refuses to return more than a given number of matches for `threshold`/`percentage`, whose output size is not knowable in advance - an over-broad cutoff on a large matrix could previously take the machine down. The count is established before anything is allocated
- **the landmark transforms now have a fastcore backend too.** [`TPStransform`][navis.transforms.thinplate.TPStransform] and [`MovingLeastSquaresTransform`][navis.transforms.MovingLeastSquaresTransform] use navis-fastcore's Rust implementation; the deprecated `morphops`/`molesq` path agrees with it to ~1e-13. Neither materialises the `(n_points, n_landmarks)` intermediate those libraries build, so `batch_size` stops mattering and peak memory no longer scales with the landmark count - which is what made `MovingLeastSquaresTransform` impractical at the landmark counts real registrations use (3400 landmarks needed ~23 GB at the default batch size). Transforming points is ~10-15x faster. Note that `TPStransform` still *fits* its spline with `morphops` even on the fastcore backend, because numpy's LAPACK-backed solve is faster there; only the point transform switches
- **CMTK and elastix point transforms can now run without the external binaries.** [`CMTKtransform`][navis.transforms.CMTKtransform] and [`ElastixTransform`][navis.transforms.ElastixTransform] now use navis-fastcore's in-process Rust implementation instead of shelling out to `streamxform`/`transformix`: no CMTK, no elastix, no subprocess and no temporary files. Results match the binaries to ~1e-6 (including which points fail), and `xform_brain` is 4-30x faster end-to-end. Image transforms (`xform_image`, `to_dfield`) still require CMTK. Control this with `navis.config.default_transform_backend` (`"auto"` by default, i.e. fastcore) or per transform via `backend=`
- elastix transforms are now *invertible* when navis-fastcore is installed - something `transformix` cannot do at all. Note the bridging graph does not use this by default: set `navis.config.elastix_invertible = True` to let it (see the notes on [`ElastixTransform`][navis.transforms.ElastixTransform] for why it is off)
- `find_bridging_path`, `find_all_bridging_paths` and `shortest_bridging_seq` gained a `prefer_forward` argument (default `True`): where two templates are connected by both a purpose-built registration and the inverse of its counterpart, use the purpose-built one - regardless of weight. Set it to `False` to have your graph's weights taken entirely at face value
- `MovingLeastSquaresTransform` gained a `.matrix_affine` property, analogous to `TPStransform.matrix_affine`: since moving least squares is a *locally* weighted affine transform, this returns the global (least-squares) affine that it converges to far away from the landmarks
- **[`Voxels`][navis.Voxels] gained a proper toolkit**, all of it working straight off the sparse voxels:
    - morphology and set algebra: `dilate`, `erode`, `opening`, `closing`, `thin`, `fill_cavities`, `union`, `intersection`, `difference`, `symmetric_difference`. Per-voxel values are carried through; set operations align neurons onto a common lattice and refuse to combine ones that do not line up
    - measurements: `surface_area`, `centroid`, `distance_transform`, `connected_components`, `iou`, `dice`, `grid_nbytes`/`voxels_nbytes`
    - shorthands `.mesh()` and `.skeletonize()`

- **[`navis.skeletonize`][navis.skeletonize] now accepts [`Voxels`][navis.Voxels]** (via the new [`navis.conversion.voxels2skeleton`][navis.conversion.voxels2skeleton]), closing a gap its own docstring used to flag. Defaults to `method="wavefront"` - ~4x faster than `"teasar"` and radii come free from the ring contraction rather than being snapped to the voxel lattice; `"teasar"` and `"thin"` remain available
- **existing functions stopped densifying.** [`navis.drop_fluff`][navis.drop_fluff] and `graph_utils._connected_components` now handle `Voxels`; [`navis.smooth_voxels`][navis.smooth_voxels], [`navis.thin_voxels`][navis.thin_voxels] and [`navis.downsample_neuron`][navis.downsample_neuron] no longer allocate the grid (the latter could trip the new memory cap on exactly the sparse neurons worth downsampling). Voxel adjacency - behind `neuron2nx`/`neuron2igraph` - is ~100x faster and no longer needs the *undeclared* scikit-learn dependency. `smooth_voxels`/`thin_voxels` keep a `backend` argument if you want the old scipy/scikit-image route
- **new `navis.ml` module: helpers for preparing neurons as machine-learning inputs.** These live under their own `navis.ml.*` namespace (deliberately *not* lifted to the top level) and come with three tutorials:
    - **normalize** - [`navis.ml.normalize_neuron`][navis.ml.normalize_neuron] maps a neuron into a canonical frame with a single rigid-plus-uniform-scale transform: center on the centroid/bbox/soma/an explicit point, PCA-orient (signs disambiguated deterministically and handedness preserved - neurons are never mirrored), and scale to unit RMS / extent / enclosing sphere. `return_matrix=True` returns the 4x4 matrix so predictions made in the normalized frame can be mapped back
    - **augment** - geometry/sampling augmentations, each taking a `random_state` and returning a copy: [`jitter_neuron`][navis.ml.jitter_neuron], [`warp_neuron`][navis.ml.warp_neuron], [`rotate_neuron`][navis.ml.rotate_neuron], [`scale_neuron`][navis.ml.scale_neuron], [`translate_neuron`][navis.ml.translate_neuron] and [`drop_nodes`][navis.ml.drop_nodes]. [`augment_neuron`][navis.ml.augment_neuron] chains them in one seeded, reproducible call
    - **sample & chunk** - turn a variable-sized neuron into a fixed-size model input: [`sample_cable`][navis.ml.sample_cable] (arclength-uniform sampling along a skeleton), [`sample_surface`][navis.ml.sample_surface] (area-weighted mesh-surface sampling), both with per-point provenance, plus [`chunk_neuron`][navis.ml.chunk_neuron] (tile a neuron into evenly-sized fragments for batching)

- **new [`navis.write_rds`][navis.write_rds] and [`navis.write_rda`][navis.write_rda]: hand data to R's [natverse](https://natverse.org) without `rpy2`.** `Skeleton` :material-arrow-right-thin: `nat::neuron` (including nat's `SegList`/`SubTrees` topology, so `resample()`, `prune_*()` etc. work on the result), `Dotprops` :material-arrow-right-thin: `nat::dotprops`, `Mesh`/`Volume` :material-arrow-right-thin: `rgl::mesh3d`, `Voxels` :material-arrow-right-thin: `nat::im3d` and `NeuronList` :material-arrow-right-thin: `nat::neuronlist` (with its meta data attached); plain dicts, lists, arrays and `DataFrames` become the corresponding R types. The serialisation is done by [rdata](https://github.com/vnmabus/rdata), so no R installation and no `rpy2` are required on either end; read the files in R with `readRDS()`/`load()`
- **new [`navis.read_rds`][navis.read_rds]**, the counterpart to [`navis.read_rda`][navis.read_rda] for `.rds` files (a single, unnamed R object rather than a whole workspace)
- **`neuprint` interface: new `dedup` option for fetching synapses.** In insect connectomes synapses are polyadic - a single presynapse connects to multiple postsynapses - and `neuprint-python` de-duplicates presynapses so each is reported only once. [`fetch_synapses`][navis.interfaces.neuprint.fetch_synapses] (a new thin wrapper over `neuprint.fetch_synapses`), [`fetch_skeletons`][navis.interfaces.neuprint.fetch_skeletons] and [`fetch_mesh_neuron`][navis.interfaces.neuprint.fetch_mesh_neuron] now take `dedup` (default `True`, i.e. unchanged behaviour). With `dedup=False` each presynapse is instead reported once per postsynaptic site it connects to
- **`parallel=True` now runs on a backend you can choose - including one you supply.** Where per-neuron work runs used to be a hard-wired `pathos` pool; it is now pluggable via the new [`navis.set_parallel_backend`][] (also usable as a context manager) and a per-call `backend=` parameter. Ships `pathos`, `joblib`, the standard library's process pool (`processes`), `threads` and `serial`; [`navis.list_parallel_backends`][] shows what's installed and third parties can register their own with `navis.compute.register_backend`. **`parallel=True` therefore no longer needs `pathos`** - though `pathos`/`joblib` are still needed to ship a lambda - and [`navis.set_parallel_backend`][] accepts any `concurrent.futures.Executor`, which is how you point {{ navis }} at a cluster without {{ navis }} ever needing scheduler options of its own.

    `backend="auto"` (the default) prefers `joblib`, then `pathos`, then the standard library's pool. **This changes which backend existing `parallel=True` calls use**: `pathos` was the only option before. The two behave the same, but `joblib` keeps its workers alive between calls where `pathos` builds a fresh pool each time, which makes a sequence of parallel calls measurably faster; pass `backend="pathos"` to keep the old one. Note also that `inplace=False` is only silently turned into an in-place run where the workers get *copies* of the neurons - a thread-based or in-process backend now honours it.

- **`parallel=True` can now run across a cluster**, via two new backends: `dask` (a [dask.distributed](https://distributed.dask.org) cluster, from a `LocalCluster` to a compute centre) and `submitit` (an array job on SLURM). Install with `pip install navis[cluster]` - deliberately *not* part of `navis[all]`. Neither backend is ever selected automatically.

    Nothing about how you call {{ navis }} changes: `parallel=True` still only means "spread this over the neurons", and all scheduler configuration stays on your own object - [`navis.set_parallel_backend`][] takes a `dask.distributed.Client` (or a cluster) and a `submitit.Executor` directly. Both bundle neurons into fewer, larger units of work, aiming for enough units to keep every worker busy while capping each at ~128 MB of neurons; `chunksize=` still overrides it per call. See the [multiprocessing tutorial](../generated/gallery/6_misc/tutorial_misc_00_multiprocess) for both in context.

- **NBLAST runs on the same backends, so a big one can go to a cluster.** The built-in NBLAST backend used to build its own `ProcessPoolExecutor`, which meant [`navis.set_parallel_backend`][] had no effect on it; it now dispatches through the same layer as everything else. The unit of work is a *block of the score matrix* rather than a neuron, sized from a per-block runtime budget so that each is seconds to minutes of work no matter how many neurons you have. Scores are unchanged - bit-identical, on every backend. On a single machine, where the default is now `joblib` rather than a private pool, a *repeated* NBLAST is **~1.75x faster** because the workers are no longer thrown away and rebuilt between calls (150 neurons all-by-all on 8 cores: 6.3s :octicons-arrow-right-24: 3.6s); they stay resident for a minute or so afterwards and `navis.compute.shutdown()` reclaims them at once.

    !!! warning "navis-fastcore does not distribute"
        The [navis-fastcore](https://github.com/schlegelp/fastcore-rs) NBLAST backend computes the whole matrix in one Rust call with its own threads, so it ignores the parallel backend entirely and runs everything locally. It is not the default (`navis.config.default_nblast_backend` is `"builtin"`), but it is what `"auto"` picks where it is installed - so leave that alone, or pass `backend="builtin"`, when you want a distributed NBLAST.

##### Improvements
- **two dependencies are gone and two are no longer imported up front.** `six` was a Python-2 compatibility shim and `pypng` had not been imported anywhere for several releases; both are dropped from the requirements. `morphops` and `molesq` are now imported on first use rather than at module level, taking ~280 ms off `import navis`. `molesq` is now only reached by the deprecated `"python"` backend; `morphops` is still needed by every thin plate spline transform, on either backend, since it does the *fit*
- **the plotting tutorials have been restructured**: *General* (the modes and backends, plus one page on colouring that absorbs the old depth-colouring tutorial), *2D plotting* and *3D plotting* (one page each for skeletons, meshes and volumes), *Other plots* (connectors, barcodes, topology, XKCD) and *Examples*. The styling pages compare settings side by side rather than one render at a time - and the 3D ones do it with `plot3d(..., snapshot=True, ax=...)`, so every panel is a real octarine render. Old URLs under `generated/gallery/1_plotting/` have moved to `1a_plotting_general/`, `1b_plotting_2d/`, `1c_plotting_3d/`, `1d_plotting_misc/` and `1e_plotting_examples/`
- **there is a [connectors tutorial](../generated/gallery/1d_plotting_misc/tutorial_plotting_misc_00_connectors)** covering every `cn_*` parameter; connector plotting was previously a three-panel aside in the 3D skeletons page. `cn_zorder` is documented for the first time - it has always worked, and appeared in no docstring
- **connectors are drawn in one artist per neuron, in a fixed shuffled order.** They used to go down one `type` at a time, which meant the type painted last won wherever markers overlap - so a rare type could bury a common one. On the example neuron that is not hypothetical: 232 presynapses sat on top of 1933 postsynapses and the antennal lobe read as an *output* region. Interleaving the draw order makes the visible mix a fair sample of the real one; the permutation is seeded, so the same data still gives the same figure
- **`cn_color_by` colours connectors by any column of the connector table** (or by an array with one value per connector) instead of by their `type` - ROI, confidence, partner id, whatever the table carries. Numerical data gets a colormap, categorical data one colour per level, and `cn_palette` picks the palette. The scale is worked out across all neurons in the call, so the same value is the same colour throughout; missing values are drawn grey. Supported by [`navis.plot2d`][] and by `plot3d`'s plotly and k3d backends - the latter two draw markers rather than stalks when it is set, since a line there carries a single colour
- **`cn_legend=True` explains the connector colours** - one entry per connector *type* for the whole axes rather than one per type per neuron, or a colorbar when `cn_color_by` is numerical. `navis.plot2d` only; call `ax.legend()` afterwards as usual
- **[`navis.plot2d`][] now honours `cn_layout={"display": "lines"}`**, which draws a stalk from each connector back to the node it belongs to instead of a free-floating marker. That has always been the *default* in `navis.config.default_connector_colors` and has always been what plotly and k3d drew; `matplotlib` silently drew markers regardless. Pass `cn_layout={"display": "circles"}` for the old look. Meshes have no nodes to point at and still get markers
- **[`navis.plot2d`][] (`method="2d"`) now renders meshes as surfaces rather than as flat blobs.** Faces pointing away from the viewer are dropped and the rest painted furthest-first, so a [`Mesh`][navis.Mesh] or [`Volume`][navis.Volume] finally occludes itself. Previously faces went down in whatever order the mesh stored them, which meant a dense arbour collapsed into a single silhouette and the last neuron drawn covered everything under it. There is no switch to turn this off - it is a correctness fix, and because half the polygons never reach the renderer it is also considerably faster (build / draw on the example neurons):

    | | Before | After |
    |---|---|---|
    | one mesh neuron (13k faces) | 32 / 28 ms | 4.8 / 12 ms |
    | five mesh neurons | 241 / 107 ms | 12 / 19 ms |
    | one 209k-face mesh | 513 / 278 ms | 30 / 28 ms |

    A mesh in a single colour is now filled as **one path** instead of one polygon per face, which is where most of that speed-up comes from - and it fixes `alpha`, which used to composite every triangle separately, so a fold in the surface came out darker than a flat stretch.

- **`mesh_shade` lights meshes and volumes in 2d.** It was a bool for the 3d methods only; with `method="2d"` it now takes a mode - `True`/`"lambert"`, `"cel"`, `"rim"` or `"ghost"` (opacity from the grazing angle instead of brightness, so a neuron inside a neuropil stays visible). Pass a dict to tune `light`, `ambient` and `strength`. Shading *multiplies into* whatever colour a face already has, so it composes with `color`, `color_by` and `depth_coloring` rather than replacing them. `style="publication"` turns it on - pass `mesh_shade=False` alongside it to keep the flat fill. See the new [mesh plotting tutorial](../generated/gallery/1b_plotting_2d/tutorial_plotting_2d_01_meshes)
- **`depth_sort="global"` sorts exactly instead of binning.** `depth_sort=<int>` buckets everything into N bins along the view axis, so two neurites in the same bin still fall back to draw order. `"global"` merges each neuron type into a single artist and sorts all of its elements together - segment by segment for skeletons, face by face for meshes. For skeletons this is *cheaper* than the default 10 bins; for meshes it costs several times either, because sorting across neurons forces one polygon per face. `halo` is unavailable on this path and passing both warns and falls back to bins
- **`halo` and `depth_sort` now work for meshes**, not just skeletons. `depth_sort` puts mesh faces into the same depth bins skeleton nodes go into, so the two kinds of neuron share one stack - without it a mesh is drawn over every skeleton regardless of where it sits
- **`halo` is visible again without `depth_sort`.** Every neuron shared one z-order, so all the halos ended up underneath all the neurons - which on a white background means nothing at all. Each neuron now gets its own slot within the same band, so input order becomes the stacking order
- **`volume_outlines` draws its contour opaque.** A volume's alpha is a *fill* alpha - there is nothing to see through a line, so at the 10-20% volumes default to the contour was all but invisible. `volume_outlines="both"` already did this; `True` now matches it
- [`Volume.to_2d`][navis.Volume.to_2d] warns when the alpha shape falls apart and it retries at a tenth of the alpha. The retry is silent no longer: it is why a too-large `volume_outlines_alpha` looks like it did nothing rather than like it was rolled back
- **`depth_sort` bins now run towards the viewer for every view.** They are indexed along the raw depth axis, which for half the possible views (`("x", "z")`, `("x", "-y")`, ...) points *away* - so the far end of the neuron was drawn on top. Views where it already worked (including the default `("x", "y")` and `("x", "-z")`) are unchanged, and a negative `depth_sort` still flips whatever it works out
- **the graph internals now run on [navis-fastcore](https://github.com/schlegelp/fastcore-rs) instead of igraph/networkx.** Each of these was a general graph algorithm answering a question about a *rooted forest*, where the answer is a linear pass over the parent vector - so building a graph object cost more than the answer did. Measured on the example neuron (4465 nodes):

    | Function | Before | After |
    |---|---|---|
    | [`betweenness_centrality`][navis.betweenness_centrality] (`directed=True`) | 11.1 ms | 0.48 ms |
    | [`betweenness_centrality`][navis.betweenness_centrality] (`directed=False`) | 217 ms | 0.49 ms |
    | [`betweenness_centrality`][navis.betweenness_centrality] (`from_=...`) | 10.6 ms | 0.83 ms |
    | [`find_main_branchpoint`][navis.find_main_branchpoint] (`"longest_neurite"`) | 2.8 ms | 0.40 ms |
    | [`find_main_branchpoint`][navis.find_main_branchpoint] (`"betweenness"`) | 9.3 ms | 1.4 ms |
    | [`split_into_fragments`][navis.split_into_fragments] (`n=5`) | 20.1 ms | 7.6 ms |
    | [`reroot_skeleton`][navis.reroot_skeleton] | 2.3 ms | 0.75 ms |
    | [`cut_skeleton`][navis.cut_skeleton] | 7.4 ms | 3.6 ms |
    | [`collapse_nodes`][navis.collapse_nodes] | 10.8 ms | 1.9 ms |
    | [`rewire_skeleton`][navis.rewire_skeleton] | 9.5 ms | 2.9 ms |
    | [`edges2neuron`][navis.edges2neuron] (`validate=True`) | 3.5 ms | 0.60 ms |
    | [`cell_body_fiber`][navis.cell_body_fiber] | 12.3 ms | 3.0 ms |
    | `node_label_sorting` | 14.4 ms | 3.5 ms |
    | [`Skeleton.is_tree`][navis.Skeleton.is_tree] | 2.3 ms | 0.65 ms |

    `betweenness_centrality` gains the most because shortest paths in a tree are unique, so betweenness has a closed form (descendants × ancestors) and needs neither Brandes nor a graph. Values are unchanged - bit-identical to igraph's, counted in `int64` since an undirected 100k-node skeleton reaches ~5e9.

- a mesh's unique edges now come from navis-fastcore (new `navis.utils.mesh_unique_edges`) instead of `trimesh.edges_unique`, which sorts an `(n_faces * 3, 2)` array to find them. This sits underneath [`neuron2nx`][navis.neuron2nx]/[`neuron2igraph`][navis.neuron2igraph] for `Meshes` and hence everything built on a mesh graph. The results are seeded into trimesh's own cache, so a mesh that has already computed its edges pays nothing

##### Fixes
- **[`navis.subset_neuron`][navis.subset_neuron] has been rewritten onto a declarative per-type schema** (`navis/core/schema.py`) that says which attributes are aligned to which axis - a skeleton's nodes, a mesh's vertices, a dotprops' points - and what references them. The three hand-written implementations had drifted apart in exactly the places nobody was looking; there is now one, plus a test that fails if any bulk field is undeclared. [`navis.masked`][navis.masked] (above) is built on the same schema. Four bugs came out of it:
    - `subset_neuron` **mis-mapped a `Mesh`'s connectors**: it built its old→new vertex index map from the vertex indices *requested*, but `submesh` returns the survivors in sorted order and additionally drops vertices left in no face - so unless the request happened to be sorted and complete, surviving connectors came back attached to the wrong vertex, silently
    - `subset_neuron` **left a `Dotprops`' soma pointing at whatever point had moved into that slot**. A dotprops soma is a point *index*, so a subset has to renumber it; it is now remapped, or dropped if the point did not survive
    - `subset_neuron` did not preserve the node table's **column order** - it moved `parent_id` to the end as a side effect of how it re-rooted the survivors
    - `Dotprops` never took part in the staleness check: the class attribute naming its core data was misspelled `_CORE_DATA`, so `Dotprops.core_md5` was always `None`

- **[`navis.downsample_neuron`][navis.downsample_neuron] left connectors and tags pointing at nodes it had just deleted.** It thinned the node table and never touched either, so on the example neuron a `downsampling_factor=10` produced a skeleton whose 2705 connectors included 1704 referring to node IDs no longer in the table. Both are now moved onto the geodesically nearest surviving node, which is the same stretch of the same branch. Pass `preserve_nodes="connectors"` to pin connectors exactly instead of letting them move
- **[`navis.prune_by_strahler`][navis.prune_by_strahler] left tags and the soma pointing at nodes it had just pruned** - the same class of bug. Tags now lose their pruned nodes (and go away entirely if that empties them) and a soma sitting on a pruned node is cleared
- **[`navis.write_parquet`][navis.write_parquet] silently dropped connectors.** It only ever wrote the node table, so round-tripping any neuron with synapses gave it back with `connectors=None` - no warning, and neither the docstring nor the format spec mentioned it. Connectors now go into a *sidecar* file next to the main one (`neurons.parquet` gets a `neurons.connectors.parquet`), which [`navis.read_parquet`][] picks up automatically; two files rather than a zip archive, so both tables keep parquet's column pruning and predicate pushdown. Pass `write_connectors=False` to opt out. The node table's `label` column was going missing the same way and is now written too
- **[`navis.read_parquet`][navis.read_parquet] returned a `NeuronList` for a single-neuron file**, contradicting its own docstring, because dotprops always carried the neuron-ID column. A file holding exactly one neuron now reads back as a single neuron for both skeletons and dotprops; `subset=`/`limit=` still always give you a `NeuronList`
- **[`navis.collapse_nodes`][navis.collapse_nodes] raised `MemoryError` on real node IDs.** It built an igraph contraction mapping of vertex *indices* but wrote node IDs into it, so it only held together while IDs happened to run `1..N`. Given the 7e17-range IDs segmentation backends hand out, igraph tried to reserve a vector sized by the ID and died. Everything now happens in ID space
- **`navis.geodesic_matrix(directed=True)` reported a coincident child as reachable from its parent**, at distance 0, whenever both `from_` and `to_` were given. A navis-fastcore fix (`0.10.0`) that {{ navis }} inherits: the partial backend used depth as a proxy for ancestry, which only holds while every edge weight is strictly positive. Coincident nodes are routine in traced and resampled skeletons
- **weighted segment lengths were one edge too long.** `navis.graph.graph_utils._generate_segments(..., return_lengths=True)` summed the weight of every node in a segment including the terminal one, whose own child→parent edge belongs to the *parent* segment. Another navis-fastcore fix {{ navis }} inherits: lengths now measure first node to last, so they sum to exactly the neuron's cable length. [`segment_analysis`][navis.segment_analysis], `.segments` and [`persistence_points`][navis.persistence_points] are unaffected - the over-count never reached them
- **`parallel=True` could hang forever once the process had done any real work.** `pathos` starts its workers with `fork` on every platform - including macOS, where the standard library does not - and forking is unsafe after native thread pools (BLAS, or Accelerate on macOS) have come up: only the forking thread survives into the child, so the first call that touches one blocks forever with no error, no output and no progress bar. Skeletonizing a handful of meshes was enough to trigger it. {{ navis }} now starts `pathos` workers with `forkserver`/`spawn`, as it already did for the standard-library backend
- **parallel work printed `resource_tracker: There appear to be N leaked semaphore objects to clean up`.** Building a progress bar - even a disabled one, since the lock is taken in `tqdm.__new__` - makes `tqdm` allocate a cross-process lock, i.e. a named semaphore that nothing then unlinks. Workers now say up front that their bars aren't shared across processes, so no such lock is allocated
- **`omit_failures=True` mislabelled dataframes with the wrong neuron.** Functions returning a dataframe per neuron ([`branch_angles`][navis.branch_angles], [`segment_analysis`][navis.segment_analysis] & co.) paired the results with the input neurons by position. Since failed runs are dropped, every dataframe after the first failure got the wrong ID and the last neuron's results were discarded - silently, and on the serial path as much as the parallel one. Also returns an empty dataframe when everything failed, instead of raising from `pd.concat`
- **`can_zip`/`must_zip` arguments were never actually distributed per neuron.** They were validated against the neuron count and then passed to every neuron whole. In practice this meant [`navis.prune_at_depth`][navis.prune_at_depth] - the only user - raised a broadcast error when given one `source` per neuron, rather than pruning each from its own root
- **`memory_usage(estimate=True)` raised on any neuron with connectors, and `NeuronList` reported `0.0B` because of it.** The estimating path prices each column from its dtype using `dtype.itemsize` - which pandas' extension dtypes do not all have, and from pandas 3 a text column defaults to `StringDtype`. [`NeuronList.memory_usage`][navis.NeuronList.memory_usage] caught *everything* and returned 0, which is why this surfaced only as a `NeuronList` claiming to be `0.0B` in its own repr. Columns that can be sized from their dtype still are; the rest are now costed by pandas itself, and estimates are exact on the example neurons. **`NeuronList.memory_usage` now raises instead of returning 0** when it cannot size the neurons - printing a `NeuronList` still can't fail (it shows `?`), but a returned `0` was indistinguishable from a genuinely empty list
- **{{ navis }}' logger could be left silenced for the rest of the session.** [`arbor_segregation_index`][navis.arbor_segregation_index], [`bending_flow`][navis.bending_flow], [`synapse_flow_centrality`][navis.synapse_flow_centrality] and [`flow_centrality`][navis.flow_centrality] quieten the logger while they build a throwaway downsampled copy, and a neuron's HTML thumbnail does the same while it plots; if any of those steps raised, the whole library went quiet with nothing to say why (and the thumbnail additionally left matplotlib's interactive mode off and a stray figure open). All now go through the new `navis.config.quiet_logger` context manager
- `NeuronList.apply()` forwarded `chunksize` and `progress` to the applied function, raising `TypeError`. Both are now explicit parameters, as are `backend` and the already-reserved `parallel`/`n_cores`/`omit_failures`
- **`pip install navis[all]` pulled in `cloud-volume` and its dependency tree**, which was never the intention - it is a specialized dependency and is meant to be installed on its own (`navis[cloudvolume]`). The exclusion list named the *distribution* (`cloud-volume`) where it needed the *extra* (`cloudvolume`), so it silently matched nothing. `r`, `flybrains` and `cluster` were unaffected
- **[`navis.nblast`][] with `scores="both"` crashed on every multi-core run.** With `"both"`, each query occupies *two* rows of the result, but the code that reassembles the score matrix from its blocks assumed one row per query. Any NBLAST split into more than one block therefore died with `ValueError: setting an array element with a sequence`; only runs that happened to fit in a single block ever worked. While there: [`navis.nblast_smart`][] and [`navis.synblast`][] now **reject** `scores="both"` rather than accepting it and getting it wrong - neither ever implemented it
- **[`navis.nbl.extract_matches`][navis.nbl.extract_matches] ranked `NaN` as the *best* possible score.** numpy sorts `NaN` to the end, so for a query that had been scored against some targets but not others - which is what [`navis.nbl.update_scores`][navis.nbl.update_scores] and any hand-assembled matrix produce - the unscored pair was returned as that query's top match, with a `NaN` score. `NaN`s are now skipped; a query with fewer than `N` valid scores gets empty `match_k`/`score_k` for the remainder. The `percentage` criterion was worse off still: one `NaN` anywhere in a row made that row's threshold `NaN`, so the query got *no* matches at all
- **`extract_matches(..., percentage=...)` listed matches worst first for distance matrices** - the exact inverse of what it does for similarities, and of what the `N` criterion does for either. Matches are now always best first
- **[`navis.make_dotprops`][navis.make_dotprops] silently produced wrong tangent vectors for point clouds containing duplicate coordinates.** Points whose `k` nearest neighbours are *all* at distance zero are dropped, but the neighbour indices were then offset by a flat `n_dropped` - only correct if every duplicate happens to sit at the *start* of the array. Anywhere else the indices ran past the end or went negative, and because numpy reads negative indices from the back this raised nothing: it just computed each tangent from an unrelated neighbourhood. On a 40-point cloud with a 4-point duplicate block in the middle, **39 of the 40 surviving points came back with the wrong tangent**
- **[`navis.Dotprops.recalculate_tangents`][navis.Dotprops.recalculate_tangents] returned `NaN` alpha values** for points sitting on duplicate coordinates - it has no equivalent of `make_dotprops`' duplicate check and cannot drop points. Those `NaN`s then propagated into every NBLAST score the neuron took part in. Such points now get `alpha=0` and an arbitrary unit vector, matching navis-fastcore
- **[`navis.sholl_analysis`][navis.sholl_analysis] was broken for most of its `center` options - including the default.** The `"centermass"` branch rebound `center` from a preset name to an x/y/z array *before* the remaining branches compared it against `"soma"`/`"root"`, and on numpy >= 1.25 `array == "soma"` is an elementwise comparison, so the next `if` raised. Separately, a node ID given as a *numpy* integer (e.g. `center=n.root[0]`) failed the `isinstance(center, int)` check, skipped the node → coordinate lookup and was broadcast as a scalar into the distance computation - **returning wrong numbers without raising**. `center` is now resolved by type before any string comparison. Also fixed in passing: `geodesic=True, center="soma"` raised `IndexError` and `radii` did not accept numpy integers. Note that `geodesic=True` now *requires* a center that lies on the arbor (`"root"`, `"soma"` or a node ID) and raises for the default `"centermass"`
- **a batch of [`Voxels`][navis.Voxels] bugs, most of them on the sparse (voxels + values) backing**, which until now was barely exercised - values and coordinates were free to drift apart. `threshold()` filtered the coordinates but not the values; `normalize()` scaled the *coordinates* instead of the values, corrupting the geometry outright; the documented `(N, 4)` constructor input silently discarded its value column; and changing `.values` did not invalidate a cached `.grid`. Also fixed: `convert_units()` resized the neuron instead of re-labelling it (125x too small for 8 nm → µm), `.volume` squared the z voxel size and dropped y, `.density` crashed on numpy 2, `copy.deepcopy()` raised a `TypeError`, `flip()` moved the neuron and mirrored connectors in the wrong space, and `.bbox` disagreed between the two backings by one voxel
- **[`Voxels`][navis.Voxels] with no filled voxels raised `ValueError: zero-size array to reduction` on `.shape`** - and hence on `.grid`, `.bbox`, `repr()` and `summary()`. Empty neurons are not exotic: an all-zero grid auto-sparsifies to nothing, and `erode()`/`thin()`/`difference()` can consume a thin neuron entirely. `.shape` now falls back to the canvas the neuron was left on, and `strip()`/`normalize()` no-op instead of raising
- **assigning `.voxels` a different number of voxels left the old `.values` in place**, so `.nnz`/`.volume` kept counting voxels that no longer existed and `.grid` raised a broadcasting error. Mismatched values are now dropped (with a warning); values that still line up row for row are kept. Latent in `1.12.0` but easy to hit now that grids auto-sparsify
- [`navis.mesh`][navis.mesh] raised `AttributeError` on the `(N, 3)` voxel arrays it documents (it tested `.ndims`, which numpy spells `.ndim`)
- **[`navis.fix_mesh`][navis.fix_mesh] raised an `AttributeError` on `trimesh >= 4.10`.** `Trimesh.remove_duplicate_faces`/`.remove_degenerate_faces` were replaced by `.unique_faces()`/`.nondegenerate_faces()` in trimesh 3.23 and finally removed in 4.10; `fix_mesh` now picks the right pair based on the installed version. This also unbroke [`Mesh.validate()`][navis.Mesh.validate] and `Mesh(..., validate=True)`, which route through it
- **`Mesh(..., validate=True)` silently did nothing when `process=False`**: it fixed a *copy* of the mesh and threw it away
- **the plotting backends now share the code that resolves connectors, connector colors, somata and `shade_by`** (new internal `navis/plotting/_common.py`) instead of keeping a copy each - ~240 lines of near-identical code across matplotlib, plotly and k3d, and, as usual, the copies had drifted:

    - **`connectors=<numpy array of types>` raised `ValueError: truth value of an array ... is ambiguous` in every backend**, even though all three explicitly listed `np.ndarray` among the accepted types. The array was being fed to `bool()` by the guard in front of the filtering
    - **plotly gave somata the wrong color when a per-node colormap was in play** (`color_by=...`). It indexed the color list built for the *line* trace - which is ordered by segment and padded with a black sentinel per segment - with a node-table index
    - **a partial `cn_colors` dict (e.g. `{"pre": "red"}`) broke `plot2d(..., method="3d")`**: the types the dict didn't cover kept their default rgb tuple, and matplotlib can't build an array from a mixed string/tuple sequence
    - **`norm_global=False` raised `TypeError: 'bool' object is not iterable`** for both `color_by` and `shade_by`, and plotly never forwarded `norm_global` to `shade_by` at all

    Two inconsistencies are resolved by construction rather than fixed: k3d ignored `cn_mesh_colors` entirely and checked a `cn_colors` dict *before* `"neuron"`, so it disagreed with the other two on which wins; and the cap on runaway soma detection was applied three different ways (all backends now use plotly/k3d's: 10 or more somata on one neuron are taken as a detection failure and skipped, with a warning). Still outstanding: `cn_colors="neuron"` combined with `color_by=<node property>` is broken in all three backends, because there is no single "neuron color" to give the connectors in that case.

- **[`navis.plot2d`][] and [`navis.plot3d`][] documented several defaults they don't actually use**: `plot2d` advertised `linewidth=.5` (really `1`), `alpha=1` (really `None`), `connectors=True` (really `False`) and `method='3d'` (really `'2d'`); `plot3d` advertised `connectors=True` (really `False`) and `fig_autosize=False` (really `True`). The defaults are now checked against the `Settings` dataclass fields in the test suite, so they can't drift again
- **[`navis.plot3d`][navis.plot3d] raised for *any* skeleton plotted with `connectors=True` on the `plotly` backend.** {{ navis }} passed `opacity` into plotly's `scatter3d.Line`, which has no such property. Plotly's default connector layout is `display="lines"`, so the only way past it was `cn_layout={"display": "circles"}` or plotting a `Mesh`. Connector transparency now sits on the trace, and an alpha channel on an explicit `cn_colors` is honoured too instead of being emitted as a malformed `rgb(r,g,b,a)` string
- **`cn_colors` was broken in [`navis.plot2d`][navis.plot2d]** for two of its three documented forms: `"neuron"` was tested against `cn_layout` (a dict, so never equal) and reached matplotlib as a literal color name, and a `{type: color}` dict replaced each whole per-type layout entry rather than just its color
- **`color_by=<neuron property>` worked only in the `matplotlib` backend.** `plotly` and `k3d` skipped the "is this a neuron property or a per-node property?" resolution and went straight to the per-node path, so e.g. `plot3d(nl, color_by="cell_type", palette="viridis")` raised `ValueError: Column "cell_type" does not exist`. All backends now share one implementation, which also makes `color_by` require a `palette` consistently everywhere
- **`radius="auto"` was decided once for the whole [`NeuronList`][navis.NeuronList] instead of per neuron.** The auto-detection wrote its verdict back onto the shared settings object, so as soon as one neuron had too few radii every *subsequent* neuron was forced onto lines regardless of its own radii
- [`navis.plot_flat`][navis.plot_flat]: `normalize_distance=True` raised `KeyError` (it scaled a key that does not exist) and `connectors=True` combined with `highlight_connectors=[...]` raised `TypeError` (the connector angles shadowed the per-node angle lookup). Each worked on its own
- [`navis.plot2d`][navis.plot2d] with `depth_coloring=True` raised when there were no neurons to normalize against - e.g. when plotting only a [`Volume`][navis.Volume]
- [`navis.plot3d`][navis.plot3d] with `backend="octarine"` and an explicit `viewer=` raised `AttributeError` unless a {{ navis }}-created viewer already existed in the session
- [`navis.plot3d`][navis.plot3d]'s `backend` is now case-insensitive: a capitalised but otherwise valid backend (e.g. `"Plotly"`) passed validation and then fell through to "unknown backend"
- `TransformSequence` was registered as invertible even if it contained a transform that was not
- **`neuprint` interface: [`fetch_skeletons`][navis.interfaces.neuprint.fetch_skeletons] shared one client - and hence one `requests.Session` - across all its worker threads**, which `requests` does not support. `neuprint-python` normally hands each thread its own deepcopy of the client, but passing an explicit client into a thread bypasses that; {{ navis }} now does the copying itself in the pool's initializer
- `navis.interfaces.brain_image_library`: a failing *root* directory listing now raises instead of quietly returning an empty file table, and partial listing failures further down warn that the result is incomplete rather than passing it off as the whole dataset
- [`navis.conversion.mesh2skeleton`][navis.conversion.mesh2skeleton] ignored the `progress` keyword argument it accepts

## Version `1.12.0` { data-toc-label="1.12.0" }
_Date: 13/06/26_

##### Breaking
- [`mirror_brain`][navis.mirror_brain] now defaults to `mirror_axis="auto"`, i.e. takes the mirror axis from the template brain's meta data (falling back to `x`). This can change results for templates whose mirror axis is not `x`
- `TPStransform.matrix_rigid` (added in 1.11.0) was renamed to `.matrix_affine`
- {{ navis }}' internal graph algorithms no longer fall back to `networkx` - consequently `navis.config.use_igraph` is gone. `Skeleton.graph`, [`neuron2nx`][navis.neuron2nx] & co. are unaffected
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
- [`geodesic_matrix`][navis.geodesic_matrix] now uses `navis-fastcore` for `Meshes` too: ~19-68x faster and ~30-60x less memory
- [`longest_neurite`][navis.longest_neurite] with `from_root=False` no longer builds a leafs-by-leafs distance matrix just to take its maximum: 29x faster and 125x less memory (722ms/785MB -> 25ms/6.3MB for a 71k node skeleton)
- [`distal_to`][navis.distal_to] now uses `navis-fastcore` if available: 13x faster and 5x less memory (it previously asked igraph for a source-by-target block, which igraph answers by running an all-sources search)
- [`arbor_segregation_index`][navis.arbor_segregation_index], [`bending_flow`][navis.bending_flow], [`synapse_flow_centrality`][navis.synapse_flow_centrality], [`flow_centrality`][navis.flow_centrality], [`longest_neurite`][navis.longest_neurite] and `node_label_sorting` now request only the geodesic distances they actually use (see `to_` above)
- major speed-up for [`heal_skeleton`][navis.heal_skeleton] and [`stitch_skeletons`][navis.stitch_skeletons]: they now use `navis-fastcore` if available (2.5-340x faster on real, fragmented skeletons, e.g. 85s -> 0.25s for a 640k node skeleton), and the built-in fallback was rewritten around a vectorized Borůvka algorithm (5-15x faster and ~7x less memory, e.g. 8s/640MB -> 0.5s/90MB for a 220k node skeleton with 5k fragments). Results are unchanged: both produce the same minimum spanning tree
- major speed-up for [`resample_skeleton`][navis.resample_skeleton]: ~15-20x faster with the default `method="linear"` (e.g. 100ms -> 7ms for the example neuron; 425ms -> 64ms when densifying it to 132k nodes, as [`xform_brain`][navis.xform_brain] does) by interpolating all segments and columns in one go instead of fitting one `scipy.interpolate.interp1d` per column *per segment*. Non-linear methods (e.g. `"cubic"`) can't share that trick but still gain ~4x. It also no longer builds a KDTree and an indexed copy of the node table when the neuron has no soma, connectors or tags to re-map
- [`reroot_skeleton`][navis.reroot_skeleton] builds a node ID -> vertex index map once instead of scanning all vertices for each root: much faster on heavily fragmented neurons
- [`split_axon_dendrite`][navis.split_axon_dendrite] no longer runs out of memory on very large (100k+ nodes) neurons - the assignment of orphan nodes used to build a full orphans-by-all-nodes geodesic matrix - and is faster (igraph instead of `networkx`)
- [`drop_fluff`][navis.drop_fluff], [`fix_mesh`][navis.fix_mesh] and everything else built on connected components are faster: `navis-fastcore` is now used for `Meshes` too, igraph otherwise
- [`skeletonize`][navis.skeletonize] with `shave=True`: fixing up the vertex map was an O(n_bristles x n_vertices) Python loop and is now a single vectorized map - a major bottleneck on large meshes
- [`rewire_skeleton`][navis.rewire_skeleton] now skips the minimum spanning tree if the graph is already a forest (i.e. has no cycles)
- [`H5transform`][navis.transforms.H5transform] and [`GridTransform`][navis.transforms.GridTransform] use `scipy.ndimage.map_coordinates` (~2x faster), and copies of an `H5transform` now carry over the cache - previously [`xform_brain`][navis.xform_brain] copied the transform and hence never benefitted from caching
- [`skeletonize`][navis.skeletonize] for point clouds/[`Dotprops`][navis.Dotprops] uses `scipy`'s minimum spanning tree instead of `networkx` and now correctly handles duplicate points
- `betweenness_centrality`, [`plot_flat`][navis.plot_flat] and [`segment_analysis`][navis.segment_analysis] are faster
- reading from URLs with the default `parallel="auto"` now goes parallel from 5 files onwards instead of 200. The 200 was tuned for the process pool used to read local files; URLs are read in a *thread* pool and are network- rather than CPU-bound. Reading e.g. 100 neurons off a remote server no longer means 100 sequential blocking requests
- URL reads now share a single `requests.Session`, so connections to the same host are pooled and kept alive
- `read_*` functions can now read from Google buckets (`gs://...`) without `gcsfs` installed

##### Fixes
- neurons read from a list of URLs **in parallel** came back stripped of their identity: the parallel reader handed the downloaded *bytes* (instead of the URL) to the parser, so the filename was never parsed. Affected neurons had no `file`, an `origin` of `"string"`, a `name` of `"SWC"`/`"MESH"`/... and a random `id`, and any `fmt` was silently ignored - i.e. the same input produced different neurons depending on `parallel`. [`read_mesh`][navis.read_mesh] failed outright (`ReadError`) since it needs the filename to determine the file type
- [`read_swc`][navis.read_swc] & co. no longer choke on URLs with a query string (e.g. `.../neuron.swc?token=123`, whose file extension was previously parsed as `swc?token=123`) and now decode percent-encoded filenames (`%20` -> a space)
- [`flow_centrality`][navis.flow_centrality], [`synapse_flow_centrality`][navis.synapse_flow_centrality] and [`arbor_segregation_index`][navis.arbor_segregation_index] returned wrong values for **fragmented neurons**: all three work out how many leafs/synapses are *proximal* to a node as `total - distal`, which is only valid on a single-rooted neuron. Nodes in another fragment are neither distal nor proximal but were silently counted as proximal, inflating the flow. Totals are now counted within each node's own fragment. **This changes the output for fragmented neurons** (they were previously wrong); single-rooted neurons are unaffected. Note that [`synapse_flow_centrality`][navis.synapse_flow_centrality] was only affected without `navis-fastcore`, so the two backends used to disagree; `bending_flow` was never affected
- `Skeleton.small_segments` returned the segments in a different **order** depending on whether `navis-fastcore` was installed (without it, {{ navis }} walked a Python `set`, i.e. in arbitrary hash order). They are now always ordered by the node table position of their seed node, which is what `navis-fastcore` already did. Several functions `enumerate()` the segments, so the order was ending up in their output - most visibly the **row order of [`segment_analysis`][navis.segment_analysis]**
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
- two new [`Voxels`][navis.Voxels] methods:
    - [`flip()`][navis.Voxels.flip] flips the neuron along specified axes
    - [`normalize()`][navis.Voxels.normalize] scales values to a 0-1 range

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
    - new function: [`navis.thin_voxels`][] can be used to thin images and `Voxels` to single-pixel width (see also below)
    - new `thin` parameter for [`read_nrrd`][navis.read_nrrd], [`read_tiff`][navis.read_tiff]

- [`Skeletons`][navis.Skeleton]:
    - skeletons can now be initialized from a `(vertices, edges)` tuple - see also [`navis.edges2neuron`][]
    - new property: `Skeleton.vertices` gives read-only to node (vertex) coordinates

- [`Voxels`][navis.Voxels]:
    - new properties: `Voxels.nnz` and `Voxels.density`

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
- `Tree/Meshes` and `Dotprops` now support addition/subtraction (similar to the already existing multiplication and division) to allow offsetting neurons

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
- [`navis.Skeleton.simple`][] now correctly drops soma nodes if they aren't root, branch or leaf points themselves

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
- `navis.Skeleton.cable_length` is now faster
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
- New property `Skeleton.surface_area`
- New (experimental) functions `navis.read_parquet` and `navis.write_parquet` store skeletons and dotprops in parquet files (see [here](https://github.com/navis-org/navis/blob/master/navis/io/pq_io.md) for format specs)
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
- New function: `navis.betweenness_centrality`
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
- `navis.Mesh`: `__getattr__` does not search `trimesh` representation anymore
- NBLASTs: queries/targets now MUST be `navis.Dotprops` (no more automatic conversion, use `navis.make_dotprops`)
- Renamed functions to make it clear they work only on `Skeletons`:
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
- `navis.Voxels`:
    - new class representing neurons as voxels
    - new (experimental) class representing neurons as voxels
    - `navis.read_nrrd` now returns `Voxels` instead of `Dotprops` by default
    - currently works with only a selection of functions

- `navis.Skeleton`:
    - can now be initialized directly with `skeletor.Skeleton`
    - new method: `navis.Skeleton.snap`

- `navis.Mesh`:
    - `navis.in_volume`, `navis.subset_neuron` and `navis.break_fragments` now work on `Meshes`
    - new properties: `.skeleton`, `.graph` and `.igraph`
    - new methods: `navis.Mesh.skeletonize` and `navis.Mesh.snap`
    - can now be initialized with `skeletor.Skeleton` and `(vertices, faces)` tuple
    - plotting: `color_by` parameter now works with `Meshes`

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

- Most functions that work with `Skeletons` now also work with `Meshes`
- New high-level wrappers to convert neurons: `navis.voxelize`, `navis.mesh` and `navis.skeletonize`
- `navis.make_dotprops` now accepts `parallel=True` parameter for parallel processing
- `navis.smooth_skeleton` can now be used to smooth arbitrary numeric columns in the node table
- New function `navis.drop_fluff` removes small disconnected bits and pieces from neurons
- New function `navis.patch_cloudvolume` monkey-patches `cloudvolume` (see the new tutorial)
- New function `navis.write_nrrd` writes `Voxels` to NRRD files
- New functions to read/write `Meshes`: `navis.read_mesh` and `navis.write_mesh`
- New function `navis.read_nmx` reads pyKNOSSOS files
- New function `navis.smooth_mesh` smoothes meshes and `Meshes`
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
- `navis.Skeleton` now has a `soma_pos` property that can also be used to set the soma by position
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
- New `plot3d` parameter: with plotly backend, use `radius=True` plots Skeletons with radius
- New `plot2d` parameter: `orthogonal=False` sets view to perspective

##### Improvements
- Various improvements to e.g. `nx2neuron`

## Version `0.2.0` { data-toc-label="0.2.0" }
_Date: 29/06/20_

##### Breaking
- `navis.nx2neuron` now returns a `navis.Skeleton` instead of a `DataFrame`

##### Additions
- New neuron class `navis.Mesh`
- New `navis.Skeleton` property `.volume`
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
- `Skeletons`: allow rerooting by setting the `.root` attribute

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

