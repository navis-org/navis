"""
IVSCC morphometrics on EM data
==============================
<!-- difficulty: advanced -->

Turn MICrONS reconstructions into something [`navis.ivscc_features`][] can measure.

!!! important "This example is not executed"
    Like the [IVSCC tutorial](zzz_tutorial_morpho_05_ivscc), this one is *not* run when the docs are
    built - it queries live CAVE servers and needs an authentication token. The code is real and
    runnable; the outputs and plots shown here are statically embedded.

The [IVSCC tutorial](zzz_tutorial_morpho_05_ivscc) waves EM off in a sentence: the features need a
neuron labelled by compartment, "which rules out most EM reconstructions". True as far as it goes - but
it is worth asking *what exactly* is missing, because the answer is short and every item on it is
fixable.

| IVSCC needs | MICrONS gives you | What to do |
|-------------|-------------------|------------|
| a skeleton with radii | a mesh, or a precomputed skeleton | ask the skeleton service |
| the soma at the root | nucleus centroid, soma collapsed to one node | comes for free |
| axon vs dendrite | a `compartment` label - `1` soma, `2` axon, `3` dendrite | check it, and redo it from synapses when it is wrong |
| **apical vs basal dendrite** | nothing | **this tutorial** |
| **`y` = cortical depth** | volume coordinates, column tilted ~5° | one affine and [`navis.xform`][] |
| microns | nanometres (the SWC endpoint hands you microns) | `units=` |
| no spines | mesh skeletons have thousands of them | [`navis.prune_twigs`][] |

Only two of those are real work. Here is the whole pipeline:

```mermaid
graph LR
    A[proofread<br>root IDs] --> B[skeleton<br>service];
    B --> C{QC};
    C -->|axon label missing| D[split from<br>synapses];
    C --> E[upright<br>navis.xform];
    D --> E;
    E --> F[apical vs basal<br>the stem that climbs];
    F --> G[navis.ivscc_features];
```

!!! note "Requirements"
    ```shell
    pip install caveclient cloud-volume -U
    ```
    You will also need a CAVE token - see the [MICrONS tutorial](../4_remote/tutorial_remote_02_microns)
    for how to get one.
"""

# %%
import navis
import numpy as np
import pandas as pd

import navis.interfaces.microns as mi

client = mi.get_cave_client("cortex65")

# %%
# ## Picking cells
#
# Two annotation tables carry everything we need. `proofreading_status_and_strategy` says whose axon and
# dendrite have been manually cleaned; `aibs_metamodel_mtypes_v661_v2` gives each cell an m-type, and in
# this dataset the m-type names *are* layer names (`L2a`, `L4c`, `L6tall-b`, ...). That makes them a
# convenient ground truth to check our work against later.

proof = client.materialize.query_table("proofreading_status_and_strategy")
mtypes = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")

clean = proof[proof.status_dendrite & proof.status_axon]
mtype = mtypes.drop_duplicates("pt_root_id").set_index("pt_root_id").cell_type

cells = clean[["pt_root_id"]].copy()
cells["m_type"] = cells.pt_root_id.map(mtype)
cells = cells.dropna(subset=["m_type"])
# excitatory types only - `L5NP` and the inhibitory classes have no apical dendrite
cells = cells[cells.m_type.str.match(r"L\d[abc]?(tall|short|ET)?-?[abc]?$")]

# two cells per m-type, picked deterministically
cells = cells.sort_values("pt_root_id").groupby("m_type", group_keys=False).head(2)

print(len(cells), "cells,", cells.m_type.nunique(), "m-types")

# %%
# ```
# 32 cells, 16 m-types
# ```
#
# ## Skeletons in one call
#
# CAVE runs a skeleton service that will hand you a [`pcg_skel`](https://github.com/AllenInstitute/pcg_skel)
# skeleton straight out of the level-2 chunk graph. [`navis.patch_caveclient`][] teaches the client to
# return {{ navis }} neurons from it, the same way [`navis.patch_cloudvolume`][] does for meshes:

navis.patch_caveclient()

nl = navis.NeuronList([
    client.skeleton.get_skeleton_navis(int(r), output_format="swc")
    for r in cells.pt_root_id
])

# `name` is the one thing the service doesn't know about
for n in nl:
    n.name = mtype[n.id]

nl[:3]

# %%
# ```
# <class 'navis.core.neuronlist.NeuronList'> containing 3 neurons (835.2KiB)
#              type name                  id  n_nodes  n_branches  n_leafs  cable_length  soma         units
# 0  navis.Skeleton  L3a  864691134884807418     7167         100      111  14856.277344     0  1 micrometer
# 1  navis.Skeleton  L4a  864691134886499066     4762          50       64   9638.884766     0  1 micrometer
# 2  navis.Skeleton  L4b  864691134886569210     3065          29       38   6263.025391     0  1 micrometer
# ```
#
# The patch adds a `*_navis` twin to each of the service's three fetch methods, plus an `as_navis=True`
# keyword on the originals - `client.skeleton.get_skeleton(root_id, as_navis=True)` does the same thing.
# Without it you would get the raw payload and rename it yourself:
#
# ```python
# swc = client.skeleton.get_skeleton(root_id, output_format="swc")
# n = navis.Skeleton(
#     swc.rename(columns={"id": "node_id", "parent": "parent_id", "type": "label"}),
#     id=root_id, units="um",
# )
# ```
#
# !!! tip "Two output formats, two units"
#     MICrONS is in nanometres, but `output_format="swc"` divides by 1,000 on the way out while the
#     default `"dict"` format does not. The patch sets `.units` to match either way, which matters
#     because IVSCC features are reported in whatever units go in.
#
# ??? warning "Why not `get_bulk_skeletons`?"
#     Because it caps how many skeletons it returns - ten, at the time of writing - and drops the rest
#     without raising. Ask it for our 32 cells and you get 10 back, which as a `NeuronList` looks exactly
#     like a complete result. {{ navis }} warns when that happens:
#
#     ```
#     WARNING : Got 10 skeletons for 32 root IDs. The bulk endpoints return a limited number per
#     call - request fewer at a time, or use `generate_bulk_skeletons_async` for large sets.
#     ```
#
#     Thirty-two sequential calls take about a minute, so that is what we do here. For hundreds of cells,
#     use `client.skeleton.generate_bulk_skeletons_async` to have them built server-side first.
#
# Two things came along for the ride. `soma` is `0` - the service collapses the nucleus into a single
# node and roots the skeleton there, which is exactly what [`ivscc_features`][navis.ivscc_features]
# wants. And there is a `radius` column, so the diameter/surface/volume features will work:

nl[0].nodes.head()

# %%
# ```
#    node_id  label         x        y       z  radius  parent_id  type
# 0        0      1  1196.352  492.736  919.56   6.146         -1  root
# 1        1      3  1198.216  502.864  916.08   0.515          0  slab
# 2        2      3  1197.488  503.384  916.04   0.515          1  slab
# 3        3      3  1196.160  504.560  915.96   0.515          2  slab
# 4        4      3  1196.160  506.016  916.44   0.515          3  slab
# ```
#
# The `type` column the service returns is the SWC structure identifier, so renaming it to `label` puts
# it exactly where [`ivscc_features`][navis.ivscc_features] looks for it - but only three of the four
# values are ever used:

nl[0].nodes.label.value_counts().sort_index()

# %%
# ```
# label
# 1       1
# 2    4161
# 3    3005
# ```
#
# `1` soma, `2` axon, `3` dendrite. There is no `4`. **That is the whole problem**: an IVSCC feature
# table built on this would have an empty `apical_dendrite_*` block and a `basal_dendrite_*` block that
# is secretly the entire dendrite.
#
# ??? info "What if I only have the mesh?"
#     [`mi.fetch_neurons`][navis.interfaces.microns.fetch_neurons] gives you meshes, and
#     [`navis.skeletonize`][] will turn one into a skeleton. It works, but you inherit the mesh's
#     problems - here is the same cell both ways:
#
#     | Source | Nodes | Cable | Tips | Roots |
#     |--------|-------|-------|------|-------|
#     | mesh + [`navis.skeletonize`][] | 166,812 | 22.91 mm | 6,046 | 925 |
#     | `mesh, prune_twigs(5 µm)` | 116,302 | 15.20 mm | 1,032 | 925 |
#     | CAVE skeleton service | 6,625 | 13.58 mm | 79 | 1 |
#
#     Almost all of those 6,046 tips are **spines**, and they carry nearly 8 mm of phantom cable -
#     [`navis.prune_twigs`][] removes most of it. The 925 roots are mesh fragments and want
#     [`navis.heal_skeleton`][]. The service skeleton has neither problem because it is built from the
#     level-2 chunk graph with a 7.5 µm invalidation radius, which is coarser than a spine.
#
#     The radii, meanwhile, agree well - matching nodes within 3 µm, the median radius is 5.690 vs
#     5.664 µm at the soma, 0.291 vs 0.309 µm on dendrite and 0.117 vs 0.148 µm on axon. Whichever route
#     you take, the radius features are measuring the same thing.
#
# ## QC: trust, then verify
#
# Before doing anything clever, check the assumptions {{ navis }} is about to make:

qc = pd.DataFrame({
    "m_type": nl.name,
    "one_tree": [(n.nodes.parent_id < 0).sum() == 1 for n in nl],
    "has_soma": [n.soma is not None for n in nl],
    "has_axon": [(n.nodes.label == 2).any() for n in nl],
    "has_dendrite": [(n.nodes.label == 3).any() for n in nl],
}, index=nl.id)

qc.drop(columns="m_type").all()

# %%
# ```
# one_tree         True
# has_soma         True
# has_axon        False
# has_dendrite     True
# ```

qc[~qc.has_axon]

# %%
# ```
#                    m_type  one_tree  has_soma  has_axon  has_dendrite
# 864691134886828794    L2a      True      True     False          True
# ```
#
# One cell out of 32 has no axon label at all - its axon is sitting in the dendrite pile, where it would
# quietly wreck every `basal_dendrite_*` feature. {{ navis }} can redo that split from the synapses,
# which MICrONS has plenty of:

n = nl.idx[864691134886828794]

pre = client.materialize.synapse_query(pre_ids=int(n.id))
post = client.materialize.synapse_query(post_ids=int(n.id))
print(f"{len(pre)} outputs, {len(post)} inputs")

VOXEL = np.array([4, 4, 40]) / 1_000          # MICrONS voxels -> microns
xyz = np.vstack([np.vstack(pre.ctr_pt_position.values),
                 np.vstack(post.ctr_pt_position.values)]) * VOXEL

n.connectors = pd.DataFrame({
    "connector_id": np.concatenate([pre.id.values, post.id.values]),
    "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
    "type": ["pre"] * len(pre) + ["post"] * len(post),
    "node_id": n.nodes.node_id.values[n.snap(xyz)[0]],   # snap each synapse to its closest node
})

split = navis.split_axon_dendrite(n, label_only=True)
n.nodes.loc[split.nodes.compartment.values == "axon", "label"] = 2

n.nodes.label.value_counts().sort_index()

# %%
# ```
# 64 outputs, 6244 inputs
#
# label
# 1       1
# 2    1317
# 3    2159
# ```
#
# [`navis.split_axon_dendrite`][] found 2.7 mm of axon sitting on average 134 µm *below* the soma, and
# left 4.4 mm of dendrite 31 µm above it. See the
# [axon-dendrite tutorial](tutorial_morpho_03_ad_split) for what the flow metrics are doing and when to
# distrust them.
#
# ## Getting upright
#
# !!! danger "`y` must be cortical depth"
#     Half the IVSCC features are statements about depth and read it off `y`, with larger `y` meaning
#     closer to the pia. MICrONS coordinates satisfy neither half of that: `y` grows *downwards*, and the
#     cortical column is tilted roughly 5° relative to the volume's axes.
#
# The correction is a rotation and a shift, which is to say one 4×4 matrix. The numbers below come from
# [`standard_transform`](https://github.com/AllenInstitute/standard_transform), the package the MICrONS
# team publishes for exactly this. `185°` rather than `5°` because we want the extra 180° that flips `y`
# into the IVSCC convention - and doing it as a rotation rather than a sign flip keeps the coordinate
# system right-handed.

from scipy.spatial.transform import Rotation
from navis.transforms import AffineTransform

M = np.eye(4)
M[:3, :3] = Rotation.from_euler("z", 185, degrees=True).as_matrix()
M[1, 3] = 396.671                     # the pia, in microns above the volume origin

print(M.round(4))

# %%
# ```
# [[-9.96200e-01  8.72000e-02  0.00000e+00  0.00000e+00]
#  [-8.72000e-02 -9.96200e-01  0.00000e+00  3.96671e+02]
#  [ 0.00000e+00  0.00000e+00  1.00000e+00  0.00000e+00]
#  [ 0.00000e+00  0.00000e+00  0.00000e+00  1.00000e+00]]
# ```

to_pia = AffineTransform(M)
nl = navis.xform(nl, to_pia)

# %%
#
# [`navis.xform`][] takes the whole `NeuronList` in one go and moves connectors along with nodes. It also
# tries to work out whether the transform changed the spatial scale, and rescales `radius` and `units` if
# it did - here it correctly decides nothing changed.
#
# Now check it, because a wrong frame is silent. The `L4` somas form a band across the volume; if the
# tilt is really gone, that band should be flat:

l4 = mtypes.drop_duplicates("pt_root_id")
l4 = np.vstack(l4[l4.cell_type.str.match("L4")].pt_position.values) * VOXEL
l4_up = to_pia.xform(l4)


def tilt(x, depth):
    """Microns of depth gained per millimetre travelled along x."""
    return np.polyfit(x, depth, 1)[0] * 1000


print("L4 soma band, µm of depth per mm of x")
print("  volume coordinates : %+.0f" % tilt(l4[:, 0], l4[:, 1]))
print("  after `to_pia`     : %+.0f" % tilt(l4_up[:, 0], -l4_up[:, 1]))

# %%
# ```
# L4 soma band, µm of depth per mm of x
#   volume coordinates : -87
#   after `to_pia`     : +1
# ```
#
# 87 µm of drift per millimetre - roughly half a cortical layer across the width of the dataset - down to
# nothing. The same thing, drawn:

# %%
# ![raw vs upright](../../../_static/ivscc_em_tut/02_frame.png)
#
# And the m-types now sort by depth exactly as their names promise:

depth = pd.Series({n.id: -n.soma_pos[0][1] for n in nl})
chk = pd.DataFrame({"m_type": pd.Series(dict(zip(nl.id, nl.name))), "soma_depth": depth.round(0)})
chk.groupby("m_type").soma_depth.mean().sort_values().round(0)

# %%
# ```
# m_type
# L2b          100.0
# L2a          124.0
# L2c          148.0
# L3a          182.0
# L3b          258.0
# L4a          302.0
# L4b          321.0
# L4c          365.0
# L5a          376.0
# L5ET         446.0
# L5b          516.0
# L6tall-a     543.0
# L6short-a    546.0
# L6short-b    559.0
# L6tall-b     642.0
# L6tall-c     651.0
# ```
#
# ## Apical vs basal
#
# This is the one thing nobody hands you, and the reason it has to happen *after* the frame is fixed:
# the apical dendrite is defined by where it goes. It is the stem that climbs towards the pia.
#
# So: cut the dendrite into stems, and score each by how far it rises above the soma.
#
# One wrinkle first. `pcg_skel` collapses the soma using a 7.5 µm sphere, which is smaller than a real
# cortical soma - so a couple of basal dendrites usually end up hanging off the *base of the apical
# trunk* rather than off the soma node, and travel with it. Widening that sphere separates them:

SOMA_R = 15          # microns - generous enough to swallow the perisomatic branch points
MIN_CLIMB = 0.4      # an apical has to climb at least this fraction of the way to the pia


def dendritic_stems(n):
    """Dendritic stems, treating everything within `SOMA_R` of the soma as soma."""
    d = np.linalg.norm(n.nodes[["x", "y", "z"]].values - n.soma_pos[0], axis=1)
    dendrite = navis.subset_neuron(n, (d > SOMA_R) & (n.nodes.label.values == 3))
    return navis.split_components(dendrite)


def apical_stem(n):
    """The dendritic stem that climbs towards the pia, and how far it climbs."""
    soma_y = n.soma_pos[0][1]
    stems = dendritic_stems(n)
    climb = np.array([s.nodes.y.max() - soma_y for s in stems])
    return stems[climb.argmax()], climb.max() / -soma_y


climb = pd.Series({n.id: apical_stem(n)[1] for n in nl})
climb.sort_values().round(2).head(6)

# %%
# ```
# 864691134990435194    0.12
# 864691135114295961    0.15
# 864691135014681590    0.42
# 864691134949041148    0.70
# 864691134886499066    0.84
# 864691134966710559    0.85
# ```
#
# Two cells apart, every score lands between 0.42 and 1.06, and 28 of the 30 are above 0.8 - whatever
# layer the soma sits in, the apical climbs most of the way to the pia. The gap between 0.15 and 0.42 is
# where `MIN_CLIMB` goes.
#
# Writing the label is then a matter of relabelling the winning stem, plus the short piece of trunk
# between it and the soma that `SOMA_R` cut away:

parents = {n.id: n.nodes.set_index("node_id").parent_id for n in nl}

for n in nl:
    stem, frac = apical_stem(n)
    if frac < MIN_CLIMB:
        continue                       # no apical - leave the whole dendrite as basal
    ids, node = list(stem.nodes.node_id), int(stem.root[0])
    while node != n.soma:              # walk back to the soma so the trunk base comes along
        node = int(parents[n.id][node])
        ids.append(node)
    n.nodes.loc[n.nodes.node_id.isin(ids) & (n.nodes.label == 3), "label"] = 4

print("cells with an apical:", int((climb >= MIN_CLIMB).sum()), "/", len(nl))
chk.assign(climb=climb.round(2))[climb < MIN_CLIMB]

# %%
# ```
# cells with an apical: 30 / 32
#
#                        m_type  soma_depth  climb
# 864691134990435194  L6short-a       579.0   0.12
# 864691135114295961        L5b       528.0   0.15
# ```
#
# ![apical](../../../_static/ivscc_em_tut/03_apical.png)
#
# One of the two rejects is an `L6short-a` - a type whose name is a description of its stubby apical, so
# declining to call one is arguably right. The other is an `L5b` whose apical was cut during
# reconstruction. Either way they get no `apical_dendrite_*` features rather than wrong ones, which is
# the outcome you want.
#
# !!! warning "This is a heuristic"
#     It leans on cortical geometry and it will mislabel things. Obliques leaving the trunk sideways
#     within `SOMA_R` of the soma get called basal; a basal dendrite that happens to arc upwards past
#     `MIN_CLIMB × depth` would get called apical. Look at the pictures before you trust a batch, and
#     turn both knobs if your data disagrees - a deeper dataset tolerates a higher `MIN_CLIMB`, a
#     coarser skeleton wants a bigger `SOMA_R`.
#
# ## Measuring
#
# That was the entire preparation. From here it is the same one-liner as for Patch-seq data:

feats = navis.ivscc_features(nl)
feats.shape

# %%
# ```
# (32, 102)
# ```

feats.loc[864691135082864887,
          ["apical_dendrite_total_length", "apical_dendrite_num_tips",
           "apical_dendrite_max_branch_order", "apical_dendrite_extent_y",
           "apical_dendrite_bias_y", "apical_dendrite_soma_percentile_y",
           "basal_dendrite_total_length", "basal_dendrite_extent_y",
           "axon_total_length", "num_stems", "soma_radius"]].round(2)

# %%
# ```
# apical_dendrite_total_length          1449.76
# apical_dendrite_num_tips                16.00
# apical_dendrite_max_branch_order         7.00
# apical_dendrite_extent_y               348.16
# apical_dendrite_bias_y                 314.34
# apical_dendrite_soma_percentile_y        0.92
# basal_dendrite_total_length           2775.21
# basal_dendrite_extent_y                144.77
# axon_total_length                    11892.34
# num_stems                               10.00
# soma_radius                              6.16
# ```
#
# `soma_percentile_y` of 0.92 says 92% of that apical sits above the soma - the sanity check that the
# frame and the labels are both right. And all 32 cells, drawn at their true cortical depths because
# that is now a thing the coordinates mean:

# %%
# ![the column](../../../_static/ivscc_em_tut/01_column.png)
#
# ## Depth trends
#
# The [IVSCC tutorial](zzz_tutorial_morpho_05_ivscc) correlated a few features against soma depth on 25
# Patch-seq cells. Same test here:

for col in ["apical_dendrite_extent_y", "apical_dendrite_bias_y",
            "apical_dendrite_max_path_length", "basal_dendrite_extent_y",
            "basal_dendrite_num_tips", "basal_dendrite_total_length"]:
    print(f"{col:<34} r = {feats[col].corr(depth):+.2f}")

# %%
# ```
# apical_dendrite_extent_y           r = +0.91
# apical_dendrite_bias_y             r = +0.93
# apical_dendrite_max_path_length    r = +0.87
# basal_dendrite_extent_y            r = +0.19
# basal_dendrite_num_tips            r = -0.70
# basal_dendrite_total_length        r = -0.49
# ```
#
# ![depth trends](../../../_static/ivscc_em_tut/04_depth.png)
#
# The apical story is the same one the Patch-seq cells told, only sharper - `extent_y` goes from
# `r = +0.70` there to `+0.91` here. That is what you would hope for: these reconstructions are dense and
# complete rather than filled from a slice, and the alignment is a measured property of the volume rather
# than a per-slice estimate.
#
# The basal dendrites, flat in the Patch-seq set, now show something: deeper cells have *fewer* basal
# tips and less basal cable (`r = -0.70`, `-0.49`) while spanning the same vertical extent. Worth a
# larger n before believing it - but it is the kind of thing that only becomes visible once the
# compartments are separated.
#
# ## Caveats
#
# !!! warning "EM axons are cut off too - just differently"
#     Patch-seq axons leave a 300 µm slice. MICrONS axons leave a 1376 × 869 × 517 µm volume. Measuring
#     how many axon tips end within 30 µm of a volume wall: a median of **18%**, and **28 of the 32**
#     cells are affected. Axon cable here runs from 0.8 mm to 15.8 mm, and most of that spread is
#     geometry - where the soma sits relative to the walls - rather than biology. Dendrite features are
#     fine; `axon_total_length` is not comparable across cells.
#
# Two more things to keep in mind:
#
# - **Branch and tip counts are not comparable across modalities.** The service builds skeletons with a
#   7.5 µm invalidation radius, so short branches vanish along with the spines: our cells average **13**
#   apical tips against 31 for the Patch-seq set. Skeletonising the mesh instead swings it the other way
#   - ~1,000 tips for the cell that the service gives 79. Neither number is a tracing. Topology features
#   compare EM cells to each other, not to light microscopy.
# - **The pia is a plane here, and it is not.** Two cells scored `climb > 1`, meaning their tufts pass
#   above `y = 0`. The single offset in `to_pia` is a good average, not a surface fit. For layer-resolved
#   work use [`standard_transform`](https://github.com/AllenInstitute/standard_transform)'s streamline
#   version, which follows the curvature.
#
# ## Where next
#
# - [IVSCC morphometrics](zzz_tutorial_morpho_05_ivscc) - what the 102 features actually are, and how to
#   add your own
# - [Axon-dendrite splits](tutorial_morpho_03_ad_split) and
#   [label propagation](tutorial_morpho_02_label_prop) - other routes to compartment labels
# - [The MICrONS datasets](../4_remote/tutorial_remote_02_microns) - the interface used here
# - [Skeleton QC](zzz_tutorial_morpho_04_qc) - checking reconstructions before you measure them

# %%

# *[IVSCC]: In Vitro Single Cell Characterization - the Allen Institute's pipeline for profiling cortical neurons.
# *[EM]: Electron Microscopy.
# *[SWC]: A plain-text format storing a neuron skeleton as a table of connected nodes.
# *[CAVE]: Connectome Annotation Versioning Engine - the annotation and query backend for MICrONS.
# *[Patch-seq]: Recording a cell electrophysiologically, then extracting its contents for sequencing and staining it for morphology.
# *[QC]: Quality control.

# mkdocs_gallery_thumbnail_path = '_static/ivscc_em_tut/00_thumbnail.png'
