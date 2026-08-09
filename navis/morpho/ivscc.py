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

import pandas as pd
import numpy as np


from abc import ABC, abstractmethod
from itertools import combinations, permutations
from scipy.stats import wasserstein_distance

from .. import config, core, utils
from .angles import _angle_between
from .subset import subset_neuron

# Set up logging
logger = config.get_logger(__name__)

__all__ = sorted(
    [
        "ivscc_features",
    ]
)

# A mapping of label IDs to compartment names
# Note: anything above 5 is considered "undefined" or "custom"
label_to_comp = {
    -1: "root",
    0: "undefined",
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
}
comp_to_label = {v: k for k, v in label_to_comp.items()}


class CompartmentNotFoundError(Exception):
    """An exception raised when a compartment is not found.

    This is also raised when the neuron has no `label` column at all - i.e.
    when *no* compartment can be found. Either way it is
    [`navis.ivscc_features`][]'s `missing_compartments` parameter that decides
    what happens next.
    """

    pass


def _compartment_mask(nodes, compartment):
    """Boolean mask of the nodes belonging to `compartment`.

    Labels may be either the compartment name (e.g. `"axon"`) or the
    corresponding SWC label ID (e.g. `2`) - we accept both.
    """
    return nodes.label.isin(
        (compartment, comp_to_label.get(compartment, compartment))
    ).values


def _mean(values):
    """Mean across finite values.

    Returns `NaN` (rather than warning) if there is nothing to average.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def _parent_dist(neuron):
    """Distance from each node to its parent (`0` for roots)."""
    nodes = neuron.nodes
    return utils.fastcore.dag.parent_dist(
        nodes.node_id.values,
        nodes.parent_id.values,
        nodes[["x", "y", "z"]].values,
        root_dist=0,
    )


def _node_pos(neuron):
    """Map node IDs to their positional index in the node table."""
    return pd.Series(
        np.arange(len(neuron.nodes)), index=neuron.nodes.node_id.values
    )


def _branch_nodes(neuron, exclude=None):
    """Boolean mask of the nodes with 2+ children.

    Unlike the node table's `type` column this also flags *roots* with multiple
    children. That matters for compartments: their root is where we cut the
    neuron, not a soma, so a stem that splits straight away is a genuine
    bifurcation. `exclude` (i.e. the soma) is never counted - the angles between
    the neurites leaving the soma are a different quantity, see
    [`navis.soma_exit_angles`][].
    """
    nodes = neuron.nodes
    parent_ids = nodes.parent_id.values
    has_parent = parent_ids >= 0

    n_childs = np.bincount(
        _node_pos(neuron).loc[parent_ids[has_parent]].values, minlength=len(nodes)
    )
    mask = n_childs >= 2

    if exclude is not None:
        mask &= nodes.node_id.values != exclude

    return mask


def _segment_metrics(neuron, pdist):
    """Geodesic length `L` and end-to-end distance `R` for each small segment.

    `.small_segments` come back ordered distal -> proximal, i.e. the parent of
    `seg[i]` is `seg[i + 1]`. The geodesic length is therefore the sum of the
    parent distances of all but the last (proximal) node.
    """
    segments = neuron.small_segments
    if not len(segments):
        return np.zeros(0), np.zeros(0)

    pos = _node_pos(neuron)
    coords = neuron.nodes[["x", "y", "z"]].values

    # Number of edges contributed by each segment. Single-node segments (which
    # can occur for isolated fragments) contribute no cable at all.
    counts = np.array([len(s) - 1 for s in segments])
    flat = np.concatenate([np.asarray(s)[:-1] for s in segments])
    if len(flat):
        L = np.bincount(
            np.repeat(np.arange(len(segments)), counts),
            weights=pdist[pos.loc[flat].values].astype(np.float64),
            minlength=len(segments),
        )
    else:
        L = np.zeros(len(segments))

    starts = coords[pos.loc[[s[0] for s in segments]].values]
    ends = coords[pos.loc[[s[-1] for s in segments]].values]
    R = np.linalg.norm(starts - ends, axis=1)

    return L, R


def _bifurcation_angles(neuron, exclude=None, degrees=True):
    """Local and remote bifurcation angles.

    At each branch point we measure the angle between its child branches, once
    using the immediate child edges ("local") and once using the vectors
    pointing at the next branch point/tip along each child ("remote"). Branch
    points with 3+ children contribute all pairwise angles. `exclude` (i.e. the
    soma) is skipped - see `_branch_nodes`.

    Returns
    -------
    local, remote : (N, ) arrays
    """
    # Group segments by their proximal end - a node with 2+ segments hanging
    # off it is a branch point
    by_prox = {}
    for s in neuron.small_segments:
        if len(s) < 2:
            continue
        by_prox.setdefault(s[-1], []).append(s)

    # Collect (branch point, near/far end of branch A, near/far end of branch B)
    # for every pair of branches, then look up all coordinates in one go
    pairs = []
    for prox, segs in by_prox.items():
        if len(segs) < 2 or prox == exclude:
            continue
        for i, j in combinations(range(len(segs)), 2):
            pairs.append(
                (prox, segs[i][-2], segs[i][0], segs[j][-2], segs[j][0])
            )

    if not pairs:
        return np.zeros(0), np.zeros(0)

    pairs = np.asarray(pairs)
    coords = neuron.nodes.set_index("node_id")[["x", "y", "z"]]
    origin = coords.reindex(pairs[:, 0]).values

    local = _angle_between(
        coords.reindex(pairs[:, 1]).values - origin,
        coords.reindex(pairs[:, 3]).values - origin,
    )
    remote = _angle_between(
        coords.reindex(pairs[:, 2]).values - origin,
        coords.reindex(pairs[:, 4]).values - origin,
    )

    if degrees:
        local, remote = np.degrees(local), np.degrees(remote)

    return local, remote


def _branch_orders(neuron, branches):
    """Branch order for every node - `1` at the root, `+1` at each branch point."""
    nodes = neuron.nodes
    node_ids = nodes.node_id.values
    parent_ids = nodes.parent_id.values

    # Weight each child -> parent edge by whether the *parent* is a branch
    # point. The distance to the root is then simply the number of branch
    # points passed on the way there.
    weights = np.zeros(len(node_ids), dtype=np.float64)
    has_parent = parent_ids >= 0
    weights[has_parent] = branches[_node_pos(neuron).loc[parent_ids[has_parent]].values]

    orders = utils.fastcore.dag.dist_to_root(node_ids, parent_ids, weights=weights)

    return pd.Series(orders + 1, index=node_ids)


def _surface_and_volume(neuron, pdist):
    """Lateral surface area and volume of the cable, modelled as tapered cylinders."""
    nodes = neuron.nodes
    not_root = nodes.parent_id.values >= 0
    if not not_root.any():
        return 0.0, 0.0

    radii = nodes.set_index("node_id").radius
    r1 = radii.reindex(nodes.node_id.values[not_root]).values.astype(np.float64)
    r2 = radii.reindex(nodes.parent_id.values[not_root]).values.astype(np.float64)
    h = pdist[not_root].astype(np.float64)

    surface = (np.pi * (r1 + r2) * np.sqrt((r1 - r2) ** 2 + h**2)).sum()
    volume = (np.pi / 3 * (r1**2 + r1 * r2 + r2**2) * h).sum()

    return float(surface), float(volume)


class NeuronContext:
    """Per-neuron state shared between feature extractors.

    Rerooting the neuron and measuring the distance from every node to the soma
    are the expensive parts of the IVSCC pipeline. Building this context once
    per neuron - instead of once per feature class - keeps them to a single pass.

    Attributes
    ----------
    neuron :        Skeleton
                    The neuron, rerooted to its soma (if it has one).
    soma :          int | str | None
                    Node ID of the soma.
    soma_pos :      (3, ) array | None
    soma_radius :   float
                    `NaN` if the neuron has no soma or no radii.

    """

    def __init__(self, neuron: "core.Skeleton", verbose: bool = False):
        self.verbose = verbose

        soma = neuron.soma
        # `.soma` may return an iterable if the neuron has multiple soma nodes
        if utils.is_iterable(soma):
            soma = soma[0] if len(soma) else None
        self.soma = soma

        # Make sure the neuron is rooted to the soma (if present)
        if soma is not None and soma not in neuron.root:
            neuron = neuron.reroot(soma)
        self.neuron = neuron

        self._dist_to_root = None
        self._has_radii = None

        if soma is None:
            self.soma_pos = None
            self.soma_radius = np.nan
        else:
            is_soma = neuron.nodes.node_id.values == soma
            self.soma_pos = neuron.nodes.loc[is_soma, ["x", "y", "z"]].values[0]
            if self.has_radii:
                self.soma_radius = float(neuron.nodes.radius.values[is_soma][0])
            else:
                self.soma_radius = np.nan

    @property
    def has_soma(self) -> bool:
        return self.soma is not None

    @property
    def has_labels(self) -> bool:
        return "label" in self.neuron.nodes.columns

    @property
    def has_radii(self) -> bool:
        """Whether the neuron has usable radii.

        Note that navis fills in a `radius` column of zeros if the node table
        doesn't have one, so we have to check for actual values.
        """
        if self._has_radii is None:
            nodes = self.neuron.nodes
            self._has_radii = "radius" in nodes.columns and bool(
                (nodes.radius.fillna(0) > 0).any()
            )
        return self._has_radii

    @property
    def dist_to_root(self) -> pd.Series:
        """Geodesic distance from each node to the root (i.e. the soma)."""
        if self._dist_to_root is None:
            nodes = self.neuron.nodes
            node_ids = nodes.node_id.values
            dists = utils.fastcore.dag.dist_to_root(
                node_ids,
                nodes.parent_id.values,
                weights=_parent_dist(self.neuron),
            )
            self._dist_to_root = pd.Series(dists, index=node_ids)
        return self._dist_to_root


class Features(ABC):
    """Base class for a group of IVSCC features.

    Subclasses implement `extract_features()`, use `record_feature()` to add
    individual features (which takes care of prefixing them with the
    compartment name) and return `self.features`.
    """

    def __init__(self, ctx: "NeuronContext", label=None):
        self.ctx = ctx
        self.neuron = ctx.neuron
        self.verbose = ctx.verbose

        if label is None:
            self.label = ""
        else:
            label = str(label)
            self.label = label if label.endswith("_") else f"{label}_"

        self.features = {}

    # Convenience proxies into the shared context
    @property
    def soma(self):
        return self.ctx.soma

    @property
    def soma_pos(self):
        return self.ctx.soma_pos

    @property
    def soma_radius(self):
        return self.ctx.soma_radius

    def record_feature(self, name, value):
        """Record a feature."""
        self.features[f"{self.label}{name}"] = value

    def _warn(self, msg):
        if self.verbose:
            logger.warning(f"Neuron {self.neuron.id}: {msg}")

    @abstractmethod
    def extract_features(self) -> dict:
        """Extract features. Implementations must return `self.features`."""
        pass


class BasicFeatures(Features):
    """Features that can be extracted from any neuron or compartment thereof."""

    def extract_features(self):
        """Extract basic features."""
        nodes = self.neuron.nodes
        coords = nodes[["x", "y", "z"]].values
        pdist = _parent_dist(self.neuron)

        # Branch points: note that for a compartment the subset's root can be a
        # branch point too, which the node table's `type` column would miss
        branches = _branch_nodes(self.neuron, exclude=self.soma)
        branch_ids = nodes.node_id.values[branches]

        # Size
        self.record_feature("num_nodes", len(nodes))
        self.record_feature("total_length", float(pdist.astype(np.float64).sum()))
        for axis, extent in zip("xyz", coords.max(axis=0) - coords.min(axis=0)):
            self.record_feature(f"extent_{axis}", float(extent))

        # Topology
        self.record_feature("num_branches", len(self.neuron.small_segments))
        self.record_feature("num_branch_points", int(branches.sum()))
        self.record_feature("num_tips", int((nodes.type == "end").sum()))
        self.record_feature(
            "max_branch_order", int(_branch_orders(self.neuron, branches).max())
        )

        # Shape: contraction is the ratio of the end-to-end distance of a
        # segment to its geodesic length, i.e. the inverse of tortuosity. The
        # ratio can't exceed 1 - clip to absorb float32 rounding in `pdist`.
        L, R = _segment_metrics(self.neuron, pdist)
        with np.errstate(invalid="ignore", divide="ignore"):
            contraction = np.where(L > 0, np.minimum(R / L, 1), np.nan)
        self.record_feature("mean_contraction", _mean(contraction))

        local, remote = _bifurcation_angles(self.neuron, exclude=self.soma)
        self.record_feature("bifurcation_angle_local", _mean(local))
        self.record_feature("bifurcation_angle_remote", _mean(remote))

        # Radius-derived features
        if self.ctx.has_radii:
            self.record_feature("mean_diameter", _mean(nodes.radius.values) * 2)
            surface, volume = _surface_and_volume(self.neuron, pdist)
            self.record_feature("total_surface", surface)
            self.record_feature("total_volume", volume)
            self.record_feature(
                "parent_daughter_ratio", self._parent_daughter_ratio(branch_ids)
            )
        else:
            self._warn("no usable radii, skipping radius-based features.")

        if not self.ctx.has_soma:
            self._warn("no `.soma` attribute, skipping soma-related features.")
            return self.features

        # x/y bias from soma: how lopsided the arbor is around the soma.
        # Note: this is absolute for x and relative for y
        self.record_feature(
            "bias_x",
            float(
                abs(
                    (nodes.x.max() - self.soma_pos[0])
                    - (self.soma_pos[0] - nodes.x.min())
                )
            ),
        )
        self.record_feature(
            "bias_y",
            float(
                (nodes.y.max() - self.soma_pos[1])
                - (self.soma_pos[1] - nodes.y.min())
            ),
        )

        # Fraction of nodes above the soma
        self.record_feature(
            "soma_percentile_x", float((nodes.x.values > self.soma_pos[0]).mean())
        )
        self.record_feature(
            "soma_percentile_y", float((nodes.y.values > self.soma_pos[1]).mean())
        )

        # Distances from soma
        self.record_feature(
            "max_euclidean_distance",
            float(np.linalg.norm(coords - self.soma_pos, axis=1).max()),
        )
        # Note: distances to root are measured on the full neuron, so for a
        # compartment this is the path length from the soma - not the length of
        # the path within the compartment
        max_path_length = float(
            self.ctx.dist_to_root.reindex(nodes.node_id.values).max()
        )
        self.record_feature("max_path_length", max_path_length)

        # How early the first branch point occurs, relative to the arbor's reach
        self.record_feature(
            "early_branch_path", self._early_branch_path(branch_ids, max_path_length)
        )

        return self.features

    def _parent_daughter_ratio(self, branch_ids):
        """Mean ratio of daughter to parent radius across branch points."""
        nodes = self.neuron.nodes
        daughters = nodes[nodes.parent_id.isin(branch_ids)]

        if daughters.empty:
            return np.nan

        radii = nodes.set_index("node_id").radius
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = (
                radii.reindex(daughters.node_id.values).values
                / radii.reindex(daughters.parent_id.values).values
            )
        return _mean(ratio)

    def _early_branch_path(self, branch_ids, max_path_length):
        """Path length to the first branch point over the maximum path length."""
        if not len(branch_ids) or not max_path_length > 0:
            return np.nan

        first = float(self.ctx.dist_to_root.reindex(branch_ids).min())
        return first / max_path_length


class CompartmentFeatures(BasicFeatures):
    """Features for a single compartment (axon, basal dendrite, etc.).

    The compartment is extracted as a subset of the neuron but we keep a handle
    on the full neuron (via `self.ctx`) because some features are only defined
    relative to the rest of the cell.
    """

    # The compartment to extract - set by the subclasses below
    compartment = None

    def __init__(self, ctx: "NeuronContext", compartment=None):
        compartment = compartment if compartment is not None else self.compartment
        if compartment is None:
            raise ValueError(
                "`CompartmentFeatures` requires a compartment - either pass one "
                "or subclass and set the `compartment` class attribute."
            )
        # Accept SWC label IDs (e.g. `2`) as well as names (e.g. "axon") - we
        # always report under the name
        compartment = label_to_comp.get(compartment, compartment)

        if not ctx.has_labels:
            raise CompartmentNotFoundError(
                f"No 'label' column in node table for neuron {ctx.neuron.id} - "
                "can't tell compartments apart."
            )

        mask = _compartment_mask(ctx.neuron.nodes, compartment)
        if not mask.any():
            raise CompartmentNotFoundError(
                f"No {compartment} ({comp_to_label.get(compartment, compartment)}) "
                f"compartments found in neuron {ctx.neuron.id}"
            )

        # Initialize the parent class
        super().__init__(ctx, label=compartment)

        self.compartment = compartment
        # Mask into the *full* node table - kept for the soma-relative features
        self.mask = mask
        # Now subset the neuron to this compartment. Note that node IDs survive
        # this, so we can still index into the context's full-neuron measures.
        self.neuron = subset_neuron(ctx.neuron, mask)

    def extract_features(self):
        # Extract basic features via the parent class
        super().extract_features()

        if not self.ctx.has_soma:
            return self.features

        # Number of stems, i.e. neurites of this compartment sprouting from the
        # soma. This has to be counted on the *full* neuron: in the subset the
        # soma is gone and the stems have been rewired into roots.
        parents = self.ctx.neuron.nodes.parent_id.values
        self.record_feature("num_stems", int((parents[self.mask] == self.soma).sum()))

        # Where this compartment leaves the soma. A compartment can have several
        # roots (multiple stems, or fragmentation), so we use the one closest to
        # the soma and measure both features at that same node.
        roots = self.neuron.nodes[self.neuron.nodes.type == "root"]
        dists = np.linalg.norm(roots[["x", "y", "z"]].values - self.soma_pos, axis=1)
        closest = int(np.argmin(dists))
        exit_node = roots.iloc[closest]

        # Distance between the compartment's root and the soma *surface*. If we
        # don't know the soma radius we measure from its centre instead.
        radius = self.soma_radius if np.isfinite(self.soma_radius) else 0
        self.record_feature("exit_distance", float(max(dists[closest] - radius, 0)))

        # Exit theta: the radial position at which the compartment leaves the soma
        self.record_feature(
            "exit_theta",
            float(
                np.arctan2(
                    exit_node.y - self.soma_pos[1], exit_node.x - self.soma_pos[0]
                )
            ),
        )

        return self.features


class AxonFeatures(CompartmentFeatures):
    """Extract features from an axon."""

    compartment = "axon"


class BasalDendriteFeatures(CompartmentFeatures):
    """Extract features from a basal dendrite."""

    compartment = "basal_dendrite"


class ApicalDendriteFeatures(CompartmentFeatures):
    """Extract features from an apical dendrite."""

    compartment = "apical_dendrite"


class SomaFeatures(Features):
    """Extract whole-cell features centred on the soma."""

    def extract_features(self):
        if not self.ctx.has_soma:
            self._warn("no `.soma` attribute, skipping soma features.")
            return self.features

        self.record_feature("soma_radius", float(self.soma_radius))
        self.record_feature("soma_surface", float(4 * np.pi * self.soma_radius**2))

        # Number of neurites leaving the soma (across all compartments)
        parents = self.ctx.neuron.nodes.parent_id.values
        self.record_feature("num_stems", int((parents == self.soma).sum()))

        return self.features


class OverlapFeatures(Features):
    """Features that compare two compartments (e.g. overlap)."""

    # Compartments to compare
    compartments = ("axon", "basal_dendrite", "apical_dendrite")

    def extract_features(self):
        if not self.ctx.has_labels:
            self._warn("no 'label' column, skipping overlap features.")
            return self.features

        nodes = self.ctx.neuron.nodes

        # Depth (y) of the nodes of each compartment that is actually present
        depth = {}
        for c in self.compartments:
            mask = _compartment_mask(nodes, c)
            if mask.any():
                depth[c] = nodes.y.values[mask]

        for c1, c2 in permutations(depth, 2):
            y1, y2 = depth[c1], depth[c2]
            lo, hi = y2.min(), y2.max()

            # Calculate % of nodes of a given compartment type above/overlapping/below the
            # full y-extent of another compartment type
            self.features[f"{c1}_frac_above_{c2}"] = float((y1 > hi).mean())
            self.features[f"{c1}_frac_intersect_{c2}"] = float(
                ((y1 >= lo) & (y1 <= hi)).mean()
            )
            self.features[f"{c1}_frac_below_{c2}"] = float((y1 < lo).mean())

            # Calculate earth mover's distance (EMD) between the two compartments.
            # This is symmetric, so we only record it once per pair.
            if f"{c2}_emd_with_{c1}" not in self.features:
                self.features[f"{c1}_emd_with_{c2}"] = float(
                    wasserstein_distance(y1, y2)
                )

        return self.features


DEFAULT_FEATURES = [
    SomaFeatures,
    AxonFeatures,
    BasalDendriteFeatures,
    ApicalDendriteFeatures,
    OverlapFeatures,
]


def ivscc_features(
    x: "core.NeuronObject",
    features=None,
    missing_compartments: str = "ignore",
    verbose: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Calculate IVSCC features for neuron(s).

    IVSCC features describe cortical neurons in terms of their compartments
    (axon, basal and apical dendrite) and how those compartments relate to the
    soma and to each other. Compartments are read off the `label` column of the
    node table, which accepts either SWC label IDs (`2` = axon, `3` = basal
    dendrite, `4` = apical dendrite) or their names.

    !!! important "Coordinate frame"
        Features are computed in the neuron's native coordinate system and
        units - nothing is normalized or rescaled. Crucially, the depth-related
        features (`bias_y`, `soma_percentile_y` and everything produced by
        `OverlapFeatures`) assume that neurons have been aligned such that **`y`
        is the cortical depth axis** (i.e. perpendicular to the pia) and that
        `y` increases in the same direction for all neurons. Feed in unaligned
        neurons and those features will be meaningless. See
        [`navis.xform`][] / [`navis.Skeleton.reroot`][] for getting there.

    Parameters
    ----------
    x :                     Skeleton | NeuronList
                            Neuron(s) to calculate IVSCC features for.
    features :              Sequence[Features], optional
                            Provide specific features to calculate. Must be
                            subclasses of `Features` (see `DEFAULT_FEATURES`
                            in `navis.morpho.ivscc`). If `None`, will use the
                            default features.
    missing_compartments : "ignore" | "skip" | "raise"
                            What to do if a neuron is missing a compartment
                            (e.g. no axon or basal dendrite) or has no `label`
                            column at all:
                             - "ignore" (default): ignore that compartment. Its
                               features are simply not recorded, which for a
                               `NeuronList` means `NaN` in the columns other
                               neurons contributed
                             - "skip": skip the entire neuron
                             - "raise": raise a `CompartmentNotFoundError`
    verbose :               bool
                            If True, will log a warning whenever a feature is
                            skipped (e.g. because a neuron has no soma).
    progress :              bool
                            Whether to show a progress bar.

    Returns
    -------
    ivscc :                 pd.DataFrame
                            IVSCC features - one row per neuron, one column per
                            feature. Features that could not be computed for a
                            given neuron are `NaN`.

    Notes
    -----
    Features prefixed with a compartment name (e.g. `axon_total_length`) are
    computed on that compartment alone:

    | Feature                    | Description                                                                     |
    |----------------------------|---------------------------------------------------------------------------------|
    | `num_nodes`                | Number of nodes.                                                                  |
    | `total_length`             | Summed cable length.                                                              |
    | `extent_x/y/z`             | Extent along each axis.                                                           |
    | `num_branches`             | Number of linear segments between branch points, tips and roots.                  |
    | `num_branch_points`        | Number of branch points.                                                          |
    | `num_tips`                 | Number of terminal (leaf) nodes.                                                   |
    | `max_branch_order`         | Highest branch order; `1` for an unbranched neurite, `+1` at each branch point.   |
    | `mean_contraction`         | Mean ratio of end-to-end distance to geodesic length across segments.             |
    | `bifurcation_angle_local`  | Mean angle (degrees) between child branches, measured at the branch point.        |
    | `bifurcation_angle_remote` | Mean angle (degrees) between child branches, measured to the next branch/tip.     |
    | `mean_diameter`            | Mean node diameter.                                                               |
    | `total_surface`            | Lateral surface area, modelling the cable as tapered cylinders.                   |
    | `total_volume`             | Volume, modelling the cable as tapered cylinders.                                 |
    | `parent_daughter_ratio`    | Mean ratio of daughter to parent radius at branch points.                         |
    | `bias_x`                   | Asymmetry of the arbor around the soma along x (absolute).                        |
    | `bias_y`                   | Asymmetry of the arbor around the soma along y (signed).                          |
    | `soma_percentile_x/y`      | Fraction of nodes above the soma along x/y.                                       |
    | `max_euclidean_distance`   | Distance from the soma to the farthest node.                                      |
    | `max_path_length`          | Geodesic distance from the soma to the farthest node.                             |
    | `early_branch_path`        | Path length to the first branch point over `max_path_length`.                     |
    | `num_stems`                | Number of neurites of this compartment sprouting from the soma.                   |
    | `exit_distance`            | Distance between the soma surface and where the compartment leaves it.            |
    | `exit_theta`               | Radial position (radians) at which the compartment leaves the soma.               |

    Radius-based features require a `radius` column, and everything from
    `bias_x` onwards requires a soma; they are `NaN` otherwise.

    On top of those there are whole-cell features (`soma_radius`,
    `soma_surface`, `num_stems`) and, for each ordered pair of compartments,
    features describing how their depth distributions relate
    (`{c1}_frac_above_{c2}`, `_frac_intersect_`, `_frac_below_` and the
    earth mover's distance `{c1}_emd_with_{c2}`).

    Examples
    --------
    >>> import navis
    >>> import numpy as np
    >>> n = navis.example_neurons(1, kind='skeleton').copy()
    >>> # IVSCC features are computed per compartment, so the neuron needs a
    >>> # `label` column. Our example neurons don't have one, so we make it up -
    >>> # normally this would come from the SWC file.
    >>> n.nodes['label'] = np.where(n.nodes.y > n.soma_pos[0][1],
    ...                             'apical_dendrite', 'basal_dendrite')
    >>> feats = navis.ivscc_features(n, progress=False)
    >>> feats.index.tolist() == [n.id]
    True
    >>> bool(feats.apical_dendrite_max_branch_order.values[0] > 1)
    True

    """
    utils.eval_param(
        missing_compartments,
        name="missing_compartments",
        allowed_values=("ignore", "skip", "raise"),
    )
    utils.eval_param(
        x, name="x", allowed_types=(core.Skeleton, core.NeuronList)
    )

    if isinstance(x, core.Skeleton):
        x = core.NeuronList([x])

    wrong = sorted({type(n).__name__ for n in x if not isinstance(n, core.Skeleton)})
    if wrong:
        raise ValueError(f"IVSCC features require Skeletons, got: {', '.join(wrong)}")

    if features is None:
        features = DEFAULT_FEATURES

    ids, rows = [], []
    for n in config.tqdm(
        x, desc="Calculating IVSCC features", disable=not progress or config.pbar_hide
    ):
        # Everything the feature classes share about this neuron
        ctx = NeuronContext(n, verbose=verbose)

        row, skip = {}, False
        for feat in features:
            try:
                row.update(feat(ctx).extract_features())
            except CompartmentNotFoundError as e:
                if missing_compartments == "ignore":
                    if verbose:
                        logger.warning(str(e))
                    continue
                elif missing_compartments == "skip":
                    if verbose:
                        logger.warning(f"Skipping neuron {n.id}: {e}")
                    skip = True
                    break
                else:
                    raise

        if not skip:
            ids.append(n.id)
            rows.append(row)

    return pd.DataFrame(rows, index=pd.Index(ids, name="id"))
