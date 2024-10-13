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
from scipy.stats import wasserstein_distance
from typing import Union, Sequence

from .. import config, graph, core
from . import subset_neuron, tortuosity

# Set up logging
logger = config.get_logger(__name__)

__all__ = sorted(
    [
        "ivscc",
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
    """An exception raised when a compartment is not found."""

    pass


class Features(ABC):
    def __init__(self, neuron: "core.TreeNeuron", label=None, verbose=False):
        self.neuron = neuron
        self.verbose = verbose

        if label is None:
            self.label = ""
        elif not label.endswith("_"):
            self.label = f"{label}_"
        else:
            self.label = label

        # Make sure the neuron is rooted to the soma (if present)
        self.soma = self.neuron.soma
        if self.soma is not None:
            self.soma_pos = self.neuron.soma_pos[0]
            self.soma_radius = self.neuron.nodes.set_index("node_id").loc[
                self.soma, "radius"
            ]

            if self.neuron.soma not in self.neuron.root:
                self.neuron = self.neuron.reroot(self.neuron.soma)

            # Calculate geodesic distances from leafs to all other nodes (directed)
            self.leaf_dists = graph.geodesic_matrix(
                self.neuron, self.neuron.leafs.node_id.values, directed=True
            )
            # Replace infinities with -1
            self.leaf_dists[self.leaf_dists == float("inf")] = -1

        self.features = {}

    def record_feature(self, name, value):
        """Record a feature."""
        self.features[f"{self.label}{name}"] = value

    @abstractmethod
    def extract_features(self):
        """Extract features."""
        pass


class BasicFeatures(Features):
    """Base class for features."""

    def extract_features(self):
        """Extract basic features."""
        self.record_feature(
            "extent_y", self.neuron.nodes.y.max() - self.neuron.nodes.y.min()
        )
        self.record_feature(
            "extent_x", self.neuron.nodes.x.max() - self.neuron.nodes.x.min()
        )
        self.record_feature(
            "max_branch_order", (self.neuron.nodes.type == "branch").sum() + 1
        )
        self.record_feature("num_nodes", len(self.neuron.nodes))
        self.record_feature("total_length", self.neuron.cable_length)

        if self.soma is None:
            if self.verbose:
                logger.warning(
                    f"{self.neuron.id} has no `.soma` attribute, skipping soma-related features."
                )
            return

        # x/y bias from soma
        # Note: this is absolute for x and relative for y
        self.record_feature(
            "bias_x",
            abs(
                (self.neuron.nodes.x.max() - self.soma_pos[0])
                - (self.soma_pos[0] - self.neuron.nodes.x.min())
            ),
        )
        self.record_feature(
            "bias_y",
            (self.neuron.nodes.y.max() - self.soma_pos[1])
            - (self.soma_pos[1] - self.neuron.nodes.y.min()),
        )

        # Distances from soma
        self.record_feature(
            "max_euclidean_distance",
            (
                (self.neuron.nodes[["x", "y", "z"]] - self.soma_pos)
                .pow(2)
                .sum(axis=1)
                .pow(0.5)
                .sum()
                .max()
            ),
        )
        self.record_feature(
            "max_path_length",
            self.leaf_dists.loc[
                self.leaf_dists.index.isin(self.neuron.nodes.node_id)
            ].values.max(),
        )

        # Tortuosity
        self.record_feature("mean_contraction", tortuosity(self.neuron))

        # Branching (number of linear segments between branch)
        self.record_feature("num_branches", len(self.neuron.small_segments))

        return self.features


class CompartmentFeatures(BasicFeatures):
    """Base class for compartment-specific features."""

    def __init__(self, neuron: "core.TreeNeuron", compartment, verbose=False):
        if "label" not in neuron.nodes.columns:
            raise ValueError(
                f"No 'label' column found in node table for neuron {neuron.id}"
            )

        if (
            compartment not in neuron.nodes.label.values
            and comp_to_label.get(compartment, compartment)
            not in neuron.nodes.label.values
        ):
            raise CompartmentNotFoundError(
                f"No {compartment} ({comp_to_label.get(compartment, compartment)}) compartments found in neuron {neuron.id}"
            )

        # Initialize the parent class
        super().__init__(neuron, label=compartment, verbose=verbose)

        # Now subset the neuron to this compartment
        self.neuron = subset_neuron(
            self.neuron,
            (
                self.neuron.nodes.label.isin(
                    (compartment, comp_to_label[compartment])
                ).values
            ),
        )


class AxonFeatures(CompartmentFeatures):
    """Extract features from an axon."""

    def __init__(self, neuron: "core.TreeNeuron", verbose=False):
        super().__init__(neuron, "axon", verbose=verbose)

    def extract_features(self):
        # Extract basic features via the parent class
        super().extract_features()

        # Now deal witha axon-specific features:

        if self.soma is not None:
            # Distance between axon root and soma surface
            # Note: we're catering for potentially multiple roots here
            axon_root_pos = self.neuron.nodes.loc[
                self.neuron.nodes.type == "root", ["x", "y", "z"]
            ].values

            # Closest dist between an axon root and the soma
            dist = np.linalg.norm(axon_root_pos - self.soma_pos, axis=1).min()

            # Subtract soma radius from the distance
            dist -= self.soma_radius

            self.record_feature("exit_distance", dist)

            # Axon theta: The relative radial position of the point where the neurite from which
            # the axon derives exits the soma.

            # Get the node where the axon exits the soma
            exit_node = self.neuron.nodes.loc[self.neuron.nodes.type == "root"]

            # Get theta
            theta = np.arctan2(
                exit_node.y.values - self.soma_pos[1],
                exit_node.x.values - self.soma_pos[0],
            )[0]
            self.record_feature("exit_theta", theta)

        return self.features


class BasalDendriteFeatures(CompartmentFeatures):
    """Extract features from a basal dendrite."""

    def __init__(self, neuron: "core.TreeNeuron", verbose=False):
        super().__init__(neuron, "basal_dendrite", verbose=verbose)

    def extract_features(self):
        # Extract basic features via the parent class
        super().extract_features()

        # Now deal with basal dendrite-specific features
        if self.soma is not None:
            # Number of stems sprouting from the soma
            # (i.e. number of nodes with a parent that is the soma)
            self.record_feature(
                "calculate_number_of_stems",
                (self.neuron.nodes.parent_id == self.soma).sum(),
            )

        return self.features


class ApicalDendriteFeatures(CompartmentFeatures):
    """Extract features from a apical dendrite."""

    def __init__(self, neuron: "core.TreeNeuron", verbose=False):
        super().__init__(neuron, "apical_dendrite", verbose=verbose)

    def extract_features(self):
        # Extract basic features via the parent class
        super().extract_features()

        return self.features


class OverlapFeatures(Features):
    """Features that compare two compartments (e.g. overlap)."""

    # Compartments to compare
    compartments = ("axon", "basal_dendrite", "apical_dendrite")

    def extract_features(self):
        # Iterate over compartments
        for c1 in self.compartments:
            if c1 in self.neuron.nodes.label.values:
                c1_nodes = self.neuron.nodes[self.neuron.nodes.label == c1]
            elif comp_to_label.get(c1, c1) in self.neuron.nodes.label.values:
                c1_nodes = self.neuron.nodes[
                    self.neuron.nodes.label == comp_to_label[c1]
                ]
            else:
                continue
            for c2 in self.compartments:
                if c1 == c2:
                    continue
                if c2 in self.neuron.nodes.label.values:
                    c2_nodes = self.neuron.nodes[self.neuron.nodes.label == c2]
                elif comp_to_label.get(c2, c2) in self.neuron.nodes.label.values:
                    c2_nodes = self.neuron.nodes[
                        self.neuron.nodes.label == comp_to_label[c2]
                    ]
                else:
                    continue

                # Calculate % of nodes of a given compartment type above/overlapping/below the
                # full y-extent of another compartment type
                self.features[f"{c1}_frac_above_{c2}"] = (
                    c1_nodes.y > c2_nodes.y.max()
                ).sum() / len(c1_nodes)
                self.features[f"{c1}_frac_intersect_{c2}"] = (
                    (c1_nodes.y >= c2_nodes.y.min()) & (c1_nodes.y <= c2_nodes.y.max())
                ).sum() / len(c1_nodes)
                self.features[f"{c1}_frac_below_{c2}"] = (
                    c1_nodes.y < c2_nodes.y.min()
                ).sum() / len(c1_nodes)

                # Calculate earth mover's distance (EMD) between the two compartments
                if f"{c2}_emd_with_{c1}" not in self.features:
                    self.features[f"{c1}_emd_with_{c2}"] = wasserstein_distance(
                        c1_nodes.y, c2_nodes.y
                    )

        return self.features


def ivscc_features(
    x: "core.TreeNeuron",
    features=None,
    missing_compartments="ignore",
    verbose=False,
    progress=True,
) -> Union[float, pd.DataFrame]:
    """Calculate IVSCC features for neuron(s).

    Please see the `IVSCC` tutorial for details.

    Parameters
    ----------
    x :                     TreeNeuron | NeuronList
                            Neuron(s) to calculate IVSCC for.
    features :              Sequence[Features], optional
                            Provide specific features to calculate.
                            Must be subclasses of `BasicFeatures`.
                            If `None`, will use default features.
    missing_compartments : "ignore" | "skip" | "raise"
                            What to do if a neuron is missing a compartment
                            (e.g. no axon or basal dendrite):
                             - "ignore" (default): ignore that compartment
                             - "skip": skip the entire neuron
                             - "raise": raise an exception

    Returns
    -------
    ivscc :                 pd.DataFrame
                            IVSCC features for the neuron(s).

    """

    if isinstance(x, core.TreeNeuron):
        x = core.NeuronList([x])

    if features is None:
        features = DEFAULT_FEATURES

    data = {}
    for n in config.tqdm(
        x, desc="Calculating IVSCC features", disable=not progress or config.pbar_hide
    ):
        data[n.id] = {}
        for feat in features:
            try:
                f = feat(n, verbose=verbose)
            except CompartmentNotFoundError as e:
                if missing_compartments == "ignore":
                    continue
                elif missing_compartments == "skip":
                    if verbose:
                        print(f"Skipping neuron {n.id}: {e}")
                    data.pop(n.id)
                    break
                else:
                    raise e

            data[n.id].update(f.extract_features())

    return pd.DataFrame(data)


def _check_compartments(n, compartments):
    """Check if `compartments` are valid."""
    if compartments == "auto":
        if "label" not in n.nodes.columns:
            return None
        return n.nodes.label.unique()
    elif compartments is True:
        return n.nodes.label.unique()
    elif isinstance(compartments, str):
        if "label" not in n.nodes.columns or compartments not in n.nodes.label.unique():
            raise ValueError(f"Compartment not present: {compartments}")
        return [compartments]
    elif isinstance(compartments, Sequence):
        if "label" not in n.nodes.columns:
            raise ValueError("No 'label' column found in node table.")
        for c in compartments:
            if c not in n.nodes.label.unique():
                raise ValueError(f"Compartment not present: {c}")
        return compartments
    elif compartments in (None, False):
        return None

    raise ValueError(f"Invalid `compartments`: {compartments}")


DEFAULT_FEATURES = [
    AxonFeatures,
    BasalDendriteFeatures,
    ApicalDendriteFeatures,
    OverlapFeatures,
]
