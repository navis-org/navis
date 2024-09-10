"""Interface with the CAJAL (https://github.com/CamaraLab/CAJAL) library."""

import os

import pandas as pd

from cajal import swc, run_gw, sample_swc
from scipy.spatial.distance import squareform

from .. import config


def navis2cajal(neurons, progress=True):
    """Convert navis TreeNeurons to CAJAL forests.

    Parameters
    ----------
    neurons :   list
                List of TreeNeurons to convert.
    progress :  bool
                Show progress bar.

    Returns
    -------
    forests
                List of CAJAL forests.
    """
    forests = []
    for n in config.tqdm(
        neurons, disable=not progress or config.pbar_hide, desc="Converting"
    ):
        nodes = {
            row.node_id: swc.NeuronNode(
                sample_number=row.node_id,
                structure_id=row.get("label", 0),
                coord_triple=(row.x, row.y, row.z),
                radius=row.radius,
                parent_sample_number=row.parent_id,
            )
            for i, row in n.nodes.iterrows()
        }

        components, tree_index = swc.topological_sort(nodes)

        components = sorted(components, key=swc.num_nodes, reverse=True)

        forests.append(components)

    return forests


def compute_gw_distance_matrix(neurons, n_sample, distance="euclidean", num_processes=max(1, os.cpu_count() // 2), progress=True):
    """Compute the matrix of pairwise Gromov-Wasserstein distances between cells.

    Parameters
    ----------
    neurons :   NeuronLost
                List of neurons to compute distances for.
    n_sample :  int
                Number of ~ evenly distributed samples to use from each neuron for the distance computation.
    distance :  'geodesic' | 'euclidean'
                Distance metric to use for the distance computation. See
                [here](https://cajal.readthedocs.io/en/latest/computing-intracell-distance-matrices.html)
                for a detailed explanation of the differences.
    num_processes : int
                Number of processes to use for parallel computation. Defaults to half the number of available CPU cores.
    progress :  bool
                Show progress bar.

    Returns
    -------
    matrix
                Matrix of pairwise distances.

    """
    if len(neurons) < 2:
        raise ValueError("At least two neurons are required for distance computation.")

    if distance not in ("euclidean", "geodesic"):
        raise ValueError(f"Unknown distance metric: {distance}")

    forests = navis2cajal(neurons)

    if distance == "euclidean":
        icdm = [
            squareform(sample_swc.icdm_euclidean(f, n_sample))
            for f in config.tqdm(
                forests, disable=not progress or config.pbar_hide, desc="Euclidean Distances"
            )
        ]
    elif distance == "geodesic":
        if any(len(f) > 1 for f in forests):
            raise ValueError(
                "Geodesic distances are only supported for single-component neurons."
            )

        forests = [f[0] for f in forests]
        icdm = [
            squareform(sample_swc.icdm_geodesic(f, n_sample))
            for f in config.tqdm(
                forests, disable=not progress or config.pbar_hide, desc="Geodesic Distances"
            )
        ]
    else:
        raise ValueError(f"Unknown distance metric: {distance}")


    cell_dms = [(i, run_gw.uniform(i.shape[0])) for i in icdm]

    dists, _ = run_gw.gw_pairwise_parallel(
        cell_dms,
        num_processes
    )

    return pd.DataFrame(dists, index=neurons.id, columns=neurons.id)
