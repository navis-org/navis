"""Interface with the CAJAL (https://github.com/CamaraLab/CAJAL) library."""

import os
import pandas as pd

from cajal import swc, run_gw, sample_swc
from scipy.spatial.distance import squareform

from .. import config, core


logger = config.logger

# Note: looks like `cajal.swc` and `cajal.sample_swc` are the culprits
logger.warning("Please note that importing the CAJAL interface will currently break multiprogressing (parallel=True) in navis.")


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
    if not isinstance(neurons, core.NeuronList):
        neurons = core.NeuronList(neurons)

    forests = []
    for n in config.tqdm(
        neurons,
        disable=not progress or config.pbar_hide or len(neurons) == 1,
        desc="Converting"
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


def compute_gw_distance_matrix(
    queries,
    targets=None,
    n_sample=100,
    distance="euclidean",
    num_processes=max(1, os.cpu_count() // 2),
    progress=True,
):
    """Compute the matrix of pairwise Gromov-Wasserstein distances between neurons.

    Parameters
    ----------
    queries :   NeuronList | CAJAL forests
                List of queries neurons to compute distances for. See
                [`navis2cajal`][navis.interfaces.cajal.navis2cajal] for converting
                a NeuronList to CAJAL forests.
    targets :   NeuronList | CAJAL forests
                List of target neurons to compute distances to. If None, the
                queries will be used as targets (i.e. pairwise distances).
    n_sample :  int
                Number of ~ evenly distributed samples to use from each neuron for
                the distance computation.
    distance :  'geodesic' | 'euclidean'
                Distance metric to use for the distance computation. See
                [here](https://cajal.readthedocs.io/en/latest/computing-intracell-distance-matrices.html)
                for a detailed explanation of the differences. Note that 'geodesic'
                distances are only supported for single-component neurons. See
                also [`navis.heal_skeleton`][].
    num_processes : int
                Number of processes to use for parallel computation. Defaults to
                half the number of available CPU cores.
    progress :  bool
                Show progress bar.

    Returns
    -------
    matrix
                Matrix of pairwise distances.

    See Also
    --------
    navis.heal_skeleton
                Use this function to ensure that your neurons are single-component
                when using 'geodesic' distances.

    """
    if distance not in ("euclidean", "geodesic"):
        raise ValueError(f"Unknown distance metric: {distance}")

    # Force queries to be a list of CAJAL forests
    query_forests = _make_forests(queries, progress)
    # Compute intracellular distance matrices
    query_icdm = _calc_icdm(query_forests, n_sample, distance, progress)
    # Gromov-Wasserstein function requires tuples of (distance_matrix, distribution)
    query_dms = [(i, run_gw.uniform(i.shape[0])) for i in query_icdm]

    if targets is None:
        # Compute pairwise Gromov-Wasserstein distances
        dists, _ = run_gw.gw_pairwise_parallel(query_dms, num_processes=num_processes)
        return pd.DataFrame(dists, index=queries.id, columns=queries.id)
    else:
        target_forests = _make_forests(targets, progress)
        target_icdm = _calc_icdm(target_forests, n_sample, distance, progress)
        target_dms = [(i, run_gw.uniform(i.shape[0])) for i in target_icdm]
        dists, _ = run_gw.gw_query_target_parallel(query_dms, target_dms, num_processes=num_processes)
        return pd.DataFrame(dists, index=queries.id, columns=targets.id)


def _make_forests(neurons, progress):
    if isinstance(neurons, core.NeuronList):
        return navis2cajal(neurons)
    # List is assumed to be a list of forests
    elif isinstance(neurons, list):
        return neurons
    else:
        raise ValueError(
            f"Queries must be a NeuronList or a list of CAJAL forests, not {type(neurons)}."
        )


def _calc_icdm(forests, n_sample, distance, progress):
    """Compute intracellular distance matrices."""
    if distance == "euclidean":
        icdm = [
            squareform(sample_swc.icdm_euclidean(f, n_sample))
            for f in config.tqdm(
                forests,
                disable=not progress or config.pbar_hide,
                desc="Euclidean distances",
            )
        ]
    elif distance == "geodesic":
        if any(len(f) > 1 for f in forests):
            raise ValueError(
                "Geodesic distances are only supported for single-component neurons. Try using `navis.heal_skeleton`."
            )

        forests = [f[0] for f in forests]
        icdm = [
            squareform(sample_swc.icdm_geodesic(f, n_sample))
            for f in config.tqdm(
                forests,
                disable=not progress or config.pbar_hide,
                desc="Geodesic distances",
            )
        ]

    return icdm
