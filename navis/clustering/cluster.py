#    This script is part of navis (http://www.github.com/schlegelp/navis).
#    Copyright (C) 2018 Philipp Schlegel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

import os
import colorsys
import math

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import scipy.spatial

from typing import Union, Optional, List, Dict, Sequence
from typing_extensions import Literal

from concurrent.futures import ThreadPoolExecutor

from ..core.neurons import TreeNeuron
from ..core.neuronlist import NeuronList
from .. import plotting, utils, config

# Set up logging
logger = config.logger

NeuronObject = Union[TreeNeuron, NeuronList]

__all__ = sorted(['cluster_by_connectivity', 'cluster_by_synapse_placement',
                  'cluster_xyz', 'ClustResults'])


def cluster_by_connectivity(adjacency: Union[pd.DataFrame, np.ndarray],
                            similarity: Union[Literal['matching_index'],
                                              Literal['matching_index_synapses'],
                                              Literal['matching_index_weighted_synapses'],
                                              Literal['vertex'],
                                              Literal['vertex_normalized']
                                              ] = 'vertex_normalized',
                            threshold: int = 1,
                            cluster_kws: dict = {}) -> 'ClustResults':
    r"""Calculate connectivity similarity.

    This functions offers a selection of metrics to compare connectivity:

    .. list-table::
       :widths: 15 75
       :header-rows: 1

       * - Metric
         - Explanation
       * - matching_index
         - Number of shared partners divided by total number of partners.
       * - matching_index_synapses
         - Number of shared synapses (i.e. number of connections from/onto the
           same partners) divided by total number of synapses. Attention: this
           metric is tricky when there is a disparity of total number of
           connections between neuron A and B. For example, consider 100/200
           and 1/50 shared/total synapse: 101/250 results in a fairly high
           matching index of 0.404.
       * - matching_index_weighted_synapses
         - Similar to *matching_index_synapses* but slightly less prone to
           above mentioned error as it uses the percentage of shared synapses:

           .. math::

               S = \frac{\text{NeuronA}_{\text{ shared synapses}}}{\text{NeuronA}_{\text{ total synapses}}} \times \frac{\text{NeuronB}_{\text{ shared synapses}}}{\text{NeuronB}_{\text{ total synapses}}}

       * - vertex
         - Matching index that rewards shared and punishes non-shared partners.
           Based on
           `Jarrell et al., 2012 <http://science.sciencemag.org/content/337/6093/437>`_:

           .. math::

               f(x,y) = min(x,y) - C1 \times max(x,y) \times \exp(-C2 * min(x,y))

           Final score is the sum of :math:`f(x,y)` over all edges x, y between
           neurons A+B and their partners. C1 determines how negatively a case
           where one edge is much stronger than another is punished. C2
           determines the point where the similarity switches from negative to
           positive. C1 and C2 default to 0.5 and 1, respectively, but can be
           changed by passing them in a dictionary as `cluster_kws`.
       * - vertex_normalized
         - This is *vertex* similarity normalized by the lowest (hypothetical
           total dissimilarity) and highest (all edge weights the same)
           achievable score.


    Parameters
    ----------
    adjacency :         pandas DataFrame | numpy array
                        Adjacency matrix. Does not have to be symmetrical.
                        Will calculate similarity for all row neurons using
                        the column neurons as observations.
    similarity :        'matching_index' | 'matching_index_synapses' | 'matching_index_weighted_synapses' | 'vertex' | 'vertex_normalized', optional
                        Metric used to compare connectivity. See notes for
                        detailed explanation.

    Returns
    -------
    :class:`navis.ClustResults`
                         Custom cluster results class holding the distance
                         matrix and contains wrappers e.g. to plot dendograms.

    """
    ALLOWED_METHODS = ['matching_index', 'matching_index_synapses',
                       'matching_index_weighted_synapses', 'vertex',
                       'vertex_normalized']
    if similarity.lower() not in ALLOWED_METHODS:
        raise ValueError(f'"method" must be either: {", ".join(ALLOWED_METHODS)}')

    if isinstance(adjacency, np.ndarray):
        adjacency = pd.DataFrame(adjacency)

    if not isinstance(adjacency, pd.DataFrame):
        raise TypeError(f'Expected DataFrame, got "{type(adjacency)}"')

    neurons = adjacency.index

    # Prepare empty matching scores
    matching_scores = pd.DataFrame(np.zeros((len(neurons), len(neurons))),
                                   index=neurons, columns=neurons)

    if similarity.lower() in ['vertex_normalized', 'vertex']:
        vertex_score = True
    else:
        vertex_score = False

    combinations = [(nA, nB, adjacency, threshold,
                     vertex_score, cluster_kws) for nA in neurons for nB in neurons]

    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count())) as e:
        futures = e.map(_unpack_connectivity_helper, combinations)

        matching_indices = [n for n in config.tqdm(futures,
                                                   total=len(combinations),
                                                   desc='Calculating',
                                                   disable=config.pbar_hide,
                                                   leave=config.pbar_leave)]

    for i, v in enumerate(combinations):
            matching_scores.loc[v[0], v[1]] = matching_indices[i][similarity]

    results = ClustResults(matching_scores, mat_type='similarity')

    return results


def _unpack_connectivity_helper(x):
    """Helper function to unpack values from pool"""
    return _calc_connectivity_matching_index(neuronA=x[0],
                                             neuronB=x[1],
                                             connectivity=x[2],
                                             syn_threshold=x[3],
                                             vertex_score=x[4], **x[5])


def _calc_connectivity_matching_index(neuronA: int,
                                      neuronB: int,
                                      connectivity: pd.DataFrame,
                                      syn_threshold: int = 1,
                                      vertex_score: bool = True,
                                      **kwargs) -> Dict[str,
                                                        Union[float, int]]:
    """Calculate and return various matching indices between two neurons.

    Parameters
    ----------
    neuronA :         skeleton ID
    neuronB :         skeleton ID
    connectivity :    pandas DataFrame
                      Connectivity data as provided by :func:`navis.get_partners`.
    syn_threshold :   int, optional
                      Min number of synapses for a connection to be considered.
                      Default = 1
    vertex_score :    bool, optional
                      If False, no vertex score is returned (much faster!).
                      Default = True

    Returns
    -------
    dict
                      Containing all initially described matching indices

    Notes
    -----
    |matching_index =           Number of shared partners divided by total number
    |                           of partners

    |matching_index_synapses =  Number of shared synapses divided by total number
    |                           of synapses. Attention! matching_index_synapses
    |                           is tricky, because if neuronA has lots of
    |                           connections and neuronB only little, they will
    |                           still get a high matching index.
    |                           E.g. 100 of 200 / 1 of 50 = 101/250
    |                           -> ``matching index = 0.404``

    |matching_index_weighted_synapses = Similar to matching_index_synapses but
    |                           slightly less prone to above mentioned error:
    |                           % of shared synapses A * % of shared synapses
    |                           B * 2 / (% of shared synapses A + % of shared
    |                           synapses B)
    |                           -> value will be between 0 and 1; if one neuronB
    |                           has only few connections (percentage) to a shared
    |                           partner, the final value will also be small

    |vertex_normalized =        Matching index that rewards shared and punishes
    |                           non-shared partners. Vertex similarity based on
    |                           Jarrell et al., 2012:
    |                           f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
    |                           x,y = edge weights to compare
    |                           vertex_similarity is the sum of f over all vertices
    |                           C1 determines how negatively a case where one edge
    |                           is much stronger than another is punished
    |                           C2 determines the point where the similarity
    |                           switches from negative to positive

    """
    # Subset to neurons that actually connect with our neuron pair
    this_cn = connectivity.loc[[neuronA, neuronB]]

    # Get all neurons that connect to either A or B
    total = this_cn.loc[:, (this_cn >= syn_threshold).sum(axis=0) >= 1]
    n_total = total.shape[1]

    # Get all neurons that connect to both A and B
    shared = this_cn.loc[:, (this_cn >= syn_threshold).sum(axis=0) == 2]
    n_shared = shared.shape[1]

    shared_sum = shared.sum(axis=1)
    n_synapses_sharedA = shared_sum[neuronA]
    n_synapses_sharedB = shared_sum[neuronB]

    total_sum = total.sum(axis=1)
    n_synapses_totalA = total_sum[neuronA]
    n_synapses_totalB = total_sum[neuronB]

    # If neuronA == neuronB, above counts will not be single values but
    # DataFrames -> we'll simply have to get the first values
    if neuronA == neuronB:
        n_synapses_sharedA = n_synapses_sharedA.values[0]
        n_synapses_sharedB = n_synapses_sharedB.values[0]

        n_synapses_totalA = n_synapses_totalA.values[0]
        n_synapses_totalB = n_synapses_totalB.values[0]

    # Sum up
    n_synapses_shared = n_synapses_sharedA + n_synapses_sharedB
    n_synapses_total = n_synapses_totalA + n_synapses_totalB

    # Vertex similarity based on Jarrell et al., 2012
    # f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
    # x,y = edge weights to compare
    # vertex_similarity is the sum of f over all vertices
    # C1 determines how negatively a case where one edge is much stronger than
    #    another is punished
    # C2 determines the point where the similarity switches from negative to
    # positive
    C1 = kwargs.get('C1', 0.5)
    C2 = kwargs.get('C2', 1)
    similarity_indices: Dict[str, Union[float, int]] = {}

    if vertex_score:
        # Get min and max between both neurons
        this_max = np.max(this_cn, axis=0)
        this_min = np.min(this_cn, axis=0)

        # The max possible score is when both synapse counts are the same:
        # in which case score = max(x,y) - C1 * max(x,y) * e^(-C2 * max(x,y))
        max_score = this_max - C1 * this_max * np.exp(- C2 * this_max)

        # The smallest possible score is when either synapse count is 0:
        # in which case score = -C1 * max(a,b)
        min_score = -C1 * this_max

        # Implement: f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
        v_sim = this_min - C1 * this_max * np.exp(- C2 * this_min)

        # Sum over all partners
        vertex_similarity = v_sim.sum()

        similarity_indices['vertex'] = vertex_similarity

        try:
            similarity_indices['vertex_normalized'] = (
                vertex_similarity - min_score.sum()) / (max_score.sum() - min_score.sum())
        except BaseException:
            similarity_indices['vertex_normalized'] = 0

    if n_total != 0:
        similarity_indices['matching_index'] = n_shared / n_total
        similarity_indices['matching_index_synapses'] = n_synapses_shared / n_synapses_total
        if n_synapses_sharedA != 0 and n_synapses_sharedB != 0:
            similarity_indices['matching_index_weighted_synapses'] = (
                n_synapses_sharedA / n_synapses_totalA) * (n_synapses_sharedB / n_synapses_totalB)
        else:
            # If no shared synapses at all:
            similarity_indices['matching_index_weighted_synapses'] = 0
    else:
        similarity_indices['matching_index'] = 0
        similarity_indices['matching_index_synapses'] = 0
        similarity_indices['matching_index_weighted_synapses'] = 0

    return similarity_indices


def _unpack_synapse_helper(x):
    """Helper function to unpack values from pool."""
    return _calc_synapse_similarity(x[0], x[1], x[2], x[3], x[4])


def _calc_synapse_similarity(cnA: pd.DataFrame,
                             cnB: pd.DataFrame,
                             sigma: int = 2,
                             omega: int = 2,
                             restrict_cn: Optional[List[str]] = None
                             ) -> float:
    """Calculates synapse similarity score.

    Synapse similarity score is calculated by calculating for each synapse of
    neuron A: (1) the distance to the closest (eucledian) synapse in neuron B
    and (2) comparing the synapse density around synapse A and B. This is type
    sensitive: presynapses will only be matched with presynapses, post with
    post, etc. The formula is described in Schlegel et al., eLife (2017).

    Parameters
    ----------
    (cnA, cnB) :    Connector tables.
    sigma :         int, optional
                    Distance that is considered to be "close".
    omega :         int, optional
                    Radius over which to calculate synapse density.

    Returns
    -------
    synapse_similarity_score

    """
    all_values = []

    # Get the connector types that we want to compare between neuron A and B
    if isinstance(restrict_cn, type(None)):
        # If no restrictions, get all cn types in neuron A
        cn_to_check = cnA.type.unique()
    else:
        # Intersect restricted connectors and actually available types
        cn_to_check = set(cnA.type.unique()) & set(restrict_cn)

    # Iterate over all types of connectors
    for r in cn_to_check:
        # Skip if either neuronA or neuronB don't have this synapse type
        if cnB[cnB.type == r].empty:
            all_values += [0] * cnA[cnA.type == r].shape[0]
            continue

        # Get inter-neuron matrix
        dist_mat = scipy.spatial.distance.cdist(cnA[cnA.type == r][['x', 'y', 'z']],
                                                cnB[cnB.type == r][['x', 'y', 'z']])

        # Get index of closest synapse in neuron B
        closest_ix = np.argmin(dist_mat, axis=1)

        # Get closest distances
        closest_dist = dist_mat.min(axis=1)

        # Get intra-neuron matrices for synapse density checking
        distA = scipy.spatial.distance.pdist(
            cnA[cnA.type == r][['x', 'y', 'z']])
        distA = scipy.spatial.distance.squareform(distA)
        distB = scipy.spatial.distance.pdist(
            cnB[cnB.type == r][['x', 'y', 'z']])
        distB = scipy.spatial.distance.squareform(distB)

        # Calculate number of synapses closer than OMEGA. This does count itself!
        closeA = (distA <= omega).sum(axis=1)
        closeB = (distB <= omega).sum(axis=1)

        # Now calculate the scores over all synapses
        for a in range(distA.shape[0]):
            this_synapse_value = math.exp(-1 * math.fabs(closeA[a] - closeB[closest_ix[a]]) / (
                closeA[a] + closeB[closest_ix[a]])) * math.exp(-1 * (closest_dist[a]**2) / (2 * sigma**2))
            all_values.append(this_synapse_value)

    score = sum(all_values) / len(all_values)

    return score


def cluster_by_synapse_placement(x: Union[NeuronList,
                                          Dict[Union[str, int], pd.DataFrame]],
                                 sigma: int = 2,
                                 omega: int = 2,
                                 mu_score: bool = True,
                                 restrict_cn: Optional[List[str]] = None
                                 ) -> 'ClustResults':
    r"""Cluster neurons based on their synapse placement.

    Distances score is calculated by calculating for each synapse of
    neuron A: (1) the distance to the closest (eucledian) synapse in neuron B
    and (2) comparing the synapse density around synapse A and B.
    This is type-sensitive: presynapses will only be matched with presynapses,
    post with post, etc. The formula is described in
    `Schlegel et al., eLife (2017) <https://elifesciences.org/articles/16799>`_:

    .. math::

        f(i_{s},j_{k}) = \exp(\frac{-d^{2}_{sk}}{2\sigma^{2}}) \exp(\frac{|n(i_{s})-n(j_{k})|}{n(i_{s})+n(j_{k})})

    The synapse similarity score for neurons i and j being the average
    of :math:`f(i_{s},j_{k})` over all synapses s of i. Synapse k is the
    closest synapse of the same sign (pre/post) in neuron j to synapse s.
    :math:`d^{2}_{sk}` is the eucledian distance between these distances.
    Variable :math:`\sigma` (``sigma``) determines what distance between
    s and k is considered "close". :math:`n(i_{s})` and :math:`n(j_{k})` are
    defined as the number of synapses of neuron i/j that are within given
    radius :math:`\omega` (``omega``) of synapse s and j, respectively (same
    sign only). This esnures that in cases of a strong disparity between
    :math:`n(i_{s})` and :math:`n(j_{k})`, the synapse similarity will be
    close to zero even if the distance between s and k is very small.


    Parameters
    ----------
    x :                 NeuronList | Dict of pandas.DataFrame
                        Neurons to compare. Dict must map
                        ``{neuron_name: connector_table}``. If NeuronList,
                        each neuron must have ``.connectors`` connector tables.
                        Connector tables must have `x`, `y`, `y` columns. If
                        a `type`/`relation`/`label` columns is present, will
                        use this to compare connectors only within type.
    sigma :             int, optional
                        Distance between synapses that is considered to be
                        "close".
    omega :             int, optional
                        Radius over which to calculate synapse density.
    mu_score :          bool, optional
                        If True, score is calculated as mean between A->B and
                        B->A comparison.
    restrict_cn :       int | list | None, optional
                        Restrict to given connector types. Must map to
                        a `type`, `relation` or `label` column in the
                        connector tables.
                        If None, will use all connector types. Use either
                        single integer or list. E.g. ``restrict_cn=[0, 1]``
                        to use only pre- and postsynapses.

    Returns
    -------
    :class:`~navis.ClustResults`
                Object that contains distance matrix and methods to plot
                dendrograms.

    """
    if isinstance(x, dict):
        if any([not isinstance(d, pd.DataFrame) for d in x.values()]):
            raise TypeError('Values in dict must be pandas.DataFrames.')
    elif isinstance(x, NeuronList):
        if any([not n.has_connectors for n in x]):
            raise ValueError('All neurons must have connector tables as'
                             ' .connectors property.')
    else:
        raise TypeError('Expected Neuronlist or dict of connector tables,'
                        f' got {type(x)}')

    # If single value, turn into list
    if not isinstance(restrict_cn, type(None)):
        restrict_cn = utils.make_iterable(restrict_cn)

    neurons = x.id if isinstance(x, NeuronList) else list(x.keys())

    sim_matrix = pd.DataFrame(np.zeros((len(neurons), len(neurons))),
                              index=neurons,
                              columns=neurons)

    if isinstance(x, NeuronList):
        combinations = [(nA.connectors, nB.connectors, sigma, omega, restrict_cn)
                        for nA in x for nB in x]
        comb_names = [(nA.id, nB.id) for nA in x for nB in x]
    else:
        combinations = [(nA, nB, sigma, omega, restrict_cn) for nA in x.values() for nB in x.values()]
        comb_names = [(nA, nB) for nA in x.keys() for nB in x.keys()]

    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count())) as e:
        futures = e.map(_unpack_synapse_helper, combinations)

        scores = [n for n in config.tqdm(futures, total=len(combinations),
                                         desc='Processing',
                                         disable=config.pbar_hide,
                                         leave=config.pbar_leave)]

    for c, v in zip(comb_names, scores):
        sim_matrix.loc[c[0], c[1]] = v

    if mu_score:
        sim_matrix = (sim_matrix + sim_matrix.T) / 2

    res = ClustResults(sim_matrix, mat_type='similarity')

    if isinstance(x, NeuronList):
        res.neurons = x  # type: ignore

    return res


def cluster_xyz(x: Union[pd.DataFrame, np.ndarray],
                labels: Optional[List[str]] = None
                ) -> 'ClustResults':
    """Thin wrapper for ``scipy.scipy.spatial.distance``.

    Takes a list of x,y,z coordinates and calculates Euclidean distance matrix.

    Parameters
    ----------
    x :             pandas.DataFrame | numpy array (N, 3)
                    If DataFrame, must contain ``x``,``y``,``z`` columns.
    labels :        list of str, optional
                    Labels for each leaf of the dendrogram
                    (e.g. connector ids).

    Returns
    -------
    :class:`navis.ClustResults`
                      Contains distance matrix and methods to generate plots.

    Examples
    --------
    This examples assumes you understand the basics of using navis:

    >>> import navis
    >>> import matplotlib.pyplot as plt
    >>> n = navis.example_neurons(n=1)
    >>> rs = navis.cluster_xyz(n.nodes,
    ...                        labels=n.nodes.node_id.values)
    >>> fig = rs.plot_matrix()
    >>> plt.show()                                              # doctest: +SKIP

    """
    if isinstance(x, pd.DataFrame):
        # Generate numpy array containing x, y, z coordinates
        try:
            s = x[['x', 'y', 'z']].values
        except BaseException:
            raise ValueError('DataFrame must have x/y/z columns')
    elif isinstance(x, np.ndarray):
        if x.shape[1] != 3:
            raise ValueError('Array must be of shape (N, 3).')
        s = x
    else:
        raise TypeError(f'Expected DataFrame or numpy array, got "{type(x)}".')

    # Calculate euclidean distance matrix
    condensed_dist_mat = scipy.spatial.distance.pdist(s, 'euclidean')
    squared_dist_mat = scipy.spatial.distance.squareform(condensed_dist_mat)

    return ClustResults(squared_dist_mat, labels=labels, mat_type='distance')


class ClustResults:
    """Class to handle, analyze and plot similarity/distance matrices.

    Contains thin wrappers for ``scipy.cluster``.

    Attributes
    ----------
    dist_mat :  Distance matrix (0=similar, 1=dissimilar)
    sim_mat :   Similarity matrix (0=dissimilar, 1=similar)
    linkage :   Hierarchical clustering. Run :func:`navis.ClustResults.cluster`
                to generate linkage. By default, WARD's algorithm is used.
    leafs :     list of skids

    Examples
    --------
    # DEPCREATED

    """

    _PERM_MAT_TYPES = ['similarity', 'distance']

    neurons: Optional['NeuronList']

    def __init__(self,
                 mat: Union[np.ndarray, pd.DataFrame],
                 labels: Optional[List[str]] = None,
                 mat_type: Union[Literal['distance'],
                                 Literal['similarity']] = 'distance'
                 ):
        """Initialize class instance.

        Parameters
        ----------
        mat :       numpy.array | pandas.DataFrame
                    Distance or similarity matrix.
        labels :    list, optional
                    Labels for matrix.
        mat_type :  'distance' | 'similarity', default = 'distance'
                    Sets the type of input matrix:
                      - 'similarity' = high values are more similar
                      - 'distance' = low values are more similar

                    The "missing" matrix type will be computed. For clustering,
                    plotting, etc. distance matrices are used.

        """
        if mat_type not in ClustResults._PERM_MAT_TYPES:
            raise ValueError(f'Matrix type "{mat_type}" unkown.')

        if mat_type == 'similarity':
            self.dist_mat = self._invert_mat(mat)
            self.sim_mat = mat
        else:
            self.dist_mat = mat
            self.sim_mat = self._invert_mat(mat)

        self.labels = labels
        self.mat_type = mat_type

        if isinstance(labels, type(None)) and isinstance(mat, pd.DataFrame):
            self.labels = mat.columns.tolist()

    def __getattr__(self, key):
        if key == 'linkage':
            self.cluster()
            return self.linkage
        elif key == 'condensed_dist_mat':
            return scipy.spatial.distance.squareform(self.dist_mat,
                                                     checks=False)
        elif key in ['leafs', 'leaves']:
            return self.get_leafs()
        elif key == 'cophenet':
            return self.calc_cophenet()
        elif key == 'agg_coeff':
            return self.calc_agg_coeff()

    def get_leafs(self, use_labels: bool = False) -> Sequence:
        """Use to retrieve labels.

        Parameters
        ----------
        use_labels :    bool, optional
                        If True, self.labels will be returned. If False, will
                        use either columns (if matrix is pandas DataFrame)
                        or indices (if matrix is np.ndarray)

        """

        if isinstance(self.dist_mat, pd.DataFrame):
            if use_labels:
                return [self.labels[i] for i in scipy.cluster.hierarchy.leaves_list(self.linkage)]
            else:
                return [self.dist_mat.columns.tolist()[i] for i in scipy.cluster.hierarchy.leaves_list(self.linkage)]
        else:
            return scipy.cluster.hierarchy.leaves_list(self.linkage)

    def calc_cophenet(self) -> float:
        """Returns Cophenetic Correlation coefficient of your clustering.

        This (very very briefly) compares (correlates) the actual pairwise
        distances of all your samples to those implied by the hierarchical
        clustering. The closer the value is to 1, the better the clustering
        preserves the original distances.

        """
        return scipy.cluster.hierarchy.cophenet(self.linkage,
                                                self.condensed_dist_mat)

    def calc_agg_coeff(self) -> float:
        """Return the agglomerative coefficient.

        This measures the clustering structure of the linkage matrix. Because
        it grows with the number of observations, this measure should not be
        used to compare datasets of very different sizes.

        For each observation i, denote by m(i) its dissimilarity to the first
        cluster it is merged with, divided by the dissimilarity of the merger
        in the final step of the algorithm. The agglomerative coefficient is
        the average of all 1 - m(i).

        """
        # Turn into pandas DataFrame for fancy indexing
        Z = pd.DataFrame(self.linkage, columns=['obs1', 'obs2', 'dist', 'n_org'])

        # Get all distances at which an original observation is merged
        all_dist = Z[(Z.obs1.isin(self.leafs)) | (Z.obs2.isin(self.leafs))].dist.values

        # Divide all distances by last merger
        all_dist /= self.linkage[-1][2]

        # Calc final coefficient
        coeff = np.mean(1 - all_dist)

        return coeff

    def _invert_mat(self, sim_mat: pd.DataFrame) -> pd.DataFrame:
        """Invert matrix."""
        if isinstance(sim_mat, pd.DataFrame):
            return (sim_mat - sim_mat.max().max()) * -1
        else:
            return (sim_mat - sim_mat.max()) * -1

    def cluster(self, method: str = 'ward') -> None:
        """Cluster distance matrix.

        This will automatically be called when attribute linkage is requested
        for the first time.

        Parameters
        ----------
        method :    str, optional
                    Clustering method (see scipy.cluster.hierarchy.linkage
                    for reference)

        """
        # Use condensed distance matrix - otherwise clustering thinks we are
        # passing observations instead of final scores
        self.linkage = scipy.cluster.hierarchy.linkage(self.condensed_dist_mat,
                                                       method=method)

        # Save method in case we want to look it up later
        self.cluster_method = method

        logger.info(f'Clustering done using method "{method}"')

    def plot_dendrogram(self,
                        color_threshold: Optional[float] = None,
                        return_dendrogram: bool = False,
                        labels: Optional[List[str]] = None,
                        fig: Optional['matplotlib.figure.Figure'] = None,
                        **kwargs):
        """Plot dendrogram using matplotlib.

        Parameters
        ----------
        color_threshold :   int | float, optional
                            Coloring threshold for dendrogram.
        return_dendrogram : bool, optional
                            If True, dendrogram object is returned instead of
                            figure.
        labels :            list of str, dict
                            Labels in order of original observation or
                            dictionary with mapping original labels.
        kwargs
                            Passed to ``scipy.cluster.hierarchy.dendrogram()``

        Returns
        -------
        matplotlib.figure
                            If ``return_dendrogram=False`.
        sciyp.cluster.hierarchy.dendrogram
                            If ``return_dendrogram=True``.

        """
        import matplotlib.pyplot as plt

        if isinstance(labels, type(None)):
            labels = self.labels
        elif isinstance(labels, dict):
            labels = [labels[l] for l in self.labels]

        if not fig:
            fig = plt.figure()

        dn_kwargs = {'leaf_rotation': 90,
                     'above_threshold_color': 'k'}
        dn_kwargs.update(kwargs)

        dn = scipy.cluster.hierarchy.dendrogram(self.linkage,
                                                color_threshold=color_threshold,
                                                labels=labels,
                                                **dn_kwargs)
        logger.info(
            'Use matplotlib.pyplot.show() to render dendrogram.')

        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        try:
            plt.tight_layout()
        except BaseException:
            pass

        if return_dendrogram:
            return dn
        else:
            return fig

    def plot_clustermap(self, **kwargs):
        """Plot distance matrix and dendrogram using seaborn.

        Parameters
        ----------
        kwargs      dict
                    Keyword arguments to be passed to seaborn.clustermap. See
                    http://seaborn.pydata.org/generated/seaborn.clustermap.html


        Returns
        -------
        seaborn.clustermap

        """
        import matplotlib.pyplot as plt

        try:
            import seaborn as sns
        except BaseException:
            raise ImportError('Need seaborn package installed.')

        cg = sns.clustermap(self.dist_mat, row_linkage=self.linkage,
                            col_linkage=self.linkage, **kwargs)

        # Rotate labels
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

        # Make labels smaller
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), fontsize=4)
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), fontsize=4)

        # Increase padding
        cg.fig.subplots_adjust(right=.8, top=.95, bottom=.2)

        logger.info(
            'Use matplotlib.pyplot.show() to render figure.')

        return cg

    def plot_matrix(self) -> 'matplotlib.figure.Figure':
        """Plot distance matrix and dendrogram using matplotlib.

        Returns
        -------
        matplotlib figure

        """
        import pylab

        # Compute and plot first dendrogram for all nodes.
        fig = pylab.figure(figsize=(8, 8))
        ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
        Z1 = scipy.cluster.hierarchy.dendrogram(
            self.linkage, orientation='left', labels=self.labels)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Compute and plot second dendrogram.
        ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
        Z2 = scipy.cluster.hierarchy.dendrogram(
            self.linkage, labels=self.labels)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Plot distance matrix.
        axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
        idx1 = Z1['leaves']
        idx2 = Z2['leaves']
        D = self.dist_mat.copy()

        if isinstance(D, pd.DataFrame):
            D = D.values

        D = D[idx1, :]
        D = D[:, idx2]
        im = axmatrix.matshow(
            D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])

        # Plot colorbar.
        axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
        pylab.colorbar(im, cax=axcolor)

        logger.info(
            'Use matplotlib.pyplot.show() to render figure.')

        return fig

    def plot3d(self,
               k: Union[int, float] = 5,
               criterion: str = 'maxclust',
               **kwargs):
        """Plot neuron using :func:`navis.plot.plot3d`.

        Will only work if instance has neurons attached to it.

        Parameters
        ----------
        k :         int | float
        criterion : 'maxclust' | 'distance', optional
                    If ``maxclust``, ``k`` clusters will be formed. If
                    ``distance`` clusters will be created at threshold ``k``.
        **kwargs
                Will be passed to ``navis.plot3d()``. See
                ``help(plot.plot3d)`` for a list of keywords.

        See Also
        --------
        :func:`navis.plot3d`
                    Function called to generate 3d plot.

        """

        if 'neurons' not in self.__dict__:
            logger.error(
                'This works only with cluster results from neurons')
            return None

        cmap = self.get_colormap(k=k, criterion=criterion)

        kwargs.update({'color': cmap})

        return plotting.plot3d(self.neurons, **kwargs)

    def get_colormap(self,
                     k: Union[int, float] = 5,
                     criterion: str = 'maxclust'):
        """Generate colormap based on clustering.

        Parameters
        ----------
        k :         int | float
        criterion : 'maxclust' | 'distance', optional
                    If ``maxclust``, ``k`` clusters will be formed. If
                    ``distance`` clusters will be created at threshold ``k``.

        Returns
        -------
        dict
                    ``{id: (r, g, b), ...}``

        """
        cl = self.get_clusters(k, criterion, return_type='indices')

        cl = [[self.dist_mat.index.tolist()[i] for i in l] for l in cl]

        colors = [colorsys.hsv_to_rgb(1 / len(cl) * i, 1, 1)
                  for i in range(len(cl) + 1)]

        return {n: colors[i] for i in range(len(cl)) for n in cl[i]}

    def get_clusters(self,
                     k: Union[int, float],
                     criterion: str = 'maxclust',
                     return_type: str = 'labels') -> List[list]:
        """Get clusters.

        Just a thin wrapper for ``scipy.cluster.hierarchy.fcluster``.

        Parameters
        ----------
        k :             int | float
        criterion :     'maxclust' | 'distance', optional
                        If ``maxclust``, ``k`` clusters will be formed. If
                        ``distance`` clusters will be created at threshold.
                        ``k``.
        return_type :   'labels' | 'indices' | 'columns' | 'rows'
                        Determines what to construct the clusters of. 'labels'
                        only works if labels are provided. 'indices' refers
                        to index in distance matrix. 'columns'/'rows' works
                        if distance matrix is pandas DataFrame

        Returns
        -------
        list
                    list of clusters ``[[leaf1, leaf5], [leaf2, ...], ...]``

        """
        cl = scipy.cluster.hierarchy.fcluster(
            self.linkage, k, criterion=criterion)

        if not isinstance(self.labels, type(None)) and return_type.lower() == 'labels':
            return [[self.labels[j] for j in range(len(cl)) if cl[j] == i] for i in range(min(cl), max(cl) + 1)]
        elif return_type.lower() == 'rows':
            return [[self.dist_mat.columns.tolist()[j] for j in range(len(cl)) if cl[j] == i] for i in range(min(cl), max(cl) + 1)]
        elif return_type.lower() == 'columns':
            return [[self.dist_mat.index.tolist()[j] for j in range(len(cl)) if cl[j] == i] for i in range(min(cl), max(cl) + 1)]
        else:
            return [[j for j in range(len(cl)) if cl[j] == i] for i in range(min(cl), max(cl) + 1)]

    def to_tree(self):
        """Turn linkage to ete3 tree.

        See Also
        --------
        http://etetoolkit.org/
            Ete3 homepage

        Returns
        -------
        ete 3 tree

        """
        try:
            import ete3  # ignore type
        except BaseException:
            raise ImportError('Please install ete3 package to use this function.')

        max_dist = self.linkage[-1][2]
        n_original_obs = self.dist_mat.shape[0]

        list_of_childs = {n_original_obs + i: e[:2]
                          for i, e in enumerate(self.linkage)}
        list_of_parents = {int(e[0]): int(n_original_obs + i)
                           for i, e in enumerate(self.linkage)}
        list_of_parents.update(
            {int(e[1]): int(n_original_obs + i) for i, e in enumerate(self.linkage)})

        total_dist = {n_original_obs + i: e[2]
                      for i, e in enumerate(self.linkage)}
        # Process total distance into distances between nodes (root = 0)
        dist_to_parent = {
            n: max_dist - total_dist[list_of_parents[n]] for n in list_of_parents}

        names = {i: n for i, n in enumerate(self.dist_mat.columns.tolist())}

        # Create empty tree
        tree = ete3.Tree()

        # Start with root node
        root = sorted(list(list_of_childs.keys()), reverse=True)[0]
        treenodes = {root: tree.add_child()}

        for k in sorted(list(list_of_childs.keys()), reverse=True):
            e = list_of_childs[k]
            treenodes[e[0]] = treenodes[k].add_child(
                dist=dist_to_parent[e[0]], name=names.get(e[0], None))
            treenodes[e[1]] = treenodes[k].add_child(
                dist=dist_to_parent[e[1]], name=names.get(e[1], None))

        return tree
