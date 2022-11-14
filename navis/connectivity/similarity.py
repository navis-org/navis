#    This script is part of navis (http://www.github.com/navis-org/navis).
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

import numpy as np
import pandas as pd
import scipy.spatial as ssp
import scipy.stats as sst

from functools import partial
from itertools import product
from typing import Union, Optional, List
from typing_extensions import Literal

from concurrent.futures import ProcessPoolExecutor

from .. import config, core, utils

# Set up logging
logger = config.get_logger(__name__)


__all__ = sorted(['connectivity_similarity',  'synapse_similarity'])


def connectivity_similarity(adjacency: Union[pd.DataFrame, np.ndarray],
                            metric: Union[Literal['matching_index'],
                                          Literal['matching_index_synapses'],
                                          Literal['matching_index_weighted_synapses'],
                                          Literal['vertex'],
                                          Literal['vertex_normalized'],
                                          Literal['cosine'],
                                          ] = 'vertex_normalized',
                            threshold: Optional[int] = None,
                            n_cores: int = max(1, os.cpu_count() // 2),
                            **kwargs) -> pd.DataFrame:
    r"""Calculate connectivity similarity.

    This functions offers a selection of metrics to compare connectivity:

    .. list-table::
       :widths: 15 75
       :header-rows: 1

       * - Metric
         - Explanation
       * - cosine
         - Cosine similarity (see `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html>`_)
       * - rank_index
         - Normalized difference in rank of synaptic partners.
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
           changed by passing them in a dictionary as ``**kwargs``.
       * - vertex_normalized
         - This is *vertex* similarity normalized by the lowest (hypothetical
           total dissimilarity) and highest (all edge weights the same)
           achievable score.


    Parameters
    ----------
    adjacency :         pandas DataFrame | numpy array
                        (N, M) observation vector with M observations for N
                        neurons - e.g. an adjacency matrix. Will calculate
                        similarity for all rows using the columns as observations.
    metric :            'cosine' | 'rank_index'| 'matching_index' | 'matching_index_synapses' | 'matching_index_weighted_synapses' | 'vertex' | 'vertex_normalized'
                        Metric used to compare connectivity. See notes for
                        detailed explanation.
    threshold :         int, optional
                        Connections weaker than this will be set to zero.
    n_cores :           int
                        Number of parallel processes to use. Defaults to half
                        the available cores.

    Returns
    -------
    DataFrame
                        Pandas DataFrame with similarity scores. Neurons without
                        any connectivity will show up with ``np.nan`` for scores.

    """
    FUNC_MAP = {'rank_index': _calc_rank_index,
                'matching_index': _calc_matching_index,
                'matching_index_synapses': _calc_matching_index_synapses,
                'matching_index_weighted_synapses': partial(_calc_matching_index_synapses, weighted=True),
                'vertex': _calc_vertex_similarity,
                'vertex_normalized': partial(_calc_vertex_similarity, normalize=True),
                'cosine': _calc_cosine_similarity
                }

    if not isinstance(metric, str) or metric.lower() not in FUNC_MAP:
        raise ValueError(f'"metric" must be either: {", ".join(FUNC_MAP.keys())}')

    score_func = FUNC_MAP[metric.lower()]

    if isinstance(adjacency, np.ndarray):
        adjacency = pd.DataFrame(adjacency)
    elif not isinstance(adjacency, pd.DataFrame):
        raise TypeError(f'Expected DataFrame, got "{type(adjacency)}"')

    if threshold:
        # Do not manipulate original
        adjacency = adjacency.copy()
        adjacency[adjacency < threshold] = 0

    # Skip expensive checks if no empty vectors
    if (adjacency.max(axis=1) == 0).any():
        kwargs['validate'] = True
    else:
        kwargs['validate'] = False

    # Prepare combinations matching scores
    comb = combinations_generator(score_func, adjacency, **kwargs)

    # Note that while we are mapping from a generator (`comb`), the pool will
    # unfortunately not evaluate this lazily. This is a "bug" in the standard
    # library that might get fixed at some point.
    if n_cores > 1:
        with ProcessPoolExecutor(max_workers=n_cores) as e:
            futures = e.map(_distributor, comb, chunksize=50000)

            matching_indices = [n for n in config.tqdm(futures,
                                                       total=adjacency.shape[0]**2,
                                                       desc='Calc. similarity',
                                                       disable=config.pbar_hide,
                                                       leave=config.pbar_leave)]
    else:
        matching_indices = []
        for c in config.tqdm(comb,
                             total=adjacency.shape[0]**2,
                             desc='Calc. similarity',
                             disable=config.pbar_hide,
                             leave=config.pbar_leave):
            matching_indices.append(_distributor(c))

    # Create empty scores matrix
    neurons = adjacency.index.values
    matching_scores = pd.DataFrame(np.zeros((len(neurons), len(neurons))),
                                   index=neurons, columns=neurons)
    # Populate scores matrix
    comb_id = product(neurons, neurons)
    for i, v in enumerate(comb_id):
        matching_scores.at[v[0], v[1]] = matching_indices[i]

    return matching_scores


def _distributor(args):
    """Help submitting combinations to actual function.

    IMPORTANT: This function also takes care of skipping if vectors are empty!
    """
    func, vecA, vecB, args, kwargs = args

    # If either neuron is fully disconnected return NaN
    if kwargs.pop('validate', True):
        if not vecA.any() or not vecB.any():
            return np.nan

    return func(vecA, vecB, *args, **kwargs)


def combinations_generator(func, adjacency, *args, **kwargs):
    """Lazy generation of connectivity vector combinations."""
    comb = product(adjacency.values, adjacency.values)
    for i in range(adjacency.shape[0]**2):
        this = next(comb)
        #non_zero = (this[0] > 0) | (this[1] > 0)
        #yield (func, this[0][non_zero], this[1][non_zero], args, kwargs)
        yield (func, this[0], this[1], args, kwargs)


def _calc_rank_index(vecA, vecB, normalize=True):
    """Calculate rank index between two vectors."""
    rankA = sst.rankdata(vecA)
    rankB = sst.rankdata(vecB)
    dist = np.abs(rankA - rankB).sum()

    if normalize:
        # Normalize by worst case scenario
        a = np.arange(len(vecA))
        b = a[::-1]
        dist = dist / np.abs(a - b).sum()
        sim = 1 - dist
    else:
        if dist:
            sim = 1 / dist
        else:
            sim = 1
    return sim


def _calc_matching_index(vecA, vecB, normalize=True):
    """Calculate matching index between two vectors."""
    is_A = (vecA > 0)
    is_B = (vecB > 0)
    n_total = (is_A | is_B).sum()
    n_shared = (is_A & is_B).sum()

    return n_shared / n_total


def _calc_matching_index_synapses(vecA, vecB, weighted=False):
    """Calculate matching index based on synapses between two vectors."""
    is_A = (vecA > 0)
    is_B = (vecB > 0)
    is_both = is_A & is_B

    if not weighted:
        n_shared = vecA[is_both].sum() + vecB[is_both].sum()
        return n_shared / (vecA.sum() + vecB.sum())
    else:
        return vecA[is_both].sum() / vecA.sum() * vecB[is_both].sum() / vecB.sum()


def _calc_vertex_similarity2(adj, C1=0.5, C2=1, normalize=False):
    """Calculate vertex similarity between two vectors."""
    from tqdm import trange
    sims = []
    for i in trange(adj.shape[0]):
        this = np.repeat(adj[i:i+1], adj.shape[0], axis=0)
        comb = np.dstack((this, adj))
        this_max = comb.max(axis=2)
        this_min = comb.min(axis=2)

        # Implement: f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
        v_sim = this_min - C1 * this_max * np.exp(- C2 * this_min)

        # Sum over all partners
        vs = v_sim.sum(axis=1)

        if normalize:
            # The max possible score is when both synapse counts are the same:
            # in which case score = max(x,y) - C1 * max(x,y) * e^(-C2 * max(x,y))
            max_score = (this_max - C1 * this_max * np.exp(- C2 * this_max)).sum(axis=1)

            # The smallest possible score is when either synapse count is 0:
            # in which case score = -C1 * max(a,b)
            min_score = (-C1 * this_max).sum(axis=1)

            vs = (vs - min_score) / (max_score - min_score)

        sims.append(vs)

    return np.vstack(sims)


def _calc_vertex_similarity(vecA, vecB, C1=0.5, C2=1, normalize=False):
    """Calculate vertex similarity between two vectors."""
    # np.minimum is much faster than np.min(np.vstack(vecA, vecB), axis=1) here

    this_max = np.maximum(vecA, vecB)
    this_min = np.minimum(vecA, vecB)

    # Implement: f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
    v_sim = this_min - C1 * this_max * np.exp(- C2 * this_min)

    # Sum over all partners
    vertex_similarity = v_sim.sum()

    if not normalize:
        return vertex_similarity

    # The max possible score is when both synapse counts are the same:
    # in which case score = max(x,y) - C1 * max(x,y) * e^(-C2 * max(x,y))
    max_score = (this_max - C1 * this_max * np.exp(- C2 * this_max)).sum()

    # The smallest possible score is when either synapse count is 0:
    # in which case score = -C1 * max(a,b)
    min_score = (-C1 * this_max).sum()

    if min_score < max_score:
        return (vertex_similarity - min_score) / (max_score - min_score)
    else:
        return 0


def _calc_cosine_similarity(vecA, vecB):
    """Calculate cosine similarity between two vectors."""
    dist = ssp.distance.cosine(vecA, vecB)

    return 1 - dist


def synapse_similarity(x: 'core.NeuronList',
                       sigma: Union[float, int],
                       omega: Union[float, int],
                       mu_score: bool = True,
                       restrict_cn: Optional[List[str]] = None,
                       n_cores: int = max(1, os.cpu_count() // 2)
                       ) -> pd.DataFrame:
    r"""Cluster neurons based on their synapse placement.

    Distances score is calculated by calculating for each synapse of
    neuron A: (1) the (Euclidian) distance to the closest synapse in neuron B
    and (2) comparing the synapse density around synapse A and B.
    This is type-sensitive: presynapses will only be matched with presynapses,
    post with post, etc. The formula is described in
    `Schlegel et al., eLife (2017) <https://elifesciences.org/articles/16799>`_:

    .. math::

        f(i_{s},j_{k}) = \exp(\frac{-d^{2}_{sk}}{2\sigma^{2}}) \exp(\frac{|n(i_{s})-n(j_{k})|}{n(i_{s})+n(j_{k})})

    The synapse similarity score for neurons i and j being the average
    of :math:`f(i_{s},j_{k})` over all synapses s of i. Synapse k is the
    closest synapse of the same sign (pre/post) in neuron j to synapse s.
    :math:`d^{2}_{sk}` is the Euclidian distance between these distances.
    Variable :math:`\sigma` (``sigma``) determines what distance between
    s and k is considered "close". :math:`n(i_{s})` and :math:`n(j_{k})` are
    defined as the number of synapses of neuron i/j that are within given
    radius :math:`\omega` (``omega``) of synapse s and j, respectively (same
    sign only). This esnures that in cases of a strong disparity between
    :math:`n(i_{s})` and :math:`n(j_{k})`, the synapse similarity will be
    close to zero even if the distance between s and k is very small.


    Parameters
    ----------
    x :                 NeuronList
                        Neurons to compare. Must have connectors.
    sigma :             int | float
                        Distance between synapses that is considered to be
                        "close".
    omega :             int | float
                        Radius over which to calculate synapse density.
    mu_score :          bool
                        If True, score is calculated as mean between A->B and
                        B->A comparison.
    restrict_cn :       int | list | None
                        Restrict to given connector types. Must map to
                        a `type`, `relation` or `label` column in the
                        connector tables.
                        If None, will use all connector types. Use either
                        single integer or list. E.g. ``restrict_cn=[0, 1]``
                        to use only pre- and postsynapses.
    n_cores :           int
                        Number of parallel processes to use. Defaults to half
                        the available cores.

    Returns
    -------
    pandas.DataFrame

    See Also
    --------
    :func:`navis.synblast`
                        NBLAST variant using synapses.

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(5)
    >>> scores = navis.synapse_similarity(nl, omega=5000/8, sigma=2000/8)

    """
    if not isinstance(x, core.NeuronList):
        raise TypeError(f'Expected Neuronlist got {type(x)}')

    if any([not n.has_connectors for n in x]):
        raise ValueError('All neurons must have connector tables as .connectors property.')

    # If single value, turn into list
    if not isinstance(restrict_cn, type(None)):
        restrict_cn = utils.make_iterable(restrict_cn)

    combinations = [(nA.connectors, nB.connectors, sigma, omega, restrict_cn)
                    for nA in x for nB in x]

    with ProcessPoolExecutor(max_workers=n_cores) as e:
        futures = e.map(_unpack_synapse_helper, combinations, chunksize=1000)

        scores = [n for n in config.tqdm(futures, total=len(combinations),
                                         desc='Processing',
                                         disable=config.pbar_hide,
                                         leave=config.pbar_leave)]

    # Create empty score matrix
    sim_matrix = pd.DataFrame(np.zeros((len(x), len(x))),
                              index=x.id,
                              columns=x.id)
    # Populate matrix
    comb_names = [(nA.id, nB.id) for nA in x for nB in x]
    for c, v in zip(comb_names, scores):
        sim_matrix.loc[c[0], c[1]] = v

    if mu_score:
        sim_matrix = (sim_matrix + sim_matrix.T) / 2

    return sim_matrix


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
    neuron A: (1) the distance to the closest (Euclidian) synapse in neuron B
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
        this_cnA = cnA.loc[cnA.type == r, ['x', 'y', 'z']].values
        this_cnB = cnB.loc[cnB.type == r, ['x', 'y', 'z']].values

        # Skip if either neuronA or neuronB don't have this synapse type
        if not len(this_cnB):
            all_values += [0] * this_cnA.shape[0]
            continue

        # Create KDTrees
        treeA = ssp.cKDTree(this_cnA)
        treeB = ssp.cKDTree(this_cnB)

        # For each synapse in A, get closest synapse of same type in B
        closest_dist, closest_ix = treeB.query(this_cnA)

        # Check synapse density checking
        closeA = np.array([len(l) for l in treeA.query_ball_point(this_cnA, r=omega)])
        closeB = np.array([len(l) for l in treeB.query_ball_point(this_cnB, r=omega)])

        # Compute the score for these connectors
        part1 = np.exp(-1 * (np.abs(closeA - closeB[closest_ix]) / (closeA + closeB[closest_ix])))
        part2 = np.exp(-1 * (closest_dist**2) / (2 * sigma**2))
        all_values += (part1 * part2).tolist()

    return sum(all_values) / len(all_values)
