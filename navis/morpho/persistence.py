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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.

"""Module to generate and analyze persistence diagrams."""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import gaussian_kde
from typing import Union, Optional, Sequence, List, Dict, overload
from typing_extensions import Literal

from .. import utils, config, core, graph


# Setup logging
logger = config.get_logger(__name__)


@utils.map_neuronlist(desc='Calc. persistence', allow_parallel=True)
def persistence_points(x: 'core.NeuronObject',
                       descriptor: Union[
                                         Literal['root_dist']
                                         ] = 'root_dist',
                       remove_cbf: bool = False
                       ) -> pd.DataFrame:
    """Calculate points for a persistence diagram.

    Based on Li et al., PLoS One (2017). Briefly, this cuts the neuron into
    linear segments, the start (birth) and end (death) of which are assigned a
    value (see ``descriptor`` parameter). In combination, these points represent
    a fingerprint for the topology of the neuron.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | NeuronList
                Neuron(s) to calculate persistence poinst for. For MeshNeurons,
                we will use the skeleton produced by/associated with its
                ``.skeleton`` property.
    descriptor : 'root_dist'
                Descriptor function used to calculate birth and death "time" of
                the segments:
                  - ``root_dist`` distance from root
    remove_cbf : bool
                In unipolar neurons (e.g. in insects) the soma is separate and
                connects to the neuron's backbone via "cell body fiber" (CBF).
                The length of the CBF can vary quite a bit. Discounting the
                CBF can make the persistence points more stable.
                If ``remove_cbf=True`` and the neuron has a soma (!) we ignore
                the CBF for the birth & death times. Neurons will also be
                automatically be rooted onto their soma!

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> n.reroot(n.soma, inplace=True)
    >>> p = navis.persistence_points(n)

    References
    ----------
    Li Y, Wang D, Ascoli GA, Mitra P, Wang Y (2017) Metrics for comparing
    neuronal tree shapes based on persistent homology.
    PLOS ONE 12(8): e0182184. https://doi.org/10.1371/journal.pone.0182184

    """
    if descriptor not in ('root_dist', ):
        raise ValueError(f'Unknown "descriptor" parameter: {descriptor}')

    if isinstance(x, core.MeshNeuron):
        x = x.skeleton
    elif not isinstance(x, core.TreeNeuron):
        raise ValueError(f'Expected TreeNeuron(s), got "{type(x)}"')

    if remove_cbf and x.has_soma:
        # Reroot to soma
        x.reroot(x.soma, inplace=True)
        # Find the main branch point
        mbp = graph.find_main_branchpoint(x)

    # Generate segments
    segs = graph._generate_segments(x, weight='weight')

    # Grab starts and ends of each segment
    ends = np.array([s[0] for s in segs])
    starts = np.array([s[-1] for s in segs])

    if descriptor == 'root_dist':
        # Get geodesic distances to roots
        dist = graph.dist_to_root(x, weight='weight')
        death = np.array([dist[e] for e in ends])
        birth = np.array([dist[s] for s in starts])

        if remove_cbf and x.has_soma:
            # Subtract length of CBF
            cbf_length = graph.dist_between(x, mbp, x.soma)
            birth -= cbf_length
            death -= cbf_length

            # Drop segments that are entirely on the CBF
            starts = starts[death >= 0]
            ends = ends[death >= 0]
            birth = birth[death >= 0]
            death = death[death >= 0]

            # Clip negative births
            birth[birth < 0] = 0

    # Compile into a DataFrame
    pers = pd.DataFrame()
    pers['start_node'] = starts
    pers['end_node'] = ends
    pers['birth'] = birth
    pers['death'] = death

    return pers


def persistence_distances(q: 'core.NeuronObject',
                          t: Optional['core.NeuronObject'] = None,
                          augment: bool = True,
                          normalize: bool = True,
                          bw: float = .2,
                          **persistence_kwargs):
    """Calculate morphological similarity using persistence diagrams.

    This works by:
      1. Generate persistence points for each neuron.
      2. Create a weighted Gaussian from persistence points and sample 100
         evenly spaced points to create a feature vector.
      3. Calculate Euclidean distance.

    Parameters
    ----------
    q/t :       NeuronList
                Queries and targets, respectively. If ``t=None`` will run
                queries against queries. Neurons should have the same units,
                ideally nanometers.
    normalize : bool
                If True, will normalized the vector for each neuron to be within
                0-1. Set to False if the total number of linear segments matter.
    bw :        float
                Bandwidth for Gaussian kernel: larger = smoother, smaller =
                more detailed.
    augment :   bool
                Whether to augment the persistence vectors with other neuron
                properties (number of branch points & leafs and cable length).
    **persistence_kwargs
                Keyword arguments are passed to :func:`navis.persistence_points`.

    Returns
    -------
    distances : pandas.DataFrame

    See Also
    --------
    :func:`navis.persistence_points`
                The function to calculate the persistence points.
    :func:`navis.persistence_vectors`
                Use this to get and inspect the actual vectors used here.

    """
    q = core.NeuronList(q)
    all_n = q

    if t:
        t = core.NeuronList(t)
        all_n += t

    # Some sanity checks
    if len(all_n) <= 1:
        raise ValueError('Need more than one neuron.')

    soma_warn = False
    root_warn = False
    for n in all_n:
        if not soma_warn:
            if n.has_soma and n.soma not in n.root:
                soma_warn = True
        if not root_warn:
            if len(n.root) > 1:
                root_warn = True

        if root_warn and soma_warn:
            break

    if soma_warn:
        logger.warning('At least some neurons are not rooted to their soma.')
    if root_warn:
        logger.warning('At least some neurons are fragmented.')

    # Get persistence points for each skeleton
    pers = persistence_points(all_n, **persistence_kwargs)

    # Get the vectors
    vectors, samples = persistence_vectors(pers, samples=100, bw=bw)

    # Normalizing the vectors will produce more useful distances
    if normalize:
        vectors = vectors / vectors.max(axis=1).reshape(-1, 1)
    else:
        vectors = vectors / vectors.max()

    if augment:
        # Collect extra data. Note that this adds only 3 more to the existing
        # 100 observations
        vec_aug = np.vstack((all_n.cable_length,
                             all_n.n_leafs,
                             all_n.n_branches)).T

        # Normalize per metric
        vec_aug = vec_aug / vec_aug.max(axis=0)

        # If we wanted to weigh those observation equal to the 100 topology
        # observations:
        # vec_aug *= 100 / vec_aug.shape[1]

        vectors = np.append(vectors, vec_aug, axis=1)

    if t:
        # Extract source and target vectors
        q_vec = vectors[:len(q)]
        t_vec = vectors[len(q):]
        return pd.DataFrame(cdist(q_vec, t_vec), index=q.id, columns=t.id)
    else:
        return pd.DataFrame(squareform(pdist(vectors)), index=q.id, columns=q.id)


def persistence_vectors(x,
                        threshold: Optional[float] = None,
                        samples: int = 100,
                        bw: float = .2,
                        center: bool = False,
                        **kwargs):
    """Produce vectors from persistence points.

    Works by creating a Gaussian and sampling ``samples`` evenly spaced
    points across it.

    Parameters
    ----------
    x :         navis.NeuronList | pd.DataFrame | list thereof
                The persistence points (see :func:`navis.persistence_points`).
                For vectors for multiple neurons, provide either a list of
                persistence points DataFrames or a single DataFrame with a
                "neuron_id" column.
    threshold : float, optional
                If provided, segments shorter (death - birth) than this will not
                be used to create the Gaussian.
    samples :   int
                Number of points sampled across the Gaussian.
    bw :        float
                Bandwidth for Gaussian kernel: larger = smoother, smaller =
                more detailed.
    center :    bool
                Whether to center the individual curves on their highest value.
                This is done by "rolling" the axis (using ``np.roll``) which
                means that elements that roll beyond the last position are
                re-introduced at the first.

    Returns
    -------
    vectors :   np.ndarray
    samples :   np.ndarray
                Sampled distances. If ``center=True`` the absolute values don't
                make much sense anymore.

    References
    ----------
    Li Y, Wang D, Ascoli GA, Mitra P, Wang Y (2017) Metrics for comparing
    neuronal tree shapes based on persistent homology.
    PLOS ONE 12(8): e0182184. https://doi.org/10.1371/journal.pone.0182184

    See Also
    --------
    :func:`navis.persistence_points`
                The function to calculate the persistence points.
    :func:`navis.persistence_distances`
                Get distances based on (augmented) persistence vectors.

    """
    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)

    if isinstance(x, pd.DataFrame):
        pers = [x]
    elif isinstance(x, core.NeuronList):
        pers = [persistence_points(n, **kwargs) for n in x]
    elif isinstance(x, list):
        if not all([isinstance(l, pd.DataFrame) for l in x]):
            raise ValueError('Expected lists to contain only DataFrames')
        pers = x
    else:
        raise TypeError('Unable to work extract persistence vectors from data '
                        f'of type "{x}"')

    # Get the max distance
    max_pdist = max([p.birth.max() for p in pers])
    samples = np.linspace(0, max_pdist * 1.05, samples)

    # Now get a persistence vector
    vectors = []
    for p in pers:
        weights = p.death.values - p.birth.values
        if threshold:
            p = p.loc[weights >= threshold]
            weights = weights[weights >= threshold]

        # For each persistence generate a weighted Gaussian kernel
        kernel = gaussian_kde(p.birth.values,
                              weights=weights,
                              bw_method=bw)

        # And sample probabilities at the sample points
        vectors.append(kernel(samples))
    vectors = np.array(vectors)

    if center:
        # Shift each vector such that the highest value lies in the center.
        # Note that we are "rolling" the array which means that elements that
        # drop off to the right are reintroduced on the left
        for i in range(len(vectors)):
            vectors[i] = np.roll(vectors[i],
                                 -np.argmax(vectors[i]) + len(samples) // 2)

    return vectors, samples


def persistence_diagram(pers, ax=None, **kwargs):
    """Plot a persistence diagram.

    Parameters
    ----------
    pers :      pd.DataFrame
                Persistent points from :func:`navis.persistence_points`.
    ax :        matplotlib ax, optional
                Ax to plot on.
    **kwargs
                Keyword arguments are passed to `LineCollection`.

    Returns
    -------
    ax :        matplotlib ax

    """
    if not isinstance(pers, pd.DataFrame):
        raise TypeError(f'Expected DataFrame, got "{type(pers)}"')

    if not ax:
        fig, ax = plt.subplots()

    segs = []
    for i, (b, d) in enumerate(zip(pers.birth.values, pers.death.values)):
        segs.append([[b, i], [d, i]])
    lc = LineCollection(segs, **kwargs)
    ax.add_collection(lc)

    ax.set_xlim(-5, pers.death.max())
    ax.set_ylim(-5, pers.shape[0])

    ax.set_ylabel('segments')
    ax.set_xlabel('time')

    return ax


def persistence_vector_plot(x, normalize=True, ax=None,
                            persistence_kwargs={}, vector_kwargs={}):
    """Plot persistence vectors.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | NeuronList
                Neuron(s) to calculate persistence points for. For MeshNeurons,
                we will use the skeleton produced by/associated with its
                ``.skeleton`` property.

    Returns
    -------
    ax

    """
    if not isinstance(x, core.NeuronList):
        x = core.NeuronList(x)

    # Get persistence points for each skeleton
    pers = persistence_points(x, **persistence_kwargs)

    # Get the vectors
    vectors, samples = persistence_vectors(pers, **vector_kwargs)

    # Normalizing the vectors will produce more useful distances
    if normalize:
        vectors = vectors / vectors.max(axis=1).reshape(-1, 1)
    else:
        vectors = vectors / vectors.max()

    if not ax:
        fig, ax = plt.subplots()

    for n, v in zip(x, vectors):
        ax.plot(samples, v, label=n.label)

    return ax
