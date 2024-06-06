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

"""Module containing utility functions for BLASTING."""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from scipy.spatial.distance import squareform, pdist

from .. import config

logger = config.logger


def extract_matches(scores, N=None, threshold=None, percentage=None,
                    axis=0, distances='auto'):
    """Extract top matches from score matrix.

    See `N`, `threshold` or `percentage` for the criterion.

    Parameters
    ----------
    scores :        pd.DataFrame
                    Score matrix (e.g. from :func:`navis.nblast`).
    N :             int
                    Number of matches to extract.
    threshold :     float
                    Extract all matches above a given threshold.
    percentage :    float [0-1]
                    Extract all matches within a given range of the top match.
                    E.g. `percentage=0.05` will return all matches within
                    5% of the top match.
    single_cols :   bool
                    If True will return single columns with comma-separated
                    strings for match ID and match score, respectively.
    axis :          0 | 1
                    For which axis to produce matches.
    distances :     "auto" | bool
                    Whether `scores` is distances or similarities (i.e. whether
                    we need to look for the lowest instead of the highest values).
                    "auto" (default) will infer based on the diagonal of the
                    `scores` matrix. Use boolean to override.

    Returns
    -------
    pd.DataFrame
                    Note that the format is slightly different depending on
                    the criterion.

    """
    assert axis in (0, 1), '`axis` must be 0 or 1'

    if N is None and threshold is None and percentage is None:
        raise ValueError('Must provide either `N` or `threshold` or '
                         '`percentage` as criterion for match extraction.')
    elif len({N, threshold, percentage}) > 2:
        # We expect {criterion, None}
        raise ValueError('Please provide either `N`, `threshold` or '
                         '`percentage` as criterion for match extraction.')

    if distances == 'auto':
        distances = True if most(np.diag(scores.values).round(2) == 0) else False

    # Transposing is easier than dealing with the different axes further down
    if axis == 1:
        scores = scores.T

    if N is not None:
        return _extract_matches_n(scores,
                                  N=N,
                                  distances=distances)
    elif threshold is not None:
        return _extract_matches_threshold(scores,
                                          threshold=threshold,
                                          distances=distances)
    elif percentage is not None:
        return _extract_matches_perc(scores,
                                     perc=percentage,
                                     distances=distances)


def _extract_matches_n(scores, N=None, distances=False):
    """Return top N matches."""
    if not distances:
        if N > 1:
            # This partitions of the largest N values (faster than argsort)
            # Their correct order, however, is not guaranteed
            top_n = np.argpartition(scores.values, -N, axis=-1)[:, -N:]
        else:
            # For N=1 this is still faster
            top_n = np.argmax(scores.values, axis=-1).reshape(-1, 1)
    else:
        if N > 1:
            top_n = np.argpartition(scores.values, N, axis=-1)[:, :N]
        else:
            top_n = np.argmin(scores.values, axis=-1).reshape(-1, 1)

    # This make sure we order them properly
    top_scores = scores.values[np.arange(len(scores)).reshape(-1, 1), top_n]
    ind_ordered = np.argsort(top_scores, axis=1)

    if distances:
        ind_ordered = ind_ordered[:, ::-1]

    top_n = top_n[np.arange(len(top_n)).reshape(-1, 1), ind_ordered]
    top_scores = top_scores[np.arange(len(top_scores)).reshape(-1, 1), ind_ordered]

    # Now collate matches
    matches = pd.DataFrame()
    matches['id'] = scores.index.values
    for i in range(N):
        matches[f'match_{i + 1}'] = scores.columns[top_n[:, -(i + 1)]]
        matches[f'score_{i + 1}'] = top_scores[:, -(i + 1)]

    return matches


def _extract_matches_threshold(scores, threshold=.3, distances=False):
    """Extract all matches above a given threshold from score matrix."""
    if not distances:
        ind, cols = np.where(scores.values >= threshold)
    else:
        ind, cols = np.where(scores.values <= threshold)

    matches = pd.DataFrame()
    matches['query'] = scores.index[ind]
    matches['match'] = scores.columns[cols]
    matches['score'] = scores.values[ind, cols]
    matches = matches.sort_values(['query', 'match', 'score']).set_index(['query', 'match'])

    return matches


def _extract_matches_perc(scores, perc=.05, distances=False):
    """Extract all matches within a given percentage of the top match."""
    if not distances:
        thresh = np.max(scores.values, axis=1)
        thresh = thresh - np.abs(thresh * perc)
        ind, cols = np.where(scores.values >= thresh.reshape(-1, 1))
    else:
        thresh = np.min(scores.values, axis=1)
        thresh = thresh + np.abs(thresh * perc)
        ind, cols = np.where(scores.values <= thresh.reshape(-1, 1))

    matches = pd.DataFrame()
    matches.index = scores.index

    match_str = []
    scores_str = []
    for i in range(len(matches)):
        this = cols[ind == i]
        sc = scores.values[i, this]
        srt = np.argsort(sc)[::-1]
        m = scores.columns[this][srt]
        sc = sc[srt]

        match_str.append(','.join(m.astype(str)))
        scores_str.append(','.join(sc.round(3).astype(str)))

    matches['matches'] = match_str
    matches['scores'] = scores_str

    return matches


def update_scores(queries, targets, scores_ex, nblast_func, **kwargs):
    """Update score matrix by running only new query->target pairs.

    Parameters
    ----------
    queries :       Dotprops
    targets :       Dotprops
    scores_ex :     pandas.DataFrame
                    DataFrame with existing scores.
    nblast_func :   callable
                    The NBLAST to use. For example: ``navis.nblast``.
    **kwargs
                    Argument passed to ``nblast_func``.

    Returns
    -------
    pandas.DataFrame
                    Updated scores.

    Examples
    --------

    Mostly for testing but also illustrates the principle:

    >>> import navis
    >>> import numpy as np
    >>> nl = navis.example_neurons(n=5)
    >>> dp = navis.make_dotprops(nl, k=5) / 125
    >>> # Full NBLAST
    >>> scores = navis.nblast(dp, dp, n_cores=1)
    >>> # Subset and fill in
    >>> scores2 = navis.nbl.update_scores(dp, dp,
    ...                                   scores_ex=scores.iloc[:3, 2:],
    ...                                   nblast_func=navis.nblast,
    ...                                   n_cores=1)
    >>> np.all(scores == scores2)
    True

    """
    if not callable(nblast_func):
        raise TypeError('`nblast_func` must be callable.')
    # The np.isin query is much faster if we force any strings to <U18 by
    # converting to arrays
    is_new_q = ~np.isin(queries.id, np.array(scores_ex.index))
    is_new_t = ~np.isin(targets.id, np.array(scores_ex.columns))

    logger.info(f'Found {is_new_q.sum()} new queries and '
                f'{is_new_t.sum()} new targets.')

    # Reindex old scores
    scores = scores_ex.reindex(index=queries.id, columns=targets.id).copy()

    # NBLAST new queries against all targets
    if 'precision' not in kwargs:
        kwargs['precision'] = scores.values.dtype

    if any(is_new_q):
        logger.info(f'Updating new queries -> targets scores')
        qt = nblast_func(queries[is_new_q], targets, **kwargs)
        scores.loc[qt.index, qt.columns] = qt.values

    # NBLAST all old queries against new targets
    if any(is_new_t):
        logger.info(f'Updating old queries -> new targets scores')
        tq = nblast_func(queries[~is_new_q], targets[is_new_t], **kwargs)
        scores.loc[tq.index, tq.columns] = tq.values

    return scores


def compress_scores(scores, threshold=None, digits=None):
    """Compress scores.

    This will not necessarily reduce the in-memory footprint but will lead to
    much smaller file sizes when saved to disk.

    Parameters
    ----------
    scores :        pandas.DataFrame
    threshold :     float, optional
                    Scores lower than this will be capped at `threshold`.
    digits :        int, optional
                    Round scores to the Nth digit.

    Returns
    -------
    scores_comp :   pandas.DataFrame
                    Copy of the original dataframe with the data cast to 32bit
                    floats and the optional filters (see `threshold` and
                    `digits`) applied.

    """
    scores = scores.astype(np.float32)
    if digits is not None:
        scores = scores.round(digits)
    if threshold is not None:
        scores.clip(lower=threshold, inplace=True)
    return scores


def make_linkage(x, method='single', optimal_ordering=False):
    """Make linkage from input. If input looks like linkage it is passed through."""
    if isinstance(x, pd.DataFrame):
        # Make sure it is symmetric
        if x.shape[0] != x.shape[1]:
            raise ValueError(f'Scores must be symmetric, got shape {x.shape}')
        # A cheap check for whether these are mean scores
        if any(x.values[0].round(5) != x.values[:, 0].round(5)):
            logger.warning(f'Symmetrizing scores because they do not look like mean scores!')
            x = (x + x.values.T) / 2

        dists = squareform(1 - x.values, checks=False)
        Z = sch.linkage(dists, method=method, optimal_ordering=optimal_ordering)
    elif isinstance(x, np.ndarray):
        Z = x
    else:
        raise TypeError(f'Expected scores) (DataFrame) or linkage (array), got {type(x)}')

    return Z


def dendrogram(x, method='ward', **kwargs):
    """Plot dendrogram.

    This is just a convenient thin wrapper around scipy's dendrogram function
    that lets you feed NBLAST scores directly. Note that this causes some
    overhead for very large NBLASTs.

    Parameters
    ----------
    x :             DataFrame | array
                    Pandas DataFrame is assumed to be NBLAST scores. Array is
                    assumed to be a linkage.
    method :        str
                    Method for ``linkage``. Ignored if ``x`` is already a linkage.
    **kwargs
                    Keyword argument passed to scipy's ``dendrogram``.

    Returns
    -------
    dendrogram

    """
    # Some sensible defaults that help with large dendrograms
    DEFAULTS = dict(no_labels=True,
                    labels=x.index.values.astype(str) if isinstance(x, pd.DataFrame) else None)
    DEFAULTS.update(kwargs)

    # Make linkage
    Z = make_linkage(x, method=method)

    return sch.dendrogram(Z, **DEFAULTS)


def make_clusters(x, t, criterion='n_clusters', method='ward', **kwargs):
    """Form flat clusters.

    This is a thin wrapper around ``scipy.cluster.hierarchy.cut_tree`` and
    ``scipy.cluster.hierarchy.fcluster`` functions.

    Parameters
    ----------
    x :             DataFrame | array
                    Pandas DataFrame is assumed to be NBLAST scores. Array is
                    assumed to be a linkage.
    t :             scalar
                    See ``method``.
    criterion :     str
                    Method to use for creating clusters:
                     - `n_clusters` uses ``cut_tree`` to create ``t`` clusters
                     - `height` uses ``cut_tree`` to cut the dendrogram at
                        height ``t``
                     - `inconsistent`, `distance`, `maxclust`, etc are passed
                       through to ``fcluster``
    method :        str
                    Method for ``linkage``. Ignored if ``x`` is already a linkage.
    **kwargs
                    Additional keyword arguments are passed through to the
                    cluster functions ``cut_tree`` and ``fcluster``.

    Returns
    -------
    clusters :      np.ndarray

    """
    # Make linkage
    Z = make_linkage(x, method=method)

    if criterion == 'n_clusters':
        cl = sch.cut_tree(Z, n_clusters=t, **kwargs).flatten()
    elif criterion == 'height':
        cl = sch.cut_tree(Z, height=t, **kwargs).flatten()
    else:
        cl = sch.fcluster(Z, t=t, criterion=criterion, **kwargs)

    return cl


def nblast_prime(scores, n_dim=.2, metric='euclidean'):
    """Generate a smoothed version of the NBLAST scores.

    In brief:
     1. Run PCA on the NBLAST scores and extract the first N components.
     2. From that calulate a new similarity matrix.

    Requires scikit-learn.

    Parameters
    ----------
    scores :    pandas.DataFrame
                The all-by-all NBLAST scores.
    n_dim :     float | int
                The number of dimensions to use. If float (0 < n_dim < 1) will
                use `scores.shape[0] * n_dim`.
    metric :    str
                Which distance metric to use. Directly passed through to the
                `scipy.spatial.distance.pdist` function.

    Returns
    -------
    scores_new

    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError('Please install scikit-learn to use `nblast_prime`:\n'
                          '  pip3 install scikit-learn -U')

    if not isinstance(scores, pd.DataFrame):
        raise TypeError(f'`scores` must be pandas DataFrame, got "{type(scores)}"')

    if (scores.shape[0] != scores.shape[1]) or ~np.all(scores.columns == scores.index):
        logger.warning('NBLAST matrix is not symmetric - are you sure this is '
                       'an all-by-all matrix?')

    if n_dim < 1:
        n_dim = int(scores.shape[1] * n_dim)

    pca = PCA(n_components=n_dim)
    X_new = pca.fit_transform(scores.values)

    dist = pdist(X_new, metric=metric)

    return pd.DataFrame(1 - squareform(dist), index=scores.index, columns=scores.columns)


def most(x, f=.9):
    """Check if most (as opposed to all) entries are True."""
    if x.sum() >= (x.shape[0] * f):
        return True
    return False