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

import warnings

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from scipy.spatial.distance import squareform, pdist

from .. import config, utils

logger = config.logger


def _fastcore_matrix(scores, func):
    """The score matrix as an array `func` can take, or `None` to use numpy.

    navis-fastcore needs a float16/32/64 buffer that is C- or F-contiguous. A
    DataFrame's `.values` is a view onto the underlying block as long as the
    frame has a single dtype - including *after* a transpose, which is what lets
    `extract_matches` serve `axis=1` without copying a matrix that may well be
    tens of GB.

    """
    if utils.fastcore is None or not hasattr(utils.fastcore, func):
        return None

    vals = scores.values
    if vals.dtype not in (np.float16, np.float32, np.float64):
        return None
    if not (vals.flags['C_CONTIGUOUS'] or vals.flags['F_CONTIGUOUS']):
        return None

    return vals


def extract_matches(scores, N=None, threshold=None, percentage=None,
                    axis=0, distances='auto', max_matches=None):
    """Extract top matches from score matrix.

    See `N`, `threshold` or `percentage` for the criterion.

    Parameters
    ----------
    scores :        pd.DataFrame
                    Score matrix (e.g. from [`navis.nblast`][]).
    N :             int
                    Number of matches to extract.
    threshold :     float
                    Extract all matches above a given threshold.
    percentage :    float [0-1]
                    Extract all matches within a given range of the top match.
                    E.g. `percentage=0.05` will return all matches within
                    5% of the top match.
    axis :          0 | 1
                    For which axis to produce matches.
    distances :     "auto" | bool
                    Whether `scores` is distances or similarities (i.e. whether
                    we need to look for the lowest instead of the highest values).
                    "auto" (default) will infer based on the diagonal of the
                    `scores` matrix. Use boolean to override.
    max_matches :   int, optional
                    Refuse to return more than this many matches in total. Only
                    applies to `threshold`/`percentage`, whose output size is not
                    known in advance - an over-broad cutoff on a large matrix can
                    otherwise take the machine down. The count is established
                    before anything is allocated.

    Returns
    -------
    pd.DataFrame
                    Note that the format is slightly different depending on
                    the criterion.

    Notes
    -----
    Uses [navis-fastcore](https://github.com/schlegelp/fastcore-rs) if available
    (~20-90x faster for `N`, ~4-10x for `percentage`) and numpy otherwise. The
    two agree except where scores *tie*, in which case which of the equal
    matches comes first is arbitrary and differs between the backends.

    `NaN`s are skipped, not ranked: a query scored against some targets but not
    others gets its best *valid* matches rather than a `NaN` match. If a query
    has fewer than `N` valid scores, the remaining `match_k`/`score_k` come back
    empty - `None` or `NaN` depending on your pandas version.

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

    # Transposing is easier than dealing with the different axes further down.
    # N.B. this is free: for a single-dtype frame `.T` is a view, and both
    # backends read a C- or F-contiguous buffer equally happily.
    if axis == 1:
        scores = scores.T

    if N is not None:
        if N > scores.shape[1]:
            raise ValueError(f'Cannot extract N={N} matches from only '
                             f'{scores.shape[1]} candidates.')
        return _extract_matches_n(scores,
                                  N=N,
                                  distances=distances)
    elif threshold is not None:
        return _extract_matches_threshold(scores,
                                          threshold=threshold,
                                          distances=distances,
                                          max_matches=max_matches)
    elif percentage is not None:
        return _extract_matches_perc(scores,
                                     perc=percentage,
                                     distances=distances,
                                     max_matches=max_matches)


def _extract_matches_n(scores, N=None, distances=False):
    """Return top N matches."""
    arr = _fastcore_matrix(scores, 'top_matches')
    if arr is not None:
        # `axis=0` because `extract_matches` has already transposed if needed
        top_n, top_scores = utils.fastcore.top_matches(arr, N, axis=0,
                                                       distances=distances)
        return _collate_matches_n(scores, top_n, top_scores, N)

    vals = scores.values
    top_n, top_scores = _top_n_numpy(vals, N, distances)

    if np.isnan(top_scores).any():
        # numpy sorts NaN to the end - i.e. ranks it as the *best* similarity -
        # so a single unscored pair would otherwise become that query's top
        # match. Take NaNs out of the running and re-run; the copy is only paid
        # for when there actually are NaNs. Queries left with fewer than N valid
        # scores come back as None/NaN, which is what fastcore returns too.
        sentinel = np.inf if distances else -np.inf
        top_n, top_scores = _top_n_numpy(np.where(np.isnan(vals), sentinel, vals),
                                         N, distances)
        invalid = top_scores == sentinel
        top_n = np.where(invalid, -1, top_n)
        top_scores = np.where(invalid, np.nan, top_scores)

    return _collate_matches_n(scores, top_n, top_scores, N)


def _top_n_numpy(vals, N, distances):
    """Top N values per row, best first. Returns (indices, scores)."""
    if not distances:
        if N > 1:
            # This partitions of the largest N values (faster than argsort)
            # Their correct order, however, is not guaranteed
            top_n = np.argpartition(vals, -N, axis=-1)[:, -N:]
        else:
            # For N=1 this is still faster
            top_n = np.argmax(vals, axis=-1).reshape(-1, 1)
    else:
        if N > 1:
            # N.B. `kth=N - 1` (not `N`): both partition the N smallest into the
            # first N slots, but `N` is out of bounds when N == vals.shape[1]
            top_n = np.argpartition(vals, N - 1, axis=-1)[:, :N]
        else:
            top_n = np.argmin(vals, axis=-1).reshape(-1, 1)

    # This make sure we order them properly
    rows = np.arange(len(vals)).reshape(-1, 1)
    top_scores = vals[rows, top_n]
    ind_ordered = np.argsort(top_scores, axis=1)

    if distances:
        ind_ordered = ind_ordered[:, ::-1]

    # Reverse into best-first, which is the order both backends collate from
    return (top_n[rows, ind_ordered][:, ::-1],
            top_scores[rows, ind_ordered][:, ::-1])


def _collate_matches_n(scores, top_n, top_scores, N):
    """Collate best-first (n_queries, N) index/score arrays into a table."""
    cols = scores.columns.values

    matches = pd.DataFrame()
    matches['id'] = scores.index.values
    for i in range(N):
        ix = top_n[:, i]
        match = cols[ix]
        # A query with fewer than N valid (non-NaN) scores gets -1 here, which
        # would otherwise quietly wrap around to the last column
        missing = ix < 0
        if missing.any():
            match = match.astype(object)
            match[missing] = None
        matches[f'match_{i + 1}'] = match
        matches[f'score_{i + 1}'] = top_scores[:, i]

    return matches


def _extract_matches_threshold(scores, threshold=.3, distances=False,
                               max_matches=None):
    """Extract all matches above a given threshold from score matrix."""
    arr = _fastcore_matrix(scores, 'matches_above')
    if arr is not None:
        offsets, cols, values = utils.fastcore.matches_above(
            arr, threshold=threshold, axis=0, distances=distances,
            max_matches=max_matches)
        ind = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    else:
        # N.B. NaNs compare False either way, so they are excluded here too
        if not distances:
            ind, cols = np.where(scores.values >= threshold)
        else:
            ind, cols = np.where(scores.values <= threshold)
        _check_max_matches(len(ind), max_matches)
        values = scores.values[ind, cols]

    matches = pd.DataFrame()
    matches['query'] = scores.index.values[ind]
    matches['match'] = scores.columns.values[cols]
    matches['score'] = values
    matches = matches.sort_values(['query', 'match', 'score']).set_index(['query', 'match'])

    return matches


def _extract_matches_perc(scores, perc=.05, distances=False, max_matches=None):
    """Extract all matches within a given percentage of the top match."""
    arr = _fastcore_matrix(scores, 'matches_above')
    if arr is not None:
        offsets, cols, values = utils.fastcore.matches_above(
            arr, percentage=perc, axis=0, distances=distances,
            max_matches=max_matches)
    else:
        # N.B. nanmax/nanmin: a query with *some* unscored pairs should still
        # get matches, rather than a NaN threshold that nothing can clear. One
        # with no valid scores at all still gets none - as it does on fastcore -
        # so the "All-NaN slice" warning that comes with it is just noise
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if not distances:
                thresh = np.nanmax(scores.values, axis=1)
                thresh = thresh - np.abs(thresh * perc)
                ind, cols = np.where(scores.values >= thresh.reshape(-1, 1))
            else:
                thresh = np.nanmin(scores.values, axis=1)
                thresh = thresh + np.abs(thresh * perc)
                ind, cols = np.where(scores.values <= thresh.reshape(-1, 1))
        _check_max_matches(len(ind), max_matches)

        # Sort each query's matches best first, and convert to the same
        # CSR-style (offsets, indices, values) layout fastcore returns
        values = scores.values[ind, cols]
        order = np.lexsort((values if distances else -values, ind))
        cols, values = cols[order], values[order]
        offsets = np.zeros(len(scores) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(np.bincount(ind, minlength=len(scores)))

    match_str = []
    scores_str = []
    cols = scores.columns.values[cols].astype(str)
    values = values.round(3).astype(str)
    for start, stop in zip(offsets[:-1], offsets[1:]):
        match_str.append(','.join(cols[start:stop]))
        scores_str.append(','.join(values[start:stop]))

    matches = pd.DataFrame()
    matches.index = scores.index
    matches['matches'] = match_str
    matches['scores'] = scores_str

    return matches


def _check_max_matches(n, max_matches):
    """Guard the numpy path, which has no equivalent of fastcore's own check."""
    if max_matches is not None and n > max_matches:
        raise ValueError(f'Criterion yields {n} matches, which exceeds '
                         f'`max_matches={max_matches}`.')


def update_scores(queries, targets, scores_ex, nblast_func, **kwargs):
    """Update score matrix by running only new query->target pairs.

    Parameters
    ----------
    queries :       Dotprops
    targets :       Dotprops
    scores_ex :     pandas.DataFrame
                    DataFrame with existing scores.
    nblast_func :   callable
                    The NBLAST to use. For example: `navis.nblast`.
    **kwargs
                    Argument passed to `nblast_func`.

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
    >>> bool(np.all(scores == scores2))
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
                    Method for `linkage`. Ignored if `x` is already a linkage.
    **kwargs
                    Keyword argument passed to scipy's `dendrogram`.

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

    This is a thin wrapper around `scipy.cluster.hierarchy.cut_tree` and
    `scipy.cluster.hierarchy.fcluster` functions.

    Parameters
    ----------
    x :             DataFrame | array
                    Pandas DataFrame is assumed to be NBLAST scores. Array is
                    assumed to be a linkage.
    t :             scalar
                    See `method`.
    criterion :     str
                    Method to use for creating clusters:
                     - `n_clusters` uses `cut_tree` to create `t` clusters
                     - `height` uses `cut_tree` to cut the dendrogram at
                        height `t`
                     - `inconsistent`, `distance`, `maxclust`, etc are passed
                       through to `fcluster`
    method :        str
                    Method for `linkage`. Ignored if `x` is already a linkage.
    **kwargs
                    Additional keyword arguments are passed through to the
                    cluster functions `cut_tree` and `fcluster`.

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
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            'Please install scikit-learn to use `nblast_prime`:\n'
            '  pip3 install scikit-learn -U'
            )

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