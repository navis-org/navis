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

import numpy as np

from functools import partial
from inspect import signature

from .. import core, utils, config

logger = config.logger

# The alignment methods, in one place: `align_pairwise` validates against these and
# `navis.nblast_align` resolves its `align_method` through `_align_func` below.
ALIGN_METHODS = ('rigid', 'deform', 'rigid+deform', 'pca')


def _import_rcpd():
    """Import `rcpd`, with a pointer to the install if it is not there."""
    try:
        import rcpd
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            'Alignment by coherent point drift requires the `rcpd` library:\n'
            '  pip3 install rcpd -U'
            )
    return rcpd


def align_pairwise(x, y=None, method='rigid', sample=None, progress=True, **kwargs):
    """Run a pairwise alignment between given neurons.

    The coherent point drift methods ("rigid", "deform" and "rigid+deform")
    require the `rcpd` library; "pca" requires `scikit-learn`.

    Parameters
    ----------
    x :         navis.NeuronList
                Neurons to align to other neurons.
    y :         navis.NeuronList, optional
                The neurons to align to. If `None`, will run pairwise
                alignment of `x` vs `x`.
    method :    "rigid" | "deform" | "rigid+deform" | "pca"
                Which method to use for alignment. Maps to the respective
                `navis.align_{method}` function. "rigid+deform" performs a
                rigid followed by a warping alignment. The three coherent point
                drift methods fit the whole grid in one go, spread over all
                cores.
    sample :    float [0-1], optional
                If provided, will calculate the registration on only the given
                fraction of points. The rest are then moved by the fitted
                transform itself, which is exact - both registrations are
                functions of position rather than sets of moved points. Use this
                to speed things up.
    **kwargs
                Keyword arguments are passed through to the respective
                alignment function. For "rigid+deform" each goes to the fit that
                takes it, and the ones both take (`w`, `tolerance`,
                `max_iterations`) go to both.

    Returns
    -------
    np.ndarray
                Array of shape (x, y) with the pairwise-aligned neurons.

    See Also
    --------
    [`navis.nblast_align`][]
                Runs an NBLAST where neurons are first aligned pairwise.

    Examples
    --------
    >>> import navis
    >>> nl = navis.example_neurons(2, kind='skeleton')
    >>> aligned = navis.align.align_pairwise(nl, method='rigid', sample=.2)

    """
    if y is None:
        y = x

    utils.eval_param(x, name='x', allowed_types=(core.NeuronList, ))
    utils.eval_param(y, name='y', allowed_types=(core.NeuronList, ))
    utils.eval_param(method, name='method', allowed_values=ALIGN_METHODS)

    if method != 'pca':
        return _align_pairwise_cpd(x, y, method=method, sample=sample,
                                   progress=progress, **kwargs)

    # A PCA alignment has no target - each neuron is turned onto its own principal
    # axis - so it is fitted once per neuron rather than once per pair, and every
    # column of the grid is the same neuron.
    xf, _ = align_pca(x, **kwargs)

    return np.array([[n] * len(y) for n in xf])


def _align_pairwise_cpd(x, y, method, sample=None, progress=True, n_cores=None,
                        **kwargs):
    """`align_pairwise` for the coherent point drift methods, fitted in one go.

    A grid of registrations is embarrassingly parallel - no pair has any setup to
    share with any other - so it is handed to `rcpd` whole rather than fitted a pair
    at a time from Python; see `_register` for how it is spread over the cores.
    Because only the transforms come back, peak memory does not grow with the number
    of pairs the way it would if each fit returned its moved points.
    """
    x_co = [_subsample(_extract_coords(n), sample) for n in x]
    y_co = [_subsample(_extract_coords(n), sample) for n in y]

    # A neuron aligned to itself is the identity; skip the fit and reuse it.
    pairs = [(i, j) for i in range(len(x)) for j in range(len(y))
             if x[i] is not y[j]]

    chains = _fit_pairs(method, x_co, y_co, pairs, n_cores=n_cores,
                        progress=progress and len(pairs) > 1, **kwargs)
    by_pair = dict(zip(pairs, chains))

    aligned = []
    for i, n1 in enumerate(x):
        aligned.append([])
        # Hoisted: the same for every column, and on a skeleton each call rebuilds
        # the array from the node table - `n * m` conversions instead of `n`.
        co = _extract_coords(n1)
        for j, n2 in enumerate(y):
            if n1 is n2:
                aligned[-1].append(n1)
                continue
            xf = n1.copy()
            _set_coords(xf, _apply_chain(by_pair[(i, j)], co))
            aligned[-1].append(xf)

    return np.array(aligned)


def _align_func(method):
    """The align-a-NeuronList-onto-one-target callable for `method`.

    One table, read by both `align_pairwise` and [`navis.nblast_align`][], so that
    the method names have a single owner and an unknown one gets the same message
    either way. Every entry takes `(x, target=, sample=, progress=, **kwargs)` and
    returns the aligned neurons plus the transforms that moved them; `align_pca`
    needs `_align_pca_to_target` to fit that shape, having no target to align to.

    Examples
    --------
    # For doctests
    >>> import navis
    >>> n1, n2 = navis.example_neurons(2, kind='skeleton')
    >>> align = navis.transforms.align._align_func('rigid+deform')
    >>> n1_aligned, regs = align(n1, target=n2, sample=.2, progress=False)

    """
    utils.eval_param(method, name='method', allowed_values=ALIGN_METHODS)

    if method == 'pca':
        return _align_pca_to_target

    return partial(_align_to_target, method=method)


def align_rigid(x, target=None, scale=False, scale_bounds=None, w=0,
                tolerance=1e-6, max_iterations=200, mirror_axis=None,
                verbose=False, sample=None, progress=True, n_cores=None):
    """Align neurons using a rigid registration.

    Uses coherent point drift, as implemented in [rcpd](https://github.com/schlegelp/rcpd).
    Neurons are registered in parallel: one registration per core across a
    `NeuronList`, and all cores on a single pair.

    Parameters
    ----------
    x :                 navis.NeuronList
                        Neurons to align.
    target :            navis.Neuron | np.ndarray
                        The neuron that all neurons in `x` will be aligned to.
                        If `None`, neurons will be aligned to the first neuron in `x`!
    scale :             bool
                        If True, will also fit a uniform scale factor.
    scale_bounds :      (float, float), optional
                        Lower and upper limit for the fitted `scale`, e.g.
                        `(0.8, 1.25)` to allow a quarter either way. The limits
                        are imposed at every step of the fit, so what comes back
                        is the best alignment *within* them rather than a free fit
                        squashed into range afterwards; a pair that wants more
                        comes back sitting exactly on the nearer limit. Requires
                        `scale=True`.
    w :                 float [0-1)
                        `w` is used to account for outliers: higher w = more forgiving
                        of points with no counterpart in the target. The default of 0
                        fits every point.
    tolerance :         float
                        Stop once the fitted variance changes by less than this
                        *fraction* of itself.
    max_iterations :    int
                        Cap on EM iterations.
    mirror_axis :       'x' | 'y' | 'z' | 0 | 1 | 2, optional
                        If provided, each neuron is fitted twice - as given and mirrored
                        along this axis - and whichever of the two left the smaller
                        residual (`RigidTransform.rms`) is the one returned. Use this
                        when the neurons may be of either handedness, e.g. aligning
                        left-hand-side neurons onto a right-hand-side target without
                        knowing which is which. Where the mirrored fit wins, the
                        reflection is folded into the returned transform, so
                        `np.linalg.det(reg.rotation) < 0` tells you which neurons were
                        flipped; `.apply()`, `.matrix` and `~` all still hold. Where the
                        mirror plane sits does not matter - the fit is free to translate.
    verbose :           bool
                        Whether to report registrations that hit `max_iterations`
                        without converging.
    sample :            float [0-1], optional
                        If provided, will calculate the registration on only the given
                        fraction of points. Use this to speed things up. The
                        remaining points are *not* approximated: a rigid transform
                        is closed-form, so it applies exactly to points that took
                        no part in fitting it.
    progress :          bool
                        Whether to show a progress bar.
    n_cores :           int, optional
                        Number of cores to use. Defaults to all available.

    Returns
    -------
    xf :    navis.NeuronList
            The aligned neurons.
    regs :  list of rcpd.RigidTransform
            The fitted transforms, one per neuron in `x` and in the same order.
            Each carries `.scale`, `.rotation`, `.translation` and `.matrix` (a
            4x4 homogeneous matrix), plus the `.rms` / `.nrms` residuals and the
            `.sigma2`, `.iterations` and `.converged` diagnostics. Apply one to
            further points with `.apply(coords)`, or invert it with `~`.

    Notes
    -----
    Converging and converging onto the right shape are not the same thing:
    coherent point drift will settle a neuron onto an unrelated one quite happily.
    `regs[i].nrms` - the residual as a fraction of the target's own radius - is
    what says whether the alignment is any good, and being dimensionless it can be
    compared between pairs. Two different neurons of the same type land around
    0.06-0.11, two unrelated ones around 0.2.

    `mirror_axis` doubles the number of registrations, but both fits for a neuron are
    dispatched together with everything else, so given spare cores it costs less than
    twice the wall-clock. One caveat on how the winner is picked: with `w > 0` the
    residual is measured
    over the matched mass alone, so a fit that writes more points off as outliers is
    not penalised for it. Compare `regs[i].rms` against the alternative yourself if
    that matters - with `w=0` (the default) every point counts and it does not.

    Examples
    --------
    >>> import navis
    >>> n1, n2 = navis.example_neurons(2, kind='skeleton')
    >>> n1_aligned, regs = navis.align.align_rigid(n1, n2, sample=.2)
    >>> regs[0].converged
    True
    >>> # Let the alignment flip the neuron over if that fits better
    >>> import numpy as np
    >>> xf, regs = navis.align.align_rigid(n1, n2, mirror_axis='x', sample=.2)
    >>> bool(np.linalg.det(regs[0].rotation) < 0)  # was it mirrored?
    False

    """
    xf, chains = _align_to_target(
        x, target, 'rigid', sample=sample, mirror_axis=mirror_axis, verbose=verbose,
        progress=progress, n_cores=n_cores,
        scale=scale, scale_bounds=scale_bounds, w=w, tolerance=tolerance,
        max_iterations=max_iterations,
        )

    return xf, [chain[0] for chain in chains]


def align_deform(x, target=None, alpha=2.0, beta=0.2, w=0, tolerance=1e-6,
                 max_iterations=200, num_modes=100, verbose=False, sample=None,
                 progress=True, n_cores=None):
    """Align neurons using a deformable registration.

    Uses coherent point drift, as implemented in [rcpd](https://github.com/schlegelp/rcpd).
    Neurons are registered in parallel: one registration per core across a
    `NeuronList`, and all cores on a single pair.

    A deformation is free to map anything onto anything, so it is worth running a
    rigid alignment first and asking this only for what is left: use
    `align_pairwise(method="rigid+deform")`, or fit this on the points
    [`navis.align.align_rigid`][] moved. The motion coherence prior pays for
    displacement, so a pose offset left for the deformation to undo comes out of
    the same budget as the warp.

    Parameters
    ----------
    x :                 navis.NeuronList
                        Neurons to align.
    target :            navis.Neuron | np.ndarray
                        The neuron that all neurons in `x` will be aligned to.
                        If `None`, neurons will be aligned to the first neuron in `x`!
    alpha :             float
                        Weight of the motion coherence prior, relative to the data
                        term: higher = stiffer. The useful range is wide and
                        data-dependent, so treat the default as a starting point.
    beta :              float
                        Width of the kernel that makes neighbouring points move
                        together, as a **fraction of the neuron's radius** - so it
                        means the same thing whatever units the neurons are in.
                        Higher = stiffer. Coupled to `num_modes`, see Notes.
    w :                 float [0-1)
                        `w` is used to account for outliers: higher w = more forgiving
                        of points with no counterpart in the target. The default of 0
                        fits every point.
    tolerance :         float
                        Stop once the fitted variance changes by less than this
                        *fraction* of itself.
    max_iterations :    int
                        Cap on EM iterations.
    num_modes :         int
                        Number of eigenmodes of the kernel to solve the deformation
                        through. See Notes.
    verbose :           bool
                        Whether to report registrations that hit `max_iterations`
                        without converging.
    sample :            float [0-1], optional
                        If provided, will calculate the registration on only the given
                        fraction of points. Use this to speed things up. The
                        remaining points are moved by the same fitted deformation
                        as the rest - it is a function of position, not a set of
                        moved points, so there is no landmark transform involved.
    progress :          bool
                        Whether to show a progress bar.
    n_cores :           int, optional
                        Number of cores to use. Defaults to all available.

    Returns
    -------
    xf :    navis.NeuronList
            The aligned neurons.
    regs :  list of rcpd.DeformTransform
            The fitted deformations, one per neuron in `x` and in the same order.
            Move further points with `.apply(coords)`, and check `.nrms` (the
            residual as a fraction of the target's radius, directly comparable
            with a rigid fit's) before trusting one. Note there is no inverse: a
            sum of Gaussians has none in closed form.

    Notes
    -----
    **`beta` defaults lower here than in `rcpd`** (0.2 against 1), because a neuron is
    a specific enough object to tune for: at `beta=1` the kernel spans the whole
    arbor, everything moves as one and the result is not far off a rigid alignment.
    Over the ten pairs of example skeletons, mean nearest-neighbour distance between
    an aligned neuron and its target (as a fraction of the target's extent) is 0.0089
    unaligned, 0.0065 at `beta=1` and 0.0043 at `beta=0.2`, the last for a third more
    time. `alpha` barely matters over 0.5-10 on the same neurons, so it is left at
    `rcpd`'s default.

    **`beta` and `num_modes` are coupled.** The kernel's rank grows roughly as
    `(1 / beta) ** 3`, so halving `beta` needs about eight times as many modes to
    represent the same deformation. Too few is not a subtle failure: the part of the
    warp that does not fit in the retained subspace is projected away, so what comes
    back is smoother than `beta` alone would suggest. The check is to raise
    `num_modes` and see whether the answer moves - at the defaults here it does not
    (0.0043 either way at four times the modes and five times the time), but that is
    a statement about these neurons, not a guarantee.

    **Deformable fits routinely hit `max_iterations`**, so `converged=False` is
    common and not by itself a problem: allowed to run to a stop the same fits take
    ~450 iterations to leave a residual 5% smaller, for twice the time. A fit *can* also
    report `converged` without having met `tolerance` - a deformation has enough
    freedom to drive the residual below what the arithmetic can resolve, and there
    the fitted variance stops descending and rattles around a value it has already
    reached, which `rcpd` stops rather than let run out the cap.

    Examples
    --------
    >>> import navis
    >>> n1, n2 = navis.example_neurons(2, kind='skeleton')
    >>> n1_aligned, regs = navis.align.align_deform(n1, n2, sample=.2)

    """
    xf, chains = _align_to_target(
        x, target, 'deform', sample=sample, verbose=verbose, progress=progress,
        n_cores=n_cores,
        alpha=alpha, beta=beta, w=w, tolerance=tolerance,
        max_iterations=max_iterations, num_modes=num_modes,
        )

    return xf, [chain[0] for chain in chains]


def align_pca(x, individually=True):
    """Align neurons along their first principal components.

    This will in effect turn the neurons into a 1-dimensional line.
    Requires the `scikit-learn` library.

    Parameters
    ----------
    x :             navis.NeuronList | np.ndarray
                    The neurons to align.
    individually :  bool
                    Whether to align neurons along their individual or
                    collective first principical component.

    Returns
    -------
    xf :    navis.NeuronList
            The PCA-aligned neurons.
    pcas :  list
            The scikit-learn PCA object(s)

    Examples
    --------
    >>> import navis
    >>> n1, n2 = navis.example_neurons(2, kind='skeleton')
    >>> n1_aligned, pcas = navis.align.align_pca(n1, n2)  # doctest: +SKIP

    """
    try:
        from sklearn.decomposition import PCA
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            '`align_pca()` requires the `scikit-learn` library:\n'
            '  pip3 install scikit-learn -U'
            )

    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)

    assert isinstance(x, core.NeuronList)

    pcas = []
    if not individually:
        # Collect coordinates
        co = [_extract_coords(n) for n in x]
        n_points = [len(c) for c in co]  # track how many points per neuron
        co = np.vstack(co)

        pca = PCA(n_components=1)
        co_new = pca.fit_transform(X=co)

        xf = x.copy()
        i = 0
        for n, le in zip(xf, n_points):
            _set_coords(n, co_new[i: i + le])
            i += le
        pcas.append(pca)
    else:
        xf = x.copy()
        for n in xf:
            pca = PCA(n_components=1)
            _set_coords(n, pca.fit_transform(X=_extract_coords(n)))
            pcas.append(pca)
    return xf, pcas


def _align_pca_to_target(x, target=None, sample=None, progress=True, **kwargs):
    """`align_pca` under the same calling convention as the other alignments.

    A PCA alignment has no target and nothing to subsample - each neuron is turned
    onto its own principal axis - so those arguments are accepted and ignored, which
    is what lets `nblast_align` treat all four methods alike.

    Examples
    --------
    # For doctests
    >>> import navis
    >>> n1, n2 = navis.example_neurons(2, kind='skeleton')
    >>> n1_aligned, pcas = navis.transforms.align._align_pca_to_target(n1, n2)  # doctest: +SKIP

    """
    return align_pca(x, **kwargs)


def _align_to_target(x, target, method, sample=None, mirror_axis=None,
                     verbose=False, progress=True, n_cores=None, **kwargs):
    """Align every neuron in `x` onto `target`, fitting the whole list in one go.

    Shared by `align_rigid`, `align_deform` and the "rigid+deform" entry of
    `_align_func`, which differ only in which registrations they chain. Handing `rcpd`
    the whole list rather than a neuron at a time is what lets `_register` spread it
    over the cores.

    Returns the aligned copies and, per neuron, the tuple of transforms that took it
    there.
    """
    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)

    assert isinstance(x, core.NeuronList), f'Expected NeuronList, got {type(x)}'

    if target is None:
        target = x[0]

    axis = None if mirror_axis is None else _mirror_index(mirror_axis)

    target_co = _subsample(_extract_coords(target), sample)

    # A neuron that *is* the target needs no fit; the answer is the identity.
    sources = [_subsample(_extract_coords(n), sample) for n in x]
    pairs = [(i, 0) for i, n in enumerate(x) if n is not target]
    fitted_idx = [i for i, _ in pairs]
    n_fit = len(pairs)

    if axis is not None:
        # The mirrored copies go into the *same* batch rather than a second call, so
        # both halves still spread over every core. Mirroring about the origin is
        # enough: the fit is free to translate, so any plane perpendicular to the
        # axis leads to the same registration.
        flip = np.ones(3)
        flip[axis] = -1
        offset = len(sources)
        sources = sources + [co * flip for co in sources]
        pairs = pairs + [(i + offset, 0) for i, _ in pairs]

    chains = _fit_pairs(method, sources, [target_co], pairs, n_cores=n_cores,
                        progress=progress and n_fit > 1, **kwargs)

    if axis is not None:
        # Same neuron, same target, two fits: keep whichever left the smaller residual.
        # `rms` is in input units rather than dimensionless, but the two are measured
        # against the same target, so they are directly comparable. Folding the
        # reflection into the rigid fit does not move the points it produces, so a warp
        # fitted on top of the mirrored one still belongs with it.
        chains = [(_fold_mirror(m[0], axis), ) + m[1:] if m[-1].rms < o[-1].rms else o
                  for o, m in zip(chains[:n_fit], chains[n_fit:])]

    by_idx = dict(zip(fitted_idx, chains))

    regs = []
    xf = x.copy()
    for i, n in enumerate(xf):
        if i not in by_idx:
            # This neuron *is* the target, so the alignment is the identity, exactly.
            regs.append(_identity_chain(method, sources[i]))
            continue
        chain = by_idx[i]
        regs.append(chain)
        # Applied to *all* coordinates, not just the subsample used for the fit.
        _set_coords(n, _apply_chain(chain, _extract_coords(n)))
        if (axis is not None and isinstance(n, core.Mesh)
                and np.linalg.det(chain[0].rotation) < 0):
            # The mirrored fit won, and a reflection turns a mesh inside out - so flip
            # the winding to keep the normals pointing outwards, as `mirror_brain` does.
            n.faces = n.faces[:, ::-1]
        if verbose and not all(tr.converged for tr in chain):
            stalled = max(tr.iterations for tr in chain if not tr.converged)
            logger.info(
                f'Registration of {n.id} onto {getattr(target, "id", "target")} '
                f'did not converge in {stalled} iterations'
            )

    return xf, regs


def _fit_pairs(method, sources, targets, pairs, progress=True, n_cores=None,
               **kwargs):
    """Fit `method` for the given (source, target) index pairs, spread over the cores.

    Returns one tuple of transforms per pair, in the same order; pushing a source
    cloud through them in order lands it on its target. That is a single transform
    for "rigid" and for "deform", and a rigid one followed by a warp for
    "rigid+deform".

    Only the transforms come back, never the moved points, which is what keeps peak
    memory off the size of the grid: a 100 x 100 alignment is 10,000 registrations,
    and at a few thousand points each the moved clouds would run to hundreds of
    gigabytes. That is unqualified only for a rigid fit, whose transform is thirteen
    floats: a deformable one is itself `O(M)`, and chaining one onto a rigid fit
    materialises a moved cloud per pair as well - so a big "deform"/"rigid+deform"
    grid is bounded by what it returns rather than by what it fits.
    """
    steps = method.split('+')
    opts = _fit_opts(steps, kwargs)

    if not len(pairs):
        return []

    chains = [() for _ in pairs]
    for step_no, step in enumerate(steps):
        if step_no:
            # Each step after the first is fitted on the points the one before it
            # moved, rather than on the originals. For "rigid+deform" that is the
            # load-bearing part: the motion coherence prior pays for displacement, so
            # any pose left for the deformation to undo comes out of the same budget
            # as the warp - and past a large enough offset it is not undone at all,
            # but settles into a wrong optimum. Each pair moved differently, so there
            # is now one source cloud per pair rather than one per neuron.
            sources = [chain[-1].apply(sources[i])
                       for chain, (i, _) in zip(chains, pairs)]
            pairs = [(k, j) for k, (_, j) in enumerate(pairs)]
        fitted = _register(step, sources, targets, pairs, n_cores=n_cores,
                           progress=progress, opts=opts[step])
        chains = [chain + (tr, ) for chain, tr in zip(chains, fitted)]

    return chains


def _register(step, sources, targets, pairs, n_cores, progress, opts):
    """Fit every pair with `rcpd.register_{step}`, whichever way keeps the cores busiest.

    `rcpd`'s batch entry points run one registration per core, which is the right
    split across a grid but leaves all but one core idle on a single pair - where
    threading *inside* the fit is worth ~8x instead (0.3 s against 2.3 s for two
    ~4,500-node skeletons on 14 cores). Fits threaded one at a time stay ahead until
    there are enough pairs to fill the cores, so that is where this switches over:
    four pairs take 1.2 s sequentially against 2.5 s as a batch.
    """
    rcpd = _import_rcpd()

    if len(pairs) * 2 <= (n_cores or rcpd.num_threads()):
        fit = getattr(rcpd, f'register_{step}')
        return [fit(sources[i], targets[j], n_cores=n_cores, **opts)
                for i, j in config.tqdm(pairs, desc='Aligning', disable=not progress)]

    fit_batch = getattr(rcpd, f'register_{step}_batch')
    return fit_batch(sources, targets, pairs=pairs, n_cores=n_cores,
                     progress=progress, **opts)


def _fit_opts(steps, kwargs):
    """Sort `kwargs` onto the registration(s) that take them.

    An option both fits take (`w`, `tolerance`, `max_iterations`) goes to both, and
    one neither takes is an error rather than something quietly dropped into a fit
    that ignores it.
    """
    known = {}
    for step in steps:
        known.update(_FIT_OPTS[step])

    unknown = set(kwargs) - set(known)
    if unknown:
        raise TypeError(f'Unexpected keyword argument(s) for a "{"+".join(steps)}" '
                        f'alignment: {", ".join(sorted(unknown))}. Expected any '
                        f'of: {", ".join(sorted(known))}')

    return {step: {k: kwargs.get(k, default)
                   for k, default in _FIT_OPTS[step].items()}
            for step in steps}


def _fit_options(func):
    """The registration options `func` exposes, with their defaults.

    Read off the signature so that it stays the one place they are written down:
    `align_pairwise` fits its whole grid in one go rather than by calling
    these functions, but must still default e.g. `scale` to what `align_rigid`
    documents - which is not what `rcpd` defaults to.
    """
    fixed = ('x', 'target', 'sample', 'mirror_axis', 'verbose', 'progress', 'n_cores')
    return {k: p.default for k, p in signature(func).parameters.items()
            if k not in fixed}


def _identity_chain(method, co):
    """The transforms for a neuron that *is* the target: the identity, exactly.

    Spelled with a zero residual rather than the unfitted `NaN` a hand-built
    transform carries, since this alignment is exact by construction - which also
    makes the diagonal of a pairwise grid sort as the best match rather than as a
    missing one.
    """
    rcpd = _import_rcpd()
    chain = []
    for step in method.split('+'):
        if step == 'rigid':
            chain.append(rcpd.RigidTransform(1.0, np.eye(3), np.zeros(3),
                                             sigma2=0.0, nrms=0.0))
        else:
            chain.append(rcpd.DeformTransform(co, np.zeros_like(co), 0.0,
                                              sigma2=0.0, nrms=0.0))
    return tuple(chain)


def _apply_chain(chain, co):
    """Push coordinates through a chain of fitted transforms, in order."""
    for tr in chain:
        co = tr.apply(co)
    return co


def _extract_coords(n):
    """Extract xyz coordinates from given object."""
    if isinstance(n, np.ndarray):
        return n
    elif isinstance(n, core.Mesh):
        return n.vertices
    elif isinstance(n, core.Skeleton):
        return n.nodes[['x', 'y', 'z']].values
    elif isinstance(n, core.Dotprops):
        return n.points
    else:
        raise TypeError(f'Unable to extract coordinates from {type(n)}')


def _set_coords(n, new_co):
    """Set new xyz coordinates for given object."""
    if new_co.ndim == 2 and new_co.shape[1] == 1:
        new_co = new_co.flatten()

    if new_co.ndim == 2:
        if isinstance(n, core.Mesh):
            n.vertices = new_co
        elif isinstance(n, core.Skeleton):
            n.nodes[['x', 'y', 'z']] = new_co
        elif isinstance(n, core.Dotprops):
            n.points = new_co
        else:
            raise TypeError(f'Unable to extract coordinates from {type(n)}')
    # If this is a single vector
    else:
        if isinstance(n, core.Mesh):
            for i in range(3):
                n.vertices[:, i] = new_co
        elif isinstance(n, core.Skeleton):
            for i in 'xyz':
                n.nodes[i] = new_co
        elif isinstance(n, core.Dotprops):
            for i in range(3):
                n.points[:, i] = new_co
        else:
            raise TypeError(f'Unable to extract coordinates from {type(n)}')


def _subsample(co, sample):
    """Take every `1/sample`-th point, or all of them if `sample` is None.

    Nothing has to be done afterwards to spread the result over the points that took
    no part in the fit: both registrations come back as functions of position rather
    than as sets of moved points, so they apply to the rest exactly. What a smaller
    `sample` does cost a deformable fit is kernel centres, so the warp it can express
    is smoother.
    """
    if sample is None or sample == 1:
        return co
    if not (0 < sample <= 1):
        raise ValueError(f'`sample` must be >0 and <=1, got {sample}')
    return co[::int(1 / sample)]


def _mirror_index(axis):
    """Turn a `mirror_axis` of 'x'/'y'/'z' or 0/1/2 into a column index.

    `bool` is excluded although it is an `int`: `mirror_axis=True` reads as "yes,
    mirror", and there is no axis to default to - better to say so than to quietly
    mirror on y.
    """
    if isinstance(axis, str) and axis.lower() in ('x', 'y', 'z'):
        return 'xyz'.index(axis.lower())
    if (isinstance(axis, (int, np.integer)) and not isinstance(axis, bool)
            and axis in (0, 1, 2)):
        return int(axis)
    raise ValueError('`mirror_axis` must be one of "x", "y", "z" or 0, 1, 2, '
                     f'got {axis!r}')


def _fold_mirror(reg, axis):
    """Rewrite a transform fitted on mirrored coordinates to take the originals.

    The fit maps `M @ x` onto the target, where `M` negates one axis, so the transform
    doing the same job from the unmirrored coordinates is `R @ M` - the fitted rotation
    with that column negated. Its determinant is -1, which is precisely the statement
    that the alignment flipped the neuron over. `R @ M` is orthogonal like `R`, so
    `.apply()`, `.matrix` and `~` all still hold; the fit diagnostics carry over
    unchanged, since it is the same fit.
    """
    rcpd = _import_rcpd()
    rot = reg.rotation.copy()
    rot[:, axis] *= -1
    return rcpd.RigidTransform(
        reg.scale, rot, reg.translation, sigma2=reg.sigma2, nrms=reg.nrms,
        iterations=reg.iterations, converged=reg.converged,
    )


# The options each registration takes, with their defaults. Built from the public
# functions so that their signatures stay the single source of truth.
_FIT_OPTS = {'rigid': _fit_options(align_rigid),
             'deform': _fit_options(align_deform)}
