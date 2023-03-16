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

import warnings

import numpy as np
import pandas as pd

from .. import core, utils, config


def align_pairwise(x, y=None, method='rigid', progress=True, **kwargs):
    """Run a pairwise alignment between given neurons.

    Requires the `pycpd` library.

    Parameters
    ----------
    x :         navis.NeuronList
                Neurons to align to other neurons.
    y :         navis.NeuronList, optional
                The neurons to align to. If ``None``, will run pairwise
                alignment of ``x`` vs ``x``.
    method :    "rigid" | "deform" | "pca"
                Which method to use for alignment. Maps to the respective
                ``navis.align_{method}`` function.
    **kwargs
                Keyword arguments are passed through to the respective
                alignment function.

    Returns
    -------
    np.ndarray
                Array of shape (x, y) with the pairwise-aligned neurons.

    """
    squared = False
    if y is None:
        y = x
        squared = True

    utils.eval_param(x, name='x', allowed_types=(core.NeuronList, ))
    utils.eval_param(y, name='y', allowed_types=(core.NeuronList, ))
    utils.eval_param(method, name='method',
                     allowed_values=('rigid', 'deform', 'pca'))

    func = {'rigid': align_rigid,
            'deform': align_deform,
            'pca': align_pca}[method]

    aligned = []
    for n1 in config.tqdm(x,
                          desc='Aligning',
                          disable=not progress or len(x) == 1):
        aligned.append([])
        for n2 in y:
            if n1 == n2:
                xf = n1
            else:
                xf = func(n1, target=n2, **kwargs, progress=False)[0][0]
            aligned[-1].append(xf)

    return np.array(aligned)


def align_rigid(x, target=None, scale=False, verbose=False, progress=True):
    """Align neurons using a rigid registration.

    Requires the `pycpd` library.

    Parameters
    ----------
    x :             navis.NeuronList
                    Neurons to align.
    target :        navis.Neuron | np.ndarray
                    The neuron that all neurons in `x` will be aligned to.
                    If `None`, neurons will be aligned to the first neuron `x`!
    scale :         bool
                    If True, will also scale the neuron.
    progress :      bool
                    Whether to show a progress bar.

    Returns
    -------
    xf :    navis.NeuronList
            The aligned neurons.
    pca
            The pycpd registration objects.

    """
    try:
        from pycpd import RigidRegistration as Registration
    except ImportError:
        raise ImportError('`rigid_align()` requires the `pycpd` library:\n'
                          '  pip3 install pycpd -U')

    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)

    assert isinstance(x, core.NeuronList)

    if target is None:
        target = x[0]

    target_co = _extract_coords(target)

    xf = x.copy()
    regs = []
    for n in config.tqdm(xf,
                         disable=(not progress) or (len(xf) == 1),
                         desc='Aligning'):
        if n == target:
            continue
        # `w` is used to account for outliers -> higher w = more forgiving
        # the default is w=0 which can lead to failure to converge on a solution
        # in particular when scale=False
        # Our work-around here is to start at w=0 and incrementally increase w
        # if we fail to converge
        # Also note that pycpd ignores the `scale` in earlier versions. The
        # version on PyPI is currently outdated. From what I understand we need
        # the Github version.
        w = 0
        converged = False
        while w <= 0.001:
            try:
                reg = Registration(X=target_co,
                                   Y=_extract_coords(n),
                                   scale=scale,
                                   s=1,
                                   w=w
                                   )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    TY, params = reg.register()
                _set_coords(n, TY)
                converged = True
                break
            except np.linalg.LinAlgError:
                if w == 0:
                    w += 0.000000001
                else:
                    w *= 10

        if verbose and not converged:
            logger.info(f'Registration of {n.id} onto {target.id} did not converge')

        regs.append(reg)

    return xf, regs


def align_deform(x, target=None, progress=True, **kwargs):
    """Align neurons using a deformable registration.

    Requires the `pycpd` library. Note that it's often beneficial to first
    run a rough affine alignment via `rigid_align`.

    Parameters
    ----------
    x :             navis.NeuronList
                    Neurons to align.
    target :        navis.Neuron | np.ndarray
                    The neuron that all neurons in `x` will be aligned to.
                    If `None`, neurons will be aligned to the first neuron `x`!
    progress :      bool
                    Whether to show a progress bar.
    **kwargs
                    Additional keyword-argumens are passed through to
                    pycpd.DeformableRegistration. In brief: lower `alpha` and
                    higher `beta` typically make for more fitting deform. I have
                    gone as far as alpha=.01, beta=10000.

    Returns
    -------
    xf :    navis.NeuronList
            The aligned neurons.
    pca
            The pycpd registration objects.

    """
    try:
        from pycpd import DeformableRegistration as Registration
    except ImportError:
        raise ImportError('`deform_align()` requires the `pycpd` library:\n'
                          '  pip3 install pycpd -U')

    if isinstance(x, core.BaseNeuron):
        x = core.NeuronList(x)

    assert isinstance(x, core.NeuronList)

    if target is None:
        target = x[0]

    target_co = _extract_coords(target)

    # pycpd's deformable registration is very sensitive to the scale of the
    # data. We will hence normalize the neurons to be within the -1 to 1 range
    scale_factor = 0
    for n in x:
        co = _extract_coords(n)
        mx = np.abs(co).max()
        scale_factor = mx if mx > scale_factor else scale_factor

    xf = x / scale_factor
    target_co = target_co / scale_factor
    regs = []
    for n in config.tqdm(xf, disable=not progress, desc='Aligning'):
        if n == target:
            continue
        reg = Registration(X=target_co, Y=_extract_coords(n), **kwargs)
        TY, params = reg.register()
        _set_coords(n, TY)
        regs.append(reg)

    return xf * scale_factor, regs


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
    pca
            The scikit-learn PCA object(s)

    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError('`pca_align()` requires the `scikit-learn` library:\n'
                          '  pip3 install scikit-learn -U')

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


def _extract_coords(n):
    """Extract xyz coordinates from given object."""
    if isinstance(n, np.ndarray):
        return n
    elif isinstance(n, core.MeshNeuron):
        return n.vertices
    elif isinstance(n, core.TreeNeuron):
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
        if isinstance(n, core.MeshNeuron):
            n.vertices = new_co
        elif isinstance(n, core.TreeNeuron):
            n.nodes[['x', 'y', 'z']] = new_co
        elif isinstance(n, core.Dotprops):
            n.points = new_co
        else:
            raise TypeError(f'Unable to extract coordinates from {type(n)}')
    # If this is a single vector
    else:
        if isinstance(n, core.MeshNeuron):
            for i in range(3):
                n.vertices[:, i] = new_co
        elif isinstance(n, core.TreeNeuron):
            for i in 'xyz':
                n.nodes[i] = new_co
        elif isinstance(n, core.Dotprops):
            for i in range(3):
                n.points[:, i] = new_co
        else:
            raise TypeError(f'Unable to extract coordinates from {type(n)}')
