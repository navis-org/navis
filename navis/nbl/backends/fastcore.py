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

"""NBLAST backend backed by ``navis-fastcore`` (Rust).

``navis-fastcore`` reimplements classic NBLAST in Rust. It accepts navis
``Dotprops`` directly (they expose ``.points``/``.vect``/``.alpha``) and a navis
``Lookup2d`` as the scoring matrix, does its own threading, and returns a plain
numpy array which we wrap back into a labeled DataFrame.

It only implements classic ``nblast`` / ``nblast_allbyall`` (not ``nblast_smart``
or ``synblast``) and does not support ``approx_nn``, ``scores='both'`` or
arbitrary/callable/analytic (``'v1'``) scoring matrices.
"""

import numpy as np
import pandas as pd

from ... import config
from .base import NblastBackend

logger = config.get_logger(__name__)


class FastcoreBackend(NblastBackend):
    """NBLAST backend using ``navis-fastcore``."""

    name = "fastcore"
    # Preferred over the builtin backend when it can serve the request
    priority = 10

    def available(self):
        from ... import utils
        fc = utils.fastcore
        # Also guard against fastcore builds that predate the NBLAST API: the
        # older `nblast` name resolves to a submodule, not a callable.
        return (fc is not None
                and callable(getattr(fc, 'nblast', None))
                and callable(getattr(fc, 'nblast_allbyall', None)))

    def _smat_ok(self, smat):
        """Whether fastcore can consume this `smat`."""
        if isinstance(smat, str):
            return smat == 'auto'
        if smat is None or isinstance(smat, pd.DataFrame):
            return True
        # A navis Lookup2d (or anything with the expected duck-type)
        return hasattr(smat, 'cells') and hasattr(smat, 'axes')

    def unsupported(self, operation, **params):
        reasons = super().unsupported(operation)
        if reasons:
            return reasons

        reasons = []
        if params.get('approx_nn', False):
            reasons.append("'approx_nn=True' is not supported by fastcore")
        if params.get('scores', None) == 'both':
            reasons.append("scores='both' is not supported by fastcore")
        if not self._smat_ok(params.get('smat', 'auto')):
            reasons.append("fastcore only supports smat='auto', None, a "
                           "DataFrame or a Lookup2d")
        return reasons

    def _convert_smat(self, smat, use_alpha):
        """Convert navis `smat` to something fastcore understands.

        We resolve ``'auto'`` to the very same FCWB ``Lookup2d`` the built-in
        backend uses, so both backends score against an identical matrix.
        """
        from ..smat import smat_fcwb, Lookup2d
        if isinstance(smat, str) and smat == 'auto':
            return smat_fcwb(use_alpha)
        if isinstance(smat, pd.DataFrame):
            return Lookup2d.from_dataframe(smat)
        # None or a ready Lookup2d
        return smat

    def nblast(self, query, target, *, scores, normalized, use_alpha, smat,
               limit_dist, approx_nn, precision, n_cores, progress, smat_kwargs):
        from ... import utils
        symmetry = None if scores in (None, 'forward') else scores
        M = utils.fastcore.nblast(
            query, target,
            smat=self._convert_smat(smat, use_alpha),
            normalize=normalized,
            symmetry=symmetry,
            use_alpha=use_alpha,
            limit_dist=limit_dist,
            n_cores=n_cores,
            precision=precision,
            progress=progress,
        )
        out = pd.DataFrame(M, index=query.id, columns=target.id)
        out.index.name = 'query'
        out.columns.name = 'target'
        return out

    def nblast_smart(self, query, target, *, aba, t, criterion, scores,
                     return_mask, normalized, use_alpha, smat, limit_dist,
                     approx_nn, precision, n_cores, progress, smat_kwargs):
        from ... import utils
        symmetry = None if scores in (None, 'forward') else scores
        res = utils.fastcore.nblast_smart(
            query,
            # `None` lets fastcore run its dedicated all-by-all pre-pass
            None if aba else target,
            t=t,
            criterion=criterion,
            smat=self._convert_smat(smat, use_alpha),
            normalize=normalized,
            symmetry=symmetry,
            use_alpha=use_alpha,
            limit_dist=limit_dist,
            n_cores=n_cores,
            precision=precision,
            progress=progress,
            return_mask=return_mask,
        )
        M, mask = res if return_mask else (res, None)

        out = pd.DataFrame(M, index=query.id, columns=target.id)
        out.index.name = 'query'
        out.columns.name = 'target'
        if not return_mask:
            return out

        mask = pd.DataFrame(mask, index=query.id, columns=target.id)
        mask.index.name = 'query'
        mask.columns.name = 'target'
        return out, mask

    def _encode_connectors(self, query, target, by_type, cn_types):
        """Turn navis neurons into fastcore's numeric connector arrays.

        navis stores connectors as DataFrames with string type labels, whereas
        fastcore wants each neuron as an ``(N, 3)`` (``[x, y, z]``) or - when
        ``by_type`` - ``(N, 4)`` (``[x, y, z, type]``) float array. When grouping
        by type we build a single string->int map shared across query and target
        so that like types compare against each other; ``cn_types`` filtering is
        applied here (so fastcore is called with ``cn_types=None``).
        """
        keep = None if cn_types is None else set(cn_types)

        def types_of(nl):
            s = set()
            for n in nl:
                vals = n.connectors['type']
                if keep is not None:
                    vals = vals[vals.isin(keep)]
                s.update(vals.unique().tolist())
            return s

        type_map = {}
        if by_type:
            all_types = sorted(types_of(query) | types_of(target), key=str)
            type_map = {t: i for i, t in enumerate(all_types)}

        def encode(nl):
            out = []
            for n in nl:
                cn = n.connectors
                if keep is not None:
                    cn = cn[cn['type'].isin(keep)]
                pts = cn[['x', 'y', 'z']].to_numpy(dtype=np.float64)
                if by_type:
                    codes = cn['type'].map(type_map).to_numpy(dtype=np.float64)
                    pts = np.hstack([pts, codes.reshape(-1, 1)])
                out.append(np.ascontiguousarray(pts, dtype=np.float64))
            return out

        return encode(query), encode(target)

    def synblast(self, query, target, *, by_type, cn_types, scores, normalized,
                 smat, n_cores, progress):
        from ... import utils
        q_arr, t_arr = self._encode_connectors(query, target, by_type, cn_types)
        symmetry = None if scores in (None, 'forward') else scores
        M = utils.fastcore.synblast(
            q_arr, t_arr,
            by_type=by_type,
            cn_types=None,  # already applied in _encode_connectors
            smat=self._convert_smat(smat, use_alpha=False),
            normalize=normalized,
            symmetry=symmetry,
            n_cores=n_cores,
            precision=64,  # match the built-in SynBlaster's float64 scores
            progress=progress,
        )
        out = pd.DataFrame(M, index=query.id, columns=target.id)
        out.index.name = 'query'
        out.columns.name = 'target'
        return out

    def nblast_allbyall(self, x, *, normalized, use_alpha, smat, limit_dist,
                        approx_nn, precision, n_cores, progress, smat_kwargs):
        from ... import utils
        M = utils.fastcore.nblast_allbyall(
            x,
            smat=self._convert_smat(smat, use_alpha),
            normalize=normalized,
            symmetry=None,  # navis' all-by-all always returns forward scores
            use_alpha=use_alpha,
            limit_dist=limit_dist,
            n_cores=n_cores,
            precision=precision,
            progress=progress,
        )
        out = pd.DataFrame(M, index=x.id, columns=x.id)
        out.index.name = 'query'
        out.columns.name = 'target'
        return out
