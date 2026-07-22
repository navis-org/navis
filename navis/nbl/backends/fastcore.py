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

It does not support ``approx_nn``, ``scores='both'`` or arbitrary/callable/
analytic (``'v1'``) scoring matrices.

``nblast_knn`` is *exclusive* to this backend - there is no built-in
implementation - so it raises if navis-fastcore is missing or too old.
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
        # `nblast_knn` arrived later than the rest of the NBLAST API, so an
        # installed fastcore can be perfectly usable for everything else while
        # lacking this one. Note this is checked here rather than in
        # `available()`/`implements()` so that an old fastcore still serves the
        # other operations, and so that a *missing* fastcore is still reported
        # as "implements it but is not installed".
        if operation == 'nblast_knn':
            from ... import utils
            if not callable(getattr(utils.fastcore, 'nblast_knn', None)):
                reasons.append("the installed navis-fastcore is too old to "
                               "provide `nblast_knn` "
                               "(`pip install -U navis-fastcore`)")
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

    def nblast_knn(self, query, target, *, k, scores, n_candidates, normalized,
                   use_alpha, smat, limit_dist, precision, n_cores, progress,
                   voxel, n_dirs, splat, format, smat_kwargs):
        from ... import utils
        symmetry = None if scores in (None, 'forward') else scores
        idx, sc = utils.fastcore.nblast_knn(
            query,
            # `None` (rather than `query`) is meaningful: it puts fastcore on
            # the all-by-all path, which excludes each neuron from its own
            # neighbour list. Passing `query` twice would give every row a
            # self-match at 1.0.
            target,
            k=k,
            symmetry=symmetry,
            n_candidates=n_candidates,
            voxel=voxel,
            n_dirs=n_dirs,
            splat=splat,
            smat=self._convert_smat(smat, use_alpha),
            normalize=normalized,
            use_alpha=use_alpha,
            limit_dist=limit_dist,
            n_cores=n_cores,
            precision=precision,
            progress=progress,
        )

        if format == 'arrays':
            return idx, sc

        # Map the (n_query, k) indices back onto neuron IDs. Rows with fewer
        # than `k` candidates are padded with -1/-inf by fastcore; `-1` would
        # silently index the *last* neuron, so mask before taking.
        target_ids = np.asarray((target if target is not None else query).id)
        query_ids = np.asarray(query.id)
        valid = idx >= 0
        matches = target_ids[np.where(valid, idx, 0)]
        # Take the width from the result rather than from `k`: they agree today,
        # but an IndexError here would be a puzzling way to find out otherwise.
        k = idx.shape[1]

        if format == 'long':
            rank = np.broadcast_to(np.arange(1, k + 1), idx.shape)
            out = pd.DataFrame({
                'query': np.repeat(query_ids, k)[valid.ravel()],
                'target': matches[valid],
                'score': sc[valid],
                'rank': rank[valid],
            })
            return out.reset_index(drop=True)

        # 'wide': the layout `navis.nbl.extract_matches` produces, so the two
        # are interchangeable downstream.
        out = pd.DataFrame({'id': query_ids})
        for i in range(k):
            col = np.where(valid[:, i], matches[:, i], None)
            out[f'match_{i + 1}'] = col
            out[f'score_{i + 1}'] = np.where(valid[:, i], sc[:, i], np.nan)
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
