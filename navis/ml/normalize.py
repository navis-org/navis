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

import numbers

import numpy as np

from .. import core, config
from ..transforms.affine import AffineTransform
from ..transforms.xfm_funcs import xform

logger = config.get_logger(__name__)

__all__ = ["normalize_neuron"]


def normalize_neuron(
    x,
    center="centroid",
    scale="rms",
    rotate="pca",
    return_matrix=False,
):
    """Normalize a neuron's pose: center, orient and scale it canonically.

    Learning-based models should not have to spend capacity discovering that a
    neuron means the same thing after it is shifted, rotated or rescaled. This
    removes those nuisance degrees of freedom up front by mapping every neuron
    into a canonical frame with a single rigid-plus-uniform-scale transform
    ``p' = s * R @ (p - c)``:

    - **center** (`c`) moves a chosen reference point to the origin,
    - **rotate** (`R`) aligns the principal axes of the arbor with x/y/z, and
    - **scale** (`s`) normalizes the overall size.

    The transform is affine, so it is applied faithfully to *everything* the
    neuron carries - skeleton nodes, mesh vertices, dotprops points **and their
    tangent vectors**, connectors, soma and node radii, and the `.units` - via
    [`navis.xform`][]. Because it is a single matrix it is exactly invertible:
    pass ``return_matrix=True`` to get the 4x4 matrix and map model outputs in
    the normalized frame back to the original coordinates (see Examples).

    Parameters
    ----------
    x :         Skeleton | Mesh | Dotprops | NeuronList
                Neuron(s) to normalize. Each neuron in a `NeuronList` is
                normalized **independently** (into its own canonical frame).
                `Voxels` are not supported - rotating a dense grid needs
                resampling; convert to points/mesh first.
    center :    "centroid" | "bbox" | "soma" | (x, y, z) | None
                Reference point moved to the origin.
                - "centroid" (default): mean of the coordinates.
                - "bbox": midpoint of the axis-aligned bounding box.
                - "soma": the soma position (`Skeleton` with a soma only).
                - an explicit ``(x, y, z)`` coordinate.
                - None: no centering.
    rotate :    "pca" | None
                - "pca" (default): rotate so the largest-variance axis becomes
                  x, the next y and the smallest z. Signs are disambiguated
                  deterministically, so the same shape always lands in the same
                  orientation no matter how it came in, and handedness is
                  preserved (neurons are never mirrored). Orientation is unstable
                  when two principal axes have near-equal spread (e.g. a neuron
                  with no dominant axis).
                - None: no reorientation.
    scale :     "rms" | "extent" | "max" | None
                Size normalization, applied after centering/rotation.
                - "rms" (default): unit root-mean-square radius (robust to a few
                  far-flung nodes; the natural choice for most models).
                - "extent": longest principal-axis extent becomes 1 (fits the
                  arbor into a unit box along its principal axes).
                - "max": farthest point sits at radius 1 (unit enclosing sphere;
                  sensitive to outliers).
                - None: no scaling. Node/soma radii and `.units` are then left
                  untouched; otherwise they are rescaled by `s` to stay
                  physically consistent.
    return_matrix : bool
                If True, also return the 4x4 homogeneous matrix (or, for a
                `NeuronList`, a list of matrices) that maps original -> normalized
                coordinates.

    Returns
    -------
    xf :        same type as `x`
                A **copy** of `x` in its canonical pose (the input is never
                modified).
    matrix :    (4, 4) np.ndarray | list of np.ndarray
                Only if ``return_matrix=True``. `matrix @ [x, y, z, 1]` reproduces
                the normalization; ``np.linalg.inv(matrix)`` maps normalized
                coordinates back to the original frame.

    See Also
    --------
    [`navis.ml.chunk_neuron`][]
                Break a (typically pre-normalized) neuron into fixed-size
                fragments for batching.
    [`navis.ml.sample_points_uniform`][]
                Draw a uniform point sample from a neuron/cloud.
    [`navis.transforms.align`][]
                Register neurons *to each other* (rigid/deformable/PCA) rather
                than each into an absolute canonical frame.

    Examples
    --------
    >>> import navis
    >>> import numpy as np
    >>> n = navis.example_neurons(1, kind="skeleton")
    >>> norm = navis.ml.normalize_neuron(n)
    >>> co = norm.nodes[["x", "y", "z"]].values
    >>> bool(np.allclose(co.mean(axis=0), 0, atol=1e-6))   # centered
    True
    >>> float(round(np.sqrt((co ** 2).sum(axis=1).mean()), 6))  # unit RMS radius
    1.0
    >>> # Keep the matrix to map coordinates back to the original frame:
    >>> norm, M = navis.ml.normalize_neuron(n, return_matrix=True)
    >>> back = (np.linalg.inv(M) @ np.append(co[0], 1))[:3]
    >>> orig = n.nodes[["x", "y", "z"]].values[0]
    >>> bool(np.allclose(back, orig, atol=1e-4))
    True

    """
    # A NeuronList is normalized element-wise: each neuron gets its own frame.
    if isinstance(x, core.NeuronList):
        out = [
            normalize_neuron(
                n, center=center, scale=scale, rotate=rotate,
                return_matrix=return_matrix,
            )
            for n in x
        ]
        if return_matrix:
            neurons = [o[0] for o in out]
            matrices = [o[1] for o in out]
            return core.NeuronList(neurons), matrices
        return core.NeuronList(out)

    if not isinstance(x, (core.Skeleton, core.Mesh, core.Dotprops)):
        raise TypeError(
            "`normalize_neuron` expects a Skeleton, Mesh, Dotprops or a "
            f"NeuronList thereof, got {type(x)}. Voxels are not supported "
            "(rotating a grid needs resampling) - convert to points or a mesh "
            "first."
        )

    if rotate not in ("pca", None):
        raise ValueError(f'`rotate` must be "pca" or None, got {rotate!r}')

    if scale not in ("rms", "extent", "max", None):
        raise ValueError(
            f'`scale` must be "rms", "extent", "max" or None, got {scale!r}'
        )

    if isinstance(center, str) and center not in ("centroid", "bbox", "soma"):
        raise ValueError(
            '`center` must be "centroid", "bbox", "soma", an (x, y, z) '
            f"coordinate or None, got {center!r}"
        )

    M, s = _normalization_matrix(x, center, scale, rotate)
    xf = xform(x, AffineTransform(M))

    # `xform` moves coordinates, connectors and dotprops tangent vectors exactly,
    # but it rescales `radius`/`soma_radius`/`units` by a power-of-10 *guess* of
    # the magnitude (it is built for unit conversions). Our scale is exact and
    # rarely a power of 10, so overwrite those three with the exact factor, taken
    # from the original neuron so we don't depend on `xform`'s heuristic.
    if not np.isclose(s, 1.0):
        _fix_scale_dependent(xf, x, s)

    if return_matrix:
        return xf, M
    return xf


def _normalization_matrix(x, center, scale, rotate):
    """4x4 homogeneous affine mapping `x` to its canonical pose, plus the scale.

    Encodes ``p' = s * R @ (p - c)``; any component may be disabled (zero `c`,
    identity `R`, unit `s`). Returns ``(M, s)`` - `s` is handed back so the caller
    can rescale scale-dependent attributes (radii, units) with the exact factor.
    """
    co = _reference_coords(x)

    c = _center_point(x, co, center)
    q = co - c  # centered coordinates - the basis for rotation and scale

    if rotate == "pca":
        if len(q) >= 2:
            R = _pca_rotation(q)
        else:
            logger.warning(
                "Too few points to compute a PCA orientation - skipping rotation."
            )
            R = np.eye(3)
    else:
        R = np.eye(3)
    q = q @ R.T

    s = _scale_factor(q, scale)

    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = -s * (R @ c)
    return M, s


def _fix_scale_dependent(xf, x, s):
    """Rescale `xf`'s radii and units by the exact factor `s` (in place).

    `xf` is the transformed copy, `x` the original. We read from `x` and write the
    exact value into `xf` rather than adjusting `xf`'s already-(approximately-)
    scaled values, so the result is exact regardless of what `xform` guessed.
    Coordinates/connectors are left as `xform` set them (already exact).
    """
    if isinstance(x, core.Skeleton) and "radius" in x.nodes.columns:
        xf.nodes["radius"] = x.nodes["radius"].values * s
    if isinstance(getattr(x, "soma_radius", None), numbers.Number):
        xf.soma_radius = x.soma_radius * s
    if isinstance(getattr(x, "units", None), (config.ureg.Unit, config.ureg.Quantity)):
        # units scale inversely with coordinates so `coords * units` (the physical
        # size) is preserved; `.to_compact()` just picks a readable prefix.
        xf.units = (x.units / s).to_compact()


def _reference_coords(x):
    """(N, 3) float coordinates that define the neuron's pose."""
    if isinstance(x, core.Skeleton):
        return x.nodes[["x", "y", "z"]].values.astype(float)
    if isinstance(x, core.Mesh):
        return np.asarray(x.vertices, dtype=float)
    if isinstance(x, core.Dotprops):
        return np.asarray(x.points, dtype=float)
    raise TypeError(f"Unable to extract coordinates from {type(x)}")


def _center_point(x, co, center):
    """Resolve the `center` argument to an (x, y, z) point."""
    if center is None:
        return np.zeros(3)
    if not isinstance(center, str):
        c = np.asarray(center, dtype=float).reshape(-1)
        if c.size != 3:
            raise ValueError(
                "`center` must be an (x, y, z) coordinate with 3 values, got "
                f"{c.size}."
            )
        return c
    if center == "centroid":
        return co.mean(axis=0)
    if center == "bbox":
        return (co.min(axis=0) + co.max(axis=0)) / 2
    # center == "soma"
    pos = getattr(x, "soma_pos", None)
    if pos is None:
        raise ValueError(
            'center="soma" requires a neuron with a soma, but '
            f"{type(x).__name__} {getattr(x, 'id', '?')} has none. "
            'Use "centroid" or "bbox" instead.'
        )
    return np.asarray(pos, dtype=float).reshape(-1, 3).mean(axis=0)


def _scale_factor(q, scale):
    """Scalar `s` normalizing the (centered, rotated) coordinates `q`."""
    if scale is None:
        return 1.0
    if scale == "rms":
        denom = np.sqrt((q ** 2).sum(axis=1).mean())
    elif scale == "max":
        denom = np.sqrt((q ** 2).sum(axis=1)).max()
    else:  # "extent"
        denom = (q.max(axis=0) - q.min(axis=0)).max()
    if not np.isfinite(denom) or denom <= 0:
        logger.warning(
            f'Cannot compute a "{scale}" scale (degenerate/zero extent) - '
            "leaving scale unchanged."
        )
        return 1.0
    return 1.0 / float(denom)


def _pca_rotation(q):
    """Right-handed rotation aligning the principal axes of `q` with x/y/z.

    `q` must already be centered. Returns `R` (3, 3) with ``new = R @ p``: the
    largest-variance axis maps to x, the next to y, the smallest to z. Two things
    make the result a *canonical* orientation rather than an arbitrary PCA frame:

    - **Deterministic signs.** PCA axes are only defined up to sign, so the same
      shape could otherwise land flipped. We fix each of the two dominant axes'
      signs by the point of largest absolute projection (make it positive), which
      depends only on the shape, not its incoming pose.
    - **Preserved handedness.** The third axis is set as ``cross(axis0, axis1)``,
      forcing ``det(R) = +1`` (a proper rotation). We never mirror the neuron -
      morphology is chiral and a reflection would be a different shape.
    """
    cov = np.cov(q.T)
    evals, evecs = np.linalg.eigh(cov)              # ascending eigenvalues
    axes = evecs[:, np.argsort(evals)[::-1]].T      # rows = PC1, PC2, PC3
    proj = q @ axes.T
    for i in (0, 1):
        col = proj[:, i]
        if col[np.argmax(np.abs(col))] < 0:
            axes[i] = -axes[i]
    axes[2] = np.cross(axes[0], axes[1])
    return axes
