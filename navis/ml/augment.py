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

"""Data augmentation for neurons.

These transforms perturb a neuron's geometry (and, for dropout, its sampling)
to expand a training set - the standard trick for making a model invariant to
nuisance variation it should not care about. Every geometric augmentation is
applied through [`navis.xform`][], so coordinates, connectors, dotprops tangent
vectors, mesh vertices and (for scaling) radii/units all move consistently.

All functions:

- accept a `TreeNeuron`, `MeshNeuron`, `Dotprops` or a `NeuronList` (each neuron
  in a list is augmented **independently** - different random draw each). The one
  exception is `drop_nodes`, which does not support `MeshNeuron` (deleting
  vertices tears the surface - convert to skeleton/dotprops/points first),
- take a `random_state` (int or ``np.random.Generator``) for reproducibility,
- and return a **copy** (the input is never modified).
"""

import numbers

import numpy as np

from scipy.spatial.transform import Rotation

from .. import core, config
from ..transforms.affine import AffineTransform
from ..transforms.base import FunctionTransform
from ..transforms.xfm_funcs import xform
from ..graph.graph_utils import remove_nodes
from ..morpho.subset import subset_neuron

logger = config.get_logger(__name__)

__all__ = [
    "jitter_neuron",
    "rotate_neuron",
    "translate_neuron",
    "scale_neuron",
    "warp_neuron",
    "drop_nodes",
    "augment_neuron",
]

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


# --------------------------------------------------------------------------- #
# Geometric augmentations (applied through `navis.xform`)
# --------------------------------------------------------------------------- #
def jitter_neuron(x, sigma, random_state=None):
    """Perturb coordinates with Gaussian noise.

    Simulates the positional uncertainty of a reconstruction: every node/vertex/
    point (and every connector) is displaced by an independent draw from a normal
    distribution. Dotprops tangent vectors are regenerated from the jittered
    points, so orientation noise follows naturally.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | Dotprops | NeuronList
    sigma :     float | (3,) array-like
                Standard deviation of the noise **in coordinate units** (per-axis
                if a 3-vector). Note this is absolute: for a neuron normalized to
                unit RMS (see [`navis.ml.normalize_neuron`][]) a `sigma` of
                ~0.01-0.05 is a gentle jitter; for a raw neuron in nanometers you
                would use something on the order of the node spacing.
    random_state : int | np.random.Generator, optional
                Seed/generator for reproducibility.

    Returns
    -------
    same type as `x`
                A jittered copy.

    See Also
    --------
    [`navis.ml.warp_neuron`][]
                Smooth, spatially-correlated deformation (the low-frequency
                counterpart to this per-point noise).
    [`navis.ml.augment_neuron`][]
                Compose several augmentations in one call.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind="skeleton")
    >>> j = navis.ml.jitter_neuron(n, sigma=50, random_state=0)
    >>> j.n_nodes == n.n_nodes           # topology is untouched
    True

    """
    rng = np.random.default_rng(random_state)
    if isinstance(x, core.NeuronList):
        return core.NeuronList([jitter_neuron(n, sigma, random_state=rng) for n in x])

    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.repeat(sigma, 3)
    if sigma.shape != (3,) or np.any(sigma < 0):
        raise ValueError(f"`sigma` must be a non-negative scalar or (3,) array, got {sigma!r}")

    def add_noise(p):
        return p + rng.normal(0.0, 1.0, p.shape) * sigma

    xf = xform(x, FunctionTransform(add_noise))
    # `xform` rescales radius/units by a power-of-10 guess of the distance change
    # (it is built for unit conversions). Jitter only moves points, so restore
    # those attributes exactly - a large `sigma` can otherwise trip the guess.
    _restore_scale_attrs(xf, x)
    return xf


def rotate_neuron(x, axis=None, max_angle=None, random_state=None):
    """Rotate a neuron by a random rotation about its centroid.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | Dotprops | NeuronList
    axis :      None | "x" | "y" | "z" | (3,) array-like
                Rotation axis. If None, rotate about a uniformly random axis (a
                full uniformly-random orientation when `max_angle` is also None).
    max_angle : float, optional
                Maximum rotation magnitude in **degrees**. The angle is drawn
                uniformly from ``[-max_angle, max_angle]``. If None: a full
                uniformly-random rotation when `axis` is None, otherwise the full
                ``[-180, 180]`` range about the given axis.
    random_state : int | np.random.Generator, optional

    Returns
    -------
    same type as `x`
                A rotated copy (rotated in place - about the centroid - so it does
                not fly off when the neuron is not centered on the origin).

    See Also
    --------
    [`navis.ml.normalize_neuron`][]
                Remove orientation entirely (canonical PCA pose) instead of
                sampling random orientations.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind="skeleton")
    >>> r = navis.ml.rotate_neuron(n, axis="z", max_angle=30, random_state=0)

    """
    rng = np.random.default_rng(random_state)
    if isinstance(x, core.NeuronList):
        return core.NeuronList(
            [rotate_neuron(n, axis, max_angle, random_state=rng) for n in x]
        )

    R = _random_rotation(axis, max_angle, rng)
    M = _pivoted_affine(R, _coords(x))
    return xform(x, AffineTransform(M))


def translate_neuron(x, magnitude, random_state=None):
    """Shift a neuron by a random translation.

    Relocates the whole neuron by a single random offset - the pose augmentation
    for models that are **not** translation-invariant (i.e. that see absolute
    position, so where a neuron sits should not become a fixed cue). Unlike
    [`navis.ml.rotate_neuron`][] and [`navis.ml.scale_neuron`][], which act about
    the centroid and leave the neuron where it is, this moves its location.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | Dotprops | NeuronList
    magnitude : float | (3,) array-like
                Maximum absolute displacement **in coordinate units** (per-axis if
                a 3-vector). Each axis's offset is drawn uniformly from
                ``[-magnitude, magnitude]``. Like `jitter`, this is absolute: for a
                neuron normalized to unit RMS (see [`navis.ml.normalize_neuron`][])
                a `magnitude` of ~0.1-0.5 is a modest shift; for a raw neuron in
                nanometers use something on the order of its extent.
    random_state : int | np.random.Generator, optional
                Seed/generator for reproducibility.

    Returns
    -------
    same type as `x`
                A translated copy.

    See Also
    --------
    [`navis.ml.rotate_neuron`][]
                Random rotation about the centroid (leaves location unchanged).
    [`navis.ml.augment_neuron`][]
                Compose several augmentations in one call.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind="skeleton")
    >>> t = navis.ml.translate_neuron(n, magnitude=1000, random_state=0)
    >>> t.n_nodes == n.n_nodes           # topology and sizes are untouched
    True

    """
    rng = np.random.default_rng(random_state)
    if isinstance(x, core.NeuronList):
        return core.NeuronList(
            [translate_neuron(n, magnitude, random_state=rng) for n in x]
        )

    magnitude = np.asarray(magnitude, dtype=float)
    if magnitude.ndim == 0:
        magnitude = np.repeat(magnitude, 3)
    if magnitude.shape != (3,) or np.any(magnitude < 0):
        raise ValueError(
            f"`magnitude` must be a non-negative scalar or (3,) array, got {magnitude!r}"
        )

    M = np.eye(4)
    M[:3, 3] = rng.uniform(-magnitude, magnitude)
    # A pure translation leaves every pairwise distance unchanged, so `xform`'s
    # power-of-10 radius/unit guess is a no-op here (unlike jitter/warp, which go
    # through a FunctionTransform) - there is nothing to restore afterwards.
    return xform(x, AffineTransform(M))


def scale_neuron(x, scale_range=(0.8, 1.25), anisotropic=False, random_state=None):
    """Scale a neuron by a random factor about its centroid.

    Radii and `.units` are rescaled to match (by the geometric mean of the axis
    factors when `anisotropic`), so `radius` stays consistent with the coordinates
    and ``coordinate * units`` keeps its physical meaning.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | Dotprops | NeuronList
    scale_range : (low, high)
                Range of scale factors. Drawn **log-uniformly** so that, e.g.,
                0.8 and 1.25 are equally likely (they are reciprocals). Both must
                be > 0.
    anisotropic : bool
                If True, draw an independent factor per axis (a stretch/squash);
                dotprops tangent vectors are renormalized accordingly. If False
                (default), one factor for all three axes (a similarity).
    random_state : int | np.random.Generator, optional

    Returns
    -------
    same type as `x`
                A scaled copy.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind="skeleton")
    >>> s = navis.ml.scale_neuron(n, scale_range=(0.8, 1.25), random_state=0)

    """
    rng = np.random.default_rng(random_state)
    if isinstance(x, core.NeuronList):
        return core.NeuronList(
            [scale_neuron(n, scale_range, anisotropic, random_state=rng) for n in x]
        )

    lo, hi = scale_range
    if not (lo > 0 and hi > 0 and hi >= lo):
        raise ValueError(f"`scale_range` must be (low, high) with 0 < low <= high, got {scale_range!r}")

    n_factors = 3 if anisotropic else 1
    s = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n_factors))
    s_vec = np.repeat(s, 3) if not anisotropic else s

    M = _pivoted_affine(np.diag(s_vec), _coords(x))
    xf = xform(x, AffineTransform(M))

    # `xform` guesses a *power-of-10* radius/unit correction (built for unit
    # conversions); overwrite with the exact factor. Use the geometric mean for an
    # anisotropic scale (the cube-root of the volume factor).
    s_eff = float(np.exp(np.mean(np.log(s_vec))))
    _fix_scale_dependent(xf, x, s_eff)
    return xf


def warp_neuron(x, sigma=0.5, magnitude=0.05, grid=3, random_state=None):
    """Apply a smooth elastic deformation.

    A low-frequency random displacement field warps the neuron - the morphology
    equivalent of an elastic image augmentation. Displacements are defined at a
    coarse grid of control points and interpolated with a Gaussian kernel, so the
    warp is smooth (nearby points move together) and topology is preserved. Sizes
    are expressed as fractions of the neuron's bounding-box diagonal, so the same
    settings behave the same regardless of a neuron's absolute size or units.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | Dotprops | NeuronList
    sigma :     float
                Smoothness of the warp as a fraction of the bounding-box diagonal:
                the Gaussian kernel's length scale. Small (~0.1) = local wiggles,
                large (~1) = a gentle global bend. Default 0.5.
    magnitude : float
                Displacement strength as a fraction of the bounding-box diagonal
                (standard deviation of the per-control-point displacement).
                Default 0.05 (~5%).
    grid :      int
                Number of control points per axis (``grid**3`` total). Must be
                >= 2. More control points allow higher-frequency warps.
    random_state : int | np.random.Generator, optional

    Returns
    -------
    same type as `x`
                A warped copy.

    See Also
    --------
    [`navis.ml.jitter_neuron`][]
                Independent per-point noise (no spatial correlation).

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind="skeleton")
    >>> w = navis.ml.warp_neuron(n, sigma=0.5, magnitude=0.05, random_state=0)
    >>> w.n_nodes == n.n_nodes
    True

    """
    rng = np.random.default_rng(random_state)
    if isinstance(x, core.NeuronList):
        return core.NeuronList(
            [warp_neuron(n, sigma, magnitude, grid, random_state=rng) for n in x]
        )

    if int(grid) != grid or grid < 2:
        raise ValueError(f"`grid` must be an integer >= 2, got {grid!r}")
    if sigma <= 0 or magnitude < 0:
        raise ValueError("`sigma` must be > 0 and `magnitude` >= 0.")

    field = _warp_field(_coords(x), sigma, magnitude, int(grid), rng)
    if field is None:  # degenerate (zero-extent) neuron - nothing to warp
        return x.copy()
    xf = xform(x, FunctionTransform(field))
    # Warp only moves points; restore radius/units so `xform`'s power-of-10
    # distance-change guess can't rescale them (see `jitter_neuron`).
    _restore_scale_attrs(xf, x)
    return xf


# --------------------------------------------------------------------------- #
# Dropout (changes sampling, not just geometry)
# --------------------------------------------------------------------------- #
def drop_nodes(x, fraction=0.1, random_state=None):
    """Randomly drop a fraction of a neuron's nodes/points.

    Simulates a sparser or partial reconstruction.

    - **TreeNeuron**: drops random *non-branch, non-root* nodes and reconnects
      their children through the gap (via [`navis.remove_nodes`][]), so the arbor
      stays connected and its branching structure is preserved - only the node
      density drops. Connectors on a dropped node are reattached to its nearest
      surviving ancestor (the node its children collapse into), so none are left
      pointing at a removed node.
    - **Dotprops**: drops random points (a point cloud has no topology to keep).
    - **MeshNeuron**: not supported - deleting vertices tears the surface. Convert
      to skeleton/dotprops/points first.

    Parameters
    ----------
    x :         TreeNeuron | Dotprops | NeuronList
    fraction :  float
                Fraction of nodes/points to drop, in ``[0, 1)``. For skeletons this
                is a fraction of *all* nodes but only droppable (non-branch,
                non-root) nodes are removed; if the fraction exceeds their number
                it is capped (with a warning).
    random_state : int | np.random.Generator, optional

    Returns
    -------
    same type as `x`
                A copy with fewer nodes/points.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind="skeleton")
    >>> d = navis.ml.drop_nodes(n, fraction=0.2, random_state=0)
    >>> d.n_nodes < n.n_nodes
    True

    """
    rng = np.random.default_rng(random_state)
    if isinstance(x, core.NeuronList):
        return core.NeuronList([drop_nodes(n, fraction, random_state=rng) for n in x])

    if not (0 <= fraction < 1):
        raise ValueError(f"`fraction` must be in [0, 1), got {fraction!r}")

    if isinstance(x, core.TreeNeuron):
        n_total = len(x.nodes)
        protected = set(np.atleast_1d(x.root).tolist())
        protected |= set(x.branch_points.node_id.values.tolist())
        candidates = x.nodes.node_id.values[~np.isin(x.nodes.node_id.values, list(protected))]
        n_drop = int(round(fraction * n_total))
        if n_drop > len(candidates):
            logger.warning(
                f"`drop_nodes` can drop at most the {len(candidates)} non-branch/"
                f"non-root nodes; capping the requested {n_drop}."
            )
            n_drop = len(candidates)
        if n_drop == 0:
            return x.copy()
        which = rng.choice(candidates, size=n_drop, replace=False)
        out = remove_nodes(x, which, inplace=False)
        _rewire_connectors(x, out, which)
        return out

    if isinstance(x, core.Dotprops):
        n_total = len(x.points)
        n_drop = int(round(fraction * n_total))
        n_keep = max(1, n_total - n_drop)  # never drop everything
        keep = np.sort(rng.choice(n_total, size=n_keep, replace=False))
        return subset_neuron(x, keep, inplace=False)

    raise TypeError(
        f"`drop_nodes` supports TreeNeuron and Dotprops, not {type(x).__name__}. "
        "MeshNeurons would tear on vertex removal - convert to dotprops/points first."
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def augment_neuron(
    x,
    drop=None,
    warp=None,
    rotate=None,
    scale=None,
    translate=None,
    jitter=None,
    random_state=None,
):
    """Apply a pipeline of augmentations in one call.

    A convenience wrapper that chains the individual augmentations in a sensible
    order - ``drop -> warp -> rotate -> scale -> translate -> jitter`` (topology
    first, smooth deformation before rigid ones, independent noise last). Each
    argument is ``None`` (or ``False``) to skip that step, its primary value to
    run it with defaults, ``True`` to run it with defaults, or a dict of keyword
    arguments for full control. A skipped step draws nothing from the RNG, so
    toggling one step off does not change the random draws of the others.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron | Dotprops | NeuronList
    drop :      float | dict, optional
                `fraction` for [`navis.ml.drop_nodes`][].
    warp :      float | dict, optional
                `magnitude` for [`navis.ml.warp_neuron`][].
    rotate :    bool | float | dict, optional
                `max_angle` for [`navis.ml.rotate_neuron`][]; ``True`` = a full
                random rotation with defaults.
    scale :     (low, high) | dict, optional
                `scale_range` for [`navis.ml.scale_neuron`][].
    translate : float | (3,) | dict, optional
                `magnitude` for [`navis.ml.translate_neuron`][].
    jitter :    float | dict, optional
                `sigma` for [`navis.ml.jitter_neuron`][].
    random_state : int | np.random.Generator, optional
                One seed drives every step (drawn in sequence), so the whole
                pipeline is reproducible.

    Returns
    -------
    same type as `x`
                An augmented copy.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1, kind="skeleton")
    >>> aug = navis.ml.augment_neuron(
    ...     n, drop=0.1, rotate=True, scale=(0.8, 1.25), jitter=50, random_state=0
    ... )

    """
    rng = np.random.default_rng(random_state)
    if isinstance(x, core.NeuronList):
        return core.NeuronList([
            augment_neuron(n, drop, warp, rotate, scale, translate, jitter, random_state=rng)
            for n in x
        ])

    steps = [
        (drop, "fraction", drop_nodes),
        (warp, "magnitude", warp_neuron),
        (rotate, "max_angle", rotate_neuron),
        (scale, "scale_range", scale_neuron),
        (translate, "magnitude", translate_neuron),
        (jitter, "sigma", jitter_neuron),
    ]
    out = x
    for value, primary, fn in steps:
        kwargs = _step_kwargs(value, primary)
        if kwargs is None:
            continue
        out = fn(out, random_state=rng, **kwargs)
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _coords(x):
    """(N, 3) float coordinates that define the neuron's pose."""
    if isinstance(x, core.TreeNeuron):
        return x.nodes[["x", "y", "z"]].values.astype(float)
    if isinstance(x, core.MeshNeuron):
        return np.asarray(x.vertices, dtype=float)
    if isinstance(x, core.Dotprops):
        return np.asarray(x.points, dtype=float)
    raise TypeError(f"Unable to extract coordinates from {type(x)}")


def _random_rotation(axis, max_angle, rng):
    """A (3, 3) rotation matrix per the `axis`/`max_angle` policy."""
    if axis is None and max_angle is None:
        return Rotation.random(random_state=rng).as_matrix()

    if axis is None:
        vec = rng.normal(size=3)
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    elif isinstance(axis, str):
        if axis not in _AXES:
            raise ValueError(f'`axis` must be "x", "y", "z", a vector or None, got {axis!r}')
        vec = np.asarray(_AXES[axis])
    else:
        vec = np.asarray(axis, dtype=float).reshape(3)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("`axis` vector must be non-zero.")
        vec = vec / norm

    ma = 180.0 if max_angle is None else float(max_angle)
    angle = np.deg2rad(rng.uniform(-ma, ma))
    return Rotation.from_rotvec(angle * vec).as_matrix()


def _pivoted_affine(A, coords):
    """4x4 affine applying linear map `A` about the centroid of `coords`.

    ``p' = A @ (p - c) + c`` keeps a non-centered neuron in place while rotating/
    scaling it.
    """
    c = coords.mean(axis=0)
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = c - A @ c
    return M


def _warp_field(coords, sigma, magnitude, grid, rng):
    """Build a smooth displacement function ``p -> p + d(p)`` (or None if degenerate).

    Control points sit on a `grid`^3 lattice padded around the bounding box; each
    gets a random displacement, interpolated to query points with a normalized
    Gaussian kernel.
    """
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    if diag == 0:
        return None

    length = sigma * diag
    axes = [np.linspace(lo[i] - length, hi[i] + length, grid) for i in range(3)]
    control = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    disp = rng.normal(0.0, magnitude * diag, size=control.shape)
    inv = 1.0 / (2.0 * length ** 2)

    def field(p):
        # Squared distances (N, M) between query points and control points.
        d2 = ((p[:, None, :] - control[None, :, :]) ** 2).sum(axis=-1)
        w = np.exp(-d2 * inv)
        denom = w.sum(axis=1, keepdims=True)
        denom[denom < 1e-12] = 1.0
        return p + (w @ disp) / denom

    return field


def _fix_scale_dependent(xf, x, s):
    """Rescale `xf`'s radii and units by the exact factor `s` (in place).

    Mirrors the fix in `navis.ml.normalize` - see there for the rationale (xform's
    radius/unit correction is quantized to powers of 10).
    """
    if isinstance(x, core.TreeNeuron) and "radius" in x.nodes.columns:
        xf.nodes["radius"] = x.nodes["radius"].values * s
    if isinstance(getattr(x, "soma_radius", None), numbers.Number):
        xf.soma_radius = x.soma_radius * s
    if isinstance(getattr(x, "units", None), (config.ureg.Unit, config.ureg.Quantity)):
        xf.units = (x.units / s).to_compact()


def _rewire_connectors(orig, out, dropped):
    """Reattach `out`'s connectors that sat on dropped nodes (in place).

    `remove_nodes` rewires the skeleton but leaves the connector table pointing
    at the now-removed node IDs, orphaning every connector that sat on a dropped
    node. We follow the same rewiring the topology took: a connector on a dropped
    node moves to the node its children collapsed into - its nearest *surviving*
    ancestor - so it stays anchored to a real, on-arbor node. Roots and branch
    points are never dropped, so walking up the parent chain always lands on a
    surviving node.
    """
    if not getattr(out, "has_connectors", False):
        return
    dropped = set(np.asarray(dropped).tolist())
    on_dropped = out.connectors["node_id"].isin(dropped)
    if not on_dropped.any():
        return

    # Map every dropped node to its nearest surviving ancestor (memoized so the
    # whole chain of a long collapsed run is resolved in one walk).
    parent_of = dict(zip(orig.nodes.node_id.values, orig.nodes.parent_id.values))
    survivor = {}
    for nid in dropped:
        chain = []
        cur = nid
        while cur in dropped and cur not in survivor:
            chain.append(cur)
            cur = parent_of.get(cur, -1)
        end = survivor.get(cur, cur)
        for c in chain:
            survivor[c] = end

    conn = out.connectors.copy()
    conn["node_id"] = conn["node_id"].map(lambda i: survivor.get(i, i))
    out.connectors = conn


def _restore_scale_attrs(xf, x):
    """Restore `xf`'s radii and units from `x` (in place).

    Position-only augmentations (jitter, warp) must not change radius, soma
    radius or `.units`, but they go through `xform`, which rescales those by a
    power-of-10 *guess* of the distance change (it is built for unit
    conversions). A large enough perturbation trips that guess, so we write the
    original values straight back - unlike `_fix_scale_dependent`, this rescales
    by nothing and leaves the exact units object untouched.
    """
    if isinstance(x, core.TreeNeuron) and "radius" in x.nodes.columns:
        xf.nodes["radius"] = x.nodes["radius"].values
    if isinstance(getattr(x, "soma_radius", None), numbers.Number):
        xf.soma_radius = x.soma_radius
    if isinstance(getattr(x, "units", None), (config.ureg.Unit, config.ureg.Quantity)):
        xf.units = x.units


def _step_kwargs(value, primary):
    """Map an `augment_neuron` argument to kwargs (None/False = skip the step).

    `None` and `False` both skip; `True` runs the step with its defaults; a dict
    is passed through as keyword arguments; anything else is the step's primary
    value. Treating `False` as "skip" keeps ``rotate=False`` (rotate is documented
    as accepting a bool) intuitive and stops ``scale=False`` from crashing.
    """
    if value is None or value is False:
        return None
    if value is True:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {primary: value}
