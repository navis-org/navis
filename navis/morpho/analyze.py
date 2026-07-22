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

import pint
import warnings

import numpy as np

from dataclasses import dataclass
from scipy.spatial import cKDTree

from .. import config, core, utils

from typing import Optional, Sequence, Union

# Set up logging
logger = config.get_logger(__name__)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


def find_soma(x: 'core.TreeNeuron', *, dist_factor: float = 3.0) -> Optional[int]:
    """Try finding a neuron's soma.

    Uses the neuron's `.soma_detection_radius` and/or `.soma_detection_label`
    attributes to find candidate soma nodes, then returns the single most
    likely soma node:

    - Candidates are nodes whose `radius >= soma_detection_radius` (nodes with
      missing radius - `NaN` or `<= 0` - are ignored) and, if the node table
      has a `label` column, whose label also matches `soma_detection_label`.
    - Each candidate is scored by the mean radius of the candidates within
      `dist_factor` times its own radius, and the fattest node of the best
      scoring cluster is returned. Because it is the fattest *region* that wins,
      individual thick nodes elsewhere on the arbor (e.g. on a primary neurite)
      are not mistaken for the soma.

    If neither `.soma_detection_radius` nor `.soma_detection_label` is set, or
    no candidate is found, returns `None`.

    Parameters
    ----------
    x :             TreeNeuron
    dist_factor :   float
                    Candidate nodes within `dist_factor` times a candidate's
                    radius are treated as belonging to the same cluster. Larger
                    values average over a wider neighbourhood.

    Returns
    -------
    node_id :       int | None
                    Node ID of the detected soma, or `None` if no soma found.

    Examples
    --------
    >>> import navis
    >>> n = navis.example_neurons(1)
    >>> int(navis.find_soma(n))
    4177

    """
    if not isinstance(x, core.TreeNeuron):
        raise TypeError(f'Input must be TreeNeuron, not "{type(x)}"')

    soma_radius = getattr(x, 'soma_detection_radius', None)
    soma_label = getattr(x, 'soma_detection_label', None)

    check_labels = not isinstance(soma_label, type(None)) and 'label' in x.nodes.columns
    check_radius = not isinstance(soma_radius, type(None))

    # If neither criterion is configured there is nothing to detect
    if not check_labels and not check_radius:
        return None

    radii = x.nodes.radius.values

    # Build a full-length boolean mask of candidate soma nodes (masking is much
    # faster than repeatedly subsetting the node table)
    mask = np.ones(len(x.nodes), dtype=bool)
    have_usable_radius = False
    applied_any = False
    if check_radius:
        # Treat both NaN and <= 0 as "missing" (e.g. `guess_radius` writes -1).
        # If no node has a usable radius we skip the radius criterion entirely
        # rather than rejecting everything - a labelled soma can still be found.
        has_radius = ~np.isnan(radii) & (radii > 0)
        if has_radius.any():
            thr = _resolve_threshold(x, soma_radius)
            mask &= has_radius & (radii >= thr)
            have_usable_radius = True
            applied_any = True

    if check_labels:
        # np.asarray guards against a categorical `label` column throwing warnings
        labels = np.asarray(x.nodes.label.values).astype(str)
        mask &= labels == str(soma_label)
        applied_any = True

    # Radius was requested but unusable and there is no label to fall back on
    if not applied_any:
        return None

    cand_idx = np.flatnonzero(mask)
    if not cand_idx.size:
        return None

    node_ids = x.nodes.node_id.values

    if have_usable_radius:
        # Spatial clustering: the soma is the fattest *region*, not necessarily
        # the fattest single node. Score each candidate by the mean radius of
        # the candidates in the ball of `dist_factor * R` around it: a lone
        # thick node (e.g. on a primary neurite) is diluted by its thinner
        # neighbours while a soma is fat throughout. The fattest node of the
        # winning ball is returned.
        coords = x.nodes[['x', 'y', 'z']].values.astype(float)[cand_idx]
        cand_r = radii[cand_idx]
        balls = cKDTree(coords).query_ball_point(coords, dist_factor * cand_r)
        scores = np.array([cand_r[b].mean() for b in balls])
        inl = balls[int(np.argmax(scores))]
        best = node_ids[cand_idx][inl][int(np.argmax(cand_r[inl]))]
    else:
        # Label-only path (no radius to seed a ball): use the largest connected
        # label component and return its most central node.
        from ..graph.graph_utils import connected_components_of

        comps = connected_components_of(x, set(node_ids[cand_idx].tolist()))
        comp = np.array(sorted(max(comps, key=len)))
        coords = x.nodes.set_index('node_id').loc[comp, ['x', 'y', 'z']].values.astype(float)
        best = comp[int(np.argmin(np.linalg.norm(coords - coords.mean(0), axis=1)))]

    return node_ids.dtype.type(best)


def _resolve_threshold(x: 'core.TreeNeuron', soma_radius) -> float:
    """Resolve a soma-detection radius to a plain number in neuron space.

    Unlike `find_soma_mesh` this must not raise (it runs on every `.soma`
    access): for dimensionless or anisotropic neurons it falls back to the bare
    magnitude, matching navis's historically lenient behaviour.
    """
    thr = x.map_units(soma_radius, on_error='ignore')
    if isinstance(thr, pint.Quantity):
        thr = thr.magnitude
    return float(thr)


@dataclass
class SomaEllipsoid:
    """Oriented ellipsoid describing a soma detected on a mesh.

    Returned by [`navis.find_soma_mesh`][]. The ellipsoid is centered on the
    soma and oriented along its principal axes.

    Attributes
    ----------
    center :            (3, ) array
                        XYZ center of the soma (the point of maximum inscribed
                        radius). This is what gets assigned to a MeshNeuron's
                        `.soma_pos`.
    radii :             (3, ) array
                        Semi-axis lengths, sorted descending (a >= b >= c).
    axes :              (3, 3) array
                        Principal axes as columns (unit vectors), matching
                        `radii`.
    inscribed_radius :  float
                        Radius of the largest sphere that fits inside the mesh
                        at `center`. This is the robust scalar radius that drove
                        detection.
    n_vertices :        int
                        Number of mesh vertices used for the ellipsoid fit.

    """

    center: np.ndarray
    radii: np.ndarray
    axes: np.ndarray
    inscribed_radius: float
    n_vertices: int

    @property
    def equiv_radius(self) -> float:
        """Radius of the sphere with the same volume as the ellipsoid."""
        return float(np.prod(self.radii) ** (1 / 3))

    @property
    def volume(self) -> float:
        """Volume of the ellipsoid."""
        return float(4 / 3 * np.pi * np.prod(self.radii))

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask of which points lie inside the ellipsoid.

        Parameters
        ----------
        points :    (N, 3) | (3, ) array

        Returns
        -------
        (N, ) bool array

        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        # Map into the unit sphere: xi = (x - c) @ (axes / radii)
        xi = (points - self.center) @ (self.axes / self.radii)
        return (xi**2).sum(axis=1) <= 1.0

    def distance_to_surface(
        self, points: np.ndarray, atol: float = 1e-9, max_iter: int = 64
    ) -> np.ndarray:
        """Signed Euclidean distance to the ellipsoid surface.

        Parameters
        ----------
        points :    (N, 3) | (3, ) array

        Returns
        -------
        (N, ) float array
                    Positive outside, negative inside, ~0 on the surface.

        """
        x = np.atleast_2d(np.asarray(points, dtype=float))
        # Align to principal axes
        p = (x - self.center) @ self.axes
        a = self.radii
        a2 = a * a
        r2 = (p**2 / a2).sum(axis=1)
        out = r2 > 1.0 + 1e-12
        dist = np.empty(len(p), dtype=float)

        # Outside points: Newton iteration for the nearest surface point
        if out.any():
            po = p[out]
            t = np.zeros(len(po))
            for _ in range(max_iter):
                denom = t[:, None] + a2
                f = (a2 * po**2 / denom**2).sum(1) - 1.0
                fp = (-2.0 * a2 * po**2 / denom**3).sum(1)
                dt = -f / fp
                t += dt
                if np.all(np.abs(dt) < atol):
                    break
            xs = a2 * po / (t[:, None] + a2)
            dist[out] = np.linalg.norm(xs - po, axis=1)

        # Inside points: radial projection onto the surface
        inn = ~out
        if inn.any():
            idx = np.where(inn)[0]
            pi = p[inn]
            s = np.sqrt(r2[inn])
            nz = s > atol
            if nz.any():
                xs = pi[nz] / s[nz, None]
                dist[idx[nz]] = -np.linalg.norm(xs - pi[nz], axis=1)
            if (~nz).any():  # exactly at the center
                dist[idx[~nz]] = -a.min()

        return dist

    def __repr__(self) -> str:
        c = ", ".join(f"{v:.1f}" for v in self.center)
        r = ", ".join(f"{v:.1f}" for v in self.radii)
        return (
            f"SomaEllipsoid(center=[{c}], radii=[{r}], "
            f"inscribed_radius={self.inscribed_radius:.1f}, "
            f"n_vertices={self.n_vertices})"
        )


def _grid(lo: np.ndarray, hi: np.ndarray, pitch: float) -> np.ndarray:
    """Regular XYZ grid covering the [lo, hi] box at the given pitch."""
    ax = [np.arange(lo[i], hi[i] + pitch, pitch) for i in range(3)]
    gx, gy, gz = np.meshgrid(*ax, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _interior_depth_field(coll, lo, hi, pitch):
    """Depth of interior grid points sampled over the [lo, hi] box.

    `depth = -signed distance to the surface`, i.e. the radius of the largest
    sphere that fits inside the mesh at each interior sample point.

    Parameters
    ----------
    coll :          ncollpyde.Volume
    lo, hi :        (3, ) array
                    Lower/upper corners of the sampling box.
    pitch :         float
                    Grid spacing.

    Returns
    -------
    points :        (M, 3) array
                    Interior sample points.
    depth :         (M, ) array
                    Depth (inscribed radius) at each interior point.

    """
    pts = _grid(lo, hi, pitch)
    d = coll.distance(pts, signed=True)
    interior = d < 0
    return pts[interior], -d[interior]


def _fit_soma_ellipsoid(vertices: np.ndarray, center: np.ndarray):
    """PCA/covariance ellipsoid fit to soma-patch vertices about `center`.

    Adapted from skeliner's `Soma.fit`: eigen-decompose the second-moment
    matrix of the vertices about `center`; the eigenvectors are the principal
    axes and the semi-axes are `sqrt(5 * eigenvalue)` (~2 sigma, a rough
    95%-of-mass envelope). Sorted so that a >= b >= c.

    Returns
    -------
    radii :     (3, ) array
    axes :      (3, 3) array

    """
    diff = vertices - center
    moment = diff.T @ diff / max(len(vertices) - 1, 1)
    ev, evec = np.linalg.eigh(moment)
    order = np.argsort(ev)[::-1]
    radii = np.sqrt(5.0 * np.clip(ev[order], 0, None))
    axes = evec[:, order]
    return radii, axes


def _estimate_edge_length(vertices: np.ndarray, faces: np.ndarray, sample: int = 20_000) -> float:
    """Deterministic estimate of the mean triangle edge length.

    Uses a strided sample of faces so the cost is independent of mesh size. This
    is only used as a (rarely binding) floor on the search-grid pitch, so an
    estimate is plenty - computing the exact mean over every edge (as
    `MeshNeuron.sampling_resolution` does) would dominate on very large meshes.
    """
    if not len(faces):
        return 1.0
    if len(faces) > sample:
        faces = faces[:: len(faces) // sample]
    tris = vertices[faces]                      # (S, 3, 3)
    edges = tris - tris[:, [1, 2, 0]]           # v0->v1, v1->v2, v2->v0
    return float(np.linalg.norm(edges, axis=2).mean())


@utils.map_neuronlist(desc="Detecting somas", allow_parallel=True)
def find_soma_mesh(
    x: "core.MeshNeuron",
    *,
    min_soma_radius: Union[str, float] = "1 micron",
    dist_factor: float = 3.0,
    min_vertices: int = 20,
    max_points: int = 200_000,
    n_rays: int = 10,
    inplace: bool = False,
) -> Optional[Union["SomaEllipsoid", "core.MeshNeuron"]]:
    """Detect the soma of a MeshNeuron from its geometry.

    Finds the thickest region of the mesh - the point of largest inscribed
    sphere - and fits an oriented ellipsoid to the surrounding surface. The
    approach is inspired by the [skeliner](https://github.com/berenslab/skeliner)
    library. Note that this works directly on the mesh (no skeletonization
    required) and returns the soma as an ellipsoid.

    For skeletons (TreeNeurons) use [`navis.find_soma`][] instead.

    Parameters
    ----------
    x :                 MeshNeuron | NeuronList
                        Neuron(s) to detect the soma for.
    min_soma_radius :   str | float
                        Minimum inscribed-sphere radius for a region to count as
                        a soma. If the neuron has `.units` set you can pass a
                        string such as "1 micron" (the default); otherwise pass a
                        number in the neuron's coordinate space. This is the main
                        knob for accepting/rejecting a soma and should be tuned to
                        your data.
    dist_factor :       float
                        The soma surface is taken as all vertices within
                        `dist_factor * inscribed_radius` of the center. Larger
                        values include more of the surrounding surface in the
                        ellipsoid fit.
    min_vertices :      int
                        Minimum number of surface vertices in the soma patch. Fat
                        regions with fewer vertices are rejected (this is what
                        separates a real soma from a thick neurite).
    max_points :        int
                        Upper bound on the number of samples in the coarse search
                        grid. Keeps the cost bounded on large meshes.
    n_rays :            int
                        Number of rays `ncollpyde` uses for the inside/outside
                        test. Higher values are more robust on non-watertight
                        meshes at a small linear cost.
    inplace :           bool
                        If True, set `x.soma_pos` to the detected center (or
                        `None`) and return the neuron. If False (default) return
                        the [`navis.SomaEllipsoid`][] (or `None`).

    Returns
    -------
    SomaEllipsoid
                        If a soma was found and `inplace=False`.
    None
                        If no soma was found and `inplace=False`.
    MeshNeuron
                        If `inplace=True` (with `.soma_pos` set).

    See Also
    --------
    [`navis.find_soma`][]
                        The equivalent for skeletons (TreeNeurons).

    Examples
    --------
    >>> import navis
    >>> m = navis.example_neurons(1, kind='mesh')
    >>> soma = navis.find_soma_mesh(m)
    >>> soma                                                    # doctest: +SKIP
    SomaEllipsoid(center=[...], radii=[...], inscribed_radius=..., n_vertices=...)
    >>> # Set the soma position on the neuron
    >>> _ = navis.find_soma_mesh(m, inplace=True)
    >>> m.soma_pos is not None
    True

    """
    import ncollpyde

    if not isinstance(x, core.MeshNeuron):
        raise TypeError(f'Expected MeshNeuron, got "{type(x)}"')

    def _finish(ell):
        if inplace:
            x.soma_pos = None if ell is None else ell.center
            x._soma_ellipsoid = ell
            return x
        return ell

    V = np.asarray(x.vertices, dtype=float)
    F = np.asarray(x.faces)
    if len(V) < min_vertices:
        return _finish(None)

    # Unit-aware radius threshold (e.g. "1 micron" -> 125 for 8nm voxels)
    r_min = x.map_units(min_soma_radius, on_error="ignore")
    if not isinstance(r_min, (int, float)) or isinstance(r_min, bool):
        raise ValueError(
            f"Unable to interpret `min_soma_radius={min_soma_radius!r}` for "
            f"neuron {x.id}. Pass a plain number in the neuron's coordinate "
            "space or give the neuron isometric `.units`."
        )

    coll = ncollpyde.Volume(V, F, n_rays=n_rays)
    # Cheap, deterministic estimate of the mean edge length, used only as a floor
    # on the grid pitch. The exact `x.sampling_resolution` is an O(faces) pass
    # over every edge and would dominate on very large meshes.
    sr = _estimate_edge_length(V, F)
    lo, hi = V.min(0), V.max(0)
    ext = hi - lo

    # Coarse grid: aim fine enough to land inside a soma of radius `r_min`, but
    # cap the number of samples (and never go finer than the mesh resolution).
    # The fine-refinement stage below recovers precision, so the coarse grid
    # only has to get *close*.
    pitch = max(r_min, sr)
    grid_coarsened = False
    if np.prod(np.ceil(ext / pitch) + 1) > max_points:
        pitch = max(float((np.prod(ext) / max_points) ** (1 / 3)), sr)
        # A regular grid at spacing `pitch` is only guaranteed to sample inside a
        # sphere of radius >= pitch * sqrt(3) / 2. If that exceeds `r_min` a soma
        # near the floor could be stepped over - flag it, but only warn if we
        # actually fail to find a soma (below), to avoid noise on the common
        # large-bbox case where detection still succeeds.
        grid_coarsened = pitch * (3**0.5 / 2) > r_min

    def _no_soma(reason=None):
        # The winding check is an O(faces) operation that only drives a warning,
        # so we compute it lazily here (on failure) rather than up front on every
        # call - a bad winding is a likely culprit when detection fails.
        parts = [reason] if reason else []
        try:
            if not bool(x.trimesh.is_winding_consistent):
                parts.append(
                    "The mesh is not winding-consistent, which can make the "
                    "inside/outside test unreliable."
                )
        except Exception:
            pass
        if grid_coarsened:
            parts.append(
                f"The search grid was coarsened (pitch {pitch:.0f}) because the "
                "bounding box is large relative to `min_soma_radius`, so a small "
                "soma may have been missed - try increasing `max_points` or "
                "`min_soma_radius`."
            )
        if parts:
            logger.warning(f"No soma detected for neuron {x.id}. " + " ".join(parts))
        return _finish(None)

    P, depth = _interior_depth_field(coll, lo, hi, pitch)
    if not len(P):
        return _no_soma(
            "No interior sample points were found - the mesh may be too thin, "
            "too coarsely sampled, or not closed."
        )
    c0, R0 = P[depth.argmax()], float(depth.max())

    # Fine refine in a small box around the anchor to sharpen center + radius.
    pitch2 = max(sr / 2, R0 / 12)
    P2, depth2 = _interior_depth_field(coll, c0 - 2 * R0, c0 + 2 * R0, pitch2)
    if not len(P2):
        P2, depth2 = P, depth
    c1, R_max = P2[depth2.argmax()], float(depth2.max())

    # Rejection gates (all must pass):
    #  1. inscribed radius big enough to be a soma
    #  2. enough surrounding surface vertices (a tube has far fewer than a blob)
    #  3. at least a couple of genuinely deep samples (not a lone spurious voxel)
    # Soma-surface vertices: those within `dist_factor * R_max` of the center. A
    # bounding-box pre-filter followed by an exact radius test keeps this O(V)
    # with a small constant - building a KDTree over every vertex (O(V log V))
    # just for a single ball query is wasteful and dominates on large meshes.
    r = dist_factor * R_max
    box_idx = np.flatnonzero(np.all(np.abs(V - c1) <= r, axis=1))
    sub = V[box_idx] - c1
    inlier = box_idx[np.einsum("ij,ij->i", sub, sub) <= r * r]
    n_cand = int((depth2 >= 0.5 * R_max).sum())
    if R_max < r_min or len(inlier) < min_vertices or n_cand < 2:
        return _no_soma()

    radii, axes = _fit_soma_ellipsoid(V[inlier], np.asarray(c1, dtype=float))
    ell = SomaEllipsoid(
        center=np.asarray(c1, dtype=float),
        radii=radii,
        axes=axes,
        inscribed_radius=R_max,
        n_vertices=len(inlier),
    )
    return _finish(ell)