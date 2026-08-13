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

"""Stripping the surface a neuron mesh keeps on its inside.

Segmented neurons come with a lot of membrane that is not the cell's outside:
mitochondria and vesicles floating free in the cytoplasm, and - worse - the ones
touching the membrane from within, which push it into an *invagination*. On the
mesh both read as surface; to anything that walks that surface an invagination
is a tunnel, which is why a wavefront skeletonization comes back covered in
bristles.

The property that defines them - the cell encloses them - is also what makes
them invisible from outside, so that is what gets asked. Both functions here are
thin wrappers around `navis-fastcore`, which does the ray casting:
[`navis.openness`][] is the question on its own and [`navis.drop_internals`][]
is the repair built on it.

"""

import numpy as np

from .. import config, utils
from ..core import schema

__all__ = ["openness", "drop_internals"]

logger = config.get_logger(__name__)


def openness(x, mask=None, n_rays=16, offset=None, seed=1985, threads=None):
    """Fraction of rays leaving each face that escape the mesh.

    For every face, this fires a cosine-weighted spray of rays into the
    hemisphere above it and counts how many get away. Outer membrane lands at
    0.5-1.0 and the wall of an invagination at 0, with little in between - so
    the value is really a two-way classification, and a threshold anywhere in
    the gap does the same thing.

    Useful on its own to see *where* a mesh is bad before deciding what to do
    about it - colour a mesh by this and the invaginations light up.

    Parameters
    ----------
    x :         navis.Mesh | navis.Volume | trimesh.Trimesh
                Mesh to score. Faces must be wound outward - see the warning in
                [`navis.drop_internals`][].
    mask :      (F, ) bool array, optional
                Cast only from these faces. The whole mesh still blocks rays -
                only the *sources* are restricted - so each value is exactly the
                one a full sweep would have produced.
    n_rays :    int
                Rays per face. 16 is plenty: the signal is bimodal, so this only
                has to resolve "none got out" from "some did".
    offset :    float, optional
                How far off the surface to start each ray. Defaults to 5% of the
                median edge length, which keeps this scale-free.
    seed :      int
                Fixes the spray.
    threads :   int, optional
                Size of the thread pool. Defaults to all available cores.

    Returns
    -------
    (F, ) float array
                Fractions in [0, 1], one per face - or one per selected face, in
                face order, if `mask` was given.

    Examples
    --------
    >>> import navis
    >>> import numpy as np
    >>> m = navis.example_neurons(1, kind='mesh')
    >>> op = navis.openness(m)
    >>> len(op) == m.n_faces
    True
    >>> # This mesh is clean, so almost every face can see out
    >>> bool((op > 0).mean() > 0.95)
    True

    See Also
    --------
    [`navis.drop_internals`][]
                The repair this scores for: cut the buried faces away and close
                the mesh back up.

    """
    if not utils.is_mesh(x):
        raise TypeError(f'Expected mesh-like, got "{type(x)}"')

    return utils.fastcore.openness(
        np.asarray(x.vertices),
        np.asarray(x.faces),
        mask=mask,
        n_rays=n_rays,
        offset=offset,
        seed=seed,
        threads=threads,
    )


@utils.map_neuronlist(desc="Dropping", allow_parallel=True)
@utils.rebuilds("vertices")
def drop_internals(
    x,
    threshold=0.05,
    n_rays=16,
    smooth=10,
    iterations=3,
    hops=20,
    seed=1985,
    threads=None,
    inplace=False,
):
    """Remove the surface a mesh keeps on its inside.

    Anything the cell encloses goes: free-floating organelles, and the
    invaginations where one has pushed into the membrane from within. Both are
    found the same way - by [`navis.openness`][], which asks whether a face can
    see out - and the holes cutting them leaves are triangulated shut, so the
    result is a surface, not a surface with bites taken out of it.

    This is what to reach for when a mesh skeletonizes into a hairball. On a
    heavily invaginated FAFB neuron it takes the skeleton from 3,227 leafs to
    ~500 without losing a real branch, because each invagination was a tunnel
    the wave front could take a shortcut through.

    Only vertices are ever *removed* - caps re-use the ones already on the
    boundary - so connectors, extra edges, the skeleton correspondence and
    anything you attached yourself come through pointing where they pointed
    before.

    Note this also closes openings the mesh arrived with, e.g. a neurite
    truncated at the edge of the dataset. For skeletonization that is what you
    want: a stump left open splits the wave front just as a pocket mouth does.

    !!! warning "Faces must be wound outward"

        Rays are fired into the hemisphere the face normal points into, so a
        consistently inward-wound mesh reads as entirely buried and comes back
        empty. A mesh wound *inconsistently* is worse, because it fails quietly:
        the faces that disagree read as buried and are cut out of otherwise
        healthy membrane. [`navis.fix_mesh`][] will sort the winding out.

    Parameters
    ----------
    x :         navis.Mesh/List | navis.Volume | trimesh.Trimesh
                Mesh(es) to strip.
    threshold : float
                Faces whose smoothed openness falls below this are cut. The
                operating range is 0.05-0.10; the buried mode sits at exactly
                zero, so any small value works, while above ~0.1 the cut starts
                eating real membrane.
    n_rays :    int
                Rays per face - see [`navis.openness`][].
    smooth :    int
                Diffusion rounds applied to the field before thresholding.
                Thresholding a raw field cuts along a ragged contour; a few
                rounds of this shorten it several fold without moving where the
                cut sits.
    iterations : int
                Passes of the whole cycle. More than one because capping a
                pocket mouth turns a partially-open neighbour into a fully
                buried one, so later passes catch what the first could not see.
    hops :      int, optional
                How far past the previous pass's caps to re-cast. A pass only
                changes the mesh where the last one cut, so the field is only
                stale near the new caps. `None` re-casts everything, which
                costs about twice as much and changes next to nothing.
    seed :      int
                Fixes the ray spray.
    threads :   int, optional
                Size of the thread pool. Defaults to all available cores.
    inplace :   bool
                If True, will strip `x`. If False, will strip and return a copy.

    Returns
    -------
    stripped
                Mesh(es) of the same type as the input.

    Examples
    --------
    >>> import navis
    >>> m = navis.example_neurons(1, kind='mesh')
    >>> stripped = navis.drop_internals(m)
    >>> # This mesh is clean, so there is very little to take away
    >>> bool(stripped.n_faces > m.n_faces * 0.95)
    True

    See Also
    --------
    [`navis.openness`][]
                The test this is built on, on its own - to see where a mesh is
                bad, or to pick a threshold.
    [`navis.drop_fluff`][]
                Drops small *disconnected* pieces by size, wherever they sit.
                The two overlap rather than complement each other: a
                free-floating organelle is both small and inside, so either
                function takes it. This one is the right tool when you want the
                inside cleared regardless of size - and it is the only one that
                reaches an invagination, which is part of the main component and
                so invisible to any connectivity criterion.
    [`navis.fill_holes`][]
                Closes openings without removing anything.

    """
    if not utils.is_mesh(x):
        raise TypeError(f'Expected mesh-like, got "{type(x)}"')

    if not inplace:
        x = x.copy()

    # Extra edges go out of the `.vertices` setter's way, exactly as
    # `simplify_mesh` does: left in place they would be dropped outright the
    # moment the vertex count changed. They go back in un-remapped and are
    # repaired along with everything else. `Volume` and bare trimeshes have none.
    extra_edges = getattr(x, "_extra_edges", None)
    if extra_edges is not None:
        x._extra_edges = None

    vertices, faces, keep, passes = utils.fastcore.drop_internals(
        np.asarray(x.vertices),
        np.asarray(x.faces),
        threshold=threshold,
        n_rays=n_rays,
        smooth=smooth,
        iterations=iterations,
        hops=hops,
        seed=seed,
        threads=threads,
    )

    x.vertices = vertices
    x.faces = np.asarray(faces, dtype=np.int64)

    if extra_edges is not None:
        x._extra_edges = extra_edges

    for i, p in enumerate(passes):
        logger.debug(
            f"drop_internals pass {i}: {p['buried']:,} of {p['faces_before']:,} "
            f"faces buried ({p['recast']:,} re-cast), {p['capped']:,} caps "
            f"-> {p['faces_after']:,} faces"
        )

    # Every surviving vertex *is* one of the old ones - the repair only ever
    # removes, and the caps re-use what was already on the boundary - so this is
    # the strongest of the accounts a rebuild can give. See `schema.Rebuild`.
    return x, schema.Rebuild(kept=keep)
