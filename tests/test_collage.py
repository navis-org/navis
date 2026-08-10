"""Tests for the collage/grid layouts.

The layouts themselves are `navis-fastcore` (rasterise, then pack); what is checked
here is the navis half — that the page, the views, the scaling and the neuron
handling all line up, and above all that the invariants the layouts promise actually
hold on real neurons: nothing overlaps, nothing leaves the page, nothing leaves the
mask, and the neurons you passed in come back unmodified.
"""

import matplotlib

# Use a headless backend so tests don't try to open windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import navis
from navis.plotting import collage as C


@pytest.fixture(scope="module")
def skels():
    return navis.example_neurons(5)


@pytest.fixture(scope="module")
def mesh():
    return navis.NeuronList([navis.example_neurons(1, kind="mesh")])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


PAGE = (8.27, 11.69)


def page_extent(ax, view=("x", "-y"), page=PAGE):
    """The page as `((xlo, xhi), (ylo, yhi))`, sign of the view applied."""
    sx = -1 if view[0].startswith("-") else 1
    sy = -1 if view[1].startswith("-") else 1
    return (0, sx * page[0]), (0, sy * page[1])


# -----------------------------------------------------------------------------
# View bookkeeping
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "view,want",
    [
        (("x", "-y"), (0, 1, 2, 1, -1)),
        (("x", "y"), (0, 1, 2, 1, 1)),
        (("-z", "y"), (2, 1, 0, -1, 1)),
        (("y", "x"), (1, 0, 2, 1, 1)),
        (("+x", "-z"), (0, 2, 1, 1, -1)),
    ],
)
def test_view_plane(view, want):
    assert C._view_plane(view) == want


@pytest.mark.parametrize("view", [("x", "x"), ("a", "y"), ("x",), 42, ("x", "y", "z")])
def test_view_plane_rejects_nonsense(view):
    with pytest.raises(ValueError):
        C._view_plane(view)


@pytest.mark.parametrize("color,expect_list", [
    ("red", False),                       # a name
    ((1, 0, 0), False),                   # an (r, g, b)
    ((1, 0, 0, 0.5), False),              # an (r, g, b, a)
    (None, False),
    (["red", "blue", "green"], True),     # one per neuron
])
def test_per_neuron_colors(color, expect_list):
    got = C._per_neuron_colors(color, 3)
    assert (got is not None) == expect_list


def test_per_neuron_colors_checks_the_count():
    with pytest.raises(ValueError, match="each of the 3 neurons"):
        C._per_neuron_colors(["red", "blue"], 3)


# -----------------------------------------------------------------------------
# Grid
# -----------------------------------------------------------------------------


def test_grid_places_one_neuron_per_cell(skels):
    fig, ax, placed = navis.plot_collage(skels, cols=2, soma=False)
    assert len(placed) == len(skels)
    assert (ax.get_xlim(), ax.get_ylim()) == page_extent(ax)

    # Three rows of two cells; every neuron inside its own cell
    cell = np.array([PAGE[0] / 2, PAGE[1] / 3])
    for i, n in enumerate(placed):
        row, col = divmod(i, 2)
        centre = n.bbox.mean(axis=1)[[0, 1]] * [1, -1]  # view is ("x", "-y")
        want = ((col + 0.5) * cell[0], PAGE[1] - (row + 0.5) * cell[1])
        assert np.allclose(centre, want, atol=1e-6), f"neuron {i} is not in its cell"


def test_grid_leaves_the_originals_alone(skels):
    before = [n.nodes[["x", "y", "z"]].values.copy() for n in skels]
    navis.plot_collage(skels, soma=False)
    for n, was in zip(skels, before):
        assert np.array_equal(n.nodes[["x", "y", "z"]].values, was)


def test_grid_replays_a_layout(skels):
    _, _, placed = navis.plot_collage(skels, cols=2, soma=False)
    before = [n.nodes[["x", "y", "z"]].values.copy() for n in placed]
    _, _, again = navis.plot_collage(placed=placed, color="r", soma=False)
    for n, was in zip(again, before):
        assert np.array_equal(n.nodes[["x", "y", "z"]].values, was)


def test_grid_drops_dangling(skels):
    _, _, placed = navis.plot_collage(skels, cols=2, drop_dangling=True, soma=False)
    assert len(placed) == 4


def test_grid_uniform_scale_keeps_relative_sizes(skels):
    _, _, varied = navis.plot_collage(skels, cols=3, soma=False)
    _, _, uniform = navis.plot_collage(
        skels, cols=3, uniform_scale=True, soma=False
    )
    orig = C._view_sizes(skels, ("x", "-y"))
    got = C._view_sizes(uniform, ("x", "-y"))
    # One scale for all of them means every ratio is preserved
    ratios = got / orig
    assert np.allclose(ratios, ratios[0], rtol=1e-9)
    # ...whereas the default scales each to its own cell, so they are not
    assert not np.allclose(
        (C._view_sizes(varied, ("x", "-y")) / orig),
        (C._view_sizes(varied, ("x", "-y")) / orig)[0],
        rtol=1e-6,
    )


def test_grid_sorts_largest_first(skels):
    _, _, placed = navis.plot_collage(skels, cols=1, sort=True, soma=False)
    sizes = C._view_sizes(skels, ("x", "-y")).prod(axis=1)
    assert [n.id for n in placed] == [skels[i].id for i in np.argsort(-sizes)]


def test_grid_needs_something_to_plot():
    with pytest.raises(ValueError, match="Need either"):
        navis.plot_collage(None)


def test_grid_handles_an_empty_list():
    fig, ax, placed = navis.plot_collage(navis.NeuronList([]))
    assert len(placed) == 0
    assert (ax.get_xlim(), ax.get_ylim()) == page_extent(ax)


# -----------------------------------------------------------------------------
# Collage
# -----------------------------------------------------------------------------


def bboxes(nl, view=("x", "-y")):
    """Each neuron's `(xlo, ylo, xhi, yhi)` in page coordinates."""
    ix, iy, _, sx, sy = C._view_plane(view)
    out = []
    for n in nl:
        b = n.bbox
        xs = sorted([sx * b[ix, 0], sx * b[ix, 1]])
        ys = sorted([sy * b[iy, 0], sy * b[iy, 1]])
        out.append((xs[0], ys[0], xs[1], ys[1]))
    return np.array(out)


def assert_on_the_page(nl, page=PAGE, view=("x", "-y"), tol=1e-6):
    b = bboxes(nl, view)
    assert (b[:, 0] >= -tol).all() and (b[:, 1] >= -tol).all(), "off the left/bottom"
    assert (b[:, 2] <= page[0] + tol).all(), "off the right"
    assert (b[:, 3] <= page[1] + tol).all(), "off the top"


def assert_boxes_disjoint(nl, view=("x", "-y"), tol=1e-9):
    b = bboxes(nl, view)
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            overlap = (
                b[i, 0] < b[j, 2] - tol
                and b[j, 0] < b[i, 2] - tol
                and b[i, 1] < b[j, 3] - tol
                and b[j, 1] < b[i, 3] - tol
            )
            assert not overlap, f"neurons {i} and {j} overlap"


def test_collage_boxes_fit_and_do_not_overlap(skels):
    scale, centres, rotated, fits = C._layout_boxes(
        skels, navis.NeuronList([]), ("x", "-y"), np.array(PAGE), 0.02, False, 18
    )
    placed = [
        C._place(n, scale, c, ("x", "-y"), rotate=r)
        for n, c, r in zip(skels, centres, rotated)
    ]
    assert_on_the_page(placed)
    assert_boxes_disjoint(placed)


def test_collage_occupancy_packs_tighter_than_boxes(skels):
    page = np.array(PAGE)
    box = C._layout_boxes(skels, navis.NeuronList([]), ("x", "-y"), page, 0.02, False, 18)
    occ = C._layout_occupancy(
        skels, navis.NeuronList([]), ("x", "-y"), page, 0.02, False, 100, 6,
        scale_lo=box[0],
    )
    assert occ is not None
    assert occ[0] > box[0], "packing the arbors should allow a larger scale"


def test_collage_occupancy_neurons_do_not_collide(skels):
    """The arbors themselves, not just the boxes, have to stay apart."""
    page = np.array(PAGE)
    res, padding = 100, 0.02
    scale, centres, rotated, _ = C._layout_occupancy(
        skels, navis.NeuronList([]), ("x", "-y"), page, padding, False, res, 6
    )
    placed = navis.NeuronList([
        C._place(n, scale, c, ("x", "-y"), rotate=r)
        for n, c, r in zip(skels, centres, rotated)
    ])

    # Rasterise the placed neurons onto one page and check nothing lands twice
    page_px = tuple(np.ceil(page * res).astype(int)[::-1])
    canvas = np.zeros(page_px, dtype=bool)
    for n in placed:
        co = C._page_coords(n, ("x", "-y"))
        px = np.rint(co * res).astype(int)
        keep = (
            (px[:, 0] >= 0) & (px[:, 0] < page_px[1])
            & (px[:, 1] >= 0) & (px[:, 1] < page_px[0])
        )
        px = px[keep]
        hit = canvas[px[:, 1], px[:, 0]]
        assert not hit.any(), "two neurons share a pixel"
        canvas[px[:, 1], px[:, 0]] = True


def test_segments_of_a_mesh_are_unique_edges(mesh):
    """Walking the faces would list every interior edge twice, for the same pixels."""
    m = mesh[0]
    edges = C._segments(m)
    faces = np.asarray(m.faces)
    assert len(edges) < len(faces) * 3 / 1.5, "expected roughly half of the directed edges"
    # Every edge is unique as an unordered pair, and names a real vertex
    pairs = np.sort(edges, axis=1)
    assert len(np.unique(pairs, axis=0)) == len(pairs)
    assert edges.max() < len(m.vertices)


def test_move_takes_the_connectors_along(skels):
    """Placing a neuron must not leave its connectors where the neuron used to be."""
    n = skels[0].copy()
    assert n.has_connectors, "this test needs a neuron with connectors"
    nodes = n.nodes[["x", "y", "z"]].values.copy()
    cn = n.connectors[["x", "y", "z"]].values.copy()

    C._move(n, np.array([1.0, 2.0, 3.0]))

    assert np.allclose(n.nodes[["x", "y", "z"]].values - nodes, [1, 2, 3])
    assert np.allclose(n.connectors[["x", "y", "z"]].values - cn, [1, 2, 3])


def test_collage_leaves_the_originals_alone(skels):
    before = [n.nodes[["x", "y", "z"]].values.copy() for n in skels]
    navis.plot_collage(skels, layout="dense", soma=False, occupancy=True)
    for n, was in zip(skels, before):
        assert np.array_equal(n.nodes[["x", "y", "z"]].values, was)


def test_collage_backfill_does_not_change_the_scale(skels):
    page = np.array(PAGE)
    alone = C._layout_boxes(
        skels[:3], navis.NeuronList([]), ("x", "-y"), page, 0.02, False, 18
    )
    withfill = C._layout_boxes(
        skels[:3], skels[3:], ("x", "-y"), page, 0.02, False, 18
    )
    assert alone[0] == withfill[0], "backfill must not influence the scale"
    # The backfill neurons are appended after the main ones
    assert len(withfill[1]) == 3 + int(withfill[3].sum())


def test_collage_rotation_buys_a_bigger_scale(skels):
    """Allowing a quarter turn can only help, and on A4 it does."""
    page = np.array(PAGE)
    upright = C._layout_boxes(
        skels, navis.NeuronList([]), ("x", "-y"), page, 0.02, False, 18
    )
    turned = C._layout_boxes(
        skels, navis.NeuronList([]), ("x", "-y"), page, 0.02, True, 18
    )
    assert not upright[2].any(), "nothing may be turned when rotation is off"
    assert turned[2].any(), "expected at least one neuron to be turned"
    assert turned[0] > upright[0], "turning some of them should allow a larger scale"

    placed = [
        C._place(n, turned[0], c, ("x", "-y"), rotate=r)
        for n, c, r in zip(skels, turned[1], turned[2])
    ]
    assert_on_the_page(placed)
    assert_boxes_disjoint(placed)


@pytest.mark.parametrize("view", [("x", "-y"), ("x", "y"), ("-z", "-y"), ("y", "x")])
def test_page_turn_matches_rotate90(skels, view):
    """The mask must stand in for the neuron as drawn, in every view.

    `_rotate90` turns the raw coordinates; the rasteriser turns the page ones. A view
    with one negative axis mirrors between the two, which swaps the handedness - get
    it wrong and the packed mask is a point reflection of the neuron plotted.
    """
    n = skels[0].copy()
    turned = C._rotate90(n, view)
    want = C._page_coords(turned, view)

    upright = C._page_coords(skels[0], view)
    got = upright.copy()
    for _ in range(C._page_turn(view)):
        got = np.column_stack([-got[:, 1], got[:, 0]])

    # Both are only ever used after a shift to their own corner, so compare that way
    assert np.allclose(want - want.min(axis=0), got - got.min(axis=0))


def test_collage_rotate90_is_a_quarter_turn(skels):
    n = skels[0].copy()
    before = C._page_coords(n, ("x", "-y"))
    C._rotate90(n, ("x", "-y"))
    after = C._page_coords(n, ("x", "-y"))
    # A quarter turn swaps the extents
    assert np.allclose(np.ptp(after, axis=0), np.ptp(before, axis=0)[::-1])
    # ...and preserves every pairwise distance
    d0 = np.linalg.norm(before[:50] - before[:50].mean(axis=0), axis=1)
    d1 = np.linalg.norm(after[:50] - after[:50].mean(axis=0), axis=1)
    assert np.allclose(d0, d1)


@pytest.mark.parametrize("view", [("x", "-y"), ("x", "y"), ("-z", "-y"), ("y", "x")])
def test_collage_honours_the_view(skels, view):
    fig, ax, _ = navis.plot_collage(skels, layout="dense", view=view, soma=False)
    assert (ax.get_xlim(), ax.get_ylim()) == page_extent(ax, view)


def test_collage_handles_meshes_and_a_mix(skels, mesh):
    fig, ax, _ = navis.plot_collage(mesh, layout="dense", occupancy=True)
    assert (ax.get_xlim(), ax.get_ylim()) == page_extent(ax)

    mixed = navis.NeuronList(list(skels[:2]) + list(mesh))
    fig, ax, _ = navis.plot_collage(mixed, layout="dense", occupancy=True, soma=False)
    assert (ax.get_xlim(), ax.get_ylim()) == page_extent(ax)


def test_collage_handles_an_empty_list():
    fig, ax, _ = navis.plot_collage(navis.NeuronList([]), layout="dense")
    assert (ax.get_xlim(), ax.get_ylim()) == page_extent(ax)


def test_collage_checks_the_colour_count(skels):
    with pytest.raises(ValueError, match="each of the 5 neurons"):
        navis.plot_collage(skels, layout="dense", color=["r", "g"])
    # With backfill the colours must cover both sets
    with pytest.raises(ValueError, match="each of the 5 neurons"):
        navis.plot_collage(skels[:3], layout="dense", backfill=skels[3:], color=["r", "g", "b"])


def test_collage_padding_that_cannot_work_raises(skels):
    with pytest.raises(ValueError, match="smaller `padding`"):
        navis.plot_collage(skels, layout="dense", padding=100)


# -----------------------------------------------------------------------------
# Masks
# -----------------------------------------------------------------------------


def square_image(n=100):
    """A black square on white paper - dark is the shape."""
    img = np.ones((n, n))
    img[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 0.0
    return img


def ring_mask(page=PAGE, res=100):
    """An annulus as a page-sized bool mask."""
    h, w = int(np.ceil(page[1] * res)), int(np.ceil(page[0] * res))
    rows, cols = np.ogrid[0:h, 0:w]
    r = np.hypot((rows - h / 2) / h, (cols - w / 2) / w)
    return (r > 0.12) & (r < 0.42)


def test_mask_from_image_array():
    mask = C._mask_from_image(square_image(), page_size=(4, 4), res=50)
    assert mask.shape == (200, 200)
    assert mask.dtype == bool
    assert 0.2 < mask.mean() < 0.3, f"expected ~25% coverage, got {mask.mean():.2%}"


def test_mask_from_image_inverts():
    img = square_image()
    normal = C._mask_from_image(img, page_size=(4, 4), res=50)
    flipped = C._mask_from_image(img, page_size=(4, 4), res=50, invert=True)
    assert not (normal & flipped).any()


def test_mask_from_image_rejects_bad_input():
    with pytest.raises(ValueError, match="2d or 3d"):
        C._mask_from_image(np.zeros((2, 2, 2, 2)))


def test_mask_from_image_reads_a_file(tmp_path):
    path = tmp_path / "square.png"
    plt.imsave(path, square_image(), cmap="gray")
    from_file = C._mask_from_image(str(path), page_size=(4, 4), res=50)
    from_array = C._mask_from_image(square_image(), page_size=(4, 4), res=50)
    assert np.array_equal(from_file, from_array)


def test_resolve_mask_tells_a_page_mask_from_a_picture():
    """A bool array is already a mask; anything else is a picture of one."""
    page_mask = ring_mask()
    assert C._resolve_mask(page_mask, PAGE) is page_mask
    assert C._resolve_mask(None, PAGE) is None

    # A float array of the same shape is a *picture*, so it is scaled onto the page
    as_image = C._resolve_mask(square_image(), PAGE)
    assert as_image.dtype == bool
    assert as_image.shape == (int(np.ceil(PAGE[1] * 100)), int(np.ceil(PAGE[0] * 100)))
    assert np.array_equal(as_image, C._mask_from_image(square_image(), PAGE))


def test_collage_takes_a_mask_as_an_image(skels, tmp_path):
    path = tmp_path / "square.png"
    plt.imsave(path, square_image(), cmap="gray")
    fig, ax, placed = navis.plot_collage(
        skels, layout="dense", mask=str(path), color="k", soma=False
    )
    assert len(placed) == len(skels)
    assert (ax.get_xlim(), ax.get_ylim()) == page_extent(ax)


def test_mask_confines_the_neurons(skels):
    """Nothing may be drawn outside the mask."""
    page = np.array(PAGE)
    res = 100
    mask = ring_mask(PAGE, res)
    layout = C._layout_occupancy(
        skels, navis.NeuronList([]), ("x", "-y"), page, 0.02, False, res, 6, mask=mask
    )
    assert layout is not None, "the neurons should fit into this ring"
    scale, centres, rotated, _ = layout
    placed = [
        C._place(n, scale, c, ("x", "-y"), rotate=r)
        for n, c, r in zip(skels, centres, rotated)
    ]

    page_px = tuple(np.ceil(page * res).astype(int)[::-1])
    allowed = C._resample_mask(mask, page_px)
    for i, n in enumerate(placed):
        co = C._page_coords(n, ("x", "-y"))
        px = np.rint(co * res).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, page_px[1] - 1)
        px[:, 1] = np.clip(px[:, 1], 0, page_px[0] - 1)
        assert allowed[px[:, 1], px[:, 0]].all(), f"neuron {i} escaped the mask"


def test_resample_mask_stretches_to_the_page():
    small = np.zeros((10, 10), dtype=bool)
    small[5:, :] = True
    out = C._resample_mask(small, (100, 100))
    assert out.shape == (100, 100)
    assert not out[:50].any() and out[50:].all()


def test_resample_mask_warns_on_a_wrong_aspect(caplog):
    C._resample_mask(np.zeros((10, 10), dtype=bool), (100, 50))
    assert any("aspect ratio" in r.message for r in caplog.records)


def test_resample_mask_rejects_non_2d():
    with pytest.raises(ValueError, match="2d array"):
        C._resample_mask(np.zeros((4, 4, 4), dtype=bool), (4, 4))


# -----------------------------------------------------------------------------
# Dispatch and backends
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["grid", "dense"])
def test_plot_collage_dispatches_on_layout(skels, layout):
    fig, ax, placed = navis.plot_collage(skels, layout=layout, color="k", soma=False)
    assert len(placed) == len(skels)
    assert (ax.get_xlim(), ax.get_ylim()) == page_extent(ax)
    assert_on_the_page(placed)


def test_plot_collage_layouts_really_differ(skels):
    """A grid keeps its cells; a dense pack does not."""
    _, _, grid = navis.plot_collage(skels, layout="grid", soma=False)
    _, _, dense = navis.plot_collage(skels, layout="dense", soma=False)
    g = np.array([n.bbox.mean(axis=1) for n in grid])
    d = np.array([n.bbox.mean(axis=1) for n in dense])
    assert not np.allclose(g, d)


@pytest.mark.parametrize("bad,kwargs", [
    ("layout", dict(layout="spiral")),
    ("backend", dict(backend="crayon")),
])
def test_plot_collage_rejects_unknown_choices(skels, bad, kwargs):
    with pytest.raises(ValueError, match=bad):
        navis.plot_collage(skels, **kwargs)


def test_plot_collage_needs_something_to_plot():
    with pytest.raises(ValueError, match="Need either"):
        navis.plot_collage(None)


@pytest.mark.parametrize("layout", ["grid", "dense"])
def test_placed_replays_either_layout(skels, layout):
    _, _, placed = navis.plot_collage(skels, layout=layout, soma=False)
    before = [n.nodes[["x", "y", "z"]].values.copy() for n in placed]
    # The layout is skipped entirely, so the layout argument no longer matters
    _, _, again = navis.plot_collage(placed=placed, color="r", soma=False)
    for n, was in zip(again, before):
        assert np.array_equal(n.nodes[["x", "y", "z"]].values, was)


octarine_only = pytest.mark.skipif(
    "octarine" not in navis.plotting.ddd.BACKENDS, reason="octarine not installed"
)


@octarine_only
@pytest.mark.parametrize("layout", ["grid", "dense"])
def test_octarine_backend_renders_onto_the_page(skels, layout):
    fig, ax, placed = navis.plot_collage(
        skels, layout=layout, backend="octarine", color="w", dpi=60
    )
    assert (ax.get_xlim(), ax.get_ylim()) == page_extent(ax)
    assert len(ax.images) == 1, "the render should arrive as a single image"

    # The image is placed in data coordinates, i.e. on the page - which is what
    # lets matplotlib overlays line up with it
    x0, x1, y0, y1 = ax.images[0].get_extent()
    assert -0.1 <= x0 < x1 <= PAGE[0] + 0.1
    assert -PAGE[1] - 0.1 <= y1 < y0 <= 0.1, "view is ('x', '-y'), so y runs negative"
    assert ax.images[0].get_array().any(), "the render is blank"


@octarine_only
def test_backend_does_not_change_the_layout(skels):
    """The neurons are placed before anything is drawn, so both must agree."""
    _, _, mpl = navis.plot_collage(
        skels, layout="dense", occupancy=True, allow_rotation=True, soma=False
    )
    _, _, oct_ = navis.plot_collage(
        skels, layout="dense", occupancy=True, allow_rotation=True,
        backend="octarine", dpi=60,
    )
    for a, b in zip(mpl, oct_):
        assert np.allclose(
            a.nodes[["x", "y", "z"]].values, b.nodes[["x", "y", "z"]].values
        )


@octarine_only
def test_octarine_replays_a_matplotlib_layout(skels):
    _, _, placed = navis.plot_collage(skels, layout="dense", soma=False)
    fig, ax, again = navis.plot_collage(
        placed=placed, backend="octarine", color="w", dpi=60
    )
    assert len(ax.images) == 1
    for a, b in zip(placed, again):
        assert np.array_equal(
            a.nodes[["x", "y", "z"]].values, b.nodes[["x", "y", "z"]].values
        )
