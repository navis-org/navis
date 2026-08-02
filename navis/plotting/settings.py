import pint

import numpy as np
import matplotlib as mpl

from dataclasses import dataclass, field, fields
from typing import Union, List, Tuple, Optional
from typing_extensions import Literal

# Global flag whether to validate settings
VALIDATE_SETTINGS = True


@dataclass
class Settings:
    """Class that works a bit like a dictionary but can validate keys and has some extra features."""

    # We can define synonyms for arguments, so that they can be used interchangeably
    _synonyms: List[Tuple] = field(default_factory=list)
    _name = "Settings"

    def __setattr__(self, key, value, check_valid=False):
        if check_valid and key not in self.properties:
            raise AttributeError(
                f"The '{key}' argument is invalid for {self._name}. Valid arguments are: {', '.join(self.properties)}"
            )
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.properties

    @property
    def properties(self):
        return tuple(
            [
                p
                for p in dir(self)
                if not p.startswith("_")
                and (p != "properties")   # we need this to avoid infinite recursion
                and not callable(getattr(self, p, None))
            ]
        )

    def update_settings(self, **kwargs):
        # Deal with synonyms
        for syn in self._synonyms:
            present = [s for s in syn if s in kwargs]
            if len(present) > 1:
                raise ValueError(f"Multiple synonyms for the same argument: {present}")

            for s in syn[1:]:
                if s in kwargs:
                    kwargs[syn[0]] = kwargs.pop(s)

        for k, v in kwargs.items():
            self.__setattr__(k, v, check_valid=VALIDATE_SETTINGS)

        # Remember which ones were actually asked for - a default and the same
        # value passed explicitly are not always the same thing (see `was_set`)
        self.__dict__.setdefault("_explicit", set()).update(kwargs)

        # For method chaining
        return self

    def was_set(self, key):
        """Whether `key` was passed explicitly rather than left at its default.

        Use this where "the user did not say" has to mean something other than
        the default value does - e.g. a default that only applies when nothing
        else has already settled the question.

        """
        return key in self.__dict__.get("_explicit", ())

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def get(self, key, default=None):
        value = self.__dict__.get(key, default)
        if value is None:
            value = default
        return value

    def pop(self, key, default=None):
        return self.__dict__.pop(key, default)


@dataclass
class BasePlottingSettings(Settings):
    """Plotting parameters common to all functions/backends."""

    _name = "BasePlottingSettings"

    # For Skeletons
    soma: bool = True
    radius: bool = False  # True | False | "auto"
    linewidth: float = 1
    linestyle: str = "-"

    # For Dotprops
    dps_scale_vec: float = "auto"

    # All neurons
    connectors: bool = False
    connectors_only: bool = False
    cn_size: Optional[float] = None
    cn_alpha: Optional[float] = None
    cn_layout: dict = field(default_factory=dict)
    cn_colors: dict = field(default_factory=dict)
    cn_mesh_colors: bool = False
    # Colour connectors by a column of the connector table (or an array of one
    # value per connector) instead of by their `type`.
    cn_color_by: Optional[Union[str, np.ndarray]] = None
    cn_palette: Optional[Union[str, list, dict]] = None
    color: Optional[
        Union[
            str,
            Tuple[float, float, float],
            List[Union[str, Tuple[float, float, float]]],
            dict,
        ]
    ] = None
    color_by: Optional[Union[str, np.ndarray, List[np.ndarray]]] = None
    shade_by: Optional[Union[str, np.ndarray, List[np.ndarray]]] = None
    palette: Optional[Union[str, np.ndarray]] = None
    alpha: Optional[float] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    smin: Optional[float] = None
    smax: Optional[float] = None
    norm_global: bool = True

    # Other
    scatter_kws: dict = field(default_factory=dict)

    _synonyms: List[Tuple] = field(
        default_factory=lambda: [
            ("linestyle", "ls"),
            ("linewidth", "lw"),
            ("color", "colors", "c"),
        ]
    )


@dataclass
class Matplotlib2dSettings(BasePlottingSettings):
    """Additional plotting parameters for Matplotlib 2d backend."""

    _name = "matplotlib backend"

    method: Literal["2d", "3d", "3d_complex"] = "2d"
    group_neurons: bool = False
    autoscale: bool = True
    orthogonal: bool = True
    scalebar: Union[int, float, str, pint.Quantity] = False
    volume_outlines: bool = False
    volume_outlines_alpha: float = 0.001
    rasterize: bool = False
    view: Tuple[str, str] = ("x", "y")
    figsize: Optional[Tuple[float, float]] = None
    ax: Optional[mpl.axes.Axes] = None
    # Surface shading for meshes and volumes. In 2d this takes a mode name (see
    # `dd.MESH_SHADE_MODES`) or a dict with "mode" plus any of "light", "ambient"
    # and "strength"; the 3d methods only understand the bool.
    mesh_shade: Union[bool, str, dict] = False
    non_view_axes3d: Literal["hide", "show", "fade"] = "hide"
    cn_zorder: Optional[int] = None
    cn_legend: bool = False

    depth_coloring: bool = False
    depth_scale: bool = True
    # Normalizer for depth coloring. Generated from the data in `plot2d` unless
    # explicitly provided.
    norm: Optional[mpl.colors.Normalize] = None

    # Named bundle of the settings below - see `dd.PLOT_STYLES`
    style: Optional[str] = None
    # Background-coloured stroke under each neuron: width in points, or a dict
    # with "width" and/or "color"
    halo: Union[bool, float, dict] = False
    # Number of depth bins to interleave neurons by (True picks a default);
    # negative flips which end of the depth axis is nearest the viewer.
    # "global" sorts exactly instead, at the cost of one artist per neuron type
    depth_sort: Union[bool, int, str] = False
    # Width from a topological measure instead of a constant
    taper: Optional[Literal["strahler", "subtree"]] = None


@dataclass
class PlotlySettings(BasePlottingSettings):
    """Additional plotting parameters for Plotly backend."""

    _name = "plotly backend"

    fig: Optional[Union["plotly.Figure", dict]] = None
    inline: bool = True
    title: Optional[str] = None
    fig_autosize: bool = True
    hover_name: Optional[str] = False
    hover_id: bool = False
    legend: bool = True
    legend_orientation: Literal["h", "v"] = "v"
    legend_group: Optional[str] = None
    volume_legend: bool = False
    width: Optional[int] = None
    height: Optional[int] = 600
    # Deliberate sentinel: plotly's two consumers of `linewidth` want different
    # defaults - 3 for a skeleton's line width (1 is too thin to see) and 1 for
    # the radius multiplier used when `radius=True`. Both defaults live at their
    # call site in graph_objs.py, so the field itself can't state either.
    linewidth: Optional[float] = None
    linestyle: str = "-"

    # Surface shading / scene styling (see graph_objs.py for presets)
    lighting: Union[bool, str, dict] = True
    lightposition: Optional[dict] = None
    flatshading: bool = False
    background: Optional[Union[str, dict]] = None
    projection: Literal["perspective", "orthographic"] = "perspective"
    dragmode: Literal["turntable", "orbit"] = "orbit"
    hide_axes: bool = True

    _synonyms: List[Tuple] = field(
        default_factory=lambda: [
            ("linestyle", "ls"),
            ("linewidth", "lw"),
            ("color", "colors", "c"),
            ("background", "bg"),
            ("projection", "proj"),
        ]
    )


@dataclass
class OctarineSettings(BasePlottingSettings):
    """Additional plotting parameters for Octarine backend."""

    _name = "octarine backend"

    clear: bool = False
    center: bool = True
    viewer: Optional[Union["octarine.Viewer", Literal["new"]]] = None
    random_ids: bool = False
    camera: Literal["ortho", "perspective"] = "ortho"
    control: Literal["trackball", "panzoom", "fly", "orbit"] = "trackball"
    show: bool = True
    size: Optional[Tuple[int, int]] = None
    offscreen: bool = False
    spacing: Optional[Tuple[float, float, float]] = None

    # `snapshot=True` swaps the interactive viewer for a matplotlib figure with
    # the rendered image on it - see `plotting/render.py`. The settings below it
    # only have an effect in that case.
    snapshot: bool = False
    # "auto" means: default view for a new viewer, leave an existing one alone.
    # `None` never touches the camera.
    view: Optional[Union[str, Tuple[str, str], dict]] = "auto"
    margin: float = 0.05
    hide_axes: bool = True
    bgcolor: Optional[str] = None
    # Supersampling on top of `size`. `None` leaves pygfx's own default, which
    # is 2 for an offscreen canvas.
    pixel_ratio: Optional[float] = None
    figsize: Optional[Tuple[float, float]] = None
    dpi: Optional[int] = None
    ax: Optional[mpl.axes.Axes] = None

    @property
    def _neuron_settings(self):
        """Names of the settings that describe the neurons rather than the view.

        This is what `octarine.Viewer.add_neurons` is given. It is derived
        rather than listed because `add_neurons` has no `**kwargs`: a window or
        figure option that slipped into it would surface as a `TypeError` from
        octarine, and a hand-maintained list has to be updated for every field
        added here.

        """
        drawing = {
            f.name for f in fields(BasePlottingSettings) if not f.name.startswith("_")
        }
        # `scatter_kws` is for points, not neurons; `cn_color_by`/`cn_palette` are
        # resolved by the backends that can draw per-connector colors, which the
        # octarine plugin cannot; `random_ids` is octarine-only
        return (drawing - {"scatter_kws", "cn_color_by", "cn_palette"}) | {"random_ids"}


@dataclass
class K3dSettings(BasePlottingSettings):
    """Additional plotting parameters for K3d backend."""

    _name = "k3d backend"

    height: int = 600
    inline: bool = True
    legend_group: Optional[str] = None
    plot: Optional["k3d.Plot"] = None
