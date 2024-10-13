import pint

import numpy as np
import matplotlib as mpl

from dataclasses import dataclass, field
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

        # For method chaining
        return self

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

    # For TreeNeurons
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
    rasterize: bool = False
    view: Tuple[str, str] = ("x", "y")
    figsize: Optional[Tuple[float, float]] = None
    ax: Optional[mpl.axes.Axes] = None
    mesh_shade: bool = False
    non_view_axes3d: Literal["hide", "show", "fade"] = "hide"

    depth_coloring: bool = False
    depth_scale: bool = True


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
    linewidth: Optional[float] = None  # for plotly, linewidth 1 is too thin, we default to 3 in graph_objs.py
    linestyle: str = "-"


@dataclass
class VispySettings(BasePlottingSettings):
    """Additional plotting parameters for Vispy backend."""

    _name = "vispy backend"

    clear: bool = False
    center: bool = True
    combine: bool = False
    title: Optional[str] = None
    viewer: Optional["navis.Viewer"] = None
    shininess: float = 0
    shading: str = "smooth"
    size: Optional[Tuple[int, int]] = (800, 600)
    show: bool = True
    name: Optional[str] = None


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

    # These are viewer-specific settings that we must not pass to the plotting
    # function
    _viewer_settings: tuple[str] = (
        "clear",
        "center",
        "viewer",
        "camera",
        "control",
        "show",
        "size",
        "offscreen",
        "scatter_kws",
        "spacing"
    )


@dataclass
class K3dSettings(BasePlottingSettings):
    """Additional plotting parameters for K3d backend."""

    _name = "k3d backend"

    height: int = 600
    inline: bool = True
    legend_group: Optional[str] = None
    plot: Optional["k3d.Plot"] = None
