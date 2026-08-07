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

import copy
import numbers
import pint
import uuid
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from io import StringIO

from typing import Union, List, Optional, Any
from typing_extensions import Literal

from .. import utils, config, core
from . import schema

__all__ = ["Neuron"]

# Set up logging
logger = config.get_logger(__name__)

# This is to prevent pint to throw a warning about numpy integration
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])


#: Sentinel for "no attribute", so that an attribute set to `None` is not
#: mistaken for a missing one.
_MISSING = object()


def _try_getattr(obj, key: str):
    """Fetch `key` off `obj`, treating "not implemented here" as absent.

    `hasattr` only swallows `AttributeError`, so a placeholder like
    `Voxels.soma` - inherited from `BaseNeuron` and raising
    `NotImplementedError` - makes `.has_soma`/`.n_soma` explode instead of
    answering the question they were asked.
    """
    try:
        return getattr(obj, key)
    except (AttributeError, NotImplementedError):
        return _MISSING


def _class_attr(cls, key: str):
    """Look `key` up on `cls` and its bases without invoking any descriptor.

    Sits on the attribute-miss path, which `hasattr` probing makes hot, so the
    cheap C-level `getattr` rejects the common miss first. The class dicts then
    confirm the hit: unlike `getattr` (or `inspect.getattr_static`, which is
    ~5x slower again) they don't see metaclass attributes such as `type.mro`,
    which an instance could never reach anyway.
    """
    if getattr(cls, key, _MISSING) is _MISSING:
        return _MISSING
    for klass in cls.__mro__:
        attr = vars(klass).get(key, _MISSING)
        if attr is not _MISSING:
            return attr
    return _MISSING


def _implements(cls, key: str) -> bool:
    """Test whether `cls` itself provides `key`.

    `hasattr(cls, key)` is too generous: `BaseNeuron` declares placeholders
    (`soma`, `bbox`) that only raise `NotImplementedError` and every subclass
    inherits them. Converting to a neuron type that merely inherits the
    placeholder gains you nothing, so only count what the class brings itself.
    """
    attr = _class_attr(cls, key)
    return attr is not _MISSING and attr is not _class_attr(BaseNeuron, key)


def _extension_bytes(column: pd.Series) -> int:
    """Size of a column whose dtype has no dependable width.

    pandas' extension dtypes cannot be priced from the dtype alone: a
    `StringDtype` exposes no `itemsize`, an Arrow-backed one reports `0`, and a
    categorical's real cost is its codes, whose width pandas picks by a rule of
    its own. So ask pandas rather than keeping a copy of how it stores things -
    modelling another library's layout is what broke this in the first place.

    It answers from the array's buffers, and `deep=False` is what keeps that
    O(1) instead of walking the values.
    """
    return int(column.memory_usage(index=False, deep=False))


#: Columns of `BaseNeuron.attached`, written once so the builder, the link rows
#: and the empty frame cannot drift apart.
ATTACHED_COLUMNS = ["name", "kind", "axis", "names", "shape"]


def _attachment(name: str, kind: str, axis: str, value, names: str = "") -> dict:
    """One row of `BaseNeuron.attached`."""
    return {
        "name": name,
        "kind": kind,
        "axis": axis,
        "names": names,
        "shape": None if value is None else tuple(np.shape(value)),
    }


def _sizeof(value, deep: bool, estimate: bool, seen: set) -> int:
    """Bytes held by a value, following the containers a neuron nests things in.

    A neuron holds more than its own tables. A mesh keeps its skeleton - that is
    the whole point of the vertex-to-node link - a masked neuron keeps the
    snapshot it will be restored from, and provenance keeps an array per axis.
    None of those are arrays themselves, and all of them are real memory.

    Derived caches built by other libraries (`_trimesh`, the igraph and networkx
    views) are still not counted: they are `TEMP_ATTR`, so they come and go with
    any change, and pricing them would mean modelling someone else's layout.
    """
    if isinstance(value, np.ndarray):
        return value.nbytes

    # An ndarray costs the same either way, so only the pandas containers branch
    # on `estimate`. Estimating prices each column from its dtype - much faster
    # than walking it - and materialises the column only for the dtypes that
    # cannot be priced that way. Iterating `dtypes` rather than columns is the
    # point: building a Series per column costs more than the sizing itself.
    if isinstance(value, pd.DataFrame):
        if not estimate:
            return int(value.memory_usage(deep=deep).sum())
        return sum(
            dtype.itemsize * value.shape[0]
            if isinstance(dtype, np.dtype)
            else _extension_bytes(value[col])
            for col, dtype in value.dtypes.items()
        )

    if isinstance(value, pd.Series):
        if not estimate:
            return int(value.memory_usage(deep=deep))
        if isinstance(value.dtype, np.dtype):
            return value.dtype.itemsize * value.shape[0]
        return _extension_bytes(value)

    if isinstance(value, (BaseNeuron, schema.Provenance)):
        # By identity, so a neuron reachable twice is paid for once - and so
        # that a cycle cannot send us round for ever
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return _sizeof(vars(value), deep, estimate, seen)

    # Containers, but only over the items that could hold anything. A skeleton's
    # `tags` is a dict of lists of node IDs: descending blindly costs one call
    # per tagged node to be told that each one is free, and `memory_usage` is on
    # the `NeuronList` repr path. `__dict__` catches the two things worth
    # descending into that are not containers at all - a nested neuron, and its
    # provenance.
    if isinstance(value, dict):
        value = value.values()
    elif not isinstance(value, (list, tuple)):
        return 0

    return sum(
        _sizeof(v, deep, estimate, seen)
        for v in value
        if type(v) in _BULK_TYPES or hasattr(v, "__dict__")
    )


#: Types `_sizeof` prices directly, for the exact-type fast path above.
_BULK_TYPES = frozenset((np.ndarray, pd.DataFrame, pd.Series, dict, list, tuple))


def Neuron(
    x: Union[nx.DiGraph, str, pd.DataFrame, "Skeleton", "Mesh"], **metadata
):
    """Constructor for Neuron objects. Depending on the input, either a
    `Skeleton` or a `Mesh` is returned.

    Parameters
    ----------
    x
                        Anything that can construct a [`navis.Skeleton`][]
                        or [`navis.Mesh`][].
    **metadata
                        Any additional data to attach to neuron.

    See Also
    --------
    [`navis.read_swc`][]
                        Gives you more control over how data is extracted from
                        SWC file.
    [`navis.example_neurons`][]
                        Loads some example neurons provided.

    """
    try:
        return core.Skeleton(x, **metadata)
    except utils.ConstructionError:
        try:
            return core.Mesh(x, **metadata)
        except utils.ConstructionError:
            pass
        except BaseException:
            raise
    except BaseException:
        raise

    raise utils.ConstructionError(f'Unable to construct neuron from "{type(x)}"')


class UnitObject:
    """Base class for things that have units."""

    @property
    def units(self) -> Union[numbers.Number, np.ndarray]:
        """Units for coordinate space."""
        # Note that we are regenerating the pint.Quantity from the string
        # That is to avoid problems with pickling e.g. when using multiprocessing
        unit_str = getattr(self, "_unit_str", None)

        if utils.is_iterable(unit_str):
            values = [config.ureg(u) for u in unit_str]
            conv = [v.to(values[0]).magnitude for v in values]
            return config.ureg.Quantity(np.array(conv), values[0].units)
        else:
            return config.ureg(unit_str)

    @property
    def units_xyz(self) -> np.ndarray:
        """Units for coordinate space. Always returns x/y/z array."""
        units = self.units

        if not utils.is_iterable(units):
            units = config.ureg.Quantity([units.magnitude] * 3, units.units)

        return units

    @units.setter
    def units(self, units: Union[pint.Unit, pint.Quantity, str, None]):
        # Note that we are storing the string, not the actual pint.Quantity
        # That is to avoid problems with pickling e.g. when using multiprocessing

        # Do NOT remove the is_iterable condition - otherwise we might
        # accidentally strip the units from a pint Quantity vector
        if not utils.is_iterable(units):
            units = utils.make_iterable(units)

        if len(units) not in [1, 3]:
            raise ValueError(
                "Must provide either a single unit or one for "
                "for x, y and z dimension."
            )

        # Make sure we actually have valid unit(s)
        unit_str = []
        for v in units:
            if isinstance(v, str):
                # This makes sure we have meters (i.e. nm, um, etc) because
                # "microns", for example, produces odd behaviour like
                # "millimicrons" on division
                v = v.replace("microns", "um").replace("micron", "um")
                unit_str.append(str(v))
            elif isinstance(v, (pint.Unit, pint.Quantity)):
                unit_str.append(str(v))
            elif isinstance(v, type(None)):
                unit_str.append(None)
            elif isinstance(v, numbers.Number):
                unit_str.append(str(config.ureg(f"{v} dimensionless")))
            else:
                raise TypeError(f'Expect str or pint Unit/Quantity, got "{type(v)}"')

        # Some clean-up
        if len(set(unit_str)) == 1:
            unit_str = unit_str[0]
        else:
            # Check if all base units (e.g. "microns") are the same
            unique_units = set([str(config.ureg(u).units) for u in unit_str])
            if len(unique_units) != 1:
                raise ValueError(
                    'Non-isometric units must share the same base,'
                    f' got: {", ".join(unique_units)}'
                )
            unit_str = tuple(unit_str)

        self._unit_str = unit_str

    @property
    def is_isometric(self):
        """Test if neuron is isometric."""
        u = self.units
        if utils.is_iterable(u) and len(set(u)) > 1:
            return False
        return True


class BaseNeuron(UnitObject):
    """Base class for all neurons."""

    name: Optional[str]
    id: Union[int, str, uuid.UUID]

    #: Unit space for this neuron. Some functions, like soma detection are
    #: sensitive to units (if provided)
    #: Default = micrometers
    units: Union[pint.Unit, pint.Quantity]

    volume: Union[int, float]

    connectors: Optional[pd.DataFrame]

    #: Attributes used for neuron summary
    SUMMARY_PROPS = ["type", "name", "units"]

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ["name"]

    #: Temporary attributes that need clearing when neuron data changes
    TEMP_ATTR = ["_memory_usage"]

    #: Core data table(s) used to calculate hash
    CORE_DATA = []

    #: Element axes: which attributes are aligned to which axis, and what
    #: references them. See `navis/core/schema.py`. Empty means this neuron
    #: type cannot be subset element-wise.
    AXES = {}

    #: Correspondences between one of our axes and an axis of another
    #: representation - a mesh's vertex-to-node map. See `navis/core/schema.py`.
    #: Empty means this neuron type has no other representation to carry.
    LINKS = ()

    #: Data attributes that are deliberately *not* aligned to any axis, e.g. a
    #: mesh's soma position (a coordinate, not a vertex index). Declaring them
    #: keeps `test_schema_is_complete` honest: anything array-like that is
    #: neither declared in an axis, nor temporary, nor listed here is a field
    #: that would silently survive a subset unfiltered.
    AXIS_INDEPENDENT = ()

    def __init__(self, **kwargs):
        # Set a random ID -> may be replaced later
        self.id = uuid.uuid4()

        # Make a copy of summary and temp props so that if we register
        # additional properties we don't change this for every single neuron
        self.SUMMARY_PROPS = self.SUMMARY_PROPS.copy()
        self.TEMP_ATTR = self.TEMP_ATTR.copy()

        self._lock = 0
        for k, v in kwargs.items():
            self._register_attr(name=k, value=v)

        # Base neurons has no data
        self._current_md5 = None

    def __getattr__(self, key):
        """Get attribute."""
        if key.startswith("has_"):
            key = key[key.index("_") + 1 :]
            data = _try_getattr(self, key)
            if data is not _MISSING:
                if isinstance(data, pd.DataFrame):
                    if not data.empty:
                        return True
                    else:
                        return False
                # This is necessary because np.any does not like strings
                elif isinstance(data, str):
                    if data == "NA" or not data:
                        return False
                    return True
                elif utils.is_iterable(data) and len(data) > 0:
                    return True
                elif data:
                    return True
            return False
        elif key.startswith("n_"):
            key = key[key.index("_") + 1 :]
            data = _try_getattr(self, key)
            if data is not _MISSING:
                if isinstance(data, pd.DataFrame):
                    return data.shape[0]
                elif utils.is_iterable(data):
                    return len(data)
                elif isinstance(data, str) and data == "NA":
                    return "NA"
            return None

        # Private/dunder misses are the hot path here - `copy`, `pickle` and
        # `hasattr` probe a lot of them - so don't pay for the nicer message.
        if key.startswith("_"):
            raise AttributeError(f'Attribute "{key}" not found')

        # A property that raises AttributeError itself (e.g. `Mesh.soma`, which
        # points you at `.soma_pos`) also ends up here - Python cannot tell that
        # apart from a genuine miss. Re-run the descriptor so its own, more
        # specific error survives instead of being replaced by the message below.
        # Anything the class declares must be a descriptor that raised: a plain
        # class attribute would have been returned before `__getattr__` ran.
        descr = _class_attr(type(self), key)
        if descr is not _MISSING:
            return descr.__get__(self, type(self))

        raise AttributeError(self._missing_attr_msg(key))

    def _missing_attr_msg(self, key: str) -> str:
        """Explain a missing attribute in terms of the neuron types that have it.

        A bare `Attribute "cable_length" not found` is a dead end: it doesn't
        say that the attribute exists on a *different* neuron type, nor how to
        get one. Reaching for a skeleton-only property on a mesh or dotprops is
        common enough to be worth naming the way out.
        """
        converters = {
            core.Skeleton: "navis.skeletonize(x)",
            core.Mesh: "navis.mesh(x)",
            core.Voxels: "navis.voxelize(x, pitch=...)",
            core.Dotprops: "navis.make_dotprops(x)",
        }
        others = [
            cls
            for cls in converters
            if not isinstance(self, cls) and _implements(cls, key)
        ]

        msg = f'{type(self).__name__} has no attribute "{key}".'
        if not others:
            return msg

        names = " or ".join(cls.__name__ for cls in others)
        msg += f" It is available on {names}"
        if len(others) == 1:
            msg += f" - convert with `{converters[others[0]]}`."
        else:
            msg += " - convert first."
        return msg

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.summary())

    def __copy__(self):
        return self.copy(deepcopy=False)

    def __deepcopy__(self, memo):
        result = self.copy(deepcopy=True)
        memo[id(self)] = result
        return result

    def __eq__(self, other):
        """Implement neuron comparison."""
        if isinstance(other, BaseNeuron):
            # We will do this sequentially and stop as soon as we find a
            # discrepancy -> this saves tons of time!
            for at in self.EQ_ATTRIBUTES:
                comp = getattr(self, at, None) == getattr(other, at, None)
                if isinstance(comp, np.ndarray):
                    if not comp.all():
                        return False
                # `not comp` also catches numpy scalars (np.bool_(False) is
                # not False, so the old `comp is False` check missed them)
                elif not comp:
                    return False
            # If all comparisons have passed, return True
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Generate a hashable value."""
        # We will simply use the neuron's memory address
        return id(self)

    def __add__(self, other):
        """Implement addition."""
        if isinstance(other, BaseNeuron):
            return core.NeuronList([self, other])
        return NotImplemented

    def __imul__(self, other):
        """Multiplication with assignment (*=)."""
        return self.__mul__(other, copy=False)

    def __itruediv__(self, other):
        """Division with assignment (/=)."""
        return self.__truediv__(other, copy=False)

    def __iadd__(self, other):
        """Addition with assignment (+=)."""
        return self.__add__(other, copy=False)

    def __isub__(self, other):
        """Subtraction with assignment (-=)."""
        return self.__sub__(other, copy=False)

    def _repr_html_(self):
        frame = self.summary().to_frame()
        frame.columns = [""]
        # return self._gen_svg_thumbnail() + frame._repr_html_()
        return frame._repr_html_()

    def _gen_svg_thumbnail(self):
        """Generate 2D plot for thumbnail."""
        import matplotlib.pyplot as plt

        prev_int = plt.isinteractive()
        plt.ioff()  # turn off interactive mode
        try:
            with config.quiet_logger(level="WARNING", pbars=True):
                fig = plt.figure(figsize=(2, 2))
                ax = fig.add_subplot(111)
                fig, ax = self.plot2d(connectors=False, ax=ax)
                output = StringIO()
                fig.savefig(output, format="svg")
        finally:
            # Without the `finally`, a failed plot leaves the session with
            # interactive mode off and a stray figure holding the canvas
            if prev_int:
                plt.ion()  # turn on interactive mode
            _ = plt.clf()
        return output.getvalue()

    def _clear_temp_attr(self, exclude: list = []) -> None:
        """Clear temporary attributes."""
        if self.is_locked:
            logger.debug(f"Neuron {self.id} at {hex(id(self))} locked.")
            return

        # Must set checksum before recalculating e.g. node types
        # -> otherwise we run into a recursive loop
        self._current_md5 = self.core_md5
        self._stale = False

        for a in [at for at in self.TEMP_ATTR if at not in exclude]:
            try:
                delattr(self, a)
                logger.debug(f"Neuron {self.id} {hex(id(self))}: attribute {a} cleared")
            except AttributeError:
                logger.debug(
                    f'Neuron {self.id} at {hex(id(self))}: Unable to clear temporary attribute "{a}"'
                )
            except BaseException:
                raise

    def _register_attr(self, name, value, summary=True, temporary=False):
        """Set and register attribute.

        Use this if you want an attribute to be used for the summary or cleared
        when temporary attributes are cleared.
        """
        setattr(self, name, value)

        # If this is an easy to summarize attribute, add to summary
        if summary and name not in self.SUMMARY_PROPS:
            if isinstance(value, (numbers.Number, str, bool, np.bool_, type(None))):
                self.SUMMARY_PROPS.append(name)
            else:
                logger.error(
                    f'Attribute "{name}" of type "{type(value)}" '
                    "can not be added to summary"
                )

        if temporary:
            self.TEMP_ATTR.append(name)

    def _unregister_attr(self, name):
        """Remove and unregister attribute."""
        if name in self.SUMMARY_PROPS:
            self.SUMMARY_PROPS.remove(name)

        if name in self.TEMP_ATTR:
            self.TEMP_ATTR.remove(name)

        delattr(self, name)

    @property
    def core_md5(self) -> str:
        """MD5 checksum of core data.

        Generated from `.CORE_DATA` properties.

        Returns
        -------
        md5 :   string
                MD5 checksum of core data. `None` if no core data.

        """
        hash = ""
        for prop in self.CORE_DATA:
            cols = None
            # See if we need to parse props into property and columns
            # e.g. "nodes:node_id,parent_id,x,y,z"
            if ":" in prop:
                prop, cols = prop.split(":")
                cols = cols.split(",")

            data = getattr(self, prop, None)
            # `None` is not hashable as an array (it would become a 0-d object
            # array) and means "not set", which is itself part of the state we
            # are describing - so skip it rather than blowing up
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    if cols:
                        data = data[cols]
                    data = data.values

                hash += schema.hash_array(data)

        return hash if hash else None

    @property
    def datatables(self) -> List[str]:
        """Names of all DataFrames attached to this neuron."""
        return [k for k, v in self.__dict__.items() if isinstance(v, pd.DataFrame)]

    @property
    def extents(self) -> np.ndarray:
        """Extents of neuron in x/y/z direction (includes connectors)."""
        # Not `hasattr`: that only swallows AttributeError, so a neuron type
        # left with `BaseNeuron`'s placeholder would raise NotImplementedError
        # here instead of the explanation below.
        bbox = _try_getattr(self, "bbox")
        if bbox is _MISSING:
            raise ValueError(
                "Neuron must implement `.bbox` (bounding box) "
                "property to calculate extents."
            )
        return bbox[:, 1] - bbox[:, 0]

    @property
    def id(self) -> Any:
        """ID of the neuron.

        Must be hashable. If not set, will assign a random unique identifier.
        Can be indexed by using the `NeuronList.idx[]` locator.
        """
        return getattr(self, "_id", None)

    @id.setter
    def id(self, value):
        try:
            hash(value)
        except BaseException:
            raise ValueError("id must be hashable")
        self._id = value

    @property
    def label(self) -> str:
        """Label (e.g. for legends)."""
        # If explicitly set return that label
        if getattr(self, "_label", None):
            return self._label

        # If no label set, produce one from name + id (optional)
        name = getattr(self, "name", None)
        id = getattr(self, "id", None)

        # If no name, use type
        if not name:
            name = self.type

        label = name

        # Use ID only if not a UUID
        if not isinstance(id, uuid.UUID):
            # And if it can be turned into a string
            try:
                id = str(id)
            except BaseException:
                id = ""

            # Only use ID if it is not the same as name
            if id and name != id:
                label += f" ({id})"

        return label

    @label.setter
    def label(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f'label must be string, got "{type(value)}"')
        self._label = value

    @property
    def name(self) -> str:
        """Neuron name."""
        return getattr(self, "_name", None)

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def connectors(self) -> pd.DataFrame:
        """Connector table. If none, will return `None`."""
        return getattr(self, "_connectors", None)

    #: Columns a connector table must have. A tuple means "any of these", and
    #: the first is what it gets renamed to. Subclasses that need more (a
    #: skeleton's connectors have always had to say which node they sit on)
    #: extend this rather than the setter.
    CONNECTOR_COLUMNS = ["x", "y", "z"]

    @connectors.setter
    def connectors(self, v):
        if isinstance(v, type(None)):
            self.detach("_connectors")
        else:
            if "connectors" not in schema.declared_axes(self):
                # `Voxels` has no schema at all, so nothing has declared the
                # axis its connectors live on. Say so here rather than leaning
                # on `attach` to conjure one, which it deliberately no longer
                # does - an axis it has never heard of is a user's typo far
                # more often than it is a new kind of element.
                schema.declare_axis(self, schema.CONNECTOR_AXIS)
            self.attach(
                "_connectors",
                utils.validate_table(
                    v, required=self.CONNECTOR_COLUMNS, rename=True, restrict=False
                ),
                axis="connectors",
            )

    @property
    def presynapses(self):
        """Table with presynapses (filtered from connectors table).

        Requires a "type" column in connector table. Will look for type labels
        that include "pre" or that equal 0 or "0".
        """
        return self._filter_connectors("pre")

    @property
    def postsynapses(self):
        """Table with postsynapses (filtered from connectors table).

        Requires a "type" column in connector table. Will look for type labels
        that include "post" or that equal 1 or "1".
        """
        return self._filter_connectors("post")

    def _filter_connectors(self, kind: str) -> pd.DataFrame:
        """Filter the connector table down to pre- or postsynapses."""
        if not isinstance(getattr(self, "connectors", None), pd.DataFrame):
            raise ValueError("No connector table found.")

        types = self.connectors["type"].unique()
        # Make an educated guess which label means what
        label = utils.guess_connector_type(types, kind)

        if label is None:
            logger.debug(f"Unable to find {kind}synapses in types: {types}")
            return self.connectors.iloc[0:0]  # return empty DataFrame

        return self.connectors[self.connectors["type"] == label]

    @property
    def is_stale(self) -> bool:
        """Test if temporary attributes might be outdated."""
        # If we know we are stale, just return True
        if getattr(self, "_stale", False):
            return True
        else:
            # Only check if we believe we are not stale
            self._stale = self._current_md5 != self.core_md5
        return self._stale

    @property
    def is_locked(self):
        """Test if neuron is locked."""
        return getattr(self, "_lock", 0) > 0

    @property
    def is_masked(self) -> bool:
        """Test if neuron is currently restricted to part of itself.

        See Also
        --------
        [`navis.BaseNeuron.mask`][]
        """
        return bool(getattr(self, "_mask_stack", None))

    def attach(
        self,
        name: str,
        data,
        axis: Optional[str] = None,
        *,
        ids=None,
        on_rebuild: str = "drop",
    ):
        """Attach data that selections should carry along.

        Anything attached this way is subset, filtered and re-indexed by
        `subset_neuron`, `mask` and everything built on them, exactly as the
        neuron's own tables are - it is the same declaration
        (`navis/core/schema.py`), just made per-neuron instead of per-class.

        Parameters
        ----------
        name :      str
                    Attribute to hang the data off, e.g. `n.compartment`.
        data :      array | DataFrame | None
                    One entry per element of `axis`. `None` detaches.
        axis :      str, optional
                    Axis the data is aligned to - `"vertices"`, `"nodes"`,
                    `"connectors"`, ... Must be one the neuron has; a name that
                    is not there raises rather than declaring it, since nothing
                    would ever select it. Leave it out for data that brings its
                    own elements and becomes an axis in its own right.
        ids :       str, optional
                    Column of `data` holding stable element IDs, when it is a
                    new axis. Without it elements are identified by position,
                    which is fine but means references to them have to be
                    remapped rather than merely filtered.
        on_rebuild : "drop" | "carry"
                    What happens when a function *rebuilds* the axis instead of
                    selecting from it - [`resample_skeleton`][navis.resample_skeleton]
                    and friends. Some of the elements are then new, and there is
                    no value to carry onto them, so the default is to let the
                    data go with a warning. `"carry"` keeps the values of the
                    elements the rebuild says it kept, and still drops if any
                    element turns out to be genuinely new.

        Examples
        --------
        Carry a per-vertex label through a mask

        >>> import navis, numpy as np
        >>> n = navis.example_neurons(1, kind='mesh')
        >>> n.attach('compartment', np.arange(n.n_vertices), axis='vertices')
        >>> with navis.masked(n, np.arange(100)):
        ...     len(n.compartment) == n.n_vertices < 100
        True

        See Also
        --------
        [`navis.BaseNeuron.attach_link`][]
                    For data that *names* elements of another axis.

        """
        if data is None:
            return self.detach(name)

        # Anything the class defines, not just a property. Attaching writes an
        # instance attribute, which shadows a method as silently as it shadows a
        # property - and `n.attach("attach", ...)` leaves nothing able to attach
        # anything ever again. Note this only looks at the class, so re-attaching
        # to a name of your own is still fine.
        if hasattr(type(self), name):
            raise AttributeError(
                f'Cannot attach to "{name}": {type(self).__name__} already '
                f"defines it. Pick a name of your own."
            )

        axis = axis or name
        if not isinstance(data, pd.DataFrame):
            data = np.asarray(data)

        declared = schema.declared_axes(self)
        if axis not in declared:
            if axis != name:
                # An `axis` that is not there is a typo far more often than it
                # is intent, and a silent one: the data would be declared
                # against elements nothing ever selects, so nothing would ever
                # carry it - which is the whole reason to attach it. Bringing
                # elements of its own is the one thing that legitimately makes
                # an axis, and that is spelled by leaving `axis` alone.
                raise KeyError(
                    f'{type(self).__name__} has no "{axis}" axis '
                    f'(has: {sorted(declared)}). Omit `axis` if "{name}" '
                    "brings elements of its own and should become one."
                )
            schema.declare_axis(self, schema.Axis(name=axis, data=(name,), ids=ids))
        else:
            existing = declared[axis]
            if name == existing.data[0]:
                # The axis' own elements are being replaced - same door as the
                # `.nodes`/`.vertices`/`.points` setters use
                self._replacing(axis, data)
            elif len(data) != schema.axis_length(self, existing):
                # Anything else has to line up with the elements already there
                raise ValueError(
                    f'"{name}" has {len(data)} entries but axis "{axis}" has '
                    f"{schema.axis_length(self, existing)} elements."
                )

        setattr(self, name, data)
        schema.declare_aligned(self, axis, name, on_rebuild)
        schema.stamp_links(self, axis)
        # Attaching does not go through `_clear_temp_attr` - it changes what the
        # neuron carries, not how it is built - but it does change its size
        self.__dict__.pop("_memory_usage", None)

    def _replacing(self, axis_name: str, replacement) -> None:
        """Deal with attached data before an axis' elements are replaced.

        The `.nodes`, `.vertices` and `.points` setters predate the schema and
        write their private attribute directly, so nothing else gets a chance.
        Without this, data attached to the old elements is left exactly where it
        was - at the old length, describing elements that are gone, and indexing
        cleanly enough that nothing ever complains.
        """
        if schema.is_replacing(self, axis_name):
            # Somebody who knows better is driving - see `schema.replacing`
            return
        axis = schema.declared_axes(self).get(axis_name)
        if axis is not None:
            self._orphan_aligned(axis, replacement)

    def _attached_aligned(self, axis) -> list:
        """Attributes aligned to an axis that were attached, not declared.

        A [`Dotprops`][navis.Dotprops]' tangent vectors are aligned to its points
        and are the class' own business: the setter that maintains them is the
        one asking us, and orphaning them would take away data the type needs to
        function. Only what `attach` put there is ours to carry or drop.
        """
        declared = type(self).AXES.get(axis.name)
        companions = set(declared.data) if declared is not None else set()
        return [
            attr
            for attr in axis.data[1:]
            if attr not in companions and getattr(self, attr, None) is not None
        ]

    def _orphan_aligned(self, axis, replacement) -> None:
        """Deal with data aligned to an axis whose elements are being replaced.

        Assigning is not selecting: nothing here can say where the old elements
        went, so anything describing them has to go. The one case we *can* tell
        apart is an id-bearing axis whose new elements are a subset of the old
        ones - that is a selection written as an assignment, and the IDs say
        exactly which survived, so the aligned data is carried instead of
        dropped.

        Note what this must *not* do: infer that a reused ID is the same element.
        A function that rebuilds an axis is free to mint IDs however it likes, so
        only the subset case - where every new element was already there - is
        safe to read as a selection. Anything else says so through
        [`schema.apply_rebuild`][] or gets dropped.
        """
        aligned = self._attached_aligned(axis)
        if not aligned:
            return

        if axis.positional:
            # No identity to compare, so the count is the only signal there is -
            # the same guess reference repair has always made.
            if len(replacement) == schema.axis_length(self, axis):
                return
        else:
            # One hash join answers all three questions - are the new elements
            # all old ones, are they unique, and where did each come from - where
            # `np.isin` plus `np.unique` would sort the IDs twice over.
            was = pd.Index(schema.axis_ids(self, axis))
            now = np.asarray(replacement[axis.ids].values)
            where = was.get_indexer(now)
            if (where >= 0).all() and len(np.unique(where)) == len(now):
                for attr in aligned:
                    # Through `_select_aligned`, which knows a DataFrame from an
                    # array - `attach` accepts either
                    setattr(
                        self,
                        attr,
                        schema._select_aligned(getattr(self, attr), where, None),
                    )
                return

        for attr in aligned:
            self.detach(attr)
        logger.warning(
            f"Replacing {type(self).__name__} {self.id}'s '{axis.name}' with "
            f"different elements: dropped {', '.join(aligned)}, which described "
            "the old ones. Select the axis instead of assigning to it if you "
            "want them carried."
        )

    def attach_link(
        self,
        name: str,
        mapping: str,
        *,
        source: str,
        target_axis: str,
        target: str = "",
        column: Optional[str] = None,
        cascade: str = "propagate",
        dangling: str = "drop",
        on_rebuild: str = "drop",
    ):
        """Declare that some of this neuron's data names elements of an axis.

        A link is one array wearing two hats: aligned to `source`, so a
        selection there carries it, and naming elements of `target_axis`, so a
        selection *there* filters and re-indexes it. See
        `navis/core/schema.py`.

        Parameters
        ----------
        name :          str
                        Names the far end, for `get_mapping` and friends.
        mapping :       str
                        Attribute the values live in - `"vertex_id"`, or the
                        table they are a column of when `column` is given.
        source :        str
                        Axis the mapping is aligned to.
        target_axis :   str
                        Axis its values name.
        target :        str, optional
                        Attribute holding the object that owns `target_axis`.
                        Empty means this neuron.
        column :        str, optional
                        Column of `mapping` holding the values, when it is a
                        table rather than an array of its own.
        cascade :       "propagate" | "keep"
                        What a selection of `source` does to the far end.
        dangling :      "drop" | "blank"
                        What becomes of a source element whose target is gone.
        on_rebuild :    "drop" | "snap"
                        What happens when a function *rebuilds* the target axis
                        instead of selecting from it. A different question from
                        `dangling`: the target is not gone, it moved, and
                        `"snap"` follows it to wherever the rebuild says
                        references should now point. This is what keeps a
                        connector on its branch through
                        [`resample_skeleton`][navis.resample_skeleton].

        Examples
        --------
        A mitochondria table that sits on the nodes, and goes when they do

        >>> import navis, numpy as np, pandas as pd
        >>> n = navis.example_neurons(1, kind='skeleton')
        >>> mito = pd.DataFrame({'mito_id': np.arange(20),
        ...                      'node_id': n.nodes.node_id.values[:20]})
        >>> n.attach('mito', mito, ids='mito_id')
        >>> n.attach_link('nodes', 'mito', column='node_id',
        ...               source='mito', target_axis='nodes', cascade='keep')

        See Also
        --------
        [`navis.BaseNeuron.attach`][]
                    For data that is merely aligned to an axis.

        """
        link = schema.Link(
            name=name,
            source=source,
            mapping=mapping,
            column=column,
            target_axis=target_axis,
            target=target,
            cascade=cascade,
            dangling=dangling,
            on_rebuild=on_rebuild,
        )

        # A link is identified by `source->name`, so reusing both replaces
        # whatever the class declared under them. Pointing the replacement at
        # the *same* values is the supported way to change a built-in link's
        # policy and is silent; pointing it somewhere else stops the built-in
        # being maintained at all, which is silent and destructive - a
        # skeleton's connectors simply stop being pruned.
        replaced = next((lk for lk in type(self).LINKS if lk.key == link.key), None)
        if replaced is not None and (link.mapping, link.column) != (
            replaced.mapping,
            replaced.column,
        ):
            logger.warning(
                f'Link "{link.key}" replaces the one {type(self).__name__} '
                f"declares over {replaced.where}, which will no longer be "
                f'maintained - nothing will repair it when "{link.target_axis}" '
                f"changes. Give this link a name of its own unless you mean to "
                f"replace it; to change only the policy of the built-in one, "
                f"declare it over the same mapping."
            )

        schema.declare_link(self, link)
        schema.stamp_link(self, link)

    def detach(self, name: str):
        """Drop attached data, and any links or declarations that named it."""
        self.__dict__.pop("_memory_usage", None)
        for axis in schema.declared_axes(self).values():
            if name not in axis.data:
                continue
            if name != axis.data[0]:
                schema.undeclare_aligned(self, axis.name, name)
                continue
            if axis.name in type(self).AXES:
                # The class declared this axis, so emptying its table is not the
                # same as taking the axis away - a neuron with no connectors is
                # still a neuron that *can* have them.
                setattr(self, name, None)
                schema.stamp_links(self, axis.name)
                return
            # An axis this neuron alone declared goes away with its data
            self.__dict__.get("_axes", {}).pop(axis.name, None)

        schema.undeclare_link(self, name)
        self.__dict__.pop(name, None)

    def attached(self) -> pd.DataFrame:
        """What `attach` and `attach_link` have put on this neuron.

        Only what *this* neuron carries beyond its type: the axes every
        [`Skeleton`][navis.Skeleton] has are a property of the class, not
        something to report per neuron. Attached data is otherwise invisible -
        it is a plain attribute, and nothing in the summary mentions it.

        Returns
        -------
        pandas.DataFrame
                    One row per attachment, with columns:

                    - `name`: the attribute it hangs off, or - for a link -
                      the `source->target` key `get_mapping` addresses it by
                    - `kind`: `"aligned"` (one value per element of `axis`),
                      `"axis"` (elements of its own), or `"link"`
                    - `axis`: the axis it is aligned to
                    - `names`: for a link, the axis its values name
                    - `shape`: shape of the data, `None` if it is not set -
                      which for a link is what says it cannot be followed yet

        Examples
        --------
        >>> import navis, numpy as np
        >>> n = navis.example_neurons(1, kind='skeleton')
        >>> n.attach('score', np.arange(n.n_nodes), axis='nodes')
        >>> n.attached()[['name', 'kind', 'axis']]
            name     kind   axis
        0  score  aligned  nodes

        See Also
        --------
        [`navis.NeuronList.attached`][]
                    The same, summarised over a list of neurons.

        """
        return pd.DataFrame(self._attachment_rows(), columns=ATTACHED_COLUMNS)

    def _attachment_rows(self) -> List[dict]:
        """The rows of `attached`, before they become a frame.

        Separate because `NeuronList.attached` wants every neuron's rows and one
        frame at the end - building an empty one per neuron and throwing it away
        is most of what a list of untouched neurons would otherwise cost.
        """
        rows = []
        class_axes = type(self).AXES
        for name, axis in schema.declared_axes(self).items():
            # An axis the class did not declare is itself an attachment, so its
            # primary table counts too - which `data[0] not in own` says exactly.
            own = set(class_axes[name].data) if name in class_axes else set()
            rows.extend(
                _attachment(
                    attr,
                    "axis" if i == 0 else "aligned",
                    name,
                    getattr(self, attr, None),
                )
                for i, attr in enumerate(axis.data)
                if attr not in own
            )

        for link in self.__dict__.get("_links", ()):
            names = (
                link.target_axis
                if not link.target
                else f"{link.target}.{link.target_axis}"
            )
            rows.append(
                _attachment(
                    link.key,
                    "link",
                    link.source,
                    schema.link_mapping(self, link),
                    names=names,
                )
            )
        return rows

    def get_mapping(self, source: str, target: str) -> np.ndarray:
        """Map every element of one endpoint onto an element of another.

        Composes links, so a correspondence nobody declared directly - a mesh's
        connectors onto its skeleton's nodes, via the vertices they sit on - is
        still available. Links describe what is there; they do not build it, so
        this raises rather than generating a mapping that is missing.

        Parameters
        ----------
        source :    str
                    Endpoint to map *from*: one of this neuron's axes, or the
                    name of a link on it.
        target :    str
                    Endpoint to map *to*.

        Returns
        -------
        np.ndarray
                    One entry per `source` element, in its order, naming a
                    `target` element - an ID where the target axis has IDs, an
                    index where it does not. `-1` where it maps to nothing.

        Raises
        ------
        navis.core.schema.MappingError
                    If the mapping is not there, or describes elements that have
                    changed since.

        Examples
        --------
        Which skeleton node does each of a mesh's vertices belong to? Note the
        skeleton has to exist before there is a mapping onto it to read.

        >>> import navis
        >>> m = navis.example_neurons(1, kind='mesh')
        >>> m.skeleton is not None
        True
        >>> len(m.get_mapping('vertices', 'skeleton')) == m.n_vertices
        True

        See Also
        --------
        [`navis.BaseNeuron.select_across`][]
                    For the other direction, which is a selection rather than a
                    mapping.

        """
        return schema.get_mapping(self, source, target)

    def select_across(self, source: str, target: str, selection) -> np.ndarray:
        """Which `source` elements map into a selection of `target` elements.

        Links are directed and only followed forwards - a mesh vertex has one
        skeleton node, but a node has many vertices, so backwards is not a
        mapping at all. As a *selection* it is perfectly well defined, and that
        is what this answers.

        Parameters
        ----------
        source :    str
                    Endpoint to select *on*, i.e. what the mask is over.
        target :    str
                    Endpoint `selection` refers to.
        selection : list | np.ndarray
                    Anything [`navis.subset_neuron`][] accepts for `target`:
                    element IDs, indices, or a boolean mask.

        Returns
        -------
        np.ndarray
                    Boolean mask over `source`, ready to hand to
                    [`navis.subset_neuron`][] or [`navis.masked`][].

        Examples
        --------
        Which vertices belong to the first 50 nodes of a mesh's skeleton?

        >>> import navis
        >>> m = navis.example_neurons(1, kind='mesh')
        >>> nodes = m.skeleton.nodes.node_id.values[:50]
        >>> int(m.select_across('vertices', 'skeleton', nodes).sum())
        364

        See Also
        --------
        [`navis.BaseNeuron.get_mapping`][]

        """
        return schema.select_across(self, source, target, selection)

    def _adopt(self, other: "BaseNeuron") -> "BaseNeuron":
        """Take over another neuron's state, in place.

        Used by unmasking to swap a neuron's contents wholesale while keeping
        the object itself - the whole point of masking in place is that
        references held elsewhere see the change.
        """
        if other is self:
            return self

        lock = self.__dict__.get("_lock")
        self.__dict__.clear()
        self.__dict__.update(other.__dict__)
        if lock is not None:
            self.__dict__["_lock"] = lock
        return self

    def mask(self, mask, inplace: bool = False, warn_cut: bool = True) -> "BaseNeuron":
        """Restrict this neuron to part of itself.

        The neuron *becomes* the masked region: every property and every navis
        function sees only that part until it is unmasked. Masks nest - each
        `mask()` can be undone by one `unmask()`.

        Parameters
        ----------
        mask :      see [`navis.subset_neuron`][]
                    Anything `subset_neuron` accepts, including a callable that
                    takes this neuron and returns a selection.
        inplace :   bool
                    If False, mask a copy and return it, leaving this neuron
                    untouched.
        warn_cut :  bool
                    Warn if the mask cuts across branches, leaving nodes that
                    look like the ends of the arbour but are not. Silent for
                    masks that keep whole subtrees, e.g. a compartment.

        Returns
        -------
        Neuron
                    The masked neuron.

        See Also
        --------
        [`navis.masked`][]
                    Context manager - prefer it where the mask has a natural
                    scope, since it unmasks even if something raises.
        [`navis.BaseNeuron.unmask`][]
        [`navis.BaseNeuron.apply_mask`][]

        """
        from .masking import mask_neuron

        return mask_neuron(self, mask, inplace=inplace, warn_cut=warn_cut)

    def unmask(self, reset: bool = True, warn_cut: bool = True) -> "BaseNeuron":
        """Undo the innermost mask, in place.

        Parameters
        ----------
        reset :     bool
                    If True, restore the neuron exactly as it was and discard
                    anything done while masked. If False, fold edits made to the
                    masked region back into the whole neuron.
        warn_cut :  bool
                    With `reset=False`, warn if folding the mask back left the
                    neuron in more pieces than it was - usually a leaf-sensitive
                    edit having eroded the mask boundary.

        Returns
        -------
        self

        Raises
        ------
        navis.core.masking.MaskingError
                    If the neuron is not masked.
        navis.core.schema.MergeError
                    With `reset=False`, if the masked region was restructured by
                    something that could not keep track of where its elements
                    came from. Refusing beats folding them back wrongly.

        """
        from .masking import unmask_neuron

        return unmask_neuron(self, reset=reset, warn_cut=warn_cut)

    def apply_mask(self, inplace: bool = False) -> "BaseNeuron":
        """Make the innermost mask permanent.

        The masked region becomes the neuron; there is nothing left to go back
        to.

        Parameters
        ----------
        inplace :   bool
                    If False, return a copy and leave this neuron masked.

        Returns
        -------
        Neuron

        """
        from .masking import apply_mask_neuron

        return apply_mask_neuron(self, inplace=inplace)

    @property
    def type(self) -> str:
        """Neuron type."""
        return "navis.BaseNeuron"

    @property
    def soma(self):
        """The soma of the neuron (if any)."""
        raise NotImplementedError(f"`soma` property not implemented for {type(self)}.")

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box of neuron."""
        raise NotImplementedError(f"Bounding box not implemented for {type(self)}.")

    def convert_units(
        self, to: Union[pint.Unit, str], inplace: bool = False
    ) -> Optional["BaseNeuron"]:
        """Convert coordinates to different unit.

        Only works if neuron's `.units` is not dimensionless.

        Parameters
        ----------
        to :        pint.Unit | str
                    Units to convert to. If string, must be parsable by pint.
                    See examples.
        inplace :   bool, optional
                    If True will convert in place. If not will return a
                    copy.

        Examples
        --------
        >>> import navis
        >>> n = navis.example_neurons(1)
        >>> n.units
        <Quantity(8, 'nanometer')>
        >>> n.cable_length
        266476.8
        >>> n2 = n.convert_units('um')
        >>> n2.units
        <Quantity(1.0, 'micrometer')>
        >>> n2.cable_length
        2131.8

        """
        units = self.units
        # `.units = None` - the default whenever a loader doesn't know better -
        # round-trips to a *dimensionless* Quantity, which sails past an
        # isinstance check and then fails inside pint with "Cannot convert from
        # 'dimensionless' to 'micrometer'": an error that names neither navis
        # nor `.units`, and doesn't hint that they can simply be set.
        if not isinstance(units, (pint.Unit, pint.Quantity)) or units.dimensionless:
            raise ValueError(
                f'Unable to convert to "{to}": this neuron has no units set. '
                "Either assign them first (e.g. `n.units = '8 nanometer'`) or "
                "scale the coordinates directly (`n * 8 / 1000` converts 8nm "
                "voxels to microns)."
            )

        n = self.copy() if not inplace else self

        # Catch pint's UnitStrippedWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Get factor by which we have to multiply to get to target units
            conv = n.units.to(to).magnitude
            # Multiply by conversion factor
            n *= conv

        n._clear_temp_attr(exclude=["classify_nodes"])

        return n

    def copy(self, deepcopy=False) -> "BaseNeuron":
        """Return a copy of the neuron."""
        copy_fn = copy.deepcopy if deepcopy else copy.copy
        # Attributes not to copy
        no_copy = ["_lock"]
        # Generate new empty neuron
        x = self.__class__()
        # Override with this neuron's data
        x.__dict__.update(
            {k: copy_fn(v) for k, v in self.__dict__.items() if k not in no_copy}
        )

        return x

    def summary(self, add_props=None) -> pd.Series:
        """Get a summary of this neuron."""
        # Do not remove the list -> otherwise we might change the original!
        props = list(self.SUMMARY_PROPS)

        # Make sure ID is always in second place
        if "id" in props and props.index("id") != 2:
            props.remove("id")
            props.insert(2, "id")
        # Add .id to summary if not a generic UUID
        elif not isinstance(self.id, uuid.UUID) and "id" not in props:
            props.insert(2, "id")

        if add_props:
            props, ix = np.unique(np.append(props, add_props), return_inverse=True)
            props = props[ix]

        # This is to catch an annoying "UnitStrippedWarning" with pint
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = pd.Series([getattr(self, at, "NA") for at in props], index=props)

        return s

    def plot2d(self, **kwargs):
        """Plot neuron using [`navis.plot2d`][].

        Parameters
        ----------
        **kwargs
                Will be passed to [`navis.plot2d`][].
                See `help(navis.plot2d)` for a list of keywords.

        See Also
        --------
        [`navis.plot2d`][]
                    Function called to generate 2d plot.

        """
        from ..plotting import plot2d

        return plot2d(self, **kwargs)

    def plot3d(self, **kwargs):
        """Plot neuron using [`navis.plot3d`][].

        Parameters
        ----------
        **kwargs
                Keyword arguments. Will be passed to [`navis.plot3d`][].
                See `help(navis.plot3d)` for a list of keywords.

        See Also
        --------
        [`navis.plot3d`][]
                    Function called to generate 3d plot.

        Examples
        --------
        >>> import navis
        >>> nl = navis.example_neurons()
        >>> #Plot with connectors
        >>> viewer = nl.plot3d(connectors=True)

        """
        from ..plotting import plot3d

        return plot3d(core.NeuronList(self, make_copy=False), **kwargs)

    def map_units(
        self,
        units: Union[pint.Unit, str],
        on_error: Union[Literal["raise"], Literal["ignore"]] = "raise",
    ) -> Union[int, float]:
        """Convert units to match neuron space.

        Only works if neuron's `.units` is isometric and not dimensionless.

        Parameters
        ----------
        units :     number | str | pint.Quantity | pint.Units
                    The units to convert to neuron units. Simple numbers are just
                    passed through.
        on_error :  "raise" | "ignore"
                    What to do if an error occurs (e.g. because `neuron` does not
                    have units specified). If "ignore" will simply return `units`
                    unchanged.

        See Also
        --------
        [`navis.core.to_neuron_space`][]
                    The base function for this method.

        Examples
        --------
        >>> import navis
        >>> # Example neurons are in 8x8x8nm voxel space
        >>> n = navis.example_neurons(1)
        >>> n.map_units('1 nanometer')
        0.125
        >>> # Numbers are passed-through
        >>> n.map_units(1)
        1
        >>> # For neuronlists
        >>> nl = navis.example_neurons(3)
        >>> nl.map_units('1 nanometer')
        [0.125, 0.125, 0.125]

        """
        return core.core_utils.to_neuron_space(units, neuron=self, on_error=on_error)

    def memory_usage(self, deep=False, estimate=False):
        """Return estimated memory usage of this neuron.

        Works by going over the data the neuron holds - numpy arrays and pandas
        DataFrames such as vertices, nodes, anything you attached - and summing
        up their size in memory. Data held *inside* something else counts too:
        a mesh's skeleton, the snapshot a masked neuron will be restored from,
        and the arrays provenance keeps per axis. Caches built by other
        libraries (`trimesh`, igraph, networkx) do not - they are temporary and
        rebuilt on demand.

        Parameters
        ----------
        deep :      bool
                    Passed to pandas DataFrames. If True will also inspect
                    memory footprint of `object` dtypes.
        estimate :  bool
                    If True, we will only estimate the size. This is
                    considerably faster but will slightly underestimate the
                    memory usage.

        Returns
        -------
        int
                    Memory usage in bytes.

        """
        # We will use a very simply caching here
        # We don't check whether neuron is stale because that causes
        # additional overhead and we want this function to be as fast
        # as possible
        if hasattr(self, "_memory_usage"):
            mu = self._memory_usage
            if mu["deep"] == deep and mu["estimate"] == estimate:
                return mu["size"]

        size = int(_sizeof(self.__dict__, deep, estimate, {id(self)}))
        self._memory_usage = {"deep": deep, "estimate": estimate, "size": size}

        return size
