import re
from typing import Optional, Union, Sequence

import numpy as np
import pint

DIMENSIONLESS = pint.Unit("dimensionless")

INF_RE = re.compile(r"(?P<is_neg>-?)inf(inity)?")

ureg = pint.UnitRegistry()
ureg.define("@alias micron = micrometer")


def parse_quantity(item: str) -> pint.Quantity:
    """Parse strings into ``pint.Quantity``, accounting for infinity.

    Parameters
    ----------
    item : str
        A quantity string like those used by ``pint``.

    Returns
    -------
    pint.Quantity
    """
    item = item.strip()
    try:
        q = ureg.Quantity(item)
    except pint.UndefinedUnitError as e:
        first, *other = item.split()
        m = INF_RE.match(first)
        if m is None:
            raise e

        val = float("inf")
        if m.groupdict()["is_neg"]:
            val *= -1
        unit = ureg.Unit(" ".join(other))
        q = ureg.Quantity(val, unit)

    return q


def as_unit(unit: Optional[Union[str, pint.Unit]]) -> pint.Unit:
    """Convert a string (or None) into a ``pint.Unit``

    Parameters
    ----------
    unit : Optional[Union[str, pint.Unit]]

    Returns
    -------
    pint.Unit
        If the ``unit`` argument was ``None``, return dimensionless.
    """
    if unit is None:
        return DIMENSIONLESS

    if isinstance(unit, pint.Unit):
        return unit

    return ureg.Unit(unit)


def reduce_units(units: Sequence[Optional[Union[str, pint.Unit]]]) -> pint.Unit:
    """Reduce a sequence of units or unit-like strings down to a single ``pint.Unit``.

    Dimensionless units are ignored.

    Parameters
    ----------
    units : Sequence[Optional[Union[str, pint.Unit]]]
        ``None`` is treated as dimensionless.

    Returns
    -------
    pint.Unit
        Consensus units of the sequence.

    Raises
    ------
    ValueError
        If more than one non-dimensionless unit is found.
    """
    # use np.unique instead of set operations here,
    # because setting aliases in the registry affects
    # __eq__ (comparisons as used by np.unique) but not
    # __hash__ (as used by sets)
    unit_set = np.unique([DIMENSIONLESS] + [as_unit(u1) for u1 in units])
    if len(unit_set) == 1:
        return DIMENSIONLESS
    actuals = list(unit_set)
    actuals.remove(DIMENSIONLESS)
    if len(actuals) == 1:
        return actuals[0]
    raise ValueError(f"More than one real unit found: {sorted(unit_set)}")
