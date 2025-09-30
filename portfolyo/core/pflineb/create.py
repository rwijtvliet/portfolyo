from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping

import pint

from ...toolsb.types import Col, PintTimeDataframe, PintTimeSeries
from ..commodity import Commodity
from . import flat, flat_helper, nested, nested_helper, pfline

if TYPE_CHECKING:
    from .flat import FlatPfLine
    from .nested import NestedPfLine
    from .pfline import PfLine


def create_pfline(data: Any, commodity: Commodity | None = None) -> PfLine:
    """Create a PfLine instance from the provided data, if possible."""
    # Catch easy cases.
    if isinstance(data, pfline.PfLine):
        return data if data.commodity is commodity else data.set_commodity(commodity)

    # Data must be processed to see, which descendent class we need to return.
    errors = {}
    for name, fn in {"flat": create_flatpfline, "nested": create_nestedpfline}.items():
        # Try passing data to other creation functions.
        try:
            return fn(data, commodity)
        except (ValueError, TypeError, KeyError) as e:
            errors[name] = e
            pass
    errormsg = "\n".join(f"- {name}: {e.args[0]}" for name, e in errors.items())
    raise ValueError(
        f"Cannot create flat or nested PfLine from the provided data, with the following reasons:\n{errormsg}"
    )


def create_flatpfline(
    data: (
        Mapping[Col, PintTimeSeries | pint.Quantity]
        | PintTimeDataframe
        | PintTimeSeries
        | Iterable[PintTimeSeries | pint.Quantity]
    ),
    commodity: Commodity | None,
) -> FlatPfLine:
    """Create a FlatPfLine instance from the provided data, if possible.

    Parameters
    ----------
    data
        Generally: mapping with one or more attributes or items ``w``, ``q``, ``r``, ``p``;
        all timeseries. Most commonly a ``pandas.DataFrame`` or a dictionary of
        ``pandas.Series``, but may also be e.g. another PfLine object.
        If they contain (distinct) ``pint`` data types, ``data`` may also be a single
        ``pandas.Series`` or a collection of ``pandas.Series``.
    commodity, optional (default: none)
        Commodity the data applies to. Used to set units etc.

    Returns
    -------
        Flat portfolio line
    """
    # Catch easy cases.
    if isinstance(data, flat.FlatPfLine):  # data already correct instance; quick-return
        return data.set_commodity(commodity)
    elif isinstance(data, nested.NestedPfLine):  # data is a PfLine, but not a flat one: flatten
        return data.flatten().set_commodity(commodity)

    # Data must be processed to find dataframe and kind.
    df = flat_helper.create_df(data)
    kind = flat_helper.get_kind(df)
    df = flat_helper.apply_commodity(df, commodity)
    return flat.FlatPfLine(df, kind, commodity)


def create_nestedpfline(data: Mapping[str, Any], commodity: Commodity | None) -> NestedPfLine:
    """Create a NestedPfLine instance from the provided data, if possible.

    Parameters
    ----------
    data
        Generally: mapping, between strings (as keys) and portfolio lines, or objects
        that can be converted into portfolio lines (as values).
    commodity, optional (default: none)
        Commodity the data applies to. Used to set units etc.

    Returns
    -------
        Nested portfolio line
    """
    # Catch easy cases.
    if isinstance(data, nested.NestedPfLine):  # data already correct instancE; quick-return
        return data.set_commodity(commodity)
    elif isinstance(data, flat.FlatPfLine):
        raise TypeError("Cannot create nested portfolio line from a flat portfolio line.")

    # Data must be processed to find children and kind.
    children, kind = nested_helper.children_and_kind(data)
    return nested.NestedPfLine(children, kind, commodity)
