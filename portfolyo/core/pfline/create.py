from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import classes, flat_helper, nested_helper
from .enums import Kind, Structure  # noqa

if TYPE_CHECKING:
    from .classes import FlatPfLine, NestedPfLine, PfLine


def create_pfline(data: Any) -> PfLine:
    """Create a PfLine instance from the provided data, if possible."""
    if isinstance(data, classes.PfLine):
        # Data already correct instance. Quick-return.
        return data

    # Data must be processed to see, which descendent class we need to return.
    errors = {}
    for name, fn in {"flat": create_flatpfline, "nested": create_nestedpfline}.items():
        # Try passing data to other creation functions.
        try:
            return fn(data)
        except (ValueError, TypeError, KeyError) as e:
            errors[name] = e
            pass
    errormsg = "\n".join(f"- {name}: {e.args[0]}" for name, e in errors.items())
    raise ValueError(
        f"Cannot create flat or nested PfLine, with the following reasons:\n{errormsg}"
    )


def create_flatpfline(data: Any) -> FlatPfLine:
    """Create a FlatPfLine instance from the provided data, if possible.

    Parameters
    ----------
    data: Any
        Generally: mapping with one or more attributes or items ``w``, ``q``, ``r``, ``p``;
        all timeseries. Most commonly a ``pandas.DataFrame`` or a dictionary of
        ``pandas.Series``, but may also be e.g. another PfLine object.
        If they contain a (distinct) ``pint`` data type, ``data`` may also be a single
        ``pandas.Series`` or a collection of ``pandas.Series``.

    Returns
    -------
    FlatPfLine
    """
    # In some situations, no processing is needed to return a flat pfline.
    if isinstance(data, classes.FlatPfLine):
        # Data already correct instance. Quick-return.
        return data
    elif isinstance(data, classes.PfLine):
        # The data is a PfLine, but not a flat one.
        return data.flatten()

    # Data must be processed to see, which descendent class we need to return.
    df, kind = flat_helper.dataframe_and_kind(data)
    if cls := classes.constructor(Structure.FLAT, kind):
        return cls(df)
    raise ValueError(
        f"Did not find a flat portfolio line class for the kind of the data ({kind})."
    )


def create_nestedpfline(data: Any) -> NestedPfLine:
    """Create a NestedPfLine instance from the provided data, if possible."""
    if isinstance(data, classes.NestedPfLine):
        # Data already correct instance. Quick-return.
        return data
    elif isinstance(data, classes.PfLine):
        # The data is a PfLine, but not a nested one.
        raise TypeError(
            "Cannot create nested portfolio line from a flat portfolio line."
        )

    # Data must be processed to see, which descendent class we need to return.
    children, kind = nested_helper.children_and_kind(data)
    if cls := classes.constructor(Structure.NESTED, kind):
        return cls(children)
    raise ValueError(
        f"Did not find a nested portfolio line class for the kind of the data ({kind})."
    )
