"""Verify input data and turn into object needed in NestedPfLine instantiation."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ... import tools
from ...tools.types import PintTimeDataframe
from ..commodity import Commodity
from . import pfline


def create_children(data: Mapping | PintTimeDataframe) -> dict[str, pfline.PfLine]:
    """From data, create a dictionary of PfLines. Also, do some data verification."""

    # Turn dataframe into dictionary.
    if isinstance(data, pd.DataFrame):
        data = {colname: data[colname] for colname in data.columns.get_level_values(0).unique()}

    # Create dictionary of PfLines.
    children = {name: pfline.create(child) for name, child in data.items()}

    # Assert valid keys.
    for name in children:
        if not isinstance(name, str):
            raise TypeError(f"Name must be string; got {name} ({type(name)}).")
        elif name in ["w", "q", "p", "r"]:
            raise ValueError("Name cannot be one of 'w', 'q', 'p', 'r'.")

    # Assert number of children.
    if len(children) == 0:
        raise ValueError("Must provide at least 1 child.")

    # Keep only overlapping part of indices.
    idx = tools.index.intersect(child.index for child in children.values())
    if len(idx) == 0:
        raise ValueError("PfLine indices have no overlap.")
    return {name: child.loc[idx] for name, child in children.items()}


def apply_commodity(
    children: dict[str, pfline.PfLine], commodity: Commodity | None
) -> dict[str, pfline.PfLine]:
    """Apply ``commodity`` to ``children``, i.e., convert to correct units and do few checks."""
    return {name: child.set_commodity(commodity) for name, child in children.items()}
