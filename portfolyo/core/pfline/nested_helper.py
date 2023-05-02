"""Verify input data and turn into object needed in NestedPfLine instantiation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple
from collections import defaultdict
import pandas as pd

from ... import tools
from .base import Kind, PfLine


def make_mapping(data: Any) -> Mapping[Any, Any]:
    """From data, create a mapping."""
    # Shortcut if PfLine is passed.

    if isinstance(data, Mapping):
        return data

    elif isinstance(data, pd.DataFrame):
        children = {}
        # Get all sub-dataframes (or series) and turn into dictionary.
        for col in data.columns.get_level_values(0).unique():
            children[col] = data[col]
        return children

    raise TypeError(
        f"Parameter ``data`` must be dict (or other Mapping) or pandas.DataFrame; got {type(data)}."
    )


def children_and_kind(children: Dict[str, PfLine]) -> Tuple[Dict[str, PfLine], Kind]:
    """Check number, kind, and indices of children; return corrected children and kind."""
    # Number of children.
    if len(children) == 0:
        raise ValueError("Must provide at least 1 child.")
    # Keep only overlapping part of indices.
    idx = tools.intersect.indices(*[child.index for child in children.values()])
    if len(idx) == 0:
        raise ValueError("PfLine indices have no overlap.")
    children = {name: child.loc[idx] for name, child in children.items()}

    # Kind of children.
    kindset = set([child.kind for child in children.values()])
    if len(kindset) != 1:
        kinds1 = defaultdict(list)
        for name, child in children.items():
            kinds1[child.kind].append(name)
        kinds2 = {kind: ", ".join(names) for kind, names in kinds1.items()}
        kinds3 = " and ".join([f"{kind} ({names})" for kind, names in kinds2.items()])
        raise ValueError(f"All children must be of the same kind; found {kinds3}.")
    kind = next(iter(kindset))

    return children, kind


def dataframe(children: Dict[str, PfLine], kind: Kind) -> pd.DataFrame:
    """Create dataframe with aggregated values."""
    # All children have same kind, so same columns. Also, same indices, so same index.
    df = sum(child._df for child in children.values())
    if kind is Kind.COMPLETE:
        df["p"] = df["r"] / df["q"]
    return df
