"""Verify input data and turn into object needed in NestedPfLine instantiation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from ... import tools
from . import classes, create
from .enums import Kind


def children_and_kind(data: Any) -> Tuple[Dict[str, classes.PfLine], Kind]:
    mapping = _mapping(data)
    children = _children(mapping)
    kind = _kind(children)
    return children, kind


def _mapping(data: Any) -> Mapping[Any, Any]:
    """From data, create a mapping."""

    if isinstance(data, Mapping):
        return data

    elif isinstance(data, pd.DataFrame):
        children = {}
        # Get all sub-dataframes (or series) and turn into dictionary.
        for col in data.columns.get_level_values(0).unique():
            children[col] = data[col]
        return children

    raise TypeError(
        f"Parameter ``data`` must be dict (or other Mapping) or pandas.DataFrame; got {type(data).__name__}."
    )


def _children(mapping: Mapping) -> Dict[str, classes.PfLine]:
    """From mapping, create dictionary of PfLines."""

    # Create dictionary of PfLines.
    children = {name: create.pfline(child) for name, child in mapping.items()}

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
    idx = tools.intersect.indices(*[child.index for child in children.values()])
    if len(idx) == 0:
        raise ValueError("PfLine indices have no overlap.")
    return {name: child.loc[idx] for name, child in children.items()}


def _kind(children: Dict[str, classes.PfLine]) -> Kind:
    """Kind of data, based on children."""

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

    return kind
