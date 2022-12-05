"""Verify input data and turn into object needed in MultiPfLine instantiation."""

from __future__ import annotations

from typing import Any, Counter, Dict, Mapping

import pandas as pd

from ... import tools
from .base import Kind, PfLine


def make_mapping(data: Any) -> Mapping[Any, Any]:
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
        f"Parameter ``data`` must be dict (or other Mapping) or pandas.DataFrame; got {type(data)}."
    )


def verify_and_trim_dict(children: Dict[str, PfLine]) -> Dict[str, PfLine]:
    """Do data checks on dictionary of children so that they fit well together."""
    # Data check.
    try:
        _assert_pfline_kindcompatibility(children)
    except AssertionError as e:
        raise ValueError("The data is not suitable for creating a MultiPfLine.") from e
    # Trim.
    children = _intersect_indices(children)
    return children


def _assert_pfline_kindcompatibility(children: Dict) -> None:
    """Check pflines in dictionary, and raise error if their kind is not compatible."""

    if len(children) == 0:
        raise AssertionError("Must provide at least 1 child.")

    if len(children) == 1:
        return  # No possible compatibility errors if only 1 child.

    # Check kind.

    kindcounter = Counter(child.kind for child in children.values())

    if len(kindcounter) == 1:
        return  # No compatibility error if all children of same kind.

    if (
        kindcounter[Kind.VOLUME_ONLY] == kindcounter[Kind.PRICE_ONLY] == 1
        and kindcounter[Kind.ALL] == 0
    ):
        return  # Children of distinct can only be combined in this exact setting.

    raise AssertionError(
        "All children must be of the same kind, or there must be exactly one volume-only "
        "child and one price-only child."
    )


def _intersect_indices(children: Dict[str, PfLine]) -> Dict[str, PfLine]:
    """Keep only the overlapping part of each PfLine's index."""

    if len(children) < 2:
        return children  # No index errors if only 1 child.

    indices = [child.index for child in children.values()]

    # Check frequency.

    freqs = set([i.freq for i in indices])
    if len(freqs) != 1:
        raise ValueError(
            f"PfLines have unequal frequencies; found {', '.join(str(f) for f in freqs)}."
            " Resample first to obtain equal frequencies."
        )

    # Check/fix indices.
    # Workaround for error in pandas intersection (#46702):
    idx = tools.intersect.indices(*indices)
    if len(idx) == 0:
        raise ValueError("PfLine indices describe non-overlapping periods.")

    children = {name: child.loc[idx] for name, child in children.items()}
    return children
