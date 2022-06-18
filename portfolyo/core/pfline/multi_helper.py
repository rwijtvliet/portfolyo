"""Verify input data and turn into object needed in MultiPfLine instantiation."""

from __future__ import annotations

from . import multi
from .base import PfLine, Kind
from ...tools import stamps

from typing import Counter, Mapping, Dict, Union


def make_childrendict(data: Union[Mapping, multi.MultiPfLine]) -> Dict[str, PfLine]:
    """From data, create a dictionary of PfLine instances. Also, do some data verification."""
    children = _data_to_childrendict(data)
    try:
        _assert_pfline_kindcompatibility(children)
    except AssertionError as e:
        raise ValueError("The data is not suitable for creating a MultiPfLine.") from e
    children = _intersect_indices(children)
    return children


def _data_to_childrendict(data: Union[Mapping, multi.MultiPfLine]) -> Dict[str, PfLine]:
    """Check data, and turn into a dictionary."""

    if isinstance(data, multi.MultiPfLine):
        return data.children
    if not isinstance(data, Mapping):
        raise TypeError(
            "`data` must be dict or other Mapping (or a MultiPfLine instance)."
        )

    children = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError("Keys must be strings.")
        if isinstance(value, PfLine):
            children[key] = value
        else:
            children[key] = PfLine(value)  # try to cast to PfLine instance

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

    if len(set([i.freq for i in indices])) != 1:
        raise ValueError("PfLines have unequal frequency; resample first.")

    # Check/fix indices.
    idx = stamps.intersection(*indices)  # workaround for pandas intersection (#46702)
    if len(idx) == 0:
        raise ValueError("PfLine indices describe non-overlapping periods.")

    children = {name: child.loc[idx] for name, child in children.items()}
    return children
