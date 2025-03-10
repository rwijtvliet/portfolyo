"""Prepare/verify input data for PfState initialisation."""

import warnings
from typing import Any, Iterable

import pandas as pd

from ... import tools
from ..pfline import Kind, PfLine, create


def make_pflines(
    offtakevolume: Any, unsourcedprice: Any, sourced: Any = None
) -> Iterable[PfLine]:
    """Take offtake, unsourced, sourced information. Do some data massaging and return
    3 PfLines: for offtake volume, unsourced price, and sourced price and volume.
    """
    # Ensure unsourced and offtake are specified.
    if offtakevolume is None or unsourcedprice is None:
        raise ValueError("Must specify offtake volume and unsourced prices.")

    # Offtake volume.
    offtakevolume = prepare_offtakevolume(offtakevolume)
    idx = offtakevolume.index

    # Unsourced prices.
    unsourcedprice = prepare_unsourcedprice(unsourcedprice, idx)

    # Sourced volume/prices.
    sourced = prepare_sourced(sourced, idx)

    return offtakevolume, unsourcedprice, sourced


def prepare_offtakevolume(offtakevolume: Any) -> PfLine:
    offtakevolume = create.pfline(offtakevolume)  # force to be PfLine.
    if "q" not in offtakevolume.kind.available:
        raise ValueError("Parameter ``offtakevolume`` does not contain volume.")
    elif offtakevolume.kind is Kind.COMPLETE:
        warnings.warn(
            "Parameter ``offtakevolume``: also contains price infomation; this is discarded."
        )
        offtakevolume = offtakevolume.volume
    return offtakevolume


def prepare_unsourcedprice(unsourcedprice: Any, ref_idx: pd.DatetimeIndex) -> PfLine:
    unsourcedprice = create.pfline(unsourcedprice)  # force to be PfLine.
    if "p" not in unsourcedprice.kind.available:
        raise ValueError("Parameter ``unsourcedprice`` does not contain prices.")
    elif unsourcedprice.kind is Kind.COMPLETE:
        warnings.warn(
            "Parameter ``unsourcedprice``: also contains volume infomation; this is discarded."
        )
        unsourcedprice = unsourcedprice.price
    try:
        tools.testing.assert_indices_compatible(ref_idx, unsourcedprice.index)
    except AssertionError as e:
        raise ValueError from e
    if len(tools.intersect.indices(ref_idx, unsourcedprice.index)) < len(ref_idx):
        raise ValueError(
            "Parameter ``unsourcedprice`` does not cover entire delivery period of"
            " ``offtakevolume``."
        )
    return unsourcedprice.loc[ref_idx]


def prepare_sourced(sourced: Any, ref_idx: pd.DatetimeIndex) -> PfLine:
    if sourced is None:
        return create.flatpfline(pd.DataFrame({"q": 0, "r": 0}, ref_idx))

    sourced = create.pfline(sourced)
    if sourced.kind is not Kind.COMPLETE:
        raise ValueError("Parameter ``sourced`` does not contain price and volume.")
    try:
        tools.testing.assert_indices_compatible(ref_idx, sourced.index)
    except AssertionError as e:
        raise ValueError from e
    # HACK: Workaround for error in pandas intersection (#46702):
    if len(tools.intersect.indices(ref_idx, sourced.index)) < len(ref_idx):
        raise ValueError(
            "Parameter ``sourced``: does not cover entire delivery period of"
            " ``offtakevolume``."
        )
    return sourced.loc[ref_idx]
