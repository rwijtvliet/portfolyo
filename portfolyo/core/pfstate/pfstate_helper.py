"""Prepare/verify input data for PfState initialisation."""


import warnings
from typing import Any, Iterable

from ... import tools
from ..pfline import FlatPfLine, Kind, NestedPfLine, PfLine  # noqa


def make_pflines(
    offtakevolume: Any, unsourcedprice: Any, sourced: Any = None
) -> Iterable[PfLine]:
    """Take offtake, unsourced, sourced information. Do some data massaging and return
    3 PfLines: for offtake volume, unsourced price, and sourced price and volume.
    """
    # Ensure unsourced and offtake are specified.
    if offtakevolume is None or unsourcedprice is None:
        raise ValueError("Must specify offtake volume and unsourced prices.")

    # Get everything as PfLine.
    # . Offtake volume.
    offtakevolume = PfLine(offtakevolume)  # force to be PfLine.
    if offtakevolume.kind is Kind.PRICE:
        raise ValueError("Parameter ``offtakevolume`` does not contain volume.")
    elif offtakevolume.kind is Kind.COMPLETE:
        warnings.warn(
            "Parameter ``offtakevolume``: also contains price infomation; this is discarded."
        )
        offtakevolume = offtakevolume.volume
    # . Unsourced prices.
    unsourcedprice = PfLine(unsourcedprice)  # force to be PfLine.
    if unsourcedprice.kind is Kind.VOLUME:
        raise ValueError("Parameter ``unsourcedprice`` does not contain prices.")
    elif unsourcedprice.kind is Kind.COMPLETE:
        warnings.warn(
            "Parameter ``unsourcedprice``: also contains volume infomation; this is discarded."
        )
        unsourcedprice = unsourcedprice.price
    # . Sourced volume and prices.
    if sourced is not None:
        sourced = PfLine(sourced)
        if sourced.kind is not Kind.COMPLETE:
            raise ValueError("Parameter ``sourced`` does not contain price and volume.")

    # Check/fix indices.
    # . Frequencies.
    freqs = [
        o.index.freq for o in (offtakevolume, unsourcedprice, sourced) if o is not None
    ]
    if len(set(freqs)) != 1:
        raise ValueError("PfLines have unequal frequency; resample first.")
    # . Keep only overlapping part of indices.
    idx = offtakevolume.index
    if len(tools.intersect.indices(idx, unsourcedprice.index)) < len(idx):
        raise ValueError(
            "Parameter ``unsourcedprice``: does not cover entire delivery period of ``offtakevolume``."
        )
    unsourcedprice = unsourcedprice.loc[idx]
    if sourced is not None:
        # Workaround for error in pandas intersection (#46702):
        if len(tools.intersect.indices(idx, sourced.index)) < len(idx):
            raise ValueError(
                "Parameter ``sourced``: does not cover entire delivery period of ``offtakevolume``."
            )
        sourced = sourced.loc[idx]

    return offtakevolume, unsourcedprice, sourced
