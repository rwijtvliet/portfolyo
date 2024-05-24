# import pandas as pd
# import portfolyo as pf
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .. import tools
from ..core import pfstate
from ..core.pfline import PfLine, create
from ..core.pfline.enums import Structure
from ..core.pfstate import PfState


def general(pfl_or_pfs: Iterable[PfLine | PfState]) -> None:
    """
    Based on passed parameters calls either concat_pflines() or concat_pfstates().

    Parameters
    ----------
    pfl_or_pfs: Iterable[PfLine | PfState]
        The input values. Can be either a list of Pflines or PfStates to concatenate.

    Returns
    -------
    None

    Notes
    -----
    Input portfolio lines must contain compatible information, i.e., same frequency,
    timezone, start-of-day, and kind. Their indices must be gapless and without overlap.

    For nested pflines, the number and names of their children must match; concatenation
    is done on a name-by-name basis.

    Concatenation returns the same result regardless of input order.

    """
    if all(isinstance(item, PfLine) for item in pfl_or_pfs):
        return concat_pflines(pfl_or_pfs)
    elif all(isinstance(item, PfState) for item in pfl_or_pfs):
        return concat_pfstates(pfl_or_pfs)
    else:
        raise NotImplementedError(
            "Concatenation is implemented only for PfState or PfLine."
        )


def concat_pflines(pfls: Iterable[PfLine]) -> PfLine:
    """
    Concatenate porfolyo lines along their index.

    Parameters
    ----------
    pfls: Iterable[PfLine]
        The input values.

    Returns
    -------
    PfLine
        Concatenated version of PfLines.

    Notes
    -----
    Input portfolio lines must contain compatible information, i.e., same frequency,
    timezone, start-of-day, and kind. Their indices must be gapless and without overlap.

    For nested pflines, the number and names of their children must match; concatenation
    is done on a name-by-name basis.

    Concatenation returns the same result regardless of input order.
    """
    if len(pfls) < 2:
        raise NotImplementedError(
            "Cannot perform operation with less than 2 portfolio lines."
        )
    if len({pfl.kind for pfl in pfls}) != 1:
        raise TypeError("Not possible to concatenate PfLines of different kinds.")
    if len({pfl.index.freq for pfl in pfls}) != 1:
        raise TypeError("Not possible to concatenate PfLines of different frequencies.")
    if len({pfl.index.tz for pfl in pfls}) != 1:
        raise TypeError("Not possible to concatenate PfLines of different time zones.")
    if len({tools.startofday.get(pfl.index, "str") for pfl in pfls}) != 1:
        raise TypeError(
            "Not possible to concatenate PfLines of different start_of_day."
        )
    # we can concatenate only pflines of the same type: nested of flat
    # with this test and check whether pfls are the same types and they have the same number of children
    if len({pfl.structure for pfl in pfls}) != 1:
        raise TypeError("Not possible to concatenate PfLines of different structures.")
    if pfls[0].structure is Structure.NESTED:
        child_names = pfls[0].children.keys()
        for pfl in pfls:
            diffs = set(child_names) ^ set(pfl.children.keys())
            if len(diffs) != 0:
                raise TypeError(
                    "Not possible to concatenate PfLines with different children names."
                )
    # If we reach here, all pfls have same kind, same number and names of children.

    # concat(a,b) and concat(b,a) should give the same result:
    sorted_pfls = sorted(pfls, key=lambda pfl: pfl.index[0])
    if pfls[0].structure is Structure.FLAT:
        # create flat dataframe of parent
        dataframes_flat = [pfl.df for pfl in sorted_pfls]
        # concatenate dataframes into one
        concat_data = pd.concat(dataframes_flat, axis=0)
        try:
            # Call create.flatpfline() and catch any ValueError
            return create.flatpfline(concat_data)
        except ValueError as e:
            # Handle the error
            raise ValueError(
                "Error by creating PfLine. PfLine is either not gapless or has overlaps"
            ) from e
    child_data = {}
    child_names = pfls[0].children.keys()
    for cname in child_names:
        # for every name in children need to concatenate elements
        child_values = [pfl.children[cname] for pfl in sorted_pfls]
        child_data[cname] = concat_pflines(child_values)

    # create pfline from dataframes: ->
    # call the constructor of pfl to check check gaplesnes and overplap
    return create.nestedpfline(child_data)


def concat_pfstates(pfss: Iterable[PfState]) -> PfState:
    """
    Concatenate porfolyo states along their index.

    Parameters
    ----------
    pfss: Iterable[PfState]
         The input values.

    Returns
    -------
     PfState
         Concatenated version of PfStates.

    """
    if len(pfss) < 2:
        print("Concatenate needs at least two elements.")
        return
    offtakevolume = concat_pflines([pfs.offtakevolume for pfs in pfss])
    sourced = concat_pflines([pfs.sourced for pfs in pfss])
    unsourcedprice = concat_pflines([pfs.unsourcedprice for pfs in pfss])
    return pfstate.PfState(offtakevolume, unsourcedprice, sourced)
