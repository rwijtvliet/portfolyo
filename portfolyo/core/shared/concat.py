# import pandas as pd
# import portfolyo as pf
from __future__ import annotations
from typing import Union
import pandas as pd
from portfolyo import tools

from ..pfstate import PfState
from collections import defaultdict
from ..pfline.enums import Structure

from ..pfline import PfLine, create
from .. import pfstate


def general(*pfl_or_pfs: Union[PfLine, PfState]) -> None:
    """
    Based on passed parameters calls either concat_pflines() or concat_pfstates().

    Parameters
    ----------
    pfl_or_pfs: Union[PfLine, PfState]
        The input values. Can be either a list of Pflines or PfStates to concatenate.

    Returns
    -------
    None

    """
    if all(isinstance(item, PfLine) for item in pfl_or_pfs):
        return concat_pflines(*pfl_or_pfs)
    elif all(isinstance(item, PfState) for item in pfl_or_pfs):
        return concat_pfstates(*pfl_or_pfs)
    else:
        raise NotImplementedError(
            "Concatenation is implemented only for PfState or PfLine."
        )


def concat_pflines(*pfls: PfLine) -> PfLine:
    """
    This only works if the input portfolio lines have contain compatible information:
    (the same frequency, timezone, start-of-day, kind, etc) and
    their indices are gapless and without overlap.

    For nested ones, check if they have the same children. If not, error. If yes, do the concatenation on each child (pair them up)

    Parameters
    ----------
    pfls: List[PfLine]
        The input values.

    Returns
    -------
    PfLine
        Concatenated version of List[PfLine].

    """
    if len(pfls) < 2:
        print("Concatenate needs at least two elements.")
        return
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
        sorted_children = defaultdict(list)
        for pfl in pfls:
            for name, child in pfl.items():
                sorted_children[name].append(child)
        for name, children in sorted_children.items():
            if len(children) != len(pfls):
                raise ValueError

        children_names_sets = [{name for name in pfl.children} for pfl in pfls]
        if len(children_names_sets) != len(pfls[0].children):
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
        return create.flatpfline(concat_data)
    child_data = {}
    for name, children in sorted_children.items():
        # for every name in children need to concatenate elements
        child_data[name] = concat_pflines(*children)

    # create pfline from dataframes: ->
    # call the constructor of pfl to check check gapplesnes and overplap
    output = create.nestedpfline(child_data)
    return output


def concat_pfstates(*pfss: PfState) -> PfState:
    """
    Parameters
    ----------
    pfls: List[PfLine]
        The input values.

    Returns
    -------
    PfLine
        Concatenated version of List[PfLine].

    """
    if len(pfss) < 2:
        print("Concatenate needs at least two elements.")
        return
    offtakevolume = concat_pflines(*[pfs.offtakevolume for pfs in pfss])
    sourced = concat_pflines(*[pfs.sourced for pfs in pfss])
    unsourcedprice = concat_pflines(*[pfs.unsourcedprice for pfs in pfss])
    return pfstate.PfState(offtakevolume, unsourcedprice, sourced)
