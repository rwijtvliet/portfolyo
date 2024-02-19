# import pandas as pd
# import portfolyo as pf
from typing import TYPE_CHECKING, Union, List
from ..pfline import classes
from collections import defaultdict
from ..pfline.enums import Structure

if TYPE_CHECKING:  # needed to avoid circular imports
    from ..pfline import PfLine
    from ..pfstate import PfState
# import itertools
from portfolyo import tools


def general(pfl_or_pfs: Union[PfLine, PfState]) -> None:
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
    if isinstance(pfl_or_pfs, classes.FlatPfLine) or isinstance(
        pfl_or_pfs, classes.NestedPfLine
    ):
        concat_pflines(pfl_or_pfs)
    elif isinstance(pfl_or_pfs, PfState):
        concat_pfstates(pfl_or_pfs)
    else:
        raise NotImplementedError(
            "Concatenation is implemented only for PfState or PfLine."
        )


def concat_pflines(pfls: List[PfLine]) -> PfLine:
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

    # for a, b in itertools.combinations(pfls, 2):
    #     if a.kind == b.kind:
    #         raise TypeError("Not possible to concatenate PfLines of different kinds.")

    # we need to check every element on the list and compare each element to each other element
    # would it be better to check every element at first for freq and such and then divide by flat and nested
    # or run checks on flat/nested first
    # TODO: to ask whether children and parent have the same tz and so on->yes
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
    # with this test and check wether pfls are the same types and they have the same number of children
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
        # If we reach here, all pfls have same kind, same number and names of children.

        children_names_sets = [{name for name in pfl.children} for pfl in pfls]
        if len(children_names_sets) != len(pfls[0].children):
            raise TypeError(
                "Not possible to concatenate PfLines with different children names."
            )

        # TODO: concat(a,b and concat(b,a) should give the same result)
        # TODO: we can probably to create pfline to check gapplesnes and overplap,
        # it's supposed to give error


def concat_pfstates(pfs: List[PfState]) -> PfState:
    pass
