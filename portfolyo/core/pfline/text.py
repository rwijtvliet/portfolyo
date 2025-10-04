from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import colorama
import pandas as pd

from ..shared import text as shared_text
from .enums import Kind, Structure

if TYPE_CHECKING:
    from .pflinee import PfLine


INFO = {
    Kind.VOLUME: ("volume",),
    Kind.PRICE: ("price",),
    Kind.REVENUE: ("revenue", "(i.e., monetary value)"),
    Kind.COMPLETE: ("complete", "(i.e., volume, price, and monetary value)"),
}


def _what(pfl: PfLine, *, long: bool = False) -> str:
    parts = INFO[pfl.kind]
    return " ".join(parts) if long else parts[0]


def _children_info(pfl: PfLine) -> Iterable[str]:
    """Info about the children of the portfolio line."""
    childtxt = [f"'{name}' ({_what(child)})" for name, child in pfl.children.items()]
    return [". Children: " + ("none" if not childtxt else ", ".join(childtxt))]


def _flatdatablock(pfl: PfLine, num_of_ts: int) -> Iterable[str]:
    """The timestamps and data to be shown in a block, next to the tree."""
    # Obtain dataframe with index = timestamp as string and columns = one or more of 'wqpr'.
    df = pd.DataFrame(pfl)
    # . (roughly) reduce number of timestamps to increase speed of conversion to strings.
    if len(df.index) > num_of_ts * 2:
        df = pd.concat([df.iloc[:num_of_ts, :], df.iloc[-num_of_ts:, :]], axis=0)
    # . turn values into strings.
    df = shared_text.df_with_strvalues(df, pfl.commodity.col_to_units)
    # . turn index into strings and reduce to wanted number of datapoints
    df = shared_text.df_with_strindex(df, num_of_ts)
    # . column withs
    col_space = {k: v for k, v in shared_text.COLWIDTHS.items() if k in df}
    # Turn into list of strings.
    df_str = df.to_string(col_space=col_space, index_names=False, header=False)
    return df_str.split("\n")


def _childrenlines(pfl: PfLine, num_of_ts: int, depth: int) -> Iterable[str]:
    """Treeview of only the children."""
    out = []
    if pfl.structure is Structure.FLAT:
        return out
    for childnum, (name, child) in enumerate(pfl.kids.items()):
        is_last = childnum == len(pfl.children) - 1
        is_only = len(pfl.children) == 1
        out.extend(nestedtree(name, child, num_of_ts, depth + 1, is_last, is_only))
    return out


# Highest-level functions.


def pflheader(pfl: PfLine) -> list[str]:
    firstline = [f"PfLine with {_what(pfl, long=True)} information."]
    return firstline + shared_text.objectheader(pfl.index, pfl.commodity)


def nestedtree(
    name: str,
    pfl: PfLine,
    num_of_ts: int,
    depth: int = 0,
    is_last: bool = True,
    is_only: bool = False,
) -> Iterable[str]:
    """Treeview of the portfolio line."""
    out = []
    tree = shared_text.treedict(depth, is_last, pfl.structure is Structure.NESTED)
    # Name.
    out.append(tree["00"] + tree["01"] + name)
    # Top-level body block.
    if is_only and depth > 0:
        txtlines = ["(single contributor to parent data; has same values)"]
    else:
        txtlines = _flatdatablock(pfl, num_of_ts)
    for txtline in txtlines:
        out.append(tree["10"] + tree["11"] + colorama.Style.RESET_ALL + txtline)
    # Add children if any.
    for txtline in _childrenlines(pfl, num_of_ts, depth):
        out.append(tree["10"] + txtline)
    return out


def pfl_as_string(pfl: PfLine, num_of_ts: int, color: bool) -> str:
    lines = pflheader(pfl)
    if pfl.structure is Structure.NESTED:
        lines.extend(_children_info(pfl))
    # if flatten:
    #     lines.extend(shared_text.dataheader(pfl.commodity.col_to_units))
    #     lines.extend([""])
    #     lines.extend(_flatdatablock(pfl, num_of_ts))
    # else:
    spaces = " " * (shared_text.MAX_DEPTH + 5)
    columns_and_units = {
        col: unit for col, unit in pfl.commodity.col_to_units.items() if col in pfl.kind.available
    }
    lines.extend([spaces + txtline for txtline in shared_text.dataheader(columns_and_units)])
    lines.extend(nestedtree("(this pfline)", pfl, num_of_ts))
    txt = "\n".join(lines)
    return txt if color else shared_text.remove_color(txt)


class TextMethods:
    # def __repr__(self):
    #     lines = _header(self)
    #     return pfl_as_string(self, True, 20, False)

    def print(self, num_of_ts: int = 5, color: bool = True) -> None:
        """Treeview of the portfolio line.

        Parameters
        ----------
        num_of_ts, optional (default: 5)
            How many timestamps to show for each PfLine.
        color, optional (default: True)
            Make tree structure clearer by including colors. May not work on all output devices.

        Returns
        -------
        None
        """
        print(pfl_as_string(self, num_of_ts, color))
