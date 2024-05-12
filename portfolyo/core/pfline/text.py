from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import colorama
import pandas as pd

from ..shared import text as shared_text
from . import classes
from .enums import Kind

if TYPE_CHECKING:
    from .classes import PfLine


def _what(pfl: PfLine) -> str:
    return {
        Kind.VOLUME: "volume",
        Kind.PRICE: "price",
        Kind.REVENUE: "revenue",
        Kind.COMPLETE: "complete",
    }[pfl.kind]


def _children_info(pfl: PfLine) -> Iterable[str]:
    """Info about the children of the portfolio line."""
    childtxt = [f"'{name}' ({_what(child)})" for name, child in pfl.items()]
    return [". Children: " + ("none" if not childtxt else ", ".join(childtxt))]


def _flatdatablock(pfl: PfLine, cols: Iterable[str], num_of_ts: int) -> Iterable[str]:
    """The timestamps and data to be shown in a block, next to the tree."""
    # Obtain dataframe with index = timestamp as string and columns = one or more of 'qwpr'.
    df = pfl.df[list(cols)]
    # . (roughly) reduce number of timestamps to increase speed of conversion to strings.
    if len(df.index) > num_of_ts * 2:
        df = pd.concat([df.iloc[:num_of_ts, :], df.iloc[-num_of_ts:, :]], axis=0)
    # . turn values into strings.
    df = shared_text.df_with_strvalues(df)
    # . turn index into strings and reduce to wanted number of datapoints
    df = shared_text.df_with_strindex(df, num_of_ts)
    # . column withs
    col_space = {k: v for k, v in shared_text.COLWIDTHS.items() if k in df}
    # Turn into list of strings.
    df_str = df.to_string(col_space=col_space, index_names=False, header=False)
    return df_str.split("\n")


def _childrenlines(
    pfl: PfLine, cols: Iterable[str], num_of_ts: int, depth: int
) -> Iterable[str]:
    """Treeview of only the children."""
    out = []
    if isinstance(pfl, classes.FlatPfLine):
        return out
    for c, (name, child) in enumerate(pfl.items()):
        is_last, is_only = (c == len(pfl) - 1), (len(pfl) == 1)
        out.extend(
            nestedtree(name, child, cols, num_of_ts, depth + 1, is_last, is_only)
        )
    return out


# Highest-level functions.


def nestedtree(
    name: str,
    pfl: PfLine,
    cols: Iterable[str],
    num_of_ts: int,
    depth: int = 0,
    is_last: bool = True,
    is_only: bool = False,
) -> Iterable[str]:
    """Treeview of the portfolio line."""
    out = []
    tree = shared_text.treedict(depth, is_last, isinstance(pfl, classes.NestedPfLine))
    # Name.
    out.append(tree["00"] + tree["01"] + name)
    # Top-level body block.
    if is_only and depth > 0:
        txtlines = ["(only contributor to parent data; has same values)"]
    else:
        txtlines = _flatdatablock(pfl, cols, num_of_ts)
    for txtline in txtlines:
        out.append(tree["10"] + tree["11"] + colorama.Style.RESET_ALL + txtline)
    # Add children if any.
    for txtline in _childrenlines(pfl, cols, num_of_ts, depth):
        out.append(tree["10"] + txtline)
    return out


def pfl_as_string(pfl: PfLine, flatten: bool, num_of_ts: int, color: bool) -> str:
    lines = [f"PfLine object with {_what(pfl)} information."]
    lines.extend(shared_text.index_info(pfl.index))
    if isinstance(pfl, classes.NestedPfLine):
        lines.extend(_children_info(pfl))
    cols = pfl.kind.available
    if flatten:
        lines.extend(shared_text.dataheader(cols))
        lines.extend([""])
        lines.extend(_flatdatablock(pfl, cols, num_of_ts))
    else:
        spaces = " " * (shared_text.MAX_DEPTH + 5)
        lines.extend([spaces + txtline for txtline in shared_text.dataheader(cols)])
        lines.extend(nestedtree("(this pfline)", pfl, cols, num_of_ts))
    txt = "\n".join(lines)
    return txt if color else shared_text.remove_color(txt)


class PfLineText:
    def __repr__(self):
        return pfl_as_string(self, True, 20, False)

    def print(
        self: PfLine, flatten: bool = False, num_of_ts: int = 5, color: bool = True
    ) -> None:
        """Treeview of the portfolio line.

        Parameters
        ----------
        flatten : bool, optional (default: False)
            if True, show only the top-level (aggregated) information.
        num_of_ts : int, optional (default: 5)
            How many timestamps to show for each PfLine.
        color : bool, optional (default: True)
            Make tree structure clearer by including colors. May not work on all output
            devices.

        Returns
        -------
        None
        """
        print(pfl_as_string(self, flatten, num_of_ts, color))
