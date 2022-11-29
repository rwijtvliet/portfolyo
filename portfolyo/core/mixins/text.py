"""String representation of PfLine and PfState objects."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable

import colorama
import pandas as pd

from ... import tools
from .. import pfline

if TYPE_CHECKING:
    from ..pfline import PfLine
    from ..pfstate import PfState

COLORS = ["WHITE", "YELLOW", "CYAN", "GREEN", "RED", "BLUE", "MAGENTA", "BLACK"]
TREECOLORS = [colorama.Style.BRIGHT + getattr(colorama.Fore, f) for f in COLORS]
_UNITS = {"w": "MW", "q": "MWh", "p": "Eur/MWh", "r": "Eur"}
VALUEFORMAT = {"w": "{:,.1f}", "q": "{:,.0f}", "p": "{:,.2f}", "r": "{:,.0f}"}
DATETIMEFORMAT = "%Y-%m-%d %H:%M:%S %z"
COLWIDTHS = {"ts": 25, "w": 12, "q": 11, "p": 11, "r": 13}
MAX_DEPTH = 6


def _remove_color(text: str) -> str:
    """Remove all color from text."""
    for color in [colorama.Style.RESET_ALL, *TREECOLORS]:
        text = text.replace(color, "")
    return text


def _df_with_strvalues(df: pd.DataFrame, units: Dict = _UNITS):
    """Turn dataframe with single column names ('w', 'p', etc) into text strings."""
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Dataframe must have single column index; has MultiIndex.")
    str_series = {}
    for name, s in df.items():
        sin = s.pint.to(units.get(name)).pint.magnitude
        formt = VALUEFORMAT.get(name).format
        sout = sin.apply(formt).str.replace(",", " ", regex=False)
        str_series[name] = sout.mask(s.isna(), "")
    return pd.DataFrame(str_series)


def _df_with_strindex(df: pd.DataFrame, num_of_ts: int):
    """Turn datetime index of dataframe into text, and reduce number of rows."""
    df.index = df.index.map(
        lambda ts: ts.strftime(DATETIMEFORMAT).ljust(COLWIDTHS["ts"])
    )
    if len(df.index) > num_of_ts:
        i1, i2 = num_of_ts // 2, (num_of_ts - 1) // 2
        inter = pd.DataFrame([[".."] * len(df.columns)], [".."], df.columns)
        df = pd.concat([df.iloc[:i1], inter, df.iloc[-i2:]])
    return df


def _what(pfl: PfLine) -> str:
    return {
        pfline.Kind.PRICE_ONLY: "price",
        pfline.Kind.VOLUME_ONLY: "volume",
        pfline.Kind.ALL: "price and volume",
    }[pfl.kind]


def _index_info(i: pd.DatetimeIndex) -> Iterable[str]:
    """Info about the index."""
    return [
        f". Start: {i[0]                    } (incl)    . Timezone    : {i.tz or 'none'}  ",
        f". End  : {tools.right.index(i)[-1]} (excl)    . Start-of-day: {i[0].time()}  ",
        f". Freq : {i.freq} ({len(i)} datapoints)",
    ]


def _children_info(pfl: PfLine) -> Iterable[str]:
    """Info about the children of the portfolio line."""
    childtxt = [f"'{name}' ({_what(child)})" for name, child in pfl.items()]
    return [". Children: " + ("none" if not childtxt else ", ".join(childtxt))]


def _treedict(depth: int, is_last_child: bool, has_children: bool) -> Dict[str, str]:
    """Dictionary with 4 strings ('00', '01', '10', '11') that are used in drawing the tree."""
    colors = {"0": TREECOLORS[depth], "1": TREECOLORS[depth + 1]}
    tree = {}
    # 00 = first chars on header text line, #10 = first chars on other text lines
    if depth == 0:
        tree["00"] = colors["0"] + "─"
    else:
        tree["00"] = colors["0"] + ("└" if is_last_child else "├")
    tree["10"] = " " if is_last_child else (colors["0"] + "│")
    # 01 = following chars on header line, #11 = following chars on other text lines
    tree["01"] = (colors["1"] + "●" + colors["0"]) if has_children else "─"
    tree["01"] += "─" * (MAX_DEPTH - depth) + " "
    tree["11"] = ((colors["1"] + "│") if has_children else " ") + colors["0"]
    tree["11"] += " " * (MAX_DEPTH - depth + 3)
    return tree


def _dataheader(cols: Iterable[str] = "wqpr", units: Dict = _UNITS) -> Iterable[str]:
    out = [" " * 25] * 2  # width of timestamps
    for c in cols:
        width = COLWIDTHS[c] + 1
        out[0] += f"{c:>{width}}"
        out[1] += f"{units[c]:>{width}}"
    return out


# Main 3 functions that recursively call each other.


def _nestedtree(
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
    tree = _treedict(depth, is_last, isinstance(pfl, pfline.MultiPfLine))
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


def _flatdatablock(pfl: PfLine, cols: Iterable[str], num_of_ts: int) -> Iterable[str]:
    """The timestamps and data to be shown in a block, next to the tree."""
    # Obtain dataframe with index = timestamp as string and columns = one or more of 'qwpr'.
    df = pfl.df(cols, flatten=True)
    # . reduce number of timestamps to increase speed of conversion to strings.
    if len(df.index) > num_of_ts * 2:
        df = pd.concat([df.iloc[:num_of_ts, :], df.iloc[-num_of_ts:, :]], axis=0)
    # . turn values into strings.
    df = _df_with_strvalues(df)
    # . turn index into strings and reduce to wanted number of datapoints
    df = _df_with_strindex(df, num_of_ts)
    # . column withs
    col_space = {k: v for k, v in COLWIDTHS.items() if k in df}
    # Turn into list of strings.
    df_str = df.to_string(col_space=col_space, index_names=False, header=False)
    return df_str.split("\n")


def _childrenlines(
    pfl: PfLine, cols: Iterable[str], num_of_ts: int, depth: int
) -> Iterable[str]:
    """Treeview of only the children."""
    out = []
    if isinstance(pfl, pfline.SinglePfLine):
        return out
    for c, (name, child) in enumerate(pfl.items()):
        is_last, is_only = (c == len(pfl) - 1), (len(pfl) == 1)
        out.extend(
            _nestedtree(name, child, cols, num_of_ts, depth + 1, is_last, is_only)
        )
    return out


# Highest-level functions.


def pfl_as_string(pfl: PfLine, flatten: bool, num_of_ts: int, color: bool) -> str:
    lines = [f"PfLine object with {_what(pfl)} information."]
    lines.extend(_index_info(pfl.index))
    if isinstance(pfl, pfline.MultiPfLine):
        lines.extend(_children_info(pfl))
    cols = pfl.available
    if flatten:
        lines.extend(_dataheader(cols))
        lines.extend([""])
        lines.extend(_flatdatablock(pfl, cols, num_of_ts))
    else:
        spaces = " " * (MAX_DEPTH + 5)
        lines.extend([spaces + txtline for txtline in _dataheader(cols)])
        lines.extend(_nestedtree("(this pfline)", pfl, cols, num_of_ts))
    txt = "\n".join(lines)
    return txt if color else _remove_color(txt)


def pfs_as_string(pfs: PfState, num_of_ts: int, color: bool) -> str:
    lines = ["PfState object."]
    lines.extend(_index_info(pfs.index))
    spaces = " " * (MAX_DEPTH + 5)
    lines.extend([spaces + txtline for txtline in _dataheader("wqpr")])
    lines.extend(_nestedtree("offtake", pfs.offtakevolume, "wqpr", num_of_ts))
    lines.extend(_nestedtree("pnl_cost", pfs.pnl_cost, "wqpr", num_of_ts))
    txt = "\n".join(lines)
    return txt if color else _remove_color(txt)


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


class PfStateText:
    def __repr__(self):
        return pfs_as_string(self, 5, False)

    def print(self: PfState, num_of_ts: int = 5, color: bool = True) -> None:
        """Treeview of the portfolio state.

        Parameters
        ----------
        num_of_ts : int, optional (default: 5)
            How many timestamps to show for each PfLine.
        color : bool, optional (default: True)
            Make tree structure clearer by including colors. May not work on all output
            devices.

        Returns
        -------
        None
        """
        print(pfs_as_string(self, num_of_ts, color))
