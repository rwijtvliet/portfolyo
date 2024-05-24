"""String representation of PfLine and PfState objects."""

from typing import Dict, Iterable

import colorama
import pandas as pd

from ... import tools

COLORS = ["WHITE", "YELLOW", "CYAN", "GREEN", "RED", "BLUE", "MAGENTA", "BLACK"]
TREECOLORS = [colorama.Style.BRIGHT + getattr(colorama.Fore, f) for f in COLORS]
_UNITS = {"w": "MW", "q": "MWh", "p": "Eur/MWh", "r": "Eur"}
VALUEFORMAT = {"w": "{:,.1f}", "q": "{:,.0f}", "p": "{:,.2f}", "r": "{:,.0f}"}
DATETIMEFORMAT = "%Y-%m-%d %H:%M:%S %z"
COLWIDTHS = {"ts": 25, "w": 12, "q": 11, "p": 11, "r": 13}
MAX_DEPTH = 6


def remove_color(text: str) -> str:
    """Remove all color from text."""
    for color in [colorama.Style.RESET_ALL, *TREECOLORS]:
        text = text.replace(color, "")
    return text


def df_with_strvalues(df: pd.DataFrame, units: Dict = _UNITS):
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


def df_with_strindex(df: pd.DataFrame, num_of_ts: int):
    """Turn datetime index of dataframe into text, and reduce number of rows."""
    df.index = df.index.map(
        lambda ts: ts.strftime(DATETIMEFORMAT).ljust(COLWIDTHS["ts"])
    )
    if len(df.index) > num_of_ts:
        i1, i2 = num_of_ts // 2, (num_of_ts - 1) // 2
        inter = pd.DataFrame([[".."] * len(df.columns)], [".."], df.columns)
        df = pd.concat([df.iloc[:i1], inter, df.iloc[-i2:]])
    return df


def index_info(i: pd.DatetimeIndex) -> Iterable[str]:
    """Info about the index."""
    return [
        f". Start: {i[0]                    } (incl)    . Timezone    : {i.tz or 'none'}  ",
        f". End  : {tools.right.index(i)[-1]} (excl)    . Start-of-day: {i[0].time()}  ",
        f". Freq : {i.freq} ({len(i)} datapoints)",
    ]


def treedict(depth: int, is_last_child: bool, has_children: bool) -> Dict[str, str]:
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


def dataheader(cols: Iterable[str] = "wqpr", units: Dict = _UNITS) -> Iterable[str]:
    out = [" " * 25] * 2  # width of timestamps
    for c in cols:
        width = COLWIDTHS[c] + 1
        out[0] += f"{c:>{width}}"
        out[1] += f"{units[c]:>{width}}"
    return out
