"""
Code to quickly get objects for testing.
"""

import datetime as dt
from typing import Dict, Union

import numpy as np
import pandas as pd

from .. import tools
from ..core.pfline import Kind, MultiPfLine, PfLine, SinglePfLine
from ..core.pfstate import PfState
from . import mockup

OK_COL_COMBOS = ["w", "q", "p", "pr", "qr", "qp", "wp", "wr"]


def get_index(
    freq: str = "D",
    tz: str = "Europe/Berlin",
    startdate: str = None,
    periods: int = None,
    start_of_day: dt.time = None,
    *,
    _random: bool = True,
) -> pd.DatetimeIndex:
    """Get index."""
    if not periods:
        standard = {"AS": 4, "QS": 5, "MS": 14, "D": 400, "H": 10_000, "15T": 50_000}
        periods = standard.get(freq, 10)
        if _random:
            periods = np.random.randint(periods // 2, periods * 2)
        if tools.freq.up_or_down(freq, "H") <= 0 and tz is None:
            # Shorten index to not include timestamp that do not exist in Europe/Berlin.
            periods = min(periods, 4000)
    if not startdate:
        a, m, d = 2020, 1, 1
        a += np.random.randint(-4, 4) if _random else (periods % 20 - 10)
        if tools.freq.up_or_down(freq, "MS") <= 0:
            m += np.random.randint(0, 12) if _random else (periods % 12)
        if tools.freq.up_or_down(freq, "D") <= 0:
            d += np.random.randint(0, 28) if _random else (periods % 28)
        if tools.freq.up_or_down(freq, "H") <= 0 and tz is None:
            # Start index after DST-start to not include timestamps that do not exist in Europe/Berlin.
            m, d = 4, 2
        startdate = f"{a}-{m}-{d}"
    if not start_of_day:
        start_of_day = dt.time(hour=0, minute=0)
    starttime = f"{start_of_day.hour:02}:{start_of_day.minute:02}:00"
    start = f"{startdate} {starttime}"
    return pd.date_range(start, freq=freq, periods=periods, tz=tz)


def get_value(
    name: str = "w", has_unit: bool = True, *, _random: bool = True
) -> Union[float, tools.unit.Q_]:
    """Get a single value."""
    if not _random:
        np.random.seed(0)
    magn = np.random.random() * 200
    if not has_unit:
        return magn
    else:
        unit = tools.unit.from_name(name)
        return tools.unit.Q_(magn, unit)


def get_series(
    i: pd.DatetimeIndex = None,
    name: str = "w",
    has_unit: bool = True,
    *,
    _random: bool = True,
) -> pd.Series:
    """Get Series with index `i` and name `name`. Values from mock-up functions (if in
    'wqpr') or random between 100 and 200."""
    if i is None:
        i = get_index(_random=_random)
    i.name = "ts_left"

    if not _random:
        np.random.seed(0)

    if name == "w":
        # random average, and 3 random amplitudes with sum < 1
        avg = (10 + 30 * np.random.random()) * np.random.choice([1, 10, 100])
        ampls = np.random.rand(3) * np.array([0.3, 0.2, 0.1])
        return mockup.w_offtake(i, avg, *ampls, has_unit=has_unit)
    elif name == "q":
        q = get_series(i, "w", True) * i.duration
        q = q.rename("q")
        return q if has_unit else q.pint.m
    elif name == "p":
        # random average, and 3 random amplitudes with sum < 1
        avg = 50 + 100 * np.random.random()
        ampls = np.random.rand(3) * np.array([0.25, 0.04, 0.3])
        return mockup.p_marketprices(i, avg, *ampls, has_unit=has_unit)
    elif name == "r":
        r = get_series(i, "q", has_unit) * get_series(i, "p", has_unit)
        return r.rename("r")
    else:
        return pd.Series(100 + 100 * np.random.rand(len(i)), i, name=name)


def get_dataframe(
    i: pd.DatetimeIndex = None,
    columns="wp",
    has_unit: bool = True,
    *,
    _random: bool = True,
) -> pd.DataFrame:
    """Get DataFrame with index `i` and columns `columns`. Columns (e.g. `q` and `w`)
    are not made consistent."""
    if i is None:
        i = get_index(_random=_random)
    i.name = "ts_left"
    series = {col: get_series(i, col, has_unit, _random=_random) for col in columns}
    return pd.DataFrame(series)


# Portfolio line.


def get_singlepfline(
    i: pd.DatetimeIndex = None, kind: Kind = Kind.ALL, *, _random: bool = True
) -> SinglePfLine:
    """Get single portfolio line, i.e. without children."""
    if not isinstance(kind, Kind):
        kind = Kind(kind)
    columns = {Kind.VOLUME_ONLY: "q", Kind.PRICE_ONLY: "p", Kind.ALL: "qr"}[kind]
    return SinglePfLine(get_dataframe(i, columns))


def get_multipfline(
    i: pd.DatetimeIndex = None, kind: Kind = Kind.ALL, *, _random: bool = True
) -> MultiPfLine:
    """Get multi portfolio line. With 2 (singlepfline) children of the same ``kind``."""
    if not isinstance(kind, Kind):
        kind = Kind(kind)
    if i is None:
        i = get_index(_random=_random)
    return MultiPfLine(
        {
            "A": get_singlepfline(i, kind, _random=_random),
            "B": get_singlepfline(i, kind, _random=_random),
        }
    )


def get_pfline(
    i: pd.DatetimeIndex = None,
    kind: Kind = Kind.ALL,
    max_nlevels: int = 3,
    childcount: int = 2,
    prefix: str = "",
    *,
    _random: bool = True,
) -> PfLine:
    """Get portfolio line, without children or with children in random number of levels.
    (including the current level; max_nlevels must be >= 1.)"""
    if not isinstance(kind, Kind):
        kind = Kind(kind)
    # Gather information.
    if i is None:
        i = get_index(_random=_random)
    if not _random:
        np.random.seed(0)
    nlevels = np.random.randint(0, max_nlevels)
    # Create single PfLine.
    if nlevels == 0:
        return get_singlepfline(i, kind, _random=_random)
    # Gather information.
    if childcount == 2 and kind is Kind.ALL and np.random.rand() < 0.33:
        kinds = [Kind.PRICE_ONLY, Kind.VOLUME_ONLY]
    else:
        kinds = [kind] * childcount
    # Create multi PfLine.
    children = {}
    for c, knd in enumerate(kinds):
        name = f"part {prefix}{c}."
        children[name] = get_pfline(
            i, knd, max_nlevels - 1, prefix=f"{prefix}{c}.", _random=_random
        )
    return MultiPfLine(children)


# Portfolio state.


def get_pfstate(
    i: pd.DatetimeIndex = None, avg=None, *, _random: bool = True
) -> PfState:
    """Get portfolio state."""
    if i is None:
        i = get_index(_random=_random)
    if not _random:
        np.random.seed(0)
    if avg is None:
        avg = 200 ** np.random.rand()  # between 1 and 200
    wo = -1 * mockup.w_offtake(i, avg)
    pu = mockup.p_marketprices(i)
    ws, ps = mockup.wp_sourced(wo)
    return PfState.from_series(wo=wo, pu=pu, ws=ws, ps=ps)


def get_pfstates(
    i: pd.DatetimeIndex = None, num=3, *, _random: bool = True
) -> Dict[str, PfState]:
    """Get dictionary of portfolio states."""
    if i is None:
        i = get_index(_random=_random)
    names = ["Pf 1", "Portfolio 2 (long name)", "Portfolio number three"]
    for n in range(3, num):
        names.append(f"Portfolio {n+1}")
    return {name: get_pfstate(i, _random=_random) for name in names[:num]}
