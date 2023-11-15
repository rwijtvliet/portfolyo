"""
Code to quickly get objects for testing.
"""

import datetime as dt
from typing import Dict, Union

import numpy as np
import pandas as pd

from .. import tools
from ..core.pfline import FlatPfLine, Kind, NestedPfLine, PfLine, create
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
    _seed: bool = True,
) -> pd.DatetimeIndex:
    """Get index."""
    if _seed:
        np.random.seed(_seed)
    if not periods:
        standard = {"AS": 4, "QS": 5, "MS": 14, "D": 400, "H": 10_000, "15T": 50_000}
        periods = standard.get(freq, 10)
        if _seed:
            periods = np.random.randint(periods // 2, periods * 2)
        if tools.freq.up_or_down(freq, "H") <= 0 and tz is None:
            # Shorten index to not include timestamp that do not exist in Europe/Berlin.
            periods = min(periods, 4000)
    if not startdate:
        a, m, d = 2020, 1, 1
        a += np.random.randint(-4, 4) if _seed else (periods % 20 - 10)
        if tools.freq.up_or_down(freq, "MS") <= 0:
            m += np.random.randint(0, 12) if _seed else (periods % 12)
        if tools.freq.up_or_down(freq, "D") <= 0:
            d += np.random.randint(0, 28) if _seed else (periods % 28)
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
    name: str = None, has_unit: bool = True, magn: float = None, *, _seed: int = None
) -> Union[float, tools.unit.Q_]:
    """Get a single value."""
    if _seed:
        np.random.seed(_seed)
    if magn is None:
        magn = np.random.random() * 200
    if not has_unit:
        return magn
    else:
        unit = tools.unit.from_name(name)
        return tools.unit.Q_(magn, unit)


def get_series(
    i: pd.DatetimeIndex = None,
    name: str = None,
    has_unit: bool = True,
    *,
    _seed: int = None,
) -> pd.Series:
    """Get Series with index `i` and name `name`. Values from mock-up functions (if in
    'wqpr') or random between 100 and 200."""
    if i is None:
        i = get_index(_seed=_seed)
    i.name = "ts_left"

    if _seed:
        np.random.seed(_seed)

    if name == "w":
        # random average, and 3 random amplitudes with sum < 1
        avg = 30 + 10 * np.random.random()
        ampls = np.random.rand(3) * np.array([0.3, 0.2, 0.1])
        return mockup.w_offtake(i, avg, *ampls, has_unit=has_unit)
    elif name == "q":
        q = get_series(i, "w", True) * i.duration
        q = q.rename("q")
        return q if has_unit else q.pint.m
    elif name == "p":
        # random average, and 3 random amplitudes with sum < 1
        avg = 100 + 20 * np.random.random()
        ampls = np.random.rand(3) * np.array([0.25, 0.04, 0.3])
        return mockup.p_marketprices(i, avg, *ampls, has_unit=has_unit)
    elif name == "r":
        r = get_series(i, "q", has_unit) * get_series(i, "p", has_unit)
        return r.rename("r")
    elif name == "nodim":
        dtype = "pint[dimensionless]" if has_unit else float
        return pd.Series(0.9 + 0.2 * np.random.rand(len(i)), i, dtype=dtype)
    else:
        return pd.Series(100 + 100 * np.random.rand(len(i)), i, name=name)


def get_dataframe(
    i: pd.DatetimeIndex = None,
    columns="wp",
    has_unit: bool = True,
    *,
    _seed: int = None,
) -> pd.DataFrame:
    """Get DataFrame with index `i` and columns `columns`. Columns (e.g. `q` and `w`)
    are not made consistent."""
    if i is None:
        i = get_index(_seed=_seed)
    i.name = "ts_left"
    series = {col: get_series(i, col, has_unit, _seed=_seed) for col in columns}
    return pd.DataFrame(series)


# Portfolio line.


def get_pfline(
    i: pd.DatetimeIndex = None,
    kind: Kind = Kind.COMPLETE,
    nlevels: int = 1,
    *,
    _seed: int = None,
) -> FlatPfLine:
    """Get a portfolio line."""
    if nlevels == 1:
        return get_flatpfline(i, kind, _seed=_seed)
    else:
        return get_nestedpfline(i, kind, nlevels, _seed=_seed)


def get_flatpfline(
    i: pd.DatetimeIndex = None, kind: Kind = Kind.COMPLETE, *, _seed: int = None
) -> FlatPfLine:
    """Get flat portfolio line, i.e. without children."""
    columns = {
        Kind.VOLUME: "q",
        Kind.PRICE: "p",
        Kind.REVENUE: "r",
        Kind.COMPLETE: "qr",
    }[kind]
    return create.flatpfline(get_dataframe(i, columns, _seed=_seed))


def get_nestedpfline(
    i: pd.DatetimeIndex = None,
    kind: Kind = Kind.COMPLETE,
    nlevels: int = 2,
    *,
    _seed: int = None,
) -> NestedPfLine:
    """Get nested portfolio line. With 2 children of the same ``kind``. If ``nlevels``==2, the children are both flat; if not,
    the portfolio line is multiply nested."""
    if i is None:
        i = get_index(_seed=_seed)
    if nlevels == 2:
        return create.nestedpfline(
            {
                "A": get_flatpfline(i, kind, _seed=_seed),
                "B": get_flatpfline(i, kind, _seed=_seed),
            }
        )
    else:
        return create.nestedpfline(
            {
                "A": get_nestedpfline(i, kind, nlevels - 1, _seed=_seed),
                "B": get_nestedpfline(i, kind, nlevels - 1, _seed=_seed),
            }
        )


def get_randompfline(
    i: pd.DatetimeIndex = None,
    kind: Kind = Kind.COMPLETE,
    max_nlevels: int = 3,
    max_childcount: int = 2,
    prefix: str = "",
    *,
    _seed: int = None,
) -> PfLine:
    """Get portfolio line, without children or with children in random number of levels.
    (including the current level; max_nlevels must be >= 1.)"""
    # Gather information.
    if i is None:
        i = get_index(_seed=_seed)
    if _seed:
        np.random.seed(_seed)
    nlevels = np.random.randint(1, max_nlevels + 1)
    # Create flat PfLine.
    if nlevels == 1:
        return get_flatpfline(i, kind, _seed=_seed)
    # Create nested PfLine.
    children = {}
    childcount = np.random.randint(1, max_childcount + 1)
    for c in range(childcount):
        name = f"part {prefix}{c}."
        children[name] = get_randompfline(
            i,
            kind,
            max_nlevels - 1,
            max_childcount,
            prefix=f"{prefix}{c}.",
            _seed=_seed,
        )
    return create.nestedpfline(children)


# Portfolio state.


def get_pfstate(i: pd.DatetimeIndex = None, avg=None, *, _seed: int = None) -> PfState:
    """Get portfolio state."""
    if i is None:
        i = get_index(_seed=_seed)
    if _seed:
        np.random.seed(_seed)
    if avg is None:
        avg = 200 ** np.random.rand()  # between 1 and 200
    wo = -1 * mockup.w_offtake(i, avg)
    pu = mockup.p_marketprices(i)
    ws, ps = mockup.wp_sourced(wo)
    return PfState.from_series(wo=wo, pu=pu, ws=ws, ps=ps)


def get_pfstates(
    i: pd.DatetimeIndex = None, num=3, *, _seed: int = None
) -> Dict[str, PfState]:
    """Get dictionary of portfolio states."""
    if i is None:
        i = get_index(_seed=_seed)
    names = ["Pf 1", "Portfolio 2 (long name)", "Portfolio number three"]
    for n in range(3, num):
        names.append(f"Portfolio {n+1}")
    return {name: get_pfstate(i, _seed=_seed) for name in names[:num]}
