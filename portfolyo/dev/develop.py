"""
Code to quickly get objects for testing.
"""

import datetime as dt
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from .. import tools
from ..core.pfline import FlatPfLine, Kind, NestedPfLine, PfLine, create
from ..core.pfstate import PfState
from . import mockup

OK_COL_COMBOS = ["w", "q", "p", "pr", "qr", "qp", "wp", "wr"]

INDEX_LEN = {"YS": 4, "QS": 5, "MS": 14, "D": 400, "h": 10_000, "15min": 50_000}


def get_index(
    freq: str = "D",
    tz: str = "Europe/Berlin",
    startdate: str = None,
    periods: int = None,
    start_of_day: dt.time = None,
    *,
    _seed: int = None,
) -> pd.DatetimeIndex:
    """Get index."""
    # Prepare values.
    if _seed:
        np.random.seed(_seed)
    if not periods:
        standard_len = INDEX_LEN.get(freq, 10)
        periods = np.random.randint(standard_len // 2, standard_len * 2)
    if not startdate:
        a, m, d = 2016, 1, 1  # earliest possible
        a += np.random.randint(0, 8) if _seed else (periods % 8)
        if tools.freq.up_or_down(freq, "MS") <= 0:
            m += np.random.randint(0, 12) if _seed else (periods % 12)
        if tools.freq.up_or_down(freq, "D") <= 0:
            d += np.random.randint(0, 28) if _seed else (periods % 28)
        startdate = f"{a}-{m}-{d}"
    if not start_of_day:
        start_of_day = dt.time(hour=0, minute=0)
    # Create index.
    start = tools.stamp.create(startdate, tz, start_of_day)
    i = pd.date_range(start, periods=periods, freq=freq)  # tz included in start
    # Some checks.
    if tools.freq.up_or_down(freq, "h") <= 0:
        i = _shorten_index_if_necessary(i, start_of_day)
    return i


def _shorten_index_if_necessary(i, start_of_day) -> pd.DatetimeIndex:
    """Shorten index with (quarter)hourly values if necessary to ensure that an integer
    number of calendar days is included."""
    if (i[-1] - i[0]).total_seconds() < 23 * 3600:
        raise ValueError("Index must contain at least one full day")
    # Must ensure that index is integer number of days.
    for _ in range(0, 100):  # max 100 quarterhours in a day (@ end of DST)
        if tools.right.stamp(i[-1], i.freq).time() == start_of_day:
            return i
        i = i[:-1]
    raise ValueError("Can't find timestamp to end index on.")


def get_value(
    name: str = None, has_unit: bool = True, magn: float = None, *, _seed: int = None
) -> float | tools.unit.Q_:
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


def _get_namefn(nlevels: int) -> Callable[[int], str]:
    """Name function. Turns int into string in various ways."""
    if nlevels % 5 == 0:

        def fn(i: int) -> str:
            return str(i)  # simply the number

    elif nlevels % 5 == 1:

        def fn(i: int) -> str:
            return chr(i + 65)  # capital letter

    elif nlevels % 5 == 2:

        def fn(i: int) -> str:
            return chr(i + 65 + 32)  # small letter

    elif nlevels % 5 == 3:
        symbols = [(40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]

        def fn(i: int) -> str:
            i += 1  # make sure start with 1
            roman = ""
            for value, symbol in symbols:
                while i >= value:
                    roman += symbol
                    i -= value
            return roman  # roman numeral

    else:

        def fn(i: int) -> str:
            i += 1  # make sure start with 1
            tens, ones = divmod(i, 10)
            suffix = "th"
            if tens != 1:
                if ones == 1:
                    suffix = "st"
                elif ones == 2:
                    suffix = "nd"
                elif ones == 3:
                    suffix = "rd"
            return f"{i}{suffix}"  # 1st, 2nd, 3rd, 4th, ...

    return fn


def get_pfline(
    i: pd.DatetimeIndex = None,
    kind: Kind = Kind.COMPLETE,
    nlevels: int = 1,
    childcount: int = 2,
    *,
    positive: bool = False,
    _ancestornames: Tuple[str] = (),
    _seed: int = None,
) -> PfLine:
    """
    Create a portfolio line.

    Parameters
    ----------
    i : pd.DatetimeIndex, optional (default: random)
        Datetimeindex to use for the portfolio line.
    kind : Kind, optional (default: COMPLETE)
    nlevels : int (default: 1)
        Number of levels. Must be >=1 If 1, return flat portfolio line.
    childcount : int, optional (default: 2)
        Number of children on each level. (Ignored if `nlevels` == 1)
    positive : bool, optional (default: False)
        If True, return only positive values. If False, make 1/2 of pflines negative.
    _ancestornames : Tuple[str], optional (default: ())
        Text to start the childrens' names with (concatenated with '-')
    _seed : int, optional (default: no seed value)
        Seed value for the randomizer.

    Returns
    -------
    PfLine
    """
    # Gather information.
    if i is None:
        i = get_index(_seed=_seed)
    if _seed:
        np.random.seed(_seed)
    # Create flat pfline.
    if nlevels == 1:
        columns = {
            Kind.VOLUME: "q",
            Kind.PRICE: "p",
            Kind.REVENUE: "r",
            Kind.COMPLETE: "qr",
        }[kind]
        df = get_dataframe(i, columns, _seed=_seed)
        if not positive and np.random.randint(1, 4) == 1:
            df = -1 * df  # HACK: `-df` leads to error in pint. Maybe fixed in future
        return create.flatpfline(df)
    # Create nested PfLine.
    if childcount < 1:
        raise ValueError("Nested PfLine must have at least 1 child.")
    namefn = _get_namefn(nlevels)
    children = {}
    for c in range(childcount):
        names = (*_ancestornames, namefn(c))
        name = "-".join(names)
        children[name] = get_pfline(
            i, kind, nlevels - 1, childcount, _ancestornames=names, _seed=_seed
        )
    return create.nestedpfline(children)


def get_flatpfline(
    i: pd.DatetimeIndex = None, kind: Kind = Kind.COMPLETE, *, _seed: int = None
) -> FlatPfLine:
    """Get flat portfolio line, i.e. without children."""
    return get_pfline(i, kind, 1, _seed=_seed)


def get_nestedpfline(
    i: pd.DatetimeIndex = None,
    kind: Kind = Kind.COMPLETE,
    nlevels: int = 2,
    childcount: int = 2,
    *,
    _seed: int = None,
) -> NestedPfLine:
    """Get nested portfolio line with children of the same ``kind``. If ``nlevels``==2, the children are both flat; if not,
    the portfolio line is multiply nested."""
    if nlevels <= 1:
        raise ValueError(
            "Nested PfLine must have at least 2 levels. Use `get_flatpfline` or the more general `get_pfline` instead."
        )
    return get_pfline(i, kind, nlevels, childcount, _seed=_seed)


def get_randompfline(
    i: pd.DatetimeIndex = None,
    kind: Kind = Kind.COMPLETE,
    max_nlevels: int = 3,
    max_childcount: int = 2,
    *,
    positive: bool = False,
    _ancestornames: Tuple[str] = (),
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
        return get_pfline(i, kind, 1, positive=positive, _seed=_seed)
    # Create nested PfLine.
    namefn = _get_namefn(nlevels)
    children = {}
    childcount = np.random.randint(1, max_childcount + 1)
    for c in range(childcount):
        names = (*_ancestornames, namefn(c))
        name = "-".join(names)
        children[name] = get_randompfline(
            i,
            kind,
            max_nlevels - 1,
            max_childcount,
            positive=positive,
            _ancestornames=names,
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
