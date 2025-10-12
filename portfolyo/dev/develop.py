"""Code to quickly get objects for testing."""

import datetime as dt
from typing import Callable, Iterable, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pint

from .. import tools
from ..core.commodity import Commodity, gas_ger
from ..core.pfline import (
    FlatPfLine,
    Kind,
    NestedPfLine,
    PfLine,
    create_flatpfline,
    create_nestedpfline,
)

# from ..core.pfstate import PfState
from ..tools.types import Col, FloatTimeSeries, Frequencylike, PintTimeDataframe, PintTimeSeries
from ..tools.unit import Q_, ureg
from . import mockup

NOTSPECIFIED = "notspecified"  # sentinel value to allow user to pass `None` as valid value

COL_TO_UNITS = {
    "w": ureg.MW,
    "q": ureg.MWh,
    "p": ureg.euro / ureg.MWh,
    "r": ureg.euro,
    "duration": ureg.hour,
    "nodim": ureg.dimensionless,
}

_PERIODS = {"YS": 1, "QS": 4, "MS": 12, "D": 400, "h": 10_000, "15min": 40_000, "5min": 120_000}


def _periods_and_startdate(freq: Frequencylike) -> tuple[int, str]:
    # Periods.
    for freq2, periods in _PERIODS.items():
        if tools.freq.up_or_down(freq, freq2) >= 0:
            break
    else:
        raise ValueError("Couldn't find a fitting frequency.")
    periods = np.random.randint(periods, periods * 3)

    # Startdate.
    y, m, d = 2016, 1, 1  # earliest possible
    y += np.random.randint(0, 8)  # if _seed else (periods % 8)
    if tools.freq.up_or_down(freq, "MS") <= 0:
        m += np.random.randint(0, 12)  # if _seed else (periods % 12)
    if tools.freq.up_or_down(freq, "D") <= 0:
        d += np.random.randint(0, 28)  # if _seed else (periods % 28)
    startdate = f"{y}-{m}-{d}"

    return periods, startdate


def get_scalar(
    col: Literal["w", "q", "p", "r", "nodim", "duration"],
    has_unit: bool = True,
    magn: float | None = None,
    *,
    _seed: int | None = None,
) -> float | pint.Quantity:
    """Get a single value."""
    if _seed:
        np.random.seed(_seed)
    if magn is None:
        magn = np.random.random() * 200
    if not has_unit:
        return magn
    else:
        return Q_(magn, COL_TO_UNITS[col])


def get_index(
    commodity: Commodity | None,
    freq: Frequencylike | None = None,
    tz: str | ZoneInfo | None = NOTSPECIFIED,
    startofday: dt.time | str | None = None,
    *,
    _seed: int | None = None,
) -> pd.DatetimeIndex:
    """Get index. Commodity must be specified; other parameters can be specified to override defaults
    in commodity.

    Parameters
    ----------
    commmodity
        Commodity for which to create the index. None to leave unspecified.
    freq, optional (default: shortest allowed by commodity, or daily if no commodity)
        Frequency of datetimeindex.
    tz, optional (default: as specified by commodity, or None if no commodity)
        Timezone of datetimeindex.
    startofday, optional (default: as specified by commodity, or midnight if no commodity)
        Time of first element in datetimeindex.

    Returns
        DatetimeIndex with specified properties.

    Notes
    -----
    If overrides are specified, the returned object might not be valid for specified commodity.
    """
    # Data prep.
    if freq is None:
        freq = commodity.freq if commodity is not None else "D"
    freq = tools.freq.coerce(freq)

    if tz == NOTSPECIFIED:
        tz = commodity.tz if commodity is not None else None
    tz = tools.tz.coerce(tz)

    if startofday is None:
        startofday = commodity.startofday if commodity is not None else tools.startofday.MIDNIGHT
    startofday = tools.startofday.coerce(startofday)

    # Calculate index.
    if _seed:
        np.random.seed(_seed)
    periods, startdate = _periods_and_startdate(freq)
    idx = pd.date_range(
        f"{startdate} {tools.startofday.to_string(startofday)}", periods=periods, freq=freq, tz=tz
    )
    # Some checks.
    if tools.freq.is_shorter_than_daily(freq):
        idx = _ensure_start_end_same_time(idx, startofday)
    return idx


def _ensure_start_end_same_time(idx: pd.DatetimeIndex, start_of_day: dt.time) -> pd.DatetimeIndex:
    """Shorten index with shorter-than-daily values, if necessary to ensure that an integer number
    of calendar days is included."""
    if tools.stamp.to_right(idx[-1], idx.freq).time() == start_of_day:
        return idx  # already correct
    while not idx.empty:
        if idx[-1].time() == start_of_day:
            return idx[:-1]  # exclude last element
        idx = idx[:-1]  # remove last element
    raise ValueError("Can't find timestamp to end index on.")


def get_series(
    commodity: Commodity | None,
    idx: pd.DatetimeIndex | None = None,
    col: Col = "w",
    *,
    _seed: int | None = None,
) -> PintTimeSeries:
    """Get PintSeries.

    Parameters
    ----------
    commmodity
        Commodity for which to create the series. Used for unit and (if ``idx`` not specified) to
        create datetimeindex. If None, pick unit as specified in COL_TO_UNITS.
    idx, optional (default: create random)
        DatetimeIndex to use.
    col, optional (default: 'w')
        Column to create series for.

    Returns
    -------
        Series with specified properties.
    """
    idx = get_index(commodity, _seed=_seed) if idx is None else idx.copy()
    idx.name = "ts_left"

    if _seed:
        np.random.seed(_seed)

    if col == "w":
        # random average, and 3 random amplitudes with sum < 1
        avg = 30 + 10 * np.random.random()
        ampls = np.random.rand(3) * np.array([0.3, 0.2, 0.1])
        w = mockup.w_offtake(commodity, idx, avg, *ampls)
        return w if commodity else w.astype(f'pint[{COL_TO_UNITS["w"]}]')
    elif col == "q":
        q = get_series(commodity, idx, "w") * tools.index.duration(idx)
        unit = commodity.col_to_units["q"] if commodity else COL_TO_UNITS["q"]
        return q.rename("q").pint.to(unit)
    elif col == "p":
        # random average, and 3 random amplitudes with sum < 1
        avg = 100 + 20 * np.random.random()
        ampls = np.random.rand(3) * np.array([0.25, 0.04, 0.3])
        p = mockup.p_marketprices(commodity, idx, avg, *ampls)
        return p if commodity else p.astype(f"pint[{COL_TO_UNITS['p']}]")
    elif col == "r":
        r = get_series(commodity, idx, "q") * get_series(commodity, idx, "p")
        unit = commodity.col_to_units["r"] if commodity else COL_TO_UNITS["r"]
        return r.rename("r").pint.to(unit)


def get_dataframe(
    commodity: Commodity | None,
    idx: pd.DatetimeIndex | None = None,
    cols: Iterable[Literal["w", "q", "p", "r"]] = ("w", "p"),
    *,
    _seed: int | None = None,
) -> PintTimeDataframe:
    """Get DataFrame with index ``idx`` and columns ``cols``. Columns (e.g. `q` and `w`) are not
    made consistent."""
    idx = get_index(commodity, _seed=_seed) if idx is None else idx.copy()
    return pd.DataFrame({col: get_series(commodity, idx, col, _seed=_seed) for col in cols})


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
    commodity: Commodity | None,
    kind: Kind = Kind.COMPLETE,
    nlevels: int = 1,
    childcount: int = 2,
    *,
    positive: bool = False,
    _ancestornames: tuple[str] = tuple(),
    _seed: int | None = None,
) -> PfLine:
    """
    Create a portfolio line.

    Parameters
    ----------
    commodity
        Commodity to use. None to not use a commodity.
    kind, optional (default: COMPLETE)
    nlevels, optional (default: 1)
        Number of levels. Must be >=1. If 1, return flat portfolio line.
    childcount, optional (default: 2)
        Number of children on each level. (Ignored if `nlevels` == 1)
    positive, optional (default: False)
        If True, return only positive values. If False, make 1/2 of pflines negative.
    _ancestornames, optional (default: ())
        Text to start the childrens' names with (concatenated with '-')
    _seed, optional (default: no seed value)
        Seed value for the randomizer.

    Returns
    -------
        The portfolio line.
    """
    if commodity is None:
        return get_pfline(
            gas_ger,  # use gas_ger as commodity, later remove.
            kind,
            nlevels,
            childcount,
            positive=positive,
            _ancestornames=_ancestornames,
            _seed=_seed,
        ).set_commodity(None)

    # Gather information.
    idx = get_index(commodity, _seed=_seed)

    # Create flat pfline.
    if nlevels == 1:
        df = get_dataframe(commodity, idx, kind.summable, _seed=_seed)
        if not positive and np.random.randint(1, 4) == 1:
            df = -df
        return create_flatpfline(df, commodity)
    # Create nested PfLine.
    if childcount < 1:
        raise ValueError("Nested PfLine must have at least 1 child.")
    namefn = _get_namefn(nlevels)
    children = {}
    for c in range(childcount):
        names = (*_ancestornames, namefn(c))
        name = "-".join(names)
        children[name] = get_pfline(
            idx, kind, nlevels - 1, childcount, _ancestornames=names, _seed=_seed
        )
    return create_nestedpfline(children)


def get_flatpfline(
    commodity: Commodity | None,
    idx: pd.DatetimeIndex | None = None,
    kind: Kind = Kind.COMPLETE,
    *,
    _seed: int | None = None,
) -> FlatPfLine:
    """Get flat portfolio line, i.e. without children."""
    return get_pfline(commodity, idx, kind, 1, _seed=_seed)


def get_nestedpfline(
    commodity: Commodity | None,
    idx: pd.DatetimeIndex | None = None,
    kind: Kind = Kind.COMPLETE,
    nlevels: int = 2,
    childcount: int = 2,
    *,
    _seed: int | None = None,
) -> NestedPfLine:
    """Get nested portfolio line with children of the same ``kind``. If ``nlevels``==2, the children are both flat; if not,
    the portfolio line is multiply nested."""
    if nlevels <= 1:
        raise ValueError(
            "Nested PfLine must have at least 2 levels. Use `get_flatpfline` or the more general `get_pfline` instead."
        )
    return get_pfline(commodity, idx, kind, nlevels, childcount, _seed=_seed)


def get_randompfline(
    commodity: Commodity | None,
    idx: pd.DatetimeIndex | None = None,
    kind: Kind = Kind.COMPLETE,
    max_nlevels: int = 3,
    max_childcount: int = 2,
    *,
    positive: bool = False,
    _ancestornames: tuple[str] = tuple(),
    _seed: int | None = None,
) -> PfLine:
    """Get portfolio line, without children or with children in random number of levels.
    (including the current level; max_nlevels must be >= 1.)"""
    # Gather information.
    if idx is None:
        idx = get_index(commodity, _seed=_seed)
    if _seed:
        np.random.seed(_seed)
    nlevels = np.random.randint(1, max_nlevels + 1)
    # Create flat PfLine.
    if nlevels == 1:
        return get_pfline(commodity, idx, kind, 1, positive=positive, _seed=_seed)
    # Create nested PfLine.
    namefn = _get_namefn(nlevels)
    children = {}
    childcount = np.random.randint(1, max_childcount + 1)
    for c in range(childcount):
        names = (*_ancestornames, namefn(c))
        name = "-".join(names)
        children[name] = get_randompfline(
            commodity,
            idx,
            kind,
            max_nlevels - 1,
            max_childcount,
            positive=positive,
            _ancestornames=names,
            _seed=_seed,
        )
    return create_nestedpfline(children, commodity)


# Portfolio state.

#
# def get_pfstate(
#     idx: pd.DatetimeIndex | None = None, avg=None, *, _seed: int | None = None
# ) -> PfState:
#     """Get portfolio state."""
#     if idx is None:
#         idx = get_index(_seed=_seed)
#     if _seed:
#         np.random.seed(_seed)
#     if avg is None:
#         avg = 200 ** np.random.rand()  # between 1 and 200
#     wo = -1 * mockup.w_offtake(idx, avg)
#     pu = mockup.p_marketprices(idx)
#     ws, ps = mockup.wp_sourced(wo)
#     return PfState.from_series(wo=wo, pu=pu, ws=ws, ps=ps)
#
#
# def get_pfstates(
#     idx: pd.DatetimeIndex | None = None, num=3, *, _seed: int | None = None
# ) -> Dict[str, PfState]:
#     """Get dictionary of portfolio states."""
#     if idx is None:
#         idx = get_index(_seed=_seed)
#     names = ["Pf 1", "Portfolio 2 (long name)", "Portfolio number three"]
#     for n in range(3, num):
#         names.append(f"Portfolio {n+1}")
#     return {name: get_pfstate(idx, _seed=_seed) for name in names[:num]}
