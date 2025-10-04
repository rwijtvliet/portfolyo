import datetime as dt
from dataclasses import InitVar, dataclass, field
from typing import Iterable
from zoneinfo import ZoneInfo

import pint
from pandas.tseries.offsets import BaseOffset

from ... import toolsb


@dataclass(frozen=True, kw_only=True)
class Commodity:
    """Class to describe the particulars of a commodity.

    Parameters
    ----------
    name
        Display name for the commodity.
    freq
        Shortest timeperiod of this commodity, e.g. when traded on spot or intraday markets.
    tz
        Timezone. (Explicit `None` if no timezone, i.e., timezone-agnostic.)
    units
        Units to use. Must contain at least a unit for the following dimensions:
        - energy or emissions, like GWh or tCO2;
        - energy rate or emissions rate, like MW or kgCO2/h;
        - energy price or emissions price, like ctEur/kWh or Usd/tCO2;
        - monetary value, like Eur or Usd.
        The dimensions must be compatible with each other, and are used to determine if the commodity
        describes energy or emissions, as well as the currency.
    peak_fn, optional
        Function that returns boolean array indicating if specified DatetimeIndex is a peak-hour.
        Specify only if commodity has notion of peak hours and offpeak hours.
    startofday, optional
        Starting/ending time of daily-or-longer delivery periods of this commodity. E.g. for
        European natural gas '06:00'. Default: midnight.
    """

    # Fields. Type = type after __post_init__
    # . Required fields.
    name: str
    freq: BaseOffset  # init with BaseOffset | str
    tz: ZoneInfo  # init with str | dt.tzinfo | "pytz.BaseTzInfo" | ZoneInfo | None
    units: InitVar[Iterable[str | pint.Unit]]
    # . Optional fields.
    peak_fn: toolsb.peakfn.PeakFunction | None = None
    startofday: dt.time = toolsb.startofday.MIDNIGHT  # init with dt.time | str | dt.timedelta
    # . Calculated fields.
    unitpref: toolsb.unit.UnitPref = field(init=False)
    col_to_dimty: dict[toolsb.types.Col, pint.util.UnitsContainer] = field(init=False)
    col_to_units: dict[toolsb.types.Col, pint.Unit] = field(init=False)

    def __post_init__(self, units):
        object.__setattr__(self, "freq", toolsb.freq.coerce(self.freq))
        object.__setattr__(self, "startofday", toolsb.startofday.coerce(self.startofday))

        # Post-processing units.
        object.__setattr__(self, "unitpref", toolsb.unit.UnitPref.from_objs(units, "raise"))
        # . Ensure no mixing of energy and emissions units, and ensure each column has a unit.
        toolsb.wqpr.validate_compatible(self.unitpref.keys())
        toolsb.wqpr.validate_complete(self.unitpref.keys())
        # . Store base dimensionality for each column.
        col_to_dimty = {}
        col_to_units = {}
        for col in toolsb.types.COLS:
            allowed_dimties = toolsb.wqpr.col_to_dimties(col)
            for dimty in allowed_dimties:
                if dimty in self.unitpref:
                    break
            else:
                raise ValueError(
                    f"Could not find a unit for column '{col}'. Expected a unit for one of the"
                    f" following dimensionalities: {allowed_dimties}."
                )
            col_to_dimty[col] = dimty
            col_to_units[col] = self.unitpref[dimty]
        object.__setattr__(self, "col_to_dimty", col_to_dimty)
        object.__setattr__(self, "col_to_units", col_to_units)

        # Post-processing timezone.
        object.__setattr__(self, "tz", toolsb.tz.coerce(self.tz))


power_ger = Commodity(
    name="Power, Germany",
    freq="15min",
    units=["MWh", "Eur/MWh", "MW", "Eur"],
    tz="Europe/Berlin",
    peak_fn=toolsb.product.germanpower_peakfn,
)
power_generic = Commodity(
    name="Power, Generic",
    freq="15min",
    units=["MWh", "Eur/MWh", "MW", "Eur"],
    tz=None,
    peak_fn=toolsb.product.germanpower_peakfn,
)
gas_ger = Commodity(
    name="Gas, Germany",
    freq="D",
    units=["MWh", "Eur/MWh", "MW", "Eur"],
    tz="Europe/Berlin",
    startofday="06:00",
)
coal_ger = Commodity(
    name="Coal, Germany",
    freq="D",
    units=["ktce", "Eur/tce", "tce/h", "Eur"],
    tz="Europe/Berlin",
)
co2_ger = Commodity(
    name="CO2, Germany",
    freq="D",
    units=["ktCo2", "Eur/tCo2", "tCo2/h", "Eur"],
    tz="Europe/Berlin",
)
