import datetime as dt
import zoneinfo
from dataclasses import InitVar, dataclass, field
from typing import Iterable

import pint
import pytz
from pandas.tseries.offsets import BaseOffset

from ... import toolsb


@dataclass(frozen=True)
class Commodity:
    """Class to describe the particulars of a commodity.

    Parameters
    ----------
    name
        Display name for the commodity.
    freq
        Shortest timeperiod of this commodity, e.g. when traded on spot or intraday markets.
    units
        Units to use. Must contain at least a unit for the following dimensions:
        - energy or emissions, like GWh or tCO2;
        - energy rate or emissions rate, like MW or kgCO2/h;
        - energy price or emissions price, like ctEur/kWh or Usd/tCO2;
        - monetary value, like Eur or Usd.
        The dimensions must be compatible with each other, and are used to determine if the commodity describes
        energy or emissions, as well as the currency.
    tz
        Timezone. (Explicit `None` if no timezone, i.e., timezone-agnostic.)
    peak_fn, optional
        Function that returns boolean array indicating if specified DatetimeIndex is a peak-hour.
        Specify only if commodity has notion of peak hours and offpeak hours.
    startofday, optional
        Starting/ending time of daily-or-longer delivery periods of this commodity. E.g. for
        European natural gas '06:00'. Default: midnight.
    """

    name: str
    freq: str | BaseOffset
    units: InitVar[Iterable[str | pint.Unit]]
    tz: str | dt.tzinfo | None
    peak_fn: toolsb.peakfn.PeakFunction | None = None
    startofday: dt.time | str | dt.timedelta = toolsb.startofday.MIDNIGHT
    unitpref: toolsb.unit.UnitPref = field(init=False)
    col_to_dimty: dict[toolsb.types.Col, pint.util.UnitsContainer] = field(init=False)
    col_to_units: dict[toolsb.types.Col, pint.Unit] = field(init=False)

    def __post_init__(self, units):
        # if self.freq not in (freqs := tools.freq.FREQUENCIES):
        #     raise ValueError(
        #         f"``freq`` must be one of {', '.join(freqs)}; got {self.freq}."
        #     )
        object.__setattr__(self, "freq", toolsb.freq.coerce(self.freq))
        object.__setattr__(self, "startofday", toolsb.startofday.coerce(self.startofday))

        # Post-processing units.
        object.__setattr__(self, "unitpref", toolsb.unit.UnitPref.from_objs(units, "raise"))
        # . Ensure no mixing of energy and emissions units.
        toolsb.wqpr.validate_compatible(self.unitpref.keys())
        # . Ensure a unit is available for each important dimension.
        toolsb.wqpr.validate_complete(self.unitpref.keys())
        # . Store base dimensionality for each important dimension.
        col2dimty = {}
        col2units = {}
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
            col2dimty[col] = dimty
            col2units[col] = self.unitpref[dimty]
        object.__setattr__(self, "col_to_dimty", col2dimty)
        object.__setattr__(self, "col_to_units", col2units)

        # Post-processing timezone.
        if self.tz is not None and not isinstance(self.tz, zoneinfo.ZoneInfo):
            if isinstance(self.tz, str):
                object.__setattr__(self, "tz", zoneinfo.ZoneInfo(self.tz))
            elif isinstance(self.tz, pytz.BaseTzInfo):
                object.__setattr__(self, "tz", zoneinfo.ZoneInfo(self.tz.zone))
            else:
                raise TypeError(
                    f"Expected timezone, timezone name (like Europe/Berlin), or None for parameter ``tz``; got {type(self.tz)}."
                )


power_ger = Commodity(
    "Power, Germany",
    "15min",
    ["MWh", "Eur/MWh", "MW", "Eur"],
    "Europe/Berlin",
    peak_fn=toolsb.product.germanpower_peakfn,
)
power_generic = Commodity(
    "Power, Generic",
    "15min",
    ["MWh", "Eur/MWh", "MW", "Eur"],
    None,
    peak_fn=toolsb.product.germanpower_peakfn,
)
gas_ger = Commodity(
    "Gas, Germany",
    "D",
    ["MWh", "Eur/MWh", "MW", "Eur"],
    "Europe/Berlin",
    startofday="06:00",
)
coal_ger = Commodity(
    "Coal, Germany",
    "D",
    ["ktce", "Eur/tce", "tce/h", "Eur"],
    "Europe/Berlin",
)
co2_ger = Commodity(
    "CO2, Germany",
    "D",
    ["ktCo2", "Eur/tCo2", "tCo2/h", "Eur"],
    "Europe/Berlin",
)
