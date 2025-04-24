import datetime as dt
from dataclasses import InitVar, dataclass
from typing import Iterable

import pint
from pandas.tseries.offsets import BaseOffset

from ... import toolsb


@dataclass(frozen=True)
class Commodity:
    """Class to describe the particulars of a commodity.

    Parameters
    ----------
    freq
        Shortest timeperiod of this commodity, e.g. when traded on spot or intraday markets.
    peakfn, optional
        Function that returns boolean array indicating if specified DatetimeIndex is a peak-hour.
        Specify only if commodity has notion of peak hours and offpeak hours.
    startofday, optional
        Starting/ending time of daily-or-longer delivery periods of this commodity. E.g. for
        European natural gas '06:00'. Default: midnight.
    units, optional
    """

    freq: str | BaseOffset
    is_peak_hour: toolsb.peakfn.PeakFunction | None = None
    startofday: dt.time | str | dt.timedelta = toolsb.startofday.MIDNIGHT
    preferred_units: InitVar[Iterable[str | pint.Unit] | None] = None

    def __post_init__(self, preferred_units):
        # if self.freq not in (freqs := tools.freq.FREQUENCIES):
        #     raise ValueError(
        #         f"``freq`` must be one of {', '.join(freqs)}; got {self.freq}."
        #     )
        object.__setattr__(self, "freq", toolsb.freq.coerce(self.freq))
        object.__setattr__(self, "startofday", toolsb.startofday.coerce(self.startofday))
        object.__setattr__(
            self, "unitpref", toolsb.unit.UnitPref.from_objs(preferred_units, "raise")
        )


power = Commodity(
    "15min", toolsb.product.germanpower_peakfn, preferred_units=["MWh", "Eur/MWh", "MW"]
)
gas = Commodity("D", startofday="06:00", preferred_units=["MWh", "Eur/MWh", "MW"])
coal = Commodity("D", preferred_units=["ktce", "Eur/tce", "tce/h"])
co2 = Commodity("D", preferred_units=["ktCo2", "Eur/tCo2", "tCo2/h"])
