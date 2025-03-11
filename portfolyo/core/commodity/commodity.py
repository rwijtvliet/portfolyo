import datetime as dt
from dataclasses import dataclass

from ... import tools


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

    freq: str
    is_peak_hour: tools.peakperiod.PeakFunction = None
    startofday: dt.time | str | dt.timedelta = None
    units: str = None

    def __post_init__(self):
        # if self.freq not in (freqs := tools.freq.FREQUENCIES):
        #     raise ValueError(
        #         f"``freq`` must be one of {', '.join(freqs)}; got {self.freq}."
        #     )
        tools.freq.assert_freq_valid(self.freq)


power = Commodity(
    "15min",
    tools.peakperiod.factory(dt.time(hour=8), dt.time(hour=20), [1, 2, 3, 4, 5]),
    0,
)
gas = Commodity("D", None, 6)
