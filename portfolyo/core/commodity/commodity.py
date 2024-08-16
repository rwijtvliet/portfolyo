import datetime as dt
from dataclasses import dataclass

from ... import tools


@dataclass
class Commodity:
    """Class to describe the particulars of a commodity."""

    freq: str
    is_peak_hour: tools.peakperiod.PeakFunction = None
    offset_hours: int = 0

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
