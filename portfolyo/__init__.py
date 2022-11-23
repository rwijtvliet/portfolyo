"""Package to analyse and manipulate timeseries related to power and gas offtake portfolios."""

from .core.pfline import PfLine, SinglePfLine, MultiPfLine, Kind
from .core.pfstate import PfState
from .core.mixins.plot import plot_pfstates
from .core import changefreq
from . import dev

from .core import extendpandas  # extend functionalty of pandas
from .core import suppresswarnings

extendpandas.apply()
suppresswarnings.apply()

# Methods/attributes directly available at package root.

from .tools.stamps import (
    FREQUENCIES,
    floor_ts,
    ceil_ts,
    ts_leftright,
    freq_longest,
    freq_shortest,
    freq_up_or_down,
    right_to_left,
)

from .tools import frames, nits, zones, stamps
from .tools.frames import fill_gaps, wavg, standardize, set_frequency
from .tools.nits import Q_
from .tools.zones import force_tzaware, force_tzagnostic

from .prices.hedge import hedge
from .prices.utils import is_peak_hour

from . import _version

__version__ = _version.get_versions()["version"]
