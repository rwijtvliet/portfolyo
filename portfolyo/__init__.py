"""Package to analyse and manipulate timeseries related to power and gas offtake portfolios."""

from pathlib import Path

from . import dev, tools, toolsb
from .__version__ import __version__
from .core import extendpandas  # extend functionalty of pandas
from .core import suppresswarnings
from .core.commodity import Commodity, gas_ger, power_ger
from .core.pfline import PfLine, create
from .core.pflineb import Kind, PfLineb, Structure
from .core.pfstate import PfState
from .tools import testing
from .tools2.concat import general as concat
from .tools2.intersect import indexable as intersection
from .tools2.plot import plot_pfstates
from .tools.changefreq import averagable as asfreq_avg
from .tools.changefreq import summable as asfreq_sum
from .tools.freq import assert_freq_valid
from .tools.hedge import hedge
from .tools.peakfn import PeakFunction
from .tools.peakfn import factory as create_peakfn
from .tools.product import germanpower_peakfn, is_peak_hour
from .tools.right import index as right_index
from .tools.right import stamp as right_stamp
from .tools.standardize import frame as standardize
from .tools.tzone import force_agnostic, force_aware
from .tools.unit import normalize_frame as pintframe

# toolsb
from .toolsb.index import duration
from .toolsb.peakfn import PeakFunction, base_duration, offpeak_duration, peak_duration
from .toolsb.unit import Q_, ureg
from .toolsb.wavg import general as wavg

VOLUME = Kind.VOLUME
PRICE = Kind.PRICE
REVENUE = Kind.REVENUE
COMPLETE = Kind.COMPLETE

extendpandas.apply()
suppresswarnings.apply()


__all__ = ["tools", "dev", "PfLine", "PfState"]
