"""Package to analyse and manipulate timeseries related to power and gas offtake portfolios."""

from pathlib import Path

# from .core.pfstate import PfState
from . import dev, tools
from .__version__ import __version__
from .core import extendpandas  # extend functionalty of pandas
from .core import suppresswarnings
from .core.commodity import Commodity, gas_ger, power_ger
from .core.pfline import FlatPfLine, Kind, NestedPfLine, PfLine, Structure
from .tools import testing
from .tools2.concat import general as concat
from .tools2.intersect import indexable as intersection
from .tools2.plot import plot_pfstates
from .tools.changefreq import averagable as asfreq_avg
from .tools.changefreq import summable as asfreq_sum

# toolsb
from .tools.index import duration
from .tools.peakfn import PeakFunction, base_duration
from .tools.peakfn import factory as create_peakfn
from .tools.peakfn import offpeak_duration, peak_duration
from .tools.product import germanpower_peakfn
from .tools.unit import Q_, ureg
from .tools.wavg import general as wavg

# from .tools.hedge import hedge


VOLUME = Kind.VOLUME
PRICE = Kind.PRICE
REVENUE = Kind.REVENUE
COMPLETE = Kind.COMPLETE

NESTED = Structure.NESTED
FLAT = Structure.FLAT

extendpandas.apply()
suppresswarnings.apply()


# __all__ = ["tools", "dev", "PfLine", "PfState"]
