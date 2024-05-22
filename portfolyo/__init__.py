"""Package to analyse and manipulate timeseries related to power and gas offtake portfolios."""

from . import _version, dev, tools
from .core import extendpandas  # extend functionalty of pandas
from .core import suppresswarnings
from .core.pfline import Kind, PfLine, Structure, create
from .core.pfstate import PfState
from .tools import testing
from .tools2.changeyear import map_to_year
from .tools2.concat import general as concat
from .tools2.intersect import indexable as intersection
from .tools2.plot import plot_pfstates
from .tools.changefreq import averagable as asfreq_avg
from .tools.changefreq import summable as asfreq_sum
from .tools.changeyear import characterize_index, map_frame_to_year
from .tools.freq import FREQUENCIES
from .tools.hedge import hedge
from .tools.peakfn import PeakFunction
from .tools.peakfn import factory as create_peakfn
from .tools.product import germanpower_peakfn, is_peak_hour
from .tools.standardize import frame as standardize
from .tools.tzone import force_agnostic, force_aware
from .tools.unit import Q_, Unit, ureg
from .tools.wavg import general as wavg

VOLUME = Kind.VOLUME
PRICE = Kind.PRICE
REVENUE = Kind.REVENUE
COMPLETE = Kind.COMPLETE

extendpandas.apply()
suppresswarnings.apply()

__version__ = _version.get_versions()["version"]
__all__ = ["tools", "dev", "PfLine", "PfState"]
