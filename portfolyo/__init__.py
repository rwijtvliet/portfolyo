"""Package to analyse and manipulate timeseries related to power and gas offtake portfolios."""

from . import _version, dev, testing, tools
from .core import extendpandas  # extend functionalty of pandas
from .core import suppresswarnings
from .utilities.plot import plot_pfstates
from .core.pfline import Kind, PfLine, Structure, create
from .core.pfstate import PfState
from .tools2.concat import general as concat
from .tools2.plot import plot_pfstates
from .prices.hedge import hedge
from .prices.utils import is_peak_hour
from .tools.changefreq import averagable as asfreq_avg
from .tools.changefreq import summable as asfreq_sum
from .tools.changeyear import characterize_index, map_frame_to_year
from .tools.freq import FREQUENCIES
from .tools.standardize import frame as standardize
from .tools.tzone import force_agnostic, force_aware
from .tools.unit import Q_, Unit, ureg
from .tools.wavg import general as wavg
from .tools2.concat import general as concat
from .tools2.intersect import indexable as intersection

# from .core.shared.concat import general as concat

VOLUME = Kind.VOLUME
PRICE = Kind.PRICE
REVENUE = Kind.REVENUE
COMPLETE = Kind.COMPLETE

extendpandas.apply()
suppresswarnings.apply()

__version__ = _version.get_versions()["version"]
__all__ = ["tools", "dev", "testing", "PfLine", "PfState"]
