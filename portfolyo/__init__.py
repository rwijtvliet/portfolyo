"""Package to analyse and manipulate timeseries related to power and gas offtake portfolios."""


from . import _version, dev, tools
from .core import extendpandas  # extend functionalty of pandas
from .core import suppresswarnings
from .core.mixins.plot import plot_pfstates
from .core.pfline import Kind, MultiPfLine, PfLine, SinglePfLine
from .core.pfstate import PfState
from .prices.hedge import hedge
from .prices.utils import is_peak_hour
from .tools.changeyear import characterize_index, map_frame_to_year
from .tools.freq import FREQUENCIES
from .tools.standardize import frame as standardize
from .tools.tzone import force_agnostic, force_aware
from .tools.unit import Q_

extendpandas.apply()
suppresswarnings.apply()

__version__ = _version.get_versions()["version"]
__all__ = ["tools", "dev", "PfLine", "PfState"]
