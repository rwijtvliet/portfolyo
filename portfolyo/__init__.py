"""Package to analyse and manipulate timeseries related to power and gas offtake portfolios."""

from . import dev, tools
from .core import extendpandas  # extend functionalty of pandas
from .core import suppresswarnings
from .core.pfline import Kind, PfLine, Structure, create
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
from .tools.standardize import frame as standardize
from .tools.tzone import force_agnostic, force_aware
from .tools.unit import Q_, Unit, ureg
from .tools.wavg import general as wavg

import tomli
from pathlib import Path

VOLUME = Kind.VOLUME
PRICE = Kind.PRICE
REVENUE = Kind.REVENUE
COMPLETE = Kind.COMPLETE

extendpandas.apply()
suppresswarnings.apply()


# def get_version():
#     # Find the pyproject.toml file relative to this file
#     pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

#     # Open and read the pyproject.toml file using tomli
#     with pyproject_path.open("rb") as f:
#         pyproject_data = tomli.load(f)
#         return pyproject_data["tool"]["poetry"]["version"]


def get_version():
    # Find the __version__.py file in the same directory as this script
    version_file = Path(__file__).parent / "__version__.py"

    # Read the file and extract the version number
    version_globals = {}
    with version_file.open("r", encoding="utf-8") as f:
        exec(f.read(), version_globals)  # Execute the file to get __version__

    return version_globals["__version__"]  # Return the extracted version


__version__ = get_version()
__all__ = ["tools", "dev", "PfLine", "PfState"]
