"""
Extend pandas classes; add new attributes.
"""

import pandas as pd

from ..tools import duration, right, wavg


def apply():
    pd.Series.wavg = wavg.series
    pd.DataFrame.wavg = wavg.dataframe
    pd.DatetimeIndex.duration = property(duration.index)
    pd.DatetimeIndex.right = property(right.index)
    pd.Timestamp.duration = property(duration.stamp)
    pd.Timestamp.right = property(right.stamp)


# from .. import tools
# def apply():
#     pd.Series.wavg = tools.wavg.series
#     pd.DataFrame.wavg = tools.wavg.dataframe
#     pd.DatetimeIndex.duration = property(tools.duration.index)
#     pd.DatetimeIndex.right = property(tools.right.index)
#     pd.Timestamp.duration = property(tools.duration.stamp)
#     pd.Timestamp.right = property(tools.right.stamp)
