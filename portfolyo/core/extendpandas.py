"""
Extend pandas classes; add new attributes.
"""

import pandas as pd

from .. import tools


def apply():
    pd.Series.wavg = tools.wavg.series
    pd.DataFrame.wavg = tools.wavg.dataframe
    pd.DatetimeIndex.duration = property(tools.duration.index)
    pd.DatetimeIndex.right = property(tools.right.index)
    pd.Timestamp.duration = property(tools.duration.stamp)
    pd.Timestamp.right = property(tools.right.stamp)
