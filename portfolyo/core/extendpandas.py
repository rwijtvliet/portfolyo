"""
Extend pandas classes; add new attributes.
"""

import pandas as pd

from .. import tools


def apply():
    pd.Series.wavg = tools.wavg.series
    pd.DataFrame.wavg = tools.wavg.dataframe
    pd.DatetimeIndex.duration = property(tools.index.duration)
    pd.DatetimeIndex.to_right = property(tools.index.to_right)
    pd.Timestamp.duration = property(tools.stamp.duration)
    pd.Timestamp.right = property(tools.stamp.to_right)
