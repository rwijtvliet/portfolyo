"""Module to work with timestamps."""

import pandas as pd

from . import freq as tools_freq
from .types import Frequencylike


def to_right(stamp: pd.Timestamp, freq: Frequencylike) -> pd.Timestamp:
    f"""Right-bound timestamp belonging to left-bound timestamp.

    Parameters
    ----------
    stamp
        Timestamp for which to calculate the right-bound timestamp.
    freq : {tools_freq.DOCS}
        Frequency to use in determining the right-bound timestamp.

    Returns
    -------
        Corresponding right-bound timestamp.
    """
    return stamp + tools_freq.to_jump(freq)
