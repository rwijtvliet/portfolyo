"""Module to work with timestamps."""

import pandas as pd

from . import freq2 as tools_freq
from .types import Frequencylike


def to_right(ts: pd.Timestamp, freq: Frequencylike) -> pd.Timestamp:
    f"""Right-bound timestamp belonging to left-bound timestamp.

    Parameters
    ----------
    ts
        Timestamp for which to calculate the right-bound timestamp.
    freq : {tools_freq.DOCS}
        Frequency to use in determining the right-bound timestamp.

    Returns
    -------
        Corresponding right-bound timestamp.
    """
    return ts + tools_freq.to_jump(freq)
