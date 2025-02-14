"""Module to preprocess indices, frequencies, timedeltas."""

import pandas as pd
from .. import tools


def timedelta_to_jump(tdelta: pd.Timedelta) -> pd.Timedelta | pd.DateOffset:
    """Guess the jump object from a time delta.

    Parameters
    ----------
    tdelta
        (Typical) time delta between start and end of delivery period.

    Returns
    -------
        Term that can be (repeatedly) added to / subtracted from a left-bound
        timestamp to find the next / previous left-bound timestamps.
    """
    # For shorter-than-daily timedeltas, the frequency is (at longest) hourly.
    # All these frequencies have a fixed length, so we can just keep the timedelta.
    if tdelta < pd.Timedelta(hours=23):
        return tdelta
    elif pd.Timedelta(hours=23) <= tdelta <= pd.Timedelta(hours=25):
        return pd.DateOffset(days=1)
    elif pd.Timedelta(days=27) <= tdelta <= pd.Timedelta(days=32):
        return pd.DateOffset(months=1)
    elif pd.Timedelta(days=89) <= tdelta <= pd.Timedelta(days=93):
        return pd.DateOffset(months=3)
    elif pd.Timedelta(days=364) <= tdelta <= pd.Timedelta(days=367):
        return pd.DateOffset(years=1)
    else:
        raise ValueError(
            f"Parameter ``tdelta`` ({tdelta}) doesn't seem to be fit to any of the allowed frequencies ({tools.freq2.DOCS})."
        )
