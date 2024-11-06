"""Module to create a timestamp from any input."""

import datetime as dt
from typing import Any

import pandas as pd


def create(ts: Any, tz: str = None, start_of_day: dt.time = None) -> pd.Timestamp:
    """Creates a timestamp.

    Parameters
    ----------
    ts : Any
        Timestamp or something that can be turned into a timestamp.
    tz : str, optional
        Timezone of the timestamp to create. Not used if ``ts`` is already a valid
        pandas timestamp. I.e., the timezone of ``ts`` takes precedence.
    start_of_day : dt.time, optional
        Time of the timestamp. Not used if ``ts`` is already a valid pandas timestamp
        or contains a time part. I.e., the start_of_day of ``ts`` takes precedence.

    Returns
    -------
    pd.Timestamp
    """
    # Already a timestamp; nothing to do.
    if isinstance(ts, pd.Timestamp):
        return ts

    # Try to create a timestamp with the correct timezone.
    ts = pd.Timestamp(ts, tz=tz)
    if ts is pd.NaT:
        return ts

    # See if we need to change the time.
    if ts.time() != dt.time(hour=0, minute=0, second=0) or start_of_day is None:
        return ts

    ts = ts.replace(hour=start_of_day.hour, minute=start_of_day.minute)
    return ts
