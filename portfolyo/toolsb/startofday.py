"""Tools for start-of-day."""

import datetime as dt

# Developer notes:
# Because of the existence of half-hour timezones, a situation can occur where the start-of-day is not
# at the full hour. E.g., when viewing hourly data from the Asia/Kolkata timezone which has been localized to another timezone,
# the timestamps will be off by x+0.5 hours. This data is valid. However, we do not consider this case for portfolyo lines.
# The data must first be localized to the correct timezone before turning it into a portfolyo line.

# Assumptions:
# . Times are not checked at a the below-second resolution.


def assert_valid(startofday: dt.time) -> None:
    """Validate if the given start-of-day has the necessary properties to be used in portfolio lines.
    Time must be 'at the hour'.

    Parameters
    ----------
    startofday
        Time to be checked.

    Raises
    ------
    AssertionError
        If the time is not valid.
    """
    if not startofday.minute == startofday.second == 0:
        raise AssertionError(
            "Start-of-day must be at a full hour (not necessarily midnight)."
        )


def to_string(startofday: dt.time) -> str:
    """Turn time into HH:MM:SS string."""
    return f"{startofday:%H:%M:%S}"  # min and second should be 0


def to_tdelta(startofday: dt.time) -> dt.timedelta:
    """Turn time into timedelta to previous midnight (24h day)."""
    return dt.timedelta(
        hours=startofday.hour, minutes=startofday.minute, seconds=startofday.second
    )
