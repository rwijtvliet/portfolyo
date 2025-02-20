"""Tools for start-of-day."""

import datetime as dt

from . import _decorator as tools_decorator

# Developer notes:
# Because of the existence of half-hour timezones, a situation can occur where the start-of-day is not
# at the full hour. E.g., when viewing hourly data from the Asia/Kolkata timezone which has been localized to another timezone,
# the timestamps will be off by x+0.5 hours. This data is valid. However, we do not consider this case for portfolyo lines.
# The data must first be localized to the correct timezone before turning it into a portfolyo line.

# Assumptions:
# . Times are not checked at a the below-second resolution.


def _from_str(timestr: str) -> dt.time:
    """Turn string into time."""
    for timefmt in ("%H:%M", "%H:%M:%S", "%H%M"):
        try:
            return dt.datetime.strptime(timestr, timefmt).time()
        except ValueError:
            continue
    raise ValueError(f"Could not interpret string '{timestr}' as valid time.")


def _from_tdelta(tdelta: dt.timedelta) -> dt.time:
    """Turn timedelta into time."""
    second = int(tdelta.total_seconds())
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return dt.time(hour, minute, second)


def validate(startofday: dt.time | None) -> dt.time:
    """Validate if the given start-of-day has the necessary properties to be used in portfolio lines.

    Parameters
    ----------
    startofday
        Time to be checked. Time must be 'at the hour'. If None: use midnight.

    Returns
    -------
        Same time.
    """
    # Convert to correct type.
    if startofday is None:
        startofday = dt.time(0, 0)  # midnight
    elif isinstance(startofday, str):
        startofday = _from_str(startofday)
    elif isinstance(startofday, dt.timedelta):
        startofday = _from_tdelta(startofday)

    # Validate.
    if not startofday.minute == startofday.second == 0:
        raise ValueError(
            "Start-of-day must be at a full hour (not necessarily midnight)."
        )

    return startofday


check = tools_decorator.apply_validation(validate, "startofday")


@check()
def to_string(startofday: dt.time) -> str:
    """Turn time into HH:MM:SS string."""
    return f"{startofday:%H:%M:%S}"  # min and second should be 0


@check()
def to_tdelta(startofday: dt.time) -> dt.timedelta:
    """Turn time into timedelta to previous midnight (24h day)."""
    return dt.timedelta(
        hours=startofday.hour, minutes=startofday.minute, seconds=startofday.second
    )
