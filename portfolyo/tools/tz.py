"""Module to work with timezones; esp. to convert to common class (=zoneinfo.ZoneInfo)."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    import pytz

# Conversion and validation.
# --------------------------

# (No validation function, because only thing to check would be if it's None or a ZoneInfo object.)


def coerce(tz: str | dt.tzinfo | pytz.BaseTzInfo | ZoneInfo | None) -> ZoneInfo | None:
    """Convert ``tz`` into ZoneInfo object; raise Error if unsuccessful."""
    if tz is None:
        return None

    # Already a ZoneInfo.
    if isinstance(tz, ZoneInfo):
        return tz

    # Pytz timezone.
    try:
        import pytz  # import here and check for import error, so it's not a required dependency

        if isinstance(tz, pytz.BaseTzInfo):
            return ZoneInfo(tz.zone)
    except ImportError:
        pass

    # String.
    if isinstance(tz, str):
        return ZoneInfo(tz)

    # Something else, like dt.tzinfo (e.g. datetime.timezone, dateutil.tz) which might be used by pandas.
    return ZoneInfo(str(tz))  # convert to string and try


# --------------------------


def is_equal(
    tz1: str | dt.tzinfo | pytz.BaseTzInfo | ZoneInfo | None,
    tz2: str | dt.tzinfo | pytz.BaseTzInfo | ZoneInfo | None,
) -> bool:
    """Compare two timezones for equality."""
    return coerce(tz1) == coerce(tz2)
