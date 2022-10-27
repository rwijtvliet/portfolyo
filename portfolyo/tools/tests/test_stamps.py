from portfolyo.tools import stamps, nits
from portfolyo import testing
import pandas as pd
import numpy as np
import pytest

freqs_small_to_large = ["T", "5T", "15T", "30T", "H", "2H", "D", "MS", "QS", "AS"]


# tzBerlin = "Europe/Berlin"
# tzKolkata = "Asia/Kolkata"

# # Quarterhourly.
# # . Normal period.
# qh1_Berlin_l = pd.date_range("2020-03-01", periods=192, freq="15T")
# qh1_Berlin = pd.date_range("2020-03-01", periods=192, freq="15T", tz=tzBerlin)
# qh1_Kolkata_l = pd.date_range("2020-03-01 04:30", periods=192, freq="15T")
# qh1_Kolkata = pd.date_range("2020-03-01 04:30", periods=192, freq="15T", tz=tzKolkata)
# qh1_Kolkata_fl = pd.date_range("2020-03-01", periods=192, freq="15T", tz=tzKolkata)
# # . Start of DST.
# qh2_Berlin_l = pd.Index(
#     [
#         i
#         for i in pd.date_range("2020-03-28", periods=192, freq="15T")
#         if not (i.hour == 2 and i.day == 29)
#     ]
# )
# qh2_Berlin = pd.date_range("2020-03-28", periods=188, freq="15T", tz=tzBerlin)
# qh2_Kolkata_l = pd.date_range("2020-03-28 04:30", periods=188, freq="15T")
# qh2_Kolkata = pd.date_range("2020-03-28 04:30", periods=188, freq="15T", tz=tzKolkata)
# qh2_Kolkata_fl = pd.date_range("2020-03-28", periods=188, freq="15T", tz=tzKolkata)
# # . End of DST.
# qh3_Berlin_l = pd.Index(
#     [
#         *pd.date_range("2020-10-24", periods=108, freq="15T"),
#         *pd.date_range("2020-10-25 02:00", periods=88, freq="15T"),
#     ]
# )  # repeated values on [02:00, 03:00)
# qh3_Berlin = pd.date_range("2020-10-24", periods=196, freq="15T", tz=tzBerlin)
# qh3_Kolkata_l = pd.date_range("2020-10-24 03:30", periods=196, freq="15T")
# qh3_Kolkata = pd.date_range("2020-10-24 03:30", periods=196, freq="15T", tz=tzKolkata)
# qh3_Kolkata_fl = pd.date_range(
#     "2020-10-23 23:00", periods=188, freq="15T", tz=tzKolkata
# )

# # Hourly.
# # . Normal period.
# h1_Berlin_l = pd.date_range("2020-03-01", periods=48, freq="H")
# h1_Berlin = pd.date_range("2020-03-01", periods=48, freq="H", tz=tzBerlin)
# h1_Kolkata_l = pd.date_range("2020-03-01 04:30", periods=48, freq="H")
# h1_Kolkata = pd.date_range("2020-03-01 04:30", periods=48, freq="H", tz=tzKolkata)
# h1_Kolkata_fl = pd.date_range("2020-03-01", periods=48, freq="H", tz=tzKolkata)
# # . Start of DST.
# h2_Berlin_l = pd.Index(
#     [
#         i
#         for i in pd.date_range("2020-03-28", periods=48, freq="H")
#         if not (i.hour == 2 and i.day == 29)
#     ]
# )
# h2_Berlin = pd.date_range("2020-03-28", periods=47, freq="H", tz=tzBerlin)
# h2_Kolkata_l = pd.date_range("2020-03-28 04:30", periods=47, freq="H")
# h2_Kolkata = pd.date_range("2020-03-28 04:30", periods=47, freq="H", tz=tzKolkata)
# h2_Kolkata_fl = pd.date_range("2020-03-28 01:00", periods=47, freq="H", tz=tzKolkata)
# # . End of DST.
# h3_Berlin_l = pd.Index(
#     [
#         *pd.date_range("2020-10-24", periods=27, freq="H"),
#         *pd.date_range("2020-10-25 02:00", periods=22, freq="H"),
#     ]
# )  # repeated values on [02:00, 03:00)
# h3_Berlin = pd.date_range("2020-10-24", periods=49, freq="H", tz=tzBerlin)
# h3_Kolkata_l = pd.date_range("2020-10-24 03:30", periods=49, freq="H")
# h3_Kolkata = pd.date_range("2020-10-24 03:30", periods=49, freq="H", tz=tzKolkata)
# h3_Kolkata_fl = pd.date_range("2020-10-23 23:00", periods=49, freq="H", tz=tzKolkata)


# # Daily.
# # . Normal period.
# d1_Berlin_l = pd.date_range("2020-03-01", periods=15, freq="D")
# d1_Berlin = pd.date_range("2020-03-01", periods=15, freq="D", tz=tzBerlin)
# d1_Kolkata_l = pd.date_range("2020-03-01 04:30", periods=15, freq="D")
# d1_Kolkata = pd.date_range("2020-03-01 04:30", periods=15, freq="D", tz=tzKolkata)
# d1_Kolkata_fl = pd.date_range("2020-03-01", periods=15, freq="D", tz=tzKolkata)
# # . Start of DST.
# d2_Berlin_l = pd.date_range("2020-03-20", periods=15, freq="D")
# d2_Berlin = pd.date_range("2020-03-20", periods=15, freq="D", tz=tzBerlin)
# d2_Kolkata_fl = pd.date_range("2020-03-20", periods=15, freq="D", tz=tzKolkata)
# # . End of DST.
# d3_Berlin_l = pd.date_range("2020-10-20", periods=15, freq="D")
# d3_Berlin = pd.date_range("2020-10-20", periods=15, freq="D", tz=tzBerlin)
# d3_Kolkata_fl = pd.date_range("2020-10-20", periods=15, freq="D", tz=tzKolkata)


# @pytest.mark.parametrize(
#     ("i", "tz", "tz_in", "expected"),
#     [
#         # Quarterhourly values.
#         # . Normal period.
#         # .. tz-aware input.
#         (qh1_Berlin, None, None, qh1_Berlin_l),
#         (qh1_Berlin, tzBerlin, None, qh1_Berlin),
#         (qh1_Berlin, tzKolkata, None, qh1_Kolkata),
#         (qh1_Kolkata, None, None, qh1_Kolkata_l),
#         (qh1_Kolkata, tzKolkata, None, qh1_Kolkata),
#         (qh1_Kolkata, tzBerlin, None, qh1_Berlin),
#         # .. Tz-naive input.
#         (qh1_Berlin_l, None, None, qh1_Berlin_l),
#         (qh1_Berlin_l, None, tzKolkata, qh1_Berlin_l),
#         (qh1_Berlin_l, None, tzBerlin, qh1_Berlin_l),
#         (qh1_Berlin_l, tzBerlin, None, qh1_Berlin),
#         (qh1_Berlin_l, tzKolkata, tzBerlin, qh1_Kolkata),
#         (qh1_Berlin_l, tzKolkata, None, qh1_Kolkata_fl),
#         (qh1_Berlin_l, tzKolkata, tzKolkata, qh1_Kolkata_fl),
#         (qh1_Kolkata_l, None, None, qh1_Kolkata_l),
#         (qh1_Kolkata_l, tzKolkata, None, qh1_Kolkata),
#         (qh1_Kolkata_l, tzBerlin, tzKolkata, qh1_Berlin),
#         # .. Test if tz_in is ignored for tz-aware input.
#         (qh1_Kolkata, None, tzBerlin, qh1_Kolkata_l),
#         (qh1_Kolkata, tzKolkata, tzBerlin, qh1_Kolkata),
#         (qh1_Kolkata, tzBerlin, tzBerlin, qh1_Berlin),
#         # . Start of DST.
#         # .. tz-aware input.
#         (qh2_Berlin, None, None, qh2_Berlin_l),
#         (qh2_Berlin, tzBerlin, None, qh2_Berlin),
#         (qh2_Berlin, tzKolkata, None, qh2_Kolkata),
#         (qh2_Kolkata, None, None, qh2_Kolkata_l),
#         (qh2_Kolkata, tzKolkata, None, qh2_Kolkata),
#         (qh2_Kolkata, tzBerlin, None, qh2_Berlin),
#         # .. Tz-naive input.
#         (qh2_Berlin_l, None, None, qh2_Berlin_l),
#         (qh2_Berlin_l, None, tzBerlin, qh2_Berlin_l),
#         (qh2_Berlin_l, None, tzKolkata, ValueError),  # *3
#         (qh2_Berlin_l, tzBerlin, None, qh2_Berlin),
#         (qh2_Berlin_l, tzKolkata, tzBerlin, ValueError),  # *4
#         (qh2_Kolkata_l, None, None, qh2_Kolkata_l),
#         (qh2_Kolkata_l, tzKolkata, None, qh2_Kolkata),
#         (qh2_Kolkata_l, tzBerlin, tzKolkata, ValueError),  # *4
#         # . End of DST.
#         # .. tz-aware input.
#         (qh3_Berlin, None, None, qh3_Berlin_l),
#         (qh3_Berlin, tzBerlin, None, qh3_Berlin),
#         (qh3_Berlin, tzKolkata, None, qh3_Kolkata),
#         (qh3_Kolkata, None, None, qh3_Kolkata_l),
#         (qh3_Kolkata, tzKolkata, None, qh3_Kolkata),
#         (qh3_Kolkata, tzBerlin, None, qh3_Berlin),
#         # .. Tz-naive input.
#         (qh3_Berlin_l, None, None, qh3_Berlin_l),
#         (qh3_Berlin_l, None, tzBerlin, qh3_Berlin_l),
#         (qh3_Berlin_l, None, tzKolkata, ValueError),  # *3
#         (qh3_Berlin_l, tzBerlin, None, qh3_Berlin),
#         (qh3_Berlin_l, tzKolkata, None, ValueError),  # *3
#         (qh3_Kolkata_l, None, None, qh3_Kolkata_l),
#         (qh3_Kolkata_l, tzKolkata, None, qh3_Kolkata),
#         (qh3_Kolkata_l, tzBerlin, tzKolkata, ValueError),  # *4
#         # Hourly values.
#         # . Normal period.
#         # .. tz-aware input.
#         (h1_Berlin, None, None, h1_Berlin_l),
#         (h1_Berlin, tzBerlin, None, h1_Berlin),
#         (h1_Berlin, tzKolkata, None, ValueError),  # *2
#         (h1_Kolkata, None, None, ValueError),  # *2
#         (h1_Kolkata, tzKolkata, None, ValueError),  # *2
#         (h1_Kolkata, tzBerlin, None, h1_Berlin),  # *1
#         # .. Tz-naive input.
#         (h1_Berlin_l, None, None, h1_Berlin_l),
#         (h1_Berlin_l, None, tzKolkata, h1_Berlin_l),
#         (h1_Berlin_l, tzBerlin, None, h1_Berlin),
#         (h1_Berlin_l, tzKolkata, tzBerlin, ValueError),  # *2
#         (h1_Berlin_l, tzBerlin, tzKolkata, ValueError),  # *2
#         (h1_Berlin_l, tzKolkata, None, h1_Kolkata_fl),
#         (h1_Berlin_l, tzKolkata, tzKolkata, h1_Kolkata_fl),
#         (h1_Kolkata_l, None, None, ValueError),  # *1*2
#         (h1_Kolkata_l, tzKolkata, None, ValueError),  # *1*2
#         (h1_Kolkata_l, tzBerlin, tzKolkata, h1_Berlin),  # *1
#         # .. Test if tz_in is ignored for localized input frames.
#         (h1_Kolkata, None, tzBerlin, ValueError),  # *1*2
#         (h1_Kolkata, tzKolkata, tzBerlin, ValueError),  # *1*2
#         (h1_Kolkata, tzBerlin, tzBerlin, h1_Berlin),  # *1
#         # . Start of DST.
#         # .. tz-aware input.
#         (h2_Berlin, None, None, h2_Berlin_l),
#         (h2_Berlin, tzBerlin, None, h2_Berlin),
#         (h2_Berlin, tzKolkata, None, ValueError),  # *2
#         (h2_Kolkata, None, None, ValueError),  # *1*2
#         (h2_Kolkata, tzKolkata, None, ValueError),  # *1*2
#         (h2_Kolkata, tzBerlin, None, h2_Berlin),  # *1
#         # .. Tz-naive input.
#         (h2_Berlin_l, None, None, h2_Berlin_l),
#         (h2_Berlin_l, None, tzBerlin, h2_Berlin_l),
#         (h2_Berlin_l, tzBerlin, None, h2_Berlin),
#         (h2_Berlin_l, tzKolkata, tzBerlin, h2_Kolkata),
#         (h2_Kolkata_l, None, None, ValueError),  # *1*2
#         (h2_Kolkata_l, tzKolkata, None, ValueError),  # *1*2
#         (h2_Kolkata_l, tzBerlin, tzKolkata, h2_Berlin),  # *1
#         # . End of DST.
#         # .. tz-aware input.
#         (h3_Berlin, None, None, h3_Berlin_l),
#         (h3_Berlin, tzBerlin, None, h3_Berlin),
#         (h3_Berlin, tzKolkata, None, ValueError),  # *2
#         (h3_Kolkata, None, None, ValueError),  # *1*2
#         (h3_Kolkata, tzKolkata, None, ValueError),  # *1*2
#         (h3_Kolkata, tzBerlin, None, h3_Berlin),  # *1
#         # .. Tz-naive input.
#         (h3_Berlin_l, None, None, h3_Berlin_l),
#         (h3_Berlin_l, tzBerlin, None, h3_Berlin),
#         (h3_Kolkata_l, None, None, ValueError),  # *1*2
#         (h3_Kolkata_l, tzKolkata, None, ValueError),  # *1*2
#         (h3_Kolkata_l, tzBerlin, tzKolkata, h3_Berlin),  # *1
#         # Daily values.
#         # . Normal period.
#         # .. tz-aware input.
#         (d1_Berlin, None, None, d1_Berlin_l),
#         (d1_Berlin, tzBerlin, None, d1_Berlin),
#         (d1_Berlin, tzKolkata, None, ValueError),  # *2
#         (d1_Kolkata, None, None, d1_Kolkata_l),  # *1
#         (d1_Kolkata, tzKolkata, None, ValueError),  # *1*2
#         (d1_Kolkata, tzBerlin, None, d1_Berlin),  # *1
#         # .. Tz-naive input.
#         (d1_Berlin_l, None, None, d1_Berlin_l),
#         (d1_Berlin_l, None, tzKolkata, d1_Berlin_l),
#         (d1_Berlin_l, tzBerlin, None, d1_Berlin),
#         (d1_Berlin_l, tzKolkata, tzBerlin, ValueError),  # *2
#         (d1_Berlin_l, tzBerlin, tzKolkata, ValueError),  # *2
#         (d1_Berlin_l, tzKolkata, None, d1_Kolkata_fl),
#         (d1_Berlin_l, tzKolkata, tzKolkata, d1_Kolkata_fl),
#         (d1_Kolkata_l, None, None, ValueError),  # *1*2
#         (d1_Kolkata_l, tzKolkata, None, ValueError),  # *1*2
#         (d1_Kolkata_l, tzBerlin, tzKolkata, d1_Berlin),  # *1
#         # . Start of DST.
#         # .. tz-aware input.
#         (d2_Berlin, None, None, d2_Berlin_l),
#         (d2_Berlin, tzBerlin, None, d2_Berlin),
#         (d2_Berlin, tzKolkata, None, ValueError),  # *2
#         # .. Tz-naive input.
#         (d2_Berlin_l, None, None, d2_Berlin_l),
#         (d2_Berlin_l, None, tzBerlin, d2_Berlin_l),
#         (d2_Berlin_l, tzBerlin, None, d2_Berlin),
#         # . End of DST.
#         # .. tz-aware input.
#         (d3_Berlin, None, None, d3_Berlin_l),
#         (d3_Berlin, tzBerlin, None, d3_Berlin),
#         (d3_Berlin, tzKolkata, None, ValueError),  # *2
#         # .. Tz-naive input.
#         (d3_Berlin_l, None, None, d3_Berlin_l),
#         (d3_Berlin_l, tzBerlin, None, d3_Berlin),
#     ],
# )
# # *1: Index specifies non-boundary timestamps (e.g., hours that start/end at 0:30).
# # *2: Periods in the index fall on non-boundary timestamps in the target timezone.
# # *3: Index specifies local timestamps that do not exist (or should doubly exist) in source timezone.
# # *4: Index cannot be converted because source and target timezone don't
# def test_settimezone(i, tz, tz_in, expected):
#     """Test if timezones can be correctly set, and an error is raised if that's expected."""
#     if isinstance(expected, type) and issubclass(expected, Exception):
#         with pytest.raises(expected):
#             _ = stamps.set_timezone(i, tz, tz_in)
#     else:
#         result = stamps.set_timezone(i, tz, tz_in)
#         testing.assert_index_equal(result, expected)


# @pytest.mark.parametrize(
#     ("i", "tz", "floating", "expected"),
#     [
#         # . Quarterhourly.
#         # . . No conversion: should give back original.
#         (qh1_Berlin, tzBerlin, None, qh1_Berlin),
#         (qh1_Kolkata, tzKolkata, None, qh1_Kolkata),
#         # . . Conversion should be done correctly.
#         (qh1_Berlin, tzKolkata, None, qh1_Kolkata),
#         (qh1_Berlin, tzKolkata, False, qh1_Kolkata),
#         (qh1_Kolkata, tzBerlin, None, qh1_Berlin),
#         (qh1_Kolkata, tzBerlin, False, qh1_Berlin),
#         # . . Conversion while keeping local time.
#         (qh1_Kolkata_fl, tzBerlin, True, qh1_Berlin),
#         # . . Should discard floating argument if input tz-aware and tz same.
#         (qh1_Berlin, tzBerlin, False, qh1_Berlin),
#         (qh1_Berlin, tzBerlin, True, qh1_Berlin),
#         # . Hourly.
#         # . . No conversion: should give back original.
#         (h1_Berlin, tzBerlin, None, h1_Berlin),
#         (h1_Kolkata, tzKolkata, None, h1_Kolkata),
#         # . . Conversion should be done correctly.
#         (h1_Berlin, tzKolkata, None, h1_Kolkata),
#         (h1_Berlin, tzKolkata, False, h1_Kolkata),
#         (h1_Kolkata, tzBerlin, None, h1_Berlin),
#         (h1_Kolkata, tzBerlin, False, h1_Berlin),
#         # . . Conversion while keeping local time.
#         (h1_Kolkata_fl, tzBerlin, True, h1_Berlin),
#         # . . Should discard floating argument if input tz-aware and tz same.
#         (h1_Berlin, tzBerlin, False, h1_Berlin),
#         (h1_Berlin, tzBerlin, True, h1_Berlin),
#     ],
# )
# def test_forcetzaware_handshorter_tzawareinput_normalperiod(*args, **kwargs):
#     test_forcetzaware(*args, **kwargs)


# def test_forcetzaware(i, tz, floating, expected):
#     """Test if timezones can be correctly set, and an error is raised if that's expected."""
#     if isinstance(expected, type) and issubclass(expected, Exception):
#         with pytest.raises(expected):
#             _ = stamps.force_tzaware(i, tz, floating)
#     else:
#         result = stamps.force_tzaware(i, tz, floating)
#         testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2020", "2021", "AS"),
        ("2020", "2020-04", "QS"),
        ("2020", "2020-02", "MS"),
        ("2020", "2020-01-02", "D"),
        ("2020", "2020-01-01 01:00", "H"),
        ("2020", "2020-01-01 00:15", "15T"),
        ("2020-03-29", "2020-03-30", "D"),
        ("2020-03-01", "2020-04-01", "MS"),
        ("2020-10-25", "2020-10-26", "D"),
        ("2020-10-01", "2020-11-01", "MS"),
    ],
)
def test_guessfrequency(start, end, expected, tz):
    """Test if correct frequency is guessed from start and end timestamp."""
    tdelta = pd.Timestamp(end, tz=tz) - pd.Timestamp(start, tz=tz)
    if expected is None:
        with pytest.raises(ValueError):
            _ = stamps.guess_frequency(tdelta)
    else:
        result = stamps.guess_frequency(tdelta)
        assert result == expected


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        # Hourly.
        ("2020-03-29 01:00", "2020-03-29 03:00", "H"),
        ("2020-10-25 01:00", "2020-10-25 02:00+0200", "H"),
        ("2020-10-25 01:00", "2020-10-25 02:00+0100", None),
        ("2020-10-25 02:00+0200", "2020-10-25 02:00+0100", "H"),
        ("2020-10-25 02:00+0200", "2020-10-25 03:00", None),
        ("2020-10-25 02:00+0100", "2020-10-25 03:00", "H"),
        # Quarterhourly.
        ("2020-03-29 01:45", "2020-03-29 03:00", "15T"),
        ("2020-10-25 01:45", "2020-10-25 02:00+0200", "15T"),
        ("2020-10-25 01:45", "2020-10-25 02:00+0100", None),
        ("2020-10-25 02:45+0200", "2020-10-25 02:00+0100", "15T"),
        ("2020-10-25 02:45+0200", "2020-10-25 03:00", None),
        ("2020-10-25 02:45+0100", "2020-10-25 03:00", "15T"),
    ],
)
def test_guessfrequency_dst(start, end, expected):
    """Test if correct frequency is guessed from start and end timestamp."""
    tz = "Europe/Berlin"
    tdelta = pd.Timestamp(end, tz=tz) - pd.Timestamp(start, tz=tz)
    if expected is None:
        with pytest.raises(ValueError):
            _ = stamps.guess_frequency(tdelta)
    else:
        result = stamps.guess_frequency(tdelta)
        assert result == expected


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("remove_freq", [True, False])
@pytest.mark.parametrize(
    ("start", "periods", "freq", "expected_start"),
    [
        # Natural-boundary.
        # Natural-boundary timestamps; without DST-transition.
        ("2020-02-01 1:00", 24, "H", "2020-02-01"),
        ("2020-02-01 0:15", 96, "15T", "2020-02-01"),
        ("2020-02-02", 28, "D", "2020-02-01"),
        # Natural-boundary timestamps; with DST-start.
        ("2020-03-29 1:00", 24, "H", "2020-03-29"),
        ("2020-03-29 4:00", 24, "H", "2020-03-29 3:00"),
        ("2020-03-29 0:15", 96, "15T", "2020-03-29"),
        ("2020-03-29 0:30", 96, "15T", "2020-03-29 0:15"),
        ("2020-03-29 1:30", 96, "15T", "2020-03-29 1:15"),
        ("2020-03-29 3:30", 96, "15T", "2020-03-29 3:15"),
        ("2020-03-29 3:15", 96, "15T", "2020-03-29 3:00"),
        ("2020-03-02", 31, "D", "2020-03-01"),
        # Natural-boundary timestamps; with DST-end.
        ("2020-10-25 1:00", 24, "H", "2020-10-25"),
        ("2020-10-25 4:00", 24, "H", "2020-10-25 3:00"),
        ("2020-10-25 0:15", 96, "15T", "2020-10-25"),
        ("2020-10-25 0:30", 96, "15T", "2020-10-25 0:15"),
        ("2020-10-25 1:30", 96, "15T", "2020-10-25 1:15"),
        ("2020-10-25 3:30", 96, "15T", "2020-10-25 3:15"),
        ("2020-10-25 3:15", 96, "15T", "2020-10-25 3:00"),
        ("2020-10-02", 31, "D", "2020-10-01"),
        # Natural-boundary timestamps; with DST-start and DST-end.
        ("2020-04-01", 12, "MS", "2020-03-01"),
        ("2020-07-01", 12, "MS", "2020-06-01"),
        ("2020-04-01", 4, "QS", "2020-01-01"),
        ("2020-07-01", 4, "QS", "2020-04-01"),
        # Unnatural-boundary.
        # Unnatural-boundary timestamps; without DST-transition.
        ("2020-02-01 01:30", 24, "H", "2020-02-01 00:30"),
        ("2020-02-01 01:32", 96, "15T", "2020-02-01 01:17"),
        ("2020-02-02 06:00", 5, "D", "2020-02-01 06:00"),
        # Unnatural-boundary timestamps; with DST-start.
        ("2020-03-29 1:30", 24, "H", "2020-03-29 0:30"),
        ("2020-03-29 4:30", 24, "H", "2020-03-29 3:30"),
        ("2020-03-29 0:17", 96, "15T", "2020-03-29 0:02"),
        ("2020-03-29 0:32", 96, "15T", "2020-03-29 0:17"),
        ("2020-03-29 1:32", 96, "15T", "2020-03-29 1:17"),
        ("2020-03-29 3:32", 96, "15T", "2020-03-29 3:17"),
        ("2020-03-29 3:17", 96, "15T", "2020-03-29 3:02"),
        ("2020-03-02 06:00", 31, "D", "2020-03-01 06:00"),
        # Natural-boundary timestamps; with DST-end.
        ("2020-10-25 1:30", 24, "H", "2020-10-25 0:30"),
        ("2020-10-25 4:30", 24, "H", "2020-10-25 3:30"),
        ("2020-10-25 0:17", 96, "15T", "2020-10-25 0:02"),
        ("2020-10-25 0:32", 96, "15T", "2020-10-25 0:17"),
        ("2020-10-25 1:32", 96, "15T", "2020-10-25 1:17"),
        ("2020-10-25 3:32", 96, "15T", "2020-10-25 3:17"),
        ("2020-10-25 3:17", 96, "15T", "2020-10-25 3:02"),
        ("2020-10-02 06:00", 31, "D", "2020-10-01 06:00"),
        # Natural-boundary timestamps; with DST-start and DST-end.
        ("2020-04-01 06:00", 12, "MS", "2020-03-01 06:00"),
        ("2020-07-01 06:00", 12, "MS", "2020-06-01 06:00"),
        ("2020-04-01 06:00", 4, "QS", "2020-01-01 06:00"),
        ("2020-07-01 06:00", 4, "QS", "2020-04-01 06:00"),
    ],
)
def test_righttoleft(start, periods, freq, expected_start, tz, remove_freq):
    """Test if index of rightbound timestamps can be make leftbound."""
    i = pd.date_range(start, periods=periods, freq=freq, tz=tz)
    expected = pd.date_range(expected_start, periods=periods, freq=freq, tz=tz)
    if remove_freq:
        i.freq = None
    result = stamps.right_to_left(i)
    testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("remove_freq", [True, False])
@pytest.mark.parametrize(
    ("start", "periods", "freq", "expected_start"),
    [
        # Index with DST-start.
        ("2020-03-29 1:00", 24, "H", "2020-03-29"),
        ("2020-03-29 3:00", 24, "H", "2020-03-29 1:00"),
        ("2020-03-29 4:00", 24, "H", "2020-03-29 3:00"),
        ("2020-03-29 0:15", 96, "15T", "2020-03-29"),
        ("2020-03-29 1:30", 96, "15T", "2020-03-29 1:15"),
        ("2020-03-29 1:45", 96, "15T", "2020-03-29 1:30"),
        ("2020-03-29 3:00", 96, "15T", "2020-03-29 1:45"),
        ("2020-03-29 3:15", 96, "15T", "2020-03-29 3:00"),
        # Index with DST-end.
        ("2020-10-25 1:00", 24, "H", "2020-10-25"),
        ("2020-10-25 2:00+0200", 24, "H", "2020-10-25 1:00"),
        ("2020-10-25 2:00+0100", 24, "H", "2020-10-25 2:00+0200"),
        ("2020-10-25 3:00", 24, "H", "2020-10-25 2:00+0100"),
        ("2020-10-25 4:00", 24, "H", "2020-10-25 3:00"),
        ("2020-10-25 0:15", 96, "15T", "2020-10-25"),
        ("2020-10-25 0:30", 96, "15T", "2020-10-25 0:15"),
        ("2020-10-25 1:30", 96, "15T", "2020-10-25 1:15"),
        ("2020-10-25 2:15+0200", 96, "15T", "2020-10-25 2:00+0200"),
        ("2020-10-25 2:30+0200", 96, "15T", "2020-10-25 2:15+0200"),
        ("2020-10-25 2:00+0100", 96, "15T", "2020-10-25 2:45+0200"),
        ("2020-10-25 2:15+0100", 96, "15T", "2020-10-25 2:00+0100"),
        ("2020-10-25 2:30+0100", 96, "15T", "2020-10-25 2:15+0100"),
        ("2020-10-25 1:30", 96, "15T", "2020-10-25 1:15"),
        ("2020-10-25 3:30", 96, "15T", "2020-10-25 3:15"),
        ("2020-10-25 3:15", 96, "15T", "2020-10-25 3:00"),
    ],
)
def test_makeleftbound_dst_tzaware(start, periods, freq, expected_start, remove_freq):
    """Test if index of rightbound timestamps can be make leftbound, across DST changeover."""
    i = pd.date_range(
        pd.Timestamp(start, tz="Europe/Berlin"), periods=periods, freq=freq
    )
    expected = pd.date_range(
        pd.Timestamp(expected_start, tz="Europe/Berlin"), periods=periods, freq=freq
    )
    if remove_freq:
        i.freq = None
    result = stamps.right_to_left(i)
    testing.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    ("i", "expected"),
    [
        (
            pd.DatetimeIndex(
                [
                    "2020-03-29 01:00:00",
                    "2020-03-29 03:00:00",
                    "2020-03-29 04:00:00",
                    "2020-03-29 05:00:00",
                ],
                dtype="datetime64[ns]",
            ),
            pd.DatetimeIndex(
                [
                    "2020-03-29 00:00:00",
                    "2020-03-29 02:00:00",
                    "2020-03-29 03:00:00",
                    "2020-03-29 04:00:00",
                ],
                dtype="datetime64[ns]",
            ),
        ),
        (
            pd.DatetimeIndex(
                [
                    "2020-10-25 01:00:00",
                    "2020-10-25 02:00:00",
                    "2020-10-25 02:00:00",
                    "2020-10-25 03:00:00",
                ],
                dtype="datetime64[ns]",
            ),
            pd.DatetimeIndex(
                [
                    "2020-10-25 00:00:00",
                    "2020-10-25 01:00:00",
                    "2020-10-25 01:00:00",
                    "2020-10-25 02:00:00",
                ],
                dtype="datetime64[ns]",
            ),
        ),
    ],
)
def test_makeleftbound_dst_tznaive(i, expected):
    result = stamps.right_to_left(i)
    testing.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    ("idxs", "expected"),
    [
        # Days, with and without timezone.
        (
            [
                pd.date_range("2020", freq="D", periods=31),
                pd.date_range("2020-01-20", freq="D", periods=40),
            ],
            pd.date_range("2020-01-20", freq="D", periods=12),
        ),
        (
            [
                pd.date_range("2020", freq="D", periods=31, tz="Europe/Berlin"),
                pd.date_range("2020-01-20", freq="D", periods=40, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-01-20", freq="D", periods=12, tz="Europe/Berlin"),
        ),
        # Error: incompatible timezones.
        (
            [
                pd.date_range("2020", freq="D", periods=31),
                pd.date_range("2020-01-20", freq="D", periods=40, tz="Europe/Berlin"),
            ],
            None,
        ),
        # Error: distinct frequencies.
        (
            [
                pd.date_range("2020", freq="H", periods=31),
                pd.date_range("2020-01-20", freq="D", periods=40),
            ],
            None,
        ),
        # No overlap.
        (
            [
                pd.date_range("2020", freq="H", periods=24),
                pd.date_range("2020-01-20", freq="H", periods=72),
            ],
            pd.date_range("2020", freq="H", periods=0),
        ),
        # Months, with and without timezone.
        (
            [
                pd.date_range("2020", freq="MS", periods=31),
                pd.date_range("2020-05-01", freq="MS", periods=40),
            ],
            pd.date_range("2020-05-01", freq="MS", periods=27),
        ),
        (
            [
                pd.date_range("2020", freq="MS", periods=31, tz="Europe/Berlin"),
                pd.date_range("2020-05-01", freq="MS", periods=40, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-05-01", freq="MS", periods=27, tz="Europe/Berlin"),
        ),
        # Test if names retained.
        (
            [
                pd.date_range("2020", freq="MS", periods=31, name="ts_left"),
                pd.date_range("2020-05-01", freq="MS", periods=40),
            ],
            pd.date_range("2020-05-01", freq="MS", periods=27, name="ts_left"),
        ),
        # DST.
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=71, tz="Europe/Berlin"),
                pd.date_range("2020-03-29", freq="H", periods=71, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-03-29", freq="H", periods=47, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=71, tz="Europe/Berlin"),
                pd.date_range("2020-03-30", freq="H", periods=72, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-03-30", freq="H", periods=24, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=72),
                pd.date_range("2020-03-29", freq="H", periods=72),
            ],
            pd.date_range("2020-03-29", freq="H", periods=48),
        ),
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=72),
                pd.date_range("2020-03-30", freq="H", periods=72),
            ],
            pd.date_range("2020-03-30", freq="H", periods=24),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=73, tz="Europe/Berlin"),
                pd.date_range("2020-10-25", freq="H", periods=73, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-10-25", freq="H", periods=49, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=73, tz="Europe/Berlin"),
                pd.date_range("2020-10-26", freq="H", periods=72, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-10-26", freq="H", periods=24, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=72),
                pd.date_range("2020-10-25", freq="H", periods=72),
            ],
            pd.date_range("2020-10-25", freq="H", periods=48),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=72),
                pd.date_range("2020-10-26", freq="H", periods=72),
            ],
            pd.date_range("2020-10-26", freq="H", periods=24),
        ),
        # Distinct timezones.
        (
            [
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Europe/Berlin"),
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Asia/Kolkata"),
            ],
            pd.date_range("2020-01-01", freq="15T", periods=78, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Asia/Kolkata"),
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Europe/Berlin"),
            ],
            pd.date_range(
                "2020-01-01 04:30", freq="15T", periods=78, tz="Asia/Kolkata"
            ),
        ),
        (
            [
                pd.date_range("2020-01-01", freq="H", periods=24, tz="Asia/Kolkata"),
                pd.date_range("2020-01-01", freq="H", periods=24, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-01-01", freq="H", periods=0, tz="Asia/Kolkata"),
        ),
    ],
)
def test_intersection(idxs, expected):
    """Test if intersection works correctly."""
    if expected is None:
        with pytest.raises(ValueError):
            _ = stamps.intersection(*idxs)
    else:
        result = stamps.intersection(*idxs)
        testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("iterable", [False, True])  # make iterable or not
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("ts", "fut", "freq", "expected_floor", "expected_ceil"),
    [
        ("2020", 0, "D", "2020", None),
        ("2020", 0, "MS", "2020", None),
        ("2020", 0, "QS", "2020", None),
        ("2020", 0, "AS", "2020", None),
        ("2020", 1, "D", "2020-01-02", None),
        ("2020", 1, "MS", "2020-02", None),
        ("2020", 1, "QS", "2020-04", None),
        ("2020", 1, "AS", "2021", None),
        ("2020", -1, "D", "2019-12-31", None),
        ("2020", -1, "MS", "2019-12", None),
        ("2020", -1, "QS", "2019-10", None),
        ("2020", -1, "AS", "2019", None),
        ("2020-02-01", 0, "D", "2020-02-01", None),
        ("2020-02-01", 0, "MS", "2020-02-01", None),
        ("2020-02-01", 0, "QS", "2020-01-01", "2020-04-01"),
        ("2020-02-01", 0, "AS", "2020", "2021"),
        ("2020-01-01 23:55", 0, "D", "2020", "2020-01-02"),
        ("2020-01-24 1:32", 0, "MS", "2020", "2020-02"),
        ("2020-03-03 3:33", 0, "QS", "2020", "2020-04"),
        ("2020-10-11 12:34", 0, "AS", "2020", "2021"),
        ("2020-01-01 23:55", 1, "D", "2020-01-02", "2020-01-03"),
        ("2020-01-24 1:32", 1, "MS", "2020-02", "2020-03"),
        ("2020-03-03 3:33", 1, "QS", "2020-04", "2020-07"),
        ("2020-10-11 12:34", 1, "AS", "2021", "2022"),
        ("2020-01-01 23:55", -1, "D", "2019-12-31", "2020-01-01"),
        ("2020-01-24 1:32", -1, "MS", "2019-12", "2020"),
        ("2020-03-03 3:33", -1, "QS", "2019-10", "2020"),
        ("2020-10-11 12:34", -1, "AS", "2019", "2020"),
        ("2020-03-29 00:00", 0, "H", "2020-03-29 00:00", None),
        ("2020-10-25 00:00", 0, "H", "2020-10-25 00:00", None),
    ],
)
@pytest.mark.parametrize("function", ["floor", "ceil"])
def test_floorceilts_1(
    function: str,
    ts: str,
    fut: int,
    freq: str,
    expected_floor: str,
    expected_ceil: str,
    tz: str,
    iterable: bool,
):
    """Test if timestamp or iterable of timestamps is correctly floored and ceiled."""
    # Prepare.
    fn = stamps.floor_ts if function == "floor" else stamps.ceil_ts
    if function == "floor" or expected_ceil is None:
        expected_single = expected_floor
    else:
        expected_single = expected_ceil

    ts = pd.Timestamp(ts)
    expected_single = pd.Timestamp(expected_single)

    if tz:
        ts = ts.tz_localize(tz)
        expected_single = expected_single.tz_localize(tz)

    # Test.
    if not iterable:
        # Test single value.
        assert fn(ts, freq, fut) == expected_single

    else:
        # Test index.
        periods = 10
        index = pd.date_range(ts, periods=periods, freq=freq)  # causes rounding of ts
        index += ts - index[0]  # undoes the rounding

        result_iter = fn(index, freq, fut)
        result_iter.freq = None  # disregard checking frequencies here
        expected_iter = pd.date_range(expected_single, periods=periods, freq=freq)
        expected_iter.freq = None  # disregard checking frequencies here
        pd.testing.assert_index_equal(result_iter, expected_iter)


@pytest.mark.parametrize(
    ("ts", "tz", "freq", "expected_floor", "expected_ceil"),
    [
        ("2020-04-21 15:25", None, "H", "2020-04-21 15:00", "2020-04-21 16:00"),
        (
            "2020-04-21 15:25",
            "Europe/Berlin",
            "H",
            "2020-04-21 15:00",
            "2020-04-21 16:00",
        ),
        (
            "2020-04-21 15:25+02:00",
            "Europe/Berlin",
            "H",
            "2020-04-21 15:00+02:00",
            "2020-04-21 16:00+02:00",
        ),
        (
            "2020-04-21 15:25",
            "Asia/Kolkata",
            "H",
            "2020-04-21 15:00",
            "2020-04-21 16:00",
        ),
        ("2020-03-29 01:50", None, "15T", "2020-03-29 01:45", "2020-03-29 02:00"),
        ("2020-03-29 03:05", None, "15T", "2020-03-29 03:00", "2020-03-29 03:15"),
        (
            "2020-03-29 01:50+01:00",
            "Europe/Berlin",
            "15T",
            "2020-03-29 01:45+01:00",
            "2020-03-29 03:00+02:00",
        ),
        (
            "2020-03-29 03:05+02:00",
            "Europe/Berlin",
            "15T",
            "2020-03-29 03:00+02:00",
            "2020-03-29 03:15+02:00",
        ),
        (
            "2020-03-29 01:50",
            "Europe/Berlin",
            "15T",
            "2020-03-29 01:45",
            "2020-03-29 03:00",
        ),
        ("2020-03-29 03:05", None, "15T", "2020-03-29 03:00", "2020-03-29 03:15"),
        ("2020-10-25 02:50", None, "15T", "2020-10-25 02:45", "2020-10-25 03:00"),
        ("2020-10-25 02:05", None, "15T", "2020-10-25 02:00", "2020-10-25 02:15"),
        (
            "2020-10-25 02:50+02:00",
            "Europe/Berlin",
            "15T",
            "2020-10-25 02:45+02:00",
            "2020-10-25 02:00+01:00",
        ),
        (
            "2020-10-25 02:05+02:00",
            "Europe/Berlin",
            "15T",
            "2020-10-25 02:00+02:00",
            "2020-10-25 02:15+02:00",
        ),
        (
            "2020-10-25 02:50+01:00",
            "Europe/Berlin",
            "15T",
            "2020-10-25 02:45+01:00",
            "2020-10-25 03:00+01:00",
        ),
        (
            "2020-10-25 02:05+01:00",
            "Europe/Berlin",
            "15T",
            "2020-10-25 02:00+01:00",
            "2020-10-25 02:15+01:00",
        ),
        (
            "2020-10-25 02:30+02:00",
            "Europe/Berlin",
            "H",
            "2020-10-25 02:00+02:00",
            "2020-10-25 02:00+01:00",
        ),
        (
            "2020-10-25 02:30+01:00",
            "Europe/Berlin",
            "H",
            "2020-10-25 02:00+01:00",
            "2020-10-25 03:00+01:00",
        ),
    ],
)
@pytest.mark.parametrize("function", ["floor", "ceil"])
def test_floorceilts_2(function, ts, tz, freq, expected_floor, expected_ceil):
    """Test flooring and ceiling during DST transitions."""
    ts_in = pd.Timestamp(ts, tz=tz)
    if function == "floor":
        result = stamps.floor_ts(ts_in, freq)
        expected = pd.Timestamp(expected_floor, tz=tz)
    else:
        result = stamps.ceil_ts(ts_in, freq)
        expected = pd.Timestamp(expected_ceil, tz=tz)

    assert result == expected


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("ts", "freq", "is_boundary"),
    [
        ("2020", "H", True),
        ("2020", "D", True),
        ("2020", "MS", True),
        ("2020", "AS", True),
        ("2020-04-01", "H", True),
        ("2020-04-01", "D", True),
        ("2020-04-01", "MS", True),
        ("2020-04-01", "AS", False),
        ("2020-01-01 15:00", "H", True),
        ("2020-01-01 15:00", "D", False),
        ("2020-01-01 15:00", "MS", False),
        ("2020-01-01 15:00", "AS", False),
        ("2020-01-01 15:45", "H", False),
        ("2020-01-01 15:45", "D", False),
        ("2020-01-01 15:45", "MS", False),
        ("2020-01-01 15:45", "AS", False),
    ],
)
def test_assertboundary(ts, freq, is_boundary, tz):
    """Test if boundary timestamps are correctly identified."""
    ts = pd.Timestamp(ts, tz=tz)
    if is_boundary:
        _ = stamps.assert_boundary_ts(ts, freq)
    else:
        with pytest.raises(AssertionError):
            _ = stamps.assert_boundary_ts(ts, freq)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("tss", "freq", "are_boundary"),
    [
        (["2020", "2021"], "H", True),
        (["2020", "2021"], "D", True),
        (["2020", "2021"], "MS", True),
        (["2020", "2021"], "AS", True),
        (["2020-04-01", "2021"], "H", True),
        (["2020-04-01", "2021"], "D", True),
        (["2020-04-01", "2021"], "MS", True),
        (["2020-04-01", "2021"], "AS", False),
        (["2020-01-01 15:00", "2021"], "H", True),
        (["2020-01-01 15:00", "2021"], "D", False),
        (["2020-01-01 15:00", "2021"], "MS", False),
        (["2020-01-01 15:00", "2021"], "AS", False),
        (["2020-01-01 15:45", "2021"], "H", False),
        (["2020-01-01 15:45", "2021"], "D", False),
        (["2020-01-01 15:45", "2021"], "MS", False),
        (["2020-01-01 15:45", "2021"], "AS", False),
    ],
)
def test_assertboundary_asindex(tss, freq, are_boundary, tz):
    """Test if boundary timestamps are correctly identified."""
    i = pd.Index([pd.Timestamp(ts, tz=tz) for ts in tss])
    if are_boundary:
        _ = stamps.assert_boundary_ts(i, freq)
    else:
        with pytest.raises(AssertionError):
            _ = stamps.assert_boundary_ts(i, freq)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("start", "end", "freq", "trimfreq", "expected_start", "expected_end"),
    [
        ("2020-01-05", "2020-12-21", "D", "H", "2020-01-05", "2020-12-21"),
        ("2020-01-05", "2020-12-21", "D", "D", "2020-01-05", "2020-12-21"),
        ("2020-01-05", "2020-12-21", "D", "MS", "2020-02", "2020-12"),
        ("2020-01-05", "2020-12-21", "D", "QS", "2020-04", "2020-10"),
        ("2020-01-05", "2020-12-21", "D", "AS", None, None),
    ],
)
def test_trimindex(start, end, freq, trimfreq, expected_start, expected_end, tz):
    """Test if indices are correctly trimmed."""
    i = pd.date_range(start, end, freq=freq, inclusive="left", tz=tz)
    result = stamps.trim_index(i, trimfreq)
    if expected_start is not None:
        expected = pd.date_range(
            expected_start, expected_end, freq=freq, inclusive="left", tz=tz
        )
    else:
        expected = pd.DatetimeIndex([], freq=freq, tz=tz)
    pd.testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq_as_attr", [True, False])
@pytest.mark.parametrize(
    ("ts_left", "freq", "expected_ts_right"),
    [
        ("2020", "15T", "2020-01-01 00:15"),
        ("2020", "H", "2020-01-01 01:00"),
        ("2020", "D", "2020-01-02"),
        ("2020", "MS", "2020-02-01"),
        ("2020", "QS", "2020-04"),
        ("2020", "AS", "2021"),
        ("2020-04-21", "15T", "2020-04-21 00:15"),
        ("2020-04-21", "H", "2020-04-21 01:00"),
        ("2020-04-21", "D", "2020-04-22"),
        ("2020-04-21 15:00", "15T", "2020-04-21 15:15"),
        ("2020-04-21 15:00", "H", "2020-04-21 16:00"),
    ],
)
def test_tsright(ts_left, freq, tz, freq_as_attr, expected_ts_right):
    """Test if right timestamp is correctly calculated."""
    if freq_as_attr:
        ts = pd.Timestamp(ts_left, freq=freq, tz=tz)
        result = stamps.ts_right(ts)
    else:
        ts = pd.Timestamp(ts_left, tz=tz)
        result = stamps.ts_right(ts, freq)
    expected = pd.Timestamp(expected_ts_right, tz=tz)
    assert result == expected


@pytest.mark.parametrize("tz_left", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("tz_right", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("tss", "expected_tss"),
    [
        (("2020-01-01", "2020-02-02"), ("2020-01-01", "2020-02-02")),
        (("2020-02-02", "2020-01-01"), ("2020-01-01", "2020-02-02")),
        (("2020-01-01", None), ("2020-01-01", "2021-01-01")),
        ((None, "2020-02-02"), ("2020-01-01", "2020-02-02")),
        (("2020-01-01", "2020-01-01"), ("2020-01-01", "2020-01-01")),
        (("2020-03-03 3:33", "2021-10-09"), ("2020-03-03 3:33", "2021-10-09")),
        (("2021-10-09", "2020-03-03 3:33"), ("2020-03-03 3:33", "2021-10-09")),
        (("2020-03-03 3:33", "2021-10-09"), ("2020-03-03 3:33", "2021-10-09")),
        (("2020-03-03 3:33", None), ("2020-03-03 3:33", "2021-01-01")),
        ((None, "2021-10-09"), ("2021-01-01 0:00", "2021-10-09")),
        (
            (None, None),
            (
                pd.Timestamp(pd.Timestamp.today().year + 1, 1, 1),
                pd.Timestamp(pd.Timestamp.today().year + 2, 1, 1),
            ),
        ),
    ],
)
def test_tsleftright(tss: tuple, expected_tss: tuple, tz_left: str, tz_right: str):
    """Test if start and end of interval are correctly calculated."""
    tzs = [tz_left, tz_right]
    tss = [pd.Timestamp(ts) for ts in tss]  # turn into timestamps for comparison
    if tss[0] == tss[1] and tz_left != tz_right:
        return  # too complicated to test; would have to recreate function here.
    swap = tss[0] > tss[1]
    tss = [ts.tz_localize(tz) for ts, tz in zip(tss, tzs)]  # add timezone info

    result = stamps.ts_leftright(*tss)

    exp_tzs = [tz for ts, tz in zip(tss, tzs) if tz is not None and ts is not pd.NaT]
    if swap:
        exp_tzs.reverse()
    if not len(exp_tzs):
        exp_tzs = ["Europe/Berlin"]
    if len(exp_tzs) == 1:
        exp_tzs = exp_tzs * 2
    exp_result = [
        pd.Timestamp(ts).tz_localize(tz) for ts, tz in zip(expected_tss, exp_tzs)
    ]

    for a, b in zip(result, exp_result):
        assert a == b


@pytest.mark.parametrize(
    ("ts", "freq", "hours"),
    [
        (pd.Timestamp("2020-01-01"), "D", 24),
        (pd.Timestamp("2020-01-01"), "MS", 744),
        (pd.Timestamp("2020-01-01"), "QS", 2184),
        (pd.Timestamp("2020-01-01"), "AS", 8784),
        (pd.Timestamp("2020-03-01"), "D", 24),
        (pd.Timestamp("2020-03-01"), "MS", 744),
        (pd.Timestamp("2020-03-29"), "D", 24),
        (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "D", 24),
        (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "MS", 744),
        (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "QS", 2183),
        (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "AS", 8784),
        (pd.Timestamp("2020-03-01", tz="Europe/Berlin"), "D", 24),
        (pd.Timestamp("2020-03-01", tz="Europe/Berlin"), "MS", 743),
        (pd.Timestamp("2020-03-29", tz="Europe/Berlin"), "D", 23),
    ],
)
def test_duration(ts, freq, hours):
    """Test if duration in correctly calculated."""
    assert stamps.duration(ts, freq) == nits.Q_(hours, "h")


@pytest.mark.parametrize("freq1", freqs_small_to_large)
@pytest.mark.parametrize("freq2", freqs_small_to_large)
def test_frequpordown(freq1, freq2):
    i1 = freqs_small_to_large.index(freq1)
    i2 = freqs_small_to_large.index(freq2)
    outcome = np.sign(i1 - i2)
    assert stamps.freq_up_or_down(freq1, freq2) == outcome


@pytest.mark.parametrize("count", range(1, 30))
def test_longestshortestfreq(count):
    indices = np.random.randint(0, len(freqs_small_to_large), count)
    freqs = np.array(freqs_small_to_large)[indices]
    assert stamps.freq_longest(*freqs) == freqs_small_to_large[max(indices)]
    assert stamps.freq_shortest(*freqs) == freqs_small_to_large[min(indices)]


@pytest.mark.parametrize("tz", (None, "Europe/Berlin"))
@pytest.mark.parametrize(
    ("stamp_in", "stamp_out"),
    [
        ("2020-01-01 00:00", "2019-12-31 00:00"),
        ("2020-01-01 05:00", "2019-12-31 00:00"),
        ("2020-01-01 05:59", "2019-12-31 00:00"),
        ("2020-01-01 06:00", "2020-01-01 00:00"),
        ("2020-01-01 07:00", "2020-01-01 00:00"),
        ("2020-01-02 00:00", "2020-01-01 00:00"),
        ("2020-03-29 05:00", "2020-03-28 00:00"),
        ("2020-03-29 07:00", "2020-03-29 00:00"),
        ("2020-03-30 05:00", "2020-03-29 00:00"),
        ("2020-03-30 07:00", "2020-03-30 00:00"),
        ("2020-10-25 05:00", "2020-10-24 00:00"),
        ("2020-10-25 07:00", "2020-10-25 00:00"),
        ("2020-10-26 05:00", "2020-10-25 00:00"),
        ("2020-10-26 07:00", "2020-10-26 00:00"),
    ],
)
def test_gasday_de(stamp_in: str, tz: str, stamp_out: str):
    """Test if correct gas day is identified."""
    ts = pd.Timestamp(stamp_in, tz=tz)
    expected = pd.Timestamp(stamp_out, tz=tz)
    result = stamps.gasday_de(ts)
    assert result == expected


# TODO: add tests for other functions
