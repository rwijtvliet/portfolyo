import pandas as pd
import pytest
from portfolyo import tools


@pytest.mark.parametrize("tz_param", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("offset_hours", [0, 6])
def test_tsleftright_nonespecified(tz_param: str, offset_hours: int):
    """Test if start and end of interval are correctly calculated, if none is specified."""
    # None specified, so tz parameter and offset_hours should be used in both return values.
    nextyear = pd.Timestamp.today().year + 1
    expected = (
        pd.Timestamp(year=nextyear, month=1, day=1, hour=offset_hours, tz=tz_param),
        pd.Timestamp(year=nextyear + 1, month=1, day=1, hour=offset_hours, tz=tz_param),
    )
    result = tools.stamps.ts_leftright(None, None, tz_param, offset_hours)
    for a, b in zip(result, expected):
        assert a == b


@pytest.mark.parametrize("tz_param", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("tz_specified", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("tss", "offset_hours", "expected_tss"),
    [
        (("2020-01-01", None), 0, ("2020-01-01", "2021-01-01")),
        (("2020-01-01", None), 6, ("2020-01-01", "2021-01-01 06:00")),
        ((None, "2020-02-02"), 0, ("2020-01-01", "2020-02-02")),
        ((None, "2020-02-02"), 6, ("2020-01-01 06:00", "2020-02-02")),
        (("2020-03-03 3:33", None), 0, ("2020-03-03 3:33", "2021-01-01")),
        (("2020-03-03 3:33", None), 6, ("2020-03-03 3:33", "2021-01-01 06:00")),
        ((None, "2021-10-09"), 0, ("2021-01-01", "2021-10-09")),
        ((None, "2021-10-09"), 6, ("2021-01-01 06:00", "2021-10-09")),
    ],
)
def test_tsleftright_onespecified(
    tss: tuple, expected_tss: tuple, tz_specified: str, tz_param: str, offset_hours: int
):
    """Test if start and end of interval are correctly calculated, if one is specified."""
    # One specified, so tz parameter should be ignored.
    # There should be no timezone errors and no swapping is necessary.
    tss = [pd.Timestamp(ts, tz=tz_specified) for ts in tss]  # one will be NaT
    expected = [pd.Timestamp(ts, tz=tz_specified) for ts in expected_tss]
    result = tools.stamps.ts_leftright(*tss, tz_param, offset_hours)

    for a, b in zip(result, expected):
        assert a == b


@pytest.mark.parametrize("tz_param", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    "tzs_specified",
    [
        (None, None),
        ("Europe/Berlin", "Europe/Berlin"),
        (None, "Europe/Berlin"),
        ("Europe/Berlin", "Asia/Kolkota"),
    ],
)
@pytest.mark.parametrize("offset_hours", [0, 6])
@pytest.mark.parametrize(
    ("tss", "expected_tss"),
    [
        (("2020-01-01", "2020-02-02"), ("2020-01-01", "2020-02-02")),
        (("2020-01-01", "2020-02-02"), ("2020-01-01", "2020-02-02")),
        (("2020-02-02", "2020-01-01"), ("2020-01-01", "2020-02-02")),
        (("2020-02-02", "2020-01-01"), ("2020-01-01", "2020-02-02")),
        (("2020-01-01", "2020-01-01"), ("2020-01-01", "2020-01-01")),
        (("2020-01-01", "2020-01-01"), ("2020-01-01", "2020-01-01")),
        (("2020-03-03 3:33", "2021-10-09"), ("2020-03-03 3:33", "2021-10-09")),
        (("2021-10-09", "2020-03-03 3:33"), ("2020-03-03 3:33", "2021-10-09")),
        (("2020-03-03 3:33", "2021-10-09"), ("2020-03-03 3:33", "2021-10-09")),
    ],
)
def test_tsleftright_bothspecified(
    tss: tuple,
    expected_tss: tuple,
    tzs_specified: str,
    tz_param: str,
    offset_hours: int,
):
    """Test if start and end of interval are correctly calculated, if both are specified."""
    # Both specified, so tz parameter and offset_hours should be ignored
    # There should be a timezone error if they are unequal, and there should be swapping if necessary.
    tss = [pd.Timestamp(ts, tz=tz) for ts, tz in zip(tss, tzs_specified)]

    if tzs_specified[0] != tzs_specified[1]:
        with pytest.raises(ValueError):
            _ = tools.stamps.ts_leftright(*tss, tz_param, offset_hours)

    expected = [pd.Timestamp(ts, tz=tzs_specified[0]) for ts in expected_tss]
    result = tools.stamps.ts_leftright(*tss, tz_param, offset_hours)

    for a, b in zip(result, expected):
        assert a == b
