import datetime as dt

import pandas as pd
import pytest

from portfolyo import tools

f_germanpower = tools.peakfn.factory(dt.time(hour=8), dt.time(hour=20))


@pytest.mark.parametrize("withunits", ["units", "nounits"])
@pytest.mark.parametrize("testcol", ["base", "peak", "offpeak"])
@pytest.mark.parametrize(
    "bpoframe",
    [
        pd.DataFrame(
            {
                "peak": [100.0, 100, 100, 100],
                "base": [80.0, 80, 80, 80],
                "offpeak": [68.2051282, 69.4736842, 68.9770355, 68.421053],
            },
            pd.date_range("2020", periods=4, freq="MS", tz="Europe/Berlin"),
        ),
        pd.DataFrame(
            {
                "peak": [100.0, 100, 100, 100],
                "base": [80.0, 80, 80, 80],
                "offpeak": [68.8510638, 68.8699360, 68.9361702, 68.9361702],
            },
            pd.date_range("2020", periods=4, freq="YS", tz="Europe/Berlin"),
        ),
    ],
)
def test_completebpoframe_averagable(bpoframe, testcol: str, withunits: str):
    """Test if missing column can be reconstructed."""
    if withunits == "units":
        bpoframe = bpoframe.astype("pint[Eur/MWh]")
    df = bpoframe.drop(columns=testcol)
    result = tools.peakconvert.complete_bpoframe(df, f_germanpower, is_summable=False)
    tools.testing.assert_dataframe_equal(result, bpoframe)


@pytest.mark.parametrize("withunits", ["units", "nounits"])
@pytest.mark.parametrize("testcol", ["base", "peak", "offpeak"])
@pytest.mark.parametrize(
    "bpoframe",
    [
        pd.DataFrame(
            {
                "peak": [100.0, 100, 100, 100],
                "base": [280.0, 180, 80, 380],
                "offpeak": [180.0, 80, -20, 280],
            },
            pd.date_range("2020", periods=4, freq="MS", tz="Europe/Berlin"),
        ),
        pd.DataFrame(
            {
                "peak": [100.0, 100, 100, 100],
                "base": [280.0, 180, 80, 380],
                "offpeak": [180.0, 80, -20, 280],
            },
            pd.date_range("2020", periods=4, freq="YS", tz="Europe/Berlin"),
        ),
    ],
)
def test_completebpoframe_summable(bpoframe, testcol: str, withunits: str):
    """Test if missing column can be reconstructed."""
    if withunits == "units":
        bpoframe = bpoframe.astype("pint[Eur/MWh]")
    df = bpoframe.drop(columns=testcol)
    result = tools.peakconvert.complete_bpoframe(df, f_germanpower, is_summable=True)
    tools.testing.assert_dataframe_equal(result, bpoframe)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("withunits", ["units", "nounits"])
@pytest.mark.parametrize("testcol", ["base", "peak", "offpeak"])
@pytest.mark.parametrize(
    ("b", "p", "o", "o_EuropeBerlin", "ts_left", "freq"),
    [
        (100, 100, 100, None, "2020-01-01", "MS"),  # 31 days, 23 working days
        (100, 200, 41.02564103, None, "2020-01-01", "MS"),
        (100, 300, -17.94871795, None, "2020-01-01", "MS"),
        (100, 100, 100, None, "2020-01-01", "QS"),  # 91 days, 65 working days
        (100, 200, 44.44444444, 44.40484676, "2020-01-01", "QS"),
        (100, 300, -11.11111111, -11.19030649, "2020-01-01", "QS"),
        (100, 100, 100, None, "2020-01-01", "D"),  # weekday
        (100, 200, 0, None, "2020-01-01", "D"),  # weekday
        (100, 300, -100, None, "2020-01-01", "D"),  # weekday
    ],
)
def test_moreconversions_averagable(
    b: float,
    p: float,
    o: float,
    ts_left: str,
    freq: str,
    withunits: str,
    tz: str,
    o_EuropeBerlin: float,
    testcol: str,
):
    """Test if base, peak and offpeak values can be calculated from single values."""
    # Handle timezone.
    i = pd.date_range(ts_left, freq=freq, tz=tz, periods=1)
    if tz is not None and o_EuropeBerlin is not None:
        o = o_EuropeBerlin

    expected = pd.DataFrame({"peak": [p], "base": [b], "offpeak": [o]}, i).astype(float)

    # Handle units.
    if withunits == "units":
        expected = expected.astype("pint[MW]")

    df = expected.drop(columns=testcol)

    # Do testing.
    result = tools.peakconvert.complete_bpoframe(df, f_germanpower, is_summable=False)
    tools.testing.assert_dataframe_equal(result, expected)
