import pandas as pd
import pytest

from portfolyo import toolsb


_DAILYSOURCE = pd.Series(
    1.0, pd.date_range("2020-10-21 06:00", "2022-04-21 06:00", freq="D", inclusive="left")
)


@pytest.mark.parametrize(
    "freq,result_if_summable",
    [
        (
            "MS",
            pd.Series(
                [30.0, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31],
                pd.date_range("2020-11-01 06:00", "2022-04-01 06:00", freq="MS", inclusive="left"),
            ),
        ),
        (
            "QS-JAN",
            pd.Series(
                [90.0, 91, 92, 92, 90],
                pd.date_range(
                    "2021-01-01 06:00", "2022-04-01 06:00", freq="QS-JAN", inclusive="left"
                ),
            ),
        ),
        (
            "QS-APR",  # same as QS-JAN above
            pd.Series(
                [90.0, 91, 92, 92, 90],
                pd.date_range(
                    "2021-01-01 06:00", "2022-04-01 06:00", freq="QS-APR", inclusive="left"
                ),
            ),
        ),
        (
            "YS-JAN",
            pd.Series(
                [365.0],
                pd.date_range(
                    "2021-01-01 06:00", "2022-01-01 06:00", freq="YS-JAN", inclusive="left"
                ),
            ),
        ),
        (
            "YS-FEB",
            pd.Series(
                [365.0],
                pd.date_range(
                    "2021-02-01 06:00", "2022-02-01 06:00", freq="YS-FEB", inclusive="left"
                ),
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [float, "pint[MWh]", "pint[Eur]"])
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(toolsb.changefreq.summable, id="summable"),
        pytest.param(toolsb.changefreq.averagable, id="avgable"),
    ],
)
def test_downsample_hourlysource(freq, result_if_summable, dtype, fn):
    """Test if data is correctly downsampled."""
    source = _DAILYSOURCE.astype(dtype)
    if fn is toolsb.changefreq.summable:
        result = result_if_summable.astype(dtype)
    else:
        result = pd.Series(1.0, result_if_summable.index).astype(dtype)
    toolsb.testing.assert_series_equal(fn(source, freq), result)


@pytest.mark.parametrize(
    "freq,value_if_summable",
    [("h", 1 / 24), ("15min", 1 / 96)],
)
@pytest.mark.parametrize("dtype", [float, "pint[MWh]", "pint[Eur]"])
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(toolsb.changefreq.summable, id="summable"),
        pytest.param(toolsb.changefreq.averagable, id="avgable"),
    ],
)
def test_upsample_hourlysource(freq, value_if_summable, dtype, fn):
    """Test if data is correctly upsampled."""
    source = _DAILYSOURCE.astype(dtype)
    result_idx = pd.date_range("2020-10-21 06:00", "2022-04-21 06:00", freq=freq, inclusive="left")
    if fn is toolsb.changefreq.summable:
        result = pd.Series(value_if_summable, result_idx).astype(dtype)
    else:
        result = pd.Series(1.0, result_idx).astype(dtype)
    toolsb.testing.assert_series_equal(fn(source, freq), result)


_QUARTERLYSOURCE = pd.Series(
    [180.0, 91, 184, 92, 180],
    pd.date_range("2021-01-01 06:00", "2022-04-01 06:00", freq="QS-JAN", inclusive="left"),
)


@pytest.mark.parametrize(
    "freq,result_if_summable",
    [
        (
            "MS",  # upsampled
            pd.Series(
                [62.0, 56, 62, 30, 31, 30, 62, 62, 60, 31, 30, 31, 62, 56, 62],
                pd.date_range("2021-01-01 06:00", "2022-04-01 06:00", freq="MS", inclusive="left"),
            ),
        ),
        (
            "QS-APR",  # unchanged
            pd.Series(
                [180.0, 91, 184, 92, 180],
                pd.date_range(
                    "2021-01-01 06:00", "2022-04-01 06:00", freq="QS-APR", inclusive="left"
                ),
            ),
        ),
        (
            "YS-JAN",  # downsampled
            pd.Series(
                [547.0],
                pd.date_range(
                    "2021-01-01 06:00", "2022-01-01 06:00", freq="YS-JAN", inclusive="left"
                ),
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [float, "pint[MWh]", "pint[Eur]"])
def test_summable_quarterlysource(freq, result_if_summable, dtype):
    """Test if data is correctly upsampled."""
    source = _QUARTERLYSOURCE.astype(dtype)
    result = result_if_summable.astype(dtype)
    toolsb.testing.assert_series_equal(toolsb.changefreq.summable(source, freq), result)


_TESTCASES = [  # period, freqs, result
    (
        ("2020-02-01", "2023-01-01"),
        ("MS", "QS-APR"),
        ("2020-04-01", "2023-01-01"),
    ),
    (
        ("2020-01-01", "2023-01-01"),
        ("MS", "QS-APR"),
        ("2020-01-01", "2023-01-01"),
    ),
    (
        ("2020-01-01", "2023-01-01"),
        ("MS", "YS-APR"),
        ("2020-04-01", "2022-04-01"),
    ),
    (
        ("2020-04-01", "2022-04-01"),
        ("YS-APR", "QS"),
        ("2020-04-01", "2022-04-01"),
    ),
]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("period,freq,result"), _TESTCASES)
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_index_new_freq(
    period: tuple[str, str],
    starttime: str,
    tz: str,
    freq: tuple[str, str],
    result: tuple[str, str],
):
    idx = pd.date_range(
        f"{period[0]} {starttime}",
        f"{period[1]} {starttime}",
        freq=freq[0],
        inclusive="left",
        tz=tz,
    )
    expected_result = pd.date_range(
        f"{result[0]} {starttime}",
        f"{result[1]} {starttime}",
        freq=freq[1],
        inclusive="left",
        tz=tz,
    )
    toolsb.testing.assert_index_equal(toolsb.changefreq.index(idx, freq[1]), expected_result)
