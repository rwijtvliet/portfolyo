import pandas as pd
import pytest

from portfolyo import testing, tools


@pytest.mark.parametrize("types", ["series", "df"])
@pytest.mark.parametrize("ignore_tz", [True, False])
def test_frames_ignore_tz(types: str, ignore_tz: bool):
    idx_a = pd.date_range(
        "2020", "2022", freq="MS", inclusive="left", tz="Europe/Berlin"
    )
    a = pd.Series(range(0, 24), idx_a)

    idx_b = pd.date_range("2020-02", "2021-09", freq="MS", inclusive="left")
    b = pd.Series(range(0, 19), idx_b)

    exp_idx_a = pd.date_range(
        "2020-02", "2021-09", freq="MS", inclusive="left", tz="Europe/Berlin"
    )
    exp_idx_b = idx_b
    exp_a = pd.Series(range(1, 20), exp_idx_a)
    exp_b = pd.Series(range(0, 19), exp_idx_b)

    if types == "series":
        if not ignore_tz:
            with pytest.raises(ValueError):
                _ = tools.intersect.frames(a, b, ignore_tz=ignore_tz)
            return
        result_a, result_b = tools.intersect.frames(a, b, ignore_tz=ignore_tz)
        testing.assert_series_equal(result_a, exp_a)
        testing.assert_series_equal(result_b, exp_b)
    else:
        a, b = pd.DataFrame({"col_a": a}), pd.DataFrame({"col_b": b})
        if not ignore_tz:
            with pytest.raises(ValueError):
                _ = tools.intersect.frames(a, b, ignore_tz=ignore_tz)
            return
        exp_a, exp_b = pd.DataFrame({"col_a": exp_a}), pd.DataFrame({"col_b": exp_b})
        result_a, result_b = tools.intersect.frames(a, b, ignore_tz=ignore_tz)
        testing.assert_frame_equal(result_a, exp_a)
        testing.assert_frame_equal(result_b, exp_b)


@pytest.mark.parametrize("types", ["series", "df"])
@pytest.mark.parametrize("ignore_start_of_day", [True, False])
def test_frames_ignore_start_of_day(types: str, ignore_start_of_day: bool):
    idx_a = pd.date_range("2020 00:00", "2022 00:00", freq="MS", inclusive="left")
    a = pd.Series(range(0, 24), idx_a)

    idx_b = pd.date_range("2020-02 06:00", "2021-09 06:00", freq="MS", inclusive="left")
    b = pd.Series(range(0, 19), idx_b)

    exp_idx_a = pd.date_range(
        "2020-02 00:00", "2021-09 00:00", freq="MS", inclusive="left"
    )
    exp_idx_b = idx_b
    exp_a = pd.Series(range(1, 20), exp_idx_a)
    exp_b = pd.Series(range(0, 19), exp_idx_b)
    if types == "series":
        if not ignore_start_of_day:
            with pytest.raises(ValueError):
                _ = tools.intersect.frames(
                    a, b, ignore_start_of_day=ignore_start_of_day
                )
            return
        result_a, result_b = tools.intersect.frames(
            a, b, ignore_start_of_day=ignore_start_of_day
        )
        testing.assert_series_equal(result_a, exp_a)
        testing.assert_series_equal(result_b, exp_b)
    else:
        a, b = pd.DataFrame({"col_a": a}), pd.DataFrame({"col_b": b})
        if not ignore_start_of_day:
            with pytest.raises(ValueError):
                _ = tools.intersect.frames(
                    a, b, ignore_start_of_day=ignore_start_of_day
                )
            return
        exp_a, exp_b = pd.DataFrame({"col_a": exp_a}), pd.DataFrame({"col_b": exp_b})
        result_a, result_b = tools.intersect.frames(
            a, b, ignore_start_of_day=ignore_start_of_day
        )
        testing.assert_frame_equal(result_a, exp_a)
        testing.assert_frame_equal(result_b, exp_b)


@pytest.mark.parametrize("types", ["series", "df"])
@pytest.mark.parametrize("ignore_freq", [True, False])
def test_frames_ignore_freq(types: str, ignore_freq: bool):
    idx_a = pd.date_range("2022-04-01", "2024-07-01", freq="QS", inclusive="left")
    a = pd.Series(range(0, 9), idx_a)

    idx_b = pd.date_range("2021-01-01", "2024-01-01", freq="AS", inclusive="left")
    b = pd.Series(range(0, 3), idx_b)

    exp_idx_a = pd.date_range("2023-01-01", "2024-01-01", freq="QS", inclusive="left")
    exp_idx_b = pd.date_range("2023-01-01", "2024-01-01", freq="AS", inclusive="left")
    exp_a = pd.Series(range(3, 7), exp_idx_a)
    exp_b = pd.Series(range(2, 3), exp_idx_b)
    if types == "series":
        if not ignore_freq:
            with pytest.raises(ValueError):
                _ = tools.intersect.frames(a, b, ignore_freq=ignore_freq)
            return
        result_a, result_b = tools.intersect.frames(a, b, ignore_freq=ignore_freq)
        testing.assert_series_equal(result_a, exp_a)
        testing.assert_series_equal(result_b, exp_b)
    else:
        a, b = pd.DataFrame({"col_a": a}), pd.DataFrame({"col_b": b})
        if not ignore_freq:
            with pytest.raises(ValueError):
                _ = tools.intersect.frames(a, b, ignore_freq=ignore_freq)
            return
        exp_a, exp_b = pd.DataFrame({"col_a": exp_a}), pd.DataFrame({"col_b": exp_b})
        result_a, result_b = tools.intersect.frames(a, b, ignore_freq=ignore_freq)
        testing.assert_frame_equal(result_a, exp_a)
        testing.assert_frame_equal(result_b, exp_b)
