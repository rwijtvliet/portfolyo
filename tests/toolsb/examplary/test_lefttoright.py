import pandas as pd
import pytest

from portfolyo import toolsb

LEFT_FREQ_RIGHT = [
    ("2020", "min", "2020-01-01 00:01"),
    ("2020", "5min", "2020-01-01 00:05"),
    ("2020", "15min", "2020-01-01 00:15"),
    ("2020", "h", "2020-01-01 01:00"),
    ("2020", "D", "2020-01-02"),
    ("2020", "MS", "2020-02"),
    ("2020", "QS", "2020-04"),
    ("2020", "QS-APR", "2020-04"),
    ("2020", "QS-JUL", "2020-04"),
    ("2020", "QS-OCT", "2020-04"),
    ("2020", "YS", "2021"),
    ("2020-05", "min", "2020-05-01 00:01"),
    ("2020-05", "5min", "2020-05-01 00:05"),
    ("2020-05", "15min", "2020-05-01 00:15"),
    ("2020-05", "h", "2020-05-01 01:00"),
    ("2020-05", "D", "2020-05-02"),
    ("2020-05", "MS", "2020-06"),
    ("2020-05", "QS-FEB", "2020-08"),
    ("2020-05", "QS-MAY", "2020-08"),
    ("2020-05", "QS-AUG", "2020-08"),
    ("2020-05", "QS-NOV", "2020-08"),
    ("2020-05", "YS-MAY", "2021-05"),
    ("2020-04-21 15:00", "min", "2020-04-21 15:01"),
    ("2020-04-21 15:00", "5min", "2020-04-21 15:05"),
    ("2020-04-21 15:00", "15min", "2020-04-21 15:15"),
    ("2020-04-21 15:00", "h", "2020-04-21 16:00"),
]


@pytest.mark.parametrize(
    "left,freq,right",
    [
        (pd.Timestamp(left), freq, pd.Timestamp(right))
        for (left, freq, right) in LEFT_FREQ_RIGHT
    ],
)
def test_lefttoright_stamp(left, freq, right):
    assert toolsb._lefttoright.stamp(left, freq) == right


@pytest.mark.parametrize(
    "left,right",
    [
        (
            pd.date_range(left, periods=3, freq=freq),
            pd.date_range(right, periods=3, freq=freq),
        )
        for (left, freq, right) in LEFT_FREQ_RIGHT
    ],
)
def test_lefttoright_index(left, right):
    pd.testing.assert_index_equal(toolsb._lefttoright.index(left), right)
