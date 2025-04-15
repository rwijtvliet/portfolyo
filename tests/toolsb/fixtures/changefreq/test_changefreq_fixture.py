import pandas as pd

from portfolyo import toolsb


def test_resample_index(inputidx, targetfreq, outputidx):
    """Test resampling of indices."""
    toolsb.testing.assert_index_equal(toolsb.changefreq.index(inputidx, targetfreq), outputidx)


def test_resample_frame(inputframe, outputframe, targetfreq, avg_or_sum, frame_type):
    """Test resampling of series and dataframes."""
    resample_fn = {
        "sum": toolsb.changefreq.summable,
        "avg": toolsb.changefreq.averagable,
    }[avg_or_sum]
    assert_fn = {
        pd.DataFrame: toolsb.testing.assert_frame_equal,
        pd.Series: toolsb.testing.assert_series_equal,
    }[frame_type]

    assert_fn(resample_fn(inputframe, targetfreq), outputframe)
