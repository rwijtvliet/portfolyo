import pytest
from portfolyo import toolsb


def do_test_wavg_series(s, weights, wavg, fn):
    if isinstance(wavg, type) and issubclass(wavg, Exception):
        with pytest.raises(Exception):
            _ = fn(s, weights)

    else:
        toolsb.testing.assert_scalar_equal(fn(s, weights), wavg)


def do_test_wavg_dataframe(df, weights, axis, wavg, fn):
    if isinstance(wavg, type) and issubclass(wavg, Exception):
        with pytest.raises(Exception):
            _ = fn(df, weights, axis)

    else:
        toolsb.testing.assert_series_equal(fn(df, weights, axis).sort_index(), wavg.sort_index())


class TestWavgValues1dWeights0d:
    def test_wavg(
        self,
        wavgfnseries,
        values1d,
        weights0d,
        wavg_for_values1d_and_weights0d,
    ):
        do_test_wavg_series(values1d, weights0d, wavg_for_values1d_and_weights0d, wavgfnseries)


class TestWavgValues1dWeights1d:
    def test_wavg(
        self,
        wavgfnseries,
        values1d,
        weights1d,
        wavg_for_values1d_and_weights1d,
    ):
        do_test_wavg_series(values1d, weights1d, wavg_for_values1d_and_weights1d, wavgfnseries)


class TestWavgValues2dWeights0d:
    def test_wavg(
        self,
        wavgfndataframe,
        values2d,
        weights0d,
        axis,
        wavg_for_values2d_and_weights0d,
    ):
        do_test_wavg_dataframe(
            values2d, weights0d, axis, wavg_for_values2d_and_weights0d, wavgfndataframe
        )


class TestWavgValues2dWeights1d:
    def test_wavg(
        self,
        wavgfndataframe,
        values2d,
        weights1d,
        axis,
        wavg_for_values2d_and_weights1d,
    ):
        do_test_wavg_dataframe(
            values2d, weights1d, axis, wavg_for_values2d_and_weights1d, wavgfndataframe
        )


class TestWavgValues2dWeights2d:
    def test_wavg(
        self,
        wavgfndataframe,
        values2d,
        weights2d,
        axis,
        wavg_for_values2d_and_weights2d,
    ):
        do_test_wavg_dataframe(
            values2d, weights2d, axis, wavg_for_values2d_and_weights2d, wavgfndataframe
        )
