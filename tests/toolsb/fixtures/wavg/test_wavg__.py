import pytest
import pandas as pd
import numpy as np


def assertquantityequality(q1, q2):
    assert str(q1.units) == str(
        q2.units
    )  # TODO: remove str() when 'different unit registries' error is found
    assert q1.magnitude == q2.magnitude or np.isnan(q1.magnitude) and np.isnan(q2.magnitude)


def assertseriesequality(s1, s2):
    assert str(s1.pint.units) == str(s2.pint.units)
    nans1, nans2 = s1.isna(), s2.isna()
    pd.testing.assert_series_equal(nans1, nans2)
    pd.testing.assert_series_equal(s1[~nans1].pint.magnitude, s2[~nans2].pint.magnitude)


def do_test_wavg_series(s, weights, wavg, fn):
    if isinstance(wavg, type) and issubclass(wavg, Exception):
        with pytest.raises(wavg):
            _ = fn(s, weights)

    else:
        assertquantityequality(fn(s, weights), wavg)
        # pf.testing.assert_value_equal(wavgfnseries(values1d, weights0d), wavg_for_values1d_and_weights0d)


def do_test_wavg_dataframe(df, weights, axis, wavg, fn):
    if isinstance(wavg, type) and issubclass(wavg, Exception):
        with pytest.raises(wavg):
            _ = fn(df, weights, axis)

    else:
        assertseriesequality(fn(df, weights, axis), wavg)
        # pf.testing.assert_value_equal(wavgfnseries(values1d, weights0d), wavg_for_values1d_and_weights0d)


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
