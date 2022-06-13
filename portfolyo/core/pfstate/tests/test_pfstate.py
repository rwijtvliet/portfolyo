from portfolyo import dev, testing, PfState  # noqa

# Assert correct working of _make_pflines
# . check unsourced and offtake are specified.
# . check that inconsistent data raises error.
# . check with keys having unequal indexes: unequal freq, timeperiod.
# . check if missing values have expected result.


def test_make_pflines():
    pass


# . check correct working of attributes .from_series().
def test_from_series():
    pass


# . timeseries can be accessed with .offtake, .sourced, .unsourced, .unsourcedprice, .netposition, .pnl_cost, .hedgefraction, etc.
def test_pfstate_consistency():
    pass


# . initialisation with pfstate returns identical pfstate.
def test_init_from_pfstate():
    pass


# # . check correct working of attributes .df().
# @pytest.mark.parametrize("tz", ["Europe/Berlin", None])
# @pytest.mark.parametrize("freq", ["MS", "D"])
# @pytest.mark.parametrize(
#     "columns",
#     np.random.choice(["w", "q", "pr", "qr", "pq", "wp", "wr"]),
# )
# def test_df(tz, freq, columns):
#     i = dev.get_index(tz, freq)
#     vals = np.random.rand(len(i), 2)
#     result = pd.DataFrame(vals, i, columns)
#     expected_df = PfState(result).df()

#     pd.testing.assert_frame_equal(result, expected_df)


# # . check correct working of attributes .changefreq().
# @pytest.mark.parametrize("freq", np.random.choice("MS", 3, False))
# @pytest.mark.parametrize("newfreq", np.random.choice("QS", 3, False))
# @pytest.mark.parametrize(
#     "columns",
#     np.random.choice(["pr", "qr", "pq", "wp", "wr"]),
# )
# def test_changefreq(freq, newfreq, columns):
#     df = dev.get_dataframe(dev.get_index("Europe/Berlin", freq), columns)
#     pfs1 = PfState(dev.get_pfline(df))
#     pfs2 = pfs1.asfreq(newfreq)

#     # Compare the dataframes.
#     df1, df2 = pfs1.df(columns), pfs2.df(columns)
#     if df2.empty:
#         return

#     # To compare, only keep time intervals that are in both objects.
#     df1 = df1[(df1.index >= df2.index[0]) & (df1.ts_right <= df2.ts_right[-1])]

#     if df1.empty:
#         return

#     assert np.isclose(df1.columns[0].sum(), df2.columns[0].sum())
#     assert np.isclose(df1.columns[1].sum(), df2.columns[1].sum())
