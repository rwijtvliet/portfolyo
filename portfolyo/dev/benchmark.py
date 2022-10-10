"""Run several benchmarks to see how fast or slow the code is, and write to excel file."""

# import pandas as pd
# import portfolyo as pf
# import cProfile, pstats, io
# from pstats import SortKey

# LENGTHS = [5, 500, 50_000, 5_000_000]
# REPEAT = 10


# def benchmark_pfline():
#     df = pd.DataFrame(columns=[], index=LENGTHS, dtype=float)
#     for l in LENGTHS:
#         pr = cProfile.Profile()
#         i = pd.date_range("2020", periods=l, freq="15T")
#         pr.enable()
#         for _ in range(REPEAT):
#             pfl = pf.dev.get_singlepfline(i, pf.Kind.ALL)
#         pr.disable()

#     return pr


# pr.disable()
