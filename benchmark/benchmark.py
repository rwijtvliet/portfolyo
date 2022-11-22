"""Run several benchmarks to see how fast or slow the code is, and write to excel file."""

# using venv 'pf39b'

import datetime as dt
import sys
from typing import Callable

import numpy as np
import pandas as pd
import tqdm

import portfolyo as pf

LENGTHS = [100, 1000, 10_000, 100_000, 300_000]
RUNS = [10, 10, 10, 3, 1]

SHEET_NAME = "reference"


def benchmark_pfline(df):
    for le, ru in tqdm.tqdm(zip(LENGTHS, RUNS)):
        i = pd.date_range("2020", periods=le, freq="15T")
        for cols in ["q", "p", "qr", "wp"]:
            print(f"{le=} {cols=}")
            data = pd.DataFrame({c: np.linspace(1, 1000, le) for c in cols}, i)
            df.loc[le, ("single", "creation", cols)] = timeit(
                lambda: pf.PfLine(data), ru
            )
            pfl = pf.PfLine(data)
            df.loc[le, ("single", "to_str", cols)] = timeit(lambda: str(pfl), ru)
            df.loc[le, ("single", "pfl+const", cols)] = timeit(lambda: pfl + 2, ru)
            other = pf.PfLine(data + 1)
            df.loc[le, ("single", "pfl+pfl", cols)] = timeit(lambda: pfl + other, ru)
            df.loc[le, ("single", "pfl*const", cols)] = timeit(lambda: pfl * 2, ru)
            if pfl.kind is pf.Kind.VOLUME_ONLY or pfl.kind is pf.Kind.ALL:
                df.loc[le, ("single", "access_q", cols)] = timeit(lambda: pfl.q, ru)
                df.loc[le, ("single", "access_w", cols)] = timeit(lambda: pfl.w, ru)
            df.loc[le, ("single", "memory_use", cols)] = sys.getsizeof(pfl)
            twochildren = [
                pf.PfLine(pd.DataFrame({c: np.linspace(1, 1000, le) for c in cols}, i)),
                pf.PfLine(pd.DataFrame({c: np.linspace(2, 2000, le) for c in cols}, i)),
            ]
            children = {f"child{c}": twochildren[c % 2] for c in range(10)}
            df.loc[le, ("multi10", "creation", cols)] = timeit(
                lambda: pf.PfLine(children), ru
            )
            pfl = pf.PfLine(children)
            df.loc[le, ("multi10", "to_str", cols)] = timeit(lambda: str(pfl), ru)
            if pfl.kind is pf.Kind.VOLUME_ONLY or pfl.kind is pf.Kind.ALL:
                df.loc[le, ("multi10", "access_q", cols)] = timeit(lambda: pfl.q, ru)
                df.loc[le, ("multi10", "access_w", cols)] = timeit(lambda: pfl.w, ru)
            df.loc[le, ("multi10", "memory_use", cols)] = sys.getsizeof(pfl)


def benchmark_pfstate(df):
    for le, ru in tqdm.tqdm(zip(LENGTHS, RUNS)):
        print(f"{le=}")
        i = pd.date_range("2020", periods=le, freq="15T")
        w_offtake = pf.dev.w_offtake(i)
        offtake = pf.PfLine(w_offtake)
        unsourced = pf.PfLine(pf.dev.p_marketprices(i))
        sourced = pf.PfLine(pf.dev.wp_sourced(w_offtake))

        df.loc[le, ("pfstate", "creation", "nosourced")] = timeit(
            lambda: pf.PfState(offtake, unsourced), ru
        )
        df.loc[le, ("pfstate", "creation", "sourced")] = timeit(
            lambda: pf.PfState(offtake, unsourced, sourced), ru
        )
        pfs = pf.PfState(offtake, unsourced, sourced)
        df.loc[le, ("pfstate", "to_str", "")] = timeit(lambda: str(pfs), ru)
        other = pf.PfState(offtake * 2, unsourced * 4)
        df.loc[le, ("pfstate", "pfs+pfs", "")] = timeit(lambda: pfs + other, ru)
        df.loc[le, ("pfstate", "pfs*const", "")] = timeit(lambda: pfs * 2, ru)
        df.loc[le, ("pfstate", "access_q", "")] = timeit(lambda: pfs.q, ru)
        df.loc[le, ("pfstate", "access_w", "")] = timeit(lambda: pfs.w, ru)
        df.loc[le, ("pfstate", "access_unsourced", "")] = timeit(
            lambda: pfs.unsourced, ru
        )
        df.loc[le, ("pfstate", "memory_use", "")] = sys.getsizeof(pfs)


def timeit(f: Callable, runs: int):
    start = dt.datetime.now()
    try:
        for _ in range(runs):
            _ = f()
    except Exception:
        return -1
    end = dt.datetime.now()
    return (end - start).total_seconds() / runs


df = pd.DataFrame(
    columns=pd.MultiIndex.from_arrays([[], [], []]), index=LENGTHS, dtype=float
)
benchmark_pfline(df)
benchmark_pfstate(df)

with pd.ExcelWriter("benchmark.xlsx") as writer:
    df.to_excel(writer, sheet_name=f"{SHEET_NAME}_{dt.date.today()}")


# if __name__ == "__main__":
#     main()
