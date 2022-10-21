"""Run several benchmarks to see how fast or slow the code is, and write to excel file."""

import datetime as dt
import sys

import numpy as np
import pandas as pd
import portfolyo as pf
import tqdm

LENGTHS = [100, 1000, 10_000, 100_000, 1_000_000]
RUNS = 3


def benchmark_pfline(df):
    for le in tqdm.tqdm(LENGTHS):
        i = pd.date_range("2020", periods=le, freq="15T")
        for cols in tqdm.tqdm(["q", "p", "qr", "wp"]):
            data = pd.DataFrame({c: np.linspace(1, 1000, le) for c in cols}, i)
            df.loc[le, ("single", "creation", cols)] = timeit(lambda: pf.PfLine(data))
            pfl = pf.PfLine(data)
            df.loc[le, ("single", "to_str", cols)] = timeit(lambda: str(pfl))
            df.loc[le, ("single", "pfl+const", cols)] = timeit(lambda: pfl + 2)
            other = pf.PfLine(data + 1)
            df.loc[le, ("single", "pfl+pfl", cols)] = timeit(lambda: pfl + other)
            df.loc[le, ("single", "pfl*const", cols)] = timeit(lambda: pfl * 2)
            if pfl.kind is pf.Kind.VOLUME_ONLY or pfl.kind is pf.Kind.ALL:
                df.loc[le, ("single", "access_q", cols)] = timeit(lambda: pfl.q)
                df.loc[le, ("single", "access_w", cols)] = timeit(lambda: pfl.w)
            df.loc[le, ("single", "memory_use", cols)] = sys.getsizeof(pfl)
            children = {
                f"child{c}": pf.PfLine(
                    pd.DataFrame({c: np.linspace(1, 1000, le) for c in cols}, i)
                )
                for c in range(10)
            }
            df.loc[le, ("multi10", "creation", cols)] = timeit(
                lambda: pf.PfLine(children)
            )
            pfl = pf.PfLine(children)
            df.loc[le, ("multi10", "to_str", cols)] = timeit(lambda: str(pfl))
            if pfl.kind is pf.Kind.VOLUME_ONLY or pfl.kind is pf.Kind.ALL:
                df.loc[le, ("multi10", "access_q", cols)] = timeit(lambda: pfl.q)
                df.loc[le, ("multi10", "access_w", cols)] = timeit(lambda: pfl.w)
            df.loc[le, ("multi10", "memory_use", cols)] = sys.getsizeof(pfl)


def benchmark_pfstate(df):
    for le in tqdm.tqdm(LENGTHS):
        i = pd.date_range("2020", periods=le, freq="15T")
        w_offtake = pf.dev.w_offtake(i)
        offtake = pf.PfLine(w_offtake)
        unsourced = pf.PfLine(pf.dev.p_marketprices(i))
        sourced = pf.PfLine(*pf.dev.wp_sourced(w_offtake))

        df.loc[le, ("pfstate", "creation", "nosourced")] = timeit(
            lambda: pf.PfState(offtake, unsourced)
        )
        df.loc[le, ("pfstate", "creation", "sourced")] = timeit(
            lambda: pf.PfState(offtake, unsourced, sourced)
        )
        pfs = pf.PfState(offtake, unsourced, sourced)
        df.loc[le, ("pfstate", "to_str", "")] = timeit(lambda: str(pfs))
        other = pf.PfState(offtake * 2, unsourced * 4)
        df.loc[le, ("pfstate", "pfs+pfs", "")] = timeit(lambda: pfs + other)
        df.loc[le, ("pfstate", "pfs*const", "")] = timeit(lambda: pfs * 2)
        df.loc[le, ("pfstate", "access_q", "")] = timeit(lambda: pfs.q)
        df.loc[le, ("pfstate", "access_w", "")] = timeit(lambda: pfs.w)
        df.loc[le, ("pfstate", "access_unsourced", "")] = timeit(lambda: pfs.unsourced)
        df.loc[le, ("pfstate", "memory_use", "")] = sys.getsizeof(pfs)


def timeit(f):
    start = dt.datetime.now()
    try:
        for _ in range(RUNS):
            _ = f()
    except Exception:
        return -1
    end = dt.datetime.now()
    return (end - start).total_seconds() / RUNS


def main():
    df = pd.DataFrame(
        columns=pd.MultiIndex.from_arrays([[], [], []]), index=LENGTHS, dtype=float
    )
    benchmark_pfline(df)
    benchmark_pfstate(df)
    with pd.ExcelWriter("benchmark.xlsx") as writer:
        df.to_excel(writer, sheet_name=f"{dt.date.today()}")


if __name__ == "__main__":
    main()
