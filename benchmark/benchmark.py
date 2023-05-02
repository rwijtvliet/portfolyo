"""Run several benchmarks to see how fast or slow the code is, and write to excel file."""

# using venv 'pf39b'

import datetime as dt
import sys
from collections import deque
from itertools import chain
from typing import Callable

import numpy as np
import pandas as pd
import tqdm

import portfolyo as pf

LENGTHS = [100, 1000, 10_000, 100_000, 300_000]
RUNS = [10, 10, 10, 3, 1]

SHEET_NAME = "new_major_version"


def total_size(o, handlers={}, verbose=False):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    Source: https://code.activestate.com/recipes/577504/
    """

    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: lambda d: chain.from_iterable(d.items()),
        set: iter,
        frozenset: iter,
        **handlers,  # user handlers take precedence
    }
    seen = set()  # track which object id's have already been seen
    default_size = sys.getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=sys.stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def benchmark_pfline(df):
    for le, ru in tqdm.tqdm(zip(LENGTHS, RUNS)):
        i = pd.date_range("2020", periods=le, freq="15T")
        for cols in ["q", "p", "qr", "wp"]:
            print(f"{le=} {cols=}")
            data = pd.DataFrame({c: np.linspace(1, 1000, le) for c in cols}, i)
            df.loc[le, ("flat", "creation", cols)] = timeit(lambda: pf.PfLine(data), ru)
            pfl = pf.PfLine(data)
            df.loc[le, ("flat", "to_str", cols)] = timeit(lambda: str(pfl), ru)
            df.loc[le, ("flat", "pfl+const", cols)] = timeit(lambda: pfl + 2, ru)
            other = pf.PfLine(data + 1)
            df.loc[le, ("flat", "pfl+pfl", cols)] = timeit(lambda: pfl + other, ru)
            df.loc[le, ("flat", "pfl*const", cols)] = timeit(lambda: pfl * 2, ru)
            if pfl.kind is pf.Kind.VOLUME or pfl.kind is pf.Kind.COMPLETE:
                df.loc[le, ("flat", "access_q", cols)] = timeit(lambda: pfl.q, ru)
                df.loc[le, ("flat", "access_w", cols)] = timeit(lambda: pfl.w, ru)
            df.loc[le, ("flat", "memory_use", cols)] = total_size(pfl)
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
            if pfl.kind is pf.Kind.VOLUME or pfl.kind is pf.Kind.COMPLETE:
                df.loc[le, ("multi10", "access_q", cols)] = timeit(lambda: pfl.q, ru)
                df.loc[le, ("multi10", "access_w", cols)] = timeit(lambda: pfl.w, ru)
            df.loc[le, ("multi10", "memory_use", cols)] = total_size(pfl)


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
        df.loc[le, ("pfstate", "memory_use", "")] = total_size(pfs)


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

with pd.ExcelWriter("benchmark.xlsx", mode="a") as writer:
    df.to_excel(writer, sheet_name=SHEET_NAME)


# if __name__ == "__main__":
#     main()
