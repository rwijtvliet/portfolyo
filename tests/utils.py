"""Create uniform ids for testcases."""

from typing import Any

import pandas as pd

import portfolyo as pf
from portfolyo.core.pfline import Kind, Structure, classes


def _id_fn(data: Any) -> str:
    """Readable id of test case"""
    if isinstance(data, list):
        return f'[{"|".join(_id_fn(e) for e in data)}]'
    elif isinstance(data, tuple):
        return f'({"|".join(_id_fn(e) for e in data)})'
    elif isinstance(data, dict):
        return "{" + "|".join(f"{k}:{_id_fn(v)}" for k, v in data.items()) + "}"
    elif isinstance(data, pd.DatetimeIndex):
        return f"idx[{data[0]}-{data[-1]}-{data.freq}-{data.tz}]"
    elif isinstance(data, pd.Series):
        typ = "Timeseries" if isinstance(data.index, pd.DatetimeIndex) else "Series"
        vals = "|".join([f"{v:.2f}" for v in data.values[:2]])  # max 2
        if len(data.values) > 2:
            vals += "..."
        return f"{typ}[{vals}]"
    elif isinstance(data, pd.DataFrame):
        return f"Df[{'|'.join(data.columns)}]"
    elif isinstance(data, classes.PfLine):
        return f"{data.__class__.__name__}"
    elif isinstance(data, pf.PfState):
        return f"{data.__class__.__name__}"
    elif isinstance(data, pf.Q_):
        return f"Q({data.units})"
    elif isinstance(data, type):
        return data.__name__
    elif isinstance(data, Kind):
        return str(data)
    elif isinstance(data, Structure):
        return str(data)
    elif isinstance(data, str):
        return data
    elif isinstance(data, type) and issubclass(data, Exception):
        return data.__name__
    elif data is None:
        return "None"
    return str(data)


def id_fn(data: Any) -> str:
    return f" {_id_fn(data)} "
