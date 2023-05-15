"""Create uniform ids for testcases."""
from typing import Any, Dict
import pandas as pd

import portfolyo as pf
from portfolyo.core.pfline import Kind, Structure, classes


def id_fn(data: Any):
    """Readable id of test case"""
    if isinstance(data, Dict):
        return str({key: id_fn(val) for key, val in data.items()})
    elif isinstance(data, pd.Series):
        if isinstance(data.index, pd.DatetimeIndex):
            return f"Timeseries({data.dtype})"
        return f"Series(idx:{''.join(str(i) for i in data.index)})"
    elif isinstance(data, pd.DataFrame):
        return f"Df({''.join(str(c) for c in data.columns)})"
    elif isinstance(data, classes.PfLine):
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
    elif data is None:
        return "None"
    return str(data)
