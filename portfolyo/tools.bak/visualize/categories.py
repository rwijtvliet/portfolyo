from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .. import unit as tools_unit


@dataclass
class Category:
    """Class to store characteristics of a category on a categorical datetime axis."""

    x: int
    y: Any
    ts: pd.Timestamp
    label: str
    short_label: str
    prio: int  # the higher the number, the least important the category. 0 indicates highest priority.


class Categories:
    def __init__(self, s: pd.Series):
        self._categories: Iterable[Category] = create_categories(s)

    def categories(self, max_count: int = None) -> Iterable[Category]:
        """Return subset of categories with `max_count` or fewer elements."""
        if max_count is None:
            max_count = np.inf
        categories = self._categories
        while len(categories) > max_count:
            prio = max(cat.prio for cat in categories)
            categories = [cat for cat in categories if cat.prio < prio]
        return categories

    def _get_subset(self, attr: str, max_count: int = None) -> Iterable:
        values = [getattr(cat, attr) for cat in self.categories(max_count)]
        if not isinstance(values[0], tools_unit.Q_):
            return np.array(values)
        unit = values[0].units
        magnitudes = [value.to(unit).m for value in values]
        return tools_unit.PA_(magnitudes, unit)

    def x(self, max_count: int = None) -> Iterable[int]:
        return self._get_subset("x", max_count)

    def y(self, max_count: int = None) -> Iterable[float | tools_unit.Q_]:
        return self._get_subset("y", max_count)

    def ts(self, max_count: int = None) -> Iterable[pd.Timestamp]:
        return self._get_subset("ts", max_count)

    def labels(self, max_count: int = None) -> Iterable[str]:
        return self._get_subset("label", max_count)

    def short_labels(self, max_count: int = None) -> Iterable[str]:
        return self._get_subset("short_label", max_count)


def create_categories(s: pd.Series) -> Iterable[Category]:
    if s.index.freq == "YS":
        prios = priolist(len(s))

        def category(i, ts, y):
            label = f"{ts.year}"
            short_label = label[-2:] if i > 0 else label
            return Category(i, y, ts, label, short_label, prios[i])

    elif s.index.freq == "QS":

        def category(i, ts, y):
            num = ts.quarter
            prio = 0 if i == 0 else {1: 1, 2: 3, 3: 2, 4: 3}[num]
            year = f"{ts.year}" if i == 0 or num == 1 else ""
            label = short_label = f"Q{num}\n{year}"
            return Category(i, y, ts, label, short_label, prio)

    elif s.index.freq == "MS":
        prios = priolist(12)

        def category(i, ts, y):
            num, name = ts.month, ts.month_name()[:3]
            prio = 0 if i == 0 else prios[num - 1] + 1
            year = f"{ts.year}" if i == 0 or num == 1 else ""
            label = f"{name}\n{year}"
            short_label = f"{name[0]}\n{year}"
            return Category(i, y, ts, label, short_label, prio)

    else:
        raise ValueError("Daily (or shorter) data should not be plotted as categories.")

    return [category(i, ts, y) for i, (ts, y) in enumerate(s.items())]


def priolist(n):
    """Create list with n priority numbers, which indicate how important a given element
    is on a categorical scale. The higher the number, the more easily it can be dropped.
    """
    prio = 0
    prios = [prio]
    while len(prios) < n:
        prio += 1
        prios = [prios[i // 2] if i % 2 == 0 else prio for i in range(len(prios) * 2)]
    return prios[:n]
