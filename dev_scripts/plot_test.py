"""Create several plots to see if plotting still works."""

import pandas as pd


def plots():
    """Create 16 plots: (w, q, p, r) x (hourly, monthly) x (children, no children)."""
    i = pd.date_range("2020", "2021", freq="H")
    print(i)
