"""Verify input data and turn into object needed in SinglePfLine instantiation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ... import tools
from . import base, interop
from .base import Kind


def make_dataframe(data) -> pd.DataFrame:
    """From data, create a DataFrame with column `q`, column `p`, or columns `q` and `r`,
    with relevant units set to them. Also, do some data verification."""
    # Shortcut if PfLine is passed.
    if isinstance(data, base.PfLine):
        # return data.flatten()._df  <-- don't use, causes infinite recursion due to __init__ calls.
        # Works for SinglePfLine and MultiPfLine.
        if data.kind is Kind.VOLUME_ONLY:
            return pd.DataFrame({"q": data.q})
        elif data.kind is Kind.PRICE_ONLY:
            return pd.DataFrame({"p": data.p})
        else:  # data.kind == 'all'
            return pd.DataFrame({"q": data.q, "r": data.r})

    # Turn into interoperability object
    if isinstance(data, interop.InOp):
        inop = data
    else:
        inop = interop.InOp.from_data(data).to_timeseries()

    # Check data types.
    if inop.agn is not None or inop.nodim is not None:
        err = []
        if inop.nodim is not None:
            err.append(f"explicityly dimensionless ({inop.nodim})")
        if inop.agn is not None:
            err.append(f"dimension-agnostic ({inop.agn})")
        raise ValueError(
            f"Found {' and '.join(err)} data. Use 'w', 'q', 'p', 'r' (e.g. as dictionary"
            " keys), or explicitly pass values with a ``pint`` unit, to indicate dimensionality."
        )

    # Make actual dataframe.
    return _dataframe_from_series(inop.w, inop.q, inop.p, inop.r)


def _dataframe_from_series(
    w: pd.Series = None, q: pd.Series = None, p: pd.Series = None, r: pd.Series = None
) -> pd.DataFrame:

    # Get price information.
    if p is not None and w is None and q is None and r is None:
        # We only have price information. Return immediately.
        return pd.DataFrame({"p": p.pint.to_base_units()})  # kind is PRICE_ONLY

    # Get quantity information (and check consistency).
    if q is None and w is None:
        if r is not None:
            if p is not None:
                q = r / p
            else:
                q = interop.InOp(q=0).to_timeseries(r.index).q
        elif p is None:
            raise ValueError(
                "Must supply (a) volume, (b) price, (c) revenue, or (d) a (consistent) combination of these."
            )
    if q is None:
        q = w * w.index.duration
    elif w is not None and not tools.frame.series_allclose(q, w * w.index.duration):
        raise ValueError("Passed values for ``q`` and ``w`` not consistent.")

    # Get revenue information (and check consistency).
    if p is None and r is None:
        return pd.DataFrame({"q": q.pint.to_base_units()})  # kind is VOLUME_ONLY
    if r is None:  # must calculate from p
        # Edge case: p==nan or p==inf. If q==0, assume r=0. If q!=0, raise error
        r = p * q
        i = r.isna() | np.isinf(r.pint.m)
        if i.any():
            if (abs(q.pint.m[i]) > 1e-5).any():
                raise ValueError(
                    "Found timestamps with ``p``==na yet ``q``!=0. Unknown ``r``."
                )
            r[i] = 0
    elif p is not None and not tools.frame.series_allclose(r, p * q):
        # Edge case: remove lines where p==nan or p==inf and q==0 before judging consistency.
        i = p.isna() | np.isinf(p.pint.m)
        if not (abs(q.pint.m[i]) < 1e-5).all() or not tools.frame.series_allclose(
            r[~i], p[~i] * q[~i]
        ):
            raise ValueError("Passed values for ``q``, ``p`` and ``r`` not consistent.")
    q, r = q.pint.to_base_units(), r.pint.to_base_units()
    return pd.DataFrame({"q": q, "r": r}).dropna()  # kind is ALL
