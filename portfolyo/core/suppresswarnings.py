"""Suppress unhelpful/known/expected warnings."""

import pint
import warnings


def apply():
    warnings.filterwarnings("ignore", category=pint.UnitStrippedWarning)
    warnings.filterwarnings(
        "ignore",
        r".*does not support magnitudes of <class 'int'>. Converting magnitudes to float.*",
        RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore", r".*Timestamp.freq is deprecated and will be removed.*", FutureWarning
    )
