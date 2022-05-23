"""
Issue:
Custom class ('GameDurations') implements self + pint.Quantity and pint.Quantity + self
The latter is not called: pint.Quantity.__add__ is (expectedly) called before
GameDurations.__radd__, but the pint.Quantity.__add__ raises a DimensionalityError.

Example:
"""

from __future__ import annotations
from pint import Quantity as Q_


class GameDurations:
    def __init__(self, gamedurations: dict) -> None:
        self.durdict = gamedurations

    def __repr__(self) -> str:
        return "\n".join(f"{game}: {dur}" for game, dur in self.durdict.items())

    def __add__(self, other) -> GameDurations:
        return GameDurations({game: dur + other for game, dur in self.durdict.items()})

    __radd__ = __add__


games = GameDurations({"football": Q_("90 minutes"), "monopoly": Q_("7 hours")})
games
# football: 90 minute
# monopoly: 7 hour

# working as expected:
games + Q_("30 minutes")
# football: 120 minute
# monopoly: 7.5 hour

# not working, because Quantity raises Error, so GamesDuration.__radd__ is not called
Q_("30 minutes") + games
# DimensionalityError

"""
Solution:
in `quantity.py`, change line 1048 to `return NotImplemented`.
"""
