import enum
from typing import Iterable


class Kind(enum.Enum):
    """Enumerate what kind of information (which dimensions) is present in a PfLine."""

    # abbreviation, available columns (in order), summable (pfl1 + pfl2) columns, human-readable text
    VOLUME = "vol", "wq", "q"
    PRICE = "pri", "p", "p"
    REVENUE = "rev", "r", "r"
    COMPLETE = "cmp", "wqpr", "qr"

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value[0] == value:
                return member

    @classmethod
    def from_cols(cls, cols: Iterable[str]):
        cols = set(cols)
        for kind in cls:
            if set(kind.available) == cols:
                return kind
        raise ValueError("No fitting 'kind' found.")

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(self.value[1])

    @property
    def summable(self) -> tuple[str, ...]:
        return tuple(self.value[2])

    def __repr__(self):
        return f"<{self.value[0]}>"

    def __str__(self):
        return self.value[0]


class Structure(enum.Enum):
    """Enumerate if the PfLine has children or not."""

    FLAT = enum.auto()
    NESTED = enum.auto()
