import enum


class Kind(enum.Enum):
    """Enumerate what kind of information (which dimensions) is present in a PfLine."""

    # abbreviation, available columns, summable (pfl1 + pfl2) columns
    VOLUME = "vol", "wq", "q"
    PRICE = "pri", "p", "p"
    REVENUE = "rev", "r", "r"
    COMPLETE = "all", "wqpr", "qr"

    @classmethod
    def _missing_(cls, val):
        for member in cls:
            if member.value[0] == val:
                return member

    @property
    def available(self):
        return tuple(self.value[1])

    @property
    def summable(self):
        return tuple(self.value[2])

    def __repr__(self):
        return f"<{self.value[0]}>"

    def __str__(self):
        return self.value[0]


class Structure(enum.Enum):
    """Enumerate if the PfLine has children or not."""

    FLAT = enum.auto()
    NESTED = enum.auto()
