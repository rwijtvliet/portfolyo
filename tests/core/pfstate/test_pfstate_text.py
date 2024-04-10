"""Test if portfolio state can be printed."""

import portfolyo as pf


def test_pfstate_print():
    """Test if portfolio state can be printed."""
    pfs = pf.dev.get_pfstate()
    pfs.print()
