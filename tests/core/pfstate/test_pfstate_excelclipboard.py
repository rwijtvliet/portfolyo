"""Test if portfolio line can be exported."""


import portfolyo as pf


def test_pfstate_toexcel():
    """Test if data can be exported to excel."""
    pfs = pf.dev.get_pfstate()
    pfs.to_excel("test.xlsx")


def test_pfstate_toclipboard():
    """Test if data can be copied to clipboard."""
    pfs = pf.dev.get_pfstate()
    pfs.to_clipboard()
