import portfolyo as pf


def test_getversion():
    """Test if the version number of the package can be read."""
    version = pf.__version__
    assert len(version)
    parts = version.split(".")
    assert len(parts) == 3
    for part in parts:
        _ = int(part)  # check is number
