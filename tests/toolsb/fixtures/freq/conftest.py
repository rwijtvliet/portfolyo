import numpy as np
import pytest


@pytest.fixture(
    scope="session",
    params=[
        ("min", "5min"),
        ("min", "D"),
        ("min", "QS-JAN"),
        ("min", "QS-FEB"),
        ("D", "QS-JAN"),
        ("D", "QS-FEB"),
        ("D", "YS-FEB"),
        ("MS", "QS-JAN"),
        ("MS", "QS-FEB"),
        ("QS-JAN", "YS-JAN"),
        ("QS-JAN", "YS-APR"),
        ("QS-APR", "YS-JAN"),
    ],
)
def freqs_shortlong_asstr(request) -> tuple[str, str]:
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        ("QS-JAN", "QS-APR"),
        ("QS-JAN", "QS-JUL"),
        ("QS-JAN", "QS-OCT"),
        ("QS-APR", "QS-JUL"),
        ("QS-APR", "QS-OCT"),
        ("QS-JUL", "QS-OCT"),
        ("QS-FEB", "QS-MAY"),
    ],
)
def freqs_equivalent_asstr(request) -> tuple[str, str]:
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        ("QS-JAN", "QS-FEB"),
        ("QS-JAN", "QS-MAR"),
        ("QS-JAN", "QS-MAY"),
        ("QS-JAN", "QS-JUN"),
        ("QS-FEB", "QS-MAR"),
        ("QS-FEB", "QS-APR"),
        ("YS-JAN", "YS-FEB"),
        ("YS-JAN", "YS-MAR"),
        ("YS-JAN", "YS-APR"),
        ("QS-JAN", "YS-FEB"),
        ("QS-JAN", "YS-MAR"),
    ],
)
def freqs_incompatible_asstr(request) -> tuple[str, str]:
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        ("min", "5min", "D", "MS", "QS-JAN", "YS-JAN"),
        ("15min", "h", "QS-JAN", "YS-APR"),
        ("h", "D", "QS-APR", "YS-JAN"),
        ("min", "min", "5min", "D", "D", "MS", "QS-JAN", "YS-JAN"),  # repeated entries
    ],
)
def freqs_sorted_asstr(request) -> tuple[str, ...]:
    return request.param


@pytest.fixture(scope="session")
def freqs_unsorted_asstr(freqs_sorted_asstr, seed) -> tuple[str, ...]:
    np.random.seed(seed)
    return tuple(np.random.default_rng().permutation(freqs_sorted_asstr))


@pytest.fixture(scope="session")
def freqs_unsortable_asstr(freqs_incompatible_asstr, seed) -> tuple[str, ...]:
    np.random.seed(seed)
    freqs_unsortable = ["min", "5min", "D", "MS", *freqs_incompatible_asstr]
    return tuple(np.random.default_rng().permutation(freqs_unsortable))


@pytest.fixture(
    scope="session",
    params=[
        (
            ("min", "5min", "D", "MS", "QS-JAN", "QS-APR"),
            ("min", "5min", "D", "MS", "QS-APR", "QS-JAN"),
        ),
        (
            ("15min", "h", "h", "QS-JAN", "QS-APR", "QS-JUL", "YS-APR"),
            ("15min", "h", "h", "QS-JAN", "QS-JUL", "QS-APR", "YS-APR"),
            ("15min", "h", "h", "QS-APR", "QS-JAN", "QS-JUL", "YS-APR"),
            ("15min", "h", "h", "QS-APR", "QS-JUL", "QS-JAN", "YS-APR"),
            ("15min", "h", "h", "QS-JUL", "QS-JAN", "QS-APR", "YS-APR"),
            ("15min", "h", "h", "QS-JUL", "QS-APR", "QS-JAN", "YS-APR"),
        ),
    ],
)
def freqs_sorted_inclcompatible_asstr(request) -> tuple[tuple[str, ...]]:
    return request.param


@pytest.fixture(scope="session")
def freqs_unsorted_inclcompatible_asstr(freqs_sorted_inclcompatible_asstr, seed) -> tuple[str, ...]:
    np.random.seed(seed)
    return tuple(np.random.default_rng().permutation(freqs_sorted_inclcompatible_asstr[0]))
