"""Test if children can be added to and dropped from a portfolyo line."""

from typing import Dict, Iterable

import pandas as pd
import pytest

import portfolyo as pf
from portfolyo import PfLine, testing

tz = "Europe/Berlin"

# Set 1.
ref_i = pd.date_range("2020", freq="MS", periods=3, tz=tz)
ref_series = {
    "A": pd.Series([-3.5, -5, -5], ref_i),
    "B": pd.Series([3.0, 5, 3], ref_i),
    "C": pd.Series([0.5, 2, -2], ref_i),
}
ref_children = {
    "vol": {
        "childA": pf.flatpfline({"w": ref_series["A"]}),
        "childB": pf.flatpfline({"w": ref_series["B"]}),
        "childC": pf.flatpfline({"w": ref_series["C"]}),
    },
    "pri": {
        "childA": pf.flatpfline({"p": ref_series["A"] * 100}),
        "childB": pf.flatpfline({"p": ref_series["B"] * 100}),
        "childC": pf.flatpfline({"p": ref_series["C"] * 100}),
    },
    "rev": {
        "childA": pf.flatpfline({"r": ref_series["A"] * 1000}),
        "childB": pf.flatpfline({"r": ref_series["B"] * 1000}),
        "childC": pf.flatpfline({"r": ref_series["C"] * 1000}),
    },
    "all": {
        "childA": pf.flatpfline({"w": ref_series["A"], "r": ref_series["A"] * 1000}),
        "childB": pf.flatpfline({"w": ref_series["B"], "r": ref_series["B"] * 1000}),
        "childC": pf.flatpfline({"w": ref_series["C"], "r": ref_series["C"] * 1000}),
    },
}
ref_pfl = {kind: pf.nestedpfline(ref_children[kind]) for kind in ref_children}

# Children with partial overlap.
i2 = pd.date_range("2020-02", freq="MS", periods=3, tz=tz)
series2 = {"D": pd.Series([8.0, 7, 6], i2)}
children2 = {
    "vol": {"childD": pf.flatpfline({"w": series2["D"]})},
    "pri": {"childD": pf.flatpfline({"p": series2["D"] * 100})},
    "rev": {"childD": pf.flatpfline({"r": series2["D"] * 1000})},
    "all": {"childD": pf.flatpfline({"w": series2["D"], "r": series2["D"] * 1000})},
}
i12 = pd.date_range("2020-02", freq="MS", periods=2, tz=tz)
children12 = {kind: {**ref_children[kind], **children2[kind]} for kind in ref_children}
children12_trimmed = {
    kind: {n: c.loc[i12] for n, c in {**ref_children[kind], **children2[kind]}.items()}
    for kind in ref_children
}
pfl12 = {kind: pf.nestedpfline(children12_trimmed[kind]) for kind in children12_trimmed}

# Child with no overlap.
i3 = pd.date_range("2022", freq="MS", periods=3, tz=tz)
series3 = {"D": pd.Series([8.0, 7, 6], i3)}
children3 = {
    "vol": {"childD": pf.flatpfline({"w": series3["D"]})},
    "pri": {"childD": pf.flatpfline({"p": series3["D"] * 100})},
    "rev": {"childD": pf.flatpfline({"r": series3["D"] * 1000})},
    "all": {"childD": pf.flatpfline({"w": series3["D"], "r": series3["D"] * 1000})},
}

# Child with other frequency.
i4 = pd.date_range("2020", freq="D", periods=3, tz=tz)
series4 = {"D": pd.Series([8.0, 7, 6], i4)}
children4 = {
    "vol": {"childD": pf.flatpfline({"w": series4["D"]})},
    "pri": {"childD": pf.flatpfline({"p": series4["D"] * 100})},
    "rev": {"childD": pf.flatpfline({"r": series4["D"] * 1000})},
    "all": {"childD": pf.flatpfline({"w": series4["D"], "r": series4["D"] * 1000})},
}

# Child with other timezone.
i5 = pd.date_range("2020", freq="MS", periods=3, tz=None)
series5 = {"D": pd.Series([8.0, 7, 6], i5)}
children5 = {
    "vol": {"childD": pf.flatpfline({"w": series5["D"]})},
    "pri": {"childD": pf.flatpfline({"p": series5["D"] * 100})},
    "rev": {"childD": pf.flatpfline({"r": series5["D"] * 1000})},
    "all": {"childD": pf.flatpfline({"w": series5["D"], "r": series5["D"] * 1000})},
}


def do_test_setchild(
    pfl: PfLine,
    to_add: Dict[str, PfLine],
    expected: PfLine,
    how: str,
):
    """Helper; test if setting child results in correct result."""

    def do_set(pfl, name, child):
        if how == "inplace":
            pfl[name] = child
        else:
            pfl = pfl.set_child(name, child)
        return pfl

    if type(expected) is type and issubclass(expected, Exception):
        for name, child in to_add.items():
            with pytest.raises(expected):
                do_set(pfl, name, child)
        return

    for name, child in to_add.items():
        pfl = do_set(pfl, name, child)
    result = pfl

    assert result == expected
    testing.assert_frame_equal(result.df, expected.df)


def do_test_dropchild(pfl: PfLine, to_drop: Iterable[str], expected: PfLine, how: str):
    """Helper; test if dropping child results in correct result."""

    def do_drop(pfl, name):
        if how == "inplace":
            del pfl[name]
        else:
            pfl = pfl.drop_child(name)
        return pfl

    for name in to_drop:
        if type(expected) is type and issubclass(expected, Exception):
            with pytest.raises(expected):
                do_drop(pfl, name)
            return
        else:
            pfl = do_drop(pfl, name)
    result = pfl

    assert result == expected
    testing.assert_frame_equal(result.df, expected.df)


@pytest.mark.parametrize(
    "children,expected",
    [
        (ref_children["vol"], ref_pfl["vol"]),
        (ref_children["pri"], ref_pfl["pri"]),
        (ref_children["rev"], ref_pfl["rev"]),
        (ref_children["all"], ref_pfl["all"]),
    ],
)
@pytest.mark.parametrize("how", ["inplace", "newobj"])
@pytest.mark.parametrize("addorreplace", ["add", "replace"])
def test_setchild(children: dict, how: str, addorreplace: str, expected: PfLine):
    """Test if child can be added/overwritten to a pfline."""
    *rest, (name, child) = children.items()
    to_add = {name: child}
    if addorreplace == "add":
        startchildren = dict(rest)
    else:
        startchildren = {**children, name: rest[0][1]}  # Start with repeated child.
    pfl = pf.nestedpfline(startchildren)

    if how == "inplace":
        expected = Exception

    do_test_setchild(pfl, to_add, expected, how)


@pytest.mark.parametrize(
    "children,to_add,expected",
    [
        (ref_children["vol"], children2["vol"], pfl12["vol"]),
        (ref_children["pri"], children2["pri"], pfl12["pri"]),
        (ref_children["rev"], children2["rev"], pfl12["rev"]),
        (ref_children["all"], children2["all"], pfl12["all"]),
    ],
)
@pytest.mark.parametrize("how", ["inplace", "newobj"])
@pytest.mark.parametrize("addorreplace", ["add", "replace"])
def test_setchild_overlap(
    children: Dict[str, PfLine],
    to_add: Dict[str, PfLine],
    how: str,
    addorreplace: str,
    expected: PfLine,
):
    """Test if child with partially overlapping index can be added/overwritten to a pfline."""
    if addorreplace == "add":
        startchildren = children
    else:
        to_be_overwritten = {n: c for n, c in zip(to_add.keys(), children.values())}
        startchildren = {**children, **to_be_overwritten}
    pfl = pf.nestedpfline(startchildren)

    if how == "inplace":
        expected = Exception

    do_test_setchild(pfl, to_add, expected, how)


@pytest.mark.parametrize(
    "children,to_add",
    [
        (ref_children["vol"], ref_children["pri"]),
        (ref_children["vol"], ref_children["rev"]),
        (ref_children["vol"], ref_children["all"]),
    ],
)
@pytest.mark.parametrize("how", ["inplace", "newobj"])
@pytest.mark.parametrize("addorreplace", ["add", "replace"])
def test_setchild_error_kind(
    children: Dict[str, PfLine],
    to_add: Dict[str, PfLine],
    how: str,
    addorreplace: str,
):
    """Test if child with incorrect kind cannot be added/overwritten to a pfline."""
    if addorreplace == "add":
        startchildren = children
    else:
        to_be_overwritten = {n: c for n, c in zip(to_add.keys(), children.values())}
        startchildren = {**children, **to_be_overwritten}
    pfl = pf.nestedpfline(startchildren)

    do_test_setchild(pfl, to_add, Exception, how)


@pytest.mark.parametrize(
    "children,to_add,whyerror",
    [
        (ref_children["vol"], children3["vol"], "nooverlap"),
        (ref_children["pri"], children3["pri"], "nooverlap"),
        (ref_children["rev"], children3["rev"], "nooverlap"),
        (ref_children["all"], children3["all"], "nooverlap"),
        (ref_children["vol"], children4["vol"], "otherfreq"),
        (ref_children["pri"], children4["pri"], "otherfreq"),
        (ref_children["rev"], children4["rev"], "otherfreq"),
        (ref_children["all"], children4["all"], "otherfreq"),
        (ref_children["vol"], children5["vol"], "othertz"),
        (ref_children["pri"], children5["pri"], "othertz"),
        (ref_children["rev"], children5["rev"], "othertz"),
        (ref_children["all"], children5["all"], "othertz"),
    ],
)
@pytest.mark.parametrize("how", ["inplace", "newobj"])
@pytest.mark.parametrize("addorreplace", ["add", "replace"])
def test_setchild_error(
    children: Dict[str, PfLine],
    to_add: Dict[str, PfLine],
    how: str,
    addorreplace: str,
    whyerror: str,
):
    """Test if child with non-overlapping index cannot be added/overwritten to a pfline."""
    if addorreplace == "add":
        startchildren = children
    else:
        to_be_overwritten = {n: c for n, c in zip(to_add.keys(), children.values())}
        startchildren = {**children, **to_be_overwritten}
    pfl = pf.nestedpfline(startchildren)

    do_test_setchild(pfl, to_add, Exception, how)


@pytest.mark.parametrize(
    "children",
    [
        ref_children["vol"],
        ref_children["pri"],
        ref_children["rev"],
        ref_children["all"],
    ],
)
@pytest.mark.parametrize("how", ["inplace", "newobj"])
def test_dropchild(children: Dict[str, PfLine], how: str):
    """Test if child can be deleted from a pfline."""
    *rest, (name, child) = children.items()
    pfl = pf.nestedpfline(children)
    expected = pf.nestedpfline(dict(rest))

    if how == "inplace":
        expected = Exception

    do_test_dropchild(pfl, [name], expected, how)


@pytest.mark.parametrize(
    "pfl,to_drop",
    [
        (pf.nestedpfline(ref_children["vol"]), "NonexistentChild"),
        (pf.nestedpfline({"childA": ref_children["vol"]}), "childA"),
        (pf.nestedpfline(ref_children["vol"]).flatten(), "NonexistentChild"),
    ],
)
@pytest.mark.parametrize("how", ["inplace", "newobj"])
def test_dropchild_error(pfl: PfLine, to_drop: str, how: str):
    """Test if error raised when dropping non-existing child or last child."""
    do_test_dropchild(pfl, [to_drop], Exception, how)
