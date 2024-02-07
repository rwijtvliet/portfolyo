from typing import Iterable, List, Union
import pandas as pd

from portfolyo import tools


def get_idx(
    startdate: str,
    starttime: str,
    tz: str,
    freq: str,
    enddate: str,
) -> pd.DatetimeIndex:
    # Empty index.
    if startdate is None:
        return pd.DatetimeIndex([], freq=freq, tz=tz)
    # Normal index.
    ts_start = pd.Timestamp(f"{startdate} {starttime}", tz=tz)
    ts_end = pd.Timestamp(f"{enddate} {starttime}", tz=tz)
    return pd.date_range(ts_start, ts_end, freq=freq, inclusive="left")


def get_frames(
    idxs: Iterable[pd.DatetimeIndex], ref_idx: pd.DatetimeIndex = None
) -> List[Union[pd.Series, pd.DataFrame]]:
    frames = []
    for i, idx in enumerate(idxs):
        # Get data.
        if ref_idx is None:
            startnum = 0
            index = idx
        else:
            startnum = idx.get_loc(ref_idx[0]) if len(ref_idx) > 0 else 0
            index = ref_idx
        # Create series.
        fr = pd.Series(range(startnum, startnum + len(index)), index, dtype=int)
        # Possibly turn into dataframe.
        if i % 2 == 0:
            fr = pd.DataFrame({"a": fr})
        frames.append(fr)
    return frames


def freq():
    # ignore freq
    a = pd.date_range(
        "2020-01-01",
        "2023-01-01",
        freq="15T",
        inclusive="left",
    )
    b = pd.date_range(
        "2020-01-20",
        "2023-01-01",
        freq="H",
        inclusive="left",
    )
    intersect = tools.intersect.indices_flex(a, b, ignore_freq=True, ignore_tz=True)
    print(intersect)


def start_day():
    a = pd.date_range(
        "2022-04-01 00:00", "2022-05-10 00:00", freq="D", inclusive="left"
    )
    b = pd.date_range(
        "2022-04-01 06:00", "2022-07-15 06:00", freq="D", inclusive="left"
    )
    intersect = tools.intersect.indices_flex(a, b, ignore_start_of_day=True)
    print(intersect)


def tz():
    a = pd.date_range(
        "2022-04-01 00:00",
        "2022-05-10 00:00",
        freq="H",
        tz="Europe/Berlin",
        inclusive="left",
    )
    b = pd.date_range(
        "2022-04-25 00:00",
        "2022-05-15 00:00",
        freq="H",
        tz="Europe/Berlin",
        inclusive="left",
    )
    intersect = tools.intersect.indices_flex(
        a, b, ignore_tz=True, ignore_start_of_day=True
    )
    print(intersect)


def all():
    a = pd.date_range(
        "2022-01-01 00:00",
        "2023-01-01 00:00",
        freq="15T",
        tz="Asia/Kolkata",
        inclusive="left",
    )
    b = pd.date_range(
        "2022-01-20 06:00",
        "2023-01-01 06:00",
        freq="H",
        tz=None,
        inclusive="left",
    )
    intersect = tools.intersect.indices_flex(
        a, b, ignore_freq=True, ignore_tz=True, ignore_start_of_day=True
    )
    print(intersect)


def frame():
    idxs = [
        get_idx("2020-01-01", "00:00", None, "15T", "2022-02-02"),
        get_idx("2020-01-20", "00:00", "Europe/Berlin", "15T", "2022-02-02"),
    ]
    frames = get_frames(idxs)

    # Normal case.
    result_frames = tools.intersect.frames(
        *frames,
        ignore_start_of_day=False,
        ignore_tz=True,
        ignore_freq=False,
    )
    expected_frame_a = get_frames(idxs, idxs)
    expected_frame_b = get_frames(idxs, None)
    expected_frames = [expected_frame_a, expected_frame_b]
    print(result_frames)
    print(expected_frames)


# tz()
# start_day()
# freq()
# all()
frame()
