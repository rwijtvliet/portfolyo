import pandas as pd

from portfolyo import tools


def freq():
    # ignore freq
    a = pd.date_range(
        "2022-04-01",
        "2024-07-01",
        freq="QS",
        tz="Asia/Kolkata",
        inclusive="left",
    )
    b = pd.date_range(
        "2021-01-01",
        "2024-01-01",
        freq="AS",
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


# tz()
# start_day()
# freq()
all()
