"""Map series with quarterhourly, hourly, or daily values onto another index or year,
trying to align weekdays, holidays, and dst-changeover days. Always takes values from
same calender month (but different year)."""

import warnings

from .. import tools
from ..core.pfline import Kind, PfLine, Structure


def map_to_year(pfl: PfLine, year: int, holiday_country: str) -> PfLine:
    """Transfer the data to a hypothetical other year.

    Parameters
    ----------
    pfl : PfLine
        Portfolio line that must be mapped.
    year : int
        Year to transfer the data to.
    holiday_country : str, optional (default: None)
        Country or region for which to assume the holidays. E.g. 'DE' (Germany), 'NL'
        (Netherlands), or 'USA'. See ``holidays.list_supported_countries()`` for
        allowed values.

    Returns
    -------
    PfLine

    Notes
    -----
    Useful for daily (and shorter) data. Copies over the data but takes weekdays (and
    holidays) of target year into consideration. See ``portfolyo.map_frame_as_year()``
    for more information.
    Inaccurate for monthly data and longer, because we only have one value per month,
    and can therefore not take different number of holidays/weekends (i.e., offpeak
    hours) into consideration.
    """

    # Guard clause.
    if tools.freq.shortest(pfl.index.freq, "MS") == "MS":
        warnings.warn(
            "This PfLine has a monthly frequency or longer; changing the year is inaccurate, as"
            " details (number of holidays, weekends, offpeak hours, etc) cannot be taken into account."
        )

    # Do mapping.

    if pfl.structure is Structure.NESTED:
        return PfLine(
            {
                name: map_to_year(child, year, holiday_country)
                for name, child in pfl.items()
            }
        )

    # pfl is FlatPfLine.

    if pfl.kind is Kind.VOLUME:
        df = pfl.dataframe(
            "w"
        )  # Averageble data to allow mapping unequal-length periods
        df2 = tools.changeyear.map_frame_to_year(df, year, holiday_country)
    elif pfl.kind is Kind.PRICE:
        df = pfl.dataframe(
            "p"
        )  # Averageble data to allow mapping unequal-length periods
        df2 = tools.changeyear.map_frame_to_year(df, year, holiday_country)
    elif pfl.kind is Kind.REVENUE:
        # Assume that revenue is scales proportionately with duration of period.
        # E.g. 290 Eur in leapyear Feb --> 280 Eur in non-leapyear Feb.
        df = pfl.dataframe("r")
        df *= tools.duration.index(df.index)  # Make averageble
        df2 = tools.changeyear.map_frame_to_year(df, year, holiday_country)
        df2 /= tools.duration.index(df2.index)  # Make summable again
    else:  # CompletePfLine
        df = pfl.dataframe(["w", "p"])  # Averagable
        df2 = tools.changeyear.map_frame_to_year(df, year, holiday_country)

    return PfLine(df2)
