=======
Indices
=======
   
Because not all energy markets define delivery periods in the same way, we need to be somewhat flexible with handling timeseries.

.. contents:: Page contents:
   :depth: 1
   :local:

------------------------------
Calender and non-calendar days
------------------------------

Calendar days
-------------

* A "day" is normally assumed to be the time period from midnight to midnight. In most cases, this is 24h; in timezones that observe daylight-saving-time, each year also has one 23h-day and one 25h-day.

* Equivalently, a "month", "quarter" or "year" are the time periods from midnight on the first day of the month/quarter/year until midnight on the first day of the following month/quarter/year.

* Upsampling daily-or-longer timeseries to hours or quarterhours results in a timeseries of which the first (quarter)hour starts at midnight, and the final (quarter)hour ends at midnight of the following day.

We say that the "offset" of this market is 0 hours (past midnight).

Non-calendar days
-----------------

In some markets, however, the start of a day does not line up with the calendar.

* In the German Gas market, for example, a "day" is defined as the time period from 06:00 until 06:00. The day called '21 April' is actually the time period from 06:00 on 21 April, until 06:00 on 22 April.

* Months, quarters and years are then also starting at 06:00 on the first calendar day of the month/quarter/year, and ending at 06:00 on the first calendar day of the following month/quarter/year.

* Upsampling to hours or quarterhours gives a timeseries of which the first element starts at 06:00, and the final one ends at 06:00.

In this case, the "offset" of this market is 6 hours (past midnight).

----------------------------
Assumptions regarding offset
----------------------------

When handling timeseries with daily-or-longer indices, the offset is clear from the timestamps. For example, the following daily index has a 6-hour offset:

.. exec_code::

   # --- hide: start --- 
   import pandas as pd
   print(repr(pd.date_range('2024-01-05 06:00', freq='D', periods=4)))

This remains true during a DST-transition:

.. exec_code::

   # --- hide: start --- 
   import pandas as pd
   print(repr(pd.date_range('2024-03-29 06:00', freq='D', periods=4, tz='Europe/Berlin')))

However, when handling timeseries with shorter-than-daily indices, the offset cannot be deduced. *In this case, it is assumed that the index starts with the first (quarter)hour in the day.* Considering the following index:

.. exec_code::

   # --- hide: start ---
   import pandas as pd
   i = pd.date_range('2024-01-05 06:00', freq= 'H', periods = 40)
   print(repr(i))

It contains 40 hourly values, i.e., it does not contain an integer number of days. We cannot infer, how a day is defined, and in these situations it is therefore assumed, that it is the offset of the first timestamp - '06:00' in this case.

DST and daily-or-longer indices
-------------------------------

If the market under consideration does not operate on shorter-than-daily frequencies, you might be tempted to disregard the offset and use *calendar* days/months/quarters/years. E.g., you'd use the "21 April 00:00" timestamp to indicate the time period "21 April 06:00 - 22 April 06:00".

Unfortunately, that causes problems during an DST-transition. If the start of DST is on 30 March, and the clock misses the 02:00-03:00 hour, then it is the 29 March 06:00 - 30 March 06:00 delivery period that is only 23h long. Using calendar days, however, the missing hour would be in the 30 March calendar day, which causes errors (e.g. when calculating energies from powers etc.)

The offset of a market is therefore also relevant when working with daily indices.