==========
Resampling
==========

When resampling, i.e, changing the frequency of a timeseries (e.g., from daily to monthly values or vice versa) the dimension of the data prescribes how this should be done. Here we consider what rules apply to power, energy, price, and revenue data.

.. contents:: Page contents:
   :depth: 1
   :local:

-------------
Time-summable
-------------

So-called (by me, at least) "time-summable" values are values that only make sense when the *duration* of the time period they apply to is also specified. For an intuitive example: if I tell you, that I have earned 10 000 Eur, you won't know whether to congratulate or pity me, unless I tell you if that number applies to a year or a week.

Time-summable values cannot be expressed as a continuous function *f(t)*. Instead, they only make sense as a *discrete list*, with each value applies to an entire time period, but not to any one moment within that time period. If we imagine dividing the time period up into increasingly small intervals, the value that applies to each of them eventually approaches 0.

To give another example of a time-summable quantity: consider the distances traveled by a car during various time intervals. This quantity has several characteristics. Firstly, the values do not apply to *moments* in time - to travel a distance, we need a *period* of time. Secondly, in order to judge if the distances are small or large, we must consider how much time they were driven in. Thirdly, if we combine several time periods, the distances can (must) be summed to get the correct distance in the total period.

Relevant to our context is that the dimensions **energy** and **revenue** are time-summable.

---------------
Time-averagable
---------------

For "time-averagable" values the reverse is true. They *can* be judged without knowing the duration of the corresponing time period. For an example, consider my hourly salary. Its value is not affected by me working 1 hour, 5 hours, or 0.5 hours.

Time-averagable values *can* be thought of as continuous functions. In practice, we might still deal with discrete values that apply to a certain time period - however, if we imagine dividing the time period up into increasingly small intervals, the value that applies to each of them eventually settles on a non-zero value.

To give another example of a time-averagable quantity: consider the velocity of a car. This quantity has several characteristics that are different from the distance described above. Firstly, the velocity can be measured at every moment in time. We cannot directly measure the velocity during a period of time, we can only calculate it as the *average* value. Secondly, velocity values can be compared directly, even if they apply to time periods with unequal durations. Thirdly, if we combine several time periods, the velocities must be averaged (weighted with their durations) to get the correct velocity during the total period.

Relevant to our context is that the dimensions **power** and **temperature** are time-averagable. **Price** is time-averagable under certain circumstances.

-------
Derived
-------

Some values do not fall in either category, and are calculated from other values instead. 

Relevant to our context: the dimension **price** is, in those circumstances where it applies to a certain volume, always equal to the revenue divided by the volume. This means that, when combining the prices of several time periods, we need to average the individual values, but weighted with the **energy** in each period, not the duration ("volume-weighted").

If a price does *not* apply to a certain volume, e.g., when considering current market prices for various delivery periods, it must be treated as time-averagable - see the final remark of this page.

------------------
Upsampling example
------------------

If, in the year 2024, a customer has an energy offtake of 1 GWh (``q = 1000``), i.e., with an average power of 0.11 MW (``w = 0.11``), and pays a price of 30 Eur/MWh (``p = 30``), then the revenue is 30 kEur (``r = 30000``). Let's say the average temperature in the year is 7.98 degC (``t = 7.98``):

=========================  ========  ======  =======  =======  ====
Yearly                            w       q        p        r     t
resolution                       MW     MWh  Eur/MWh      Eur  degC
=========================  ========  ======  =======  =======  ====
2024-01-01 00:00:00+01:00  0.113843  1000.0     30.0  30000.0  7.98
=========================  ========  ======  =======  =======  ====

If we resample these values to a higher-frequency timeseries (e.g. quarteryearly), then the values of the summable dimensions (``q`` and ``r``) become smaller, as their values need to be "spread" over the resulting rows. If nothing more is known about how the energy is consumed, we assume that the consumption rate is constant throughout the period. This means we have to distribute the values over the new rows, **in proportion to their duration**. (Because the 3rd and 4th quarter have more days than the 1st and 2nd quarter, they get a larger fraction of the original value.)

The values of the averagable dimensions (``w`` and ``t``) are **unchanged**, i.e., they are simply copies of the original value. Also the value of the derived quantity ``p`` turns out to be unchanged. The resulting values are therefore:

=========================  ========  ======  =======  =======  ====
Upsampled to quarterly            w       q        p        r     t
resolution                       MW     MWh  Eur/MWh      Eur  degC
=========================  ========  ======  =======  =======  ====
2024-01-01 00:00:00+01:00  0.113843  248.52     30.0  7455.60  7.98
2024-04-01 00:00:00+02:00  0.113843  248.63     30.0  7459.02  7.98
2024-07-01 00:00:00+02:00  0.113843  251.37     30.0  7540.98  7.98
2024-10-01 00:00:00+02:00  0.113843  251.48     30.0  7544.40  7.98
=========================  ========  ======  =======  =======  ====

This is the best guess we can make without using any additional information about how the values are distributed throughout the year. Note that each row is consistent, i.e., ``q`` equals ``w`` times the duration in hours, and ``r`` equals ``p`` times ``q``. 

--------------------
Downsampling example
--------------------

Something similar happens when going in the reverse direction, but a bit more intricate. Let's start with the following quarteryearly values:

=========================  ========  ======  =======  =======  ====
Quarterly                         w       q        p        r     t
resolution                       MW     MWh  Eur/MWh      Eur  degC
=========================  ========  ======  =======  =======  ====
2024-01-01 00:00:00+01:00  0.137426   300.0    37.77  11330.1   1.3
2024-04-01 00:00:00+02:00  0.082418   180.0    25.30   4554.0  12.3
2024-07-01 00:00:00+02:00  0.090580   200.0    21.30   4260.0  15.1
2024-10-01 00:00:00+02:00  0.144862   320.0    30.80   9856.0   3.2
=========================  ========  ======  =======  =======  ====

If we resample to a lower-frequency timeseries (e.g. yearly), we need to **sum** the values of the summable dimensions ``q`` and ``r`` (the duration does not need to be considered). 

For the time-averagable dimensions (``w`` and ``t``), the **average** of the individual values must be calculated, **weighted with the duration** of each row. (Alternatively, for the power ``w``: this is always ``q/duration`` and can always be calculated from these values after *they* are downsampled.)

For the derived dimension ``p``, this is also an average of the individual values, but weighted with the volume ``q`` of each row. (Alternatively: the price is always ``r/q`` and can always be calculated from these values after *they* are downsampled.)

The resulting downsampled values are:

=========================  ========  ======  =======  =======  ====
Downsampled to yearly             w       q        p        r     t
resolution                       MW     MWh  Eur/MWh      Eur  degC
=========================  ========  ======  =======  =======  ====
2024-01-01 00:00:00+01:00  0.113843  1000.0     30.0  30000.0  7.98
=========================  ========  ======  =======  =======  ====

(Note that the 'simple row-average' of the power, temperature, and price give us incorrect values.)

--------------------------------
Downsampling example, price-only
--------------------------------

To illustrate the point that downsampling prices is different when we have "no volume information", consider the price values from the previous example, but let's assume they represent the futures base price for each quarter:

=========================  =======
Quarterly                        p
resolution                 Eur/MWh
=========================  =======
2024-01-01 00:00:00+01:00    37.77
2024-04-01 00:00:00+02:00    25.30
2024-07-01 00:00:00+02:00    21.30
2024-10-01 00:00:00+02:00    30.80
=========================  =======

How high is the (arbitrage-free) base price for the entire year?

In this case, price is treated as time-averagable and weighted with the *duration* of each period. We obtain a slightly lower value than before:

=========================  =======
Downsampled to yearly            p
resolution                 Eur/MWh
=========================  =======
2024-01-01 00:00:00+01:00    28.78
=========================  =======

The reason for the higher price in the previous example, is that, there, it is weighted with the *energy* in each period. We had more energy in the expensive quarters, and less in the cheaper ones, which results in a higher price for the entire year.

-----------------------------
Resampling with ``portfolyo``
-----------------------------

When changing the frequency of a ``PfLine`` or ``PfState`` object, the considerations above are automatically taken into account. If you are in the situation of having to change the frequency of a ``pandas.Series`` or ``DataFrame`` with a ``DatetimeIndex``, however, the relevant functions are also available at the ``portfolyo.changefreq`` module.