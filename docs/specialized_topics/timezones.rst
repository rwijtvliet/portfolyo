=========
Timezones
=========

Working with timezones can be a real headache. Here are some common scenarios and how to handle them.

.. note:: In general, a lot of problems and ambiguity can be removed by **picking a timezone and sticking to it** - and converting all incoming timeseries into this timezone.

------------
Preparations
------------

The frequency must be set or inferrable. This means that gaps are filled (e.g. with ``fr.resample().asfreq()``) and, if necessary, timestamps are localized (with ``fr.tz_localize()``). In addition, each timestamp must fall at a period boundary, i.e., when dealing with daily values, the timestamps are all at midnight, and not at e.g. noon. This is all checked by the functions below.

Finally, the timestamps should be left-bound. This cannot be checked by the functions, so it is assumed.

-------------------------
Convert to timezone-aware
-------------------------

The most common case is where we have a timezone tied to a geographic location. In that case, we can do our conversions with the ``portfolyo.force_tzaware()`` function. In this example we will use the "Europe/Berlin" timezone. Here is how we can convert any timeseries (or dataframe) ``fr`` into this timezone:

* The index already has the correct timezone set. This can be checked with ``fr.index.tz``. In this case, nothing needs to be done (and if we pass it to ``portfolyo.force_tzaware()``, it is returned as-as).

* The index has a timezone set (i.e., is "tz-aware") but to a different timezone,let's say "Japan". In that case, we need to decide if we want to convert the series while keeping the local time or while keeping the universal time.

  - Keeping the local time ("floating conversion") means that the value belonging to ``2020-01-01 15:00`` in Japan is converted to ``2020-01-01 15:00`` in Europe/Berlin.

  - Keeping the universal time ("non-floating conversion") means that the ``15:00`` value is converted to ``07:00`` in Europe/Berlin. For daily (and longer) values, this is not possible, as any given calendar day in Japan falls on two calendar days in Europe/Berlin.

  For these conversions, we use the ``portfolyo.force_tzaware()`` function and specify a value for the ``floating`` parameter.

* The index is tz-agnostic and has exactly 24h for each day. In this case, a conversion to e.g. the "Europe/Berlin" timezone is lossy if we are dealing with hourly values (and shorter) - after all,the target timezone has a missing/repeated hour at the start/end of DST. This is also done by the ``portfolyo.force_tzaware()`` function.

* The index is not tz-aware, but its timestamps still imply a certain timezone and show local time (also called "wall time") in that zone. (Again using "Europe/Berlin", this means that it contains the timestamps ``2020-03-29 00:00``, ``01:00``, ``03:00``, ``...``, as well as ``2020-10-25 00:00``, ``01:00``, ``02:00``, ``02:00``, ``03:00``, ``...``. As mentioned above, **this data cannot by converted directly** by ``portfolyo.force_tzaware()``: it must first be localized with the ``fr.tz_localize()`` method that ``pandas`` provides.


----------------------------
Convert to timezone-agnostic
----------------------------

Alternatively, we might not want to deal with timezones at all, and convert any timeseries (or dataframe) ``fr`` to be "tz-agnostic". The data is then no longer tied to a specific geographic location, and therefore do not have DST and every day has exactly 24h. In that case, we can do our conversions with the ``force_tzagnostic()`` function. Here's how we can use this function:

* The index is already "standardized". No conversion is needed (and passing it to ``force_tzagnostic()`` returns it as-is).

* The index has a timezone set. The conversion is done with ``force_tzagnostic()``. As above: if the source data has hourly (or shorter) values in a timezone with a DST-transition (like "Europe/Berlin"), the conversion is lossy.

* The index is not tz-aware, but its timestamps are the "wall time" values for a specific timezone. As above: **this data cannot be converted directly** but must first be localized with ``fr.tz_localize()``.


-----------
Assumptions
-----------

The conversion functions assume that we are dealing with "time-averable" data, such as values in [MW] or [Eur/MWh].

Two examples:

* Daily values: 10 [MW] on 2020-10-25 (which has 24h in a standardized context) is converted into 10 [MW] on 2020-10-25 in the Europe/Berlin timezone (which has 25h). If the values have a unit of [MWh] or [Eur], this is probably not what we want.

* Hourly values: a value of 4 [MW] on 2020-10-25 02:00-03:00 (standardized) is duplicated to TWO values of 4 [MW] each in the Europe/Berlin timezone. This is probably what we want, regardless of the unit.

When converting "time-summable" data, such as values in [MWh] or [Eur], do the unit conversion into [MW] or [Eur/MWh] before doing the timezone conversion.