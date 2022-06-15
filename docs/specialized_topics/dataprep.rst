========================
Preprocessing input data
========================

User-supplied input data should adhere to certain specifications, and depending on our data source, some cleanup or manipulation might be necessary.

.. contents:: Page contents:
   :depth: 1
   :local:

``portfolyo`` mainly works with ``pandas Series`` and ``pandas DataFrame`` objects as input data. Let's say we have an object ``fr`` as input data, which may be either; it should:

* have a ``DatetimeIndex``, 
* with left-bound timestamps,
* which is either timezone-agnostic or localized to a certain geographic timezone,
* and gapless,
* and has a quarterhourly, hourly, daily, monthly, quarterly or yearly frequency.

Below are some common operations to ensure these characteristics. The are best done in the order presented. For convenience there is also a ``portfolyo.standardize()`` function, which can be used to do all necessary preprocessing in the majority of cases.


-------------
DatetimeIndex
-------------

The index of ``fr`` must be a ``pandas.DatetimeIndex``. Each timestamp in the index describes a *period* in time. (The reason a ``pandas.PeriodIndex`` is not used for this, is that these cannot handle timezones that use daylight-savings time.) 

If we are dealing with a DataFrame whose timestamps are not in the index but in one of the columns, we can use the ``fr.set_index()`` method, like so:

.. code-block:: python

    >>> fr = fr.set_index("columname")

If the index contains the correct timestamps but e.g. as strings, we can create a ``DatetimeIndex`` like so:

.. code-block:: python

    >>> fr.index = pandas.DatetimeIndex(fr.index)

-------------------
Localize input data
-------------------

There are several thing to consider when working with timezones; the first is localization. Localization is necessary when the index (a) is not timezone-aware (meaning, that ``fr.index.tz`` is ``None``; the timestamps show the local ("wall") time), but (b) *does* assume a specific timezone. 

A common example is hourly (or shorter) data, in timezones with daylight-savings time, like "Europe/Berlin". A non-localized timeseries will have a missing/repeated hour at the start/end of DST: it contains the timestamps ``2020-03-29 00:00``, ``01:00``, ``03:00``, ``...``, as well as ``2020-10-25 00:00``, ``01:00``, ``02:00``, ``02:00``, ``03:00``, ``...``.

.. note:: This is often the case when reading data from Excel, as Excel cannot handle timezones or UTC-offsets. 

The timezone can be added to the data with the ``fr.tz_localize()`` method:

.. code-block:: python

    >>> fr = fr.tz_localize("Europe/Berlin", ambiguous="infer")

This makes the index unambiguous, with a UTC-offset on each timestamp (e.g. the two ``2020-10-25 02:00:00`` timestamps become ``2020-10-25 02:00:00+02:00`` and ``2020-10-25 02:00:00+01:00``).

.. warning:: Localization is only necessary in the situation mentioned above: if the input data has **local ("wall") timestamps** for a **specific geographic location**.

Note that localization is not necessary in the following cases:

* The index is already localized (``fr.index.tz`` is not ``None``). In this case, ``fr.tz_localize`` raises a ``TypeError``.

* The index is "standardized timezone-agnostic", i.e., with 24h per calendar day and no DST-transitions. In this case, the data is not tied to a specific geographic location and therefore should not be localized. [#f1]_

Don't worry if our data does not yet have the timezone we want to use in our application, we will take care of that further below.

---------
Frequency
---------

The index must have a frequency (``fr.index.freq``); it must be one of the ones in ``portfolyo.FREQUENCIES``. The following abbreviations are used by ``pandas`` and throughout this package:

* ``15T``: quarterhourly; 
* ``H``: hourly;
* ``D``: daily;
* ``MS``: monthly;
* ``QS``: quarterly;
* ``AS``: yearly.

If the frequency is not set, we can try to make pandas infer it:

.. code-block:: python

    >>> fr.index.freq = pandas.infer_freq(fr.index)

However, if ``fr.index.freq`` is still ``None`` after this, or if an error is raised, it is because of one of these reasons:

Too few datapoints
------------------
If there is only one timestamp in the index, e.g., ``2020-01-01 0:00``, it is impossible for ``pandas.infer_freq`` to know if this represents an hour, a day, or the entire year. In this case, we can manually set the frequency with e.g.:

.. code-block:: python
    
    >>> fr.index.freq = "D"

Gaps in data
------------
If the index has gaps - e.g., it has timestamps for Jan 5, Jan 6, Jan 7, and Jan 10, the frequency can also not be determined. In this case, if we are dealing with daily values, the Jan 8 and Jan 9 timestamps need to be inserted, e.g. with:

.. code-block:: python
    
    >>> fr = fr.resample("D").asfreq()

Because their values are unknown, these timestamps get a ``numpy.nan`` value. (We could use the ``portfolyo.fill_gaps()`` function to do a linear interpolation and fill the gaps.)

.. _righttoleft:

------------------------
Left-bound DatetimeIndex
------------------------

Another assumtion is that timestamps in the index must be at the *start* of their periods. E.g., if we have hourly values, the timestamp with time ``04:00`` must describe the hour starting at 04:00 (i.e., 04:00-05:00) and not the hour ending at 04:00 (i.e.,03:00-04:00).

If the index has right-bound timestamps, we can convert it to the wanted left-bound format with ``portfolyo.right_to_left()``:

.. code-block:: python

    >>> fr.index = portfolyo.right_to_left(fr.index)

---------------
Target timezone
---------------

Finally, we can convert our input data into the timezone we want to use throughout our application. We choose either (a) to make all data timezone-aware and localized to the same geographic location, or (b) to make all data timezone-agnostic, with 24h for each day, without DST-transitions. 

We can use one of the timezone conversion functions to do this:

.. code-block:: python

    >>> fr = portfolyo.force_tzaware(fr, tz="Europe/Berlin")

or 

.. code-block:: python

    >>> fr = portfolyo.force_tzagnostic(fr)

Note that these conversions are not always lossless, in which case assumptions are made abouth the data. For more details, see :doc:`../specialized_topics/timezones`.

-----------
In one step
-----------

If the input data has no gaps, and a frequency that is either set or can be inferred, then we can do all of the above operations with one call to the ``portfolyo.standardize()`` function, i.e.:

.. code-block:: python

    >>> fr = portfolyo.standardize(fr, 'aware', tz='Europe/Berlin', bound='right')

This function also tries to localize ``fr`` if it is not timezone-aware, and the frequency cannot be inferred.


 

.. rubric:: Footnotes

.. [#f1] However, there is no harm in doing the localization to thee target timezone if it is possible. In specific situations, localization is not possible (if we (a) have (quarter)hourly values that we (b) want to localize to a timezone with daylight-savings-time such as "Europe/Berlin" and (c) the moment of the DST-transition is included in the input data) and ``fr.tz_localize()`` raises a ``NonExistentTimeError`` or a ``AmbiguousTimeError``. 