========
Tutorial
========

In this tutorial we will take a deep dive into working with the ``portfolyo`` package.

----------
Input data
----------

The starting point for our tutorial is raw timeseries data in form of a ``pandas`` ``Series`` or ``DataFrame``. 

For real-life applications, the input data will likely be read from an Excel workbook, database, or other data service. Before this data can be used, it must be "standardized" so that it has a certain format; most importantly it must have a gapless ``DatetimeIndex`` with left-bound timestamps in a certain frequency. These timestamps are either "timezone-aware" (= localized to a certain geographic timezone), or "timezone-agnostic" (= explicitly not tied to a location or timezone).

For more information on standardizing input data, see :doc:`../specialized_topics/dataprep`.

For this tutorial, we'll use the all-in-one function ``portfolyo.standardize()`` to get the input data in the correct format. 

