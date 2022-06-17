======
PfLine
======

The basic building block of the ``portfolyo`` package is the "portfolio line" (``PfLine``). Instances of this class store a timeseries containing volume information, price information, or both. This page discusses their most important properties, and how to use them. 

.. contents:: Page contents:
   :depth: 1
   :local:

----
Kind
----

An important property of a ``PfLine`` is its "kind", which tells us which kind of information it contains. There are 3 possible values: ``"p"``, ``"q"`` and ``"all"``.

* ``.kind == "p"`` 

  This is a portfolio line which only contains price information.

  E.g.: the gas forward price curve for a certain market area, in daily resolution, for tomorrow until the end of the frontyear+2.

* ``.kind == "q"``

  This is a portfolio line that only contains volume information.

  E.g.: the expected offtake volume of a certain portfolio, for the coming calender year, in quarterhourly resolution. 

  The volume in each timestamp can be retrieved by the user in units of energy (e.g., MWh) or in units of power (e.g., MW).

* ``.kind == "all"``

  This a portfolio line that contains both price and volume information.

  E.g.: the sourced volume of the same portfolio, again for the coming calender year but in monthly resolution. 

  For each timestamp we have a volume (the contracted volume, both as energy and as power), a price (for which the volume was contracted) and a revenue (i.e., the multiplication of the energy and the price). 

Under the hood, not all information that can be retrieved by the user is stored; redundant information is discarded and recalculated whenever necessary. For the volume, for example, only the energy is stored. The power can be calculated by dividing the energy (in MWh) by the duration of the timestamp (in h).

----------------
Single and Multi
----------------

Two classes exist that implement a portfolio line: ``SinglePfLine`` and ``MultiPfLine``. Both are descendents of the ``PfLine`` base class. The difference between them is that the latter is a *collection* of one or more child portfolio lines. 

For example, if a price portfolio line contains a single price value for each timestamp, it is a ``SinglePfLine``. If, however, the price has several components that must be added together, these can be stored as children of a ``MultiPfLine``. Similarly, if sourced volume is split into e.g. forward and spot, these can be stored as children of a ``MultiPfLine``.

#TODO: add reference to tutorial where children are shown.

.. note::

   In practice, the methods and properties of these classes are very similar, and when initialising an instance of the ``PfLine`` class (of which ``SinglePfLine`` and ``MultiPfLine`` are descendents) the correct object type is automatically returned. 

Unless explicitly specifying one of its children, a ``MultiPfLine`` always shows the aggregate values. If the child values are no longer of interest, the ``.flatten()`` method returns the equivalent ``SinglePfLine``.

--------------
Initialisation
--------------

There are many ways to specify the timeseries from which to initialise a portfolio line; here we will discuss the most common ones. In all cases it is assumed that :doc:`the data has been prepared and standardized<../specialized_topics/dataprep>`.

In General
==========

``portfolyo`` tries to determine the dimension of information (e.g., if it is a price or a volume) using its key (if it has one) and its unit (also, if it has one) - see the section on :ref:`Compatilibity of abbrevation and unit <nameunitcompatibility>`.

To initialise a price-and-volume portfolio line (i.e., with ``.kind == "all"``), we must supply at least 2 of the following timeseries: prices, volumes, revenues. To initialise a price-only portfolio line, we must only provide supply price timeseries. To initialise a volume-only portfolio line, we must only provide a volume timeseries.

.. note::
   
   Regarding volumes: It is not necessary to specify both ``w`` and ``q``; for a given timestamp we can calculate the energy from its power and vice versa.  If both *are* specified, they should be consistent with each other. 

   The same goes when volume (either ``w`` or ``q``), price (``p``) and revenue (``r``) are specified.


DataFrame or dictionary of timeseries
=====================================

Or any other ``Mapping`` from (string) key values to ``pandas.Series``. The keys (dataframe column names) must each be one of the following: ``w`` (power), ``q`` (energy), ``p`` (price), ``r`` (revenue). Depending on the keys, the ``.kind`` of the portfolio line is determined.

.. exec_code::
   
   import portfolyo as pf, pandas as pd
   index = pd.date_range('2024', freq='AS', periods=3)
   input_dict = {'w': pd.Series([200, 220, 300], index)}
   pfl = pf.PfLine(input_dict)
   print(pfl)

Timeseries with unit
====================

To create a price-only or volume-only portfolio line, we may provide a ``pandas.Series`` with a valid unit for volume or price. It is automatically converted to the default unit.

.. exec_code::

   import portfolyo as pf, pandas as pd
   index = pd.date_range('2024', freq='AS', periods=3)
   input_series = pd.Series([10, 11.5, 10.8], index, dtype='pint[ctEur/kWh]')
   pfl = pf.PfLine(input_series)
   print(pfl)
   

Dictionary of portfolio lines
=============================

Or any other ``Mapping`` from (string) key values to ``PfLine`` objects. The keys are used as the children names. 

.. exec_code::

   import portfolyo as pf, pandas as pd
   index = pd.date_range('2024', freq='AS', periods=3)
   pfl1 = pf.PfLine(pd.Series([0.2, 0.22, 0.3], index, dtype='pint[GW]')) 
   pfl2 = pf.PfLine(pd.Series([100, 150, 200.0], index, dtype='pint[MW]')) 
   input_dict = {'southeast': pfl1, 'northwest': pfl2}
   pfl = pf.MultiPfLine(input_dict) #TODO: change to pf.PfLine
   print(pfl)

Note that the sum values are shown. The individual children's portfolio lines may be accessed using their name as an index, e.g. ``pfl['southeast']``.  

Nesting is not limited to one level, and instead of having each value be a ``PfLine`` objects, it is actually sufficient that each value can be used to initialise a ``PfLine`` object. 

   




---
API
---



.. autoclass:: portfolyo.PfLine
   :members:
   :inherited-members: 