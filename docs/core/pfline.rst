======
PfLine
======

The basic building block of the ``portfolyo`` package is the "portfolio line" (``PfLine``). Instances of this class store a timeseries containing volume information, price information, or both. This page discusses their most important properties, and how to use them. 

It is assumed that you are familiar with the following dimension abbrevations: ``w`` for power, ``q`` for energy, ``p`` for price, and ``r`` for revenue; see :doc:`this page for more information <../specialized_topics/dimensions>`.

.. contents:: Page contents:
   :depth: 1
   :local:

----
Kind
----

An important characteristic of a ``PfLine`` is its "kind", which tells us which kind of information it contains. It is available as the property ``.kind`` and has one of 3 possible values: ``"p"``, ``"q"`` and ``"all"``.

* ``.kind is Kind.PRICE_ONLY`` 

  This is a portfolio line which only contains price information. 
  
  As an examples, consider the forward price curve for a certain market area, or the fixed offtake price that a customer is paying for a certain delivery period.

* ``.kind is Kind.VOLUME_ONLY``

  This is a portfolio line that only contains volume information. 
  
  For example: the expected/projected development of offtake volume of a customer or customer group.

  The volume in each timestamp can be retrieved by the user in units of energy (e.g., MWh) or in units of power (e.g., MW).

* ``.kind is Kind.ALL``

  This a portfolio line that contains both price and volume information. 
  
  For example: the volume that has been sourced to hedge a certain portfolio, e.g. in monthly or quarterly blocks. 

  For each timestamp we have a volume (the contracted volume, both as energy and as power), a price (for which the volume was contracted) and a revenue (i.e., the multiplication of the energy and the price). 

Under the hood, not all information that can be retrieved by the user is stored; redundant information is discarded and recalculated whenever necessary. For the volume, for example, only the energy is stored. The power can be calculated by dividing the energy (in MWh) by the duration of the timestamp (in h).

--------------
Initialisation
--------------

There are many ways to specify the timeseries from which to initialise a portfolio line; here we will discuss the most common ones. In all cases it is assumed that :doc:`the data has been prepared and standardized<../specialized_topics/dataprep>`.

In General
==========

``portfolyo`` tries to determine the dimension of information (e.g., if it is a price or a volume) using its key (if it has one) and its unit (also, if it has one) - see the section on :ref:`Compatilibity of abbrevation and unit <nameunitcompatibility>`.

To initialise a price-only portfolio line, we must only provide supply price timeseries. To initialise a volume-only portfolio line, we must only provide a volume timeseries. To initialise a price-and-volume portfolio line (i.e., with ``.kind is Kind.ALL``), we must supply at least 2 of the following timeseries: prices, volumes, revenues. 

.. note::
   
   Specifying volumes: it is not necessary to specify both power (``w``) and energy (``q``); for a given timestamp we can calculate one from the other. If both *are* specified, they should be consistent with each other. 

   Specifying volumes and prices: the same goes when specifying volume (either ``w`` or ``q``), price (``p``) and revenue (``r``). Only two are needed, and if more *are* specified, they should be consistent.


DataFrame or dictionary of timeseries...
========================================

or any other ``Mapping`` from (string) key values to ``pandas.Series``. 

The keys (dataframe column names) must each be one of the following: ``w`` (power), ``q`` (energy), ``p`` (price), ``r`` (revenue). Depending on the keys, the ``.kind`` of the portfolio line is determined.

.. exec_code::
   
   import portfolyo as pf
   import pandas as pd
   index = pd.date_range('2024', freq='AS', periods=3)
   input_dict = {'w': pd.Series([200, 220, 300], index)}
   pf.PfLine(input_dict)
   # --- hide: start ---
   print(repr(pf.PfLine(input_dict)))

Timeseries with unit
====================

Under the condition that a valid ``pint`` unit is present, we may also provide a single timeseries (``pandas.Series``), or an iterable of timeseries. They are automatically converted to the default unit as can be seen in this example:

.. exec_code::

   # --- hide: start ---
   import portfolyo as pf, pandas as pd
   index = pd.date_range('2024', freq='AS', periods=3)
   # --- hide: stop ---
   # using the imports and index from the previous example
   input_series = pd.Series([10, 11.5, 10.8], index, dtype='pint[ctEur/kWh]')
   pf.PfLine(input_series)
   # --- hide: start ---
   print(repr(pf.PfLine(input_series)))
   

Dictionary of portfolio lines...
================================

or any other ``Mapping`` from (string) key values to ``PfLine`` objects. 

The keys are used as the children names: 

.. exec_code::

   # --- hide: start ---
   import portfolyo as pf, pandas as pd
   index = pd.date_range('2024', freq='AS', periods=3)
   # --- hide: stop ---
   pfl1 = pf.PfLine(pd.Series([0.2, 0.22, 0.3], index, dtype='pint[GW]')) 
   pfl2 = pf.PfLine(pd.Series([100, 150, 200.0], index, dtype='pint[MW]')) 
   dict_of_children = {'southeast': pfl1, 'northwest': pfl2}
   pfl = pf.MultiPfLine(dict_of_children)
   # --- hide: start ---
   print(repr(pf.MultiPfLine(dict_of_children))) #TODO: change to pf.PfLine

Note that the sum values are shown. 

Nesting is not limited to one level, and, instead of having each value be a ``PfLine`` objects, it is actually sufficient that each value can be used to initialise a ``PfLine`` object. 
  

------------------------
With or without children
------------------------

As seen in the final initialisation example, we can create a nested ``PfLine`` instance, which has named children that are also ``PfLine`` instances. This in contrast to the 'childless' instance we created in the first initialisation example. 

It is important to note that both instances contain the exact same methods and properties that are described in the rest of this document. 

.. note:: When checking the types, you will see that these are actually different classes. The childless portfolio line is an instance of the ``SinglePfLine`` class, whereas the other portfolio line is an instance of the ``MultiPfLine`` class. Both are descendents of the ``PfLine`` base class.  When initialising a ``PfLine``, the correct type will be returned based on the input data. 

For nested portfolio lines it is important to know that, unless explicitly stated by the user, we are looking at and working with the the top level, i.e., the sum/aggregate. 

If we want to access the children, we can use their name as an index, e.g. ``pfl['southeast']``, or we can iterate over the object with ``for childname in pfl`` or ``for (name, child) in pfl.items()``.

An example of where a nested portfolio line makes sense, is to combine procurement on the forward market and spot trade into a single "sourced" portfolio line. We can then easily work with the aggregate values, but the values of the individual markets are also still available.

If we are no longer interested in the children, we can keep only the top-level information with the ``.flatten()`` method.


--------------
Accessing data
--------------

In order to get our data out of a ``PfLine`` instance, the following options are available.

Index
=====

The ``.index`` property returns the ``pandas.DatetimeIndex`` that applies to the data. This includes the frequency that tells us how long the time periods are that start at each of the timestamps in the index. 

.. exec_code::

   # --- hide: start --- 
   # --- hide: stop ---
   import portfolyo as pf, pandas as pd
   index = pd.date_range('2024', freq='AS', periods=3)
   input_df = pd.DataFrame({'w':[200, 220, 300], 'p': [100, 150, 200]}, index)
   pfl = pf.PfLine(input_df)

   pfl.index
   # --- hide: start ---
   print(repr(pfl.index))

For convenience, ``portfolyo`` adds a ``.duration`` and a ``ts_right`` property, which do as we would predict:

.. exec_code::

   # --- hide: start --- 
   import portfolyo as pf, pandas as pd
   index = pd.date_range('2024', freq='AS', periods=3)
   input_df = pd.DataFrame({'w':[200, 220, 300], 'p': [100, 150, 200]}, index)
   pfl = pf.PfLine(input_df)
   # --- hide: stop ---
   # continuation from previous code example
   pfl.index.duration, pfl.index.ts_right
   # --- hide: start ---
   print(repr(pfl.index.duration),'\n', repr(pfl.index.ts_right))



Timeseries
==========

The properties ``.w``, ``.q``, ``.p`` and ``.r`` always return the information as a ``pandas.Series``. These have a ``pint`` unit, which can be stripped using ``.pint.m`` (or ``.pint.magnitude``).

.. exec_code::

   # --- hide: start --- 
   import portfolyo as pf, pandas as pd
   index = pd.date_range('2024', freq='AS', periods=3)
   input_df = pd.DataFrame({'w':[200, 220, 300], 'p': [100, 150, 200]}, index)
   pfl = pf.PfLine(input_df)
   # --- hide: stop ---
   # continuation from previous code example
   pfl.r
   # --- hide: start ---
   print(repr(pfl.r))

.. Comment: the output of the line above may contain a central dot "pint[MW·h]" which may cause some encoding problems for the output.

#TODO: Link to reference for more information

DataFrame
=========

If we want to get more than one timeseries out, we can use the ``.df()`` method, which has several options to control the exact format and contents of the dataframe. 

.. exec_code::

   # --- hide: start ---
   import portfolyo as pf, pandas as pd
   index = pd.date_range('2024', freq='AS', periods=3)
   input_df = pd.DataFrame({'w':[200, 220, 300], 'p': [100, 150, 200]}, index)
   pfl = pf.PfLine(input_df)
   # --- hide: stop ---
   # continuation from previous code example
   pfl.df(units=False)
   # --- hide: start ---
   print(f'\n{str(pfl.df(units=False))}')
   # --- hide: stop ---


#TODO: Link to reference for more information

Price- or Volume-only
=====================

With the ``.price`` and ``.volume`` properties, we are able to extract a ``"p"``- or ``"q"``-``PfLine``. The returned ``PfLine`` has no children, as this would lead to incorrect values in some situations (in case we want to get the price part of a ``"all"-PfLine`` - in that case, the aggregate price is the average of the children's prices, weighted with their volumes - which we lose if we keep only the prices).

----------------------
Operations: Arithmatic
----------------------

There are many ways to change and interact with ``PfLine`` instances. An intuitive way is through arithmatic. The following operations are currently defined/implemented. Unless specified otherwise, the returned object is of the same kind, and has the same children, as the original. 

====================== ========================== ================== ================== ================== ================== ================== ==================
Kind                                              Price-only porfolio line (``"p"``)    Volume-only portfolio line (``"q"``)  Price and volume portfolio line (``"all"``)
------------------------------------------------- ------------------------------------- ------------------------------------- -------------------------------------
Children                                          No                 Yes                No                 Yes                No                 Yes
Operation              Operand
====================== ========================== ================== ================== ================== ================== ================== ==================
Negation          n/A  -``PfLine``                Yes                Yes                Yes                Yes                Yes                Yes
Addition / subtraction ± float                    Yes; value is interpreted as price in No                 No                 No                 No
                                                  default unit; returns ``PfLine`` 
                                                  without children.
---------------------- -------------------------- ------------------------------------- ------------------ ------------------ ------------------ ------------------
\                      ± price                    Yes; returns ``PfLine`` without       No                 No                 No                 No
                                                  children.
---------------------- -------------------------- ------------------------------------- ------------------ ------------------ ------------------ ------------------
\                      ± volume                   No                 No                 Yes; returns ``PfLine`` without       No                 No
                                                                                        children.
---------------------- -------------------------- ------------------ ------------------ ------------------------------------- ------------------ ------------------
\                      ± ``"all"``-Pfline         No                 No                 No                 No                 Yes; returns ``PfLine`` without
                                                                                                                              children.
---------------------- -------------------------- ------------------ ------------------ ------------------ ------------------ -------------------------------------
Multiplication         * float                    Yes                Yes                Yes                Yes                No                 No
\                      * price                    No                 No                 Yes; returns ``PfLine`` without       No                 No
                                                                                        children, of kind ``"all"``.
---------------------- -------------------------- ------------------ ------------------ ------------------------------------- ------------------ ------------------
\                      * volume                   Yes; returns ``PfLine`` without       No                 No                 No                 No
                                                  children, of kind ``"all"``.
---------------------- -------------------------- ------------------------------------- ------------------ ------------------ ------------------ ------------------
\                      * ``"all"``-Pfline         No                 No                 No                 No                 No
Division               / float                    Yes                Yes                Yes                Yes                No                 No
\                      / price                    Yes; returns ``pandas.Series`` of     No                 No                 No                 No
                                                  floats
---------------------- -------------------------- ------------------------------------- ------------------ ------------------ ------------------ ------------------
\                      / volume                   No                 No                 Yes; returns ``pandas.Series`` of     No                 No            
                                                                                        floats
---------------------- -------------------------- ------------------ ------------------ ------------------------------------- ------------------ ------------------
====================== ========================== ================== ================== ================== ================== ================== ==================

Remarks:

* The operands are defined liberally. E.g. "price" means any object that can be interpreted as a price: a single ``pint.Quantity`` with a valid price unit; a ``pandas.Series`` with a ``pint`` unit, a dictionary with a single key ``"p"``, a ``pandas.Dataframe`` with a single column ``"p"``, or a ``PfLine`` of kind ``"p"``; see :doc:`interoperability`. A single value is understood to apply to each timestamp in the index of the portfolio line.

  The following code example shows a few equivalent operations.

  .. exec_code::

     import portfolyo as pf, pandas as pd
     index = pd.date_range('2024', freq='AS', periods=3)
     pfl = pf.PfLine(pd.Series([2, 2.2, 3], index, dtype='pint[MW]')) 
     pfl_1 = pfl + {'q': 50.0}
     pfl_2 = pfl + pf.Q_(50000.0, 'kWh')
     pfl_3 = pfl + {'q': pd.Series([50, 50, 50.0], index)}
     pfl_4 = pfl + pf.PfLine({'q': pd.Series([50, 50, 50.0], index)})
     pfl_1 == pfl_2 == pfl_3 == pfl_4
     # --- hide: start ---
     print(repr(pfl_1 == pfl_2 == pfl_3 == pfl_4))

* Notice how the multiplication of a ``"p"`` and ``"q"`` portfolio line results in a ``"all"`` portfolio line.

  .. exec_code::
     
     import portfolyo as pf, pandas as pd
     index = pd.date_range('2024', freq='AS', periods=3)
     vol = pf.PfLine(pd.Series([2, 2.2, 3], index, dtype='pint[MW]')) 
     pri = pf.PfLine(pd.Series([100, 150, 200.0], index, dtype='pint[Eur/MWh]')) 
     vol * pri
     # --- hide: start ---
     print(repr(vol * pri))

* There are not many operations in which a portfolio line with children is not flattened (i.e., in which its children are kept). The reason is due to ambiguity. E.g., if we have two children with a resulting volume of 40 MWh, there is no "natural" way to add 10 MWh to this - we could add it to one and nothing to the other; we could split it evenly, or proportionally; we could add a third child with the additional volume. For this reason, the object is flattened beforehand. 

* Only when multiplying with (dividing by) a float or dimensionless value, or when negating, are ``PfLine``'s children retained; the reason is that in those cases, applying the operation to each child seperately gives us the expected result.

--------------------------
Operations: set timeseries
--------------------------

Sometimes we may want to replace one part of a ``PfLine``, while keeping the others the same. For this we can use the ``.set_w()``, ``.set_q()``, ``.set_p()`` and ``.set_r()`` methods. These methods accept float values, ``pint.Quantity`` objects, and ``pandas.Series``, and return a new ``PfLine`` with the selected information replaced. The returned portfolio lines are flattened, i.e., without children. 

It is also possible to set a ``"p"``- or ``"q"``-``PfLine`` as the price or volume; for this we use the ``.set_price()`` and ``set_volume()`` methods.





---
API
---



.. autoclass:: portfolyo.PfLine
   :members:
   :inherited-members: 