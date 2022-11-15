================
Interoperability
================

We try to maximize interoperability between the ``PfLine`` and ``PfState`` classes and common Python classes. At the same time, we want the outcome of any operation to be unambiguous, without having to guess at or be surprised by its outcome.

.. contents:: Page contents:
   :depth: 2
   :local:


.. note:: This section assumes you are familiar with :doc:`../specialized_topics/dimensions`

For our purposes, the most common data container is the timeseries, though on some occasions we are dealing with single values (e.g. a single, time-independent, price of 45 Eur/MWh). There are several ways for the user to provide this information.

In the code examples below, the following imports are assumed and variables are assumed:

.. exec_code::
   
   import portfolyo as pf
   import pandas as pd
   idx = pd.date_range("2023", freq="AS", periods=2)

---------
One value
---------

To pass a single value, the following objects can be used:

* A ``float`` or ``int`` value.

* A ``pint.Quantity``, which is unit-aware. For convenience, the ``Quantity`` class (with the relevant unit registry) is available at ``porfolyo.Q_()``:

  .. exec_code::
      
     # --- hide: start ---
     import portfolyo as pf
     print(repr(pf.Q_(50, "Eur/MWh")))
     # --- hide: stop ---
     pf.Q_(50, "Eur/MWh")
     
  The unit is converted to the default unit for its dimension once it is used in any of the ``portfolyo`` objects, see also :ref:`this section<nameunitcompatibility>` further below.
  
  See `pint's website <https://pint.readthedocs.io>`_ for more information about ``pint``.

.. hint:: Using a ``pint.Quantity`` expresses a more deliberate intent, and therefore allows us to catch dimensionality errors more easily. For dimensionless values, such as fractions, we could even use a dimensionless ``Quantity`` (though this quickly becomes cumbersome).

------------------
One or more values
------------------

If we have to specify several individual values, we can use:

* A dictionary with the one or more of the dimension abbrevations (``"w"``, ``"q"``, ``"p"``, ``"r"``, ``"nodim"``) as the keys, and ``float``, ``int`` or ``pint.Quantity`` instances as the values. E.g.:

  .. exec_code::
     
     # --- hide: start ---
     import portfolyo as pf
     import pandas as pd
     # --- hide: stop ---
     {"p": 50.0, "w": pf.Q_(120, 'MW')}
     # --- hide: start ---
     print(repr({"p": 50.0, "w": pf.Q_(120.0, 'MW')}))

* Or we can use any other ``Mapping`` from string values to ``float``s, ``int``s, or ``pint.Quantity`` objects, e.g., a ``pandas.Series`` with a string index. It is recommended, however, to use ``Series`` only for timeseries information.

.. note:: Because we have to explicitly state the dimension abbreviation, these objects help us avoid dimensionality errors. For this reason, we may want to use them, even for *single* values.
  
.. _singletimeseries:

--------------
One timeseries
--------------

.. warning:: To avoid unexpected behavior, timeseries (``pandas.Series`` and ``pandas.DataFrame`` objects) should be of a certain form. See :doc:`../specialized_topics/dataprep`.

For timeseries, ``pandas.Series`` are used. These can be "unit-agnostic" (i.e., of datatype ``float`` or ``int``), or unit-aware as in the example below. [#ts]_

.. exec_code::
   
   # --- hide: start ---
   import portfolyo as pf 
   import pandas as pd
   idx = pd.date_range("2023", freq="AS", periods=2)
   # --- hide: stop ---
   pd.Series([50, 56.0], idx, dtype="pint[Eur/MWh]")  # unit-aware
   # --- hide: start ---
   print(repr(pd.Series([50, 56.0], idx, dtype="pint[Eur/MWh]")))

.. warning:: The ``name`` attribute of a ``pandas.Series`` is always ignored.

----------------------
One or more timeseries
----------------------

To pass several timeseries, we can use:

* A dictionary with the one or more of the dimension abbrevations (``"w"``, ``"q"``, ``"p"``, ``"r"``, ``"nodim"``) as the keys, and timeseries as the values. E.g.:

  .. exec_code::

     # --- hide: start ---
     import portfolyo as pf 
     import pandas as pd
     idx = pd.date_range("2023", freq="AS", periods=2)
     # --- hide: stop ---
     {"p": pd.Series([50, 56], idx), "w": pd.Series([120, 125], idx, dtype="pint[MW]")}
     # --- hide: start ---
     print(repr({"p": pd.Series([50, 56.0], idx), "w": pd.Series([120, 125.0], idx, dtype="pint[MW]")}))
    
  Each of the timeseries can have a unit or be unit-agnostic.

* Or we can use any other ``Mapping`` from string values to timeseries, e.g., a ``pandas.DataFrame`` with a datetime-index. In this case:

  .. exec_code::
    
     # --- hide: start ---
     import portfolyo as pf 
     import pandas as pd
     idx = pd.date_range("2023", freq="AS", periods=2)
     # --- hide: stop ---
     pd.DataFrame({"p": [50, 56], "w": [120, 125]}, idx)
     # --- hide: start ---
     print(repr(pd.DataFrame({"p": [50, 56.0], "w": [120, 125.0]}, idx)))

  Dataframes can also be made unit-aware. [#df]_

.. note:: The same applied here: because we have to explicitly state the dimension abbreviation, these objects help us avoid dimensionality errors. For this reason, we may want to use them, even for *single* timeseries.
  
------------
Combinations
------------

Dictionaries are the most versatily of these objects. They can be used to pass a single value, multiple values, a single timeseries, multiple timeseries, or a combination of these:

.. exec_code::
    
   # --- hide: start ---
   import portfolyo as pf 
   import pandas as pd
   idx = pd.date_range("2023", freq="AS", periods=2)
   # --- hide: stop ---
   d1 = {"p": 50}
   d2 = {"p": 50, "w": 120}
   d3 = {"p": pd.Series([50, 56], idx)}
   d4 = {"p": pd.Series([50, 56], idx), "w": pd.Series([120, 125], idx)}
   d5 = {"p": pd.Series([50, 56], idx), "w": 120}
    

.. _ducktyping:

-----------------------------
Duck typing for other objects
-----------------------------

Any object can be used, as long as it has an ``.items()`` method returning (key, value)-tuples (e.g. if it inherits from the ``Mapping`` `abstract base class <https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping>`_ and therefore implements ``__getitem__``, ``__iter__`` and ``__len__`` methods), and all keys are valid dimension abbrevations.

.. _nameunitcompatibility:

-------------------------------------
Compatilibity of abbrevation and unit
-------------------------------------

Information can have a key (one of the dimension abbrevations: ``"w"``, ``"q"``, ``"p"``, ``"r"``, ``"nodim"``) and/or a unit. In a DataFrame, a timeseries' key is the corresponding column name. A timeseries 'by itself' has no key; its name is ignored.

There is a one-to-one relationship between dimension abbrevation and unit; see :doc:`../specialized_topics/dimensions`.

* In some of the objects discussed above, we specify both a key *and* a unit. In that case, ``portfolyo`` checks if the unit has the correct dimensionality. If so, but it is not the default unit, a conversion to the default unit is done. 

  E.g., the key ``"p"`` and unit ``ctEur/kWh`` of ``{"p": pd.Series([5.0, 5.6], idx, dtype="pint[ctEur/kWh]")}`` are consistent. The values will be changed to the default unit (=Eur/MWh) upon further processing. Using ``"q"`` instead of ``"p"`` results in a dimensionality error, and using ``"x"`` results in a KeyError.

* In some objects, only the unit is specified. Here, the dimension is deduced from the unit, and the unit is converted into the default (if necessary). 

  E.g., the timeseries ``pd.Series([5.0, 5.6], idx, dtype="pint[ctEur/kWh]")`` (NB: without the dictionary key) is such an object.

* In other objects, only the key is specified. In that case, the unit is deduced from the key - the default unit is assumed. 

  E.g., the key ``"p"`` of ``{"p": pd.Series([50, 56], idx)}`` indicates that we are dealing with prices, and the default unit of Eur/MWh is assumed.

* If both are not provided, the dimension must be inferrable from the context, and the unit is assumed to be the default for that dimension. 

  E.g. when adding a ``float`` value to a ``PfLine`` containing prices, the value is assumed to also be a price, in the default unit (= Eur/MWh).


---------
Footnotes
---------

.. [#ts]
    
   If we want to add unit-awareness to such a series, we can use the ``.astype()`` method with a pint-unit (e.g. "pint[MW]") as its argument (as in line 3). Alternatively, we can create it from scratch with the ``dtype`` parameter (as in line 5):

   .. code-block:: python 
       :emphasize-lines: 3,4

       >>> idx = pandas.date_range("2023", freq="AS", periods=2)
       >>> s0 = pandas.Series([50, 56], idx)  # unit-agnostic
       >>> s1 = s0.astype("pint[Eur/MWh]")  # unit-aware

       >>> s2 = pandas.Series([50, 56], idx, dtype="pint[Eur/MWh]")  # same as s1

       >>> s1
       2023-01-01    50.0
       2024-01-01    56.0
       Freq: AS-JAN, dtype: pint[Eur/MWh]

.. [#df]

   There are several ways to create a unit-aware dataframe; the easiest is to create it from unit-aware series (as in line 4). Alternatively, if we already have the unit-agnostic dataframe ready, we can also use the ``.astype()`` method here (line 7):

   .. code-block:: python
      :emphasize-lines: 4, 7

      >>> idx = pandas.date_range("2023", freq="AS", periods=2)
      >>> s_price = pandas.Series([50, 56], idx, dtype="pint[Eur/MWh]")
      >>> s_volume = pandas.Series([120, 125], idx, dtype="pint[MW]")
      >>> df1 = pandas.DataFrame({"p": s_price, "w": s_volume})

      >>> df_agn = pandas.DataFrame({"p": [50, 56], 'w': [120, 125]}, idx) # unit-agnostic
      >>> df2 = df_agn.astype({'p': 'pint[Eur/MWh]', 'w': 'pint[MW]'}) # same as df1

      >>> df1.dtypes
      p    pint[Eur/MWh]
      w         pint[MW]
      dtype: object