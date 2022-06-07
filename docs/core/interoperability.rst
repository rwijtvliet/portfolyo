================
Interoperability
================

We try to maximize interoperability between the ``PfLine`` and ``PfState`` classes and common Python classes. At the same time, we want the outcome of any operation to be unambiguous, without having to guess at or be surprised by its outcome.

.. note:: This section assumes you are familiar with :doc:`../specialized_topics/dimensions`

For our purposes, the most common data container is the timeseries, though on some occasions we are dealing with single values (e.g. a single, time-independent, price of 45 Eur/MWh). There are several ways for the user to provide this information.

----------------
Types of objects
----------------

Single value
------------

To pass a single value, the following objects can be used:

* A ``float`` or ``int`` value.

* A ``pint`` ``Quantity``, which is unit-aware. For convenience, the ``Quantity`` class (with the relevant unit registry) is available at ``porfolyo.Q_()``:

  .. code-block::

      >>> portfolyo.Q_(50, 'Eur/MWh')
      50 <Unit('euro / megawatthour')>

  The unit is converted to the default unit for its dimension once it is used in any of the ``portfolyo`` objects, see also the section :ref:`<Compatilibity of name and unit>`.
  
  See `pint's website <https://pint.readthedocs.io>`_ for more information about ``pint``.

.. hint:: Using a ``pint`` ``Quantity`` expresses a more deliberate intent, and therefore allows us to catch dimensionality errors more easily. For dimensionless values, such as fractions, we could even use a dimensionless ``pint`` ``Quantity`` (though this quickly becomes cumbersome).

Several values
--------------

To pass several individual values, we can use:

* A dictionary with the one or more of the dimension abbrevations (``'w'``, ``'q'``, ``'p'``, ``'r'``, ``'nodim'``) as the keys, e.g.:

  .. code-block::

      >>> {'p': 50.0, 'w': 120}

* Or the equivalent ``pandas`` ``Series``:

  .. code-block::

      >>> pandas.Series({'p': 50.0, 'w': 120})
      p     50.0
      w    120.0
      dtype: float64

The individual values of the dictionary and the series can be simple ``float`` (or ``int``) values (as in the examples above) or they can be ``pint`` ``Quantities``.

If any of the keys is not recognised, a ``KeyError`` is raised.

.. note:: Of course we can also use a dictionary or series to pass a *single* value. The advantage is that we can explicitly state the dimension using one of the abbrevations, which we cannot do when passing only the value or ``Quantity``.

Single timeseries
-----------------

.. warning:: To ensure operability, timeseries (``pandas`` ``Series`` and ``DataFrame`` objects) should be of a certain form. See :doc:`../specialized_topics/dataprep`.

For timeseries, ``pandas`` ``Series`` are used. These can be "unit-agnostic" (i.e., of datatype ``float`` or ``int``) or unit-aware. [#ts]_

.. code-block::

      >>> idx = pandas.date_range('2023', freq='AS', periods=2)
      >>> pandas.Series([50, 56], idx).astype("pint[Eur/MWh]")  # unit-aware
      2023-01-01    50.0
      2024-01-01    56.0
      Freq: AS-JAN, dtype: pint[Eur/MWh]

.. warning:: The ``name`` attribute of a ``pandas`` ``Series`` is always ignored by ``portfolyo``.

Several timeseries
------------------

To pass several timeseries, we can use:

* A ``pandas`` ``DataFrame`` with ``'w'``, ``'q'``, ``'p'`` ``'r'`` and/or ``'nodim'`` as its column name(s), e.g.:

  .. code-block::
      
      >>> idx = pandas.date_range('2023', freq='AS', periods=2)
      >>> pandas.DataFrame({'p': [50, 56], 'w': [120, 125]}, idx)
                   p    w
      2023-01-01  50  120
      2024-01-01  56  125

  Dataframes can also be unit-aware. [#df]_

* A dictionary with ``'w'``, ``'q'``, ``'p'`` ``'r'`` and/or ``'nodim'`` as its key(s), e.g.:

  .. code-block::

      >>> {'p': pandas.Series([50, 56], idx), 'w': pandas.Series([120, 125], idx)}

.. note:: Here too: we can of course also use a dictionary or dataframe to pass a *single* timeseries. The advantage is that we can explicitly state the dimension using one of the abbrevations, which we cannot do when only passing the timeseries.

Combinations
------------

Dictionaries are the most versatily of these objects. They can be used to pass a single values, multiple values, a single timeseries, multiple timeseries, or a combination of these:

.. code-block::
    
    >>> idx = pandas.date_range('2023', freq='AS', periods=2)
    >>> d1 = {'p': 50}
    >>> d2 = {'p': 50, 'w': 120}
    >>> d3 = {'p': pandas.Series([50, 56], idx)}
    >>> d4 = {'p': pandas.Series([50, 56], idx), 'w': pandas.Series([120, 125], idx)}
    >>> d5 = {'p': pandas.Series([50, 56], idx), 'w': 120}

Duck typing for other objects
-----------------------------

For other objects than the ones described above: if an object is subscriptable (i.e., implements ``object[]``), *and* at least one of ``object[name]``, with ``name`` one the dimension abbrevations, returns a value or timeseries, then this is used as the data. 

.. note:: Only access by index (``object['p']``) is checked; access by attribute (``object.p``) is not.

------------------------------
Compatilibity of name and unit
------------------------------

Information can have a name (one of the dimension abbrevations: ``'w'``, ``'q'``, ``'p'``, ``'r'``, ``'nodim'``) and/or a unit. 

* In some of the objects discussed above, we specify both a name *and* a unit. In that case, ``portfolyo`` checks if the unit has the correct dimensionality. If so, but it is not the default unit, a conversion to the default unit is done. 

  E.g., the name ``'p'`` and unit ``ctEur/kWh`` of ``{'p': pandas.Series([5.0, 5.6], idx, dtype='pint[ctEur/kWh]')}`` are consistent. The values will be changed to the default unit (=Eur/MWh) upon further processing. Using ``q`` instead of ``'p'`` results in a dimensionality error.

* In some objects, only the unit is specified. In that case, the dimension is deduced from the unit, and the unit is again converted into the default (if necessary). 

  E.g., the timeseries ``pandas.Series([5.0, 5.6], idx, dtype='pint[ctEur/kWh]')`` from the previous example (so without the dictionary key) is such an object.

* In other objects, only the name is specified. In that case, the unit is deduced from the name - the default unit is assumed. 

  E.g., the name ``'p'`` of ``{'p': pandas.Series([50, 56], idx)}`` indicates that we are dealing with prices, and the default unit of Eur/MWh is assumed.

* If both are not provided, the dimension must be inferrable from the context, and the unit is assumed to be the default for that dimension. 

  E.g. when adding a ``float`` value to a ``PfLine`` containing prices, the value is assumed to also be a price, in the default unit (= Eur/MWh).


.. [#ts]
    
    If we want to add unit-awareness to such a series, we can use the ``.astype()`` method with a pint-unit (e.g. "pint[MW]") as its argument. Alternatively, we can create it from scratch with the ``dtype`` parameter:

    .. code-block::

        >>> idx = pandas.date_range('2023', freq='AS', periods=2)
        >>> s0 = pandas.Series([50, 56], idx)  # unit-agnostic
        >>> s1 = s0.astype("pint[Eur/MWh]")  # unit-aware
        >>> s2 = pandas.Series([50, 56], idx, dtype="pint[Eur/MWh]")  # unit-aware
        >>> s1  # s2 is the same
        2023-01-01    50.0
        2024-01-01    56.0
        Freq: AS-JAN, dtype: pint[Eur/MWh]

.. [#df]

    There are several ways to create a unit-aware dataframe; the easiest is to create it from unit-aware series:

    .. code-block::
          
        >>> idx = pandas.date_range('2023', freq='AS', periods=2)
        >>> df0 = pandas.DataFrame({('p', 'Eur/MWh'): [50, 56], ('w', 'MW'): [120, 125]}, idx) # unit-agnostic
        >>> df1 = df0.pint.quantify()  # unit-aware; bottom column level used as unit
        >>> s_p = pandas.Series([50, 56], idx, dtype="pint[Eur/MWh]")
        >>> s_w = pandas.Series([120, 125], idx, dtype="pint[MW]")
        >>> df1 = pandas.DataFrame({'p': s_p, 'w': s_w})