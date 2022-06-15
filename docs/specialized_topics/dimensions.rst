====================
Dimensions and Units
====================

We are usually dealing with data that has a certain physical dimension, which determines what kind of information it is and which are valid units. For example, if we are dealing with energy (i.e., "capacity to perform work"), some valid units are Joule, Megawatt-hour, calorie, and BTU. 

.. contents:: Page contents:
   :depth: 1
   :local:

-------------
Abbreviations
-------------

In ``portfolyo``, a convention is used for the abbreviation the most relevant dimensions, as well as their default units.

============  ==============================================  ============
Abbreviation  Dimension                                       Default unit
============  ==============================================  ============
``w``         Power (energy per duration)                     Megawatt [MW]
``q``         Energy (power times duration)                   Megawatt-hour [MWh]
``p``         Price (revenue per energy)                      Euro per Megawatt-hour [Eur/MWh]
``r``         Revenue  (or any other                          Euro [Eur]
              absolute monetary value)   
``duration``  Duration (of a delivery period)                 hours [h]                   
``nodim``     dimensionless                                   none
============  ==============================================  ============

Remarks:

* Concerning "Power": not to be confused with power as a synonym for electricity.

* Concerning "Revenue": Unfortunately, there is no perfect word here. We cannot universally use "cost" or "income", as it depends on the situation, which one applies. "Value" is too vague, and "monetary value" too long.

----------------------------
Missing or non-default units
----------------------------

Using ``pint``, a user can specify not only the magnitude of a value (or Series or DataFrame), but also its *units*. For more information on how this is handled, see :ref:`here <nameunitcompatibility>`.

In short: 

* If both an abbreviation and a unit are specified, they should be compatible.

* If a unit is specified, the values are automatically converted to the default unit.

* If no abbreviation is specified, the unit is used to deduce it (using the table above).

------------------------
Other naming conventions
------------------------

"volume" and "quantity" are often used to indicate either power or energy without having to specify, which one. In most situations, it is clear from context which delivery period is meant, so knowing one, we can calculate the other. E.g. 100 MW in January means 74 400 MWh.




