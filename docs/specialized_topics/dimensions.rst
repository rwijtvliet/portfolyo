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

In ``portfolyo``, a convention is used for the abbreviation the most relevant dimensions.

============  ==============================================
Abbreviation  Dimension                                     
============  ==============================================
``w``         Power (energy per duration)
``q``         Energy (power times duration)
``p``         Price (revenue per energy)
``r``         Revenue  (or any other
              absolute monetary value)
``duration``  Duration (of a delivery period)
``nodim``     dimensionless
============  ==============================================

Remarks:

* Concerning "Power": not to be confused with power as a synonym for electricity.

* Concerning "Revenue": Unfortunately, there is no perfect word here. We cannot universally use "cost" or "income", as it depends on the situation, which one applies. "Value" is too vague, and "monetary value" too long.

----------------------------
Missing or non-default units
----------------------------

Using ``pint``, a user can specify not only the magnitude of a value (or Series or DataFrame), but also its *units*. For more information on how this is handled, see :ref:`here <nameunitcompatibility>`.

In short: 

* A unit must be specified.

* If no abbreviation is specified, the unit is used to deduce it (using the table above).

* If both an abbreviation and a unit are specified, they should be compatible.

------------------------
Other naming conventions
------------------------

"volume" and "quantity" are often used to indicate either power or energy without having to specify, which one. In most situations, it is clear from context which delivery period is meant, so knowing one, we can calculate the other. E.g. 100 MW for the duration of January equals 74 400 MWh.




