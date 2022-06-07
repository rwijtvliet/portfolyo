====================
Dimensions and Units
====================

We are usually dealing with data that has a certain physical dimension, which determines what kind of information it is and which are valid units. For example, if we are dealing with energy (i.e., "capacity to perform work"), some valid units are Joule, Megawatt-hour, calorie, and BTU. 

In ``portfolyo``, a convention is used for the abbreviation the most relevant dimensions, as well as their default units.

============  ==============================================  ============
Abbreviation  Dimension                                       Default unit
============  ==============================================  ============
``w``         Power [#pow]_ (energy per duration)             Megawatt [MW]
``q``         Energy (power times duration)                   Megawatt-hour [MWh]
``p``         Price (revenue per energy)                      Euro per Megawatt-hour [Eur/MWh]
``r``         Revenue [#rev]_ (or any other                   Euro [Eur]
              absolute monetary value)   
``duration``  Duration (of a delivery period)                 hours [h]                   
``nodim``     dimensionless                                   none
============  ==============================================  ============

------------------------
Other naming conventions
------------------------

"volume" and "quantity" are often used to indicate either power or energy without having to specify, which one. In most situations, it is clear from context which delivery period is meant, so knowing one, we can calculate the other. Also, 

.. [#pow] Not to be confused with power as a synonym for electricity.
.. [#rev] Unfortunately, there is no perfect word here. We cannot universally use "cost" or "income", as it depends on the situation, which one applies. "Value" is too vague, and "monetary value" too long.