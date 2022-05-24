=======
Objects
=======

The objects at the center of the library are ``PfLine`` and ``PfState``.

PfLine
======

A ``PfLine`` object stores a timeseries containing volume information, price information, or both.

For example: 

* The gas forward price curve for a certain market area, in daily resolution, for tomorrow until the end of the frontyear+4. This ``PfLine`` only contains price information with e.g. a unit of Eur/MWh.

* The expected offtake volume of a certain portfolio, for the coming calender year, in quarterhourly resolution. This is a ``PfLine`` that only contains volume information. The volume in each timestamp can be retrieved by the user in units of energy (e.g., MWh) or in units of power (e.g., MW). 

* The sourced volume of the same portfolio, again for the coming calender year but in monthly resolution. This is a ``PfLine`` that, for each timestamp, contains a volume (the contracted volume, both as energy and as power), a price (for which the volume was contracted) and a revenue (in Eur, being the multiplication of the energy and the price).

More information can be found here: :doc:`/core/pfline`.

PfState
=======

A ``PfState`` object stores several ``PfLine`` objects that all relate to the same portfolio. It stores information about the offtake in the portfolio; how much volume has currently been sourced and at what price; how much volume is consequently still unsourced and what is its current market price; what is the current best-guess for the procurement price of the portfolio (i.e., combining sourced and unsourced volumes to satisfy offtake), etc. 
 
More information can be found here: :doc:`/core/pfstate`.