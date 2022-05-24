======
PfLine
======

A ``PfLine`` object stores a timeseries containing volume information, price information, or both.

For example: 

* The gas forward price curve for a certain market area, in daily resolution, for tomorrow until the end of the frontyear+4. This ``PfLine`` only contains price information with e.g. a unit of Eur/MWh.

* The expected offtake volume of a certain portfolio, for the coming calender year, in quarterhourly resolution. This is a ``PfLine`` that only contains volume information. The volume in each timestamp can be retrieved by the user in units of energy (e.g., MWh) or in units of power (e.g., MW). 

* The sourced volume of the same portfolio, again for the coming calender year but in monthly resolution. This is a ``PfLine`` that, for each timestamp, contains a volume (the contracted volume, both as energy and as power), a price (for which the volume was contracted) and a revenue (in Eur, being the multiplication of the energy and the price).

Under the hood, not all information that can be retrieved by the user is stored; redundant information is discarded and recalculated whenever necessary. For the volume, for example, only the energy is stored. The power can be calculated by dividing the energy (in MWh) by the duration of the timestamp (in h).

.. autoclass:: portfolyo.PfLine
   :members:
   :inherited-members: