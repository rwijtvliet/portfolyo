=======
PfState
=======

A ``PfState`` object stores the timeseries that relate to a portfolio at a certain moment in time. It stores information about the offtake in the portfolio; how much volume has currently been sourced and at what price; how much volume is consequently still unsourced and what is its current market price; what is the current best-guess for the procurement price of the portfolio (i.e., combining sourced and unsourced volumes to satisfy offtake), etc. 
 
Like in ``PfLine`` objects: here too, part of the information is not stored but calculated on demand; e.g. the unsourced volume is calculated from the offtake and sourced volumes, etc. 

.. autoclass:: portfolyo.PfState
   :members:
   :inherited-members: