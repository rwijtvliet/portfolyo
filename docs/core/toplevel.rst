.. |_| unicode:: 0xA0 
   :trim:

===============
Top-level tools
===============

Some tools for working with ``pandas.Series``, ``pandas.DataFrame``, ``portfolyo.PfLine`` and ``portfolyo.PfState`` objects are available at the root of the package. They are concisely listed below.

----------------------
Work on pandas objects 
----------------------

* ``portfolyo.asfreq_avg()`` Changes the frequency of a Series or DataFrame with "averagable" data. See :doc:`this page<../specialized_topics/resampling>` for more information.

* ``portfolyo.asfreq_sum()`` Changes the frequency of a Series or DataFrame with "summable" data. See :doc:`this page<../specialized_topics/resampling>` for more information. 

* ``portfolyo.wavg()`` Calculates weighted average of a Series or DataFrame.

* ``portfolyo.standardize()`` Ensures/asserts a Series or DataFrame follows necessary rules to initialize PfLine with.
  
-------------------------
Work on portfolyo objects
-------------------------

* ``portfolyo.concat()`` Concatenates PfLines into one PfLine.
  
* ``portfolyo.plot_pfstates()`` Plots several PfStates in one figure.


