.. portfolyo documentation master file, created by
   sphinx-quickstart on Mon May 23 12:24:25 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=========
Portfolyo
=========

.. image:: https://img.shields.io/pypi/v/portfolyo
   :target: https://pypi.org/project/portfolyo
   :alt: Pypi

.. image:: https://github.com/rwijtvliet/portfolyo/actions/workflows/ci.yaml/badge.svg
   :target: https://github.com/rwijtvliet/portfolyo/actions/workflows/ci.yaml
   :alt: Github - CI

.. image:: https://github.com/rwijtvliet/portfolyo/actions/workflows/pre-commit.yaml/badge.svg
   :target: https://github.com/rwijtvliet/portfolyo/actions/workflows/pre-commit.yaml
   :alt: Github - Pre-commit

.. image:: https://img.shields.io/codecov/c/gh/rwijtvliet/portfolyo
   :target: https://app.codecov.io/gh/rwijtvliet/portfolyo
   :alt: Codecov

.. image:: https://readthedocs.org/projects/portfolyo/badge/?version=latest
    :target: https://portfolyo.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Portfolyo is a Python package to analyse and manipulate **timeseries** related to **power 
and gas offtake portfolios**.

A good place to start learing about this package is :doc:`the quickstart guide<tutorial/quickstart>`.

------------
Installation
------------

.. code-block:: bash

   pip install portfolyo


--------
Contents
--------

.. toctree::
   :maxdepth: 1

   ./index

.. toctree::
   :maxdepth: 2
   :caption: Core functionality

   core/objects
   core/pfline
   core/pfstate
   core/interoperability

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/quickstart

   tutorial/part1
   tutorial/part2
   tutorial/part3
   tutorial/part4

.. toctree::
   :maxdepth: 2
   :caption: Specialized topics

   specialized_topics/dataprep
   specialized_topics/timezones
   specialized_topics/dimensions
   specialized_topics/resampling

.. toctree::
   :maxdepth: 2
   :caption: Full reference

   full_reference