portfolyo
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

Analysing and manipulating timeseries related to power and gas offtake portfolios.


Installation
------------

.. code-block:: bash

   pip install portfolyo


Developing
----------

This project uses ``black`` to format code and ``flake8`` for linting. We also support ``pre-commit`` to ensure
these have been run. To configure your local environment please install these development dependencies and set up
the commit hooks.

.. code-block:: bash

   $ pip install black flake8 pre-commit
   $ pre-commit install