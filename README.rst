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

.. image:: https://img.shields.io/codecov/c/gh/rwijtvliet/portfolyo
   :target: https://app.codecov.io/gh/rwijtvliet/portfolyo
   :alt: Codecov

.. image:: https://readthedocs.org/projects/portfolyo/badge/?version=latest
    :target: https://portfolyo.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Portfolyo is a Python package to analyse and manipulate timeseries related to power 
and gas offtake portfolios.

Installation
------------

.. code-block:: bash

   pip install portfolyo

NB: this package is under active development and the API will change without prior notice. To ensure your code will continue to work, pin the version number that you install:

.. code-block:: bash

   pip install portfolyo==x.x.x


Documentation
-------------

Documentation is hosted on readthedocs:

https://portfolyo.readthedocs.io/

Repository
----------

The git repository is hosted on github:

http://www.github.com/rwijtvliet/portfolyo


Developing
----------

This project uses ``black`` to format code and ``flake8`` for linting. We also support ``pre-commit`` to ensure
these have been run. To configure your local environment please install these development dependencies and set up
the commit hooks.

.. code-block:: bash

   $ pip install -r requirements-dev.txt
   $ pre-commit install

Feature branches are merged into the `develop` branch. This branch is merged into the `main` branch whenever a new stable release is published.