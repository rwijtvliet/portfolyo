=========
portfolyo
=========

.. image:: https://img.shields.io/pypi/v/portfolyo
   :target: https://pypi.org/project/portfolyo
   :alt: Pypi

.. image:: https://github.com/rwijtvliet/portfolyo/actions/workflows/ci-on-pullreq.yaml/badge.svg
   :target: https://github.com/rwijtvliet/portfolyo/actions/workflows/ci-on-pullreq.yaml
   :alt: Github - CI (on pullrequest)

.. image:: https://github.com/rwijtvliet/portfolyo/actions/workflows/ci-on-push.yaml/badge.svg
   :target: https://github.com/rwijtvliet/portfolyo/actions/workflows/ci-on-push.yaml
   :alt: Github - CI (on push)

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


------------
Installation
------------

.. code-block:: bash

   pip install portfolyo

NB: this package is under active development and the API will change without prior notice. To ensure your code will continue to work, pin the version number that you install:

.. code-block:: bash

   pip install portfolyo==x.y.z

   # or, in pyproject.toml
   portfolyo = "x.y.z"


-------------
Documentation
-------------

Documentation is hosted on readthedocs:

https://portfolyo.readthedocs.io/


----------
Repository
----------

The git repository is hosted on github:

http://www.github.com/rwijtvliet/portfolyo


----------
Developing
----------

This project uses ``black`` to format code and ``flake8`` for linting. We also support ``pre-commit`` to ensure
these have been run. To configure your local environment please install these development dependencies and set up
the commit hooks.

.. code-block:: bash

   poetry install --with dev,test
   pre-commit install

Feature branches are merged into the ``develop`` branch via pull request.


Internal dependencies
---------------------

Inside the package, modules depend on each other in the following chain. A module may depend on another module if it is further to the left:

tools >> pfline >> pfstate >> tools2


----------
Publishing
----------

To publish a new release from ``develop``, create a new branch, increment the version number and push to github. For convenience, there is a ``create_release_branch.sh`` script that accomplishes the same, which takes one argument:

.. code-block:: bash

   create_release_branch.sh major # or minor, or patch, or specific version number

Then, from the github website, the release can be published by clicking the "tags" button. Be sure to select the correct branch.

When done, merge the release branch into ``develop`` and ``main``, also via pull request, and delete it.
