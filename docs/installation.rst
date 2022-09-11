.. highlight:: shell

============
Installation
============

Using Poetry (recommended)
---------------------------

If you don't have `Poetry` installed, you can install it with the following command:

.. code-block:: console

    $ pip install poetry

`event_aug` can then be installed from its `GitHub repository`_.

Clone the public repository and enter the working directory:

.. code-block:: console

    $ git clone https://github.com/NeelayS/event_aug
    $ cd event_aug/

To install the basic version of the package:

.. code-block:: console

    $ poetry install --without dev

In case you are using an older version of `Poetry` (before 1.0), you can use the following command instead:

.. code-block:: console

    $ poetry install --no-dev

If you want to install the development version of the package:

.. code-block:: console

    $ poetry install

If you wish to use the Youtube video downloading functionality of the package, you can additionally run:

.. code-block:: console

    $ poetry install -E youtube


.. _Github repository: https://github.com/NeelayS/event_aug


Using Pip
----------

`event_aug` can also be installed using `pip`.

To install the basic version of the package:

.. code-block:: console

    $ pip install git+https://github.com/NeelayS/event_aug.git

To use the YouTube video downloading functionality of the package:

.. code-block:: console

    $ pip install git+https://github.com/NeelayS/event_aug.git#[youtube]