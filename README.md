<h1 align="center">EventAug</h1>
<h3 align="center">A Python package for augmenting event-camera datasets</h3>

<div align='center'>

[![Tests](https://github.com/NeelayS/event_aug/actions/workflows/package-test.yml/badge.svg)](https://github.com/NeelayS/event_aug/actions/workflows/package-test.yml)

</div>


# Installation


## From source (recommended)


### Using Poetry (recommended)


If you don't have `Poetry` installed, you can install it with the following command:

```bash

pip install poetry

```

Clone the repository and enter working directory:

```bash

git clone https://github.com/NeelayS/event_aug
cd event_aug

```

To install the basic version of the package:


```bash

poetry install --no-dev

```

If you wish to install the development version of the package:

```bash

poetry install

```

If you wish to use the Youtube video downloading functionality of the package, you can additionally run:

```bash

poetry install -E youtube

```


### Using Pip


To install the basic version of the package:

```bash

pip install git+https://github.com/NeelayS/event_aug.git

```

To use the YouTube video downloading functionality of the package:

```bash

pip install git+https://github.com/NeelayS/event_aug.git#[youtube]

```


<!-- ## From PyPI


**TBA** -->

<!-- To install the basic version of the package: -->

<!-- ```bash

pip install event-aug

``` -->

<!-- To use the YouTube video downloading functionality of the package:

```bash

pip install event-aug[youtube]

``` -->