<h1 align="center">EventAug</h1>
<h3 align="center">A Python package for augmenting event-camera datasets</h3>

<div align='center'>

[![Tests](https://github.com/NeelayS/event_aug/actions/workflows/package-test.yml/badge.svg)](https://github.com/NeelayS/event_aug/actions/workflows/package-test.yml)
[![Docs](https://readthedocs.org/projects/event-aug/badge/?version=latest)](https://event-aug.readthedocs.io/en/latest/?badge=latest)

**[Documentation](https://event_aug.readthedocs.io/en/latest/)** | **[Tutorials](https://github.com/NeelayS/event_aug/tree/main/tutorial_ntbks)**

</div>


# About

`EventAug` is an open-source Python package that provides methods for augmentation of visual event data. This package was developed as part of the [Google Summer of Code 2022](https://summerofcode.withgoogle.com/) program. A summary of information about the project can be found at [this webpage](https://neelays.github.io/gsoc-2022/).


# Installation


## From source (recommended)


### Using Poetry (recommended)


If you don't have `Poetry` installed, you can install it with the following command:

```bash

pip install poetry

```

Clone the public repository and enter the working directory:

```bash

git clone https://github.com/NeelayS/event_aug
cd event_aug/

```

To install the basic version of the package:

```bash

poetry install --without dev

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


# Usage

Please refer to the `tutorial_ntbks/` directory for usage examples. It contains two IPython notebooks which demonstrate how the package can be used to augment event-camera datasets end-to-end using spike-encoded (custom) videos and Perlin noise.
