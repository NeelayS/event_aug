# event-aug

A Python package for augmenting event-camera datasets

## Installation


### From source (recommended)


#### Using Poetry (recommended)

Clone the repository and enter working directory:

```bash

git clone https://github.com/NeelayS/event_aug
cd event_aug

```

If you don't have `Poetry` installed, you can install it with the following command:

```bash

pip install poetry

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


#### Using Pip

To install the basic version of the package:

```bash

pip install git+https://github.com/NeelayS/event_aug.git

```

To use the YouTube video downloading functionality of the package:

```bash

pip install git+https://github.com/NeelayS/event_aug.git#[youtube]

```

### From PyPI

**TBA**

<!-- ```bash

pip install event-aug

``` -->