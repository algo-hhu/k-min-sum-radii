[![Build Status](https://github.com/algo-hhu/k-min-sum-radii/actions/workflows/mypy-flake-test.yml/badge.svg)](https://github.com/algo-hhu/k-min-sum-radii/actions)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Supported Python version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Stable Version](https://img.shields.io/pypi/v/kmsr?label=stable)](https://pypi.org/project/kmsr/)

# K-Min-Sum-Radii


TODO

## Installation

```bash
pip install kmsr
```

## Example

TODO

## Development

Install [poetry](https://python-poetry.org/docs/#installation)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install clang
```bash
sudo apt-get install clang
```

Set clang variables
```bash
export CXX=/usr/bin/clang++
export CC=/usr/bin/clang
```

Install the package
```bash
poetry install
```

If the installation does not work and you do not see the C++ output, you can build the package to see the stack trace
```bash
poetry build
```

Run the tests
```bash
poetry run python -m unittest discover tests -v
```

## Citation

If you use this code, please cite [the following paper](TODO):

```
TODO
```
