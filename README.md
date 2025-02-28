# Mesh2scattering

[![PyPI version](https://badge.fury.io/py/mesh2scattering.svg)](https://badge.fury.io/py/mesh2scattering)
[![Documentation Status](https://readthedocs.org/projects/mesh2scattering/badge/?version=latest)](https://mesh2scattering.readthedocs.io/en/latest/?badge=latest)
[![CircleCI](https://circleci.com/gh/ahms5/mesh2scattering.svg?style=shield)](https://circleci.com/gh/ahms5/mesh2scattering)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyfar/gallery/main?labpath=docs/gallery/interactive/pyfar_introduction.ipynb)

Mesh2scattering is based on [Mesh2HRTF](https://github.com/Any2HRTF/Mesh2HRTF) and is an open-source project aiming an easy-to-use software package for the numerical calculation of scattering pattern and scattering and diffusion coefficients of any surface. In a nutshell, Mesh2scattering consists of three parts:

- input: prepares geometrical data and acoustic parameters for the simulation,
- numcalc: based on the input from ``input``, it calculates the corresponding sound field
- output: processes the output from NumCalc to scattering pattern.
- process: processes the output to scattering and/or diffusion coefficients.
- utils: helping functions.

Please notice that this project does not support HRTF post processing, use [Mesh2HRTF](https://github.com/Any2HRTF/Mesh2HRTF) instead.

## Getting Started

Check out the examples folder for a tour of the most important mesh2scattering
functionality and [read the docs](https://mesh2scattering.readthedocs.io/en/latest) for the complete documentation.

## Installation

Use pip to install mesh2scattering

```bash
pip install mesh2scattering
pip install git+https://github.com/pyfar/imkar.git@9770ad090196b73f3202a187784470d3f9f9e995
```

(Requires Python 3.8 or higher)

Note that NumCalc need to be build on Linux and MacOS. For Windows it can be downloaded.

### for Linux

Install the C++ build essentials by running

```bash
sudo apt-get install build-essential
```

Go into the NumCalc directory by running

```bash
cd path/to/your/Mesh2scattering/mesh2scattering/numcalc/src
```

Compile NumCalc by running make. It is now located in the folder ``mesh2scattering/numcalc/bin``

```bash
make
```

Copy NumCalc to a folder in your program path: in the same directory run

```bash
sudo cp NumCalc /usr/local/bin/
```

Now NumCalc can be used by running NumCalc (don't do this yet).

### for MacOS

Install the C++ build essentials by installing ``xcode``
Go into the ``numcalc/src`` directory by running

```bash
cd path/to/your/Mesh2scattering/mesh2scattering/numcalc/src
```

Compile NumCalc by running ``make``. It is now located in the folder ``mesh2scattering/numcalc/bin``

```bash
make
```

Now NumCalc can be used by running ``path/to/mesh2scattering/numcalc/bin/NumCalc`` (don't do this yet)

```bash
path/to/mesh2scattering/numcalc/bin/NumCalc
```

### for Windows

download the executable from the release.

## Contributing

Check out the [contributing guidelines](https://mesh2scattering.readthedocs.io/en/stable/contributing.html).
