# NAGL

[![tests](https://github.com/SimonBoothroyd/nagl/workflows/CI/badge.svg?branch=main)](https://github.com/SimonBoothroyd/nagl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/SimonBoothroyd/nagl/branch/main/graph/badge.svg?token=Aa8STE8WBZ)](https://codecov.io/gh/SimonBoothroyd/nagl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A playground for applying graph convolutional networks to molecules, with a focus on learning continuous "atom-type"
embeddings and from these classical molecule force field parameters.

This framework is mostly based upon the [*End-to-End Differentiable Molecular Mechanics Force Field Construction*](https://arxiv.org/abs/2010.01196) 
preprint by Wang, Fass and Chodera.

## Getting Started

Examples for using this framework can be found in the [`examples`](examples) directory.

## Installation

The required dependencies for this framework can be installed using `conda`:

```
conda install -c conda-forge -c dglteam nagl "dgl >=0.7"
```

### Optional dependencies

To make the full use of the framework, it is recommended to install the following 

#### [OpenFF Toolkit](https://github.com/openforcefield/openff-toolkit):

```
# Molecule loading and featurization
conda install -c conda-forge openff-toolkit
```

#### [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning):

```
# Cleaner, distributed model training
conda install -c conda-forge pytorch-lightning
```

#### [Dask](https://github.com/dask/distributed):

```
# Distributed molecule processing / labelling.
conda install -c conda-forge distributed dask-jobqueue
```

## License

The main package is release under the [MIT license](LICENSE). Parts of the package are inspired by / modified from a 
number of third party packages whose licenses are included in the [3rd party license file](LICENSE-3RD-PARTY).

## Copyright

Copyright (c) 2021, Simon Boothroyd
