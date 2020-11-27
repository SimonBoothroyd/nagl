# NAGL

[![tests](https://github.com/simonboothroyd/nagl/workflows/tests/badge.svg?branch=main)](#)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/simonboothroyd/nagl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/simonboothroyd/nagl/context:python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A playground for applying graph convolutional networks to molecules, with a focus on learning continuous "atom-type"
embeddings and from these classical molecule force field parameters.

This framework is mostly based upon the [*End-to-End Differentiable Molecular Mechanics Force Field Construction*](https://arxiv.org/abs/2010.01196) 
preprint by Wang, Fass and Chodera.

## Installation

The required dependencies for this framework can be installed using `conda`:

```
conda env create --name nagl --file devtools/conda-envs/test_env.yaml
python setup.py develop
```

***Note**: Currently the commercial OpenEye `oechem` and `oequacpac` packages are required as dependencies.*

## Scripts

* `scripts/data-set-generation/run.py` - A script which will load a set of molecules from their SMILES representations,
  compute their AM1 partial charges and Wiberg bond orders (WBO) then pickle these ready for featurization.
  
* `scripts/training/run.py` - A script which loads a train and test set of pickled molecules, featurizes them, and
   then attempts to train a generalizable NN model which is able to predict the AM1 partial charges and WBOs.
 

## Copyright

Copyright (c) 2020, Simon Boothroyd
