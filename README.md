# NAGL

[![tests](https://github.com/SimonBoothroyd/nagl/workflows/CI/badge.svg?branch=main)](https://github.com/SimonBoothroyd/nagl/actions?query=workflow%3ACI)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/SimonBoothroyd/nagl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/SimonBoothroyd/nagl/context:python)
[![codecov](https://codecov.io/gh/SimonBoothroyd/nagl/branch/main/graph/badge.svg?token=Aa8STE8WBZ)](https://codecov.io/gh/SimonBoothroyd/nagl)
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

## Copyright

Copyright (c) 2020, Simon Boothroyd
