#! /usr/bin/env python3
"""
{begin_markdown model_runner_xam}
{spell_markdown }

# Example and Test of `ModelRunner`

`ModelRunner` is the main class that connects all of the individual components of a `curvefit`
model into a larger modeling pipeline. Here we will create examples of the building
blocks that are necessary for a ModelRunner.

## Function Documentation
Please see function documentation here: [`ModelRunner`][ModelRunner.md].

## Importing Packages
```python """
from curvefit.core.data import Data
from curvefit.models.core_model import CoreModel
from curvefit.core.parameter import Parameter, ParameterFunction, ParameterSet
from curvefit.uncertainty.residual_model import SmoothResidualModel
from curvefit.uncertainty.predictive_validity import PredictiveValidity



print('model_runner.py: OK')
"""
```
{end_markdown objective_fun_xam}
"""
