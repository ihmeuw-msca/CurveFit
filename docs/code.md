# Package Overview

For a quick getting started demo, please see [getting started](extract_md/get_started_xam.md).
For a demo of using the modeling framework from start to finish, 
please see [model runner](extract_md/model_runner_xam.md).

The main modules are listed below, and there is documentation on each of them in the Developer Docs section
of the documentation.

- `core`: functional forms, objective functions, data management objects, and model parameters
- `models`: types of models to fit to the data
- `solvers`: types of solvers that are used for the optimization problem of fitting models to data
- `uncertainty`: objects for obtaining uncertainty intervals for the forecasts
- `initializer`: objects for getting "smart" priors based on other datasets
- `utils`: utility functions

For examples on how to use the code, please see the following

- [Quick Start](extract_md/get_started_xam.md)
- [Using Covariates](extract_md/covariate_xam.md)
- [Using Random Effects](extract_md/random_effect_xam.md)
- [Understanding Parametric Functions](extract_md/param_time_fun_xam.md)
- [Initializing Priors](extract_md/prior_initializer_xam.md)
- [Running Models with Predictive Validity](extract_md/model_runner_xam.md)
