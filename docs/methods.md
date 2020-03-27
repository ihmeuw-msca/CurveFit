# Overview

CurveFit is an extendable nonlinear mixed effects model or fitting curves.
The main application in this development is COVID-19 forecasting, so that
the curves we consider are variants of logistic models. However the interface
allows any user-specified parametrized family.

Parametrized curves have several key features that make them useful for forecating:  

- We can capture key signals from noisy data.
- Parameters are interpretable, and can be modeled using covariates in a transparent way.
- Parametric forms allow for more stable inversion approaches, for current and future work.
- Parametric functions impose rigid assumptions that make forecasting more stable.



## COVID-19 functional forms

We considered two functional forms so far.

- **Generalized Logistic:** \[f(t; \alpha, \beta, p)  = \frac{p}{1 + \exp(-\alpha(t-\beta))}\]


 - **Generalized Gaussian Error Function** \[
 f(t;  \alpha, \beta, p) = \frac{p}{2}\left(\Psi(\alpha(t-\beta)\right) = \frac{p}{2}\left(1+ \frac{2}{\sqrt{\pi}}\int_{0}^{\alpha(t-\beta)} \exp\left(-\tau^2\right)d\tau\right)
\]

Each form has comparable fundamental parameters:

- **Level \(p\):**  Controls the ultimate level.
- **Slope \(\alpha\)**:  Controls speed of infection.
- **Inflection \(\beta\)**: Time at which the  rate of change is maximal.   

We can fit these parameters to data, but this by itself does not account for covariates, and cannot
connect different locations together. The next section therefore specifies statistical models that do this.
