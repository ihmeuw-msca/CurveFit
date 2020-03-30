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

We considered two functional forms so far when modeling the COVID-19 epidemic.

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

## Statistical Model

Statistical assumptions link covariates across locations. Key aspects are the following:  

- Parameters may be influenced by covariates, e.g. those that reflect social distancing

- Parameters may be modeled in a different space, e.g. \(p, \alpha\) are non-negative

- Parameters and covariate multipliers may be location-specific, with assumptions
placed on their variation.

CurveFit specification is tailored to these three requirements. Every parameter in any functional form
can be specified through a link function, covariates, fixed, and random effects. The final estimation
problem is a nonlinear mixed effects model, with user-specified priors on fixed and random effects.

For example, consider the ERF functional form with covariates \(\alpha, \beta, p\).
Assume we are fitting data in log-cumulative-death-rate space. Input data are:

- \(S_j\): social distancing covariate value at location \(j\)
- \(y_j^t\): cumulative death rate in location \(j\) at time \(t\)

We specify the statistical model as follows:

- Measurement model:
\[
\begin{aligned}
\log(y_j^t) &= \frac{p_j}{2}\left(1+ \frac{2}{\sqrt{\pi}}\int_{0}^{\alpha_j(t-\beta_j)} \exp\left(-\tau^2\right)d\tau\right) + \epsilon_{t,j} \\
\epsilon_{t,j} & \sim N(0, V_t)
\end{aligned}
\]

- \(\beta\)-model specification:
\[
\begin{aligned}
\beta_j &= \beta + \gamma_j S_j + \epsilon_j^\beta \\
\gamma_j &\sim N(\overline \gamma, V_\gamma) \\
\epsilon_j^\beta &\sim N(0, V_\beta)
\end{aligned}
\]
- \(\alpha\)-model specification:
\[
\begin{aligned}
\alpha_j &= \exp(\alpha + u_j^\alpha) \\
u_{\alpha, j} & \sim N(0, V_\alpha)
\end{aligned}
\]
- \(p\)-model specification:
\[
\begin{aligned}
p_j & = \exp(p + u_j^p) \\
u_{p,j} & \sim N(0, V_p)
\end{aligned}
\]

In this example, the user specifies

- prior mean \(\overline \gamma\)
- variance parameters \(V_t, V_\gamma, V_\beta, V_\alpha, V_p\).

CurveFit estimates:

- fixed effects \(\alpha, \beta, p\)
- random effects \(\{\gamma_j, u_j^\alpha, u_j^\beta, u_j^p\}\)

Exponential link functions are used to model non-negative parameters \(\alpha, p\).
