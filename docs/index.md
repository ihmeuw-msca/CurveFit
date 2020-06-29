# Welcome to Curve Fit!

## Background

`CurveFit` is a primarily a Python package for fitting curves using nonlinear mixed effects.
It can be used to do only that if desired. However, due to its **former** usage for the
 [IHME COVID-19 project](https://covid19.healthdata.org/united-states-of-america),
 it has modules specifically for evaluating model performance out beyond the range of time observed in the data.
 Likewise, it has modules for creating uncertainty intervals based on out of sample performance.

In our [methods documentation](methods.md) we discuss the statistical methods for `CurveFit`.
In our [code documentation](code.md), we explain the core model code and also the extensions that allow for 
evaluating model performance and generating uncertainty intervals.

## IHME COVID-19 Project

**This repository and method is no longer being used for the IHME COVID-19 project. IHME uses
a SEIIR based approach, and the code can be found here:

- [Core Code](https://github.com/ihmeuw/covid-model-seiir)
- [Pipeline](https://github.com/ihmeuw/covid-model-seiir-pipeline)

**For any IHME COVID-19 related inquiries, please contact
 [covid19@healthdata.org](mailto:covid19@healthdata.org)**.
 
To see the IHME projections visualization, click [here](https://covid19.healthdata.org/united-states-of-america).
For FAQs, click [here](http://www.healthdata.org/covid/faqs).

## Getting Started

To clone the repository and get
started, you can either do

```
git clone https://github.com/ihmeuw-msca/CurveFit.git
cd CurveFit
make install
```

### A Note on `cppad_py`

One of the dependencies for this package is cppad_py, a python interface for algorithmic differentiation.
If you experience issues installing or importing `cppad_py` after doing `make install`, 
please see [this page](https://github.com/bradbell/cppad_py) to clone and debug the build for `cppad_py`
with `setup.py`.

## Maintainers

- [Aleksandr Aravkin](https://uw-amo.github.io/saravkin/) ([saravkin@uw.edu](mailto:saravkin@uw.edu))
- [Peng Zheng](https://zhengp0.github.io/PengSite/) ([zhengp@uw.edu](mailto:zhengp@uw.edu))
- [Marlena Bannick](http://www.healthdata.org/about/marlena-norwood) ([mbannick@uw.edu](mailto:mbannick@uw.edu))
- Jize Zhang ([jizez@uw.edu](mailto:jizez@uw.edu))
- Alexey Sholokov ([aksh@uw.edu](mailto:aksh@uw.edu))
- [Bradley Bell](https://www.seanet.com/~bradbell/home.htm) ([bradbell@seanet.com](mailto:bradbell@seanet.com))

