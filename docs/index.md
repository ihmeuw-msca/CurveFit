# Welcome to Curve Fit!

## Background

`CurveFit` is a Python package for fitting curves using nonlinear mixed effects. It can be used to do only that if desired.
However, due to its current usage for the [IHME COVID-19 project](https://covid19.healthdata.org/united-states-of-america),
 it has modules specifically for evaluating model performance out beyond the range of time observed in the data.
 Likewise, it has modules for creating uncertainty intervals based on out of sample performance.

In our [methods documentation](methods.md) we discuss the statistical methods for `CurveFit`.
In our [code documentation](code.md), we explain the core model code and also the extensions that allow for 
evaluating model performance and generating uncertainty intervals.

*NOTE: This documentation is currently under construction and being updated regularly.*

## IHME COVID-19 Project

**For any IHME COVID-19 related inquiries, please contact
 [covid19@healthdata.org](mailto:covid19@healthdata.org)**.
 
To see the IHME projections visualization, click [here](https://covid19.healthdata.org/united-states-of-america).
To read the paper, click [here](https://www.medrxiv.org/content/10.1101/2020.03.27.20043752v1). For 
FAQs, click [here](http://www.healthdata.org/covid/faqs).

Please note that this code base makes up only one part of the IHME COVID-19 projection process, in particular the
 COVID-19 deaths forecasting.

## Getting Started

To clone the repository and get
started, you can either do

```
git clone https://github.com/ihmeuw-msca/CurveFit.git
cd CurveFit
pip install .
```

or use `make install`.

## Maintainers

- [Aleksandr Aravkin](https://uw-amo.github.io/saravkin/) ([saravkin@uw.edu](mailto:saravkin@uw.edu))
- [Peng Zheng](https://zhengp0.github.io/PengSite/) ([zhengp@uw.edu](mailto:zhengp@uw.edu))
- [Marlena Bannick](http://www.healthdata.org/about/marlena-norwood) ([mbannick@uw.edu](mailto:mbannick@uw.edu))


