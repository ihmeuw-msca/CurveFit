# -*- coding: utf-8 -*-
"""
    Uncertainty Estimation
"""
import numpy as np
import pandas as pd
from copy import deepcopy


def create_fe_table(models):
    df_fe = pd.DataFrame({}, columns=['Location', 'fe0', 'fe1', 'fe2'])
    for location, model in models.items():
        df_fe = df_fe.append({
            'Location': location,
            'fe0': model.result.x[0],
            'fe1': model.result.x[1],
            'fe2': model.result.x[2]
        }, ignore_index=True)
    df_fe.sort_values('Location', inplace=True)
    return df_fe


def pred(x, model,
         transform_id=None,
         transform_fun=None):
    x = x.copy()
    if transform_id is not None:
        assert transform_fun is not None
        x[transform_id] = transform_fun[0](x[transform_id])
    params = model.compute_params(x)
    return model.fun(model.t, params)


def jac_pred(x, model,
             transform_id=None,
             transform_fun=None,
             eps=1e-16):
    # !! Log transform the positive parameter
    x = x.copy()
    if transform_id is not None:
        assert transform_fun is not None
        x[transform_id] = transform_fun[0](x[transform_id])
        inv_transform_fun = transform_fun[::-1]
    else:
        inv_transform_fun = None
    x_c = x + 0j
    jac_mat = np.zeros((model.num_obs, x.size))
    for i in range(x.size):
        x_c[i] += eps*1j
        jac_mat[:, i] = pred(x_c, model,
                             transform_id=transform_id,
                             transform_fun=inv_transform_fun).imag/eps
        x_c[i] -= eps*1j
    return jac_mat


def create_fe_hess_mat(model,
                       eps=1e-16,
                       add_prior=True,
                       transform_id=None,
                       transform_fun=None):
    # jacobian matrix
    jac_mat = jac_pred(model.result.x, model,
                       transform_id=transform_id,
                       transform_fun=transform_fun,
                       eps=eps)

    # fixed effects covariance matrix
    fe_jac_mat = jac_mat[:, :model.num_fe]
    fe_hess_mat = (fe_jac_mat.T/model.obs_se**2).dot(fe_jac_mat)

    # prior information
    if add_prior:
        fe_hess_mat += np.diag(1.0/model.fe_gprior[:, 1]**2)

    return fe_hess_mat


def create_cov_mat(models,
                   eps=1e-16,
                   add_prior=True,
                   transform_id=None,
                   transform_fun=None):
    # create fe result table
    df_fe = create_fe_table(models)

    # compute the empirical variance matrix
    fe_mat = df_fe[['fe0', 'fe1', 'fe2']].values
    if transform_id is not None:
        fe_mat[:, transform_id] = transform_fun[0](fe_mat[:, transform_id])

    fe_hess_mat_empirical = np.linalg.inv(np.cov(fe_mat.T))

    fe_cov_mat = {}
    for location, model in models.items():
        fe_hess_mat_empirical_location = fe_hess_mat_empirical.copy()
        # fe_hess_mat_empirical_location[:, 2] /= np.sqrt(50.0 +
        #                                                 500.0/model.num_obs**2)
        # fe_hess_mat_empirical_location[2, :] /= np.sqrt(50.0 +
        #                                                 500.0/model.num_obs**2)
        fe_hess_mat_location = create_fe_hess_mat(model,
                                                  eps=eps,
                                                  add_prior=add_prior,
                                                  transform_id=transform_id,
                                                  transform_fun=transform_fun)
        fe_cov_mat_location = np.linalg.inv(
            fe_hess_mat_location + fe_hess_mat_empirical_location
        )
        fe_cov_mat.update({
            location: fe_cov_mat_location
        })

    return fe_cov_mat


def create_draws(t, models,
                 df_fe=None,
                 cov_mat=None,
                 transform_id=None,
                 transform_fun=None,
                 num_draws=1000):
    if cov_mat is None:
        cov_mat = create_cov_mat(models,
                                 transform_id=transform_id,
                                 transform_fun=transform_fun)
    if df_fe is None:
        df_fe = create_fe_table(models)

    draws = {}
    for location, model in models.items():
        cov_mat_location = cov_mat[location]
        fe_result = \
        df_fe[df_fe['Location'] == location][['fe0', 'fe1', 'fe2']].values[0]
        if transform_id is not None:
            fe_result[transform_id] = transform_fun[0](fe_result[transform_id])
        fe_samples = fe_result + np.random.multivariate_normal(
            np.zeros(3), cov_mat[location], size=num_draws)
        # !! Exponentiate back
        if transform_id is not None:
            fe_samples[:, transform_id] = transform_fun[1](fe_samples[:, 2])
        x_samples = np.hstack([fe_samples, np.zeros((num_draws, 3))])
        params_samples = [
            model.compute_params(x_sample)[:, 0]
            for x_sample in x_samples
        ]
        draws.update({
            location: np.vstack([
                model.fun(t, params)
                for params in params_samples
            ])
        })

    return draws


def swap_cov(models, col_covs):
    new_models = {}
    for location, model in models.items():
        new_model = deepcopy(model)
        new_model.col_covs = col_covs
        new_model.covs = [
            new_model.df[name].values
            for name in new_model.col_covs
        ]
        new_model.params = new_model.compute_params(new_model.result.x)
        new_models.update({
            location: new_model
        })

    return new_models
