# -*- coding: utf-8 -*-
"""
    Uncertainty Estimation
"""
import numpy as np
from copy import deepcopy


def pred(x, model):
    x = x.copy()
    params = model.compute_params(x)
    return model.fun(model.t, params)


def jac_pred(x, model, eps=1e-16):
    x = x.copy()
    x_c = x + 0j
    jac_mat = np.zeros((model.num_obs, x.size))
    for i in range(x.size):
        x_c[i] += eps*1j
        jac_mat[:, i] = pred(x_c, model).imag/eps
        x_c[i] -= eps*1j
    return jac_mat


def create_fe_info_mat(model,
                       eps=1e-16,
                       add_prior=True):
    # jacobian matrix
    jac_mat = jac_pred(model.result.x, model, eps=eps)

    # fixed effects information matrix
    fe_jac_mat = jac_mat[:, :model.num_fe]
    fe_info_mat = (fe_jac_mat.T/model.obs_se**2).dot(fe_jac_mat)

    # prior information
    if add_prior:
        fe_info_mat += np.diag(1.0/model.fe_gprior[:, 1]**2)

    return fe_info_mat


def create_re_info_mat(model,
                       eps=1e-16,
                       add_prior=True):
    # jacobian matrix
    jac_mat = jac_pred(model.result.x, model, eps=eps)

    # random effects information matrix
    re_jac_mat = jac_mat[:, model.num_fe:].reshape(model.num_groups,
                                                   model.num_obs,
                                                   model.num_fe)
    re_info_mat = []
    for i in range(model.num_groups):
        sub_re_info_mat = (re_jac_mat[i].T/model.obs_se**2).dot(re_jac_mat[i])
        if add_prior:
            sub_re_info_mat += np.diag(1.0/model.re_gprior[:, 1]**2)
        re_info_mat.append(sub_re_info_mat)

    return re_info_mat


def create_vcov_mat(model_all,
                    eps=1e-16,
                    re_diag_floor=10.0,
                    re_diag_id=1,
                    add_prior=True):

    fe, re = model_all.unzip_x(model_all.result.x)
    re_empirical_mat = np.linalg.inv(np.cov(re.T))
    re_info_mat = create_re_info_mat(model_all,
                                     eps=eps,
                                     add_prior=add_prior)
    vcov_mat = []
    for i in range(model_all.num_groups):
        re_info_mat[i][re_diag_id, re_diag_id] = \
            np.maximum(re_diag_floor, re_info_mat[i][re_diag_id, re_diag_id])
        sub_vcov_mat = np.linalg.inv(
            re_empirical_mat + re_info_mat[i]
        )
        vcov_mat.append(sub_vcov_mat)

    return vcov_mat


def create_params_samples(model_all, num_draws=1000):
    vcov_mat = create_vcov_mat(model_all)
    fe, re = model_all.unzip_x(model_all.result.x)

    re_samples = np.hstack([
        np.random.multivariate_normal(re[i], vcov_mat[i], size=num_draws)
        for i in range(model_all.num_groups)
    ])

    x_samples = np.hstack([
        np.repeat(fe[None, :], num_draws, axis=0),
        re_samples
    ])

    params_samples = np.dstack([
        model_all.compute_params(x_samples[i], expand=False)
        for i in range(num_draws)
    ])

    return params_samples


def create_draws(t, model_all, num_draws=1000):
    params_samples = create_params_samples(model_all,
                                           num_draws=num_draws)
    draws = {
        name: np.vstack([
            model_all.fun(t, params)
            for params in params_samples[:, i, :].T
        ])
        for i, name in enumerate(model_all.group_names)
    }
    fe, re = model_all.unzip_x(model_all.result.x)
    re_empirical_mat = np.linalg.inv(np.cov(re.T))

    return draws


def create_draws_for_all(t, model_all, covs, num_draws=1000,
                         diag_protection=None):
    assert covs.size == model_all.num_fe
    covs = covs.reshape(1, model_all.num_fe)

    fe, re = model_all.unzip_x(model_all.result.x)
    re_empirical_cov_mat = np.cov(re.T)
    if diag_protection is not None:
        re_empirical_cov_mat = np.linalg.inv(
            np.linalg.inv(re_empirical_cov_mat) + np.diag(diag_protection))

    re_samples = np.random.multivariate_normal(np.zeros(model_all.num_fe),
                                               re_empirical_cov_mat,
                                               size=num_draws)
    fe_samples = fe + re_samples
    for i in range(model_all.num_fe):
        fe_samples[:, i] = model_all.var_link_fun[i](fe_samples[:, i])
    params_samples = covs*fe_samples
    for i in range(model_all.num_params):
        params_samples[:, i] = model_all.link_fun[i](params_samples[:, i])

    draws = np.vstack([
        model_all.fun(t, params)
        for params in params_samples
    ])

    return draws, fe_samples, params_samples


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
