DATA_PATH = '../data/Slovakia_combined.csv'

PEAKED_GROUP = [
    'Wuhan City, Hubei',
    'Piemonte',
    'Comunidad de Madrid',
    'Emilia-Romagna',
    'Toscana',
    'Liguria',
    'Lombardia',
    'King and Snohomish Counties (excluding Life Care Center), WA',
]

MODEL_INFO_DICT = dict(
    col_t='Days',
    col_obs='ln(age-standardized death rate)',
    col_covs=[['intercept'], ['cov_1w'], ['intercept']],
    col_group='Location',
    param_names=['alpha', 'beta', 'p'],
    link_fun=[np.exp, lambda x: x, np.exp],
    var_link_fun=[lambda x: x, lambda x: x, lambda x: x],
    fun=curvefit.log_erf,
    col_obs_se='obs_std_tight',
)

dummy_gprior = [0.0, np.inf]
dummy_uprior = [-np.inf, np.inf]
zero_uprior = [0.0, 0.0]
fe_init = np.array([-3, 28.0, -8.05])
fe_bounds = [[-np.inf, 0.0], [15.0, 100.0], [-10, -6]]
options = {
    'ftol': 1e-10,
    'gtol': 1e-10,
    'maxiter': 500,
    'disp': True,
}

BASIC_JOINT_MODEL_FIT_DICT = dict(
    fe_gprior=[dummy_gprior]*3,
    re_bounds=[dummy_uprior]*3,
    re_gprior=[dummy_gprior, [0.0, 10.0], dummy_gprior],
    smart_initialize=True,
    smart_init_options=options,
    options={
        'ftol': 1e-10,
        'gtol': 1e-10,
        'maxiter': 10,
        'disp': True
    },
)
