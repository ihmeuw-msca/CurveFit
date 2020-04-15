import numpy as np
import pandas as pd
from copy import deepcopy
from curvefit.pipelines.ap_model import APModel
from curvefit.diagnostics.preconditioners import BetaBoundsPreConditioner
from curvefit.core.model import CurveModel
from curvefit.core.utils import data_translator
from curvefit.core.functions import gaussian_pdf


class PreConditionedAPModel(APModel):
    """
    This class is exactly equivalent to the APModel class except that it uses BetaBoundsPreConditioner
    for local models. See docs for APModel for more detailed instructions.
    """

    def __init__(self, not_peaked_groups=None, **kwargs):
        """
        Init. Wants all the parameters which its superclass, APModel, wants, plus:

        Args:
            not_peaked_groups: List[str], Optional. List of groups which are, according to expert knowledge, or
                to a simple eye-balling, have not reached the peak yet and are still clearly soaring up. Used
                for training the internal classifier which predicts whether the peak was reached or not in a
                covariate-independent style. If None, then a pre-defined set of coefficients is used for this
                classifier (avoid using this option).
            **kwargs:
        """
        self.not_peaked_groups = not_peaked_groups
        self.init_parameters_estimations = None
        self.preconditioner = None
        super().__init__(**kwargs)

    def run_init_model(self):
        __doc__ = super().run_init_model.__doc__

        super().run_init_model()
        self.preconditioner = BetaBoundsPreConditioner(df=self.all_data,
                                                       col_group=self.col_group,
                                                       col_obs=self.col_obs,
                                                       col_t=self.col_t,
                                                       peaked_groups=self.peaked_groups,
                                                       not_peaked_groups=self.not_peaked_groups
                                                       )
        self.init_parameters_estimations = self.preconditioner.get_estimations(["fe_bounds_beta"])

    def run_model(self, df, group):
        __doc__ = super().run_model.__doc__

        model = CurveModel(
            df=df[df[self.col_group] == group].copy(),
            **self.basic_model_dict
        )

        fit_dict = deepcopy(self.fit_dict)

        print(group)
        fe_gprior = fit_dict['fe_gprior']
        common_beta_bounds = fit_dict['fe_bounds'][1]
        suggested_beta_bounds = self.init_parameters_estimations["fe_bounds_beta"].get(group, None)
        if suggested_beta_bounds is not None:
            print("\t Suggested beta bounds ", suggested_beta_bounds)
            individual_beta_bounds = [max(common_beta_bounds[0], suggested_beta_bounds[0] * 1.2),
                                      common_beta_bounds[1]]
            fit_dict['fe_bounds'][1] = individual_beta_bounds
            fe_gprior[1][0] = max(individual_beta_bounds[0], fe_gprior[1][0])
            print('\t Update beta bounds to ', individual_beta_bounds)
        else:
            print('\t Use common beta bounds ', common_beta_bounds)
            individual_beta_bounds = common_beta_bounds

        fe_gprior[1][1] *= self.prior_modifier(model.num_obs)
        print('\t Update beta fe_gprior to ', fe_gprior)
        fit_dict.update({
            'fe_gprior': fe_gprior,
            'fe_bounds': [fit_dict['fe_bounds'][0], individual_beta_bounds, fit_dict['fe_bounds'][2]]
        })
        # print(fit_dict['fe_gprior'])

        model.fit_params(**fit_dict)
        if suggested_beta_bounds is not None:
            fit_dict.update({
                'fe_bounds': [fit_dict['fe_bounds'][0], common_beta_bounds, fit_dict['fe_bounds'][2]]
            })
        return model

    def summarize_result(self, print_summary=True):
        """
        Prints a table which characterizes fit quality. It has four columns:
        Location, RMSE ERF, RMSE DERF, RMSE LNR
        Where
            - RMSE ERF: residual squares for the fit in ERF space
            - RMSE DERF: residual squares for the fit in DERF space
            - RMSE LNR: residual squares for the exponential fit in DERF space, corresponds to the linear fit in ln(DERF) space,
        The table is sorted by -ln(RMSE DERF) + ln(RMSE LNR), which means that the fits where a simple exponential
        model works better than the CurveFit (which means the fit went badly) will go first.

        Returns:
            Dataframe with the data.
        """
        models = self.models
        summary = []
        df_summary = pd.DataFrame({}, columns=['Location',
                                               'RMSE ERF',
                                               'RMSE DERF',
                                               'RMSE LNR'])
        location_list = []
        rmse_gaussian_cdf_list = []
        rmse_gaussian_pdf_list = []
        rmse_gaussian_pdf_linear_list = []
        for i, (location, model) in enumerate(models.items()):
            gaussian_cdf_pred = model.fun(model.t, model.params[:, 0])
            rmse_gaussian_cdf = np.linalg.norm(gaussian_cdf_pred - model.obs) ** 2
            gaussian_pdf_obs = data_translator(model.obs, self.basic_model_dict['fun'], 'gaussian_pdf')
            gaussian_pdf_pred = gaussian_pdf(model.t, model.params[:, 0])
            rmse_gaussian_pdf = np.linalg.norm(gaussian_pdf_obs - gaussian_pdf_pred) ** 2
            rmse_gaussian_pdf_linear = self.preconditioner._statistics["linear_rmse"].get(location, 1e10)
            summary.append([location, rmse_gaussian_cdf, rmse_gaussian_pdf, rmse_gaussian_pdf_linear])

            location_list.append(location)
            rmse_gaussian_cdf_list.append(rmse_gaussian_cdf)
            rmse_gaussian_pdf_list.append(rmse_gaussian_pdf)
            rmse_gaussian_pdf_linear_list.append(rmse_gaussian_pdf_linear)

        df_summary['Location'] = location_list
        df_summary['RMSE ERF'] = rmse_gaussian_cdf_list
        df_summary['RMSE DERF'] = rmse_gaussian_pdf_list
        df_summary['RMSE LNR'] = rmse_gaussian_pdf_linear_list

        return df_summary
