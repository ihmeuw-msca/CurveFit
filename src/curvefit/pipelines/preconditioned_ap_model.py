import numpy as np
from copy import deepcopy
from curvefit.pipelines.ap_model import APModel
from curvefit.diagnostics.preconditioners import BetaBoundsPreConditioner
from curvefit.core.model import CurveModel


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
        super().__init__(**kwargs)

    def run_init_model(self):
        __doc__ = super().run_init_model.__doc__

        super().run_init_model()
        preconditioner = BetaBoundsPreConditioner(df=self.all_data,
                                                  col_group=self.col_group,
                                                  col_obs=self.col_obs,
                                                  col_t=self.col_t,
                                                  peaked_groups=self.peaked_groups,
                                                  not_peaked_groups=self.not_peaked_groups
                                                  )
        self.init_parameters_estimations = preconditioner.get_estimations(["fe_bounds_beta"])

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
            individual_beta_bounds = [max(common_beta_bounds[0], suggested_beta_bounds[0]*1.2),
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