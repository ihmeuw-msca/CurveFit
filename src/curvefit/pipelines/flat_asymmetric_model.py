import numpy as np
from copy import deepcopy

from curvefit.pipelines.ap_model import APModel
from curvefit.core.utils import data_translator
from curvefit.core.functions import gaussian_pdf
from curvefit.core.model import CurveModel
from curvefit.core.gauss_mix import GaussianMixture


class APFlatAsymmetricModel(APModel):
    """
    Fits an individual APModel and then uses a peak detector and spline fit to the peak
    to fit a mixture of Gaussian distributions allowing for a flat-like peak and a
    decline that does not match the incline.
    """
    def __init__(self, num_gaussians, **kwargs):
        super().__init__(**kwargs)

        self.num_gaussians = num_gaussians

        self.gaussian_mixes = {}
        for group in self.groups:
            self.gaussian_mixes[group] = GaussianMixture(num_gaussians=self.num_gaussians)

    def run_model(self, df, group):
        """Run each individual model.
        """
        model = CurveModel(
            df=df[df[self.col_group] == group].copy(),
            **self.basic_model_dict
        )

        fit_dict = deepcopy(self.fit_dict)
        fe_gprior = fit_dict['fe_gprior']
        fe_gprior[1][1] *= self.prior_modifier(model.num_obs)
        print(group)
        print('\t update beta fe_gprior to', fe_gprior)

        fit_dict.update({
            'fe_gprior': fe_gprior
        })
        model.fit_params(**fit_dict)

        self.gaussian_mixes[group].fit_gaussian_mixture(params=model.params)

        return model

    def predict(self, times, predict_space, predict_group):
        predictions = self.gaussian_mixes[predict_group].predict(times)
        predictions = data_translator(predict_space, input_space=gaussian_pdf,
                                      output_space=self.predict_space)
        return predictions

