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
    def __init__(self, beta_stride, mixture_size, daily_col,
                 gm_fit_threshold=10, gm_fit_dict=None, **kwargs):

        self.beta_stride = beta_stride
        self.mixture_size = mixture_size
        self.daily_col = daily_col
        self.gm_fit_threshold = gm_fit_threshold

        if gm_fit_dict is None:
            self.gm_fit_dict = {}
        else:
            self.gm_fit_dict = gm_fit_dict
        if 'bounds' in self.gm_fit_dict:
            if len(self.gm_fit_dict['bounds']) == 1:
                self.gm_fit_dict.update({
                    'bounds': np.repeat(self.gm_fit_dict['bounds'],
                                        self.mixture_size,
                                        axis=0)
                })

        self.gaussian_mixtures = {}
        super().__init__(**kwargs)

    def replace_peaked_groups(self):
        self.peaked_groups = None

    def run_model(self, df, group):
        """Run each individual model.
        """
        sub_df = df[df[self.col_group] == group].copy()
        model = CurveModel(
            df=sub_df,
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

        # Update the gaussian mixture for this particular group
        # every time the model is run
        gm = GaussianMixture(
            df=sub_df,
            col_t=self.col_t,
            col_obs=self.daily_col,
            params=model.params[:, 0],
            beta_stride=self.beta_stride,
            mixture_size=self.mixture_size,
        )
        if len(sub_df) < self.gm_fit_threshold:
            print("Too few data points to do the GM fit.")
            gm.weights = np.zeros(self.mixture_size)
            gm.weights[self.mixture_size // 2] = 1.
        else:
            gm.fit_mixture_weights(**self.gm_fit_dict)
        self.gaussian_mixtures.update({group: gm})
        return model

    def predict(self, times, predict_space, predict_group):
        """
        Predicts in the correct predict space, but first
        uses the gaussian mixture for this particular group.
        Args:
            times:
            predict_space:
            predict_group:

        Returns:

        """
        predictions = self.gaussian_mixtures[predict_group].predict(t=times)
        predictions = data_translator(
            data=predictions,
            input_space=gaussian_pdf,
            output_space=predict_space
        )
        return predictions

