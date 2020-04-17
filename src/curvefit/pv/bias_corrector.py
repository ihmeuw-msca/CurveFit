from curvefit.pv.pv import PVModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
import numpy as np
import pdb


class BiasCorrector(PVModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_bias_correction(self, theta):
        print("STARTING BIAS CORRECTION...")
        self.run_pv(theta=theta)

    def get_correction(self):
        pass

    def get_corrected_predictions(self, mp, times, predict_space, predict_group, forecast_time_start):
        """
        Get corrected predictions from a model pipeline using
        the residuals from this BiasCorrector.
        Args:
            mp: (curvefit.pipelines._pipeline.ModelPipeline)
            times: (np.array) prediction times
            predict_space: (callable) space to predict in
            predict_group: (str) group to predict for
            forecast_time_start: (int) time point at which the forecasting
                starts rather than fitting

        Returns:
            (np.array) of length times
        """
        pass


class NaiveBiasCorrector(BiasCorrector):
    """
    A bias corrector that takes the average of the residuals from the bias analysis and adds it back
    on to the predictions.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.residuals_added = None

    def get_correction(self):
        mean = self.all_residuals['residual'].mean()
        self.residuals_added = mean
        return mean

    def get_corrected_predictions(self, mp, times, predict_space, predict_group, forecast_time_start):
        print(f"GETTING CORRECTED PREDICTIONS STARTING AT {forecast_time_start}")
        mean = self.get_correction()
        predictions = mp.predict(
            times=times,
            predict_space=predict_space,
            predict_group=predict_group
        )
        forecasted = times >= forecast_time_start
        predictions[forecasted] = predictions[forecasted] - (predictions[forecasted] ** mp.theta) * mean
        return predictions


class GPBiasCorrector(BiasCorrector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gp = GaussianProcessRegressor(Matern(length_scale_bounds=(1e-1, 10)), alpha=1e-2)

    # Note (jize) -- only added for sanity check. should be removed later.
    def _plot_GP(self, times, x, y, resi_mean_pred):
        plt.figure()
        plt.plot(x, y, 'r.')
        x_fine = np.linspace(times[0], times[-1], 100)
        y_fine, std = self.gp.predict(x_fine.reshape((-1, 1)), return_std=True)
        if len(y_fine.shape) > 1:
            y_fine = y_fine[:, -1]
        plt.plot(x_fine, y_fine, 'b-')
        plt.fill(
            np.concatenate([x_fine, x_fine[::-1]]), 
            np.concatenate([
                y_fine - 1.96 * std, 
                (y_fine + 1.96 * std)[::-1]
            ]),
            alpha=0.5, 
            fc='m',
            ec='None',
        )
        plt.title('time' + str(len(y) - 1))
        print('=' * 50)
        print('time', len(y) - 1)
        print('mean')
        print(y)
        print('adjust from GP (seen)')
        print(resi_mean_pred[:len(y)])
        print('adjust from GP (future)')
        print(resi_mean_pred[len(y):])
        print('=' * 50)

    # Note (jize) -- this sums up residuals across models and fit one GP.
    # not currently being used.
    def _adjust_prediction_sum(self, times, residual_matrix, plot):
        Y = residual_matrix[~np.isnan(residual_matrix).any(axis=1)]
        if len(Y) == 0:
            return np.zeros(len(times))
        weights = np.exp(np.arange(Y.shape[0]) * 0.1) / np.sum(np.exp(np.arange(Y.shape[0]) * 0.1))
        y = np.average(Y, weights=weights, axis=0)
        x = np.arange(len(y)).reshape((-1, 1))
        self.gp.fit(x, y)
        resi_mean_pred = self.gp.predict(times.reshape((-1, 1)))

        if plot:
            self._plot_GP(times, x, y, resi_mean_pred)

        return resi_mean_pred

    def _adjust_prediction(self, times, residual_matrix, plot):
        Y = residual_matrix[~np.isnan(residual_matrix).any(axis=1)].T
        if len(Y) == 0:
            return np.zeros(len(times))
        x = np.arange(Y.shape[0]).reshape((-1, 1))
        self.gp.fit(x, Y)
        resi_mean_pred = self.gp.predict(times.reshape((-1, 1)))
        if plot:
            self._plot_GP(times, x, Y[:, -1], resi_mean_pred[:, -1])
        return resi_mean_pred[:, -1]

    def get_corrected_predictions(self, mp, times, predict_space, predict_group, forecast_time_start, plot=True):
        residual_matrix = self.pv_groups[predict_group].residual_matrix
        predictions = mp.predict(
            times=times,
            predict_space=predict_space,
            predict_group=predict_group,
        )
        predictions[forecast_time_start:] -= (
            (predictions[forecast_time_start:] ** mp.theta) * 
            self._adjust_prediction(times, residual_matrix, plot)[forecast_time_start:]
        )
        return predictions      



        


    
        
        


            
