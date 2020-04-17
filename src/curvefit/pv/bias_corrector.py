from curvefit.pv.pv import PVModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np

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
        self.gp = GaussianProcessRegressor(Matern())
    
    def _adjust_prediction(self, times, residual_matrix, forecast_time_start):
        assert residual_matrix[0] == residual_matrix[1]
        n = residual_matrix.shape[0]
        prediction_matrix = np.zeros((n, n))
        if n <= forecast_time_start:
            return prediction_matrix
        for i in range(forecast_time_start + 1, n):
            y = residual_matrix[:i, :i].T
            x = np.arange(i)
            self.gp.fit(x, y)
            resi_mean_pred = self.gp.predict(times)
            prediction_matrix[i - 1, :] += resi_mean_pred[:, -1]
        return prediction_matrix

    def get_corrected_predictions(self, mp, times, predict_space, predict_group, forecast_time_start):
        residual_matrix = mp.pv.pv_groups[predict_group].residual_matrix
        return self._adjust_prediction(times, residual_matrix, forecast_time_start)



        


    
        
        


            
