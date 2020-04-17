import numpy as np

from curvefit.pv.pv import PVModel


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


class ExponentialAverageBiasCorrector(BiasCorrector):
    """
    A bias corrector that uses exponential average of residuals to the past.
    """

    def __init__(self, alpha=1.5, **kwargs):
        super().__init__(**kwargs)
        self.residuals_added = None
        self.alpha = alpha

    def get_correction(self):
        """
        Calculates mean residuals.

        Returns:
            Numpy array with two columns: first -- residuals, second -- time for corresponding residuals
        """
        residuals_aggregated = self.all_residuals[["data_index", "residual"]]
        # we shift data index by one to make it consistent with self.df
        residuals_aggregated["data_index"] -= 1
        return residuals_aggregated.groupby(by="data_index").mean().join(self.df["Days"], how='inner').to_numpy()

    def get_corrected_predictions(self, mp, times, predict_space, predict_group, forecast_time_start):
        predictions = mp.predict(
            times=times,
            predict_space=predict_space,
            predict_group=predict_group
        )
        correction = self.get_correction()
        # This part is probably redundant but I keep it for safety
        last_t_used_in_pv = np.max(correction[:, 1])
        # if forecast_time_start <= last_t_used_in_pv:
        #     raise ValueError("forecast_time_start should not be smaller that the last time used in the PV matrix")
        forecasted = times >= max(forecast_time_start, last_t_used_in_pv)

        def predict_residual(t):
            weights = np.exp(self.alpha*(correction[:, 1] - t))
            return correction[:, 0].dot(weights)/weights.sum()

        forecast_corrections = np.array([predict_residual(t) for t in times[forecasted]])
        #import pdb; pdb.set_trace();
        print(len(forecast_corrections), (predictions[forecasted] ** mp.theta) * forecast_corrections)
        predictions[forecasted] = predictions[forecasted] - (predictions[forecasted] ** mp.theta) * forecast_corrections
        return predictions
