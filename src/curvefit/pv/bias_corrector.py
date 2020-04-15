from curvefit.pv.pv import PVModel


class BiasCorrector(PVModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_bias_correction(self, theta):
        print(f"BIAS CORRECTING")
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


class NaiveBiasCorrector(PVModel):
    """
    A bias corrector that takes the average of the residuals from the bias analysis and adds it back
    on to the predictions.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_correction(self):
        print("COMPUTING BIAS CORRECTION")
        mean = self.all_residuals['residual'].mean()
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
