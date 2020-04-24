import numpy as np
from curvefit.utils.data import data_translator


# TODO: ADD TESTS AND DOCUMENTATION
class Draws:
    def __init__(self, num_draws, prediction_times, exp_smoothing=None, max_last=None):

        self.num_draws = num_draws
        self.prediction_times = prediction_times
        self.exp_smoothing = exp_smoothing
        self.max_last = max_last

        assert type(self.num_draws) == int
        assert self.num_draws > 0

        assert type(self.prediction_times) == np.ndarray

        if self.exp_smoothing is not None:
            assert type(self.exp_smoothing) == float
            if self.max_last is None:
                raise RuntimeError("Need to pass in how many of the last models to use.")
            else:
                assert type(self.max_last) == int
                assert self.max_last > 0
        else:
            if self.max_last is not None:
                raise RuntimeError("Need to pass in exponential smoothing parameter.")

    # TODO: FIX THIS FUNCTION -- WILL NEED MODEL TO FIX IT
    def create_draws(self, mp, residual_model, group, prediction_times):
        data = mp.all_data.loc[mp.all_data[mp.col_group] == group].copy()
        max_t = int(np.round(data[mp.col_t].max()))
        num_obs = data.loc[~data[mp.col_obs_compare].isnull()][mp.col_group].count()

        predictions = mp.mean_predictions[group]

        add_noise = prediction_times > max_t
        no_noise = prediction_times <= max_t

        forecast_out_times = prediction_times[add_noise] - max_t
        error = residual_model.simulate_residuals(
            num_simulations=self.num_draws,
            covariate_specs={'num_data': num_obs, 'far_out': forecast_out_times}
        )
        no_error = np.zeros(shape=(self.num_draws, sum(no_noise)))
        all_error = np.hstack([no_error, error])

        noisy_forecast = predictions - (predictions ** mp.theta) * all_error
        noisy_forecast = data_translator(
            data=noisy_forecast, input_space=mp.predict_space, output_space=mp.predict_space
        )
        return noisy_forecast
