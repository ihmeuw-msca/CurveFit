import numpy as np


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

