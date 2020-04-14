from curvefit.pv.pv import PVModel


class BiasCorrector(PVModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_bias_correction(self, theta):
        self.run_pv(theta=theta)

    def get_corrected_predictions(self, group):
        # TODO: Implement a correction for the prediction
        # Somehow need to do different things
        # based on if the bias corrector has been run or not
        # (won't be run if there were less than look_back[0]
        # data points)
        # If it wasn't run, just return the predictions back

        # Also might want predictions to be an argument here?
        # Prediction times? Not sure.
        pass
