import numpy as np

# Model generator will need to instantiate a forecaster


class ResidualModel:
    def __init__(self, data, outcome, covariates):
        """
        Base class for a residual model. Can fit and predict out.

        Args:
            data: (pd.DataFrame) data to use
            outcome: (str) outcome column name
            covariates: List[str] covariates to predict
        """
        self.data = data
        self.outcome = outcome
        self.covariates = covariates

        assert type(self.outcome) == str
        assert type(self.covariates) == list

        self.coef = None

    def fit(self):
        pass

    def predict(self, times, predict_magnitude):
        pass


class LinearResidualModel(ResidualModel):
    def __init__(self, **kwargs):
        """
        A basic linear regression for the residuals.

        Args:
            **kwargs: keyword arguments to ResidualModel base class
        """
        super().__init__(**kwargs)

    def fit(self):
        df = self.data.copy()
        df['intercept'] = 1
        pred = np.asarray(df[['intercept'] + self.covariates])
        out = np.asarray(df[[self.outcome]])
        self.coef = np.linalg.inv(pred.T.dot(pred)).dot(pred.T).dot(out)

    def predict(self, df, predict_magnitude):
        pass


class Forecaster:
    def __init__(self, data, col_t, col_obs, col_grp):
        """
        A Forecaster will generate forecasts of residuals to create
        new, potential future datasets that can then be fit by the ModelPipeline

        Args:
            data: (pd.DataFrame) the model data
            col_t: (str) column of data that indicates time
            col_obs: (str) column of data that's in the same space
                as the forecast (linear space)
            col_grp: (str) column of data that indicates group membership
        """
        self.data = data
        self.col_t = col_t
        self.col_obs = col_obs
        self.col_grp = col_grp

        self.mean_residual_model = None
        self.std_residual_model = None

    def fit_residuals(self, residual_data, mean_outcome, std_outcome,
                      covariates, residual_model_type):
        """
        Run a regression for the mean and standard deviation
        of the scaled residuals.

        Args:
            residual_data: (pd.DataFrame) data frame of residuals
                that has the columns listed in the covariate
            mean_outcome: (str) the name of the column that has mean
                of the residuals
            std_outcome: (str) the name of the column that has the std
                of the residuals
            covariates: (str) the covariates to include in the regression
            residual_model_type: (str) what type of residual model to it
                types include 'linear'
        """
        if residual_model_type == 'linear':
            self.mean_residual_model = LinearResidualModel(data=residual_data, outcome=mean_outcome, covariates=covariates)
            self.std_residual_model = LinearResidualModel(data=residual_data, outcome=std_outcome, covariates=covariates)
        else:
            raise ValueError(f"Unknown residual model type {residual_model_type}.")

        self.mean_residual_model.fit()
        self.std_residual_model.fit()

    def predict(self, covariates):
        pass

