import numpy as np
import pandas as pd

from sklearn.linear_model import HuberRegressor
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


class BasicPreConditioner:
    """
    The purpose of a PreConditioner in the context of this project is to provide good covariate-independent
    bounds and priors for the model's parameters.
    """
    def __init__(self, df=None, col_group=None, col_t=None, **kwargs):
        self.df = df
        self.col_group = col_group
        self.col_t = col_t
        assert "d ln(age-standardized death rate)" in df.columns, \
            "d ln(age-standardized death rate) is not in the dataset. It's the space where the linear model" \
            " is always fitted, regardless of cov_obs."
        self.col_obs = "d ln(age-standardized death rate)"
        self._supported = []

    def _check_parameters_support(self, parameters=()):
        """
        Checks that all the requested parameters are in the list of supported parameters.
        Args:
            parameters: List[str]
                List of model parameters for which estimations are requested.
        Returns:
            None if all the parameters are supported, otherwise raises AssertionError
        """
        for parameter in parameters:
            assert parameter in self._supported, "Estimation %s is not implemented yet" % parameter

    def _check_groups_support(self, groups=()):
        """
        Checks that all the requested groups are in the dataframe.
        Args:
            groups: List[str]
                List of groups for which the preconditioner needs to estimate initial parameters.

        Returns:
            None if all the groups are in the dataframe, otherwise raises AssertionError
        """
        available_groups = set(self.df[self.col_group].unique())
        for group in groups:
            assert group in available_groups, "Group %s is not in the dataset provided" % group

    def get_estimations(self, parameters=(), groups=(), **kwargs):
        """
        Provides estimations of parameters for local models.
        Args:
            parameters: List[str]
                Which parameters to estimate
            groups: List[str]
                Groups for which individual models parameters are requested.
            **kwargs:

        Returns:
            Dictionary of init parameters for all parameters for all groups.
        """
        self._check_parameters_support(parameters)
        self._check_groups_support(groups)
        return None


class BetaBoundsPreConditioner(BasicPreConditioner):
    """
    Preconditioner which goal is to determine a good lower bound for beta (the coordinate of the curve peak).
    The idea is that if the data is still very well described by a linear model in a log-space, then
    the location is still on a very initial stage where the disease spreads uncontrollably, i.e. the number
    of cases grows exponentially. If it's true, then the peak is definitely past the last data point available.
    Otherwise, no constraint is provided.

    To achieve this several statistics of linear behaviour (R2, slopes, etc) are collected, and an SVM classifier
    is trained based on provided lists of peaked_groups and not_peaked groups.
    """

    def __init__(self, peaked_groups=None, not_peaked_groups=None, **kwargs):
        """
        Initializes the preconditioner.
        Args:
            df: (pd.DataFrame)
                Full dataset which includes at least columns col_group, col_t, and col_obs which
                should be "d ln(age-standardized death rate)".
            col_group:
                Group column
            col_t:
                Time column
            col_obs:
                Observations column
            **kwargs:
        """
        super().__init__(**kwargs)
        self._supported = ["fe_bounds_beta"]
        self._features_for_beta = ["R2_full", "R2_head", "R2_tail", "R2_tail_own",
                                   "slope_full", "slope_head", "slope_tail",
                                   "fraction_below", "weighted_fraction_below",
                                   ]
        self._peaked_groups = peaked_groups
        self._not_peaked_groups = not_peaked_groups
        self._statistics = {
            "linear_rmse": {},
            "linear_r2": {},
            "linear_slope": {}
        }
        # This set of coefficients defines a LinearSVC(random_state=42) classifier
        # which was trained to use features above for determining whether the peak
        # has been reached. This is not a permanent solution, rather an urgent
        # stabilizing ad-hoc.
        # These are used only if no peaked/not_peaked groups were provided,
        # otherwise (and normally) classifier is being trained each time from scratch.
        # TODO: Find better way to store classifier without getting pickle dependencies or a dump file in the repo.
        self._predefined_classifier_coefficients = {
            "coef_": np.array([[-0.32994495,
                                -0.47427811,
                                -0.33692544,
                                -0.11642247,
                                -0.50221485,
                                -0.57179481,
                                -0.41783679,
                                -0.14919415,
                                0.38239628]]),
            "intercept_": np.array([0.10990841]),
            "classes_": np.array([0, 1])
        }

    def get_estimations(self, parameters=("fe_bounds_beta", ), groups=(), error_if_cross_val_score_lower_than=0.8, **kwargs):
        """
        Estimates lower bound for beta. There is only one supported parameter: parameters=("fe_bounds_beta").
        Args:
            parameters: List[str]
                List of parameters to estimate.
                This particular class supports only one parameter estimation: "fe_bounds_beta"
            groups: List[str]
                Which groups need their local models' parameters to be estimated. If an empty list is passed
                then it estimates for all the groups in the dataset.
            error_if_cross_val_score_lower_than:
                If internal classifier was trained with cross_val_score lower than this, then
                it throws AssertionError.
            **kwargs:

        Returns:

        """
        self._check_parameters_support(parameters)
        self._check_groups_support(groups)
        result = {}
        for parameter in parameters:
            if parameter == "fe_bounds_beta":
                if self._peaked_groups is None or self._not_peaked_groups is None:
                    model = LinearSVC(random_state=42)
                    model.coef_ = self._predefined_classifier_coefficients["coef_"]
                    model.intercept_ = self._predefined_classifier_coefficients["intercept_"]
                    model.classes_ = self._predefined_classifier_coefficients["classes_"]
                else:
                    train_groups = self._peaked_groups + self._not_peaked_groups
                    train_dataset = self._extract_features_for_peak_estimation(groups=train_groups, **kwargs)
                    for group in self._peaked_groups:
                        train_dataset.at[train_dataset[self.col_group] == group, "Peaked"] = 1
                    for group in self._not_peaked_groups:
                        train_dataset.at[train_dataset[self.col_group] == group, "Peaked"] = 0
                    model = LinearSVC(random_state=42)
                    x_train = train_dataset[self._features_for_beta].to_numpy()
                    y_train = train_dataset["Peaked"].to_numpy()
                    assert np.mean(cross_val_score(model, x_train, y_train, cv=5)) >= error_if_cross_val_score_lower_than, \
                        "Classifier for predicting beta bounds is too poor." \
                        " You may want to choose different peaked_groups and not_peaked_groups."
                    model.fit(x_train, y_train)
                # print("Coefs for peaks: ", model.coef_)
                groups = self.df[self.col_group].unique() if len(groups) == 0 else groups
                predict_dataset = self._extract_features_for_peak_estimation(groups=groups, **kwargs)
                x_predict = predict_dataset[self._features_for_beta].to_numpy()
                predict_dataset["Prediction"] = model.predict(x_predict)
                fe_bounds_beta = {}
                for group in groups:
                    prediction = predict_dataset.loc[predict_dataset[self.col_group] == group]["Prediction"]
                    if len(prediction) == 0:
                        fe_bounds_beta[group] = [0, np.inf]
                    elif len(prediction) == 1:
                        fe_bounds_beta[group] = [0, np.inf] if int(prediction) == 1 \
                            else [np.max(self.df[self.df[self.col_group] == group][self.col_t]), np.inf]
                    else:
                        raise Exception("Multiple prediction for one location")
                result[parameter] = fe_bounds_beta
            else:
                raise Exception("Unsupported parameter.")
        return result

    def _extract_features_for_peak_estimation(self, groups=(), tail=0.4, skip_if_shorter_than=6, **kwargs):
        """
        Extracts features which characterize the likelihood of target value to follow a linear trend.
        Args:
            groups: List(str)
                list of groups for which to estimate
            tail:
                A fraction of timeline which we consider to be "tail". Some features describe the
                behavior of the last, tail, part of the time series with respect to the first part.
                If models for these two pieces are significantly different, then the trend has changed.
            skip_if_shorter_than:
                Skips locations (makes no prediction) for time series which are shorter than this.
            **kwargs:

        Returns:

        """
        assert 0 < tail < 1, "Tail is a fraction, it should be between 0 and 1"
        assert 0 < skip_if_shorter_than, "skip_if_shorter_than should be a positive int"

        groups = self.df[self.col_group].unique() if len(groups) == 0 else groups
        features = pd.DataFrame({self.col_group: groups})

        for idx, row in features.iterrows():
            group = row[self.col_group]
            df_loc = self.df[self.df[self.col_group] == group]

            length = len(df_loc[self.col_t])
            if length < skip_if_shorter_than:
                continue

            model_full = HuberRegressor()

            x_full = df_loc[self.col_t].to_numpy().reshape((-1, 1))
            y_full = df_loc[self.col_obs].to_numpy()
            model_full.fit(x_full, y_full)
            slope_full = model_full.coef_[0]

            tail_len = int(tail * length)

            x_tail = df_loc[self.col_t].to_numpy()[-tail_len:].reshape((-1, 1))
            y_tail = df_loc[self.col_obs].to_numpy()[-tail_len:]
            x_head = df_loc[self.col_t].to_numpy()[:-tail_len].reshape((-1, 1))
            y_head = df_loc[self.col_obs].to_numpy()[:-tail_len]

            r2_full_score = model_full.score(x_full, y_full)
            r2_head_score = model_full.score(x_head, y_head)
            r2_tail_score = model_full.score(x_tail, y_tail)

            model_head = HuberRegressor()
            model_head.fit(x_head, y_head)
            slope_head = model_head.coef_[0]

            model_tail = HuberRegressor()
            model_tail.fit(x_tail, y_tail)
            slope_tail = model_tail.coef_[0]
            features.at[idx, "R2_full"] = r2_full_score
            features.at[idx, "R2_head"] = r2_head_score
            features.at[idx, "R2_tail"] = r2_tail_score
            features.at[idx, "R2_tail_own"] = model_tail.score(x_tail, y_tail)
            features.at[idx, "slope_full"] = slope_full
            features.at[idx, "slope_head"] = slope_head
            features.at[idx, "slope_tail"] = slope_tail

            y_pred_full = model_full.predict(x_full)
            self._statistics["linear_r2"][group] = r2_full_score
            self._statistics["linear_rmse"][group] = np.linalg.norm(np.exp(y_full) - np.exp(y_pred_full))**2
            self._statistics["linear_slope"][group] = slope_full

            fraction_below_score = np.mean(model_full.predict(x_tail) > y_tail)
            weights = np.array([1 / (1 + i) ** 2 for i in range(1, tail_len + 1)][::-1])
            weighted_fraction_below_score = np.dot(weights, model_full.predict(x_tail) > y_tail)
            features.at[idx, "fraction_below"] = fraction_below_score
            features.at[idx, "weighted_fraction_below"] = weighted_fraction_below_score

        return features.dropna()
