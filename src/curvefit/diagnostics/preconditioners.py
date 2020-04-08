import numpy as np
import pandas as pd

from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import cross_val_score

from curvefit.core.utils import get_derivative_of_column_in_log_space


class BasicPreConditioner:
    def __init__(self, df=None, col_group=None, col_t=None, col_obs=None, **kwargs):
        self.df = df
        self.col_group = col_group
        self.col_t = col_t
        self.col_obs = col_obs
        self._supported = []

    def _check_parameters_support(self, parameters=()):
        for parameter in parameters:
            assert parameter in self._supported, "Estimation %s is not implemented yet" % parameter

    def _check_groups_support(self, groups=()):
        available_groups = set(self.df[self.col_group].unique())
        for group in groups:
            assert group in available_groups, "Group %s is not in the dataset provided" % group

    def get_estimations(self, parameters=(), groups=(), **kwargs):
        self._check_parameters_support(parameters)
        self._check_groups_support(groups)
        return None


class BetaBoundsPreConditioner(BasicPreConditioner):
    def __init__(self, peaked_groups, not_peaked_groups, **kwargs):
        super().__init__(**kwargs)
        self._supported = ["fe_bounds_beta"]
        self._features_for_beta = ["R2_full", "R2_head", "R2_tail", "R2_tail_own",
                                   "slope_full", "slope_head", "slope_tail",
                                   "fraction_below", "weighted_fraction_below",
                                   ]
        self._peaked_groups = peaked_groups
        self._not_peaked_groups = not_peaked_groups

    def get_estimations(self, parameters=(), groups=(), error_if_cross_val_score_lower_than=0.8, **kwargs):
        self._check_parameters_support(parameters)
        self._check_groups_support(groups)
        result = {}
        for parameter in parameters:
            if parameter == "fe_bounds_beta":
                train_groups = self._peaked_groups + self._not_peaked_groups
                train_dataset = self._extract_features_for_peak_estimation(for_groups=train_groups, **kwargs)
                for group in self._peaked_groups:
                    train_dataset.at[train_dataset[self.col_group] == group, "Peaked"] = 1
                for group in self._not_peaked_groups:
                    train_dataset.at[train_dataset[self.col_group] == group, "Peaked"] = 0
                model = HuberRegressor()
                x_train = train_dataset[self._features_for_beta].to_numpy()
                y_train = train_dataset["Peaked"].to_numpy()
                assert np.mean(cross_val_score(model, x_train, y_train)) < error_if_cross_val_score_lower_than, \
                    "Classifier for predicting beta bounds is too poor." \
                    " You may want to choose different peaked_groups and not_peaked_groups."
                model.fit(x_train, y_train)
                if len(groups) == 0:
                    # If no groups provided then predict for all groups in the dataset
                    groups = set(self.df[self.col_group].unique())
                predict_dataset = self._extract_features_for_peak_estimation(for_groups=groups, **kwargs).reset_index()
                x_predict = predict_dataset[self._features_for_beta].to_numpy()
                predict_dataset["Prediction"] = model.predict(x_predict)
                fe_bounds_beta = {}
                for group in groups:
                    prediction = predict_dataset.loc[predict_dataset[self.col_group] == group, "index"]["Prediction"]
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

    def _extract_features_for_peak_estimation(self, for_groups=None, tail=0.4, skip_if_shorter_than=6, **kwargs):
        groups = self.df[self.col_group].unique() if for_groups is None else for_groups
        peaking_data = pd.DataFrame({self.col_group: groups})

        for idx, row in peaking_data.iterrows():
            group = row[self.col_group]
            df_loc = self.df[self.df[self.col_group] == group]

            length = len(df_loc[self.col_t])
            if length < skip_if_shorter_than:
                continue

            model_full = HuberRegressor()

            x_tail = df_loc[self.col_t].to_numpy().reshape((-1, 1))
            y_full = df_loc[self.col_obs].to_numpy()
            model_full.fit(x_tail, y_full)
            slope_full = model_full.coef_[0]

            tail_len = int(tail * length)

            x_tail = df_loc[self.col_t].to_numpy()[-tail_len:].reshape((-1, 1))
            y_tail = df_loc[self.col_obs].to_numpy()[-tail_len:]
            x_head = df_loc[self.col_t].to_numpy()[:-tail_len].reshape((-1, 1))
            y_head = df_loc[self.col_obs].to_numpy()[:-tail_len]

            r2_full_score = model_full.score(x_tail, y_full)
            r2_head_score = model_full.score(x_head, y_head)
            r2_tail_score = model_full.score(x_tail, y_tail)

            model_head = HuberRegressor()
            model_head.fit(x_head, y_head)
            slope_head = model_head.coef_[0]

            model_tail = HuberRegressor()
            model_tail.fit(x_tail, y_tail)
            slope_tail = model_tail.coef_[0]
            peaking_data.at[idx, "R2_full"] = r2_full_score
            peaking_data.at[idx, "R2_head"] = r2_head_score
            peaking_data.at[idx, "R2_tail"] = r2_tail_score
            peaking_data.at[idx, "R2_tail_own"] = model_tail.score(x_tail, y_tail)
            peaking_data.at[idx, "slope_full"] = slope_full
            peaking_data.at[idx, "slope_head"] = slope_head
            peaking_data.at[idx, "slope_tail"] = slope_tail

            fraction_below_score = np.mean(model_full.predict(x_tail) > y_tail)
            weights = np.array([1 / (1 + i) ** 2 for i in range(1, tail_len + 1)][::-1])
            weighted_fraction_below_score = np.dot(weights, model_full.predict(x_tail) > y_tail)
            peaking_data.at[idx, "fraction_below"] = fraction_below_score
            peaking_data.at[idx, "weighted_fraction_below"] = weighted_fraction_below_score

        return peaking_data.dropna()


if __name__ == "__main__":
    # data
    file_path_data = '../data/Slovakia.csv'
    file_path_covariate = '../data/Slovakia covariate.csv'
    basic_info_dict = dict(
        all_cov_names=['cov_1w'],
        col_t='Days',
        col_group='Location',
        col_obs_compare='d ln(age-standardized death rate)',
    )
    df = pd.read_csv(file_path_data)
    df_cov = pd.read_csv(file_path_covariate)

    df_cov = df_cov.rename(columns={'location': 'Location'})
    df = pd.merge(df, df_cov, on='Location', how='inner').copy()

    df['intercept'] = 1.0
    df = get_derivative_of_column_in_log_space(df,
                                               col_t=basic_info_dict['col_t'],
                                               col_obs='ln(age-standardized death rate)',
                                               col_grp=basic_info_dict['col_group'])
    df['daily deaths'] = np.exp(df['d ' + tight_info_dict['col_obs']])