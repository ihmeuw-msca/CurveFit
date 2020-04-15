import numpy as np
from sklearn.linear_model import HuberRegressor

class Baseline:

    def __init__(self, observations, groups, features=None):
        if len(observations) != len(groups) or len(groups) != len(features):
            raise ValueError()
        self.observations = observations 
        self.groups = groups
        self.features = features

        self.grp_to_obs = {}
        for grp, obs in zip(groups, observations):
            self.grp_to_obs[grp] = obs

    def fit(self):
        raise NotImplementedError()

    def compare(self, estimations, groups, metric_fun):
        raise NotImplementedError()

    def add_group(self, observation, group, feature=None):
        raise NotImplementedError()


class LinearRegressionBaseline(Baseline):

    def __init__(self, observations, groups, features, regressor=HuberRegressor()):       
        super().__init__(observations, groups, features) 
        for obs, ft in zip(observations, features):
            if len(obs) != len(ft):
                raise ValueError()
        self.regressor = regressor      
        self.baseline_est = {}
        self.models = {}
    
    def fit(self):
        for grp, obs, ft in zip(self.groups, self.observations, self.features):
            self.add_group(obs, grp, ft)
    
    def compare(self, estimations, groups, metric_fun):
        # Note (jize) -- metric fun takes two args (est, obs)
        if len(estimations) != len(groups):
            raise ValueError()
        metric_fun_values = {}
        for grp, est in zip(groups, estimations):
            metric_fun_values[grp] = (
                metric_fun(self.baseline_est[grp], self.grp_to_obs[grp]), 
                metric_fun(est, self.grp_to_obs[grp]),
            )
        return metric_fun_values

    def add_group(self, observation, group, feature):
        model = self.regressor
        if len(feature.shape) == 1:
            model.fit(np.reshape(feature, (-1, 1)), observation)
            self.baseline_est[group] = model.predict(np.reshape(feature, (-1, 1)))
        else:
            model.fit(feature, observation)
            self.baseline_est[group] = model.predict(feature)
        self.models[group] = model
                


