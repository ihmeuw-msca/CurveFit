from sklearn.linear_model import HuberRegressor

class Baseline:

    def __init__(self, observations, groups, features=None):
        if len(observations) != len(groups) or len(groups) != len(features):
            raise ValueError()
        self.observations = observations 
        self.groups = groups
        self.features = features

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
        # have two metric functions in case baseline and passed-in estimates are not in the same space
        if len(estimations) != len(groups):
            raise ValueError()
        metric_fun_values = {}
        for grp, est in zip(groups, estimations):
            metric_fun_values[grp] = (metric_fun(self.baseline_est[grp]), metric_fun(est))
        return metric_fun_values

    def add_group(self, observation, group, feature):
        model = self.regressor
        model.fit(feature, observation)
        self.models[group] = model
        self.baseline_est[group] = model.predict(feature)
                


