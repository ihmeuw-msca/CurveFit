from utils import check_array_dims
from sklearn.linear_model import HuberRegressor

class Baseline:

    def __init__(self, observations, locations, features=None):
        if len(observations) != len(locations) or len(locations) != len(features):
            raise ValueError()
        self.observations = observations 
        self.locations = locations
        self.features = features

    def fit(self):
        raise NotImplementedError()

    def is_better_than_baseline(self, estimations, locations, metric_fun=None):
        raise NotImplementedError()

    def add_location(self, observation, location, feature=None):
        raise NotImplementedError()


class LinearRegressionBaseline(Baseline):

    def __init__(self, observations, locations, features):       
        super().__init__(observations, locations, features) 
        for obs, ft in zip(observations, features):
            if len(obs) != len(ft):
                raise ValueError()
    
    def fit(self):
        self.baseline_est = {}
        self.models = {}
        for loc, obs, ft in zip(self.locations, self.observations, self.features):
            self.models[loc] = HuberRegressor()
            self.models[loc].fit(ft, obs)
            self.baseline_est[loc] = self.models[loc].predict(ft)
    
    def is_better_than_baseline(self, estimations, locations, metric_fun=None, verbose=False):
        if len(estimations) != len(locations):
            raise ValueError()
        is_smaller = {} # better in the sense of smaller metric fun value
        metric_fun_values = {}
        if verbose:
            print('location \t curr model \t baseline')
        for loc, est in zip(locations, estimations):
            metric_fun_values[loc] = (metric_fun(est), metric_fun(self.baseline_est[loc]))
            is_smaller[loc] = metric_fun_values[loc][0] < metric_fun_values[loc][1]
            if verbose:
                print(loc, '\t', metric_fun_values[loc][0], '\t', metric_fun_values[loc][1])
        return is_smaller, metric_fun_values

    def add_location(self, observation, location, feature):
        model = HuberRegressor()
        model.fit(feature, observation)
        self.models[location] = model
        self.baseline_est[location] = model.predict(feature)
                


