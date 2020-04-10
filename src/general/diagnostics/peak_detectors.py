import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.svm import LinearSVC

class PeakDetector:

    def __init__(self, observations, features=None):
        self.observations = observations 
        self.features = features

    def has_peaked(self, group):
        raise NotImplementedError()


class LinearPeakDetector(PeakDetector):

    def __init__(self, observations, features, peaked, regressor=HuberRegressor(), classifier=LinearSVC(random_state=42)):
        super().__init__()
        if len(observations) != len(features) or len(observations) != len(peaked):
            raise ValueError()
        self.peaked = peaked 
        self.regressor = regressor
        self.classifier = classifier

    def compute_factors(self, observation, feature, tail_prob=0.4):
        factors = []
        model_full = self.regressor.fit(feature, observation)
        factors.append(model_full._coef[0])
        factors.append(model_full.score(feature, observation))

        head_len = int(len(observation) * (1 - tail_prob))
        obs_head = observation[:head_len]
        ft_head = feature[:head_len]
        obs_tail = observation[head_len:]
        ft_tail = feature[head_len:]

        model_head = self.regressor.fit(ft_head, obs_head)
        factors.append(model_head._coef[0])
        factors.append(model_head.score(ft_head, obs_head))

        model_tail = self.regressor.fit(ft_tail, obs_tail)
        factors.append(model_tail._coef[0])
        factors.append(model_tail.score(ft_tail, obs_tail))

        factors.append(np.mean(model_full.predict(ft_tail) > obs_tail))
        weights = np.array([1 / (1 + i) ** 2 for i in range(head_len, len(observation))][::-1])
        factors.append(np.dot(weights, model_full.predict(ft_tail) > obs_tail))

        assert len(factors) == 8

        return factors 

    def train_peak_classifier(self, tail_prob=0.4):
        factors_matrix = []
        for obs, ft in zip(self.observations, self.features):
            factors_matrix.append(self.compute_factors(obs, ft, tail_prob=tail_prob))
        factors_matrix = np.asarray(factors_matrix)
        self.classifier.fit(factors_matrix, self.peaked)
        self.predicted = self.classifier.predict(factors_matrix)

    def has_peaked(self, observation, feature):
        factors = self.compute_factors(observation, feature)
        return self.classifier.predict(factors)


    




