import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.svm import LinearSVC

class PeakDetector:

    def __init__(self, observations, groups, features=None):
        if len(observations) != len(groups):
            raise ValueError()
        self.observations = observations 
        self.groups = groups
        self.features = features

    def has_peaked(self, observation, group, feature=None):
        raise NotImplementedError()


class PieceWiseLinearPeakDetector(PeakDetector):

    def __init__(self, observations, groups, features, peaked, tail_prob=0.4, regressor=HuberRegressor(), classifier=LinearSVC(random_state=42)):
        super().__init__(observations, groups, features)
        if len(observations) != len(features) or len(observations) != len(peaked):
            raise ValueError()
        self.peaked = peaked 
        self.regressor = regressor
        self.classifier = classifier
        self.tail_prob = tail_prob

    def _record_regressor_fit(self, X, y, factors):
        model = self.regressor.fit(X, y)
        factors.append(model.coef_[0])
        factors.append(model.score(X, y))
        return model

    def compute_factors(self, observation, feature):
        head_len = int(len(observation) * (1 - self.tail_prob))
        obs_head = observation[:head_len]
        obs_tail = observation[head_len:]

        if len(feature.shape) == 1:
            ft = np.reshape(feature, (-1, 1))
        else:
            ft = feature
        ft_head = ft[:head_len]
        ft_tail = ft[head_len:]

        factors = []
        models = []
        for X, y in zip([ft, ft_head, ft_tail], [observation, obs_head, obs_tail]):
            model = self._record_regressor_fit(X, y, factors)
            models.append(model)

        model_full = models[0]
        factors.append(np.mean(model_full.predict(ft_tail) > obs_tail))
        weights = np.array([1 / (1 + i) ** 2 for i in range(head_len, len(observation))][::-1])
        factors.append(np.dot(weights, model_full.predict(ft_tail) > obs_tail))

        assert len(factors) == 8

        return factors 

    def train_peak_classifier(self):
        factors_matrix = []
        for obs, ft in zip(self.observations, self.features):
            factors_matrix.append(self.compute_factors(obs, ft))
        factors_matrix = np.asarray(factors_matrix)
        self.classifier.fit(factors_matrix, self.peaked)
        predictions = self.classifier.predict(factors_matrix)
        self.predicted = {self.groups[i]: predictions[i] for i in range(len(self.groups))}

    def has_peaked(self, observation, group, feature):
        factors = self.compute_factors(observation, feature)
        prediction = self.classifier.predict(np.atleast_2d(factors))
        self.predicted[group] = prediction[0]
        return prediction[0]


    




