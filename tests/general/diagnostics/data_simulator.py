import numpy as np

def simulate_linear_data(n_data, n_features, noisy=False):
    coef = np.random.randn(n_features) * 2
    if n_features > 1:
        X = np.random.rand(n_data, n_features) * 2
        y_true = np.dot(X, coef)
    else:
        X = np.random.randn(n_data)
        y_true = coef * X
    if noisy:
        return y_true + np.random.randn(n_data) * 0.05, X, y_true
    else:
        return y_true, X, y_true


def simulate_linear_data_multigroups(n_groups, max_n_data, n_features, noisy=False, min_n_data=1):
    groups = np.arange(n_groups)
    ys = []
    Xs = []
    ytrues = []
    for _ in range(n_groups):
        y, X, ytrue = simulate_linear_data(np.random.randint(low=min_n_data, high=max_n_data), n_features, noisy)
        ys.append(y)
        Xs.append(X)
        ytrues.append(ytrue)
    return ys, Xs, groups, ytrues

def simulate_random_data_with_labels(n_groups, max_n_data, n_features, min_n_data=6):
    ys, Xs, groups, _ = simulate_linear_data_multigroups(n_groups, max_n_data, n_features, False, min_n_data=min_n_data)
    labels = [np.random.choice([0, 1]) for _ in range(n_groups)]
    return ys, Xs, groups, labels

