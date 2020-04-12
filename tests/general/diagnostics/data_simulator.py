import numpy as np

def simulate_linear_data(n_data, n_features, noisy=False):
    coef = np.random.randn(n_features) * 2
    X = np.random.rand(n_data, n_features)
    y_true = np.dot(X, coef)
    if noisy:
        return y_true + np.random.randn(n_data) * 0.1, X, coef
    else:
        return y_true, X, coef


def simulate_linear_data_multigroups(n_groups, max_n_data, n_features, noisy=False):
    groups = np.arange(n_groups)
    ys = []
    Xs = []
    coefs = []
    for _ in range(n_groups):
        y, X, coef = simulate_linear_data(np.random.randint(low=1, high=max_n_data), n_features, noisy)
        ys.append(y)
        Xs.append(X)
        coefs.append(coef)
    return ys, Xs, groups, coefs

