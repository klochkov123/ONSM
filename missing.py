import numpy as np
from scipy.stats import norm


def get_missing_probabilities(mat, conf=0.95, tol=0.1):
    p, n = np.shape(mat)

    delta = np.mean(np.array(np.abs(mat) > 0.00000001, dtype=float), axis=1)
    stderr = np.sqrt((delta - delta * delta) / n) * norm.ppf(conf)
    idx = np.where(delta * tol >= stderr / 2)[0]

    return delta, stderr, idx


def missing_var(X, delta):
    n = np.shape(X)[1]
    S = np.matmul(X, X.T) / n
    D = np.diag(np.diag(S))
    Delta = np.diag(delta)
    return np.matmul(Delta, np.matmul(S - D, Delta)) + np.matmul(Delta, D)


def missing_covar(X, Y, delta_x, delta_y):
    assert(np.shape(X)[1] == np.shape(Y)[1])
    n = np.shape(X)[1]

    return np.matmul(np.matmul(np.diag(delta_x), X), np.matmul(Y.T, np.diag(delta_y))) / n
