import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spmatrix
from cvxopt.solvers import qp
import seaborn as sns
from scipy.linalg import sqrtm

#import read_data
#import missing
import kmeans_greedy


def gauss_var1_process(theta, sigma, T):
    # variance of innovations is sigma * I
    #
    n, m = np.shape(theta)
    assert(n == m)
    assert(np.linalg.norm(theta, 2) < 0.9999)

    LIMIT = 2000
    x0 = np.zeros(n)
    for _ in range(LIMIT):
        eps = np.random.randn(n) * sigma
        x0 = np.dot(theta, x0) + eps

    x = np.empty((n, T))
    x[:, 0] = x0
    for i in range(1, T):
        eps = np.random.randn(n) * sigma
        x[:, i] = np.dot(theta, x[:, i-1]) + eps

    return x


def lasso_from_covariance(D, c, alpha):
    # minimizes 1/2 x^{T}Dx - c^{T} x + \alpha \| v \|_1
    #
    n = np.size(c)
    assert(np.shape(D) == (n, n))

    Q = np.zeros((2 * n, 2 * n))
    Q[:n, :n] = D

    d = np.empty(2 * n)
    d[:n] = -c
    d[n:] = alpha

    G = np.concatenate((np.concatenate((-np.identity(n), -np.identity(n)), axis=1),
                        np.concatenate((np.identity(n), -np.identity(n)), axis=1)), axis=0)
    h = np.zeros(2 * n)

    res = qp(matrix(Q), matrix(d), G=matrix(G), h=matrix(h))
    x = np.reshape(np.array(res['x'].T), (2 * n,))[:n]

    return x, res['primal objective']


def random_basis(n, k):
    assert (n >= k)
    x = np.random.randn(n, k)
    return np.matmul(x, sqrtm(np.linalg.inv(np.matmul(x.T, x))))  # orthogonal normalization of columns


def u_random(n, k, cluster_num, index=None):
    assert (cluster_num >= k)

    x = random_basis(cluster_num, k)
    if index is None:
        ind = np.random.randint(cluster_num, size=n)
    else:
        ind = index
    ind_mat = kmeans_greedy.get_index_matrix(cluster_num, ind)

    return np.matmul(ind_mat, x), ind


def v_step_normalized(D0, D1, alpha_v, u):
    # make sure u^{\T}u = I!
    #
    n, k = np.shape(u)
    assert (np.shape(D0) == (n, n) and np.shape(D1) == (n, n))
    assert (k <= n)

    v = np.empty((k, n))
    loss = 0.

    for j in range(k):
        x, _loss_j = lasso_from_covariance(D0, np.dot(D1, u[:, j]), alpha_v)
        v[j, :] = x
        loss += _loss_j

    return v, loss


def u_step_normalized(cluster_num, D1, v, ind_old=None):
    k, n = v.shape
    assert (np.shape(D1) == (n, n))
    assert (k <= n)

    mat = np.matmul(v, D1)

    #check this stupid function!!! <- apparently works...
    res = kmeans_greedy.kmeans_greedy(lambda a: np.linalg.norm(a, ord='nuc'),
                                      cluster_num, mat, init_index=ind_old)
    ind_mat = kmeans_greedy.get_index_matrix(cluster_num, res.index)

    w, s, z = np.linalg.svd(np.matmul(mat, ind_mat))
    return np.matmul(w, np.matmul(z[:k, :], ind_mat.T)).T, res.index,  np.sum(s)


class _ResultInstance:
    def __init__(self, theta, u, v, index, loss):
        self.theta = theta
        self.u = u
        self.v = v
        self.loss = loss
        self.index = index


# to implement: index choice; initial index + initial u
# matrix competition (index competition?)
#
def alternating(cluster_num, k, D0, D1, alpha_v, epochs=10, u_init=None, index_init=None):
    n, _ = np.shape(D0)
    assert (np.shape(D0) == (n, n))
    assert (np.shape(D1) == (n, n))

    if u_init is None and index_init is None:
        res = kmeans_greedy.kmeans_greedy(lambda a: np.linalg.norm(a, ord='nuc'),
                                          cluster_num, np.matmul(random_basis(n, k).T, D1))
        u_start, ind_start = u_random(n, k, cluster_num, index=res.index)
    elif index_init is None:
        u_start = u_init
    else:
        u_start, ind_start = u_random(n, k, cluster_num, index=index_init)

    for e in range(epochs):
        if e == 0:
            if u_init is None:
                u_est, ind_est = u_start, ind_start
            else:
                u_est, ind_est = u_init, None
        else:
            u_est, ind_est, loss = u_step_normalized(cluster_num, D1, v_est, ind_old=ind_est)
        v_est, loss = v_step_normalized(D0, D1, alpha_v, u_est)

    theta_est = np.matmul(u_est, v_est)
    print("loss : {}".format(loss))

    return _ResultInstance(theta_est, u_est, v_est, ind_est, loss)


def matrix_competition(repeat_num, *args, **kwargs):
    ress = []
    for _ in range(repeat_num):
        ress.append(alternating(*args, **kwargs))

    _, idx = min([(res.loss, i) for (i, res) in enumerate(ress)])
    return ress[idx]


def simu():
    k = 2
    cluster_num = k
    n = 50
    T = 100

    ind_star = np.array([0 if i < n / 2 else 1 for i in range(n)])
    u_star = kmeans_greedy.get_index_matrix(cluster_num, ind_star)

    v_star = np.zeros((k, n))
    v_star[0, 1] = .7
    v_star[1, 0] = -.5
    theta_star = np.matmul(u_star, v_star)

    x = gauss_var1_process(theta_star, 1., T)
    x_train = x[:, :-1]
    y_train = x[:, 1:]

    D0 = np.matmul(x_train, x_train.T) / (T-1)
    D1 = np.matmul(x_train, y_train.T) / (T-1)

    if True:
        epochs = 10
        alpha_v = 0.2

        res = matrix_competition(1000, cluster_num, k, D0, D1, alpha_v, epochs=epochs)
        theta_est, u_est, v_est, loss = res.theta, res.u, res.v, res.loss

        _, loss_star = v_step_normalized(D0, D1, alpha_v, u_star)
        print("loss : {}".format(loss))
        print("loss*: {}".format(loss_star))
        print(np.linalg.norm(theta_est - theta_star, ord='fro'), np.linalg.norm(theta_star, ord='fro'))

    if True:
        sns.set()
        ax = sns.heatmap(theta_est, center=0)
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=8)
        #ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        plt.show()

        sns.set()
        ax = sns.heatmap(theta_star, center=0)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=8)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        plt.show()

#simu()





