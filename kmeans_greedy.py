import numpy as np


def get_index_matrix(cluster_num, ind):
    n = np.size(ind)
    ind_mat = np.zeros((n, cluster_num))
    for label in range(cluster_num):
        vec = (ind == label)
        if not np.any(vec):
            continue

        vec = vec.astype(dtype=float)
        vec = vec / np.linalg.norm(vec)
        ind_mat[:, label] = vec

    return ind_mat


def proj_from_ind(k, ind):
    n = np.size(ind)
    proj = np.zeros((n, n))
    for label in range(k):
        u = (ind == label)
        if not np.any(u):
            continue

        u = u.astype(dtype=float)
        u = np.reshape(u, (n, 1))
        proj += np.matmul(u, u.T) / np.sum(u * u)

    return proj


def find_new_home(func, k, S, ind, i):
    _, n = np.shape(S)
    #assert (np.shape(S) == (n, n))

    vals = []
    for label in range(k):
        ind_new = np.array(ind, copy=True)
        ind_new[i] = label

        vals.append(func(np.matmul(S, proj_from_ind(k, ind_new))))

    (val_best, label_best) = max([(val, label) for (label, val) in enumerate(vals)])
    if val_best == vals[ind[i]]:
        return ind[i], False

    return label_best, True


def greedy_update(func, k, S, ind):
    _, n = np.shape(S)
    #assert (np.shape(S) == (n, n))
    assert (np.shape(ind) == (n,))

    ind_new = np.array(ind, copy=True)
    ans = False
    for i in range(n):
        label, ans_i = find_new_home(func, k, S, ind_new, i)
        ind_new[i] = label
        if ans_i:
            ans = True

    return ind_new, ans


class ReturnInstance:

    def __init__(self, index, proj, message):
        self.index = index
        self.proj = proj
        self.message = message


def kmeans_greedy(func, k, S, iter_limit=10000, init_index=None):
    _, n = np.shape(S)
    #assert (np.shape(S) == (n, n))
    assert (k <= n)

    if init_index is None:
        ind = np.random.randint(k, size=n)
    else:
        ind = init_index

    message = "did not converge"
    for it in range(iter_limit):
        ind, ans = greedy_update(func, k, S, ind)
        if not ans:
            message = "success in {} iterations".format(it)
            break
        print(ind)

    return ReturnInstance(ind, proj_from_ind(k, ind), message)
