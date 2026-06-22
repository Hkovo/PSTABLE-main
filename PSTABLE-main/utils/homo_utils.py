import numpy as np
import scipy.sparse as sp


def normalize_adj(adj):
    deg1 = adj.sum(1).A1  # (N)
    deg1[deg1 == 0] = 1
    d_inv_sqrt1 = np.power(deg1, -0.5)
    D_inv_sqrt1 = sp.diags(d_inv_sqrt1)
    deg2 = adj.sum(0).A1  # (M)
    deg2[deg2 == 0] = 1
    d_inv_sqrt2 = np.power(deg2, -0.5)
    D_inv_sqrt2 = sp.diags(d_inv_sqrt2)
    return D_inv_sqrt1.dot(adj).dot(D_inv_sqrt2)


def to_homo_adj(given_hete_adjs, metapath_info):
    adj = normalize_adj(given_hete_adjs[metapath_info[0]])
    for etype in metapath_info[1:]:
        adj = adj.dot(normalize_adj(given_hete_adjs[etype]))
    # adj = adj.tolil()
    # adj[adj > 0] = 1
    return sp.csr_matrix(adj)
