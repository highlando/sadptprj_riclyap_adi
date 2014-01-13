import scipy.linalg as spla
import scipy.sparse as sps
import numpy as np


def compute_lrbt_transfos(zfc=None, zfo=None, M=None, trunck=None):
    """
    the transformation matrices for the BT MOR

    :param zfc:
        Factor of the controllability Gramian :math:`W_c = Z_cZ_c^H`
    :param zfo:
        Factor of the observability Gramian :math:`W_o = Z_oZ_o^H`
    :param M:
        mass matrix
    :param trunck:
        truncation parameters

    :return:
        the left and right transformation matrices `tl` and `tr` \
        for the balanced truncation

    """
    if M is None:
        M = sps.eye(zfo.shape[0])

    rsv_mat, sv, lsv_matt = spla.svd(np.dot(zfc.T, M*zfo))

    if trunck is None:
        k = np.where(sv > 1e-14)[0].size
        rsvk, lsvk, svk = rsv_mat[:k, :], lsv_matt.T[:k, :], sv[:k]

    svsqri = 1./np.sqrt(svk)

    svqrsi_mat = sps.diags(svsqri, 0)

    tl = np.dot(zfc, rsvk*svqrsi_mat)
    tr = np.dot(zfo, lsvk*svqrsi_mat)

    return tl, tr
