import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt

import sadptprj_riclyap_adi.lin_alg_utils as lau


def compute_lrbt_transfos(zfc=None, zfo=None, mmat=None, trunck=None):
    """
    the transformation matrices for the BT MOR

    :param zfc:
        Factor of the controllability Gramian :math:`W_c = Z_cZ_c^H`
    :param zfo:
        Factor of the observability Gramian :math:`W_o = Z_oZ_o^H`
    :param mmat:
        mass matrix
    :param trunck:
        truncation parameters

    :return:
        the left and right transformation matrices `tl` and `tr` \
        for the balanced truncation

    """
    if mmat is None:
        mmat = sps.eye(zfo.shape[0])

    lsv_mat, sv, rsv_matt = np.linalg.svd(np.dot(zfc.T, mmat*zfo))

    if trunck is None:
        k = np.where(sv > 1e-14)[0].size
        lsvk, rsvk, svk = lsv_mat[:, :k], rsv_matt.T[:, :k], sv[:k]

    svsqri = 1./np.sqrt(svk)

    svsqri_mat = sps.diags(svsqri, 0)

    tl = np.dot(zfo, rsvk*svsqri_mat)
    tr = np.dot(zfc, lsvk*svsqri_mat)

    return tl, tr


def compare_freqresp(mmat=None, amat=None, jmat=None, bmat=None,
                     cmat=None, tr=None, tl=None,
                     ahat=None, bhat=None, chat=None,
                     plot=False):
    """
    compare the frequency response of the original and the reduced model

    cf. [HeiSS08, p. 1059] but with B_2 = 0

    """
    # the low rank factors of the feedback gain
    NV = amat.shape[0]

    imunit = 1j

    absci = np.logspace(-4, 4, base=10)

    freqrel, red_freqrel = [], []

    for omega in absci:
        sadib = lau.solve_sadpnt_smw(amat=omega*imunit*mmat-amat,
                                     jmat=jmat, rhsv=bmat)
        freqrel.append(np.linalg.norm(cmat*sadib[:NV, :]))

        aib = np.linalg.solve(omega*imunit - ahat, bhat)
        red_freqrel.append(np.linalg.norm(np.dot(chat, aib)))

    if plot:
        plt.plot(absci, freqrel, absci, red_freqrel)
        plt.show()
    return freqrel, red_freqrel, absci
