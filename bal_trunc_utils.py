import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt

import sadptprj_riclyap_adi.lin_alg_utils as lau


def compute_lrbt_transfos(zfc=None, zfo=None, mmat=None,
                          trunck=dict(threshh=1e-2)):
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

    k = np.where(sv > trunck['threshh'])[0].size
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
    if ahat is None:
        ahat = np.dot(tl.T, amat*tr)
    if bhat is None:
        bhat = tl.T*bmat
    if chat is None:
        chat = cmat*tr

    NV, red_nv = amat.shape[0], ahat.shape[0]

    imunit = 1j

    absci = np.logspace(-4, 4, base=10)

    freqrel, red_freqrel = [], []

    for omega in absci:
        sadib = lau.solve_sadpnt_smw(amat=omega*imunit*mmat-amat,
                                     jmat=jmat, rhsv=bmat)
        freqrel.append(np.linalg.norm(cmat*sadib[:NV, :]))
        # print freqrel[-1]

        aib = np.linalg.solve(omega*imunit*np.eye(red_nv) - ahat, bhat)
        red_freqrel.append(np.linalg.norm(np.dot(chat, aib)))
        # print red_freqrel[-1]

    if plot:
        legstr = ['NV was {0}'.format(mmat.shape[0]),
                  'nv is {0}'.format(tr.shape[1])]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(absci, freqrel, absci, red_freqrel)
        plt.legend(legstr, loc=3)
        plt.semilogx()
        plt.semilogy()
        plt.show(block=False)
    return freqrel, red_freqrel, absci
