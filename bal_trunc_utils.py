import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt

import sadptprj_riclyap_adi.lin_alg_utils as lau


def compute_lrbt_transfos(zfc=None, zfo=None, mmat=None,
                          trunck=dict(threshh=1e-6)):
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

    Returns
    -------
    freqrel : list of floats
        the frob norm of the transferfunction at a frequency range
    red_freqrel : list of floats
        from of the tf of the reduced model at the same frequencies
    absci : list of floats
        frequencies where the tfs are evaluated at
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


def compare_stepresp(tmesh=None, a_mat=None, c_mat=None, b_mat=None,
                     m_mat=None, tl=None, tr=None,
                     iniv=None,
                     fullresp=None, fsr_soldict=None,
                     plot=True):
    """ compute the system's step response to unit inputs in time domain

    """

    from scipy.integrate import odeint

    ahat = np.dot(tl.T, a_mat*tr)
    chat = lau.matvec_densesparse(c_mat, tr)

    inivhat = np.dot(tl.T, m_mat*iniv)

    inivout = lau.matvec_densesparse(c_mat, iniv)

    # print np.linalg.norm(red_ss_rhs.flatten()), np.linalg.norm(ss_rhs)

    red_stp_rsp, ful_stp_rsp = [], []
    for ccol in range(2):  # b_mat.shape[1]):
        bmc = b_mat[:, ccol][:, :]
        red_bmc = tl.T * bmc

        def dtfunc(v, t):
            return (np.dot(ahat, v).flatten() + red_bmc.flatten())  # +\
                # red_ss_rhs.flatten())

        red_state = odeint(dtfunc, 0*inivhat.flatten(), tmesh)
        red_stp_rsp.append(np.dot(chat, red_state.T))
        ful_stp_rsp.append(fullresp(bcol=bmc, trange=tmesh, ini_vel=iniv,
                           cmat=c_mat, soldict=fsr_soldict))

    if plot:
        for ccol in range(2):  # b_mat.shape[1]):
            redoutp = red_stp_rsp[ccol].T
            fig = plt.figure(ccol)
            ax1 = fig.add_subplot(311)
            ax1.plot(tmesh, redoutp)
            fuloutp = ful_stp_rsp[ccol]
            ax2 = fig.add_subplot(312)
            ax2.plot(tmesh, fuloutp)
            fig.show()
            ax3 = fig.add_subplot(313)
            ax3.plot(tmesh, fuloutp-inivout)
            fig.show()
