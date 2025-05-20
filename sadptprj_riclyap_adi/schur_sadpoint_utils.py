import time

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.sparse import save_npz  # , load_npz
# from scipy.sparse.linalg import factorized

from sksparse.cholmod import cholesky as skscholesky

import logging
logger = logging.getLogger(__name__)


def cmp_S_col_two_rl(Ainvcbcol, C, k, nnpp, writeX=False):
    # Ainvcbcol = Ainv(B[:, k].toarray().flatten())
    vallist = []
    for j in range(k, nnpp):
        if (np.mod(j+1, int(nnpp/10)) == 0 and writeX):
            # and np.mod(k, int(nnpp/10)) == 0):
            print('X ', sep=' ', end='', flush=True)
        csval = np.inner(C[j, :].toarray().flatten(), Ainvcbcol)
        vallist.append(csval.item())
    return (k, vallist)


def get_sinv(Sfac):
    def _sinv(x):
        return cho_solve((Sfac, False), x)
    return _sinv


def schur_comp_inv(f, minv=None, B=None, infoS=None, C=None,
                   S=None, M=None, sinv=None, ret_invs=False):
    nnvv, nnpp = B.shape
    if sinv is None:
        if S is None:
            try:
                S = np.load(f'{infoS}_nvnp{nnvv}{nnpp}_S.npy')
                logger.info(f'loaded: {infoS}_nvnp{nnvv}{nnpp}_S.npy')
                Sfac = cholesky(np.array(S).reshape((nnpp, nnpp)))
                sinv = get_sinv(Sfac)
            except FileNotFoundError:
                logger.info(f'not found: {infoS}_nvnp{nnvv}{nnpp}_S.npy')
                save_npz(f'{infoS}_nvnp{nnvv}{nnpp}_M.npz', M)
                save_npz(f'{infoS}_nvnp{nnvv}{nnpp}_B.npz', B)
                print('from scipy.sparse import load_npz')
                print(f'M = load_npz("{infoS}_nvnp{nnvv}{nnpp}_M.npz")')
                print(f'B = load_npz("{infoS}_nvnp{nnvv}{nnpp}_B.npz")')
                print(f'save_npz("{infoS}_nvnp{nnvv}{nnpp}_S.npz")\n\n')
                print(f'mkdir {infoS}_nvnp{nnvv}{nnpp}_cachedir/')
                print(f'python compute_S_cline.py {infoS}_nvnp{nnvv}{nnpp} ' +
                      '0 1 2 3 --nstrips 4')
                raise UserWarning('no S -- exported the mats' +
                                  'see above for instructions')
        else:
            Sfac = cholesky(np.array(S).reshape((nnpp, nnpp)))
            sinv = get_sinv(Sfac)
    else:  # sinv already available
        pass
    if minv is None:
        # minv = factorized(M)
        mfac = skscholesky(M)
        minv = mfac.solve_A

    if C is None:
        C = B.T
    else:
        raise NotImplementedError('Can only do symmetric by now')
    (_nv, _np) = B.shape
    foo = f[:_nv]
    fto = minv(foo)
    fot = f[_nv:] - C@fto
    ftt = -sinv(fot)
    fto = fto - minv(B @ ftt)
    if ret_invs:
        return np.r_[fto, ftt], sinv, minv
    else:
        return np.r_[fto, ftt]


def comp_S(M=None, B=None, minv=None, wstrips=None, nstrips=None,
           cacheonly=False, cachedir='cache/'):
    if wstrips is None:
        wstrips = range(nstrips)

    if minv is None:
        # logging.info('factorizing M with factorized... ')
        # mlu = factorized(M)
        # logging.info('done factorizing M. ')
        logging.info('factorizing M with cholmod')
        facsm = skscholesky(M)
        logging.info('done: with factorization of A')
        mlu = facsm.solve_A
    else:
        mlu = minv

    nnvv, nnpp = B.shape
    strttm = time.time()
    if not cacheonly:
        S = np.empty((nnpp*nnpp, ))

    for s in wstrips:
        logging.info(f'computing/loading S: stripes {s+1}/{nstrips}')
        for k in range(s, nnpp, nstrips):
            sstrpfnm = f'{cachedir}NVNP{nnvv}{nnpp}row{k}of{nnpp}.npy'
            try:
                v = np.load(sstrpfnm)
            except FileNotFoundError:
                opplz = False
                for i in range(nstrips):  # only for the output
                    if np.mod(k+i, int(nnpp/10)) == 0:
                        print(f'\nS{s:3.0f}/{nstrips}: ' +
                              f't: {time.time()-strttm:6.0f}: ',
                              sep=' ', end='', flush=True)
                        opplz = True
                Ainvcbcol = mlu(B[:, k].toarray().flatten())
                _, v = cmp_S_col_two_rl(Ainvcbcol, B.T, k, nnpp,
                                        writeX=opplz)
                np.save(sstrpfnm, v)
            if not cacheonly:
                for j in range(k, nnpp):
                    S[j*nnpp + k] = v[j-k]  # j-th row - k-th entry
                    S[k*nnpp + j] = v[j-k]
        print('\n')
    if cacheonly:
        return None
    else:
        return S.reshape((nnpp, nnpp))
