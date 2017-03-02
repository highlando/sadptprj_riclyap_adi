import unittest

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import lin_alg_utils as lau

import proj_ric_utils as pru
# unittests for the helper functions


class TestProjLyap(unittest.TestCase):

    def setUp(self):
        self.NV = 500
        self.NP = 80
        self.NY = 5
        self.NU = self.NY+3
        self.verbose = True
        self.comprthresh = 1e-6  # threshhold for SVD trunc. for compr. of Z

        self.nwtn_adi_dict = dict(adi_max_steps=150,
                                  adi_newZ_reltol=1e-8,
                                  nwtn_max_steps=24,
                                  nwtn_upd_reltol=4e-7,
                                  nwtn_upd_abstol=4e-7,
                                  full_upd_norm_check=True,
                                  verbose=self.verbose)

        # -F, M spd -- coefficient matrices
        self.F = -sps.eye(self.NV) + 1e-5*sps.rand(self.NV, self.NV)
        self.M = sps.eye(self.NV, format='csr')
        # self.M = sps.diags([-1, 3, -1], [-1, 0, 1],
        #                    shape=(self.NV, self.NV), format='csr')
        try:
            self.Mlu = spsla.factorized(self.M.tocsc())
        except RuntimeError:
            print 'M is not full rank'

        # bmatrix that appears in the nonliner ric term X*B*B.T*X
        self.bmat = np.random.randn(self.NV, self.NU)

        # right-handside: C= -W*W.T
        self.W = np.random.randn(self.NV, self.NY)

        # smw formula Asmw = A - BV --- cannot use general UV since pymess
        self.U = self.bmat
        self.Usp = sps.csr_matrix(self.bmat)
        self.V = 1e-4 * np.random.randn(self.NU, self.NV)
        self.uvs = sps.csr_matrix(np.dot(self.U, self.V))
        self.uvssp = sps.csr_matrix(self.Usp * self.V)

        # initial value for newton adi
        self.Z0 = np.random.randn(self.NV, self.NY)

        # we need J sparse and of full rank
        for auxk in range(10):
            try:
                self.J = sps.rand(self.NP, self.NV,
                                  density=0.03, format='csr')
                spsla.splu((self.J * self.J.T).tocsc())
                break
            except RuntimeError:
                if self.verbose:
                    print 'J not full row-rank.. I make another try'
        try:
            spsla.splu((self.J * self.J.T).tocsc())
        except RuntimeError:
            raise Warning('Fail: J is not full rank')

        # the Leray projector
        MinvJt = lau.app_luinv_to_spmat(self.Mlu, self.J.T)
        Sinv = np.linalg.inv(self.J * MinvJt)
        self.P = np.eye(self.NV) - np.dot(MinvJt, Sinv * self.J)

    @unittest.skip('lets concentrate on the lyap ')
    def test_proj_lyap_sol(self):
        """check the solution of the projected lyap eqn

        via ADI iteration"""

        Z = pru.solve_proj_lyap_stein(amat=self.F, mmat=self.M,
                                      umat=self.U, vmat=self.V,
                                      jmat=self.J, wmat=self.W,
                                      adi_dict=self.nwtn_adi_dict)['zfac']

        MtXM = self.M.T * np.dot(Z, Z.T) * self.M
        FtXM = (self.F.T - self.uvs.T) * np.dot(Z, Z.T) * self.M

        PtW = np.dot(self.P.T, self.W)

        ProjRes = np.dot(self.P.T, np.dot(FtXM, self.P)) + \
            np.dot(np.dot(self.P.T, FtXM.T), self.P) + \
            np.dot(PtW, PtW.T)

        # TEST: result is 'projected'
        self.assertTrue(np.allclose(MtXM,
                                    np.dot(self.P.T, np.dot(MtXM, self.P))))

        # TEST: check projected residual
        self.assertTrue(np.linalg.norm(ProjRes) / np.linalg.norm(MtXM)
                        < 1e-8)

    @unittest.skip('lets concentrate on the lyap ')
    def test_proj_lyap_sol_pymess(self):
        """check the solution of the projected lyap eqn

        via the pymess ADI iteration"""
        import pymess

        optns = pymess.options()
        optns.adi.res2_tol = 5e-8
        optns.adi.output = 0
        optns.nm.output = 0
        optns.type = pymess.MESS_OP_NONE
        optns.adi.shifts.paratype = pymess.MESS_LRCFADI_PARA_ADAPTIVE_V
        delta = -0.02

        lyapeq = pymess.equation_lyap_dae2(optns, self.M, self.F, self.J.T,
                                           self.bmat, delta, self.V)

        Z, status = pymess.lradi(lyapeq, optns)

        MtXM = self.M.T * np.dot(Z, Z.T) * self.M
        FXM = (self.F - self.uvs) * np.dot(Z, Z.T) * self.M

        PtW = np.dot(self.P.T, self.bmat)

        ProjRes = np.dot(self.P.T, np.dot(FXM, self.P)) + \
            np.dot(np.dot(self.P.T, FXM.T), self.P) + \
            np.dot(PtW, PtW.T)

        # print np.linalg.norm(ProjRes)
        # print np.linalg.norm(MtXM)
        # print np.linalg.norm(MtXM-np.dot(self.P.T, np.dot(MtXM, self.P)))

        # TEST: result is 'projected'
        self.assertTrue(np.allclose(MtXM,
                                    np.dot(self.P.T, np.dot(MtXM, self.P))))

        # TEST: check projected residual
        self.assertTrue(np.linalg.norm(ProjRes) / np.linalg.norm(MtXM)
                        < 1e-8)

    @unittest.skip('lets concentrate on the lyap ')
    def test_proj_lyap_sol_pymess_trnsp(self):
        """check the solution of the transposed projected lyap eqn

        via the pymess ADI iteration"""
        import pymess

        optns = pymess.options()
        optns.adi.res2_tol = 5e-8
        optns.adi.output = 0
        optns.nm.output = 0
        optns.type = pymess.MESS_OP_TRANSPOSE
        optns.adi.shifts.paratype = pymess.MESS_LRCFADI_PARA_ADAPTIVE_V
        delta = -0.02

        A = self.F
        AUS = self.F - self.uvssp.T  # Note the transpose
        Pi = self.P

        lyapeq = pymess.equation_lyap_dae2(optns, self.M, A, self.J.T,
                                           self.bmat.T, delta, self.V.T)

        Z, status = pymess.lradi(lyapeq, optns)
        # solves `(\Pi A)^T X  M  +  M^T  X  (\Pi A) &=& -(C\Pi^T )^T(C\Pi^T )`

        X = np.dot(Z, Z.T)
        MXMt = self.M * X * self.M.T
        # TEST: result is 'projected'
        self.assertTrue(np.allclose(MXMt, np.dot(Pi.T, np.dot(MXMt, Pi))))

        PitAtXM = np.dot(Pi.T, AUS.T*X*self.M)
        # FtXM = (-self.F - 0*self.uvs).T * np.dot(Z, Z.T) * self.M

        PitW = np.dot(Pi.T, self.bmat)

        ProjRes = PitAtXM + PitAtXM.T + np.dot(PitW, PitW.T)

        # TEST: check projected residual
        print np.linalg.norm(ProjRes), np.linalg.norm(MXMt)
        self.assertTrue(np.linalg.norm(ProjRes) / np.linalg.norm(MXMt)
                        < 1e-8)

    @unittest.skip('lets concentrate on the lyap ')
    def test_proj_lyap_sol_sparseu(self):
        """check the solution of the projected lyap eqn

        via ADI iteration"""

        Z = pru.solve_proj_lyap_stein(amat=self.F, mmat=self.M,
                                      umat=self.Usp, vmat=self.V,
                                      jmat=self.J, wmat=self.W,
                                      adi_dict=self.nwtn_adi_dict)['zfac']

        MtXM = self.M.T * np.dot(Z, Z.T) * self.M
        FtXM = (self.F.T - self.uvssp.T) * np.dot(Z, Z.T) * self.M

        PtW = np.dot(self.P.T, self.W)

        ProjRes = np.dot(self.P.T, np.dot(FtXM, self.P)) + \
            np.dot(np.dot(self.P.T, FtXM.T), self.P) + \
            np.dot(PtW, PtW.T)

# TEST: result is 'projected'
        self.assertTrue(np.allclose(MtXM,
                                    np.dot(self.P.T, np.dot(MtXM, self.P))))

# TEST: check projected residual
        self.assertTrue(np.linalg.norm(ProjRes) / np.linalg.norm(MtXM)
                        < 1e-8)

    @unittest.skip('lets concentrate on the Ric ')
    def test_proj_lyap_smw_transposeflag(self):
        """check the solution of the projected lyap eqn

        via ADI iteration"""

        U = self.U
        V = self.V

        Z = pru.solve_proj_lyap_stein(amat=self.F, mmat=self.M,
                                      umat=U, vmat=V,
                                      jmat=self.J, wmat=self.W,
                                      adi_dict=self.nwtn_adi_dict)['zfac']

        Z2 = pru.solve_proj_lyap_stein(amat=self.F - self.uvs, mmat=self.M,
                                       jmat=self.J, wmat=self.W,
                                       adi_dict=self.nwtn_adi_dict)['zfac']

        Z3 = pru.solve_proj_lyap_stein(amat=self.F.T-self.uvs.T, mmat=self.M.T,
                                       jmat=self.J, wmat=self.W,
                                       adi_dict=self.nwtn_adi_dict,
                                       transposed=True)['zfac']

        Z4 = pru.solve_proj_lyap_stein(amat=self.F.T, mmat=self.M.T,
                                       jmat=self.J, wmat=self.W,
                                       umat=U, vmat=V,
                                       adi_dict=self.nwtn_adi_dict,
                                       transposed=True)['zfac']

# TEST: {smw} x {transposed}
        self.assertTrue(np.allclose(Z, Z2))
        self.assertTrue(np.allclose(Z2, Z3))
        self.assertTrue(np.allclose(Z3, Z4))
        self.assertTrue(np.allclose(Z, Z4))

    def test_proj_alg_ric_sol_pymess(self):
        """check the sol of the projected alg. Riccati Eqn

        via Newton ADI in pymess"""
        import pymess

        optns = pymess.options()
        optns.adi.res2_tol = 5e-8
        optns.adi.output = 0
        optns.nm.output = 0
        optns.adi.shifts.paratype = pymess.MESS_LRCFADI_PARA_ADAPTIVE_V
        delta = -0.02

        # optns.nm.K0 = None  # initial stabilizing feedback

        # for the default options, the observ. Riccati Equation is solved
        F, B, C = self.F.T, self.W, self.bmat.T
        ricceq = pymess.equation_riccati_dae2(optns, self.M, F, self.J.T,
                                              B, C, delta)

        Z, status = pymess.lrnm(ricceq, optns)
        # Z = pru.proj_alg_ric_newtonadi(mmat=self.M, amat=F,
        #                                jmat=self.J, bmat=B,
        #                                wmat=self.W,
        #                                nwtn_adi_dict=self.
        #                                nwtn_adi_dict)['zfac']

        # TEST: check projected residual - riccati sol
        FtXM = np.dot(self.F.T * Z, Z.T * self.M)
        PiFtXM = np.dot(self.P.T, FtXM)
        PtW = np.dot(self.P.T, self.W)
        MtXb = np.dot(self.M.T*Z, np.dot(Z.T, self.bmat))

        ProjRes = PiFtXM + PiFtXM.T - np.dot(MtXb, MtXb.T) + np.dot(PtW, PtW.T)

        MtXM = self.M.T * np.dot(Z, Z.T) * self.M
        self.assertTrue(np.linalg.norm(ProjRes) / np.linalg.norm(MtXM)
                        < 1e-7)
# TEST: result is 'projected' - riccati sol
        self.assertTrue(np.allclose(MtXM,
                                    np.dot(self.P.T, np.dot(MtXM, self.P))))

        # with the transpose option, the observ. Riccati Equation is solved
        optns.type = pymess.MESS_OP_TRANSPOSE
        F, B, C = self.F, self.bmat, self.W.T
        ricceq = pymess.equation_riccati_dae2(optns, self.M, F, self.J.T,
                                              B, C, delta)

        FtXM = np.dot(self.F.T * Z, Z.T * self.M)
        PiFtXM = np.dot(self.P.T, FtXM)
        PtW = np.dot(self.P.T, self.W)
        MtXb = np.dot(self.M.T*Z, np.dot(Z.T, self.bmat))

        ProjRes = PiFtXM + PiFtXM.T - np.dot(MtXb, MtXb.T) + np.dot(PtW, PtW.T)

        self.assertTrue(np.linalg.norm(ProjRes) / np.linalg.norm(MtXM)
                        < 1e-7)

        self.assertTrue(np.allclose(MtXM,
                                    np.dot(self.P.T, np.dot(MtXM, self.P))))

    def test_proj_alg_ric_myvspy_mess(self):
        """check the sol of the projected alg. Riccati Eqn

        via Newton ADI  -- pymess vs. my mess"""
        # Z = pru.proj_alg_ric_newtonadi(mmat=self.M, amat=self.F,
        #                                jmat=self.J, bmat=self.bmat,
        #                                wmat=self.W, z0=self.bmat,
        #                                nwtn_adi_dict=self.
        #                                nwtn_adi_dict)['zfac']

        Zpm = pru.\
            pymess_dae2_cnt_riccati(mmat=self.M, amat=self.F,
                                    jmat=self.J, bmat=self.bmat,
                                    wmat=self.W, z0=self.bmat)['zfac']

        # for '0' initial value --> z0 = None
        Z = pru.proj_alg_ric_newtonadi(mmat=self.M, amat=self.F,
                                       jmat=self.J, bmat=self.bmat,
                                       wmat=self.W,
                                       nwtn_adi_dict=self.
                                       nwtn_adi_dict)['zfac']

        MtXM = self.M.T * np.dot(Z, Z.T) * self.M
        MtXpmM = self.M.T * np.dot(Zpm, Zpm.T) * self.M

        self.assertTrue(np.allclose(MtXM, MtXpmM))

        MtXb = self.M.T * np.dot(np.dot(Z, Z.T), self.bmat)

        FtXM = self.F.T * np.dot(Z, Z.T) * self.M
        PtW = np.dot(self.P.T, self.W)

        ProjRes = np.dot(self.P.T, np.dot(FtXM, self.P)) + \
            np.dot(np.dot(self.P.T, FtXM.T), self.P) -\
            np.dot(MtXb, MtXb.T) + \
            np.dot(PtW, PtW.T)

# TEST: result is 'projected' - riccati sol
        self.assertTrue(np.allclose(MtXM,
                                    np.dot(self.P.T, np.dot(MtXM, self.P))))

# TEST: check projected residual - riccati sol
        print np.linalg.norm(ProjRes) / np.linalg.norm(MtXM)

        self.assertTrue(np.linalg.norm(ProjRes) / np.linalg.norm(MtXM)
                        < 1e-7)

    @unittest.skip('lets concentrate on the lyap ')
    def test_compress_algric_Z(self):
        Z = pru.proj_alg_ric_newtonadi(mmat=self.M, amat=self.F,
                                       jmat=self.J, bmat=self.bmat,
                                       wmat=self.W, z0=self.bmat,
                                       nwtn_adi_dict=self.
                                       nwtn_adi_dict)['zfac']

        Zred = pru.compress_Zsvd(Z, thresh=self.comprthresh)

        print '\ncompressing Z from {0} to {1} columns:'.\
            format(Z.shape[1], Zred.shape[1])

        difn, zzn, zzrn = \
            lau.comp_sqfnrm_factrd_diff(Z, Zred, ret_sing_norms=True)

        print '\n || ZZ - ZZred||_F || / ||ZZ|| = {0}\n'.\
            format(np.sqrt(difn/zzn))

        vec = np.random.randn(Z.shape[0], 1)

        print '||(ZZ_red - ZZ )*testvec|| / ||ZZ*testvec|| = {0}'.\
            format(np.linalg.norm(np.dot(Z, np.dot(Z.T, vec)) -
                   np.dot(Zred, np.dot(Zred.T, vec))) /
                   np.linalg.norm(np.dot(Zred, np.dot(Zred.T, vec))))

        self.assertTrue(True)

suite = unittest.TestLoader().loadTestsFromTestCase(TestProjLyap)
unittest.TextTestRunner(verbosity=2).run(suite)
