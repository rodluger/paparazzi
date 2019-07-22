# -*- coding: utf-8 -*-
import numpy as np
import starry
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix, hstack, vstack, diags, block_diag
from tqdm import tqdm
import celerite
from scipy.linalg import cho_factor, cho_solve


__all__ = ["Doppler", "LinearSolver"]


class Doppler(object):
    """
    Computes the Doppler `g` functions for a rigidly
    rotating star.
    
    """

    def __init__(self, lam, ydeg=10, beta=0, inc=90.0, P=1.0):
        """

        """
        # Degree of the Ylm expansion
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2

        # Grab the `A1` matrix from `starry`
        map = starry.Map(ydeg, lazy=False)
        self._A1T = map.ops.A1.eval().T

        # Grab the rotation matrix op
        self._R = map.ops.R

        # Doppler operator props
        self._beta = beta
        self._inc = inc * 180.0 / np.pi
        self._P = P
        self.lam = lam
        self._compute = True
        self._t = np.array([])

    @property
    def lam(self):
        return self._lam
    
    @lam.setter
    def lam(self, value):
        self._lam = np.atleast_1d(value)
        self.K = self._lam.shape[0]
        assert self.K >= 3, "The wavelength grid must have at least 3 bins."
        assert self.K % 2 == 1, "The number of wavelength bins must be odd."
        assert len(self._lam.shape) == 1, "Invalid wavelength grid shape."
        assert np.allclose(np.diff(self._lam), self._lam[1] - self._lam[0])
        self._precompute()
        self._compute = True

    @property
    def beta(self):
        return self._beta
    
    @beta.setter
    def beta(self, value):
        self._beta = value
        self._precompute()
        self._compute = True

    @property
    def inc(self):
        return self._inc * 180.0 / np.pi
    
    @inc.setter
    def inc(self, value):
        self._inc = value * np.pi / 180.0
        self._precompute()
        self._compute = True

    @property
    def P(self):
        return self._P
    
    @P.setter
    def P(self, value):
        self._P = value
        self._compute = True

    @property
    def mask(self):
        return self._lam_mask
    
    @property
    def lam_padded(self):
        return self._lam_padded

    def _precompute(self):
        # Inclination/velocity stuff
        sini = np.sin(self._inc)
        cosi = np.cos(self._inc)
        self._betasini = self._beta * sini
        self._axis = [0, sini, cosi]

        # Kernel width
        W = (np.abs(0.5 * np.log((1 + self._betasini) / (1 - self._betasini)))) 
        dlam = self.lam[1] - self.lam[0]
        self._kernel_width = 2 * int(np.ceil(W / dlam))

        # Pad the wavelength array to circumvent edge effects
        dlam = self.lam[1] - self.lam[0]
        W = self._kernel_width // 2
        pad_l = np.linspace(self.lam[0] - W * dlam, self.lam[0], W + 1)[:-1]
        pad_r = np.linspace(self.lam[-1], self.lam[-1] + W * dlam, W + 1)[1:]
        self._lam_padded = np.concatenate((pad_l, self.lam, pad_r))
        self.Kp = self._lam_padded.shape[0]
        self._lam_mask = np.concatenate((np.zeros_like(pad_l), 
                                         np.ones_like(self.lam), 
                                         np.zeros_like(pad_r)))
        self._lam_mask = np.array(self._lam_mask, dtype=bool)

    def _Ij(self, j, x):
        """
        
        """
        if j == 0:
            return 0.5 * np.pi * (1 - x ** 2)
        else:
            return (j - 1) / (j + 2) * (1 - x ** 2) * self._Ij(j - 2, x)

    def _sn(self, n):
        """
        
        """
        # Indices
        LAM = np.floor(np.sqrt(n))
        DEL = 0.5 * (n - LAM ** 2)
        i = int(np.floor(LAM - DEL))
        j = int(np.floor(DEL))
        k = int(np.ceil(DEL) - np.floor(DEL))
        
        # x coordinate of lines of constant Doppler shift
        # NOTE: In starry, the z axis points *toward* the observer,
        # which is the opposite of the convention used for Doppler
        # shifts, so we need to include a factor of -1 below.
        x = -(1 / self._betasini) * (np.exp(2 * self._lam_padded) - 1) / \
             (np.exp(2 * self._lam_padded) + 1)

        # Integral is only nonzero when we're
        # inside the unit disk
        idx = np.abs(x) < 1
        
        # Solve the integral
        res = np.zeros(self.Kp)
        if (k == 0) and (j % 2 == 0):
            res[idx] = (2 * x[idx] ** i * \
                (1 - x[idx] ** 2) ** (0.5 * (j + 1))) / (j + 1)
        elif (k == 1) and (j % 2 == 0):
            res[idx] = x[idx] ** i * self._Ij(j, x[idx])
        
        return res

    def _g(self):
        """
        Indexed [ylm, nlam].
        Normalized.

        """
        g = self._A1T.dot(np.array([self._sn(n) for n in range(self.N)]))
        norm = np.trapz(g[0])
        return g / norm

    def _T(self):
        """
        Toeplitz g matrix.

        """
        g = self._g()
        T = [None for n in range(self.N)]
        for n in range(self.N):
            col0 = np.pad(g[n, :self.Kp // 2 + 1][::-1], 
                          (0, self.Kp // 2), mode='constant')
            row0 = np.pad(g[n, self.Kp // 2:], 
                          (0, self.Kp // 2), mode='constant')
            T[n] = csr_matrix(toeplitz(col0, row0)[self._lam_mask])
        return T

    def D(self, t=0.0, quiet=False):
        """
        Compute the Doppler design matrix.

        """

        if (self._compute) or (not np.array_equal(t, self._t)):

            # Get the angular phase
            M = len(t)
            theta = (2 * np.pi / self._P * t) % (2 * np.pi)
            
            # Toeplitz matrices
            T = self._T()

            # Rotation matrices
            R = [self._R(self._axis, t) for t in theta]

            # The design matrix
            Dt = [None for t in range(M)]
            for t in tqdm(range(M), disable=quiet):
                TR = [None for n in range(self.N)]
                for l in range(self.ydeg + 1):
                    idx = slice(l ** 2, (l + 1) ** 2)
                    TR[idx] = np.tensordot(R[t][l].T, T[idx], axes=1)
                Dt[t] = hstack(TR)
            self._D = vstack(Dt).tocsr()

            self._t = t
            self._compute = False

        return self._D


class LinearSolver(object):
    """

    """

    def __init__(self, lam, D, F, F_sig, N, Kp, u_sig, u_mu, vT_sig, vT_rho, vT_mu):
        """

        """
        self.lam = lam
        self.D = D
        self.F = F
        self.N = N
        self.Kp = Kp
        self.f = self.F.reshape(-1, 1)
        if np.ndim(F_sig) == 0:
            self.CInv = np.ones_like(self.f) / F_sig ** 2
        elif np.ndim(F_sig) == 1:
            self.CInv = np.array([np.ones(self.f.shape[1]) / F_sig ** 2 
                                  for n in range(self.f.shape[0])])
        else:
            self.CInv = F_sig.reshape(-1, 1) ** -2
        self.CInv = self.CInv.reshape(-1)
        self.lndet2pC = np.sum(np.log(2 * np.pi * self.CInv))

        # L2 (gaussian) prior on u
        self.u_CInv = np.ones(self.N) / u_sig ** 2
        self.u_CInvmu = (self.u_CInv * u_mu).reshape(-1, 1)
        self.lndet2pC_u = np.sum(np.log(2 * np.pi * self.u_CInv))
        self.u_mu = (np.ones(self.N) * u_mu).reshape(-1, 1)

        # Gaussian process prior on vT
        self.vT_mu = (np.ones(self.Kp) * vT_mu).reshape(1, -1)
        if vT_rho > 0.0:
            kernel = celerite.terms.Matern32Term(np.log(vT_sig), np.log(vT_rho))
            gp = celerite.GP(kernel)
            vT_C = gp.get_matrix(self.lam)
            cho_C = cho_factor(vT_C)
            self.vT_CInv = cho_solve(cho_C, np.eye(self.Kp))
            self.vT_CInvmu = cho_solve(cho_C, self.vT_mu.reshape(-1)).reshape(-1, 1)
            self.lndet2pC_vT = 2 * np.sum(np.log(2 * np.pi * np.diag(cho_C[0])))
        else:
            self.vT_CInv = np.ones(self.Kp) / vT_sig ** 2
            self.vT_CInvmu = (self.vT_CInv * self.vT_mu).reshape(-1, 1)
            self.lndet2pC_vT = np.sum(np.log(2 * np.pi * self.vT_CInv))
            self.vT_CInv = np.diag(self.vT_CInv)

    def u(self, vT):
        """
        Linear solve for `u` given `v^T`.

        """
        V = block_diag([vT.reshape(-1, 1) for n in range(self.N)])
        A = np.array(self.D.dot(V).todense())
        ATCInv = np.multiply(A.T, self.CInv)
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, self.f)
        np.fill_diagonal(ATCInvA, ATCInvA.diagonal() + self.u_CInv)
        return np.linalg.solve(ATCInvA, ATCInvf + self.u_CInvmu).reshape(-1, 1)

    def vT(self, u):
        """
        Linear solve for `v^T` given `u`.

        """
        offsets = -np.arange(0, self.N) * self.Kp
        U = diags([np.ones(self.Kp) * u[n] 
                    for n in range(self.N)], offsets, 
                    shape=(self.N * self.Kp, self.Kp))
        A = np.array(self.D.dot(U).todense())
        ATCInv = np.multiply(A.T, self.CInv)
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, self.f)
        return np.linalg.solve(ATCInvA + self.vT_CInv, 
                               ATCInvf + self.vT_CInvmu).reshape(1, -1)
    
    def step(self, u=None, vT=None):
        """

        """
        # Take a step
        if vT is not None:
            u = self.u(vT)
            vT = self.vT(u)
        else:
            vT = self.vT(u)
            u = self.u(vT)
        
        # Compute the model & likelihood
        a = u.dot(vT).reshape(-1)
        model = self.D.dot(a)
        r = self.f.reshape(-1) - model.reshape(-1)
        lnlike = -0.5 * np.dot(
                            np.multiply(r.reshape(1, -1), self.CInv),
                            r.reshape(-1, 1)
                        ) - 0.5 * self.lndet2pC
        r = u - self.u_mu
        lnprior_u = -0.5 * np.dot(
                            np.multiply(r.reshape(1, -1), self.u_CInv),
                            r.reshape(-1, 1)
                        ) - 0.5 * self.lndet2pC_u
        r = vT - self.vT_mu
        lnprior_vT = -0.5 * np.dot(
                            np.dot(r.reshape(1, -1), self.vT_CInv),
                            r.reshape(-1, 1)
                        ) - 0.5 * self.lndet2pC_vT
        return u, vT, model, lnlike, lnprior_u + lnprior_vT
    
    def solve(self, u=None, vT=None, maxiter=200, quiet=False):
        """
        
        """
        lnlike = np.zeros(maxiter)
        lnprior = np.zeros(maxiter)
        for n in tqdm(range(maxiter), total=maxiter, disable=quiet):
            u, vT, model, lnlike[n], lnprior[n] = self.step(u, vT)
        
        # TODO: Convergence criterion

        print((lnlike + lnprior)[-1])

        return u, vT, model, lnlike, lnprior
