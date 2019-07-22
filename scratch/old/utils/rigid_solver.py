# -*- coding: utf-8 -*-
import numpy as np
import starry
from starry.ops.utils import is_theano
import theano.tensor as tt
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix, hstack, vstack, diags
from tqdm import tqdm


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

    @property
    def lam(self):
        return self._lam
    
    @lam.setter
    def lam(self, value):
        self._lam = np.atleast_1d(value)
        self.K = self._lam.shape[0]
        assert self.K >= 3
        assert self.K % 2 == 1
        assert len(self._lam.shape) == 1
        assert np.allclose(np.diff(self._lam), self._lam[1] - self._lam[0])
        self._precompute()

    @property
    def beta(self):
        return self._beta
    
    @beta.setter
    def beta(self, value):
        self._beta = value
        self._precompute()

    @property
    def inc(self):
        return self._inc * 180.0 / np.pi
    
    @inc.setter
    def inc(self, value):
        self._inc = value * np.pi / 180.0
        self._precompute()

    @property
    def P(self):
        return self._P
    
    @P.setter
    def P(self, value):
        self._P = value

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
        D = vstack(Dt).tocsr()
        
        return D