# -*- coding: utf-8 -*-
import numpy as np
import starry
from starry.ops.utils import is_theano
import theano.tensor as tt
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix, hstack, vstack, diags
from tqdm import tqdm


class RigidRotationSolver(object):
    """
    Computes the Doppler `g` functions for a rigidly
    rotating star.
    
    """

    def __init__(self, lmax):
        """

        """
        # Degree of the Ylm expansion
        self.lmax = lmax
        self.N = (lmax + 1) ** 2

        # Grab the `A1` matrix from `starry`
        map = starry.Map(lmax, lazy=False)
        self._A1T = map.ops.A1.eval().T

        # Grab the rotation matrix op
        self._R = map.ops.R

        #
        self.kernel_width = None
        self.D = None

    def _Ij(self, j, x):
        """
        
        """
        if j == 0:
            return 0.5 * np.pi * (1 - x ** 2)
        else:
            return (j - 1) / (j + 2) * (1 - x ** 2) * self._Ij(j - 2, x)

    def _sn(self, n, lam, vsini_c):
        """
        
        """
        # This is a vector function!
        lam = np.atleast_1d(lam)
        res = np.zeros_like(lam)
        
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
        x = -(1 / vsini_c) * (np.exp(2 * lam) - 1) / (np.exp(2 * lam) + 1)

        # Integral is only nonzero when we're
        # inside the unit disk
        idx = np.abs(x) < 1
        
        # Solve the integral
        if (k == 0) and (j % 2 == 0):
            res[idx] = (2 * x[idx] ** i * \
                (1 - x[idx] ** 2) ** (0.5 * (j + 1))) / (j + 1)
        elif (k == 1) and (j % 2 == 0):
            res[idx] = x[idx] ** i * self._Ij(j, x[idx])
        
        return res

    def _s(self, lam, vsini_c):
        """
        
        """
        res = np.zeros((self.N, len(lam)))
        for n in range(self.N):
            res[n] = self._sn(n, lam, vsini_c)
        return res

    def g(self, lam, vsini_c):
        """
        Indexed [ylm, nlam].
        Normalized.

        """
        g = self._A1T.dot(self._s(lam, vsini_c))
        norm = np.trapz(g[0])
        return g / norm

    def T(self, lam, vsini_c):
        """
        Toeplitz g matrix.

        """
        g = self.g(lam, vsini_c)
        K = len(lam)
        T = [None for n in range(self.N)]
        for n in range(self.N):
            col0 = np.pad(g[n, :K // 2 + 1][::-1], (0, K // 2), mode='constant')
            row0 = np.pad(g[n, K // 2:], (0, K // 2), mode='constant')
            T[n] = csr_matrix(toeplitz(col0, row0))
        return T

    def compute(self, lam, v_c=2.e-6, inc=90.0, theta=0.0, quiet=False):
        """
        Compute the Doppler design matrix.

        """
        # Compute some stuff
        K = len(lam)
        M = len(theta)
        sini = np.sin(inc * np.pi / 180)
        cosi = np.cos(inc * np.pi / 180)
        vsini_c = v_c * sini
        theta = np.atleast_1d(theta) * np.pi / 180
        
        # Compute the kernel half-width in wavelength bins.
        # We must pad our operator by this much on either side
        # to circumvent edge effects
        dlam = (lam[1] - lam[0])
        W = (np.abs(0.5 * np.log((1 + vsini_c) / (1 - vsini_c)))) 
        W /= dlam
        W = int(np.ceil(W))
        self.kernel_width = 2 * W

        # Pad the wavelength array
        pad_l = np.linspace(lam[0] - W * dlam, lam[0], W + 1)[:-1]
        pad_r = np.linspace(lam[-1], lam[-1] + W * dlam, W + 1)[1:]
        lam_ = np.concatenate((pad_l, lam, pad_r))
        
        # Toeplitz matrices
        T = self.T(lam_, vsini_c)

        # Rotation matrices
        axis = [0, sini, cosi]
        R = [self._R(axis, t) for t in theta]

        # The design matrix
        Dt = [None for t in range(M)]
        for t in tqdm(range(M), disable=quiet):
            TR = [None for n in range(self.N)]
            for l in range(self.lmax + 1):
                idx = slice(l ** 2, (l + 1) ** 2)
                TR[idx] = np.tensordot(R[t][l].T, T[idx], axes=1)
            Dt[t] = hstack(TR)
        D = vstack(Dt).tocsr()

        # Remove the padded region we added above
        obs = np.concatenate((np.zeros_like(pad_l), 
                              np.ones_like(lam), 
                              np.zeros_like(pad_r)))
        obs = np.array(obs, dtype=bool)
        obs = np.tile(obs, M)
        self.D = D[obs]
    
    def pad(self, array, value=1.0):
        """

        """
        if is_theano(array):
            return tt.concatenate((
                        value * tt.ones(self.kernel_width // 2),
                        array,
                        value * tt.ones(self.kernel_width // 2)
                    ))
        else:
            return np.concatenate((
                        value * np.ones(self.kernel_width // 2),
                        array,
                        value * np.ones(self.kernel_width // 2)
                    ))