# -*- coding: utf-8 -*-
import numpy as np
import starry


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
        self._A1 = map.ops.A1.eval()

    def _Ij(self, j, x):
        """
        
        """
        if j == 0:
            return 0.5 * np.pi * (1 - x ** 2)
        else:
            return (j - 1) / (j + 2) * (1 - x ** 2) * self._Ij(j - 2, x)

    def _sn(self, n, D, wsini_c):
        """
        
        """
        # This is a vector function!
        D = np.atleast_1d(D)
        res = np.zeros_like(D)
        
        # Indices
        LAM = np.floor(np.sqrt(n))
        DEL = 0.5 * (n - LAM ** 2)
        i = int(np.floor(LAM - DEL))
        j = int(np.floor(DEL))
        k = int(np.ceil(DEL) - np.floor(DEL))
        
        # x coordinate of lines of constant Doppler shift
        x = (1 / wsini_c) * (np.exp(2 * D) - 1) / (np.exp(2 * D) + 1)

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

    def _s(self, D, wsini_c):
        """
        
        """
        res = np.zeros((self.N, len(D)))
        for n in range(self.N):
            res[n] = self._sn(n, D, wsini_c)
        return res

    def g(self, D, wsini_c):
        """
        
        """
        # A1 is a sparse scipy matrix, so `*` 
        # is actually how we dot matrices!
        # NOTE: In starry, the z axis points *toward* the observer!
        return self._s(D, -wsini_c).T * self._A1