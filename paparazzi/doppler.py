# -*- coding: utf-8 -*-
import numpy as np
import starry
from scipy.sparse import csr_matrix, hstack, vstack, diags
from scipy.sparse import block_diag as sparse_block_diag
from scipy.linalg import block_diag as dense_block_diag
from scipy.linalg import cho_factor, cho_solve
import celerite
from tqdm import tqdm
import os
from .utils import Adam


__all__ = ["Doppler"]


class Doppler(object):
    """
    
    """

    def __init__(self, ydeg=15, vsini=40, inc=40.0):
        """

        """
        self.ydeg = ydeg
        self.inc = inc
        self.vsini = vsini
        self._reset_data()
    
    def _set_lnlam(self, lnlam):
        # Set the log-wavelength grid
        self._lnlam = np.atleast_1d(lnlam)
        self.K = self._lnlam.shape[0]
        assert self.K >= 3, "The wavelength grid must have at least 3 bins."
        assert self.K % 2 == 1, "The number of wavelength bins must be odd."
        assert len(self._lnlam.shape) == 1, "Invalid wavelength grid shape."
        assert np.allclose(np.diff(self._lnlam), 
                           self._lnlam[1] - self._lnlam[0])
        self._reset_cache()

    def _set_theta(self, theta):
        # Set the angular phase
        self._theta = np.atleast_1d(theta) * np.pi / 180.
        self.M = len(theta)
        self._reset_cache()

    @property
    def ydeg(self):
        """
        Degree of the spherical harmonic expansion.

        """
        return self._ydeg
    
    @ydeg.setter
    def ydeg(self, value):
        # Degree of the Ylm expansion
        self._ydeg = value
        self.N = (value + 1) ** 2

        # Grab the `A1` matrix from `starry`
        self.map = starry.Map(value)
        self._A1T = self.map.ops.A1.eval().T

        # Grab the rotation matrix op
        self._R = self.map.ops.R

        self._reset_cache()

    @property
    def vsini(self):
        """
        The projected equatorial radial velocity in km/s.
            
        """
        return self._beta * 3.e5 * np.sin(self._inc)
    
    @vsini.setter
    def vsini(self, value):
        self._beta = value / np.sin(self._inc) / 3.e5
        self._reset_cache()

    @property
    def inc(self):
        """
        The stellar inclination in degrees.
            
        """
        return self._inc * 180.0 / np.pi
    
    @inc.setter
    def inc(self, value):
        self._inc = value * np.pi / 180.0
        self.map.inc = value
        self._reset_cache()

    @property
    def lnlam(self):
        """
        The log-wavelength grid.
            
        """
        if self._lnlam is None:
            raise ValueError("Please load a dataset first.")
        return self._lnlam
    
    @property
    def lnlam_padded(self):
        """
        The padded log-wavelength grid.
            
        """
        if self._lnlam_padded is None:
            # Kernel width
            betasini = self._beta * np.sin(self._inc)
            hw = (np.abs(0.5 * np.log((1 + betasini) / 
                                      (1 - betasini)))) 
            dlam = self.lnlam[1] - self.lnlam[0]
            self.W = 2 * int(np.ceil(hw / dlam)) + 1

            # Pad the wavelength array to circumvent edge effects
            dlam = self.lnlam[1] - self.lnlam[0]
            hw = (self.W - 1) // 2
            pad_l = np.linspace(self.lnlam[0] - hw * dlam, 
                self.lnlam[0], hw + 1)[:-1]
            pad_r = np.linspace(self.lnlam[-1], 
                self.lnlam[-1] + hw * dlam, hw + 1)[1:]
            self._lnlam_padded = np.concatenate((pad_l, self.lnlam, pad_r))

        return self._lnlam_padded

    @property
    def x(self):
        """
        The `x` coordinate of lines of constant Doppler shift.
        
        """
        if self._x is None:
            # NOTE: In starry, the z axis points *toward* the observer,
            # which is the opposite of the convention used for Doppler
            # shifts, so we need to include a factor of -1 below.
            betasini = self._beta * np.sin(self._inc)
            Kp = self.lnlam_padded.shape[0]
            hw = (self.W - 1) // 2
            lam_kernel = self.lnlam_padded[Kp // 2 - hw:Kp // 2 + hw + 1]
            self._x = -(1 / betasini) * (np.exp(2 * lam_kernel) - 1) / \
                        (np.exp(2 * lam_kernel) + 1)
            self._x[self.x < -1.0] = -1.0
            self._x[self.x > 1.0] = 1.0

        return self._x

    @property
    def theta(self):
        """
        The array of rotational phases in degrees.
            
        """
        if self._theta is None:
            raise ValueError("Please load a dataset first.")
        return self._theta * 180.0 / np.pi

    def _reset_cache(self):
        self._lnlam_padded = None
        self._x = None
        self.W = None
        self._g = None
        self._T = None
        self._D = None
        self.u = None
        self.vT = None
        self.b = None

    def _reset_data(self):
        self._theta = None
        self._lnlam = None
        self.F = None
        self.ferr = None
        self.u_true = None
        self.vT_true = None
        self.b_true = None
        self.u = None
        self.vT = None
        self.b = None

    def _s(self):
        # Compute the `s^T` solution vector.
        sijk = np.zeros((self.ydeg + 1, self.ydeg + 1, 2, len(self.x)))
        
        # Initial conditions
        r2 = (1 - self.x ** 2)
        sijk[0, 0, 0] = 2 * r2 ** 0.5
        sijk[0, 0, 1] = 0.5 * np.pi * r2

        # Upward recursion in j
        for j in range(2, self.ydeg + 1, 2):
            sijk[0, j, 0] = ((j - 1.) / (j + 1.)) * r2 * sijk[0, j - 2, 0]
            sijk[0, j, 1] = ((j - 1.) / (j + 2.)) * r2 * sijk[0, j - 2, 1]
        
        # Upward recursion in i
        for i in range(1, self.ydeg + 1):
            sijk[i] = sijk[i - 1] * self.x

        # Full vector
        s = np.empty((self.N, len(self.x)))
        n = np.arange(self.N)
        LAM = np.floor(np.sqrt(n))
        DEL = 0.5 * (n - LAM ** 2)
        i = np.array(np.floor(LAM - DEL), dtype=int)
        j = np.array(np.floor(DEL), dtype=int)
        k = np.array(np.ceil(DEL) - np.floor(DEL), dtype=int)
        s[n] = sijk[i, j, k]
        return s

    @property
    def g(self):
        """
        Return the vectorized convolution kernel `g^T`.

        This is a matrix whose rows are equal to the convolution kernels
        for each term in the  spherical harmonic decomposition of the 
        surface.
        """
        if self._g is None:
            g = self._A1T.dot(self._s())
            norm = np.trapz(g[0])
            self._g = g / norm
        
        return self._g

    @property
    def T(self):
        """
        Return the Toeplitz super-matrix.

        This is a horizontal stack of Toeplitz convolution matrices, one per
        spherical harmonic.
        """
        if self._T is None:
            self._T = [None for n in range(self.N)]
            for n in range(self.N):
                diagonals = np.tile(self.g[n].reshape(-1, 1), self.K)
                offsets = np.arange(self.W)
                self._T[n] = diags(diagonals, offsets, 
                                  (self.K, self.K + self.W - 1), format="csr")
        return self._T

    @property
    def D(self, theta=0.0, quiet=False):
        """
        Return the full Doppler design matrix evaluated for angles
        `theta` (in degrees).

        """
        if self._D is None:
            
            # Rotation matrices
            sini = np.sin(self._inc)
            cosi = np.cos(self._inc)
            axis = [0, sini, cosi]
            R = [self._R(axis, t) for t in self.theta]

            # The design matrix
            Dt = [None for t in range(self.M)]
            for t in tqdm(range(self.M), disable=quiet):
                TR = [None for n in range(self.N)]
                for l in range(self.ydeg + 1):
                    idx = slice(l ** 2, (l + 1) ** 2)
                    TR[idx] = np.tensordot(R[t][l].T, self.T[idx], axes=1)
                Dt[t] = hstack(TR)
            self._D = vstack(Dt).tocsr()
        
        return self._D

    def load_data(self, theta, lnlam, F, ferr=1.e-4):
        """Load a dataset.
        
        Args:
            theta (ndarray): Array of rotational phases in degrees of length `M`.
            lnlam (ndarray): Uniformly spaced log-wavelength array of length `K`.
            F (ndarray): Matrix of observed spectra of shape `(M, K)`.
            ferr (float, optional): Uncertainty on the flux. Defaults to 1.e-4.
        """
        self._reset_data()
        self._set_theta(theta)
        self._set_lnlam(lnlam)
        self.F = F
        self.ferr = ferr
        self.F_CInv = np.ones_like(self.F) / self.ferr ** 2
    
    def generate_data(self, R=3e5, nlam=200, sigma=7.5e-6, nlines=20, 
            ntheta=11, ferr=1.e-4, image=None):
        """Generate a synthetic dataset.
        
        Args:
            R (float, optional): The spectral resolution. Defaults to 3e5.
            nlam (int, optional): Number of observed wavelength bins. 
                Defaults to 200.
            sigma (float, optional): Line width in log space. Defaults to 
                7.5e-6, equivalent to ~0.05A at 6430A.
            nlines (int, optional): Number of additional small lines to include. 
                Defaults to 20.
            ntheta (int, optional): Number of spectra. Defaults to 11.
            ferr (float, optional): Gaussian error to add to the fluxes. 
                Defaults to 1.e-3
            image (string, optional): Path to the image to expand in Ylms.
                Defaults to "vogtstar.jpg"
        """
        self._reset_data()

        # The theta array in degrees
        theta = np.linspace(-180, 180, ntheta + 1)[:-1]
        self._set_theta(theta)

        # The log-wavelength array
        dlam = np.log(1.0 + 1.0 / R)
        lnlam = np.arange(-(nlam // 2), nlam // 2 + 1) * dlam
        self._set_lnlam(lnlam)

        # Now let's generate a synthetic spectrum. We do this on the
        # *padded* wavelength grid to avoid convolution edge effects.
        # A deep line at the center of the wavelength range
        vT = 1 - 0.5 * np.exp(-0.5 * self.lnlam_padded ** 2 / sigma ** 2)

        # Scatter some smaller lines around for good measure
        for _ in range(nlines):
            amp = 0.1 * np.random.random()
            mu = 2.1 * (0.5 - np.random.random()) * self.lnlam_padded.max()
            vT -= amp * np.exp(-0.5 * 
                    (self.lnlam_padded - mu) ** 2 / sigma ** 2)

        # Now generate our map
        if image is None:
            image = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "vogtstar.jpg")
        self.map.load(image)
        u = np.array(self.map.y.eval())

        # Compute the map matrix & the flux matrix
        A = u.reshape(-1, 1).dot(vT.reshape(1, -1))
        F = self.D.dot(A.reshape(-1)).reshape(ntheta, -1)

        # Let's divide out the baseline flux. This is a bummer,
        # since it contains really important information about
        # the map, but unfortunately we can't typically
        # measure it with a spectrograph.
        b = self.map.flux(theta=theta).eval()
        F /= b.reshape(-1, 1)

        # Finally, we add some noise
        F += ferr * np.random.randn(*F.shape)

        # Store the dataset
        self.u_true = u[1:]
        self.vT_true = vT
        self.b_true = b
        self.F = F
        self.ferr = ferr
        self._F_CInv = np.ones_like(self.F) / self.ferr ** 2
    
    def compute_u(self, T=1.0, mu=0.0, sig=0.01):
        """
        Linear solve for `u` given `v^T`, `b`.

        """
        V = sparse_block_diag([self.vT.reshape(-1, 1) for n in range(self.N)])
        A_ = np.array(self.D.dot(V).todense())
        A0, A = A_[:, 0], A_[:, 1:]
        ATCInv = np.multiply(A.T, 
            (self._F_CInv / T / self.b.reshape(-1, 1) ** 2).reshape(-1))
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, 
            (self.F * self.b.reshape(-1, 1)).reshape(-1) - A0)
        cinv = np.ones(self.N - 1) / sig ** 2
        mu = np.ones(self.N - 1) * mu ** 2
        np.fill_diagonal(ATCInvA, ATCInvA.diagonal() + cinv)
        self.u = np.linalg.solve(ATCInvA, ATCInvf + cinv * mu)
    
    def compute_b(self, T=1.0, mu=1.0, sig=0.1):
        """
        Linear solve for `b` given `u` and `vT`.
        Note that the problem is linear in `1 / b`, so that's the
        space in which the Gaussian priors are applied.

        """
        A = (np.append([1], self.u)).reshape(-1, 1).dot(self.vT.reshape(1, -1))
        M = self.D.dot(A.reshape(-1)).reshape(self.M, -1)
        MT = dense_block_diag(*M)
        ATCInv = np.multiply(MT, self._F_CInv.reshape(-1) / T)
        ATCInvA = ATCInv.dot(MT.T)
        ATCInvf = np.dot(ATCInv, self.F.reshape(-1))
        cinv = np.ones(self.M) / sig ** 2
        np.fill_diagonal(ATCInvA, ATCInvA.diagonal() + 1.0 / cinv)
        mu = np.ones(self.M) * mu
        invb = np.linalg.solve(ATCInvA, ATCInvf + 1.0 / (cinv * mu))
        self.b = 1.0 / invb
    
    def compute_vT(self, cho_C, T=1.0, mu=1.0):
        """
        Linear solve for `v^T` given `u`, `b`.

        """
        Kp = self.lnlam_padded.shape[0]
        offsets = -np.arange(0, self.N) * Kp
        U = diags([np.ones(Kp)] + 
                  [np.ones(Kp) * self.u[n] for n in range(self.N - 1)], 
                  offsets, shape=(self.N * Kp, Kp))
        A = np.array(self.D.dot(U).todense())
        ATCInv = np.multiply(A.T, 
            (self._F_CInv / T / self.b.reshape(-1, 1) ** 2).reshape(-1))
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, (self.F * self.b.reshape(-1, 1)).reshape(-1))
        CInv = cho_solve(cho_C, np.eye(Kp))
        CInvmu = cho_solve(cho_C, np.ones(Kp) * mu)
        self.vT = np.linalg.solve(ATCInvA + CInv, ATCInvf + CInvmu)
    
    def solve(self, u=None, vT=None, b=None, u_guess=None, 
              vT_guess=None, b_guess=None, u_mu=0.0, u_sig=0.01, 
              vT_mu=1.0, vT_sig=0.3, vT_rho=3.e-5, b_mu=1.0, 
              b_sig=0.1, niter_adam=100, niter_linear=0, 
              T=1.0, **kwargs):
        """
        
        """
        # Gaussian process prior on `vT`
        if vT_rho > 0.0:
            kernel = celerite.terms.Matern32Term(np.log(vT_sig), np.log(vT_rho))
            gp = celerite.GP(kernel)
            vT_C = gp.get_matrix(self.lnlam_padded)
        else:
            Kp = self.lnlam_padded.shape[0]
            vT_C = np.eye(Kp) * vT_sig ** 2
        vT_cho_C = cho_factor(vT_C)

        # Simple linear solves
        if (u is not None) and (vT is not None):
            self.u = u
            self.vT = vT
            self.compute_b(T=T, mu=b_mu, sig=b_sig)
        elif (u is not None) and (b is not None):
            self.u = u
            self.b = b
            self.compute_vT(vT_cho_C, T=T, mu=vT_mu)
        elif (vT is not None) and (b is not None):
            self.b = b
            self.vT = vT
            self.compute_u(T=T, mu=u_mu, sig=u_sig)