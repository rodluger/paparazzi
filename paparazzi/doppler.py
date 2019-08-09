# -*- coding: utf-8 -*-
import numpy as np
import starry
from scipy.sparse import csr_matrix, hstack, vstack, diags
from scipy.sparse import block_diag as sparse_block_diag
from scipy.linalg import block_diag as dense_block_diag
from scipy.linalg import cho_factor, cho_solve
import celerite
import theano
import theano.tensor as tt
import theano.sparse as ts
from tqdm import tqdm
import os
from .utils import Adam, NAdam

CLIGHT = 3.0e5


__all__ = ["Doppler"]


class Doppler(object):
    """
    Doppler imaging solver class.

    Args:
        ydeg (int, optional): Spherical harmonic degree. Defaults to 15.
        vsini (int, optional): Projected equatorial velocity in km/s. 
            Defaults to 40.
        inc (float, optional): Inclination in degrees. Defaults to 40.0.
        u_mu (float or ndarray, optional): Prior mean on the spherical 
            harmonic coefficients. Defaults to 0.0.
        u_sig (float or ndarray, optional): Prior standard deviation on 
            the spherical harmonic coefficients. Defaults to 0.01.
        vT_mu (float or ndarray, optional): Prior mean on the spectrum. 
            Defaults to 1.0.
        vT_sig (float, optional): Prior standard deviation on
            the spectrum. Defaults to 0.3.
        vT_rho (float, optional): Prior lengthscale on the Gaussian Process
            prior on the spectrum. Set to zero to disable the GP.
            Defaults to 3.e-5.
        baseline_mu (float or ndarray, optional): Prior mean on the 
            baseline. Defaults to 1.0.
        baseline_sig (float or ndarray, optional): Prior standard 
            deviation on the baseline. Defaults to 0.1.
    """

    def __init__(
        self,
        ydeg=15,
        vsini=40,
        inc=40.0,
        u_mu=0.0,
        u_sig=0.01,
        vT_mu=1.0,
        vT_sig=0.3,
        vT_rho=3.0e-5,
        baseline_mu=1.0,
        baseline_sig=0.1,
    ):
        # Stellar params
        self.ydeg = ydeg
        self.inc = inc
        self.vsini = vsini

        # Reset!
        self._reset_cache()
        self._theta = None
        self._lnlam = None
        self.F = None
        self.ferr = None
        self.u_true = None
        self.vT_true = None
        self.baseline_true = None

        # Inference params
        self.u_mu = u_mu
        self.u_sig = u_sig
        self.vT_mu = vT_mu
        self._vT_sig = vT_sig
        self._vT_rho = vT_rho
        self.baseline_mu = baseline_mu
        self.baseline_sig = baseline_sig
        self._vT_CInv = None
        self._F_CInv = None
        self.vT_deconv = None

        # Initialize
        self.u = self.u_mu * np.ones(self.N - 1)
        self.vT = None

    def _reset_cache(self):
        # Reset the cache
        self._D = None
        self._kT = None

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

        # Grab the `A1` matrix from `starry`
        self._map = starry.Map(value)
        self._A1T = self._map.ops.A1.eval().T

        # Grab the rotation matrix op
        self._R = self._map.ops.R

        # Reset
        self._reset_cache()

    @property
    def vsini(self):
        """
        The projected equatorial radial velocity in km/s.
            
        """
        return self._vsini

    @vsini.setter
    def vsini(self, value):
        self._vsini = value
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
        self._map.inc = value
        self._reset_cache()

    @property
    def lnlam(self):
        """
        The log-wavelength grid.
            
        """
        if self._lnlam is None:
            raise ValueError("Please load a dataset first.")
        return self._lnlam

    def _set_lnlam(self, lnlam):
        # Set the log-wavelength grid
        self._lnlam = np.atleast_1d(lnlam)
        assert self.K >= 3, "The wavelength grid must have at least 3 bins."
        assert self.K % 2 == 1, "The number of wavelength bins must be odd."
        assert len(self._lnlam.shape) == 1, "Invalid wavelength grid shape."
        assert np.allclose(
            np.diff(self._lnlam), self._lnlam[1] - self._lnlam[0]
        )
        self._reset_cache()
        self._compute_gp()

    @property
    def lnlam_padded(self):
        """
        The padded log-wavelength grid.
            
        """
        dlam = self.lnlam[1] - self.lnlam[0]
        hw = (self.W - 1) // 2
        pad_l = np.linspace(self.lnlam[0] - hw * dlam, self.lnlam[0], hw + 1)[
            :-1
        ]
        pad_r = np.linspace(
            self.lnlam[-1], self.lnlam[-1] + hw * dlam, hw + 1
        )[1:]
        return np.concatenate((pad_l, self.lnlam, pad_r))

    @property
    def K(self):
        """
        The number of (observed) wavelength bins.
        
        """
        return self.lnlam.shape[0]

    @property
    def Kp(self):
        """
        The number of (internal, padded) wavelength bins.
        
        """
        return self.K + self.W - 1

    @property
    def W(self):
        """
        The width of the convolution kernel in bins.

        """
        betasini = self.vsini / CLIGHT
        hw = np.abs(0.5 * np.log((1 + betasini) / (1 - betasini)))
        dlam = self.lnlam[1] - self.lnlam[0]
        return 2 * int(np.floor(hw / dlam)) + 1

    @property
    def M(self):
        """
        The number of epochs in the dataset.

        """
        return self.theta.shape[0]

    @property
    def N(self):
        """
        The number of spherical harmonic coefficients.

        """
        return (self.ydeg + 1) ** 2

    @property
    def theta(self):
        """
        The array of rotational phases in degrees.
            
        """
        if self._theta is None:
            raise ValueError("Please load a dataset first.")
        return self._theta * 180.0 / np.pi

    def _set_theta(self, theta):
        # Set the angular phase (radians internally)
        self._theta = np.atleast_1d(theta) * np.pi / 180.0
        self._reset_cache()

    @property
    def vT_sig(self):
        return self._vT_sig

    @vT_sig.setter
    def vT_sig(self, value):
        self._vT_sig = value
        self._compute_gp()

    @property
    def vT_rho(self):
        return self._vT_rho

    @vT_rho.setter
    def vT_rho(self, value):
        self._vT_rho = value
        self._compute_gp()

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, value):
        self._u = value
        if value is None:
            self._map[1:, :] = 0.0
        else:
            self._map[1:, :] = self._u

    def _compute_gp(self):
        # Compute the GP prior on the spectrum
        if self._lnlam is None:
            pass
        if self.vT_rho > 0.0:
            kernel = celerite.terms.Matern32Term(
                np.log(self.vT_sig), np.log(self.vT_rho)
            )
            gp = celerite.GP(kernel)
            vT_C = gp.get_matrix(self.lnlam_padded)
        else:
            vT_C = np.eye(self.Kp) * self.vT_sig ** 2
        self._vT_cho_C = cho_factor(vT_C)
        self._vT_CInv = cho_solve(self._vT_cho_C, np.eye(self.Kp))

    def x(self):
        """
        Return the `x` coordinate of lines of constant Doppler shift.
        
        """
        betasini = self.vsini / CLIGHT
        hw = (self.W - 1) // 2
        Kp = self.Kp
        lam_kernel = self.lnlam_padded[Kp // 2 - hw : Kp // 2 + hw + 1]
        x = (
            (1 / betasini)
            * (np.exp(-2 * lam_kernel) - 1)
            / (np.exp(-2 * lam_kernel) + 1)
        )
        if np.any(np.abs(x) >= 1.0):
            raise ValueError(
                "Error computing the kernel width. This is likely a bug!"
            )
        return x

    def sT(self):
        """
        Return the `s^T` solution vector.
        
        """
        x = self.x()
        sijk = np.zeros((self.ydeg + 1, self.ydeg + 1, 2, len(x)))

        # Initial conditions
        r2 = 1 - x ** 2
        sijk[0, 0, 0] = 2 * r2 ** 0.5
        sijk[0, 0, 1] = 0.5 * np.pi * r2

        # Upward recursion in j
        for j in range(2, self.ydeg + 1, 2):
            sijk[0, j, 0] = ((j - 1.0) / (j + 1.0)) * r2 * sijk[0, j - 2, 0]
            sijk[0, j, 1] = ((j - 1.0) / (j + 2.0)) * r2 * sijk[0, j - 2, 1]

        # Upward recursion in i
        for i in range(1, self.ydeg + 1):
            sijk[i] = sijk[i - 1] * x

        # Full vector
        s = np.empty((self.N, len(x)))
        n = np.arange(self.N)
        LAM = np.floor(np.sqrt(n))
        DEL = 0.5 * (n - LAM ** 2)
        i = np.array(np.floor(LAM - DEL), dtype=int)
        j = np.array(np.floor(DEL), dtype=int)
        k = np.array(np.ceil(DEL) - np.floor(DEL), dtype=int)
        s[n] = sijk[i, j, k]
        return s

    def kT(self):
        """
        Return the vectorized convolution kernels `k^T`.

        This is a matrix whose rows are equal to the convolution kernels
        for each term in the  spherical harmonic decomposition of the 
        surface.
        """
        # Allow caching of this matrix.
        if self._kT is None:
            self._kT = self._A1T.dot(self.sT())
            norm = np.trapz(self._kT[0])
            self._kT /= norm
        return self._kT

    def D(self, quiet=False):
        """
        Return the full Doppler matrix.

        This is a horizontal stack of Toeplitz convolution matrices, one per
        spherical harmonic. These matrices are then stacked vertically for
        each rotational phase.
        
        """
        # Allow caching of this matrix.
        if self._D is None:

            if not quiet:
                print("Computing Doppler matrix...", end=" ", flush=True)

            W = self.W
            Kp = self.Kp
            K = self.K
            kT0 = self.kT()

            # Rotation axis
            sini = np.sin(self._inc)
            cosi = np.cos(self._inc)
            axis = [0, sini, cosi]

            # Pre-compute a skeleton sparse Doppler matrix row in CSR format.
            # We will directly edit its `data` attribute, whose structure is
            # *super* convenient, as we'll see below.
            indptr = (self.N * W) * np.arange(K + 1, dtype="int32")
            i0 = np.arange(W)
            i1 = Kp * np.arange(self.N)
            i2 = np.arange(K)
            indices = (
                (i0.reshape(-1, 1) + i1.reshape(1, -1)).T.reshape(-1, 1)
                + i2.reshape(1, -1)
            ).T.reshape(-1)
            data = np.ones(W * K * self.N)
            D = [
                csr_matrix((data, indices, indptr), shape=(K, self.N * Kp))
                for m in range(self.M)
            ]

            # Loop through each epoch
            for m in range(self.M):

                # Rotate the kernels
                # Note that we are computing R^T(theta) = R(-theta) here
                RT = self._R(axis, -self.theta[m] * np.pi / 180.0)
                kT = np.empty_like(kT0)
                for l in range(self.ydeg + 1):
                    idx = slice(l ** 2, (l + 1) ** 2)
                    kT[idx] = RT[l].dot(kT0[idx])

                # Populate the Doppler matrix
                D[m].data = np.tile(kT.reshape(-1), K)

            # Stack the rows and we are done!
            # TODO: This can probably be sped up.
            self._D = vstack(D).tocsr()

            if not quiet:
                print("Done.")

        return self._D

    def load_data(self, theta, lnlam, F, ferr=1.0e-4):
        """Load a dataset.
        
        Args:
            theta (ndarray): Array of rotational phases in degrees 
                of length `M`.
            lnlam (ndarray): Uniformly spaced log-wavelength array 
                of length `K`.
            F (ndarray): Matrix of observed spectra of shape `(M, K)`.
            ferr (float, optional): Uncertainty on the flux. 
                Defaults to 1.e-4.
        """
        self._set_theta(theta)
        self._set_lnlam(lnlam)
        self.F = F
        self.ferr = ferr
        self._F_CInv = np.ones_like(self.F) / self.ferr ** 2
        self.u_true = None
        self.vT_true = None
        self.baseline_true = None

    def generate_data(
        self,
        R=3e5,
        nlam=200,
        sigma=7.5e-6,
        nlines=21,
        ntheta=16,
        ferr=1.0e-4,
        u=None,
        image=None,
        theta=None,
    ):
        """Generate a synthetic dataset.
        
        Args:
            R (float, optional): The spectral resolution. Defaults to 3e5.
            nlam (int, optional): Number of observed wavelength bins. 
                Defaults to 200.
            sigma (float, optional): Line width in log space. Defaults to 
                7.5e-6, equivalent to ~0.05A at 6430A.
            nlines (int, optional): Total number of lines to include. 
                Defaults to 21.
            ntheta (int, optional): Number of spectra. Defaults to 11.
            ferr (float, optional): Gaussian error to add to the fluxes. 
                Defaults to 1.e-3
            u (ndarray, optional): The spherical harmonic vector for the map.
                Defaults to ``None``, in which case the ``image`` is loaded.
            image (string, optional): Path to the image to expand in Ylms.
                Defaults to "vogtstar.jpg"
            theta (ndarray, optional): The rotational phase in degrees
                at each epoch. Defaults to ``None``, in which case this is
                set to vary uniformly between -180 and 180 with ``ntheta``
                epochs.
        """
        # The theta array in degrees
        if theta is None:
            if ntheta <= 1:
                theta = [0.0]
            else:
                theta = np.linspace(-180, 180, ntheta + 1)[:-1]
        else:
            ntheta = len(theta)
        self._set_theta(theta)

        # The log-wavelength array
        dlam = np.log(1.0 + 1.0 / R)
        lnlam = np.arange(-(nlam // 2), nlam // 2 + 1) * dlam
        self._set_lnlam(lnlam)
        lnlam_padded = self.lnlam_padded

        # Now let's generate a synthetic spectrum. We do this on the
        # *padded* wavelength grid to avoid convolution edge effects.
        # A deep line at the center of the wavelength range
        vT = 1 - 0.5 * np.exp(-0.5 * lnlam_padded ** 2 / sigma ** 2)

        # Scatter some smaller lines around for good measure
        for _ in range(nlines - 1):
            amp = 0.1 * np.random.random()
            mu = 2.1 * (0.5 - np.random.random()) * lnlam_padded.max()
            vT -= amp * np.exp(-0.5 * (lnlam_padded - mu) ** 2 / sigma ** 2)

        # Now generate our map
        if u is None:
            if image is None:
                image = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "vogtstar.jpg"
                )
            self._map.load(image)
        else:
            if len(u) == self.N - 1:
                self._map[1:, :] = u
            elif len(u) == self.N:
                self._map[1:, :] = u[1:]
            else:
                raise ValueError("The vector `u` has the wrong size.")
        u = np.array(self._map.y.eval())

        # Compute the model
        self.u = u[1:]
        self.vT = vT
        F = self.model()

        # Add some noise
        F += ferr * np.random.randn(*F.shape)

        # Store the dataset
        self.u_true = self.u
        self.vT_true = self.vT
        self.baseline_true = self.baseline().reshape(-1)
        self.F = F
        self.ferr = ferr
        self._F_CInv = np.ones_like(self.F) / self.ferr ** 2

        # Reset the coeffs
        self.u = self.u_mu * np.ones(self.N - 1)
        self.vT = None

    def show(self, **kwargs):
        """
        Show the image of the current map solution.

        """
        self._map.show(**kwargs)

    def render(self, **kwargs):
        """
        Render the image of the current map solution.

        """
        func = theano.function([], self._map.render(**kwargs))
        return func()

    def baseline(self):
        """
        Return the photometric baseline at each epoch.

        """
        return self._map.flux(theta=self.theta).eval().reshape(-1, 1)

    def model(self):
        """
        Return the full model for the flux matrix `F`.

        """
        A = np.append([1], self.u).reshape(-1, 1).dot(self.vT.reshape(1, -1))
        # If `D` is available, use it; otherwise, do a convolution.
        if self._D is not None:

            # Just the dot product with the design matrix
            model = self.D().dot(A.reshape(-1)).reshape(self.M, -1)

        else:

            # Pre-compute some stuff
            kT0 = self.kT()
            sini = np.sin(self._inc)
            cosi = np.cos(self._inc)
            axis = [0, sini, cosi]

            # Loop through each epoch
            model = np.empty((self.M, self.K))
            for m in range(self.M):

                # Rotate the kernels (note that R^T(theta) = R(-theta))
                RT = self._R(axis, -self.theta[m] * np.pi / 180.0)
                kT = np.empty_like(kT0)
                for l in range(self.ydeg + 1):
                    idx = slice(l ** 2, (l + 1) ** 2)
                    kT[idx] = RT[l].dot(kT0[idx])

                # Compute the model for this epoch
                model[m] = np.sum(
                    [
                        np.convolve(kT[n][::-1], A[n], mode="valid")
                        for n in range(self.N)
                    ],
                    axis=0,
                )
        model /= self.baseline()
        return model

    def loss(self):
        """
        Return the loss function for the current parameters.

        """
        # Likelihood and prior
        lnlike = -0.5 * np.sum(
            (self.F - self.model()).reshape(-1) ** 2 * self._F_CInv.reshape(-1)
        )
        lnprior = (
            -0.5 * np.sum((self.u - self.u_mu) ** 2 / self.u_sig ** 2)
            + -0.5
            * np.sum(
                (self.baseline() - self.baseline_mu) ** 2
                / self.baseline_sig ** 2
            )
            + -0.5
            * np.dot(
                np.dot((self.vT - self.vT_mu).reshape(1, -1), self._vT_CInv),
                (self.vT - self.vT_mu).reshape(-1, 1),
            )
        )
        return -(lnlike + lnprior).item()

    def compute_u(self, T=1.0, baseline=None):
        """
        Linear solve for ``u`` given ``v^T`` and an optional baseline 
        and temperature. If the baseline is not given, solves the
        approximate linear problem, which assumes the difference 
        between the baseline and ``1.0`` is small.

        Returns the Cholesky decomposition of the covariance of ``u``.

        """
        # Get the design matrix
        V = sparse_block_diag([self.vT.reshape(-1, 1) for n in range(self.N)])
        A_ = np.array(self.D().dot(V).todense())
        A0, A = A_[:, 0], A_[:, 1:]

        if baseline is not None:

            # The problem is exactly linear!
            if not hasattr(baseline, "__len__"):
                baseline = np.ones(self.M) * baseline
            baseline = baseline.reshape(-1, 1)
            ATCInv = np.multiply(
                A.T, (self._F_CInv / T / baseline ** 2).reshape(-1)
            )
            ATCInvA = ATCInv.dot(A)
            ATCInvf = np.dot(ATCInv, (self.F * baseline).reshape(-1) - A0)
            cinv = np.ones(self.N - 1) / self.u_sig ** 2
            mu = np.ones(self.N - 1) * self.u_mu
            np.fill_diagonal(ATCInvA, ATCInvA.diagonal() + cinv)
            cho_C = cho_factor(ATCInvA)
            self.u = cho_solve(cho_C, ATCInvf + cinv * mu)

        else:

            # We need to Taylor expand the problem about
            # a baseline of unity.
            B = self._map.X(theta=self.theta).eval()[:, 1:]
            B = np.repeat(B, 201, axis=0)

            # We are Taylor expanding
            #
            #   f = (A0 + A . u) / (1 + B . u)
            #
            #   as
            #
            #   f = A0 + C . u + O(u^2)
            #
            # where C is
            C = A - A0.reshape(-1, 1) * B
            CTCInv = np.multiply(C.T, (self._F_CInv / T).reshape(-1))
            CTCInvC = CTCInv.dot(C)
            cinv = np.ones(self.N - 1) / self.u_sig ** 2
            np.fill_diagonal(CTCInvC, CTCInvC.diagonal() + cinv)
            cho_C = cho_factor(CTCInvC)
            CTCInvy = np.dot(CTCInv, self.F.reshape(-1) - A0)
            mu = np.ones(self.N - 1) * self.u_mu
            self.u = cho_solve(cho_C, CTCInvy + cinv * mu)

        return cho_C

    def compute_vT(self, T=1.0, baseline=None):
        """
        Linear solve for ``v^T`` given ``u`` and an optional baseline 
        and temperature.

        Returns the Cholesky decomposition of the covariance of ``vT``.

        """
        if baseline is None:
            baseline = self.baseline()
        else:
            if not hasattr(baseline, "__len__"):
                baseline = np.ones(self.M) * baseline
            baseline = baseline.reshape(-1, 1)
        Kp = self.Kp
        offsets = -np.arange(0, self.N) * Kp
        U = diags(
            [np.ones(Kp)]
            + [np.ones(Kp) * self.u[n] for n in range(self.N - 1)],
            offsets,
            shape=(self.N * Kp, Kp),
        )
        A = np.array(self.D().dot(U).todense())
        ATCInv = np.multiply(
            A.T, (self._F_CInv / T / baseline ** 2).reshape(-1)
        )
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, (self.F * baseline).reshape(-1))
        CInv = cho_solve(self._vT_cho_C, np.eye(Kp))
        CInvmu = cho_solve(self._vT_cho_C, np.ones(Kp) * self.vT_mu)
        cho_vT = cho_factor(ATCInvA + CInv)
        self.vT = cho_solve(cho_vT, ATCInvf + CInvmu)
        return cho_vT

    def solve(
        self,
        u=None,
        vT=None,
        baseline=None,
        u_guess=None,
        vT_guess=None,
        baseline_guess=None,
        niter=100,
        T=1.0,
        dlogT=-0.25,
        optimizer="NAdam",
        dcf=10.0,
        quiet=False,
        **kwargs
    ):
        """Solve the Doppler imaging problem.
        
        Returns:
            ``(loss, cho_u, cho_vT)``, a tuple containing the array of
            loss values during the optimization and the Cholesky factorization
            of the covariance matrices of ``u`` and ``vT``, if available 
            (otherwise the latter two are set to ``None``.)
        """
        # Check the optimizer is valid
        if optimizer.lower() == "nadam":
            optimizer = NAdam
        elif optimizer.lower() == "adam":
            optimizer = Adam
        else:
            raise ValueError("Invalid optimizer.")

        # Figure out what to solve for
        known = []
        if vT is not None:
            known += ["vT"]
        if u is not None:
            known += ["u"]

        if ("u" in known) and ("vT" in known):

            # Nothing to do here but ingest the values!
            self.u = u
            self.vT = vT
            return self.loss(), None, None

        elif "u" in known:

            # Easy: it's a linear problem
            self.u = u
            cho_vT = self.compute_vT()
            return self.loss(), None, cho_vT

        else:

            if ("vT" in known) and (baseline is not None):

                # Still a linear problem!
                self.vT = vT
                cho_u = self.compute_u(baseline=baseline)
                return self.loss(), cho_u, None

            else:

                # Non-linear. Let's use (N)Adam.

                if "vT" in known:

                    # We know `vT` and need to solve for
                    # `u` w/o any baseline knowledge.
                    vT_guess = vT

                else:

                    # We know *nothing*!

                    # Estimate `v^T` from the deconvolved mean spectrum
                    if vT_guess is None:

                        fmean = np.mean(self.F, axis=0)
                        fmean -= np.mean(fmean)
                        diagonals = np.tile(
                            self.kT()[0].reshape(-1, 1), self.K
                        )
                        offsets = np.arange(self.W)
                        A = diags(
                            diagonals, offsets, (self.K, self.Kp), format="csr"
                        )
                        LInv = (
                            dcf ** 2
                            * self.ferr ** 2
                            / self.vT_sig ** 2
                            * np.eye(A.shape[1])
                        )
                        vT_guess = 1.0 + np.linalg.solve(
                            A.T.dot(A).toarray() + LInv, A.T.dot(fmean)
                        )

                        # Save this for later
                        self.vT_deconv = vT_guess

                # Estimate `u` w/o baseline knowledge
                # If `baseline_guess` is `None`, this is done via
                # a Taylor expansion; see ``compute_u()``.
                if u_guess is None:

                    self.vT = vT_guess
                    self.compute_u(T=T, baseline=baseline_guess)
                    u_guess = self.u

                # Initialize the variables
                self.u = u_guess
                self.vT = vT_guess

                # Tempering params
                if T > 1.0:
                    T_arr = 10 ** np.arange(np.log10(T), 0, dlogT)
                    T_arr = np.append(T_arr, [1.0])
                    niter_bilin = len(T_arr)
                else:
                    T_arr = [1.0]
                    niter_bilin = 1

                # Loss array
                loss_val = np.zeros(niter_bilin + niter + 1)
                loss_val[0] = self.loss()

                # Iterative bi-linear solve
                if niter_bilin > 0:

                    if not quiet:
                        print("Running bi-linear solver...")

                    for n in tqdm(range(niter_bilin), disable=quiet):

                        # Compute `u` using the previous baseline
                        self.compute_u(T=T_arr[n], baseline=self.baseline())

                        # Compute `vT` using the current `u`
                        if "vT" not in known:
                            self.compute_vT(T=T_arr[n])

                        loss_val[n + 1] = self.loss()

                # Non-linear solve
                if niter > 0:

                    # Theano nonlienar solve. Variables:
                    u = theano.shared(self.u)
                    vT = theano.shared(self.vT)
                    if "vT" in known:
                        theano_vars = [u]
                    else:
                        theano_vars = [u, vT]

                    # Compute the model
                    D = ts.as_sparse_variable(self.D())
                    a = tt.reshape(
                        tt.dot(
                            tt.reshape(tt.concatenate([[1.0], u]), (-1, 1)),
                            tt.reshape(vT, (1, -1)),
                        ),
                        (-1,),
                    )
                    b = tt.dot(
                        self._map.X(theta=self.theta),
                        tt.reshape(tt.concatenate([[1.0], u]), (-1, 1)),
                    )
                    B = tt.reshape(b, (-1, 1))
                    M = tt.reshape(ts.dot(D, a), (self.M, -1)) / B

                    # Compute the loss
                    r = tt.reshape(self.F - M, (-1,))
                    cov = tt.reshape(self._F_CInv, (-1,))
                    lnlike = -0.5 * tt.sum(r ** 2 * cov)
                    lnprior = (
                        -0.5 * tt.sum((u - self.u_mu) ** 2 / self.u_sig ** 2)
                        + -0.5
                        * tt.sum(
                            (b - self.baseline_mu) ** 2
                            / self.baseline_sig ** 2
                        )
                        + -0.5
                        * tt.dot(
                            tt.dot(
                                tt.reshape((vT - self.vT_mu), (1, -1)),
                                self._vT_CInv,
                            ),
                            tt.reshape((vT - self.vT_mu), (-1, 1)),
                        )[0, 0]
                    )
                    loss = -(lnlike + lnprior)
                    best_loss = loss.eval()
                    best_u = u.eval()
                    best_vT = vT.eval()

                    if not quiet:
                        print("Running non-linear solver...")

                    upd = optimizer(loss, theano_vars, **kwargs)
                    train = theano.function([], [u, vT, loss], updates=upd)
                    for n in tqdm(
                        1 + niter_bilin + np.arange(niter), disable=quiet
                    ):
                        u_val, vT_val, loss_val[n] = train()
                        if loss_val[n] < best_loss:
                            best_loss = loss_val[n]
                            best_u = u_val
                            best_vT = vT_val

                    # We are done!
                    self.u = best_u
                    self.vT = best_vT

                # Estimate the covariance of `u` conditioned on `vT`
                # and the covariance of `vT` conditioned on `u`.
                # Note that the covariance of `u` is computed from
                # the linearization that allows us to simultaneously
                # solve for the baseline.
                u_curr = np.array(self.u)
                cho_u = self.compute_u()
                self.u = u_curr

                if "vT" not in known:
                    vT_curr = np.array(self.vT)
                    cho_vT = self.compute_vT()
                    self.vT = vT_curr
                else:
                    cho_vT = None

                return loss_val, cho_u, cho_vT
