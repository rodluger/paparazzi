# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import starry
import paparazzi as pp
import os
import subprocess
import celerite
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import block_diag, diags
from scipy.linalg import block_diag as dense_block_diag
from tqdm import tqdm
import theano.tensor as tt
import theano.sparse as ts
import theano


__all__ = ["Solver"]


def Adam(cost, params, lr=0.001, b1=0.9, b2=0.999, e=1e-8):
    """https://gist.github.com/Newmu/acb738767acb4788bac3

    """
    updates = []
    grads = tt.grad(cost, params)
    i = theano.shared(np.array(0.,dtype=theano.config.floatX))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (tt.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tt.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tt.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates


class Solver(object):
    """Solve the Doppler imaging problem.
    
    Args:
        ydeg (int, optional): Degree of the Ylm expansion. Defaults to 5.
        inc (float, optional): Stellar inclination in degrees. Defaults to 60.
        vsini (float, optional): Equatorial projected velocity in km / s. 
            Defaults to 40.0.
        P (float, optional): Rotational period in days. Defaults to 1.
    """
    def __init__(self, name="vogtstar", ydeg=5, inc=60.0, vsini=40.0, P=1.0):
        # Main properties
        self.name = "output/%s" % name
        if not os.path.exists("output"):
            os.mkdir("output")
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2
        self.inc = inc
        self.vsini = vsini
        self.P = P
         
        # Map instance (for computing `b`)
        self.map = starry.Map(self.ydeg)
        self.map.inc = inc

        # Initialize the data
        self._loaded = False
        self.u_true = None
        self.vT_true = None
        self.b_true = None
        self.d_true = None
        self.D = None
        self.t = None
        self.lam_padded = None
        self.lam = None
        self.F = None
        self.M = None
        self.K = None
        self.Kp = None
        self._solved = False
        self.u = None
        self.b = None
        self.vT = None
        self.d = None
        self.lnlike = None
        self.lnprior = None

    def load_data(self, t, lam, F, ferr=1.e-4):
        """Load a dataset.
        
        Args:
            t (ndarray): Time array in days.
            lam (ndarray): Uniformly spaced log-wavelength array.
            F (ndarray): Matrix of observed spectra of shape `(nt, nlam)`.
            ferr (float, optional): Uncertainty on the flux. Defaults to 1.e-4.
        """
        doppler = pp.Doppler(lam, ydeg=self.ydeg, 
                             vsini=self.vsini, 
                             inc=self.inc, P=self.P)
        self.D = doppler.D(t=t)
        self.lam_padded = doppler.lam_padded
        self.t = t
        self.theta = (360.0 * t) % 360.0
        self.lam = lam
        self.F = F
        self.ferr = ferr
        self.M, self.K = self.F.shape
        self.Kp = self.D.shape[1] // self.N
        self._loaded = True

    def generate_data(self, R=3e5, nlam=200, sigma=7.5e-6, nlines=20, nt=11, 
                      ferr=1.e-4, image="vogtstar.jpg"):
        """Generate a synthetic dataset.
        
        Args:
            R (float, optional): The spectral resolution. Defaults to 3e5.
            nlam (int, optional): Number of observed wavelength bins. 
                Defaults to 200.
            sigma (float, optional): Line width in log space. Defaults to 7.5e-6,
                equivalent to ~0.05A at 6430A.
            nlines (int, optional): Number of additional small lines to include. 
                Defaults to 20.
            nt (int, optional): Number of spectra. Defaults to 11.
            ferr (float, optional): Gaussian error to add to the fluxes. 
                Defaults to 1.e-3
            image (string, optional): Path to the image to expand in Ylms.
                Defaults to "vogtstar.jpg"
        """
        # The time array in units of the period
        t = np.linspace(-0.5, 0.5, nt + 1)[:-1]

        # The log-wavelength array
        dlam = np.log(1.0 + 1.0 / R)
        lam = np.arange(-(nlam // 2), nlam // 2 + 1) * dlam

        # Pre-compute the Doppler basis
        doppler = pp.Doppler(lam, ydeg=self.ydeg, 
                             vsini=self.vsini, 
                             inc=self.inc, P=self.P)
        D = doppler.D(t=t)

        # Now let's generate a synthetic spectrum. We do this on the
        # *padded* wavelength grid to avoid convolution edge effects.
        lam_padded = doppler.lam_padded

        # A deep line at the center of the wavelength range
        vT = np.ones_like(lam_padded)
        vT = 1 - 0.5 * np.exp(-0.5 * lam_padded ** 2 / sigma ** 2)

        # Scatter some smaller lines around for good measure
        for _ in range(nlines):
            amp = 0.1 * np.random.random()
            mu = 2.1 * (0.5 - np.random.random()) * lam_padded.max()
            vT -= amp * np.exp(-0.5 * (lam_padded - mu) ** 2 / sigma ** 2)

        # Now generate our "Vogtstar" map
        self.map.load(image)
        u = np.array(self.map.y.eval())

        # Compute the map matrix & the flux matrix
        A = u.reshape(-1, 1).dot(vT.reshape(1, -1))
        F = D.dot(A.reshape(-1)).reshape(nt, -1)

        # Let's divide out the baseline flux. This is a bummer,
        # since it contains really important information about
        # the map, but unfortunately we can't typically
        # measure it with a spectrograph.
        b = np.max(F, axis=1)
        F /= b.reshape(-1, 1)

        # Finally, we add some noise
        F += ferr * np.random.randn(*F.shape)

        # Store the dataset
        self.u_true = u[1:]
        self.vT_true = vT
        self.b_true = b
        self.D = D
        self.t = t
        self.theta = (360.0 * t) % 360.0
        self.lam_padded = lam_padded
        self.lam = lam
        self.F = F
        self.ferr = ferr
        self.M, self.K = self.F.shape
        self.Kp = self.D.shape[1] // self.N
        self._loaded = True

    def _compute_u(self):
        """
        Linear solve for `u` given `v^T`, `b`.

        """
        V = block_diag([self.vT.reshape(-1, 1) for n in range(self.N)])
        AFull = np.array(self.D.dot(V).todense())
        A0 = AFull[:, 0]
        A = AFull[:, 1:]
        ATCInv = np.multiply(A.T, 
            (self.F_CInv / self.b.reshape(-1, 1) ** 2).reshape(-1))
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, (self.F * self.b.reshape(-1, 1)).reshape(-1) - A0)
        np.fill_diagonal(ATCInvA, ATCInvA.diagonal() + self.u_cinv)
        self.u = np.linalg.solve(ATCInvA, 
                                 ATCInvf + self.u_cinv * self.u_mu)

    def _compute_vT(self):
        """
        Linear solve for `v^T` given `u`, `b`.

        """
        offsets = -np.arange(0, self.N) * self.Kp
        U = diags([np.ones(self.Kp)] + 
                  [np.ones(self.Kp) * self.u[n] for n in range(self.N - 1)], 
                  offsets, shape=(self.N * self.Kp, self.Kp))
        A = np.array(self.D.dot(U).todense())
        ATCInv = np.multiply(A.T, 
            (self.F_CInv / self.b.reshape(-1, 1) ** 2).reshape(-1))
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, (self.F * self.b.reshape(-1, 1)).reshape(-1))
        self.vT = np.linalg.solve(ATCInvA + self.vT_CInv, 
                                  ATCInvf + self.vT_CInvmu)

    def _compute_b(self):
        """
        Linear solve for `b` given `u` and `vT`.
        Note that the problem is linear in `1 / b`, so that's the
        space in which the Gaussian priors are applied.

        """
        A = (np.append([1], self.u)).reshape(-1, 1).dot(self.vT.reshape(1, -1))
        M = self.D.dot(A.reshape(-1)).reshape(self.M, -1)
        MT = dense_block_diag(*M)
        ATCInv = np.multiply(MT, self.F_CInv.reshape(-1))
        ATCInvA = ATCInv.dot(MT.T)
        ATCInvf = np.dot(ATCInv, self.F.reshape(-1))
        np.fill_diagonal(ATCInvA, ATCInvA.diagonal() + 1.0 / self.b_cinv)
        invb = np.linalg.solve(ATCInvA, 
                               ATCInvf + 1.0 / (self.b_cinv * self.b_mu))
        self.b = 1.0 / invb

    def solve(self, u=None, vT=None, b=None, u_guess=None, 
              vT_guess=None, b_guess=None, u_mu=0.0, u_sig=0.01, 
              vT_mu=1.0, vT_sig=0.3, vT_rho=3.e-5, b_mu=1.0, 
              b_sig=0.1, niter=100, **kwargs):
        """
        
        """
        if not self._loaded:
            raise RuntimeError("Please load or generate a dataset first.")

        # Data covariance
        self.F_CInv = np.ones_like(self.F) / self.ferr ** 2
        self.F_lndet = np.sum(np.log(2 * np.pi * self.F_CInv.reshape(-1)))

        # Prior on `u`
        self.u_cinv = np.ones(self.N - 1) / u_sig ** 2
        self.u_mu = np.ones(self.N - 1) * u_mu ** 2
        self.u_lndet = np.sum(np.log(2 * np.pi * self.u_cinv))

        # Gaussian process prior on `vT`
        self.vT_mu = (np.ones(self.Kp) * vT_mu).reshape(1, -1)
        if vT_rho > 0.0:
            kernel = celerite.terms.Matern32Term(np.log(vT_sig), np.log(vT_rho))
            gp = celerite.GP(kernel)
            vT_C = gp.get_matrix(self.lam_padded)
            cho_C = cho_factor(vT_C)
            self.vT_CInv = cho_solve(cho_C, np.eye(self.Kp))
            self.vT_CInvmu = cho_solve(cho_C, self.vT_mu.reshape(-1))
            self.vT_lndet = -2 * np.sum(np.log(2 * np.pi * np.diag(cho_C[0])))
        else:
            self.vT_CInv = np.ones(self.Kp) / vT_sig ** 2
            self.vT_CInvmu = (self.vT_CInv * self.vT_mu)
            self.vT_lndet = np.sum(np.log(2 * np.pi * self.vT_CInv))
            self.vT_CInv = np.diag(self.vT_CInv)

        # Prior on `b`
        self.b_cinv = np.ones(self.M) / b_sig ** 2
        self.b_mu = np.ones(self.M) * b_mu
        self.b_lndet = self.M * np.log(2 * np.pi / b_sig ** 2)

        # Simple linear solves
        if (u is not None) and (vT is not None):
            self.u = u
            self.vT = vT
            self._compute_b()
        elif (u is not None) and (b is not None):
            self.u = u
            self.b = b
            self._compute_vT()
        elif (vT is not None) and (b is not None):
            self.b = b
            self.vT = vT
            self._compute_u()
        
        # Non-linear
        else:

            # Get our guesses going
            if u is not None:
                self.u = u
                var_names = ["vT", "b"]
                if vT_guess is None and b_guess is None:
                    self.b = self.b_mu + b_sig * np.random.randn(self.M)
                    self._compute_vT()
                elif vT_guess is not None:
                    self.vT = vT_guess
                    self._compute_b()
                elif b_guess is not None:
                    self.b = b_guess
                    self._compute_vT()
                else: raise ValueError("Unexpected branch!")
            elif vT is not None:
                self.vT = vT
                var_names = ["u", "b"]
                if u_guess is None and b_guess is None:
                    self.b = self.b_mu + b_sig * np.random.randn(self.M)
                    self._compute_u()
                elif u_guess is not None:
                    self.u = u_guess
                    self._compute_b()
                elif b_guess is not None:
                    self.b = b_guess
                    self._compute_u()
                else: raise ValueError("Unexpected branch!")
            elif b is not None:
                self.b = b
                var_names = ["u", "vT"]
                if u_guess is None and vT_guess is None:
                    self.u = self.u_mu + u_sig * np.random.randn(self.N - 1)
                    self._compute_vT()
                elif u_guess is not None:
                    self.u = u_guess
                    self._compute_vT()
                elif vT_guess is not None:
                    self.vT = vT_guess
                    self._compute_u()
                else: raise ValueError("")
            else:
                var_names = ["u", "vT", "b"]
                if vT_guess is None and b_guess is None and u_guess is None:
                    self.b = self.b_mu + b_sig * np.random.randn(self.M)
                    self.u = self.u_mu + u_sig * np.random.randn(self.N - 1)
                    self._compute_vT()
                elif u_guess is not None:
                    self.u = u_guess
                    if vT_guess is None and b_guess is None:
                        self.b = self.b_mu + b_sig * np.random.randn(self.M)
                        self._compute_vT()
                    elif vT_guess is not None:
                        self.vT = vT_guess
                        self._compute_b()
                    elif b_guess is not None:
                        self.b = b_guess
                        self._compute_vT()
                    else: raise ValueError("Unexpected branch!")
                elif vT_guess is not None:
                    self.vT = vT_guess
                    if b_guess is None:
                        self.b = self.b_mu + b_sig * np.random.randn(self.M)
                        self._compute_u()
                    else:
                        self.b = b_guess
                        self._compute_u()
                elif b_guess is not None:
                    self.b = b_guess
                    self.u = self.u_mu + u_sig * np.random.randn(self.N - 1)
                    self._compute_vT()
                else: raise ValueError("Unexpected branch!")

            # Iterate a bit
            for i in tqdm(range(100)):
                self._compute_u()
                self.map[1:, :] = self.u
                self.b = self.map.flux(theta=self.theta).eval()
                self._compute_vT()

            # Initialize the variables to the guesses
            vars = []
            if "u" in var_names:
                u = theano.shared(self.u)
                vars += [u]
            else:
                u = tt.as_tensor_variable(self.u)
            if "vT" in var_names:
                vT = theano.shared(self.vT)
                vars += [vT]
            else:
                vT = tt.as_tensor_variable(self.vT)
            if "b" in var_names:
                d_guess = np.zeros(self.M)
                d = theano.shared(d_guess)  
                vars += [d]
            else:
                d_guess = np.zeros(self.M)
                d = tt.as_tensor_variable(d_guess)

            # The baseline is special
            self.map[1:, :] = u
            b = self.map.flux(theta=self.theta) + d            

            # Compute the model
            D = ts.as_sparse_variable(self.D)
            a = tt.reshape(tt.dot(tt.reshape(
                                  tt.concatenate([[1.0], u]), (-1, 1)), 
                                  tt.reshape(vT, (1, -1))), (-1,))
            B = tt.reshape(b, (-1, 1))
            M = tt.reshape(ts.dot(D, a), (self.M, -1)) / B

            # Compute the likelihood
            r = tt.reshape(self.F - M, (-1,))
            cov = tt.reshape(self.F_CInv, (-1,))
            lnlike = -0.5 * (tt.sum(r ** 2 * cov) + self.F_lndet)

            # Compute the prior
            lnprior = -0.5 * (tt.sum((u - self.u_mu) ** 2 * self.u_cinv) + self.u_lndet)
            lnprior += -0.5 * (tt.dot(tt.dot(tt.reshape((vT - self.vT_mu), (1, -1)), self.vT_CInv), tt.reshape((vT - self.vT_mu), (-1, 1)))[0, 0] + self.vT_lndet)
            lnprior += -0.5 * (tt.sum((b - self.b_mu) ** 2 * self.b_cinv) + self.b_lndet)

            # The full loss
            loss = -(lnlike + lnprior)

            # The optimizer
            upd = Adam(loss, vars, **kwargs)
            train = theano.function([], 
                [u, vT, b, loss, lnlike, lnprior], updates=upd)
            lnlike_val = np.zeros(niter)
            lnprior_val = np.zeros(niter)
            best_loss = np.inf
            for n in tqdm(range(niter)):
                u_val, vT_val, b_val, loss_val, lnlike_val[n], lnprior_val[n] = train()
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_u = u_val
                    best_vT = vT_val
                    best_b = b_val

            # We're done!
            self.u = best_u
            self.vT = best_vT
            self.b = best_b
            self.lnlike = lnlike_val
            self.lnprior = lnprior_val

        self._solved = True

    def plot(self, nframes=11, render_movies=False, open_plots=False, overlap=2.0):
        """

        """
        if not self._solved:
            raise RuntimeError("Please run `solve` first.")
        files = []

        # Plot the baseline
        fig, ax = plt.subplots(1, figsize=(8, 5))
        ax.plot(self.t, self.b_true, label="true")
        ax.plot(self.t, self.b, label="inferred")
        ax.legend(loc="upper right")
        ax.set_xlabel("time")
        ax.set_ylabel("baseline")
        fig.savefig("%s_baseline.pdf" % self.name, 
                    bbox_inches="tight")
        files.append("baseline.pdf")
        plt.close()

        # Plot the likelihood
        if len(np.atleast_1d(self.lnlike).flatten()) > 1:
            fig, ax = plt.subplots(2, figsize=(8, 6), sharex=True)
            ax[0].plot(-self.lnlike, label="likelihood", color="C0")
            ax[0].plot(-(self.lnlike + self.lnprior), label="total prob", color="k")
            ax[1].plot(-self.lnprior, label="prior", color="C1")
            ax[0].set_yscale("log")
            ax[1].set_yscale("log")
            ax[0].set_ylabel("negative log probability")
            ax[1].set_ylabel("negative log probability")
            ax[1].set_xlabel("iteration")
            ax[0].legend(loc="upper right")
            ax[1].legend(loc="upper right")
            fig.savefig("%s_prob.pdf" % self.name, 
                        bbox_inches="tight")
            files.append("prob.pdf")
            plt.close()

        # Plot the Ylm coeffs
        fig, ax = plt.subplots(1, figsize=(8, 5))
        ax.plot(self.u_true, label="true")
        ax.plot(self.u, label="inferred")
        ax.set_ylabel("spherical harmonic coefficient")
        ax.set_xlabel("coefficient number")
        ax.legend(loc="upper right")
        fig.savefig("%s_coeffs.pdf" % self.name, 
                    bbox_inches="tight")
        files.append("coeffs.pdf")
        plt.close()

        # Render the true map
        theta = np.linspace(-180, 180, nframes + 1)[:-1]
        self.map[1:, :] = self.u_true
        if render_movies:
            self.map.show(theta=np.linspace(-180, 180, 50), 
                          mp4="%s_true.mp4" % self.name)
            files.append("true.mp4")
        img_true_rect = self.map.render(projection="rect", res=300).eval().reshape(300, 300)

        # Render the inferred map
        self.map[1:, :] = self.u
        img = self.map.render(theta=theta).eval()
        if render_movies:
            self.map.show(theta=np.linspace(-180, 180, 50), 
                          mp4="%s_inferred.mp4" % self.name)
            files.append("inferred.mp4")
        img_rect = self.map.render(projection="rect", res=300).eval().reshape(300, 300)

        # Plot them side by side
        fig, ax = plt.subplots(2, figsize=(10, 8))
        vmin = min(np.nanmin(img_rect), np.nanmin(img_true_rect))
        vmax = max(np.nanmax(img_rect), np.nanmax(img_true_rect))
        im = ax[0].imshow(img_true_rect, origin="lower", 
                            extent=(-180, 180, -90, 90), cmap="plasma",
                            vmin=vmin, vmax=vmax)
        im = ax[1].imshow(img_rect, origin="lower", 
                            extent=(-180, 180, -90, 90), cmap="plasma",
                            vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax.ravel().tolist())
        for axis in ax:
            latlines = np.linspace(-90, 90, 7)[1:-1]
            lonlines = np.linspace(-180, 180, 13)
            for lat in latlines:
                axis.axhline(lat, color="k", lw=0.5, alpha=0.5, zorder=100)
            for lon in lonlines:
                axis.axvline(lon, color="k", lw=0.5, alpha=0.5, zorder=100)
            axis.set_xticks(lonlines)
            axis.set_yticks(latlines)
            axis.set_xlabel("Longitude [deg]", fontsize=12)
            axis.set_ylabel("Latitude [deg]", fontsize=12)
        fig.savefig("%s_rect.pdf" % self.name, 
                    bbox_inches="tight")
        files.append("rect.pdf")
        plt.close()

        # Plot the "Joy Division" graph
        fig = plt.figure(figsize=(8, 10))
        ax_img = [plt.subplot2grid((nframes, 8), (n, 0), rowspan=1, colspan=1)
                    for n in range(nframes)]
        ax_f = [plt.subplot2grid((nframes, 8), (0, 1), rowspan=1, colspan=7)]
        ax_f += [plt.subplot2grid((nframes, 8), (n, 1), rowspan=1, colspan=7, 
                    sharex=ax_f[0], sharey=ax_f[0]) for n in range(1, nframes)]
        A = (np.append([1], self.u)).reshape(-1, 1).dot(self.vT.reshape(1, -1))
        Model = self.D.dot(A.reshape(-1)).reshape(self.M, -1)
        Model /= np.reshape(self.b, (-1, 1))
        for n in range(nframes):
            ax_img[n].imshow(img[n], extent=(-1, 1, -1, 1), 
                            origin="lower", cmap="plasma", vmin=vmin,
                            vmax=vmax)
            ax_img[n].axis('off')
            m = int(np.round(np.linspace(0, self.M - 1, nframes)[n]))
            ax_f[n].plot(self.lam, self.F[m], "k.", ms=2, 
                            alpha=0.75, clip_on=False)
            ax_f[n].plot(self.lam, Model[m], 
                            "C1-", lw=1, clip_on=False)
            ax_f[n].axis('off')
        ymed = np.median(self.F)
        ydel = 0.5 * (np.max(self.F) - np.min(self.F)) / overlap
        ax_f[0].set_ylim(ymed - ydel, ymed + ydel)

        fig.savefig("%s_timeseries.pdf" % self.name, 
                    bbox_inches="tight", dpi=400)
        files.append("timeseries.pdf")
        plt.close()

        # Plot the rest frame spectrum
        fig, ax = plt.subplots(1)
        ax.plot(self.lam_padded, self.vT_true.reshape(-1), "C0-", label="true")
        ax.plot(self.lam_padded, self.vT.reshape(-1), "C1-", label="inferred")
        ax.axvspan(self.lam_padded[0], self.lam[0], color="k", alpha=0.3)
        ax.axvspan(self.lam[-1], self.lam_padded[-1], color="k", alpha=0.3)
        ax.set_xlim(self.lam_padded[0], self.lam_padded[-1])
        ax.set_xlabel(r"$\Delta \ln \lambda$")
        ax.set_ylabel(r"Normalized intensity")  
        ax.legend(loc="lower left", fontsize=14)
        fig.savefig("%s_spectrum.pdf" % self.name, bbox_inches="tight")
        files.append("spectrum.pdf")
        plt.close()

        # Open
        if open_plots:
            for file in files:
                subprocess.run(["open", "%s_%s" % (self.name, file)])


# Generate a dataset
np.random.seed(12)
solver = Solver(ydeg=5, inc=40.0, vsini=40.0, P=1.0)
solver.generate_data(nt=8, ferr=1.e-3, image="vogtstar.jpg")
solver.solve(niter=200)
solver.plot(render_movies=False, open_plots=True)