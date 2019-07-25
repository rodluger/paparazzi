# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import starry
import paparazzi as pp
import os
import pickle
import subprocess
import celerite
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import block_diag, diags
from scipy.linalg import block_diag as dense_block_diag
from tqdm import tqdm
import theano.tensor as tt
import theano.sparse as ts
import theano


__all__ = ["LinearSolver", "generate_data"]


def plot(self, nframes=11, open_plots=False):
    """
    TODO!

    """
    # Plot the baseline
    fig, ax = plt.subplots(1, figsize=(8, 5))
    ax.plot(self.t, self.b_true, label="true")
    ax.plot(self.t, self.b, label="inferred")
    ax.legend(loc="upper right")
    ax.set_xlabel("time")
    ax.set_ylabel("baseline")
    fig.savefig("%s/baseline.pdf" % self._path, 
                bbox_inches="tight")
    
    # Plot the likelihood
    fig, ax = plt.subplots(2, figsize=(8, 6), sharex=True)
    ax[0].plot(-self.lnlike, label="likelihood", color="C0")
    ax[0].plot(-(self.lnlike + self.lnprior), label="total prob", color="k")
    ax[0].axhline(-self.lnlike_true, color="C0", ls="--")
    ax[0].axhline(-(self.lnlike_true + self.lnprior_true), color="k", ls="--")
    ax[1].plot(-self.lnprior, label="prior", color="C1")
    ax[1].axhline(-self.lnprior_true, color="C1", ls="--")
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].set_ylabel("negative log probability")
    ax[1].set_ylabel("negative log probability")
    ax[1].set_xlabel("iteration")
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    fig.savefig("%s/prob.pdf" % self._path, 
                bbox_inches="tight")
    plt.close()

    # Plot the Ylm coeffs
    fig, ax = plt.subplots(1, figsize=(8, 5))
    ax.plot(self.u_true, label="true")
    ax.plot(self.u, label="inferred")
    ax.set_ylabel("spherical harmonic coefficient")
    ax.set_xlabel("coefficient number")
    ax.legend(loc="upper right")
    fig.savefig("%s/coeffs.pdf" % self._path, 
                bbox_inches="tight")
    plt.close()

    # Render the true map
    theta = np.linspace(-180, 180, nframes + 1)[:-1]
    map = starry.Map(self.ydeg, lazy=False)
    map.inc = self.inc
    map[1:, :] = self.u_true.reshape(-1)[1:]
    img_true = map.render(theta=theta)
    map.show(theta=np.linspace(-180, 180, 50), 
                mp4="%s/true.mp4" % self._path)
    img_true_rect = map.render(projection="rect", res=300).reshape(300, 300)

    # Render the inferred map
    map[1:, :] = self.u.reshape(-1)[1:]
    img = map.render(theta=theta)
    map.show(theta=np.linspace(-180, 180, 50), 
                mp4="%s/inferred.mp4" % self._path)
    img_rect = map.render(projection="rect", res=300).reshape(300, 300)

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
    fig.savefig("%s/rect.pdf" % self._path, 
                bbox_inches="tight")
    plt.close()

    # Plot the "Joy Division" graph
    fig = plt.figure(figsize=(8, 10))
    ax_img = [plt.subplot2grid((nframes, 8), (n, 0), rowspan=1, colspan=1)
                for n in range(nframes)]
    ax_f = [plt.subplot2grid((nframes, 8), (0, 1), rowspan=1, colspan=7)]
    ax_f += [plt.subplot2grid((nframes, 8), (n, 1), rowspan=1, colspan=7, 
                sharex=ax_f[0], sharey=ax_f[0]) for n in range(1, nframes)]
    F = self.f.reshape(self.M, self.K)
    F_model = self.model.reshape(self.M, self.K)
    for n in range(nframes):
        ax_img[n].imshow(img[n], extent=(-1, 1, -1, 1), 
                        origin="lower", cmap="plasma", vmin=vmin,
                        vmax=vmax)
        ax_img[n].axis('off')
        m = int(np.round(np.linspace(0, self.M - 1, nframes)[n]))
        ax_f[n].plot(self.lam, F[m] / np.median(F[m]), "k.", ms=2, 
                        alpha=0.75, clip_on=False)
        ax_f[n].plot(self.lam, F_model[m] / np.median(F[m]), 
                        "C1-", lw=1, clip_on=False)
        ax_f[n].axis('off')
    ax_f[0].set_ylim(0.9, 1.1)
    fig.savefig("%s/timeseries.pdf" % self._path, 
                bbox_inches="tight", dpi=400)
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
    fig.savefig("%s/spectrum.pdf" % self._path, bbox_inches="tight")
    plt.close()

    # Open
    if open_plots:
        for file in ["coeffs.pdf", "prob.pdf", "true.mp4", "inferred.mp4", 
                        "timeseries.pdf", "spectrum.pdf", "baseline.pdf",
                        "rect.pdf"]:
            subprocess.run(["open", "%s/%s" % (self._path, file)])


def generate_data(R=3e5, nlam=200, sigma=7.5e-6, vsini=40.0, 
                  nlines=20, inc=60.0, ydeg=5, nspec=11, ferr=1.e-4):
    """Generate a synthetic Vogtstar dataset.
    
    Args:
        R (float, optional): The spectral resolution. Defaults to 3e5.
        nlam (int, optional): Number of observed wavelength bins. 
            Defaults to 200.
        sigma (float, optional): Line width in log space. Defaults to 7.5e-6,
            equivalent to ~0.05A at 6430A.
        vsini (float, optional): Equatorial projected velocity in km / s. 
            Defaults to 40.0.
        nlines (int, optional): Number of additional small lines to include. 
            Defaults to 20.
        inc (float, optional): Stellar inclination in degrees. 
            Defaults to 60.0.
        ydeg (int, optional): Degree of the Ylm expansion. Defaults to 5.
        nspec (int, optional): Number of spectra. Defaults to 11.
        ferr (float, optional): Gaussian error to add to the fluxes. 
            Defaults to 1.e-3
    """
    # The time array in units of the period
    t = np.linspace(-0.5, 0.5, nspec + 1)[:-1]

    # The log-wavelength array
    dlam = np.log(1.0 + 1.0 / R)
    lam = np.arange(-(nlam // 2), nlam // 2 + 1) * dlam

    # Pre-compute the Doppler basis
    doppler = pp.Doppler(lam, ydeg=ydeg, vsini=vsini, inc=inc, P=1.0)
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
    map = starry.Map(ydeg, lazy=False)
    map.inc = inc
    map.load("vogtstar.jpg")
    u = np.array(map.y)

    # Compute the map matrix & the flux matrix
    A = u.reshape(-1, 1).dot(vT.reshape(1, -1))
    F = D.dot(A.reshape(-1)).reshape(nspec, -1)

    # Let's divide out the baseline flux. This is a bummer,
    # since it contains really important information about
    # the map, but unfortunately we can't typically
    # measure it with a spectrograph.
    b = np.max(F, axis=1)
    F /= b.reshape(-1, 1)

    # Finally, we add some noise
    F += ferr * np.random.randn(*F.shape)

    return u[1:], vT, b, D, lam_padded, lam, F


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


class LinearSolver(object):

    def __init__(self, ydeg, lam_padded, F, ferr, D, u_mu=0.0, 
                 u_sig=0.01, vT_mu=1.0, vT_sig=0.3, vT_rho=3.e-5, 
                 invb_sig=0.1):
        # Data and Doppler instance
        self.N = (ydeg + 1) ** 2
        self.lam_padded = lam_padded
        self.F = F
        self.D = D
        self.M, self.K = self.F.shape
        self.Kp = self.D.shape[1] // self.N

        # Data covariance
        self.F_CInv = np.ones_like(F) / ferr ** 2
        self.lndet2pC = np.sum(np.log(2 * np.pi * self.F_CInv.reshape(-1)))

        # The inverse prior variance of `u`
        # The zeroth coefficient is fixed
        self.u_cinv = np.ones(self.N - 1) / u_sig ** 2
        self.u_mu = np.ones(self.N - 1) * u_mu ** 2
        self.lndet2pC_u = np.sum(np.log(2 * np.pi * self.u_cinv))

        # Gaussian process prior on vT
        self.vT_mu = (np.ones(self.Kp) * vT_mu).reshape(1, -1)
        if vT_rho > 0.0:
            kernel = celerite.terms.Matern32Term(np.log(vT_sig), np.log(vT_rho))
            gp = celerite.GP(kernel)
            vT_C = gp.get_matrix(self.lam_padded)
            cho_C = cho_factor(vT_C)
            self.vT_CInv = cho_solve(cho_C, np.eye(self.Kp))
            self.vT_CInvmu = cho_solve(cho_C, self.vT_mu.reshape(-1))
            self.lndet2pC_vT = -2 * np.sum(np.log(2 * np.pi * np.diag(cho_C[0])))
        else:
            self.vT_CInv = np.ones(self.Kp) / vT_sig ** 2
            self.vT_CInvmu = (self.vT_CInv * self.vT_mu)
            self.lndet2pC_vT = np.sum(np.log(2 * np.pi * self.vT_CInv))
            self.vT_CInv = np.diag(self.vT_CInv)

        # Prior on the (inverse) baseline
        self.invb_cinv = np.ones(self.M) / invb_sig ** 2
        self.invb_mu = np.ones(self.M)
        self.lndet2pC_invb = self.M * np.log(2 * np.pi / invb_sig ** 2)

    def u(self, vT, b):
        """
        Linear solve for `u` given `v^T` and `b`.

        """
        V = block_diag([vT.reshape(-1, 1) for n in range(self.N)])
        AFull = np.array(self.D.dot(V).todense())
        A0 = AFull[:, 0]
        A = AFull[:, 1:]
        ATCInv = np.multiply(A.T, 
            (self.F_CInv / b.reshape(-1, 1) ** 2).reshape(-1))
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, (self.F * b.reshape(-1, 1)).reshape(-1) - A0)
        np.fill_diagonal(ATCInvA, ATCInvA.diagonal() + self.u_cinv)
        u = np.linalg.solve(ATCInvA, 
                            ATCInvf + self.u_cinv * self.u_mu)
        return u

    def vT(self, u, b):
        """
        Linear solve for `v^T` given `u` and `b`.

        """
        offsets = -np.arange(0, self.N) * self.Kp
        U = diags([np.ones(self.Kp)] + 
                  [np.ones(self.Kp) * u[n] for n in range(self.N - 1)], 
                  offsets, shape=(self.N * self.Kp, self.Kp))
        A = np.array(self.D.dot(U).todense())
        ATCInv = np.multiply(A.T, 
            (self.F_CInv / b.reshape(-1, 1) ** 2).reshape(-1))
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, (self.F * b.reshape(-1, 1)).reshape(-1))
        return np.linalg.solve(ATCInvA + self.vT_CInv, 
                               ATCInvf + self.vT_CInvmu)

    def b(self, u, vT):
        """
        Linear solve for `b` given `u` and `vT`.
        Note that the problem is linear in `1 / b`, so that's the
        space in which the Gaussian priors are applied.

        """
        A = (np.append([1], u)).reshape(-1, 1).dot(vT.reshape(1, -1))
        M = self.D.dot(A.reshape(-1)).reshape(self.M, -1)
        MT = dense_block_diag(*M)
        ATCInv = np.multiply(MT, self.F_CInv.reshape(-1))
        ATCInvA = ATCInv.dot(MT.T)
        ATCInvf = np.dot(ATCInv, self.F.reshape(-1))
        np.fill_diagonal(ATCInvA, ATCInvA.diagonal() + self.invb_cinv)
        invb = np.linalg.solve(ATCInvA, 
            ATCInvf + self.invb_cinv * self.invb_mu)
        return 1.0 / invb

    def lnprob(self, u, vT, b):
        """

        """
        # Compute the model
        A = (np.append([1], u)).reshape(-1, 1).dot(vT.reshape(1, -1))
        M = self.D.dot(A.reshape(-1)).reshape(self.M, -1)

        # Compute the likelihood
        r = (self.F * b.reshape(-1, 1) - M).reshape(-1)
        lnlike = -0.5 * np.dot(
                            np.multiply(r.reshape(1, -1), 
                            (self.F_CInv / b.reshape(-1, 1) ** 2).reshape(-1)),
                            r.reshape(-1, 1)
                        ) - 0.5 * self.lndet2pC

        # Compute the priors                
        r = u - self.u_mu
        lnprior_u = -0.5 * np.dot(
                            np.multiply(r.reshape(1, -1), self.u_cinv),
                            r.reshape(-1, 1)
                        ) - 0.5 * self.lndet2pC_u

        r = vT - self.vT_mu
        lnprior_vT = -0.5 * np.dot(
                            np.dot(r.reshape(1, -1), self.vT_CInv),
                            r.reshape(-1, 1)
                        ) - 0.5 * self.lndet2pC_vT

        inv_b = 1.0 / b - self.invb_mu
        lnprior_invb = -0.5 * np.dot(
                        np.multiply(inv_b.reshape(1, -1), self.invb_cinv),
                        inv_b.reshape(-1, 1)
                    ) - 0.5 * self.lndet2pC_invb

        return lnlike, lnprior_u, lnprior_vT, lnprior_invb

    def solve(self, u=None, vT=None, b=None, 
              u_guess=None, vT_guess=None, b_guess=None,
              niter=100):
        """
        
        """
        # Simple linear solves
        if (u is not None) and (vT is not None):
            b = self.b(u, vT)
            return (u, vT, b,) + self.lnprob(u, vT, b)
        elif (u is not None) and (b is not None):
            vT = self.vT(u, b)
            return (u, vT, b,) + self.lnprob(u, vT, b)
        elif (vT is not None) and (b is not None):
            u = self.u(vT, b)
            return (u, vT, b,) + self.lnprob(u, vT, b)

        # Bi-linear
        elif b is not None:
            raise NotImplementedError("Case not implemented.")
        elif u is not None:
            raise NotImplementedError("Case not implemented.")
        elif vT is not None:
            if u_guess is None and b_guess is None:
                u_guess = np.random.randn(self.N - 1) / np.sqrt(self.u_cinv)
                b_guess = self.b(u_guess, vT)
            elif b_guess is None:
                b_guess = self.b(u_guess, vT)
            elif u_guess is None:
                u_guess = self.b(vT, b_guess)

            u = theano.shared(u_guess)
            b = theano.shared(b_guess)
            vT = tt.as_tensor_variable(vT)

            # Compute the model
            D = ts.as_sparse_variable(self.D)
            a = tt.reshape(tt.dot(tt.reshape(tt.concatenate([[1.0], u]), (-1, 1)), 
                                  tt.reshape(vT, (1, -1))), (-1,))
            M = tt.reshape(ts.dot(D, a), (self.M, -1))

            # Compute the likelihood
            r = tt.reshape(self.F * tt.reshape(b, (-1, 1)) - M, (-1,))
            cov = tt.reshape(self.F_CInv / tt.reshape(b ** 2, (-1, 1)), (-1,))
            lnlike = -0.5 * (tt.sum(r ** 2 * cov) + self.lndet2pC)

            # Compute the priors
            r = u - self.u_mu
            lnprior_u = -0.5 * (tt.sum(r ** 2 * self.u_cinv) + self.lndet2pC_u)
            r = 1.0 / b - self.invb_mu
            lnprior_invb = -0.5 * (tt.sum(r ** 2 * self.invb_cinv) + self.lndet2pC_invb)

            # The full loss
            loss = -(lnlike + lnprior_u + lnprior_invb)

            # The optimizer
            upd = Adam(loss, [u, b])
            train = theano.function([], [lnlike, lnprior_u, lnprior_invb], updates=upd)
            lnlike_val = np.zeros(niter)
            lnprior_u_val = np.zeros(niter)
            lnprior_invb_val = np.zeros(niter)
            lnprior_vT_val = np.zeros(niter)
            for n in tqdm(range(niter)):
                lnlike_val[n], lnprior_u_val[n], lnprior_invb_val[n] = train()

            # Evaluate and return
            u = u.eval()
            vT = vT.eval()
            b = b.eval()
            return u, vT, b, lnlike_val, lnprior_u_val, lnprior_vT_val, lnprior_invb_val

        # Full tri-linear solve
        else:
            # TODO
            raise NotImplementedError("Case not yet implemented.")


# Generate a dataset
np.random.seed(12)
ydeg = 5
ferr = 1.e-3
u_true, vT_true, b_true, D, lam_padded, lam, F = generate_data(ydeg=ydeg, ferr=ferr, nspec=31)

# Solve
solver = LinearSolver(ydeg, lam_padded, F, ferr, D)
u, vT, b, lnlike, lnprior_u, lnprior_vT, lnprior_invb = solver.solve(vT=vT_true, niter=99, u_guess=u_true, b_guess=b_true)
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(u_true)
ax[0, 0].plot(u)
ax[0, 1].plot(vT_true)
ax[0, 1].plot(vT)
ax[1, 0].plot(b_true)
ax[1, 0].plot(b)
ax[1, 1].plot(-lnlike)
ax[1, 1].plot(-(lnprior_u + lnprior_vT + lnprior_invb))
ax[1, 1].set_yscale("log")
lnlike, lnprior_u, lnprior_vT, lnprior_invb = solver.lnprob(u_true, vT_true, b_true)
ax[1, 1].axhline(-lnlike, color="C0")
ax[1, 1].axhline(-(lnprior_u + lnprior_vT + lnprior_invb), color="C1")
plt.show()

