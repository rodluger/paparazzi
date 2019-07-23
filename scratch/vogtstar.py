# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import starry
import paparazzi as pp
import os
import pickle
import subprocess


class VogtStar(object):
    """

    """

    def __init__(self, ydeg=15, beta=2.e-6, inc=60., P=1.0,
                 lam_max=2.e-5, K=399, line_sigma=2.e-7,
                 nlines=30, ferr=1.e-3, vT_mu=1.0, u_mu=0.0,
                 vT_sig=0.3, u_sig=0.01, vT_rho=1e-6,
                 M=99, maxiter=100, seed=43, 
                 fit_baseline=False,
                 clobber=False):
        """
        
        """
        self.ydeg = ydeg
        self.N = (self.ydeg + 1) ** 2
        self.beta = beta
        self.inc = inc
        self.P = P
        self.K = K
        self.lam = np.linspace(-lam_max, lam_max, self.K)
        self.line_sigma = line_sigma
        self.nlines = nlines
        self.ferr = ferr
        self.vT_mu = vT_mu
        self.vT_sig = vT_sig
        self.vT_rho = vT_rho
        self.u_mu = np.ones(self.N) * u_mu
        self.u_mu[0] = 1.0
        self.u_sig = np.ones(self.N) * u_sig
        self.u_sig[0] = 1e-6 # hack to force Y00 ~ unity
        self.M = M
        self.t = np.linspace(-0.5 * self.P, 0.5 * self.P, self.M + 1)[:-1]
        self.maxiter = maxiter
        self.seed = seed
        self.fit_baseline = fit_baseline

        # Figure out the cache path
        params = (ydeg, beta, inc, P, lam_max, K, 
                  line_sigma, nlines,
                  ferr, vT_mu, vT_sig, vT_rho,
                  u_mu, u_sig, M, maxiter, seed,
                  fit_baseline)
        self._path = ".vogtstar%s" % hex(abs(hash(params)))[2:]
        if not os.path.exists(self._path):
            os.mkdir(self._path)

        # Compute or load from cache
        if not os.path.exists("%s/vogtstar.npz" % self._path) or clobber:
            np.random.seed(self.seed)
            print("Computing Doppler basis...")
            self._doppler = pp.Doppler(self.lam, ydeg=self.ydeg, 
                                       beta=self.beta, inc=self.inc, 
                                       P=self.P)
            self.lam_padded = self._doppler.lam_padded
            self._D = self._doppler.D(t=self.t)
            print("Generating synthetic dataset...")
            self._generate()
            print("Solving the bilinear problem...")
            self._solve()
            np.savez("%s/vogtstar.npz" % self._path,
                     u_true=self.u_true,
                     vT_true=self.vT_true,
                     f_true=self.f_true,
                     b_true=self.b_true,
                     f=self.f,
                     u=self.u,
                     vT=self.vT,
                     b=self.b,
                     lam_padded=self.lam_padded,
                     model=self.model,
                     lnlike=self.lnlike,
                     lnprior=self.lnprior
            )
        else:
            print("Loading from cache...")
            data = np.load("%s/vogtstar.npz" % self._path)
            self.u_true = data["u_true"]
            self.vT_true = data["vT_true"]
            self.f_true = data["f_true"]
            self.b_true = data["b_true"]
            self.f = data["f"]
            self.u = data["u"]
            self.vT = data["vT"]
            self.b = data["b"]
            self.lam_padded = data["lam_padded"]
            self.model = data["model"]
            self.lnlike = data["lnlike"]
            self.lnprior = data["lnprior"]

    def _generate(self):
        """

        """
        # Instantiate a map
        map = starry.Map(self.ydeg, lazy=False)
        map.inc = self.inc
        map.load("vogtstar.jpg")
        self.u_true = np.array(map.y).reshape(-1, 1)

        # Create a fake spectrum w/ a bunch of lines
        # Note that we generate it on a *padded* wavelength grid
        # so we can control the behavior at the edges
        lam_padded = self._doppler.lam_padded
        self.vT_true = np.ones_like(lam_padded)
        for _ in range(self.nlines):
            line_amp = 0.5 * np.random.random()
            line_mu = 2.1 * (0.5 - np.random.random()) * \
                            self.lam.max()
            self.vT_true -= line_amp * \
                            np.exp(-0.5 * (lam_padded - line_mu) ** 2 / 
                                   self.line_sigma ** 2)
        self.vT_true = self.vT_true.reshape(1, -1)

        # Compute the *true* map matrix
        a = self.u_true.dot(self.vT_true).reshape(-1)
        
        # Generate the synthetic spectral timeseries
        self.f_true = self._D.dot(a)

        # Remove baseline information if we're fitting for it
        if self.fit_baseline:
            F = self.f_true.reshape(self.M, self.K)
            self.b_true = np.max(F, axis=1) - 1
            F /= (1 + self.b_true.reshape(-1, 1))
            self.f = F.reshape(-1)
        else:
            self.b_true = np.zeros(self.M)

        # Add some noise
        self.f = self.f_true + \
            self.ferr * np.random.randn(self.M * self.K)

    def _solve(self, u=None, vT=None):
        """

        """
        solver = pp.LinearSolver(self.lam_padded,
                                 self._D, self.f.reshape(self.M, -1), 
                                 self.ferr, self.N, self._doppler.Kp,
                                 self.u_sig, self.u_mu, 
                                 self.vT_sig, self.vT_rho,
                                 self.vT_mu,
                                 self.fit_baseline)
        if u is None and vT is None:
            vT = 1.0 + 0.01 * np.random.randn(self._doppler.Kp)
        
        self.u, self.vT, self.b, self.model, self.lnlike, self.lnprior = \
            solver.solve(u=u, vT=vT, maxiter=self.maxiter)

    def plot(self, nframes=11, open_plots=False):
        """

        """
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

        # Plot the likelihood
        fig, ax = plt.subplots(1, figsize=(8, 5))
        ax.plot(-self.lnlike, label="likelihood", color="C0")
        ax.plot(-self.lnprior, label="prior", color="C1")
        ax.plot(-(self.lnlike + self.lnprior), label="total prob", color="k")
        ax.set_yscale("log")
        ax.set_ylabel("negative log probability")
        ax.set_xlabel("iteration")
        ax.legend(loc="upper right")
        fig.savefig("%s/prob.pdf" % self._path, 
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

        # Render the inferred map
        map[1:, :] = self.u.reshape(-1)[1:]
        img = map.render(theta=theta)
        map.show(theta=np.linspace(-180, 180, 50), 
                 mp4="%s/inferred.mp4" % self._path)

        # Plot the "Joy Division" graph
        fig = plt.figure(figsize=(8, 10))
        ax_img = [plt.subplot2grid((nframes, 8), (n, 0), rowspan=1, colspan=1)
                    for n in range(nframes)]
        ax_f = [plt.subplot2grid((nframes, 8), (0, 1), rowspan=1, colspan=7)]
        ax_f += [plt.subplot2grid((nframes, 8), (n, 1), rowspan=1, colspan=7, 
                    sharex=ax_f[0], sharey=ax_f[0]) for n in range(1, nframes)]
        F = self.f.reshape(self.M, self.K)
        F_model = self.model.reshape(self.M, self.K)
        vmin = min(np.nanmin(img), np.nanmin(img_true))
        vmax = max(np.nanmax(img), np.nanmax(img_true))
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
                         "timeseries.pdf", "spectrum.pdf"]:
                subprocess.run(["open", "%s/%s" % (self._path, file)])


vogt = VogtStar(clobber=True, ydeg=5, M=31, fit_baseline=True)
vogt.plot(open_plots=True)


# DEBUG
plt.plot(vogt.b)
plt.plot(vogt.b_true)
plt.show()