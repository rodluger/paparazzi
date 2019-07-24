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

    def __init__(self, 
            ydeg=15, 
            beta=1.5e-4,         # / vsini ~ 40 km/s \
            inc=60.,             # \                 /
            P=1.0,
            lam_max=3e-4,        # / R ~ 300,000 \
            K=199,               # \             /
            line_sigma=7.5e-6,   # ~ 0.05 Angstrom at 6430 Angstrom.
            nlines=1, 
            ferr=1.e-3, 
            vT_mu=1.0, 
            u_mu=0.0,
            vT_sig=0.3, 
            u_sig=0.01, 
            vT_rho=5e-5,
            M=99, 
            maxiter=100, 
            seed=43, 
            perturb_amp=0.25, 
            perturb_exp=2,
            remove_baseline=False, 
            b_sig=0.1,
            vT_guess_quality=0.,
            clobber=False
        ):
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
        self.perturb_amp = perturb_amp
        self.perturb_exp = perturb_exp
        self.remove_baseline = remove_baseline
        self.vT_guess_quality = vT_guess_quality
        if self.remove_baseline:
            self.b_sig = b_sig
        else:
            # Don't fit for b
            self.b_sig = 0.0

        # Figure out the cache path
        params = (ydeg, beta, inc, P, lam_max, K, 
                  line_sigma, nlines,
                  ferr, vT_mu, vT_sig, vT_rho,
                  u_mu, u_sig, M, maxiter, seed,
                  remove_baseline, b_sig,
                  perturb_amp, perturb_exp,
                  vT_guess_quality)
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
            A = self.vT_true
            B = 1.0 + 0.01 * np.random.randn(self._doppler.Kp)
            vT_guess = A * self.vT_guess_quality + \
                       B * (1 - self.vT_guess_quality)
            b_guess = np.ones(self.M)
            self._solve(vT_guess, b_guess)
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
                     lnprior=self.lnprior,
                     lnlike_true=self.lnlike_true,
                     lnprior_true=self.lnprior_true
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
            self.lnlike_true = data["lnlike_true"]
            self.lnprior_true = data["lnprior_true"]

    def _generate(self):
        """

        """
        # Instantiate a map
        map = starry.Map(self.ydeg, lazy=False)
        map.inc = self.inc
        map.load("vogtstar.jpg")
        self.u_true = np.array(map.y).reshape(-1, 1)

        # Note that we generate the spectrum on a *padded* wavelength grid
        # so we can control the behavior at the edges
        lam_padded = self._doppler.lam_padded
        self.vT_true = np.ones_like(lam_padded)
        if self.nlines > 1:
            # Create a fake spectrum w/ a bunch of lines
            for _ in range(self.nlines):
                line_amp = 0.5 * np.random.random()
                line_mu = 2.1 * (0.5 - np.random.random()) * \
                                self.lam.max()
                self.vT_true -= line_amp * \
                                np.exp(-0.5 * (lam_padded - line_mu) ** 2 / 
                                    self.line_sigma ** 2)
        else:
            # Create a fake spectrum w/ a single, centered line
            line_amp = 0.5
            line_mu = 0.0
            self.vT_true -= line_amp * \
                            np.exp(-0.5 * (lam_padded - line_mu) ** 2 / 
                                self.line_sigma ** 2)

        self.vT_true = self.vT_true.reshape(1, -1)

        # Compute the *true* map matrix
        a = self.u_true.dot(self.vT_true).reshape(-1)
        
        # Generate the synthetic spectral timeseries
        self.f_true = self._D.dot(a)

        # Remove baseline information if we're fitting for it
        if self.remove_baseline:
            F = self.f_true.reshape(self.M, self.K)
            self.b_true = np.max(F, axis=1)
            F /= self.b_true.reshape(-1, 1)
            self.f = F.reshape(-1)
        else:
            self.b_true = np.ones(self.M)

        # Add some noise
        self.f = self.f_true + \
            self.ferr * np.random.randn(self.M * self.K)

    def _solve(self, vT_guess, b_guess):
        """

        """
        solver = pp.LinearSolver(self.lam_padded,
                                 self._D, self.f.reshape(self.M, -1), 
                                 self.ferr, self.N, self._doppler.Kp,
                                 self.u_sig, self.u_mu, 
                                 self.vT_sig, self.vT_rho,
                                 self.vT_mu,
                                 self.b_sig)
        self.u, self.vT, self.b, self.model, self.lnlike, self.lnprior = \
            solver.solve(vT_guess, b_guess, maxiter=self.maxiter,
                         perturb_amp=self.perturb_amp,
                         perturb_exp=self.perturb_exp)
        self.lnlike_true, self.lnprior_true = \
            solver.lnprob(self.u_true, self.vT_true, self.b_true)

    def plot(self, nframes=11, open_plots=False):
        """

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


vogt = VogtStar(clobber=True, ydeg=15, M=31, remove_baseline=False, b_sig=0.1, 
                vT_guess_quality=1, perturb_amp=0, maxiter=2)


vogt.plot(open_plots=True)
