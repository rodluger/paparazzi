import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import starry
import subprocess
from scipy.linalg import cho_solve
import theano.sparse as ts

np.random.seed(13)


def plot_results(
    doppler,
    loss=[],
    cho_u=None,
    cho_vT=None,
    name="vogtstar",
    nframes=None,
    render_movies=False,
    open_plots=False,
    overlap=2.0,
    res=300,
):
    """
    Plot the results of the Doppler imaging problem for the Vogtstar.

    """
    # Get the values we'll need for plotting
    ydeg = doppler.ydeg
    theta = doppler.theta
    u_true = doppler.u_true
    vT_true = doppler.vT_true
    baseline_true = doppler.baseline_true
    u = np.array(doppler.u)
    vT = np.array(doppler.vT)
    baseline = doppler.baseline().reshape(-1)
    model = doppler.model()
    F = doppler.F
    lnlam = doppler.lnlam
    lnlam_padded = doppler.lnlam_padded
    M = doppler.M
    inc = doppler.inc

    # List of figure files we're generating
    files = []

    # Plot the baseline
    # HACK: Append the first measurement to the last to get
    # a plot going from -180 to 180 (endpoints included)
    theta_ = np.append(theta, [180.0])
    baseline_true_ = np.append(baseline_true, [baseline_true[0]])
    baseline_ = np.append(baseline, [baseline[0]])
    fig, ax = plt.subplots(1, figsize=(8, 5))
    ax.plot(theta_, baseline_true_, label="true")
    ax.plot(theta_, baseline_, label="inferred")
    if cho_u is not None:
        U = np.triu(cho_u[0])
        B = doppler._map.X(theta=doppler.theta).eval()[:, 1:]
        A = np.linalg.solve(U.T, B.T)
        baseline_sig = np.sqrt(np.sum(A ** 2, axis=0))
        baseline_sig_ = np.append(baseline_sig, [baseline_sig[0]])
        ax.fill_between(
            theta,
            baseline - baseline_sig,
            baseline + baseline_sig,
            color="C1",
            alpha=0.25,
            lw=0,
        )
    ax.legend(loc="lower left")
    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.margins(0, None)
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_ylabel("baseline")
    fig.savefig("%s_baseline.pdf" % name, bbox_inches="tight")
    files.append("baseline.pdf")
    plt.close()

    # Plot the loss
    if len(np.atleast_1d(loss).flatten()) > 1:

        # Compute the loss @ true value
        doppler.u = u_true
        doppler.vT = vT_true
        loss_true = doppler.loss()

        # Plot
        fig, ax = plt.subplots(1, figsize=(12, 5))
        ax.plot(loss, label="loss", color="C0")
        ax.axhline(loss_true, color="C1", ls="--", label="loss @ true values")
        ax.set_yscale("log")
        ax.set_ylabel("negative log probability")
        ax.set_xlabel("iteration")
        ax.legend(loc="upper right")
        fig.savefig("%s_prob.pdf" % name, bbox_inches="tight")
        files.append("prob.pdf")
        plt.close()

    # Plot the Ylm coeffs
    fig, ax = plt.subplots(1, figsize=(12, 5))
    n = np.arange(1, doppler.N)
    ax.plot(n, u_true, "C0-", label="true")
    lo = (doppler.u_mu - doppler.u_sig) * np.ones_like(u)
    hi = (doppler.u_mu + doppler.u_sig) * np.ones_like(u)
    ax.fill_between(n, lo, hi, color="C1", lw=0, alpha=0.25, label="prior")
    ax.plot(n, u, "C1-", label="inferred")
    if cho_u is not None:
        cov_u = cho_solve(cho_u, np.eye(doppler.N - 1))
        sig_u = np.sqrt(np.diag(cov_u))
        ax.fill_between(n, u - sig_u, u + sig_u, color="C1", alpha=0.5)
    ax.set_ylabel("spherical harmonic coefficient")
    ax.set_xlabel("coefficient number")
    ax.legend(loc="upper right")
    ax.margins(0.01, None)
    fig.savefig("%s_coeffs.pdf" % name, bbox_inches="tight")
    files.append("coeffs.pdf")
    plt.close()

    # Render the true map
    map = starry.Map(ydeg=ydeg, lazy=False)
    map.inc = inc
    map[1:, :] = u_true
    if nframes is None:
        nframes = len(theta)
        theta_img = np.array(theta)
    else:
        theta_img = np.linspace(-180, 180, nframes + 1)[:-1]
    if render_movies:
        map.show(theta=np.linspace(-180, 180, 50), mp4="%s_true.mp4" % name)
        files.append("true.mp4")
    img_true_rect = map.render(projection="rect", res=res).reshape(res, res)

    # Render the inferred map
    map[1:, :] = u
    img = map.render(theta=theta_img)
    if render_movies:
        map.show(
            theta=np.linspace(-180, 180, 50), mp4="%s_inferred.mp4" % name
        )
        files.append("inferred.mp4")
    img_rect = map.render(projection="rect", res=res).reshape(res, res)

    # Render the pixelwise uncertainties
    if cho_u is not None:

        # Compute the polynomial transform matrix
        xyz = map.ops.compute_rect_grid(res)
        P = map.ops.pT(xyz[0], xyz[1], xyz[2])

        # Transform it to Ylm & evaluate it
        P = ts.dot(P, map.ops.A1).eval()

        # Rotate it so north points up
        R = map.ops.R([1, 0, 0], -(90.0 - inc) * np.pi / 180.0)
        for l in range(map.ydeg + 1):
            idx = slice(l ** 2, (l + 1) ** 2)
            P[:, idx] = P[:, idx].dot(R[l])

        # Discard Y_{0, 0}, whose variance is zero
        P = P[:, 1:]

        # NOTE: This is the slow way of computing sigma
        # CPT = cho_solve(cho_u, P.T)
        # cov = np.dot(P, CPT)
        # sig = np.sqrt(np.diag(cov))

        # This is the streamlined version
        U = np.triu(cho_u[0])
        A = np.linalg.solve(U.T, P.T)
        img_sig_rect = np.sqrt(np.sum(A ** 2, axis=0)).reshape(res, res)

        # This is how I'd compute the *prior* uncertainty on the pixels
        nsamp = 1000
        prior_std = np.std(
            [
                np.dot(
                    P[0],
                    doppler.u_sig * np.random.randn(doppler.N - 1)
                    + doppler.u_mu,
                )
                for i in range(nsamp)
            ]
        )

    # Normalize to the maximum for plotting
    vmax = max(np.nanmax(img_rect), np.nanmax(img_true_rect))
    img /= vmax
    img_rect /= vmax
    img_true_rect /= vmax
    if cho_u is not None:
        img_sig_rect /= vmax
        prior_std /= vmax

    # Plot the maps side by side
    if cho_u is not None:
        fig, ax = plt.subplots(3, figsize=(10, 13))
        fig.subplots_adjust(hspace=0.3)
    else:
        fig, ax = plt.subplots(2, figsize=(10, 8))
    im = ax[0].imshow(
        img_true_rect,
        origin="lower",
        extent=(-180, 180, -90, 90),
        cmap="plasma",
        vmin=0,
        vmax=1,
    )
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.25)
    plt.colorbar(im, cax=cax, format="%.2f")
    im = ax[1].imshow(
        img_rect,
        origin="lower",
        extent=(-180, 180, -90, 90),
        cmap="plasma",
        vmin=0,
        vmax=1,
    )
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.25)
    plt.colorbar(im, cax=cax, format="%.2f")
    ax[0].annotate(
        "true",
        xy=(0, 1),
        xytext=(7, -7),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=22,
        color="w",
        zorder=101,
    )
    ax[1].annotate(
        "inferred",
        xy=(0, 1),
        xytext=(7, -7),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=22,
        color="w",
        zorder=101,
    )
    if cho_u is not None:
        im = ax[2].imshow(
            img_sig_rect,
            origin="lower",
            extent=(-180, 180, -90, 90),
            cmap="plasma",
            vmin=0,
            vmax=prior_std,
        )
        ticks = np.linspace(0, prior_std, 5)
        ticklabels = ["%.2f" % t for t in ticks]
        ticklabels[-1] = r"$\sigma_\mathrm{prior}$"
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="4%", pad=0.25)
        cb = plt.colorbar(im, cax=cax, format="%.2f", ticks=ticks)
        cb.ax.set_yticklabels(ticklabels)
        ax[2].annotate(
            "uncertainty",
            xy=(0, 1),
            xytext=(7, -7),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=22,
            color="w",
            zorder=101,
        )
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
    fig.savefig("%s_rect.pdf" % name, bbox_inches="tight")
    files.append("rect.pdf")
    plt.close()

    # Plot the "Joy Division" graph
    fig = plt.figure(figsize=(8, 10))
    ax_img = [
        plt.subplot2grid((nframes, 8), (n, 0), rowspan=1, colspan=1)
        for n in range(nframes)
    ]
    ax_f = [plt.subplot2grid((nframes, 8), (0, 1), rowspan=1, colspan=7)]
    ax_f += [
        plt.subplot2grid(
            (nframes, 8),
            (n, 1),
            rowspan=1,
            colspan=7,
            sharex=ax_f[0],
            sharey=ax_f[0],
        )
        for n in range(1, nframes)
    ]
    for n in range(nframes):
        ax_img[n].imshow(
            img[n],
            extent=(-1, 1, -1, 1),
            origin="lower",
            cmap="plasma",
            vmin=0,
            vmax=1,
        )
        ax_img[n].axis("off")
        m = int(np.round(np.linspace(0, M - 1, nframes)[n]))
        ax_f[n].plot(lnlam, F[m], "k.", ms=2, alpha=0.75, clip_on=False)
        ax_f[n].plot(lnlam, model[m], "C1-", lw=1, clip_on=False)
        ax_f[n].axis("off")
    ymed = np.median(F)
    ydel = 0.5 * (np.max(F) - np.min(F)) / overlap
    ax_f[0].set_ylim(ymed - ydel, ymed + ydel)
    fig.savefig("%s_timeseries.pdf" % name, bbox_inches="tight", dpi=400)
    files.append("timeseries.pdf")
    plt.close()

    # Plot the rest frame spectrum
    fig, ax = plt.subplots(1)
    ax.plot(lnlam_padded, vT_true.reshape(-1), "C0-", label="true")
    ax.plot(lnlam_padded, vT.reshape(-1), "C1-", label="inferred")
    if cho_vT is not None:
        cov_vT = cho_solve(cho_vT, np.eye(doppler.Kp))
        sig_vT = np.sqrt(np.diag(cov_vT))
        ax.fill_between(
            lnlam_padded, vT - sig_vT, vT + sig_vT, color="C1", alpha=0.5
        )
    ax.axvspan(lnlam_padded[0], lnlam[0], color="k", alpha=0.3)
    ax.axvspan(lnlam[-1], lnlam_padded[-1], color="k", alpha=0.3)
    ax.set_xlim(lnlam_padded[0], lnlam_padded[-1])
    ax.set_xlabel(r"$\lambdabar$")
    ax.set_ylabel(r"Normalized intensity")
    ax.legend(loc="lower left", fontsize=14)
    fig.savefig("%s_spectrum.pdf" % name, bbox_inches="tight")
    files.append("spectrum.pdf")
    plt.close()

    # Open
    if open_plots:
        for file in files:
            subprocess.run(["open", "%s_%s" % (name, file)])


def learn_everything(high_snr=False):
    """
    In this case, we know nothing: we're going to learn
    both the map and the spectrum. We're not giving the
    algorithm an initial guess, either: the spectrum is
    learned by deconvolving the data, and the initial
    guess for the map is computed via the linearized
    problem.

    """
    # High or low SNR?
    if high_snr:
        # We rely heavily on tempering here. Once we get
        # a good initial guess via the bilinear solver,
        # we run the non-linear solver with a slow learning
        # rate.
        ferr = 1e-4
        T = 5000.0
        niter = 1000
        lr = 1e-4
        name = "learn_everything_high_snr"
    else:
        # This case is easier; just a little tempering for
        # good measure, followed by a fast non-linear
        # refinement.
        ferr = 1e-3
        T = 10.0
        niter = 250
        lr = 2e-3
        name = "learn_everything_low_snr"

    # Generate data
    dop = pp.Doppler(ydeg=15)
    dop.generate_data(ferr=ferr)

    # Reset all coefficients
    dop.u = None
    dop.vT = None

    # Solve!
    loss, cho_u, cho_vT = dop.solve(niter=niter, lr=lr, T=T)
    plot_results(dop, name=name, loss=loss, cho_u=cho_u, cho_vT=cho_vT)


def learn_map(high_snr=False):
    """
    In this case, we know the spectrum and the baseline
    perfectly. The problem is linear in the map, so solving
    the Doppler problem is easy!

    """
    # High or low SNR?
    if high_snr:
        ferr = 1e-4
        name = "learn_map_high_snr"
    else:
        ferr = 1e-3
        name = "learn_map_low_snr"

    # Generate data
    dop = pp.Doppler(ydeg=15)
    dop.generate_data(ferr=ferr)

    # Compute the true baseline (assumed to be known exactly)
    dop.u = dop.u_true
    baseline = dop.baseline()

    # Reset all coefficients
    dop.vT = dop.vT_true
    dop.u = None

    # Solve!
    loss, cho_u, cho_vT = dop.solve(vT=dop.vT, baseline=baseline)
    plot_results(dop, name=name, loss=loss, cho_u=cho_u, cho_vT=cho_vT)


def learn_map_and_baseline(high_snr=False):
    """
    In this case, we know the rest frame spectrum, but we don't
    know the map coefficients or the baseline flux. The problem
    can be linearized to solve for the coefficients, and then
    refined with the non-linear solver.

    """
    # High or low SNR?
    if high_snr:
        # At high SNR, we need to do a bit of refinement
        # with the non-linear solver.
        ferr = 1e-4
        niter = 50
        lr = 1e-4
        name = "learn_map_baseline_high_snr"
    else:
        # At low SNR, a single run of the bi-linear solver
        # gets us to the optimum!
        ferr = 1e-3
        niter = 0
        lr = None
        name = "learn_map_baseline_low_snr"

    # Generate data
    dop = pp.Doppler(ydeg=15)
    dop.generate_data(ferr=ferr)

    # Reset all coefficients
    dop.vT = dop.vT_true
    dop.u = None

    # Solve!
    loss, cho_u, cho_vT = dop.solve(vT=dop.vT, niter=niter, lr=lr)
    plot_results(dop, name=name, loss=loss, cho_u=cho_u, cho_vT=cho_vT)


if __name__ == "__main__":
    learn_map(high_snr=True)
