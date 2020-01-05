import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import starry
import subprocess
from scipy.linalg import cho_solve
import theano.sparse as ts

__all__ = ["plot_results"]


def plot_results(
    doppler,
    loss=[],
    cho_y1=None,
    cho_s=None,
    name="vogtstar",
    nframes=None,
    render_movies=False,
    open_plots=False,
    overlap=2.0,
    res=300,
):
    """
    Plot the results of the Doppler imaging problem for the SPOT star.

    """
    # Get the values we'll need for plotting
    ydeg = doppler.ydeg
    udeg = doppler._udeg
    u = doppler.u
    theta = doppler.theta
    y1_true = doppler.y1_true
    s_true = doppler.s_true
    s_deconv = doppler.s_deconv
    baseline_true = doppler.baseline_true
    y1 = np.array(doppler.y1)
    s = np.array(doppler.s)
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
    if cho_y1 is not None:
        U = np.triu(cho_y1[0])
        B = doppler._map.design_matrix(theta=doppler.theta).eval()[:, 1:]
        A = np.linalg.solve(U.T, B.T)
        baseline_sig = np.sqrt(np.sum(A ** 2, axis=0))
        baseline_sig_ = np.append(baseline_sig, [baseline_sig[0]])
        ax.fill_between(
            theta_,
            baseline_ - baseline_sig_,
            baseline_ + baseline_sig_,
            color="C1",
            alpha=0.25,
            lw=0,
        )
    ax.legend(loc="lower left", fontsize=14)
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
        doppler.y1 = y1_true
        doppler.s = s_true
        loss_true = doppler.loss()

        # Print for the record
        print("True loss: %.2f" % loss_true)
        print("Best loss: %.2f" % np.min(loss))

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
    ax.plot(n, y1_true, "C0-", label="true")
    lo = (doppler.y1_mu - doppler.y1_sig) * np.ones_like(y1)
    hi = (doppler.y1_mu + doppler.y1_sig) * np.ones_like(y1)
    ax.fill_between(n, lo, hi, color="C1", lw=0, alpha=0.25, label="prior")
    ax.plot(n, y1, "C1-", label="inferred")
    if cho_y1 is not None:
        cov_y1 = cho_solve(cho_y1, np.eye(doppler.N - 1))
        sig_y1 = np.sqrt(np.diag(cov_y1))
        ax.fill_between(n, y1 - sig_y1, y1 + sig_y1, color="C1", alpha=0.5)
    ax.set_ylabel("spherical harmonic coefficient")
    ax.set_xlabel("coefficient number")
    ax.legend(loc="lower right", fontsize=14)
    ax.margins(0.01, None)
    fig.savefig("%s_coeffs.pdf" % name, bbox_inches="tight")
    files.append("coeffs.pdf")
    plt.close()

    # Render the true map
    map = starry.Map(ydeg=ydeg, udeg=udeg)
    map.inc = inc
    map[1:, :] = y1_true
    if udeg > 0:
        map[1:] = u
    if nframes is None:
        nframes = len(theta)
        theta_img = np.array(theta)
    else:
        theta_img = np.linspace(-180, 180, nframes + 1)[:-1]
    if render_movies:
        map.show(theta=np.linspace(-180, 180, 50), mp4="%s_true.mp4" % name)
        files.append("true.mp4")
    img_true_rect = (
        map.render(projection="rect", res=res).eval().reshape(res, res)
    )

    # Render the inferred map
    map[1:, :] = y1
    img = map.render(theta=theta_img).eval()
    if render_movies:
        map.show(
            theta=np.linspace(-180, 180, 50), mp4="%s_inferred.mp4" % name
        )
        files.append("inferred.mp4")
    img_rect = map.render(projection="rect", res=res).eval().reshape(res, res)

    # Render the pixelwise uncertainties
    if cho_y1 is not None:

        # Compute the polynomial transform matrix
        xyz = map.ops.compute_rect_grid(tt.as_tensor_variable(res))
        P = map.ops.pT(xyz[0], xyz[1], xyz[2])[:, : doppler.N]

        # Transform it to Ylm & evaluate it
        P = ts.dot(P, map.ops.A1).eval()

        # Rotate it so north points up
        """
        R = map.ops.R([1, 0, 0], -(90.0 - inc) * np.pi / 180.0)
        for l in range(map.ydeg + 1):
            idx = slice(l ** 2, (l + 1) ** 2)
            P[:, idx] = P[:, idx].dot(R[l])
        """

        # Discard Y_{0, 0}, whose variance is zero
        P = P[:, 1:]

        # NOTE: This is the slow way of computing sigma
        # CPT = cho_solve(cho_y1, P.T)
        # cov = np.dot(P, CPT)
        # sig = np.sqrt(np.diag(cov))

        # This is the streamlined version
        U = np.triu(cho_y1[0])
        A = np.linalg.solve(U.T, P.T)
        img_sig_rect = np.sqrt(np.sum(A ** 2, axis=0)).reshape(res, res)

        # This is how I'd compute the *prior* uncertainty on the pixels
        nsamp = 1000
        prior_std = np.std(
            [
                np.dot(
                    P[0],
                    doppler.y1_sig * np.random.randn(doppler.N - 1)
                    + doppler.y1_mu,
                )
                for i in range(nsamp)
            ]
        )

    # Normalize to the maximum for plotting
    vmax = np.nanmax(img_true_rect)
    img_rect /= vmax
    img_true_rect /= vmax
    if cho_y1 is not None:
        img_sig_rect /= vmax
        prior_std /= vmax

    # Plot the maps side by side
    if cho_y1 is not None:
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
        xy=(0, 0),
        xytext=(7, 7),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=22,
        color="k",
        zorder=101,
    )
    ax[1].annotate(
        "inferred",
        xy=(0, 0),
        xytext=(7, 7),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=22,
        color="k",
        zorder=101,
    )
    if cho_y1 is not None:
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
            xy=(0, 0),
            xytext=(7, 7),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=22,
            color="k",
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
        # axis.set_xlabel("Longitude [deg]", fontsize=16)
        # axis.set_ylabel("Latitude [deg]", fontsize=16)
        for tick in (
            axis.xaxis.get_major_ticks() + axis.yaxis.get_major_ticks()
        ):
            tick.label.set_fontsize(10)
    fig.savefig("%s_rect.pdf" % name, bbox_inches="tight")
    files.append("rect.pdf")
    plt.close()

    # Plot the "Joy Division" graph
    fig = plt.figure(figsize=(8, 11.5))
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
            img[n], extent=(-1, 1, -1, 1), origin="lower", cmap="plasma"
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
    ax.plot(lnlam_padded, s_true.reshape(-1), "C0-", label="true")
    if s_deconv is not None:
        ax.plot(
            lnlam_padded,
            s_deconv.reshape(-1),
            "C1--",
            lw=1,
            alpha=0.5,
            label="guess",
        )
    ax.plot(lnlam_padded, s.reshape(-1), "C1-", label="inferred")
    if cho_s is not None:
        cov_s = cho_solve(cho_s, np.eye(doppler.Kp))
        sig_s = np.sqrt(np.diag(cov_s))
        ax.fill_between(
            lnlam_padded, s - sig_s, s + sig_s, color="C1", alpha=0.5
        )
    ax.axvspan(lnlam_padded[0], lnlam[0], color="k", alpha=0.3)
    ax.axvspan(lnlam[-1], lnlam_padded[-1], color="k", alpha=0.3)
    ax.set_xlim(lnlam_padded[0], lnlam_padded[-1])
    ax.set_xlabel(r"$\ln\left(\lambda/\lambda_\mathrm{r}\right)$")
    ax.set_ylabel(r"Normalized intensity")
    ax.legend(loc="lower left", fontsize=12)
    fig.savefig("%s_spectrum.pdf" % name, bbox_inches="tight")
    files.append("spectrum.pdf")
    plt.close()

    # Open
    if open_plots:
        for file in files:
            subprocess.run(["open", "%s_%s" % (name, file)])
