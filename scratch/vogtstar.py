import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt
import starry
import subprocess
np.random.seed(13)


def plot(doppler, loss=[], name="vogtstar", nframes=None, 
         render_movies=False, open_plots=False, overlap=2.0):
    """

    """
    # Get the values we'll need for plotting
    ydeg = doppler.ydeg
    theta = doppler.theta
    u_true = doppler.u_true
    vT_true = doppler.vT_true
    baseline_true = doppler.baseline_true
    u = np.array(doppler.u)
    vT = np.array(doppler.vT)
    baseline = doppler.baseline()
    model = doppler.model()
    F = doppler.F
    lnlam  = doppler.lnlam
    lnlam_padded = doppler.lnlam_padded
    M = doppler.M
    inc = doppler.inc

    # List of figure files we're generating
    files = []

    # Plot the baseline
    fig, ax = plt.subplots(1, figsize=(8, 5))
    ax.plot(theta, baseline_true, label="true")
    ax.plot(theta, baseline, label="inferred")
    ax.legend(loc="upper right")
    ax.set_xlabel("time")
    ax.set_ylabel("baseline")
    fig.savefig("%s_baseline.pdf" % name, bbox_inches="tight")
    files.append("baseline.pdf")
    plt.close()

    # Plot the loss
    if len(np.atleast_1d(loss).flatten()) > 1:

        # Compute the loss @ true value
        doppler.u = u_true
        doppler.vT = vT_true
        loss_true = doppler.loss

        # Plot
        fig, ax = plt.subplots(1, figsize=(8, 5))
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
    fig, ax = plt.subplots(1, figsize=(8, 5))
    ax.plot(u_true, label="true")
    ax.plot(u, label="inferred")
    ax.set_ylabel("spherical harmonic coefficient")
    ax.set_xlabel("coefficient number")
    ax.legend(loc="upper right")
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
    img_true_rect = map.render(projection="rect", res=300).reshape(300, 300)

    # Render the inferred map
    map[1:, :] = u
    img = map.render(theta=theta_img)
    if render_movies:
        map.show(theta=np.linspace(-180, 180, 50), 
                 mp4="%s_inferred.mp4" % name)
        files.append("inferred.mp4")
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
    fig.savefig("%s_rect.pdf" % name, bbox_inches="tight")
    files.append("rect.pdf")
    plt.close()

    # Plot the "Joy Division" graph
    fig = plt.figure(figsize=(8, 10))
    ax_img = [plt.subplot2grid((nframes, 8), (n, 0), rowspan=1, colspan=1)
                for n in range(nframes)]
    ax_f = [plt.subplot2grid((nframes, 8), (0, 1), rowspan=1, colspan=7)]
    ax_f += [plt.subplot2grid((nframes, 8), (n, 1), rowspan=1, colspan=7, 
                sharex=ax_f[0], sharey=ax_f[0]) for n in range(1, nframes)]
    for n in range(nframes):
        ax_img[n].imshow(img[n], extent=(-1, 1, -1, 1), 
                         origin="lower", cmap="plasma", vmin=vmin,
                         vmax=vmax)
        ax_img[n].axis('off')
        m = int(np.round(np.linspace(0, M - 1, nframes)[n]))
        ax_f[n].plot(lnlam, F[m], "k.", ms=2, alpha=0.75, clip_on=False)
        ax_f[n].plot(lnlam, model[m], "C1-", lw=1, clip_on=False)
        ax_f[n].axis('off')
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
    ax.axvspan(lnlam_padded[0], lnlam[0], color="k", alpha=0.3)
    ax.axvspan(lnlam[-1], lnlam_padded[-1], color="k", alpha=0.3)
    ax.set_xlim(lnlam_padded[0], lnlam_padded[-1])
    ax.set_xlabel(r"$\Delta \ln \lambda$")
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

    """
    # High or low SNR?
    if high_snr:
        ferr = 1e-4
        niter1 = 25
        niter2 = 2500
    else:
        ferr = 1e-3
        niter1 = 10
        niter2 = 0

    # Generate data
    D = pp.Doppler(ydeg=15)
    D.generate_data(ferr=ferr)

    # Compute loss @ true value
    D.u = D.u_true
    D.vT = D.vT_true
    loss_true = D.loss()

    # Solve!
    loss = D.solve(niter1=niter1, niter2=niter2, lr=1e-4)

    # Plot the results
    fig, ax = plt.subplots(1, figsize=(6, 8))
    ax.plot(loss)
    ax.axhline(loss_true, color="C1", ls="--")
    ax.set_yscale("log")

    fig = plt.figure()
    plt.plot(D.vT_true)
    plt.plot(D.vT)

    fig = plt.figure()
    plt.plot(D.F.reshape(-1))
    plt.plot(D.model.reshape(-1))

    # Render the true map
    D._map[1:, :] = D.u_true
    img_true_rect = D._map.render(projection="rect", res=300).eval().reshape(300, 300)

    # Render the inferred map
    D._map[1:, :] = D.u
    img_rect = D._map.render(projection="rect", res=300).eval().reshape(300, 300)
        
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
    plt.show()


def learn_map(high_snr=False):
    """

    """
    # High or low SNR?
    if high_snr:
        ferr = 1e-4
    else:
        ferr = 1e-3

    # Generate data
    dop = pp.Doppler(ydeg=15)
    dop.generate_data(ferr=ferr)

    # Get the true baseline (assumed to be known exactly)
    dop.u = dop.u_true
    baseline = dop.baseline()
    vT = dop.vT_true

    # Compute u
    dop.solve(vT=vT, baseline=baseline)

    plot(dop, open_plots=True, render_movies=True)

learn_map(True)