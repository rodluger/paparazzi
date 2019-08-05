import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import celerite
from scipy.linalg import cho_factor
np.random.seed(13)


def learn_everything(high_snr=False):
    """

    """
    # High or low SNR?
    if high_snr:
        ferr = 1e-4
        niter_bilinear = 25
        niter_adam = 2500
    else:
        ferr = 1e-3
        niter_bilinear = 10
        niter_adam = 0

    # Generate data
    D = pp.Doppler(ydeg=15)
    D.generate_data(ferr=ferr)

    # Compute loss @ true value
    D.u = D.u_true
    D.vT = D.vT_true
    loss_true = D.loss()

    # Estimate the rest frame spectrum from the deconvolved mean spectrum
    fmean = np.mean(D.F, axis=0)
    fmean -= np.mean(fmean)
    A = D.T[0]
    LInv = 1e-4 * np.eye(A.shape[1])
    vT_guess = 1 + np.linalg.solve(A.T.dot(A).toarray() + LInv, A.T.dot(fmean))
    D.vT = vT_guess

    # Pre-compute the Cholesky factorization of the GP for vT
    sig = 0.3
    rho = 3.e-5
    kernel = celerite.terms.Matern32Term(np.log(sig), np.log(rho))
    gp = celerite.GP(kernel)
    C = gp.get_matrix(D.lnlam_padded)
    cho_C = cho_factor(C)

    # Reset the map
    D.compute_u(baseline=np.ones(D.M))

    # Iterative Bi-linear solve
    loss = np.zeros(niter_bilinear)
    
    for i in tqdm(range(niter_bilinear)):
        D.compute_u()
        D.compute_vT(cho_C=cho_C)
        loss[i] = D.loss()
    
    # Refine with NAdam, slow learning rate
    if niter_adam > 0:
        loss = np.append(loss, 
            D.solve(vT_guess=D.vT, u_guess=D.u, niter=niter_adam, lr=1e-4))

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
    D = pp.Doppler(ydeg=15)
    D.generate_data(ferr=ferr)

    # Get the true baseline (assumed to be known exactly)
    D.vT = D.vT_true
    D.u = D.u_true
    baseline = np.array(D.baseline)

    # Compute u
    D.compute_u(baseline=D.baseline)

    # Plot the map
    fig = plt.figure()
    plt.plot(D.F.reshape(-1))
    plt.plot(D.model.reshape(-1))

    D.show(projection="rect")
    plt.show()


learn_everything()