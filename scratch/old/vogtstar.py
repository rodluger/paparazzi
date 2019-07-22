# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import starry
from utils import RigidRotationSolver
from tqdm import tqdm
import exoplanet as xo
from scipy.sparse import block_diag, diags
import theano.tensor as tt
import theano.sparse as ts
import pymc3 as pm
import os
np.random.seed(13)


# The save file
filename = "output/vogtstar_monday"
clobber = False

# Run?
if (clobber) or (not os.path.exists("%s.npz" % filename)):

    # Params
    params = dict(
        ydeg = 6,
        niter = 200,
        lam_max = 2e-5,
        K = 399,                     # Number of wavs observed
        inc = 60.0,
        beta = 2.e-6,
        line_sigma = 2.e-7,
        P = 1.0,
        ferr = 1.0e-4,
        sig_vT = 0.3,
        sig_u = 0.01,
        sig_u0 = 1e-5,
        M = 11,                      # Number of observations
        ntheta = 12                  # Number of images to display
    )
    locals().update(params)

    # Instantiate a map
    N = (ydeg + 1) ** 2
    map = starry.Map(ydeg, lazy=False)
    map.inc = inc
    map.load("vogtstar.jpg")
    ylms = np.array(map.y)
    img = map.render(theta=np.linspace(-180, 180, ntheta + 1)[:-1])
    map.show(theta=np.linspace(-180, 180, 50), mp4="%s_true.mp4" % filename)

    # Log wavelength array
    lam = np.linspace(-lam_max, lam_max, K)

    # Instantiate the solver
    solver = RigidRotationSolver(lam, ydeg=ydeg, beta=beta, inc=inc, P=P)

    # Create a fake spectrum w/ a bunch of lines
    # Note that we generate it on a *padded* wavelength grid
    # so we can control the behavior at the edges
    lam_padded = solver.lam_padded
    Kp = len(lam_padded)
    I0_padded = np.ones_like(lam_padded)
    for i in range(30):
        line_amp = 0.5 * np.random.random()
        line_mu = 2.1 * (0.5 - np.random.random()) * lam_max
        I0_padded -= line_amp * np.exp(-0.5 * (lam_padded - line_mu) ** 2 / 
                                    line_sigma ** 2)
    I0 = I0_padded[solver.mask]

    # Compute the *true* map matrix
    A = ylms.reshape(-1, 1).dot(I0_padded.reshape(1, -1))
    a = A.reshape(-1)

    # Generate the synthetic spectral timeseries
    t = np.linspace(-0.5 * P, 0.5 * P, M + 1)[:-1]
    D = solver.D(t=t)
    f_true = D.dot(a)
    F_true = f_true.reshape(M, K)

    # Add some noise
    f = f_true + ferr * np.random.randn(M * K)
    F = f.reshape(M, K)

    # Let's construct our initial guesses. 
    print("Computing initial guess...")

    # Linear solve for u | vT
    def get_u(vT):
        V = block_diag([vT.reshape(-1, 1) for n in range(N)])
        A = np.array(D.dot(V).todense())
        ATA = A.T.dot(A)
        diag = np.ones(N) * ferr ** 2
        diag[0] /= sig_u0 ** 2
        diag[1:] /= sig_u ** 2
        ATA[np.diag_indices_from(ATA)] += diag
        LInvmu = np.zeros(N)
        LInvmu[0] = ferr ** 2 / sig_u0 ** 2
        ATf = np.dot(A.T, f.reshape(-1, 1)) + LInvmu.reshape(-1, 1)
        return np.linalg.solve(ATA, ATf).reshape(-1)

    # Linear solve for vT | u
    def get_vT(u):
        offsets = -np.arange(0, N) * Kp
        U = diags([np.ones(Kp) * u[n] for n in range(N)], offsets, 
                    shape=(N * Kp, Kp))
        A = np.array(D.dot(U).todense())
        ATA = A.T.dot(A)
        diag = np.ones(Kp) * ferr ** 2 / sig_vT ** 2
        ATA[np.diag_indices_from(ATA)] += diag
        LInvmu = np.ones(Kp) * ferr ** 2 / sig_vT ** 2
        ATf = np.dot(A.T, f.reshape(-1, 1)) + LInvmu.reshape(-1, 1)
        return np.linalg.solve(ATA, ATf).reshape(-1)
    
    # Random initial spectrum
    vT_guess0 = 1.0 + 0.1 * np.random.randn(Kp)
    
    # Bilinear solve
    lnlike = np.empty(niter)
    lnprior = np.empty(niter)
    vT_guess = vT_guess0
    for n in tqdm(range(niter)):
        u_guess = get_u(vT_guess)
        vT_guess = get_vT(u_guess)
        a_guess = u_guess.reshape(-1, 1).dot(vT_guess.reshape(1, -1)).reshape(-1)
        f_guess = D.dot(a_guess)
        r = f - f_guess
        lnlike[n] = -0.5 * np.dot(r.reshape(1, -1), r.reshape(-1, 1)) / ferr ** 2
        lnprior[n] = -0.5 * (
            (u_guess[0] - 1.0) ** 2 / sig_u0 ** 2 +
            np.dot(u_guess[1:].reshape(1, -1), u_guess[1:].reshape(-1, 1)) / sig_u ** 2 +
            np.dot((vT_guess - 1.0).reshape(1, -1), (vT_guess - 1.0).reshape(-1, 1)) / sig_vT ** 2
        )

    # The solution
    u = u_guess
    vT = vT_guess
    f_model = f_guess

    # Render it
    map_map = starry.Map(ydeg, lazy=False)
    map_map.inc = inc
    map_map[1:, :] = u[1:]
    map_img = map_map.render(theta=np.linspace(-180, 180, ntheta + 1)[:-1])

    # Save
    np.savez(
        "%s.npz" % filename,
        lam=lam,
        lam_padded=lam_padded,
        I0=I0,
        I0_padded=I0_padded,
        F=F,
        img=img,
        ylms=ylms,
        map_img=map_img,
        vT_guess0=vT_guess0,
        u=u,
        vT=vT,
        f_model=f_model,
        lnlike=lnlike,
        lnprior=lnprior,
        **params
    )

# Load the result
data = dict(np.load("%s.npz" % filename))
for key in data.keys():
    if data[key].ndim == 0:
        data[key] = data[key].item()
locals().update(dict(data))

# Plot the "Joy Division" graph
fig = plt.figure(figsize=(8, 10))
ax_img = [plt.subplot2grid((ntheta, 8), (n, 0), rowspan=1, colspan=1)
            for n in range(ntheta)]
ax_f = [plt.subplot2grid((ntheta, 8), (0, 1), rowspan=1, colspan=7)]
ax_f += [plt.subplot2grid((ntheta, 8), (n, 1), rowspan=1, colspan=7, 
            sharex=ax_f[0], sharey=ax_f[0]) for n in range(1, ntheta)]
F_model = f_model.reshape(M, K)
vmin = min(np.nanmin(img), np.nanmin(map_img))
vmax = max(np.nanmax(img), np.nanmax(map_img))
for n in range(ntheta):
    ax_img[n].imshow(map_img[n], extent=(-1, 1, -1, 1), 
                    origin="lower", cmap="plasma", vmin=vmin,
                    vmax=vmax)
    ax_img[n].axis('off')
    m = int(np.round(np.linspace(0, M - 1, ntheta)[n]))
    ax_f[n].plot(lam, F[m] / np.median(F[m]), clip_on=False)
    ax_f[n].plot(lam, F_model[m] / np.median(F[m]), clip_on=False)
    ax_f[n].axis('off')
ax_f[0].set_ylim(0.9, 1.1)
fig.savefig("%s_timeseries.pdf" % filename, bbox_inches="tight", dpi=400)
plt.close()

# Plot the rest frame spectrum
fig, ax = plt.subplots(1)
ax.plot(lam_padded, I0_padded, "C0-", label="True")
ax.plot(lam_padded, vT.reshape(-1), "C1-", label="Inferred")
ax.axvspan(lam_padded[0], lam[0], color="k", alpha=0.3)
ax.axvspan(lam[-1], lam_padded[-1], color="k", alpha=0.3)
ax.set_xlim(lam_padded[0], lam_padded[-1])
ax.set_xlabel(r"$\Delta \ln \lambda$")
ax.set_ylabel(r"Normalized intensity")  
ax.legend(loc="lower left", fontsize=14)
fig.savefig("%s_spectrum.pdf" % filename, bbox_inches="tight")
plt.close()

# Save an animation
map_map = starry.Map(ydeg, lazy=False)
map_map.inc = inc
map_map[1:, :] = u[1:] / u[0]
map_map.show(theta=np.linspace(-180, 180, 50), mp4="%s.mp4" % filename)