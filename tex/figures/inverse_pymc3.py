# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix, hstack, vstack, diags
import starry
from utils import RigidRotationSolver
from tqdm import tqdm
import exoplanet as xo
import theano.tensor as tt
import pymc3 as pm


# DEBUG
plt.switch_backend("Qt5Agg")
import os
if int(os.getenv('TRAVIS', 0)): 
    quit()


# Params
lmax = 5
lam_max = 2e-5
K = 199                 # Number of wavs in model
Kpad = 15
line_amp = 0.1
line_mu = 0.0
line_sigma = 3e-7
spot_amp = -0.1
spot_sigma = 0.05
spot_lat = 0.0
spot_lon = 0.0
inc = 60.0
w_c = 2.e-6
P = 1.0
t_min = -0.5
t_max = 0.5
l2sig = 0.03
M = 11                  # Number of observations
N = (lmax + 1) ** 2     # Number of Ylms

# Log wavelength array
lam = np.linspace(-lam_max, lam_max, K)

# A Gaussian absorption line
I0 = 1 - line_amp * np.exp(-0.5 * (lam - line_mu) ** 2 / line_sigma ** 2)

# Instantiate a map with a single Gaussian spot
map = starry.Map(lmax, lazy=False)
map.inc = inc
map.add_spot(amp=spot_amp, sigma=spot_sigma, 
             lat=spot_lat, lon=spot_lon)
spot = np.array(map.y)

# The Doppler `g` functions
solver = RigidRotationSolver(lmax)
g = solver.g(lam, w_c * np.sin(inc * np.pi / 180.0)).T

# Normalize them (?) Check if this is
# the best way to do this
g /= np.trapz(g[0])

# Toeplitz convolve
T = [None for n in range(N)]
for n in range(N):
    col0 = np.pad(g[n, :K // 2 + 1][::-1], (0, K // 2), mode='constant')
    row0 = np.pad(g[n, K // 2:], (0, K // 2), mode='constant')
    T[n] = csr_matrix(toeplitz(col0, row0))

# Rotation matrices
axis = [0, np.sin(inc * np.pi / 180), np.cos(inc * np.pi / 180)]
time = np.linspace(t_min, t_max, M)
theta = 2 * np.pi / P * time
R = [map.ops.R(axis, t) for t in theta]

# The design matrix
Dt = [None for t in range(M)]
for t in tqdm(range(M)):
    TR = [None for n in range(N)]
    for l in range(lmax + 1):
        idx = slice(l ** 2, (l + 1) ** 2)
        TR[idx] = np.tensordot(R[t][l].T, T[idx], axes=1)
    Dt[t] = hstack(TR)
D = vstack(Dt)

# DEBUG: for some reason the sparse dot is much slower
D = np.array(D.todense())

# Synthetic spectrum
f = D.dot(spot.reshape(-1, 1).dot(I0.reshape(1, -1)).reshape(-1))
ferr = 0.0001
np.random.seed(12)
f += ferr * np.random.randn(M * K)

# Mask the edges
inds = np.tile(np.arange(K), M)
obs = (inds > Kpad) & (inds < K - Kpad)

# Set up the model
with pm.Model() as model:

    # The spherical harmonic basis
    mu_u = np.zeros(N)
    mu_u[0] = 1.0
    cov_u = 0.03 * np.eye(N)
    cov_u[0] = 1.0
    u = pm.MvNormal("u", mu_u, cov_u, shape=(N,))
    u = tt.reshape(u, (N, 1))

    # The spectral basis
    mu_vT = np.ones(K)
    cov_vT = 0.001 * np.eye(K)
    vT = pm.MvNormal("vT", mu_vT, cov_vT, shape=(1, K))
    vT = tt.reshape(vT, (1, K))
    
    # Compute the model
    uvT = tt.reshape(tt.dot(u, vT), (N * K,))
    f_model = tt.dot(D, uvT)

    # Track some values for plotting later
    pm.Deterministic("f_model", f_model)

    # Save our initial guess
    f_model_guess = xo.eval_in_model(f_model)

    # The likelihood function assuming known Gaussian uncertainty
    pm.Normal("obs", mu=f_model[obs], sd=ferr, observed=f[obs])


# Maximum likelihood solution
with model:
    map_soln = xo.optimize()

# Plot some stuff
fig, ax = plt.subplots(1)
ax.plot(lam, I0)
ax.plot(lam, map_soln["vT"].reshape(-1))

fig, ax = plt.subplots(M, figsize=(3, 8), sharex=True, sharey=True)
F = f.reshape(M, K)
F_model = map_soln["f_model"].reshape(M, K)
if Kpad == 0:
    inds = slice(None)
else:
    inds = slice(Kpad, K - Kpad)
for m in range(M): 
    ax[m].plot(lam[inds], F[m][inds] / F[m][Kpad])
    ax[m].plot(lam[inds], F_model[m][inds] / F[m][Kpad])
    ax[m].axis('off')

ntheta = 12
img = map.render(theta=np.linspace(-180, 180, ntheta))
coeff = map_soln["u"]
coeff /= coeff[0]
map_map = starry.Map(lmax, lazy=False)
map_map.inc = inc
map_map[1:, :] = coeff[1:]
map_img = map_map.render(theta=np.linspace(-180, 180, ntheta))
vmin = min(np.nanmin(img), np.nanmin(map_img))
vmax = max(np.nanmax(img), np.nanmax(map_img))
fig, ax = plt.subplots(2, ntheta, figsize=(ntheta, 4))
for n in range(ntheta):
    ax[0, n].imshow(img[:, :, n], extent=(-1, 1, -1, 1), 
                    origin="lower", cmap="plasma", vmin=vmin,
                    vmax=vmax)
    ax[1, n].imshow(map_img[:, :, n], extent=(-1, 1, -1, 1), 
                    origin="lower", cmap="plasma", vmin=vmin,
                    vmax=vmax)
    ax[0, n].axis('off')
    ax[1, n].axis('off')

plt.show()