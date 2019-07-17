# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix, hstack, vstack
import starry
from utils import RigidRotationSolver
from tqdm import tqdm
import exoplanet as xo
import theano.tensor as tt
import theano.sparse as ts
import pymc3 as pm


# Params
lmax = 8
lam_max = 2e-5
K = 199                 # Number of wavs observed
inc = 60.0
v_c = 2.e-6
P = 1.0
t_min = -0.5
t_max = 0.5
M = 99                 # Number of observations
N = (lmax + 1) ** 2     # Number of Ylms

# Log wavelength array
lam = np.linspace(-lam_max, lam_max, K)

# A fake spectrum w/ a bunch of lines
lam_hr = np.linspace(-lam_max, lam_max, 10 * K)
I0 = np.ones_like(lam_hr)
np.random.seed(12)
for i in range(30):
    line_amp = np.random.random()
    line_mu = 1.5 * (0.5 - np.random.random()) * lam_max
    line_sigma = 1e-8 + np.abs(1e-7 * np.random.randn())
    I0 -= line_amp * np.exp(-0.5 * (lam_hr - line_mu) ** 2 / line_sigma ** 2)
I0 = np.interp(lam, lam_hr, I0)

# Instantiate a map
map = starry.Map(lmax, lazy=False)
map.inc = inc
map.load("vogtstar.jpg")
ylms = np.array(map.y)

# The Doppler design matrix
solver = RigidRotationSolver(lmax)
theta = 2 * np.pi / P * np.linspace(t_min, t_max, M)
solver.compute(lam, v_c=v_c, inc=inc, theta=theta)

# Synthetic spectrum
a = ylms.reshape(-1, 1).dot(solver.pad(I0).reshape(1, -1)).reshape(-1)
f = solver.D.dot(a)
ferr = 0.0001
np.random.seed(13)
f += ferr * np.random.randn(M * K)

# Set up the model
with pm.Model() as model:

    # The spherical harmonic basis
    mu_u = np.zeros(N)
    mu_u[0] = 1.0
    cov_u = 1e-2 * np.eye(N)
    cov_u[0, 0] = 1e-10
    u = pm.MvNormal("u", mu_u, cov_u, shape=(N,))
    u = tt.reshape(u, (-1, 1))

    # The spectral basis
    baseline = pm.Normal("baseline", 1.0, 1e-1)
    mu_vT = np.ones(K)
    cov_vT = 1e-2 * np.eye(K)
    vT = pm.MvNormal("vT", mu_vT, cov_vT, shape=(K,))
    vT_ = tt.reshape(solver.pad(vT, baseline), (1, -1))
    
    # Compute the model
    uvT = tt.reshape(tt.dot(u, vT_), (-1, 1))
    f_model = tt.reshape(ts.dot(solver.D, uvT), (-1,))

    # Track some values for plotting later
    pm.Deterministic("f_model", f_model)

    # Save our initial guess
    f_model_guess = xo.eval_in_model(f_model)

    # The likelihood function assuming known Gaussian uncertainty
    pm.Normal("obs", mu=f_model, sd=ferr, observed=f)

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
for m in range(M): 
    ax[m].plot(lam, F[m] / F[m][0])
    ax[m].plot(lam, F_model[m] / F[m][0])
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
    ax[0, n].imshow(img[n], extent=(-1, 1, -1, 1), 
                    origin="lower", cmap="plasma", vmin=vmin,
                    vmax=vmax)
    ax[1, n].imshow(map_img[n], extent=(-1, 1, -1, 1), 
                    origin="lower", cmap="plasma", vmin=vmin,
                    vmax=vmax)
    ax[0, n].axis('off')
    ax[1, n].axis('off')

map_map.show(theta=np.linspace(-180, 180, 50))

plt.show()