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
lmax = 5
lam_max = 2e-5
K = 199                 # Number of wavs in model
Kpad = 15
inc = 60.0
v_c = 2.e-6
P = 1.0
t_min = -0.5
t_max = 0.5
M = 31                  # Number of observations
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
spot = np.array(map.y)

# The Doppler `g` functions
solver = RigidRotationSolver(lmax)


g = solver.g(lam, v_c * np.sin(inc * np.pi / 180.0))

# Toeplitz convolve
T = [None for n in range(N)]
for n in range(N):
    col0 = np.pad(g[n, :K // 2 + 1][::-1], (0, K // 2), mode='constant')
    row0 = np.pad(g[n, K // 2:], (0, K // 2), mode='constant')
    T[n] = csr_matrix(toeplitz(col0, row0))

# Rotation matrices
axis = [0, np.sin(inc * np.pi / 180), np.cos(inc * np.pi / 180)]
t = np.linspace(t_min, t_max, M)
theta = 2 * np.pi / P * t
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
D = D.tocsr()
# (6169, 7164)

# Mask the edges
inds = np.tile(np.arange(K), M)
obs = (inds >= Kpad) & (inds < K - Kpad)
Kobs = K - 2 * Kpad
D = D[obs]
# (5239, 7164)


# DEBUG broken
#theta = 2 * np.pi / P * np.linspace(t_min, t_max, M)
#D_new = solver.D(lam, v_c=v_c, inc=inc, theta=theta)
# (6169, 7812)


#import pdb; pdb.set_trace()

# Synthetic spectrum
a = spot.reshape(-1, 1).dot(I0.reshape(1, -1)).reshape(-1)
f = D.dot(a)
ferr = 0.0001
np.random.seed(13)
f += ferr * np.random.randn(M * Kobs)

# Set up the model
with pm.Model() as model:

    # The spherical harmonic basis
    mu_u = np.zeros(N)
    mu_u[0] = 1.0
    cov_u = 1e-2 * np.eye(N)
    cov_u[0, 0] = 1e-10
    u = pm.MvNormal("u", mu_u, cov_u, shape=(N,))
    u = tt.reshape(u, (N, 1))

    # The spectral basis
    mu_vT = np.ones(K)
    cov_vT = 1e-2 * np.eye(K)
    vT = pm.MvNormal("vT", mu_vT, cov_vT, shape=(1, K))
    vT = tt.reshape(vT, (1, K))
    
    # Compute the model
    uvT = tt.reshape(tt.dot(u, vT), (N * K, 1))
    f_model = tt.reshape(ts.dot(D, uvT), (M * Kobs,))

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
F = f.reshape(M, Kobs)
F_model = map_soln["f_model"].reshape(M, Kobs)
for m in range(M): 
    ax[m].plot(lam[Kpad:-Kpad], F[m] / F[m][0])
    ax[m].plot(lam[Kpad:-Kpad], F_model[m] / F[m][0])
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