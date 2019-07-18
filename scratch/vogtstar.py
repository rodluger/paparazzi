# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import starry
from utils import RigidRotationSolver
from tqdm import tqdm
import exoplanet as xo
import theano.tensor as tt
import theano.sparse as ts
import pymc3 as pm


# Params
ydeg = 6
lam_max = 2e-5
K = 399                 # Number of wavs observed
inc = 60.0
beta = 2.e-6
P = 1.0
t_min = -0.5
t_max = 0.5
M = 99                  # Number of observations
N = (ydeg + 1) ** 2     # Number of Ylms

# Instantiate a map
map = starry.Map(ydeg, lazy=False)
map.inc = inc
map.load("vogtstar.jpg")
ylms = np.array(map.y)

# Log wavelength array
lam = np.linspace(-lam_max, lam_max, K)

# Instantiate the solver
solver = RigidRotationSolver(lam, ydeg=ydeg, beta=beta, inc=inc, P=P)
lam_padded = solver.lam_padded

# Create a fake spectrum w/ a bunch of lines
# Note that we generate it on a *padded* wavelength grid
# so we can control the behavior at the edges
I0_padded = np.ones_like(lam_padded)
np.random.seed(12)
for i in range(30):
    line_amp = 0.5 * np.random.random()
    line_mu = 2.1 * (0.5 - np.random.random()) * lam_max
    line_sigma = 2.e-7
    I0_padded -= line_amp * np.exp(-0.5 * (lam_padded - line_mu) ** 2 / 
                                   line_sigma ** 2)
I0 = I0_padded[solver.mask]

# Compute the *true* map matrix
A = ylms.reshape(-1, 1).dot(I0_padded.reshape(1, -1))
a = A.reshape(-1)

# Generate the synthetic spectral timeseries
t = np.linspace(t_min, t_max, M)
D = solver.D(t=t)
f_true = D.dot(a)
F_true = f_true.reshape(M, K)

# Add some noise
ferr = 0.0001
np.random.seed(13)
f = f_true + ferr * np.random.randn(M * K)
F = f.reshape(M, K)

# Show
'''
plt.plot(solver.lam_padded, I0_padded)
plt.plot(lam, F[0])
plt.show()
'''

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
    mu_vT = np.ones(solver.Kp)
    cov_vT = 1e-1 * np.eye(solver.Kp)
    vT = pm.MvNormal("vT", mu_vT, cov_vT, shape=(solver.Kp,), testval=I0_padded)
    vT_ = tt.reshape(vT, (1, -1))
    
    # Compute the model
    uvT = tt.reshape(tt.dot(u, vT_), (-1, 1))
    f_model = tt.reshape(ts.dot(D, uvT), (-1,))

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
ax.plot(lam_padded, I0_padded)
ax.plot(lam_padded, map_soln["vT"].reshape(-1))
ax.axvspan(lam_padded[0], lam[0], color="k", alpha=0.3)
ax.axvspan(lam[-1], lam_padded[-1], color="k", alpha=0.3)
ax.set_xlim(lam_padded[0], lam_padded[-1])

fig, ax = plt.subplots(M, figsize=(3, 8), sharex=True, sharey=True)
F_model = map_soln["f_model"].reshape(M, K)
for m in range(M): 
    ax[m].plot(lam, F[m] / F[m][0])
    ax[m].plot(lam, F_model[m] / F[m][0])
    ax[m].axis('off')

ntheta = 12
img = map.render(theta=np.linspace(-180, 180, ntheta))
coeff = map_soln["u"]
coeff /= coeff[0]
map_map = starry.Map(ydeg, lazy=False)
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