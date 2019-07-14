# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse import csr_matrix, hstack, vstack, diags
import starry
from utils import RigidRotationSolver
from tqdm import tqdm


# DEBUG
plt.switch_backend("Qt5Agg")
import os
if int(os.getenv('TRAVIS', 0)): 
    quit()

# Params
lmax = 5
lam_max = 2e-5
K = 299                 # Number of wavs in model
Ko = 299                # Number of wavs actually observed
line_amp = 1.0
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
l2sig = 1.0
M = 31                  # Number of observations
N = (lmax + 1) ** 2     # Number of Ylms

# Log wavelength array
lam = np.linspace(-lam_max, lam_max, K)

# A Gaussian absorption line
I0 = 1 - line_amp * np.exp(-0.5 * (lam - line_mu) ** 2 / line_sigma ** 2)

# Instantiate a map with a Gaussian spot
map = starry.Map(lmax, lazy=False)
map.inc = inc
map.add_spot(amp=spot_amp, sigma=spot_sigma, 
             lat=spot_lat, lon=spot_lon)
spot = np.array(map.y)

# The spectral map a[nylm, nwav] is the outer product
A = np.outer(spot, I0)
a = A.reshape(-1)

# The Doppler `g` functions
solver = RigidRotationSolver(lmax)
g = solver.g(lam, w_c * np.sin(inc * np.pi / 180.0)).T

# Normalize them (?) Check if this is
# the best way to do this
g /= np.trapz(g[0])

# Toeplitz convolve
if Ko == K:
    obs = slice(None, None)
else:
    obs = slice((K - Ko) // 2, -(K - Ko) // 2)
T = [None for n in range(N)]
for n in range(N):
    col0 = np.pad(g[n, :K // 2 + 1][::-1], (0, K // 2), mode='constant')
    row0 = np.pad(g[n, K // 2:], (0, K // 2), mode='constant')
    T[n] = csr_matrix(toeplitz(col0, row0)[obs])

# Rotation matrices
u = [0, np.sin(inc * np.pi / 180), np.cos(inc * np.pi / 180)]
time = np.linspace(t_min, t_max, M)
theta = 2 * np.pi / P * time
R = [map.ops.R(u, t) for t in theta]

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
f = D.dot(a)
sig = 0.0001
f += sig * np.random.randn(len(f))

# Regress
DTD = np.dot(D.T, D)

# Unit-mean L2 prior on the Y00 terms, zero-mean on the rest
L2 = l2sig ** -2 * np.eye(DTD.shape[0])
mu_a = np.zeros(DTD.shape[0])
mu_a[:K] = 1.0

ahat = np.linalg.solve(DTD + L2, np.dot(D.T, f) + np.dot(L2, mu_a))
Ahat = ahat.reshape(N, K)
fhat = np.dot(D, ahat)

# -- Plot stuff -- 

# Eigenspectra
fig, ax = plt.subplots(1, 2)
ax[0].imshow(A, aspect='auto')
ax[1].imshow(ahat.reshape(N, K), aspect='auto')

# Observation, model
fig = plt.figure()
plt.plot(f)
plt.plot(fhat)

# Y00 spectrum
fig = plt.figure()
plt.plot(A[0])
plt.plot(Ahat[0])

# Map in white light
map[1:, :] = np.sum(Ahat, axis=1)[1:]
fig = plt.figure()
plt.plot(map.flux(theta=np.linspace(-180, 180, 500)))
map.show(theta=np.linspace(-180, 180, 50))

plt.show()

