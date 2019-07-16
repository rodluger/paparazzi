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
T = [None for n in range(N)]
for n in range(N):
    col0 = np.pad(g[n, :K // 2 + 1][::-1], (0, K // 2), mode='constant')
    row0 = np.pad(g[n, K // 2:], (0, K // 2), mode='constant')
    T[n] = csr_matrix(toeplitz(col0, row0))

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
ferr = 0.0001
f += ferr * np.random.randn(len(f))

# Regress
CInv = np.ones_like(f) / ferr ** 2
winds = np.tile(np.arange(K), M)
pad = (winds < Kpad) | (winds > K - Kpad)
CInv[pad] = 1e-6
CInv = np.diag(CInv)
DTCInv = np.dot(D.T, CInv)
DTCInvf = np.dot(DTCInv, f)
DTCInvD = np.dot(DTCInv, D)
mu_a = np.zeros(DTCInvD.shape[0])
mu_a[:K] = 1.0
LInv = np.append(np.ones(K), l2sig ** -2 * np.ones(K * (N - 1)))
LInv = np.diag(LInv)
ahat = np.linalg.solve(DTCInvD + LInv, DTCInvf + np.dot(LInv, mu_a))
Ahat = ahat.reshape(N, K)
fhat = np.dot(D, ahat)

# -- Plot stuff -- 

# Eigenspectra
fig, ax = plt.subplots(1, 2)
ax[0].imshow(A, aspect='auto')
ax[1].imshow(ahat.reshape(N, K), aspect='auto')

# Observation, model
fig = plt.figure()
plt.plot(f, color="C0")
plt.plot(fhat, color="C1")

# Y00 spectrum
fig = plt.figure()
plt.plot(A[0])
plt.plot(Ahat[0])
Asig = np.sqrt(np.diag(np.linalg.inv(DTCInvD + LInv))[:K])
plt.fill_between(range(len(Asig)), Ahat[0] - Asig, Ahat[0] + Asig,
                 color="C1", alpha=0.3)

# Map in white light
map[1:, :] = np.sum(Ahat, axis=1)[1:]
fig = plt.figure()
plt.plot(map.flux(theta=np.linspace(-180, 180, 500)))
map.show(theta=np.linspace(-180, 180, 50))

plt.show()

