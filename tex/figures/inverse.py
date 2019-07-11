# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt; plt.switch_backend("Qt5Agg") # DEBUG
import numpy as np
from scipy.linalg import toeplitz
import starry
from utils import RigidRotationSolver
from tqdm import tqdm


# Params
lmax = 5
lam_max = 2e-5
K = 999                 # Number of wavs in model
Ko = 799                # Number of wavs actually observed
line_amp = 1.0
line_mu = 0.0
line_sigma = 3e-7
spot_amp = -0.1
spot_sigma = 0.05
spot_lat = 30.0
spot_lon = 0.0
inc = 90.0
w_c = 2.e-6
P = 1.0
t_min = -0.25
t_max = 0.25
M = 11                  # Number of observations
N = (lmax + 1) ** 2     # Number of Ylms

# Log wavelength array
lam = np.linspace(-lam_max, lam_max, K)

# A Gaussian absorption line
I0 = 1 - line_amp * np.exp(-0.5 * (lam - line_mu) ** 2 / line_sigma ** 2)

# Instantiate a map with a Gaussian spot
map = starry.Map(lmax, lazy=False)
map.add_spot(amp=spot_amp, sigma=spot_sigma, lat=spot_lat, lon=spot_lon)
spot = np.array(map.y)

# The spectral map a[nylm, nwav] is the outer product
A = np.outer(spot, I0)
a = A.reshape(-1)

# The Doppler `g` functions
solver = RigidRotationSolver(lmax)
g = solver.g(lam, w_c * np.sin(inc * np.pi / 180.0)).T

# Toeplitz convolve
T = np.empty((N, K, K))
for n in range(N):
    col0 = np.pad(g[n, :K // 2 + 1][::-1], (0, K // 2), mode='constant')
    row0 = np.pad(g[n, K // 2:], (0, K // 2), mode='constant')
    T[n] = toeplitz(col0, row0)

# Rotation matrices
u = [0, np.sin(inc * np.pi / 180), np.cos(inc * np.pi / 180)]
time = np.linspace(t_min, t_max, M)
theta = 2 * np.pi / P * time
R = [map.ops.R(u, t) for t in theta]

# The design matrix
D = np.empty((M, K, N * K))
for t in tqdm(range(M)):
    TR = np.empty_like(T)
    for l in range(lmax + 1):
        idx = slice(l ** 2, (l + 1) ** 2)
        TR[idx] = np.dot(T[idx].T, R[t][l].T).T
    D[t] = TR.reshape(N * K, K).T

# Trim and reshape
D = D[:, (K - Ko) // 2:-(K - Ko) // 2, :]
D = D.reshape(M * Ko, N * K)

foo = np.log10(np.abs(D))
foo[foo < -12] = np.nan
plt.imshow(foo, aspect='auto')
plt.colorbar()

plt.figure()
f = np.dot(D, a)
plt.plot(f)

plt.show()
quit()