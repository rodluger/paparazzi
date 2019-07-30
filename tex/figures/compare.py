# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.signal import convolve
import starry
from utils import RigidRotationSolver


# Instantiate a map with a Gaussian spot
lmax = 20
N = (lmax + 1) ** 2
map = starry.Map(lmax, lazy=False)
map.add_spot(amp=-0.1, sigma=0.05, lat=30, lon=30)
ylm = np.array(map.y)

# Check that the specific intensity is positive everywhere
assert np.nanmin(map.render()) > 0

# Log wavelength array
xi = np.linspace(-7e-6, 7e-6, 499)
dxi = xi[1] - xi[0]
obs = np.abs(xi) < 4e-6

# A Gaussian absorption line
amp = 1.0
mu = 0.0
sigma = 3e-7
a0 = 1 - amp * np.exp(-0.5 * (xi - mu) ** 2 / sigma ** 2)
wsini_c = 2.0e-6

# Starry
maxD = 0.5 * np.log((1 + wsini_c) / (1 - wsini_c))
D = xi[(xi >= -maxD) & (xi <= maxD)]
solver = RigidRotationSolver(lmax)
g = solver.g(D, wsini_c)
a = np.dot(ylm.reshape(-1, 1), a0.reshape(1, -1))
S = np.zeros_like(xi)
for n in range(N):
    S += convolve(a[n], g[:, n], mode="same")
S /= S[obs][0]

# Numerical
res_arr = [50, 100, 300] #, 600]
Snum = np.array([np.zeros_like(xi) for i in range(len(res_arr))])
for i, res in enumerate(res_arr):
    x, y, z = map.ops.compute_ortho_grid(res).eval()
    D = 0.5 * np.log((1 + wsini_c * x) / (1 - wsini_c * x))
    image = map.render(res=res).reshape(-1, 1)
    spec = np.interp(xi + D.reshape(-1, 1), xi, a0)
    Snum[i] = np.nansum(image * spec, axis=0)
    Snum[i] /= Snum[i, 0]

# Compare
fig, ax = plt.subplots(2, sharex=True, figsize=(8, 8))
ax[0].plot(1e6 * xi[obs], S[obs], label="starry")
ax[0].plot(1e6 * xi[obs], Snum[-1][obs], "--", label="numerical")
ax[0].set_xlim(-4, 4)
alpha = [0.2, 0.5, 1.0]
for i in range(len(Snum)):
    ax[1].plot(1e6 * xi[obs], np.abs(S[obs] - Snum[i][obs]), 
               label=r"$n = %d^2$" %res_arr[i], color="k", alpha=alpha[i])
ax[0].set_ylabel(r"spectrum")
ax[0].legend(fontsize=12, loc="lower left")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"$\lambdabar$ (arbitrary units)")
ax[1].legend(fontsize=10, loc="upper right")
ax[1].set_ylim(1e-10, 1e0)
ax[1].set_ylabel(r"residuals")

# Show the image
aximg = inset_axes(ax[0], width="15%", height="45%", loc=4, borderpad=1)
img = map.render(res=300).reshape(300, 300)
aximg.imshow(img, origin="lower", cmap="plasma", vmin=0.042, vmax=0.353)
aximg.axis('off')

fig.savefig("compare.pdf", bbox_inches="tight")