# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve
import starry
from utils import RigidRotationSolver


# Instantiate a map with a Gaussian spot
lmax = 20
N = (lmax + 1) ** 2
map = starry.Map(lmax, lazy=False)
map.add_spot(amp=-0.1, sigma=0.05, lat=30)
ylm = np.array(map.y)

# Check that the specific intensity is positive everywhere
assert np.nanmin(map.render()) > 0

# Log wavelength array
xi = np.linspace(-2e-5, 2e-5, 9999)
dxi = xi[1] - xi[0]
obs = np.abs(xi) < 1e-5

# A Gaussian absorption line
amp = 1.0
mu = 0.0
sigma = 3e-7
a0 = 1 - amp * np.exp(-0.5 * (xi - mu) ** 2 / sigma ** 2)

# The Doppler `g` function
solver = RigidRotationSolver(lmax)
npts = 1000
wsini_c = 2.0e-6
maxD = 0.5 * np.log((1 + wsini_c) / (1 - wsini_c))
D = xi[(xi >= -maxD) & (xi <= maxD)]
g = solver.g(D, wsini_c)

# Set up the plot
nt = 15
theta = np.linspace(-90, 90, nt)
fig, ax = plt.subplots(nt, 3, figsize=(6, 12))
fig.subplots_adjust(hspace=0.3)
cmap = plt.get_cmap("plasma")
img = map.render()
vmin = np.nanmin(img)
vmax = np.nanmax(img)
rng = vmax - vmin
vmin -= 0.1 * rng
vmax += 0.1 * rng

# Compute the spectrum when the spot is on 
# the backside for reference
map.rotate(180)
a = np.dot(map.y.reshape(-1, 1), a0.reshape(1, -1))
S0 = np.zeros_like(xi)
for n in range(N):
    S0 += convolve(a[n], g[:, n], mode="same")
S0 -= S0[obs][0]

# Compute & plot each spectrum
for t in range(nt):
    # Rotate & compute spectrum
    map[1:, :] = ylm[1:]
    map.rotate(theta[t])
    a = np.dot(map.y.reshape(-1, 1), a0.reshape(1, -1))
    S = np.zeros_like(xi)
    for n in range(N):
        S += convolve(a[n], g[:, n], mode="same")
    S -= S[obs][0]
    
    # Plot spectrum
    ax[t, 1].plot(xi[obs], S0[obs], "k:", lw=1, alpha=0.5)
    ax[t, 1].plot(xi[obs], S[obs], "k-")
    ax[t, 1].set_xlim(-0.0000035, 0.0000035)
    ax[t, 1].axis('off')
    
    # Plot residuals
    color = [cmap(x) for x in np.linspace(0.75, 0.0, 5)]
    lw = np.linspace(2.5, 0.5, 5)
    alpha = np.linspace(0.25, 1, 5)
    for i in range(5):
        ax[t, 2].plot(xi[obs], S[obs] - S0[obs], ls="-", 
                      lw=lw[i], color=color[i], alpha=alpha[i])
    
    ax[t, 2].set_xlim(-0.0000035, 0.0000035)
    ax[t, 2].axis('off')
    ax[t, 2].set_ylim(-30, 30)
    
    # Plot current stellar image
    img = map.render(res=100)[:, :, 0]
    ax[t, 0].imshow(img, origin="lower", 
                    extent=(-1, 1, -1, 1),
                    cmap=cmap, vmin=vmin,
                    vmax=vmax)
    ax[t, 0].set_xlim(-3, 1)
    ax[t, 0].set_ylim(-1, 1)
    ax[t, 0].axis('off')

ax[0, 1].set_title("spectrum", y=1.4)
ax[0, 2].set_title("residuals", y=1.4)

fig.savefig("spot_rigid.pdf", bbox_inches="tight")