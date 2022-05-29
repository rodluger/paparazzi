# -*- coding: utf-8 -*-
"""
Compare our method to discrete numerical integration
on the disk.

"""
from utils import patch_theano
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import starry
import paths


# Get the Ylm expansion of a Gaussian spot
ydeg = 20
N = (ydeg + 1) ** 2
spot_map = starry.Map(ydeg, lazy=False)
spot_map.spot(contrast=0.95, radius=20, lat=30, lon=0, spot_smoothing=0.125)
y = spot_map.y.reshape(-1)

# Generate a Doppler map
inc = 90
veq = 60000.0  # m/s
theta = [0.0]
wav = np.linspace(642.75, 643.25, 200)
map = starry.DopplerMap(
    ydeg, veq=veq, vsini_max=veq, inc=inc, nt=1, wav=wav, lazy=False
)
map[:, :] = y
map.spectrum = 1.0 - np.exp(-0.5 * (map.wav0 - 643.0) ** 2 / 0.0085 ** 2)

# Compute the observed spectrum using starry
F = map.flux(theta=theta).reshape(-1)
F /= F[0]

# Compute the observed spectrum numerically
# for different grid resolutions
vsini = veq * np.sin(inc * np.pi / 180)
res_arr = [12, 36, 113, 357]  # , 1129]
npts = np.zeros(len(res_arr), dtype=int)
Fnum = np.array([np.zeros_like(map.wav) for i in range(len(res_arr))])
for i, res in enumerate(res_arr):
    _, xyz = spot_map.ops.compute_ortho_grid(res)
    on_disk = np.isfinite(xyz[2])
    x = xyz[0][on_disk]
    y = xyz[1][on_disk]
    z = xyz[2][on_disk]
    npts[i] = len(x)
    D = np.sqrt((1 + vsini / map._clight * x) / (1 - vsini / map._clight * x))
    image = spot_map.render(res=res).reshape(-1, 1)[on_disk]
    spec = np.interp(
        map.wav0 * D.reshape(-1, 1), map.wav0, map.spectrum.reshape(-1)
    )
    flux = np.nansum(image * spec, axis=0)
    flux = np.interp(map.wav, map.wav0, flux)
    Fnum[i] = flux / flux[0]

# Compare
fig, ax = plt.subplots(2, sharex=True, figsize=(8, 8))
ax[0].plot(map.wav, F, lw=2.5, label="Luger et al. (2021)")
ax[0].plot(map.wav, Fnum[-1], "--", lw=2.5, label="numerical")
ax[0].margins(0, None)
alpha = [0.2, 0.4, 0.6, 1.0]
exp = np.array(np.round(np.log10(npts)), dtype=int)
for i in range(len(Fnum)):
    ax[1].plot(
        map.wav,
        np.abs(F - Fnum[i]),
        label="$n = 10^{}$".format(exp[i]),
        color="k",
        alpha=alpha[i],
    )
ax[0].set_ylabel(r"spectrum")
ax[0].legend(fontsize=10, loc="lower left")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"$\lambda$ [nm]")
ax[1].legend(fontsize=10, loc="upper right")
ax[1].set_ylim(1e-10, 1e0)
ax[1].set_ylabel(r"relative error (numerical)")

# Show the image
aximg = inset_axes(ax[0], width="15%", height="45%", loc=4, borderpad=1)
img = spot_map.render(res=300).reshape(300, 300)
aximg.imshow(
    img,
    extent=(-1, 1, -1, 1),
    origin="lower",
    cmap="plasma",
    vmin=0.042,
    vmax=0.353,
)
aximg.set_xlim(-1.02, 1.02)
aximg.set_ylim(-1.02, 1.02)
aximg.axis("off")
x = np.linspace(-1, 1, 3000)
y = np.sqrt(1 - x ** 2)
aximg.plot(0.999 * x, 0.999 * y, "k-", lw=0.5, zorder=100)
aximg.plot(0.999 * x, -0.999 * y, "k-", lw=0.5, zorder=100)

fig.savefig(paths.figures / "compare.pdf", bbox_inches="tight", dpi=300)
