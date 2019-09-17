# -*- coding: utf-8 -*-
"""
Compare our method to discrete numerical integration
on the disk.

"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import starry
import paparazzi as pp

CLIGHT = 3.0e5


# Get the Ylm expansion of a Gaussian spot
ydeg = 20
N = (ydeg + 1) ** 2
map = starry.Map(ydeg)
map.add_spot(amp=-0.03, sigma=0.05, lat=30, lon=30)
y1 = np.array(map.y.eval())[1:]

# Check that the specific intensity is positive everywhere
assert np.nanmin(map.render().eval()) > 0

# This work
vsini = 40.0  # km/s
dop = pp.Doppler(ydeg, vsini=vsini)
dop.generate_data(
    y1=y1, R=3.0e6, nlam=1999, sigma=2.0e-5, nlines=1, theta=[0.0], ferr=0.0
)
F = dop.F[0] / dop.F[0][0]

# The rest frame spectrum
s = dop.s_true
lnlam = dop.lnlam
lnlam_padded = dop.lnlam_padded
obs = (lnlam_padded >= lnlam[0]) & (lnlam_padded <= lnlam[-1])

# Numerical
npts = np.zeros(3, dtype=int)
res_arr = [50, 100, 300]  # , 600]
Fnum = np.array([np.zeros_like(lnlam) for i in range(len(res_arr))])
for i, res in enumerate(res_arr):
    x, y, z = map.ops.compute_ortho_grid(res).eval()
    on_disk = np.isfinite(z)
    x = x[on_disk]
    y = y[on_disk]
    z = z[on_disk]
    npts[i] = len(x)
    D = 0.5 * np.log((1 + vsini / CLIGHT * x) / (1 - vsini / CLIGHT * x))
    image = map.render(res=res).eval().reshape(-1, 1)[on_disk]
    spec = np.interp(lnlam_padded - D.reshape(-1, 1), lnlam_padded, s)
    spec = spec[:, obs]
    Fnum[i] = np.nansum(image * spec, axis=0)
    Fnum[i] /= Fnum[i, 0]

# Compare
fig, ax = plt.subplots(2, sharex=True, figsize=(8, 8))
ax[0].plot(1e4 * lnlam, F, lw=2.5, label="Luger et al. (2019)")
ax[0].plot(1e4 * lnlam, Fnum[-1], "--", lw=2.5, label="numerical")
ax[0].set_xlim(-3, 3)
alpha = [0.2, 0.5, 1.0]
npts_rounded = np.array(np.round(1e-4 * npts, 1) * 1e4, dtype=int)
for i in range(len(Fnum)):
    ax[1].plot(
        1e4 * lnlam,
        np.abs(F - Fnum[i]),
        label=r"$n = %d$" % npts_rounded[i],
        color="k",
        alpha=alpha[i],
    )
ax[0].set_ylabel(r"spectrum")
ax[0].legend(fontsize=12, loc="lower left")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"$\lambdabar$ (arbitrary units)")
ax[1].legend(fontsize=10, loc="upper right")
ax[1].set_ylim(1e-10, 1e0)
ax[1].set_ylabel(r"residuals")

# Show the image
aximg = inset_axes(ax[0], width="15%", height="45%", loc=4, borderpad=1)
img = map.render(res=300).eval().reshape(300, 300)
aximg.imshow(img, origin="lower", cmap="plasma", vmin=0.042, vmax=0.353)
aximg.axis("off")

fig.savefig("compare.pdf", bbox_inches="tight")
