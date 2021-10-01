import starry
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm


starry.config.quiet = True
ntimes = 100
alpha = 0.05


def timeit(ydeg=10, nt=10, nw=100, vsini=50000.0):
    wav = np.linspace(642.85, 643.15, nw)
    wav0 = np.linspace(642.00, 644.00, nw)
    map = starry.DopplerMap(
        ydeg=ydeg, nt=nt, wav=wav, wav0=wav0, lazy=False, vsini_max=vsini
    )
    map.spectrum = np.random.random(map.nw0)
    map[:, :] = np.random.randn(map.Ny)
    flux = map.flux()
    times = np.zeros(ntimes)
    for n in range(ntimes):
        start = time.time()
        flux = map.flux()
        times[n] = time.time() - start
    return times * 1e3


# Set up
fig, ax = plt.subplots(2, 2, sharey=True, figsize=(8, 4))
fig.subplots_adjust(hspace=0.4, wspace=0.1)
ax[0, 0].set_ylim(0, 20)


# Versus spherical harmonic degree
ydegs = np.arange(1, 21)
times = np.zeros((len(ydegs), ntimes))
for n, ydeg in tqdm(
    enumerate(ydegs),
    total=len(ydegs),
    disable=os.getenv("CI", "false") == "true",
):
    times[n] = timeit(ydeg=ydeg)
ax[0, 0].plot(ydegs, times, "C0-", lw=1, alpha=alpha)
ax[0, 0].plot(ydegs, np.median(times, axis=1), "C0-", lw=2)
ax[0, 0].set_xlabel("spherical harmonic degree", fontsize=12)
ax[0, 0].set_xticks([0, 5, 10, 15, 20])
ax[0, 0].set_xlim(0, 20)


# Versus number of epochs
nts = np.array(np.linspace(1, 51, 20), dtype=int)
times = np.zeros((len(nts), ntimes))
for n, nt in tqdm(
    enumerate(nts),
    total=len(nts),
    disable=os.getenv("CI", "false") == "true",
):
    times[n] = timeit(nt=nt)
ax[0, 1].plot(nts, times, "C0-", lw=1, alpha=alpha)
ax[0, 1].plot(nts, np.median(times, axis=1), "C0-", lw=2)
ax[0, 1].set_xlabel("number of epochs", fontsize=12)
ax[0, 1].set_xticks([0, 10, 20, 30, 40, 50])
ax[0, 1].set_xlim(0, 50)


# Versus number of wavelength bins
nws = np.array(np.logspace(1, 3, 20), dtype=int)
times = np.zeros((len(nws), ntimes))
for n, nw in tqdm(
    enumerate(nws),
    total=len(nws),
    disable=os.getenv("CI", "false") == "true",
):
    times[n] = timeit(nw=nw)
ax[1, 0].plot(nws, times, "C0-", lw=1, alpha=alpha)
ax[1, 0].plot(nws, np.median(times, axis=1), "C0-", lw=2)
ax[1, 0].set_xlabel("number of wavelength bins", fontsize=12)
ax[1, 0].set_xticks([0, 250, 500, 750, 1000])
ax[1, 0].set_xlim(0, 1000)


# Versus vsini
vsinis = np.linspace(1, 100, 20) * 1000
times = np.zeros((len(vsinis), ntimes))
for n, vsini in tqdm(
    enumerate(vsinis),
    total=len(vsinis),
    disable=os.getenv("CI", "false") == "true",
):
    times[n] = timeit(vsini=vsini)
ax[1, 1].plot(vsinis / 1000, times, "C0-", lw=1, alpha=alpha)
ax[1, 1].plot(vsinis / 1000, np.median(times, axis=1), "C0-", lw=2)
ax[1, 1].set_xlabel(r"$v\sin i$ [km/s]", fontsize=12)
ax[1, 1].set_xticks([0, 25, 50, 75, 100])
ax[1, 1].set_xlim(0, 100)


# Tweak appearance
for axis in ax.flatten():
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontsize(10)
axl = fig.add_subplot(111)
axl.set_zorder(ax[0, 0].zorder - 1)
axl.spines["top"].set_color("none")
axl.spines["bottom"].set_color("none")
axl.spines["left"].set_color("none")
axl.spines["right"].set_color("none")
axl.set_yticks([])
axl.set_xticks([])
axl.set_ylabel("model evaluation time [ms]", labelpad=35, fontsize=14)


# Save
fig.savefig("runtime.pdf", bbox_inches="tight")